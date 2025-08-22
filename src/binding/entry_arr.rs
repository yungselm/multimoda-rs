use anyhow::{anyhow, Context, Result};
use crossbeam::thread;
use std::collections::HashSet;

use crate::io::input::{Contour, ContourPoint, Record};
use crate::io::Geometry;
use crate::processing::resampling::prepare_geometries_comparison;
use crate::processing::align_within::{align_frames_in_geometry, hausdorff_distance, AlignLog};
use crate::processing::align_between::GeometryPair;
use crate::processing::process_utils::process_case;
use crate::processing::align_within_and_between_array;

use crate::io::output::write_obj_mesh_without_uv;

pub fn geometry_from_array_rs(
    contours: Vec<Contour>,
    walls: Vec<Contour>,
    reference_point: ContourPoint,
    step_rotation_deg: f64,
    range_rotation_deg: f64,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
    label: &str,
    records: Option<Vec<Record>>,
    delta: f64,
    max_rounds: usize,
    diastole: bool,
    sort: bool,
    write_obj: bool,
    output_path: &str,
    bruteforce: bool,
    sample_size: usize,
) -> Result<(Geometry, Vec<AlignLog>), Box<dyn std::error::Error>> {
    let mut contours = contours;
    let label = label.to_string();

    // Build catheter contours, propagating any errors
    let mut catheter = if n_points == 0 {
        Vec::new()
    } else {
        Contour::create_catheter_contours(
            &contours
                .iter()
                .flat_map(|c| c.points.clone())
                .collect::<Vec<_>>(),
            image_center,
            radius,
            n_points,
        )?
    };

    // Initial geometry setup, possibly reordering by records
    let mut geometry = if let Some(recs) = records.as_ref() {
        let mut z_coords: Vec<f64> = contours.iter().map(|c| c.centroid.2).collect();
        z_coords.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        Geometry::reorder_contours(&mut contours, recs, diastole, &z_coords);
        catheter.sort_by_key(|c| c.id);

        Geometry {
            contours,
            catheter,
            walls,
            reference_point,
            label: label.clone(),
        }
    } else {
        Geometry {
            contours,
            catheter,
            walls,
            reference_point,
            label,
        }
    };

    for (new_id, contour) in geometry.contours.iter_mut().enumerate() {
        contour.id = new_id  as u32;
        for point in contour.points.iter_mut() {
            point.frame_index = new_id as u32;
        }
    }
    for (new_id, catheter) in geometry.catheter.iter_mut().enumerate() {
        catheter.id = new_id  as u32;
        for point in catheter.points.iter_mut() {
            point.frame_index = new_id as u32;
        }
    }
    for contour in geometry.contours.iter_mut() {
        contour.sort_contour_points();
    }
    for catheter in geometry.catheter.iter_mut() {
        catheter.sort_contour_points();
    }

    // Optionally align and refine ordering
    let (mut geometry, logs) = if sort {
        let (interim, _) = align_frames_in_geometry(
            geometry, 
            step_rotation_deg, 
            range_rotation_deg, 
            false, 
            bruteforce, 
            sample_size);
        let refined = refine_ordering(interim, delta, max_rounds);
        align_frames_in_geometry(refined, step_rotation_deg, range_rotation_deg, true, bruteforce, sample_size)
    } else {
        align_frames_in_geometry(geometry, step_rotation_deg, range_rotation_deg, true, bruteforce, sample_size)
    };

    geometry = if geometry.walls.is_empty() {
        crate::processing::walls::create_wall_geometry(&geometry, false)
    } else {
        geometry
    };

    if write_obj {
        let filename_cont = format!("{}/mesh_000_single.obj", output_path);
        let filename_walls = format!("{}/wall_000_single.obj", output_path);
        let filename_cath = format!("{}/catheter_000_single.obj", output_path);
        write_obj_mesh_without_uv(
            &geometry.contours,
            &filename_cont,
            "mesh_000_single.mtl",
        )?;
        write_obj_mesh_without_uv(&geometry.walls, &filename_walls, "wall_000_single.mtl")?;
        write_obj_mesh_without_uv(
            &geometry.catheter,
            &filename_cath,
            "catheter_000_single.mtl",
        )?;
    }

    Ok((geometry, logs))
}

fn contour_area(contour: &Contour) -> f64 {
    let points = &contour.points;
    let n = points.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        area += points[i].x * points[j].y;
        area -= points[i].y * points[j].x;
    }
    area.abs() / 2.0
}

pub fn refine_ordering(mut geom: Geometry, delta: f64, max_rounds: usize) -> Geometry {
    let n = geom.contours.len();
    if n <= 1 {
        return geom;
    }

    // Find ostium (contour with highest original ID)
    let ostium_idx = geom.contours
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            contour_area(a).partial_cmp(&contour_area(b)).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx)
        .unwrap();

    // Start with current index order as the baseline
    let mut best_order: Vec<usize> = (0..n).collect();

    // If you want ostium at start of best_order initially, you can swap here,
    // but keep best_order consistent with current indices / geometry.
    // best_order.swap(0, ostium_idx);

    for _round in 0..max_rounds {
        // Recompute cost matrix for current geometry ordering
        let cost = build_cost_matrix(&geom.contours, delta);

        // Greedy path starting from ostium index (important!)
        let mut new_order = vec![ostium_idx];
        let mut remaining: HashSet<usize> = (0..n).collect();
        remaining.remove(&ostium_idx);

        while !remaining.is_empty() {
            let last = *new_order.last().unwrap();
            // choose remaining index with minimal cost from `last`
            let &best_next = remaining
                .iter()
                .min_by(|&&a, &&b| {
                    cost[last][a]
                        .partial_cmp(&cost[last][b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();
            new_order.push(best_next);
            remaining.remove(&best_next);
        }

        // If new path isn't better than current best, stop
        if path_cost(&cost, &new_order) >= path_cost(&cost, &best_order) {
            break;
        }

        // Accept new order -> reorder geometry immediately so indices are in sync
        best_order = new_order.clone();
        geom = reorder_geometry(geom, &best_order);
        // update contour ids to reflect new indices (optional but recommended)
        for (i, c) in geom.contours.iter_mut().enumerate() {
            c.id = i as u32;
        }
        // if catheter length == contours length, reassign IDs too
        if geom.catheter.len() == geom.contours.len() {
            for (i, cath) in geom.catheter.iter_mut().enumerate() {
                cath.id = i as u32;
            }
        }
    }

    geom
}

fn build_cost_matrix(contours: &[Contour], delta: f64) -> Vec<Vec<f64>> {
    let n = contours.len();
    let mut cost = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                cost[i][j] = 0.0;
            } else {
                let h = hausdorff_distance(&contours[i].points, &contours[j].points);
                // Use index-based jump penalty (indices refer to current ordering)
                let jump = (i as i32 - j as i32).abs() as f64;
                cost[i][j] = h + delta * jump;
            }
        }
    }
    cost
}

fn path_cost(cost_matrix: &[Vec<f64>], path: &[usize]) -> f64 {
    if path.len() < 2 {
        return 0.0;
    }
    let mut total = 0.0;
    for i in 0..path.len() - 1 {
        total += cost_matrix[path[i]][path[i + 1]];
    }
    total
}

fn reorder_geometry(mut geom: Geometry, new_order: &[usize]) -> Geometry {
    let mut z_coords: Vec<f64> = Vec::new();
    for contour in &geom.contours {
        z_coords.push(contour.centroid.2)
    }
    z_coords.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Reorder contours according to new_order
    let mut reordered_contours = new_order
        .iter()
        .map(|&i| geom.contours[i].clone())
        .collect::<Vec<_>>();

    // Reorder catheter only if same length (otherwise leave as-is or implement mapping)
    let mut reordered_catheter = if geom.catheter.len() == geom.contours.len() {
        new_order.iter().map(|&i| geom.catheter[i].clone()).collect()
    } else {
        // safer: keep catheter unchanged (or implement desired behavior)
        geom.catheter.clone()
    };

    for (i, contour) in reordered_contours.iter_mut().enumerate() {
        contour.centroid.2 = z_coords[i];
        for point in contour.points.iter_mut() {
            point.z = z_coords[i];
        }
    }
    for (i, catheter) in reordered_catheter.iter_mut().enumerate() {
        catheter.centroid.2 = z_coords[i];
        for point in catheter.points.iter_mut() {
            point.z = z_coords[i];
        }
    }

    geom.contours = reordered_contours;
    geom.catheter = reordered_catheter;

    // Optionally reset IDs to be consistent with new indices (caller may also do this)
    for (idx, c) in geom.contours.iter_mut().enumerate() {
        c.id = idx as u32;
    }
    if geom.catheter.len() == geom.contours.len() {
        for (idx, c) in geom.catheter.iter_mut().enumerate() {
            c.id = idx as u32;
        }
    }

    geom
}

fn geometry_pair_from_array_rs(
    geometry_dia: Geometry,
    geometry_sys: Geometry,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
) -> anyhow::Result<GeometryPair> {
    let mut catheters_dia = if geometry_dia.catheter.is_empty() {
        Contour::create_catheter_contours(
            &geometry_dia
                .contours
                .iter()
                .flat_map(|c| c.points.clone())
                .collect::<Vec<_>>(),
            image_center,
            radius,
            n_points,
        )?
    } else {
        geometry_dia.catheter.clone()
    };
    let mut catheters_sys = if geometry_dia.catheter.is_empty() {
        Contour::create_catheter_contours(
            &geometry_sys
                .contours
                .iter()
                .flat_map(|c| c.points.clone())
                .collect::<Vec<_>>(),
            image_center,
            radius,
            n_points,
        )?
    } else {
        geometry_sys.catheter.clone()
    };

    // Ensure catheter contour are sorted by ID in same manner as contours
    catheters_dia.sort_by_key(|c| c.id);
    catheters_sys.sort_by_key(|c| c.id);

    // Insert in Geometry
    let mut geometry_dia = Geometry {
        contours: geometry_dia.contours,
        catheter: catheters_dia,
        walls: geometry_dia.walls,
        reference_point: geometry_dia.reference_point,
        label: "Diastole".to_string(),
    };
    let mut geometry_sys = Geometry {
        contours: geometry_sys.contours,
        catheter: catheters_sys,
        walls: geometry_sys.walls,
        reference_point: geometry_sys.reference_point,
        label: "Systole".to_string(),
    };

    // reindex contours in geometry_dia and geometry_sys to range from 0 to length
    for (new_id, contour) in geometry_dia.contours.iter_mut().enumerate() {
        contour.id = new_id as u32;
    }
    for (new_id, contour) in geometry_sys.contours.iter_mut().enumerate() {
        contour.id = new_id  as u32;
    }
    for (new_id, catheter) in geometry_dia.catheter.iter_mut().enumerate() {
        catheter.id = new_id  as u32;
    }
    for (new_id, catheter) in geometry_sys.catheter.iter_mut().enumerate() {
        catheter.id = new_id  as u32;
    }

    let geometries = GeometryPair {
        dia_geom: geometry_dia,
        sys_geom: geometry_sys,
    };
    Ok(geometries)
}

pub fn from_array_full_rs(
    rest_geometry_dia: Geometry,
    rest_geometry_sys: Geometry,
    stress_geometry_dia: Geometry,
    stress_geometry_sys: Geometry,
    step_rotation_deg: f64,
    range_rotation_deg: f64,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
    write_obj: bool,
    rest_output_path: &str,
    stress_output_path: &str,
    diastole_output_path: &str,
    systole_output_path: &str,
    interpolation_steps: usize,
    bruteforce: bool,
    sample_size: usize,
) -> anyhow::Result<(
    (GeometryPair, GeometryPair, GeometryPair, GeometryPair),
    (Vec<AlignLog>, Vec<AlignLog>, Vec<AlignLog>, Vec<AlignLog>),
)> {
    let result = thread::scope(
        |s| -> Result<(
            GeometryPair,
            GeometryPair,
            GeometryPair,
            GeometryPair,
            Vec<AlignLog>,
            Vec<AlignLog>,
            Vec<AlignLog>,
            Vec<AlignLog>,
        )> {
            // REST thread
            let rest_handle = s.spawn(|_| -> anyhow::Result<_> {
                let geom_pair = geometry_pair_from_array_rs(
                    rest_geometry_dia,
                    rest_geometry_sys,
                    image_center,
                    radius,
                    n_points,
                )
                .context("create rest geometry pair(rest) failed")?;

                let (geom_rest, logs_dia, logs_sys) = align_within_and_between_array(
                    "rest", 
                    geom_pair, 
                    step_rotation_deg, 
                    range_rotation_deg, 
                    write_obj, 
                    rest_output_path, 
                    interpolation_steps,
                    bruteforce,
                    sample_size)
                        .context("process geometry pair(rest) failed")?;
                    Ok((geom_rest, logs_dia, logs_sys))
            });

            // STRESS thread
            let stress_handle = s.spawn(|_| -> anyhow::Result<_> {
                let geom_pair = geometry_pair_from_array_rs(
                    stress_geometry_dia,
                    stress_geometry_sys,
                    image_center,
                    radius,
                    n_points,
                )
                .context("create stress geometry pair(stress) failed")?;

                let (geom_stress, logs_dia_stress, logs_sys_stress) = align_within_and_between_array(
                    "stress", 
                    geom_pair, 
                    step_rotation_deg, 
                    range_rotation_deg, 
                    write_obj, 
                    stress_output_path, 
                    interpolation_steps,
                    bruteforce,
                    sample_size)
                        .context("process stress geometry pair(stress) failed")?;
                    Ok((geom_stress, logs_dia_stress, logs_sys_stress))
            });

            // Join REST & STRESS
            let (rest_pair, dia_logs, sys_logs) = rest_handle.join().unwrap()?;
            let (stress_pair, dia_logs_stress, sys_logs_stress) = stress_handle.join().unwrap()?;

            // Prepare diastolic & systolic geometry pairs
            let (dia_pair, sys_pair) =
                prepare_geometries_comparison(
                    rest_pair.clone(), 
                    stress_pair.clone(),
                    step_rotation_deg,
                    range_rotation_deg);
            let dia_pair_for_thread = dia_pair.clone();
            let sys_pair_for_thread = sys_pair.clone();

            // DIASTOLIC thread
            let dia_handle = if write_obj {
                Some(s.spawn(move |_| {
                    process_case(
                        "diastolic",
                        dia_pair_for_thread,
                        diastole_output_path,
                        interpolation_steps,
                    )
                    .context("process_case(diastolic) failed")
                }))
            } else {
                None
            };

            // SYSTOLIC thread
            let sys_handle = if write_obj {
                Some(s.spawn(move |_| {
                    process_case(
                        "systolic",
                        sys_pair_for_thread,
                        systole_output_path,
                        interpolation_steps,
                    )
                    .context("process_case(systolic) failed")
                }))
            } else {
                None
            };

            // Join DIASTOLIC & SYSTOLIC
            let dia_geom = if let Some(handle) = dia_handle {
                handle.join().unwrap()?
            } else {
                dia_pair // fallback: return unprocessed pair
            };
            let sys_geom = if let Some(handle) = sys_handle {
                handle.join().unwrap()?
            } else {
                sys_pair // fallback: return unprocessed pair
            };

            Ok((
                rest_pair,
                stress_pair,
                dia_geom,
                sys_geom,
                dia_logs,
                sys_logs,
                dia_logs_stress,
                sys_logs_stress,
            ))
        },
    )
    .map_err(|panic| anyhow!("Parallel processing threads panicked: {:?}", panic))?;

    let (
        rest_geom,
        stress_geom,
        dia_geom,
        sys_geom,
        dia_logs,
        sys_logs,
        dia_logs_stress,
        sys_logs_stress,
    ) = result?;

    Ok((
        (rest_geom, stress_geom, dia_geom, sys_geom),
        (dia_logs, sys_logs, dia_logs_stress, sys_logs_stress),
    ))
}

pub fn from_array_doublepair_rs(
    rest_geometry_dia: Geometry,
    rest_geometry_sys: Geometry,
    stress_geometry_dia: Geometry,
    stress_geometry_sys: Geometry,
    step_rotation_deg: f64,
    range_rotation_deg: f64,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
    write_obj: bool,
    rest_output_path: &str,
    stress_output_path: &str,
    interpolation_steps: usize,
    bruteforce: bool,
    sample_size: usize,
) -> Result<(
    (GeometryPair, GeometryPair),
    (Vec<AlignLog>, Vec<AlignLog>, Vec<AlignLog>, Vec<AlignLog>),
)> {
    let result = thread::scope(
        |s| -> Result<(
            GeometryPair,
            GeometryPair,
            Vec<AlignLog>,
            Vec<AlignLog>,
            Vec<AlignLog>,
            Vec<AlignLog>,
        )> {
            // REST thread
            let rest_handle = s.spawn(|_| -> anyhow::Result<_> {
                let geom_pair = geometry_pair_from_array_rs(
                    rest_geometry_dia,
                    rest_geometry_sys,
                    image_center,
                    radius,
                    n_points,
                )
                .context("create rest geometry pair(rest) failed")?;

                let (geom_rest, logs_dia, logs_sys) = align_within_and_between_array(
                    "rest", 
                    geom_pair, 
                    step_rotation_deg, 
                    range_rotation_deg, 
                    write_obj, 
                    rest_output_path, 
                    interpolation_steps,
                    bruteforce,
                    sample_size)
                        .context("process geometry pair(rest) failed")?;
                    Ok((geom_rest, logs_dia, logs_sys))
            });

            // STRESS thread
            let stress_handle = s.spawn(|_| -> anyhow::Result<_> {
                let geom_pair = geometry_pair_from_array_rs(
                    stress_geometry_dia,
                    stress_geometry_sys,
                    image_center,
                    radius,
                    n_points,
                )
                .context("create stress geometry pair(stress) failed")?;

                let (geom_stress, logs_dia_stress, logs_sys_stress) = align_within_and_between_array(
                    "stress", 
                    geom_pair, 
                    step_rotation_deg, 
                    range_rotation_deg, 
                    write_obj, 
                    stress_output_path, 
                    interpolation_steps,
                    bruteforce,
                    sample_size)
                        .context("process stress geometry pair(stress) failed")?;
                    Ok((geom_stress, logs_dia_stress, logs_sys_stress))
            });
            // Join threads & propagate any processing errors
            let (rest_geom_pair, dia_logs, sys_logs) = rest_handle.join().unwrap()?;
            let (stress_geom_pair, dia_logs_stress, sys_logs_stress) =
                stress_handle.join().unwrap()?;

            Ok((
                rest_geom_pair,
                stress_geom_pair,
                dia_logs,
                sys_logs,
                dia_logs_stress,
                sys_logs_stress,
            ))
        },
    )
    .map_err(|panic_payload| {
        anyhow!("Parallel processing threads panicked: {:?}", panic_payload)
    })?;

    let (rest_geom, stress_geom, dia_logs, sys_logs, dia_logs_stress, sys_logs_stress) = result?;
    Ok((
        (rest_geom, stress_geom),
        (dia_logs, sys_logs, dia_logs_stress, sys_logs_stress),
    ))
}

pub fn from_array_singlepair_rs(
    rest_geometry_dia: Geometry,
    rest_geometry_sys: Geometry,
    step_rotation_deg: f64,
    range_rotation_deg: f64,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
    write_obj: bool,
    output_path: &str,
    interpolation_steps: usize,
    bruteforce: bool,
    sample_size: usize,
) -> Result<(GeometryPair, (Vec<AlignLog>, Vec<AlignLog>))> {
    // Build the raw pair
    let geometries = geometry_pair_from_array_rs(
        rest_geometry_dia,
        rest_geometry_sys,
        image_center,
        radius,
        n_points,
    )
    .context("create geometry_pair(single) failed")?;

    let (geom_pair, dia_logs, sys_logs) = align_within_and_between_array(
        "single", 
        geometries, 
        step_rotation_deg,
        range_rotation_deg, 
        write_obj, 
        output_path, 
        interpolation_steps,
        bruteforce,
        sample_size)
        .context("process geometry_pair(single) failed")?;
    // Process it (e.g. align, interpolate, write meshes)
    let processed_pair = process_case("single", geom_pair, output_path, interpolation_steps)
        .context("process_case(single) failed")?;

    Ok((processed_pair, (dia_logs, sys_logs)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;
    use crate::io::input::{Contour, ContourPoint};

    fn create_circle_contour(id: u32, radius: f64, n_points: u32) -> Contour {
        let mut points = Vec::new();
        let centroid = (0.0, 0.0, id as f64);
        for i in 0..n_points {
            let angle = 2.0 * PI * i as f64 / n_points as f64;
            let x = radius * angle.cos();
            let y = radius * angle.sin();
            points.push(ContourPoint {
                frame_index: id,
                point_index: i,
                x,
                y,
                z: id as f64,
                aortic: false,
            });
        }
        Contour {
            id,
            points,
            centroid,
            aortic_thickness: None,
            pulmonary_thickness: None,
        }
    }

    #[test]
    fn test_refine_ordering_shuffled() {
        // Create 5 contours with increasing radii
        let radii = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let contours: Vec<Contour> = radii
            .iter()
            .enumerate()
            .map(|(i, &r)| create_circle_contour(i as u32, r, 100))
            .collect();

        // Shuffle to a nontrivial order: [1, 3, 0, 2, 4]
        let shuffle_order = vec![1, 3, 0, 2, 4];
        let shuffled_contours: Vec<Contour> = shuffle_order.iter().map(|&i| contours[i].clone()).collect();

        let geom = Geometry {
            contours: shuffled_contours,
            catheter: Vec::new(),
            walls: Vec::new(),
            reference_point: ContourPoint {
                frame_index: 0,
                point_index: 0,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            label: "test".to_string(),
        };

        // Refine ordering with high delta to prioritize ID order over Hausdorff
        let refined = refine_ordering(geom, 0.0, 5);

        // Check IDs are now in ascending order (0, 1, 2, 3, 4)
        let ids: Vec<u32> = refined.contours.iter().map(|c| c.id).collect();
        assert_eq!(ids, vec![0, 1, 2, 3, 4]);
    }
}