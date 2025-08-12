use anyhow::{anyhow, Context, Result};
use crossbeam::thread;

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
    let geometry = if let Some(recs) = records.as_ref() {
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

pub fn refine_ordering(mut geom: Geometry, delta: f64, max_rounds: usize) -> Geometry {
    let mut last_order = Vec::new();
    for _round in 0..max_rounds {
        let cost = build_cost_matrix(&geom.contours, delta);
        let order = if geom.contours.len() <= 15 {
            held_karp(&cost)
        } else {
            two_opt(&cost, 500)
        };
        if order == last_order {
            break;
        }
        last_order = order.clone();
        geom = reorder_geometry(geom, &order);
    }
    geom
}

fn reorder_geometry(mut geom: Geometry, new_order: &[usize]) -> Geometry {
    let reordered_contours = new_order
        .iter()
        .map(|&i| geom.contours[i].clone())
        .collect();
    let reordered_catheter = new_order
        .iter()
        .map(|&i| geom.catheter[i].clone())
        .collect();
    geom.contours = reordered_contours;
    geom.catheter = reordered_catheter;
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
                let jump = (contours[i].id as i32 - contours[j].id as i32).abs() as f64;
                cost[i][j] = h + delta * jump;
            }
        }
    }
    cost
}

/// Solve the shortest Hamiltonian path from 0 through all nodes exactly.
fn held_karp(cost: &[Vec<f64>]) -> Vec<usize> {
    let n = cost.len();
    let full_mask = (1 << n) - 1;
    // dp[mask][j] = best cost to start at 0, visit mask, end at j
    let mut dp = vec![vec![f64::INFINITY; n]; 1 << n];
    let mut parent = vec![vec![None; n]; 1 << n];

    dp[1 << 0][0] = 0.0;
    for mask in 1..=full_mask {
        if (mask & 1) == 0 {
            continue;
        } // must always include node 0
        for j in 0..n {
            if mask & (1 << j) == 0 {
                continue;
            }
            if j == 0 && mask != (1 << 0) {
                continue;
            }
            let prev_mask = mask ^ (1 << j);
            for i in 0..n {
                if prev_mask & (1 << i) == 0 {
                    continue;
                }
                let cost_ij = cost[i][j];
                let cand = dp[prev_mask][i] + cost_ij;
                if cand < dp[mask][j] {
                    dp[mask][j] = cand;
                    parent[mask][j] = Some(i);
                }
            }
        }
    }
    // Pick endpoint with minimal cost
    let (mut end, _) = (1..n)
        .map(|j| (j, dp[full_mask][j]))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();
    // Reconstruct path
    let mut order = Vec::with_capacity(n);
    let mut mask = full_mask;
    while let Some(p) = parent[mask][end] {
        order.push(end);
        mask ^= 1 << end;
        end = p;
    }
    order.push(0);
    order.reverse();
    order
}

/// Start with identity ordering, then do repeated 2‑opt swaps
fn two_opt(cost: &[Vec<f64>], max_iters: usize) -> Vec<usize> {
    let n = cost.len();
    let mut order: Vec<usize> = (0..n).collect();

    let mut best_impr = true;
    for _ in 0..max_iters {
        if !best_impr {
            break;
        }
        best_impr = false;
        for a in 1..n - 1 {
            for b in a + 1..n {
                // compute delta cost of reversing order[a..=b]
                let i = order[a - 1];
                let j = order[a];
                let k = order[b];
                let l = if b + 1 < n { order[b + 1] } else { usize::MAX };

                let before = cost[i][j] + if l != usize::MAX { cost[k][l] } else { 0.0 };
                let after = cost[i][k] + if l != usize::MAX { cost[j][l] } else { 0.0 };
                if after + 1e-9 < before {
                    order[a..=b].reverse();
                    best_impr = true;
                }
            }
        }
    }
    order
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
    let geometry_dia = Geometry {
        contours: geometry_dia.contours,
        catheter: catheters_dia,
        walls: geometry_dia.walls,
        reference_point: geometry_dia.reference_point,
        label: "Diastole".to_string(),
    };
    let geometry_sys = Geometry {
        contours: geometry_sys.contours,
        catheter: catheters_sys,
        walls: geometry_sys.walls,
        reference_point: geometry_sys.reference_point,
        label: "Systole".to_string(),
    };

    let mut geometries = GeometryPair {
        dia_geom: geometry_dia,
        sys_geom: geometry_sys,
    };
    geometries = geometries.adjust_z_coordinates();
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
                prepare_geometries_comparison(rest_pair.clone(), stress_pair.clone());
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
