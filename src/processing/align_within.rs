use rayon::prelude::*;
use std::collections::HashMap;
use std::collections::HashSet;
use std::f64::consts::PI;
use std::sync::{Arc, Mutex};

use crate::io::input::{Contour, ContourPoint};
use crate::io::Geometry;
use crate::processing::align_between::GeometryPair;
use crate::processing::process_utils::downsample_contour_points;

#[derive(Debug)]
pub struct AlignLog {
    pub contour_id: u32,
    pub matched_to: u32,
    pub rel_rot_deg: f64,
    pub total_rot_deg: f64,
    pub tx: f64,
    pub ty: f64,
    pub centroid: (f64, f64),
}

pub fn align_frames_in_geometries(
    geom_pair: GeometryPair,
    step_deg: f64,
    range_deg: f64,
    smooth: bool,
    bruteforce: bool,
    sample_size: usize,
) -> anyhow::Result<(GeometryPair, (Vec<AlignLog>, Vec<AlignLog>))> {
    let (diastole, dia_logs) = align_frames_in_geometry(
        geom_pair.dia_geom, 
        step_deg, 
        range_deg, 
        smooth,
        bruteforce,
        sample_size,
    );
    let (mut systole, sys_logs) = align_frames_in_geometry(
        geom_pair.sys_geom, 
        step_deg, 
        range_deg, 
        smooth,
        bruteforce,
        sample_size,
    );

    GeometryPair::translate_contours_to_match(&diastole, &mut systole);
    GeometryPair::apply_z_transformation(&diastole, &mut systole);
    let geom_pair_clean = GeometryPair {
        dia_geom: diastole,
        sys_geom: systole,
    };

    Ok((geom_pair_clean, (dia_logs, sys_logs)))
}

pub fn align_frames_in_geometry(
    geometry: Geometry,
    step_deg: f64,
    range_deg: f64,
    smooth: bool,
    bruteforce: bool,
    sample_size: usize,
) -> (Geometry, Vec<AlignLog>) {
    let (mut geometry, reference_index, reference_pos, ref_contour) = prep_data_geometry(geometry);

    let (p1, p2, updated_ref) = assign_aortic(ref_contour.clone(), &geometry);
    let ref_contour = updated_ref.clone();

    let (_line_angle, rotation_to_y, rotated_ref, aortic_flag) =
        rotate_reference_contour(p1, p2, ref_contour.clone());

    // Update the reference contour in geometry with rotated version
    geometry.contours.insert(reference_pos, rotated_ref.clone());

    // prepare reference catheter
    for catheter in geometry.catheter.iter_mut() {
        if catheter.id == reference_index {
            catheter.rotate_contour_around_point(
                rotation_to_y,
                (ref_contour.centroid.0, ref_contour.centroid.1),
            );
            catheter.sort_contour_points();
            break;
        }
    }

    let logger = Arc::new(Mutex::new(Vec::<AlignLog>::new()));

    let reference = if rotated_ref.points.len() > sample_size {
        let frac = sample_size / rotated_ref.points.len();
        let mut combined = downsample_contour_points(&rotated_ref.points, sample_size);
        if let Some(catheter) = geometry.catheter.iter().find(|c| c.id == reference_index) {
            let n_cath = catheter.points.len() * frac as usize;
            let downsampled_catheter = downsample_contour_points(&catheter.points, n_cath);
            combined.extend_from_slice(&downsampled_catheter);
        }
        combined
    } else {
        let mut combined = rotated_ref.points.clone();
        if let Some(catheter) = geometry.catheter.iter().find(|c| c.id == reference_index) {
            combined.extend_from_slice(&catheter.points);
        }
        combined
    };

    let reference_contour = Contour {
        id: reference_index,
        points: reference,
        centroid: rotated_ref.centroid,
        aortic_thickness: rotated_ref.aortic_thickness,
        pulmonary_thickness: rotated_ref.pulmonary_thickness,
    };

    let (mut geometry, id_translation) = align_remaining_contours(
        geometry,
        reference_index,
        reference_contour,
        rotation_to_y,
        step_deg,
        range_deg,
        Arc::clone(&logger),
        bruteforce,
        sample_size,
    );

    for catheter in geometry.catheter.iter_mut() {
        for (id, translation, best_rot, center) in &id_translation {
            if catheter.id == *id {
                catheter.translate_contour((-translation.0, -translation.1, translation.2));
                catheter.rotate_contour_around_point(*best_rot, *center);
                catheter.sort_contour_points();
                break;
            }
        }
    }
    println!("Processing Geometry: {}", &geometry.label);
    println!("Reference angle to vertical: {:.1} (°) \n Rotating Reference by: {:.1} (°) \n Added additional 180° rotation: {}", _line_angle.to_degrees(), rotation_to_y.to_degrees(), aortic_flag);
    // dump the collected logs as a table
    let logs = Arc::try_unwrap(logger)
        .expect("No other Arc references to logger exist")
        .into_inner()
        .expect("Logger mutex was poisoned");
    dump_table(&logs);
    let geometry = if smooth {
        geometry.smooth_contours()
    } else {
        geometry.clone()
    };
    (geometry, logs)
}

fn prep_data_geometry(mut geometry: Geometry) -> (Geometry, u32, usize, Contour) {
    geometry.contours.sort_by_key(|contour| contour.id);

    for contour in &mut geometry.contours {
        contour.sort_contour_points();
    }

    for catheter in &mut geometry.catheter {
        catheter.sort_contour_points();
    }

    // Use the contour with the highest frame index as reference.
    let reference_index = geometry
        .contours
        .iter()
        .map(|contour| contour.id)
        .max()
        .unwrap();
    let reference_pos = geometry
        .contours
        .iter()
        .position(|contour| contour.id == reference_index)
        .expect("Reference contour not found");
    let ref_contour = &mut geometry.contours.remove(reference_pos);

    (
        geometry,
        reference_index,
        reference_pos,
        ref_contour.clone(),
    )
}

/// expects: a reference Contour and the Geometry the Contour is derived from
/// returns: farthest points and Contour with assigned aortic bool
fn assign_aortic(contour: Contour, geometry: &Geometry) -> (ContourPoint, ContourPoint, Contour) {
    let ((p1, p2), _dist) = contour.find_farthest_points();

    let p1_pos = contour.points.iter().position(|pt| pt == p1).unwrap();
    let p2_pos = contour.points.iter().position(|pt| pt == p2).unwrap();

    let (first_half_indices, second_half_indices) = if p1_pos < p2_pos {
        (
            (p1_pos..=p2_pos).collect::<HashSet<_>>(),
            (0..p1_pos)
                .chain(p2_pos + 1..contour.points.len())
                .collect::<HashSet<_>>(),
        )
    } else {
        (
            (p1_pos..contour.points.len())
                .chain(0..=p2_pos)
                .collect::<HashSet<_>>(),
            (p2_pos + 1..p1_pos).collect::<HashSet<_>>(),
        )
    };

    // Compute distances first — no borrows beyond this point
    let dist_first = first_half_indices
        .iter()
        .map(|&i| contour.points[i].distance_to(&geometry.reference_point))
        .sum::<f64>();

    let dist_second = second_half_indices
        .iter()
        .map(|&i| contour.points[i].distance_to(&geometry.reference_point))
        .sum::<f64>();

    let use_first = dist_first < dist_second;

    // borrow checker complained, maybe there's a better way
    let mut new_contour = contour.clone();

    for (i, pt) in new_contour.points.iter_mut().enumerate() {
        pt.aortic = if use_first {
            first_half_indices.contains(&i)
        } else {
            second_half_indices.contains(&i)
        };
    }

    (p1.clone(), p2.clone(), new_contour)
}

/// takes a contour (should be contour with highest index)
/// aligns it vertically, and ensures aortic is to the right
/// returns the angle, rotation and the new contour
fn rotate_reference_contour(
    p1: ContourPoint,
    p2: ContourPoint,
    contour: Contour,
) -> (f64, f64, Contour, bool) {
    let dx = p2.x - p1.x;
    let dy = p2.y - p1.y;
    let line_angle = dy.atan2(dx);
    let mut rotation_to_y = (PI / 2.0) - line_angle;

    // Normalize the rotation angle to [0, 2π)
    rotation_to_y = rotation_to_y.rem_euclid(2.0 * PI);

    let mut rotated_ref = contour.clone();
    rotated_ref.rotate_contour(rotation_to_y);
    rotated_ref.sort_contour_points();
    let ((p3, p4), _dist) = rotated_ref.find_closest_opposite();
    // Determine which point is aortic
    let (aortic_pt, non_aortic_pt) = if p3.aortic { (&p3, &p4) } else { (&p4, &p3) };

    let mut aortic_flag = false;

    if aortic_pt.x < non_aortic_pt.x {
        rotation_to_y += PI;
        // re-normalize into [0, 2π) to never exceed 360°
        rotation_to_y = rotation_to_y.rem_euclid(2.0 * PI);
        rotated_ref.rotate_contour(PI);
        rotated_ref.sort_contour_points();
        aortic_flag = true;
    }

    (line_angle, rotation_to_y, rotated_ref, aortic_flag)
}

fn align_remaining_contours(
    mut geometry: Geometry,
    ref_idx: u32,
    ref_contour: Contour,
    rot: f64,
    step_deg: f64,
    range_deg: f64,
    logger: Arc<Mutex<Vec<AlignLog>>>,
    bruteforce: bool,
    sample_size: usize,
) -> (Geometry, Vec<(u32, (f64, f64, f64), f64, (f64, f64))>) {
    let mut processed_refs: HashMap<u32, (Vec<ContourPoint>, (f64, f64, f64))> =
        std::collections::HashMap::new();
    let mut id_translation = Vec::new();
    // this tracks a general rotation over a pullback
    // while frames are aligned based on local rotation
    let mut cum_rot = rot;
    // needed to adjust catheter, sample contour currently hardcoded
    let frac = sample_size / geometry.contours[0].points.len();
    let n_cath = geometry.catheter[0].points.len() * frac as usize;

    // Process contours in reverse order (highest ID first)
    for contour in geometry.contours.iter_mut().rev() {
        if contour.id == ref_idx {
            continue;
        }

        // Determine reference points and centroid
        let (base_ref_points, ref_centroid) = match processed_refs.get(&(contour.id + 1)) {
            Some((points, centroid)) => (points.clone(), *centroid),
            None => (ref_contour.points.clone(), ref_contour.centroid),
        };

        // Include corresponding catheter points in reference set
        let mut ref_points = base_ref_points;
        if let Some(cat) = geometry.catheter.iter().find(|c| c.id == contour.id + 1) {
            let cat_points = if ref_points.len() > sample_size {
                downsample_contour_points(&cat.points, n_cath)
            } else {
                cat.points.clone()
            };
            ref_points.extend_from_slice(&cat_points);
        }

        ref_points = if ref_points.len() > sample_size {
            downsample_contour_points(&ref_points, sample_size)
        } else {
            ref_points
        };

        contour.rotate_contour(cum_rot);

        // Calculate translation
        let tx = contour.centroid.0 - ref_centroid.0;
        let ty = contour.centroid.1 - ref_centroid.1;

        contour.translate_contour((-tx, -ty, 0.0));

        // Add the catheter points to the contour points, and then find the best rotation based on both
        // Later also add wall points
        // Find the catheter with the same id as the current contour
        let target = if let Some(catheter) = geometry.catheter.iter().find(|c| c.id == contour.id) {
            let combined = contour.points.clone();
            let (mut combined, cat_pts) = if combined.len() > sample_size && !catheter.points.is_empty() {
                (downsample_contour_points(&combined, sample_size),
                downsample_contour_points(&catheter.points, n_cath))
            } else {
                (combined, catheter.points.clone())
            };
            combined.extend_from_slice(&cat_pts);
            combined
        } else {
            contour.points.clone()
        };

        // Optimize rotation
        let best_rel_rot = if bruteforce {
            search_range(&ref_points, &target, step_deg, range_deg, &contour.centroid, None, range_deg)
        } else {
            find_best_rotation(&ref_points, &target, step_deg, range_deg, &contour.centroid)
        };

        contour.rotate_contour(best_rel_rot);
        contour.sort_contour_points();

        cum_rot += best_rel_rot;

        // Store transformation data for later use (speed up)
        id_translation.push((
            contour.id,
            (tx, ty, 0.0),
            cum_rot,
            (contour.centroid.0, contour.centroid.1),
        ));

        let entry = AlignLog {
            contour_id: contour.id,
            matched_to: contour.id + 1,
            rel_rot_deg: best_rel_rot.to_degrees(),
            total_rot_deg: cum_rot.to_degrees(),
            tx,
            ty,
            centroid: (contour.centroid.0, contour.centroid.1),
        };
        logger.lock().unwrap().push(entry);

        processed_refs.insert(contour.id, (contour.points.clone(), contour.centroid));

        let half_len = contour.points.len() / 2;
        for pt in contour.points.iter_mut().skip(half_len) {
            pt.aortic = true;
        }
    }

    (geometry, id_translation)
}

pub fn find_best_rotation(
    reference: &[ContourPoint],
    target: &[ContourPoint],
    step_deg: f64,
    range_deg: f64,
    centroid: &(f64, f64, f64),
) -> f64 {
    match step_deg {
        1.0..=f64::INFINITY => {
            search_range(reference, target, step_deg, range_deg, centroid, None, range_deg)
        }
        0.1..1.0 => {
            let coarse_angle = search_range(reference, target, 1.0, range_deg, centroid, None, range_deg);
            let range = if range_deg > 5.0 {5.0} else {range_deg};
            search_range(reference, target, step_deg, range, centroid, Some(coarse_angle), range_deg)
        }
        0.01..0.1 => {
            let coarse_angle = search_range(reference, target, 1.0, range_deg, centroid, None, range_deg);
            let range = if range_deg > 5.0 {5.0} else {range_deg};
            let medium_angle = search_range(reference, target, 0.1, range, centroid, Some(coarse_angle), range_deg);
            let range_small = if range_deg > 10.0 * step_deg {10.0 * step_deg} else {range_deg};
            search_range(reference, target, step_deg, range_small, centroid, Some(medium_angle), range_deg)
        }
        _ => {
            let coarse_angle = search_range(reference, target, 1.0, range_deg, centroid, None, range_deg);
            let range = if range_deg > 5.0 {5.0} else {range_deg};            
            let medium_angle = search_range(reference, target, 0.1, range, centroid, Some(coarse_angle), range_deg);
            let range_small = if range_deg > 0.1 {0.1} else {range_deg};
            let fine_angle = search_range(reference, target, 0.01, range_small, centroid, Some(medium_angle), range_deg);
            let range_fine = if range_deg > 10.0 * step_deg {10.0 * step_deg} else {range_deg};
            search_range(reference, target, step_deg, range_fine, centroid, Some(fine_angle), range_deg)
        }
    }
}

pub fn search_range(
    reference: &[ContourPoint],
    target: &[ContourPoint],
    step_deg: f64,
    range_deg: f64,
    centroid: &(f64, f64, f64),
    center_angle: Option<f64>,
    limes_deg: f64,
) -> f64 {
    let range_rad = range_deg.to_radians();
    let step_rad = step_deg.to_radians();
    if step_rad <= 0.0 { return 0.0; }

    let center = center_angle.unwrap_or(0.0);
    let limes = limes_deg.to_radians();
    let lower_limes = -limes;

    // keep linear domain first, clamp in that domain
    let mut start_angle = center - range_rad;
    let stop_angle = center + range_rad;
    start_angle = start_angle.max(lower_limes);
    let stop_angle = stop_angle.min(limes);

    if stop_angle <= start_angle {
        return ((center + PI).rem_euclid(2.0 * PI)) - PI;
    }

    let steps = (((stop_angle - start_angle) / step_rad).ceil() as usize).max(1);

    let mut angle_dist_pairs = Vec::with_capacity(steps);
    for i in 0..=steps {
        let angle_lin = start_angle + (i as f64) * step_rad;
        if angle_lin > stop_angle { break; }

        // normalize for rotation (rem_euclid to [0,2π) then map if needed)
        let angle = angle_lin.rem_euclid(2.0 * PI);
        let mapped_angle = ((angle + PI).rem_euclid(2.0 * PI)) - PI;

        let rotated: Vec<ContourPoint> = target
            .par_iter()
            .map(|p| p.rotate_point(angle, (centroid.0, centroid.1)))
            .collect();

        let hausdorff_dist = hausdorff_distance(reference, &rotated);
        angle_dist_pairs.push((mapped_angle, hausdorff_dist));
    }

    if angle_dist_pairs.is_empty() {
        return ((center + PI).rem_euclid(2.0 * PI)) - PI;
    }

    let (min_angle, _min_dist) = angle_dist_pairs
        .iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();

    *min_angle
}

/// Computes the Hausdorff distance between two point sets.
pub fn hausdorff_distance(set1: &[ContourPoint], set2: &[ContourPoint]) -> f64 {
    let forward = directed_hausdorff(set1, set2);
    let backward = directed_hausdorff(set2, set1);
    forward.max(backward) // Hausdorff distance is max of both directed distances
}

/// Computes directed Hausdorff distance from A to B
fn directed_hausdorff(contour_a: &[ContourPoint], contour_b: &[ContourPoint]) -> f64 {
    contour_a
        .par_iter() // Use parallel iteration
        .map(|pa| {
            contour_b
                .iter()
                .map(|pb| {
                    let dx = pa.x - pb.x;
                    let dy = pa.y - pb.y;
                    (dx * dx + dy * dy).sqrt()
                })
                .fold(std::f64::MAX, f64::min) // Directly find min without storing a Vec
        })
        .reduce(|| 0.0, f64::max) // Directly find max without extra allocation
}

fn dump_table(logs: &[AlignLog]) {
    // 1) Decide on column headers and collect rows as strings
    let headers = [
        "Contour",
        "Matched To",
        "Relative Rot (°)",
        "Total Rot (°)",
        "Tx",
        "Ty",
        "Centroid",
    ];
    let rows: Vec<[String; 7]> = logs
        .iter()
        .map(|e| {
            [
                e.contour_id.to_string(),
                e.matched_to.to_string(),
                format!("{:.2}", e.rel_rot_deg),
                format!("{:.2}", e.total_rot_deg),
                format!("{:.2}", e.tx),
                format!("{:.2}", e.ty),
                format!("({:.2},{:.2})", e.centroid.0, e.centroid.1),
            ]
        })
        .collect();

    // 2) Compute max width for each of the 7 columns
    let mut widths = [0usize; 7];
    for (i, &h) in headers.iter().enumerate() {
        widths[i] = h.len();
    }
    for row in &rows {
        for (i, cell) in row.iter().enumerate() {
            widths[i] = widths[i].max(cell.len());
        }
    }

    // 3a) Left‑align any data row
    fn print_row(cells: &[String], widths: &[usize]) {
        print!("|");
        for (i, cell) in cells.iter().enumerate() {
            let pad = widths[i] - cell.len();
            print!(" {}{} |", cell, " ".repeat(pad));
        }
        println!();
    }

    // 3b) Center a header row
    fn print_header(cells: &[String], widths: &[usize]) {
        print!("|");
        for (i, cell) in cells.iter().enumerate() {
            let total_pad = widths[i] - cell.len();
            let left = total_pad / 2;
            let right = total_pad - left;
            print!(" {}{}{} |", " ".repeat(left), cell, " ".repeat(right));
        }
        println!();
    }

    // 4) Top border
    print!("+");
    for w in &widths {
        print!("{}+", "-".repeat(w + 2));
    }
    println!();

    // 5) Header row
    let header_cells: Vec<String> = headers.iter().map(|&s| s.to_string()).collect();
    print_header(&header_cells, &widths);

    // 6) Separator
    print!("+");
    for w in &widths {
        print!("{}+", "-".repeat(w + 2));
    }
    println!();

    // 7) Data rows
    for row in &rows {
        print_row(&row.to_vec(), &widths);
    }

    // 8) Bottom border
    print!("+");
    for w in &widths {
        print!("{}+", "-".repeat(w + 2));
    }
    println!();
}

#[cfg(test)]
mod contour_tests {
    use super::*;
    use crate::utils::test_utils::{generate_ellipse_points, new_dummy_contour};
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_assign_aortic_splits_correctly() {
        let major = 5.0;
        let minor = 2.5;
        let num_points = 501; // Changed to odd number to avoid split at y=0
        let points = generate_ellipse_points(major, minor, num_points, 0.0, (0.0, 0.0), 0);
        let contour = Contour {
            id: 1,
            points,
            centroid: (0.0, 0.0, 0.0),
            aortic_thickness: None,
            pulmonary_thickness: None,
        };
        let geometry = Geometry {
            contours: vec![],
            catheter: vec![],
            walls: vec![],
            reference_point: ContourPoint {
                frame_index: 0,
                point_index: 0,
                x: 0.0,
                y: 5.0,
                z: 0.0,
                aortic: false,
            },
            label: "test".to_string(),
        };
        let (_p1, _p2, updated_contour) = assign_aortic(contour, &geometry);
        let aortic_points: Vec<_> = updated_contour.points.iter().filter(|p| p.aortic).collect();
        let non_aortic_points: Vec<_> = updated_contour
            .points
            .iter()
            .filter(|p| !p.aortic)
            .collect();
        let aortic_count = aortic_points.len();

        assert!(
            (aortic_count == num_points / 2 || aortic_count == (num_points / 2) + 1),
            "Expected approximately half points to be aortic, got {}",
            aortic_count
        );
        assert!(
            aortic_points.iter().all(|p| p.y > -0.001),
            "Aortic points should be in upper half (y > 0)"
        );
        assert!(
            non_aortic_points.iter().all(|p| p.y <= 0.001),
            "Non-aortic points should be in lower half (y <= 0)"
        );
    }

    #[test]
    fn test_rotate_reference_contour_aligns_aortic_right() {
        let major = 5.0;
        let minor = 2.5;
        let num_points = 501;
        let points = generate_ellipse_points(major, minor, num_points, 0.0, (0.0, 0.0), 0);
        let mut contour = Contour {
            id: 1,
            points,
            centroid: (0.0, 0.0, 0.0),
            aortic_thickness: None,
            pulmonary_thickness: None,
        };
        contour.sort_contour_points();
        let geometry = Geometry {
            contours: vec![],
            catheter: vec![],
            walls: vec![],
            reference_point: ContourPoint {
                frame_index: 0,
                point_index: 0,
                x: 15.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            label: "test".to_string(),
        };
        let (p1, p2, contour_with_aortic) = assign_aortic(contour, &geometry);
        let (_, rotation, rotated_contour, _) =
            rotate_reference_contour(p1, p2, contour_with_aortic);
        // Check rotation is applied correctly
        assert_relative_eq!(rotation, 3.0 * PI / 2.0, epsilon = 1e-2);
        // Aortic points should be on the right (x > 0)
        let aortic_right = rotated_contour
            .points
            .iter()
            .filter(|p| p.aortic)
            .all(|p| p.x > 0.0);
        assert!(
            aortic_right,
            "Aortic points should be on the right after rotation"
        );
    }

    #[test]
    fn test_align_remaining_contours() {
        let major = 5.0;
        let minor = 2.5;
        let num_points = 501;

        // Reference contour (id 3)
        let ref_points = generate_ellipse_points(major, minor, num_points, 0.0, (0.0, 0.0), 0);
        let ref_contour = Contour {
            id: 3,
            points: ref_points,
            centroid: (0.0, 0.0, 0.0),
            aortic_thickness: None,
            pulmonary_thickness: None,
        };

        // Contour 2: rotated 30 degrees, translated to (5,5)
        let contour2_points =
            generate_ellipse_points(major, minor, num_points, 30_f64.to_radians(), (5.0, 5.0), 2);
        let contour2 = Contour {
            id: 2,
            points: contour2_points,
            centroid: (5.0, 5.0, 0.0),
            aortic_thickness: None,
            pulmonary_thickness: None,
        };

        // Contour 1: rotated 60 degrees, translated to (10,10)
        let contour1_points = generate_ellipse_points(
            major,
            minor,
            num_points,
            60_f64.to_radians(),
            (10.0, 10.0),
            1,
        );
        let contour1 = Contour {
            id: 1,
            points: contour1_points,
            centroid: (10.0, 10.0, 0.0),
            aortic_thickness: None,
            pulmonary_thickness: None,
        };
        // Create dummy catheters for each contour
        let catheters = vec![
            new_dummy_contour(1),
            new_dummy_contour(2),
            new_dummy_contour(3),
        ];

        let geometry = Geometry {
            contours: vec![contour1, contour2, ref_contour.clone()],
            catheter: catheters,
            walls: vec![],
            reference_point: ContourPoint {
                frame_index: 0,
                point_index: 0,
                x: 15.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            label: "test".to_string(),
        };

        let (aligned_geometry, _) = align_frames_in_geometry(geometry, 1.0, PI, true, true, 200);

        // Check centroids are aligned to reference (0,0)
        for contour in aligned_geometry.contours {
            if contour.id == 3 {
                continue; // Skip reference
            }
            assert_relative_eq!(contour.centroid.0, 0.0, epsilon = 0.5);
            assert_relative_eq!(contour.centroid.1, 0.0, epsilon = 0.5);
        }
    }

    #[test]
    fn test_catheter_transformations() {
        use approx::assert_relative_eq;
        use std::f64::consts::PI;

        // 1. Setup the three contours (ids 1,2,3) exactly as before.
        let major = 5.0;
        let minor = 2.5;
        let num_points = 501;

        let contours_data = vec![
            (1, 5.5, 5.5, 60.0_f64.to_radians()),
            (2, 3.5, 3.5, 30.0_f64.to_radians()),
            (3, 4.5, 4.5, 0.0),
        ];

        let mut test_contours = Vec::new();
        let mut test_catheters = Vec::new();

        for (id, cx, cy, rot) in contours_data {
            // generate ellipse points for the contour
            let points = generate_ellipse_points(major, minor, num_points, rot, (cx, cy), id);
            let pts = points.clone();
            let contour = Contour {
                id,
                points,
                centroid: Contour::compute_centroid(&pts),
                aortic_thickness: None,
                pulmonary_thickness: None,
            };
            // generate its catheter (they all start around (4.5,4.5))
            let catheter_contours =
                Contour::create_catheter_contours(&contour.points, (4.5, 4.5), 0.5, 20)
                    .expect("catheter fail");
            test_contours.push(contour);
            test_catheters.extend(catheter_contours);
        }

        // 2. Build the geometry and align everything
        let geometry = Geometry {
            contours: test_contours,
            catheter: test_catheters,
            walls: vec![],
            reference_point: ContourPoint {
                frame_index: 0,
                point_index: 0,
                x: 6.0,
                y: 4.5,
                z: 0.0,
                aortic: false,
            },
            label: "test".to_string(),
        };
        let geom_old = geometry.clone();
        let (aligned, _) = align_frames_in_geometry(geometry, 1.0, PI, true, true, 200);

        for contour in &aligned.catheter {
            // skip the reference if you like, but we’ll test all three
            // find its matching catheter
            let mut catheter = geom_old
                .catheter
                .iter()
                .find(|c| c.id == contour.id)
                .expect("missing catheter")
                .clone();

            let (exp_tx, exp_ty, exp_angle) = match catheter.id {
                1 => (-1.0, -1.0, 60.0),
                2 => (1.0, 1.0, 30.0),
                3 => (0.0, 0.0, 0.0),
                _ => panic!("Unexpected catheter id"),
            };

            // translate_contour and rotate_contour_around_point are tested functions
            catheter.translate_contour((exp_tx, exp_ty, 0.0));
            catheter.rotate_contour_around_point((exp_angle as f64).to_radians(), (4.5, 4.5));

            assert_relative_eq!(contour.centroid.0, catheter.centroid.0, epsilon = 1e-6);
            assert_relative_eq!(contour.centroid.1, catheter.centroid.1, epsilon = 1e-6);

            // sanity: radius around center must remain 0.5
        }
    }
}
