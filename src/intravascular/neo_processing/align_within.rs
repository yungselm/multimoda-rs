use anyhow::anyhow;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::sync::{Arc, Mutex};

use super::wall::create_wall_frames;
use crate::intravascular::io::input::ContourPoint;
use crate::intravascular::io::geometry::{ContourType, Frame, Geometry};
use crate::intravascular::neo_processing::process_utils::downsample_contour_points;

#[derive(Debug)]
pub struct AlignLog {
    pub contour_id: u32,
    pub matched_to: u32,
    pub rot_deg: f64,
    pub tx: f64,
    pub ty: f64,
    pub centroid: (f64, f64),
}

pub fn align_frames_in_geometry(
    geometry: &mut Geometry,
    step_deg: f64,
    range_deg: f64,
    smooth: bool,
    bruteforce: bool,
    sample_size: usize,    
) -> anyhow::Result<(Geometry, Vec<AlignLog>)> {
    if geometry.frames.is_empty() {
        return Err(anyhow!("Geometry contains no frames"));
    }
    if geometry.frames[0].lumen.points.is_empty() {
        return Err(anyhow!("Lumen contours have no points"));
    }
    if sample_size == 0 {
        return Err(anyhow!("sample_size must be > 0"));
    }

    let ref_idx = geometry.
        find_ref_frame_idx().
        unwrap_or(geometry.find_proximal_end_idx()) as usize;
    let sample_ratio = sample_size as f64 / geometry.frames[0].lumen.points.len() as f64;
    let sample_size_catheter = if geometry.frames[0].extras.contains_key(&ContourType::Catheter) {
        Some((geometry.frames[0].extras[&ContourType::Catheter].points.len() as f64 * sample_ratio).ceil() as usize)
    } else {
        None
    };

    let logger = Arc::new(Mutex::new(Vec::<AlignLog>::new()));

    for i in 1..geometry.frames.len() {
        let (prev_frames, curr_frames) = geometry.frames.split_at_mut(i);
        let current = &mut curr_frames[0];
        let previous = &prev_frames[i-1];

        // TODO: Later maybe add option to move first contour to (0.0, 0.0, 0.0)
        let translation = (
            previous.centroid.0 - current.centroid.0,
            previous.centroid.1 - current.centroid.1,
            0.0,
        );

        current.translate_frame(translation);

        let _testing_points = catheter_lumen_vec_from_frames(
            current, 
            sample_size, 
            sample_size_catheter);
        let _reference_points = catheter_lumen_vec_from_frames(
            &previous, 
            sample_size, 
            sample_size_catheter);
        
        let best_rotation = if bruteforce {
            search_range(
                &_reference_points, 
                &_testing_points,
                step_deg, 
                range_deg, 
                &current.centroid, 
                None, 
                range_deg)
        } else {
            find_best_rotation(
                &_reference_points, 
                &_testing_points, 
                step_deg, 
                range_deg, 
                &current.centroid)
        };
        current.rotate_frame(best_rotation);

        let new_log = AlignLog {
            contour_id: current.id,
            matched_to: previous.id,
            rot_deg: best_rotation.to_degrees(),
            tx: translation.0,
            ty: translation.1,
            centroid: (current.centroid.0, current.centroid.1),
        };
        logger.lock().unwrap().push(new_log);
    };
    let anomalous_bool = is_anomalous_coronary(&geometry.frames[ref_idx]);
    let additional_rotation = angle_ref_point_to_right(
        &geometry.frames[ref_idx],
        anomalous_bool)?;
    for frame in geometry.frames.iter_mut() {
        frame.rotate_frame(additional_rotation);
        frame.sort_frame_points();
    }

    let mut final_geometry = if anomalous_bool {
        assign_aortic(geometry.clone())
    } else {
        geometry.clone()
    };

    let wall_frames = create_wall_frames(&final_geometry.frames, false);
    final_geometry = Geometry {
        frames: wall_frames,
        label: final_geometry.label,
    };

    if smooth {
        final_geometry = final_geometry.smooth_frames();
    }

    let logs = Arc::try_unwrap(logger)
        .expect("No other Arc references to logger exist")
        .into_inner()
        .expect("Logger mutex was poisoned");
    dump_table(&logs);

    Ok((final_geometry, logs))
}

fn catheter_lumen_vec_from_frames(
    frame: &Frame, 
    sample_size_lumen: usize, 
    sample_size_catheter: Option<usize>
) -> Vec<ContourPoint> {
    let mut lumen_points = downsample_contour_points(&frame.lumen.points, sample_size_lumen);
    let mut catheter_points = if let Some(sample_size_catheter) = sample_size_catheter {
        if let Some(catheter_contour) = frame.extras.get(&ContourType::Catheter) {
            downsample_contour_points(&catheter_contour.points, sample_size_catheter)
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };
    lumen_points.append(&mut catheter_points);
    lumen_points
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
    forward.max(backward)
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

// // TODO implement this more efficient hausdorff version
// fn directed_hausdorff(contour_a: &[ContourPoint], contour_b: &[ContourPoint]) -> f64 {
//     // Keep behavior simple for empty inputs (match prior behavior -> 0.0)
//     if contour_a.is_empty() || contour_b.is_empty() {
//         return 0.0;
//     }

//     // Decide chunk size based on number of threads to create many tasks but not too many
//     let threads = rayon::current_num_threads().max(1);
//     // make several chunks per thread for load balancing
//     let chunks_per_thread = 4;
//     let chunk_size = ((contour_a.len() + threads * chunks_per_thread - 1)
//         / (threads * chunks_per_thread))
//         .max(1);

//     // For each chunk, compute the local maximum of the minimum squared distances
//     let max_sq = contour_a
//         .par_chunks(chunk_size)
//         .map(|chunk| {
//             let mut local_max_sq = 0.0_f64;
//             for pa in chunk {
//                 // find min squared distance from pa to any pb (sequential inside chunk)
//                 let mut min_sq = f64::INFINITY;
//                 for pb in contour_b.iter() {
//                     let dx = pa.x - pb.x;
//                     let dy = pa.y - pb.y;
//                     let d2 = dx * dx + dy * dy;
//                     if d2 < min_sq {
//                         min_sq = d2;
//                     }
//                 }
//                 if min_sq.is_finite() && min_sq > local_max_sq {
//                     local_max_sq = min_sq;
//                 }
//             }
//             local_max_sq
//         })
//         .reduce(|| 0.0_f64, f64::max);

//     max_sq.sqrt()
// }

fn is_anomalous_coronary(ref_frame: &Frame) -> bool {
    // clinical definition is >1.3 but use 2.0 here to avoid false positives
    ref_frame.lumen.elliptic_ratio() > 2.0
        || ref_frame.lumen.aortic_thickness.is_some()
        || ref_frame.lumen.pulmonary_thickness.is_some()
}

fn angle_ref_point_to_right(ref_frame: &Frame, anomalous: bool) -> anyhow::Result<f64> {
    // make sure the ref_frame has a ref_point, otherwise return error
    let ref_point = ref_frame
        .reference_point
        .ok_or(anyhow!("No reference point found in frame"))?;
    // Define line between to points to align either horizontally or 
    // vertically (based on anomalous)
    let (p1, p2) = if anomalous {
        let ((p1, p2), _) = ref_frame.lumen.find_farthest_points();
        let p1_coords = (p1.x, p1.y, p1.z);
        let p2_coords = (p2.x, p2.y, p2.z);
        (p1_coords, p2_coords)
    } else {
        let p1 = ref_frame.centroid;
        let p2 = (ref_point.x, ref_point.y, ref_point.z);
        (p1, p2)
    };

    let dx = p2.0 - p1.0;
    let dy = p2.1 - p1.1;
    let line_angle = dy.atan2(dx);

    let desired = if anomalous { std::f64::consts::FRAC_PI_2 } else { 0.0 };
    let mut rotation = (desired - line_angle).rem_euclid(2.0 * std::f64::consts::PI);

    let rotate2 = |pt: (f64, f64), center: (f64, f64), angle: f64| -> (f64, f64) {
        let dx = pt.0 - center.0;
        let dy = pt.1 - center.1;
        let c = angle.cos();
        let s = angle.sin();
        let xr = dx * c - dy * s;
        let yr = dx * s + dy * c;
        (xr + center.0, yr + center.1)
    };

    let center = (p1.0, p1.1);
    let ref_pt_2d = (ref_point.x, ref_point.y);
    let other_pts = [(p1.0, p1.1), (p2.0, p2.1)];

    let rotated_ref = rotate2(ref_pt_2d, center, rotation);
    let mut all_good = true;
    for &op in &other_pts {
        // skip comparison if op is identical to ref (possible in non-anomalous case)
        if (op.0 - ref_pt_2d.0).abs() < std::f64::EPSILON && (op.1 - ref_pt_2d.1).abs() < std::f64::EPSILON {
            continue;
        }
        let r_op = rotate2(op, center, rotation);
        if !(rotated_ref.0 > r_op.0) {
            all_good = false;
            break;
        }
    }

    if !all_good {
        rotation = (rotation + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI);
    }

    Ok(rotation)
}

fn assign_aortic(mut geometry: Geometry) -> Geometry {
    for frame in &mut geometry.frames {
        let len = frame.lumen.points.len();
        if len == 0 {
            continue;
        }
        let half = len / 2;
        for (i, point) in frame.lumen.points.iter_mut().enumerate() {
            point.aortic = i >= half;
        }
    }
    geometry
}

fn dump_table(logs: &[AlignLog]) {
    let headers = [
        "Contour",
        "Matched To",
        "Rotation (°)",
        "Tx",
        "Ty",
        "Centroid",
    ];
    let rows: Vec<[String; 6]> = logs
        .iter()
        .map(|e| {
            [
                e.contour_id.to_string(),
                e.matched_to.to_string(),
                format!("{:.2}", e.rot_deg),
                format!("{:.2}", e.tx),
                format!("{:.2}", e.ty),
                format!("({:.2},{:.2})", e.centroid.0, e.centroid.1),
            ]
        })
        .collect();

    let mut widths = [0usize; 6];
    for (i, &h) in headers.iter().enumerate() {
        widths[i] = h.len();
    }
    for row in &rows {
        for (i, cell) in row.iter().enumerate() {
            widths[i] = widths[i].max(cell.len());
        }
    }

    // Left-align data
    fn print_row(cells: &[String], widths: &[usize]) {
        print!("|");
        for (i, cell) in cells.iter().enumerate() {
            let pad = widths[i] - cell.len();
            print!(" {}{} |", cell, " ".repeat(pad));
        }
        println!();
    }

    // Center a header row
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

    // Top border
    print!("+");
    for w in &widths {
        print!("{}+", "-".repeat(w + 2));
    }
    println!();

    // Header row
    let header_cells: Vec<String> = headers.iter().map(|&s| s.to_string()).collect();
    print_header(&header_cells, &widths);

    // Separator
    print!("+");
    for w in &widths {
        print!("{}+", "-".repeat(w + 2));
    }
    println!();

    // Data rows
    for row in &rows {
        print_row(&row.to_vec(), &widths);
    }

    // Bottom border
    print!("+");
    for w in &widths {
        print!("{}+", "-".repeat(w + 2));
    }
    println!();
}