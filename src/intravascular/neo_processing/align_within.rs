use anyhow::anyhow;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

use super::wall::create_wall_frames;
use crate::intravascular::io::geometry::{Contour, ContourType, Frame, Geometry};
use crate::intravascular::io::input::ContourPoint;
use crate::intravascular::neo_processing::process_utils::{
    downsample_contour_points,
    search_range,
    hausdorff_distance,};

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

    let ref_idx = geometry
        .find_ref_frame_idx()
        .unwrap_or(geometry.find_proximal_end_idx()) as usize;
    let sample_ratio = sample_size as f64 / geometry.frames[0].lumen.points.len() as f64;
    let sample_size_catheter = if geometry.frames[0]
        .extras
        .contains_key(&ContourType::Catheter)
    {
        Some(
            (geometry.frames[0].extras[&ContourType::Catheter]
                .points
                .len() as f64
                * sample_ratio)
                .ceil() as usize,
        )
    } else {
        None
    };

    let logger = Arc::new(Mutex::new(Vec::<AlignLog>::new()));

    for i in 1..geometry.frames.len() {
        let (prev_frames, curr_frames) = geometry.frames.split_at_mut(i);
        let current = &mut curr_frames[0];
        let previous = &prev_frames[i - 1];

        // TODO: Later maybe add option to move first contour to (0.0, 0.0, 0.0)
        let translation = (
            previous.centroid.0 - current.centroid.0,
            previous.centroid.1 - current.centroid.1,
            0.0,
        );

        current.translate_frame(translation);

        let _testing_points =
            catheter_lumen_vec_from_frames(current, sample_size, sample_size_catheter);
        let _reference_points =
            catheter_lumen_vec_from_frames(&previous, sample_size, sample_size_catheter);

        let best_rotation = if bruteforce {
            search_range(
                |angle: f64| {
                    let rotated: Vec<ContourPoint> = _testing_points
                        .par_iter()
                        .map(|p| p.rotate_point(angle, (current.centroid.0, current.centroid.1)))
                        .collect();
                    hausdorff_distance(&_reference_points, &rotated)
                },
                step_deg,
                range_deg,
                None,
                range_deg,
            )
        } else {
            find_best_rotation(
                &_reference_points,
                &_testing_points,
                step_deg,
                range_deg,
                &current.centroid,
            )
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
    }

    let geometry_filled = fill_holes(geometry)?;
    let mut geometry = fix_spacing(&geometry_filled);

    let anomalous_bool = is_anomalous_coronary(&geometry.frames[ref_idx]);
    let additional_rotation = angle_ref_point_to_right(&geometry.frames[ref_idx], anomalous_bool)?;
    for frame in geometry.frames.iter_mut() {
        frame.rotate_frame(additional_rotation);
        frame.sort_frame_points();
    }

    let mut final_geometry = if anomalous_bool {
        assign_aortic(geometry.clone())
    } else {
        geometry.clone()
    };

    let wall_frames = create_wall_frames(&final_geometry.frames, false, anomalous_bool);
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
    sample_size_catheter: Option<usize>,
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
    let cost_fn = |angle: f64| -> f64 {
        let rotated: Vec<ContourPoint> = target
            .par_iter()
            .map(|p| p.rotate_point(angle, (centroid.0, centroid.1)))
            .collect();
        hausdorff_distance(reference, &rotated)
    };

    match step_deg {
        1.0..=f64::INFINITY => search_range(cost_fn, step_deg, range_deg, None, range_deg),
        0.1..1.0 => {
            let coarse_angle = search_range(&cost_fn, 1.0, range_deg, None, range_deg);
            let range = if range_deg > 5.0 { 5.0 } else { range_deg };
            search_range(cost_fn, step_deg, range, Some(coarse_angle), range_deg)
        }
        0.01..0.1 => {
            let coarse_angle = search_range(&cost_fn, 1.0, range_deg, None, range_deg);
            let range = if range_deg > 5.0 { 5.0 } else { range_deg };
            let medium_angle = search_range(&cost_fn, 0.1, range, Some(coarse_angle), range_deg);
            let range_small = if range_deg > 10.0 * step_deg {
                10.0 * step_deg
            } else {
                range_deg
            };
            search_range(cost_fn, step_deg, range_small, Some(medium_angle), range_deg)
        }
        _ => {
            let coarse_angle = search_range(&cost_fn, 1.0, range_deg, None, range_deg);
            let range = if range_deg > 5.0 { 5.0 } else { range_deg };
            let medium_angle = search_range(&cost_fn, 0.1, range, Some(coarse_angle), range_deg);
            let range_small = if range_deg > 0.1 { 0.1 } else { range_deg };
            let fine_angle = search_range(&cost_fn, 0.01, range_small, Some(medium_angle), range_deg);
            let range_fine = if range_deg > 10.0 * step_deg {
                10.0 * step_deg
            } else {
                range_deg
            };
            search_range(cost_fn, step_deg, range_fine, Some(fine_angle), range_deg)
        }
    }
}

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

    let desired = if anomalous {
        std::f64::consts::FRAC_PI_2
    } else {
        0.0
    };
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
        if (op.0 - ref_pt_2d.0).abs() < std::f64::EPSILON
            && (op.1 - ref_pt_2d.1).abs() < std::f64::EPSILON
        {
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

/// Detect z-gaps larger than expected and return (hole_found, avg_diff)
fn detect_holes(geometry: &Geometry) -> (bool, f64) {
    let mut z_diffs = Vec::new();
    for i in 1..geometry.frames.len() {
        let z_prev = geometry.frames[i - 1].centroid.2;
        let z_curr = geometry.frames[i].centroid.2;
        let diff = (z_curr - z_prev).abs();
        z_diffs.push(diff);
    }
    if z_diffs.is_empty() {
        return (false, 0.0);
    }
    let avg_diff: f64 = z_diffs.iter().sum::<f64>() / z_diffs.len() as f64;
    for &diff in &z_diffs {
        if diff > 1.5 * avg_diff {
            return (true, avg_diff);
        }
    }
    (false, avg_diff)
}

/// Fill holes by inserting averaged / interpolated frames using Geometry::insert_frame.
/// Uses z thresholds:
///   < 1.5 => OK, do nothing
///  [1.5,2.5) => one missing frame -> insert averaged frame
///  [2.5,3.5) => two missing frames -> insert two interpolated frames
///  >=3.5 => error (too big to auto-fix)
pub fn fill_holes(geometry: &mut Geometry) -> anyhow::Result<Geometry> {
    let (hole, _avg_diff) = detect_holes(geometry);

    if !hole {
        return Ok(geometry.clone());
    }

    println!("⚠️ Hole detected! Attempting to fix using Geometry::insert_frame(...)");

    // Walk through frames; insert when we see a gap
    let mut i: usize = 1;
    while i < geometry.frames.len() {
        let prev = &geometry.frames[i - 1].clone();
        let curr = &geometry.frames[i].clone();

        let diff = (curr.centroid.2 - prev.centroid.2).abs();

        if diff < 1.5 {
            // normal spacing
            i += 1;
            continue;
        } else if diff >= 1.5 && diff < 2.5 {
            // one missing frame: insert averaged frame at position i
            let mid = fix_one_frame_hole(prev, curr);
            geometry.insert_frame(mid, Some(curr.id as usize));
            // After insertion, the previously-curr frame moved to i+1, so skip past curr
            i += 2;
        } else if diff >= 2.5 && diff < 3.5 {
            // two missing frames: insert two interpolated frames at position i
            let (f1, f2) = fix_two_frame_hole(prev, curr);
            geometry.insert_frame(f1, Some(curr.id as usize));
            // second frame should go after the first inserted frame -> index i+1
            geometry.insert_frame(f2, Some(curr.id as usize + 1));
            // skip past the two inserted frames and original curr
            i += 3;
        } else {
            return Err(anyhow!(
                "🛑 Detected a very large z-gap between frames at indices {} and {} (dz = {:.2} (avg diff: {:.2})) — refusing to auto-fix",
                i - 1,
                i,
                diff,
                _avg_diff,
            ));
        }
    }

    Ok(geometry.clone())
}

fn avg_opt(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    match (a, b) {
        (Some(x), Some(y)) => Some((x + y) / 2.0),
        (Some(x), None) => Some(x),
        (None, Some(y)) => Some(y),
        (None, None) => None,
    }
}

fn avg_point(p1: &ContourPoint, p2: &ContourPoint, frame_index: u32, point_index: u32) -> ContourPoint {
    ContourPoint {
        frame_index,
        point_index,
        x: (p1.x + p2.x) / 2.0,
        y: (p1.y + p2.y) / 2.0,
        z: (p1.z + p2.z) / 2.0,
        aortic: p1.aortic || p2.aortic,
    }
}

fn avg_contour(c1: &Contour, c2: &Contour, id: u32, original_frame: u32) -> Contour {
    let len = c1.points.len().min(c2.points.len());
    let points: Vec<ContourPoint> = (0..len)
        .map(|i| avg_point(&c1.points[i], &c2.points[i], original_frame, i as u32))
        .collect();

    Contour {
        id,
        original_frame,
        points,
        centroid: match (c1.centroid, c2.centroid) {
            (Some(a), Some(b)) => Some(((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0, (a.2 + b.2) / 2.0)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        },
        aortic_thickness: avg_opt(c1.aortic_thickness, c2.aortic_thickness),
        pulmonary_thickness: avg_opt(c1.pulmonary_thickness, c2.pulmonary_thickness),
        kind: c1.kind,
    }
}

fn fix_one_frame_hole(frame_1: &Frame, frame_2: &Frame) -> Frame {
    let centroid = (
        (frame_1.centroid.0 + frame_2.centroid.0) / 2.0,
        (frame_1.centroid.1 + frame_2.centroid.1) / 2.0,
        (frame_1.centroid.2 + frame_2.centroid.2) / 2.0,
    );

    let lumen = avg_contour(&frame_1.lumen, &frame_2.lumen, frame_2.lumen.id, frame_2.lumen.original_frame);

    // extras: union keys; interpolate when both present
    let mut extras = std::collections::HashMap::new();
    for key in frame_1.extras.keys().chain(frame_2.extras.keys()) {
        if extras.contains_key(key) {
            continue;
        }
        match (frame_1.extras.get(key), frame_2.extras.get(key)) {
            (Some(c1), Some(c2)) => {
                extras.insert(*key, avg_contour(c1, c2, c2.id, c2.original_frame));
            }
            (Some(c1), None) => {
                extras.insert(*key, c1.clone());
            }
            (None, Some(c2)) => {
                extras.insert(*key, c2.clone());
            }
            (None, None) => {}
        }
    }

    // average reference_point if both exist
    let reference_point = match (&frame_1.reference_point, &frame_2.reference_point) {
        (Some(p1), Some(p2)) => Some(avg_point(p1, p2, frame_2.id, 0)),
        (Some(p1), None) => Some(p1.clone()),
        (None, Some(p2)) => Some(p2.clone()),
        (None, None) => None,
    };

    Frame {
        id: frame_2.id, // placeholder; Geometry::insert_frame will reassign IDs
        centroid,
        lumen,
        extras,
        reference_point,
    }
}

/// Interpolation helpers for two-frame hole: produce two frames at t=1/3 and t=2/3
fn interpolate_opt(a: Option<f64>, b: Option<f64>, t: f64) -> Option<f64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x + (y - x) * t),
        (Some(x), None) => Some(x),
        (None, Some(y)) => Some(y),
        (None, None) => None,
    }
}

fn interp_point(p1: &ContourPoint, p2: &ContourPoint, t: f64, frame_index: u32, point_index: u32) -> ContourPoint {
    ContourPoint {
        frame_index,
        point_index,
        x: p1.x + (p2.x - p1.x) * t,
        y: p1.y + (p2.y - p1.y) * t,
        z: p1.z + (p2.z - p1.z) * t,
        aortic: p1.aortic || p2.aortic,
    }
}

fn interpolate_contour(c1: &Contour, c2: &Contour, t: f64, id: u32, original_frame: u32) -> Contour {
    let len = c1.points.len().min(c2.points.len());
    let points: Vec<ContourPoint> = (0..len)
        .map(|i| interp_point(&c1.points[i], &c2.points[i], t, original_frame, i as u32))
        .collect();

    Contour {
        id,
        original_frame,
        points,
        centroid: match (c1.centroid, c2.centroid) {
            (Some(a), Some(b)) => Some((
                a.0 + (b.0 - a.0) * t,
                a.1 + (b.1 - a.1) * t,
                a.2 + (b.2 - a.2) * t,
            )),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        },
        aortic_thickness: interpolate_opt(c1.aortic_thickness, c2.aortic_thickness, t),
        pulmonary_thickness: interpolate_opt(c1.pulmonary_thickness, c2.pulmonary_thickness, t),
        kind: c1.kind,
    }
}

fn create_interpolated_frame(frame_1: &Frame, frame_2: &Frame, t: f64) -> Frame {
    let centroid = (
        frame_1.centroid.0 + (frame_2.centroid.0 - frame_1.centroid.0) * t,
        frame_1.centroid.1 + (frame_2.centroid.1 - frame_1.centroid.1) * t,
        frame_1.centroid.2 + (frame_2.centroid.2 - frame_1.centroid.2) * t,
    );

    let lumen = interpolate_contour(&frame_1.lumen, &frame_2.lumen, t, frame_2.lumen.id, frame_2.lumen.original_frame);

    let mut extras = std::collections::HashMap::new();
    for key in frame_1.extras.keys().chain(frame_2.extras.keys()) {
        if extras.contains_key(key) {
            continue;
        }
        match (frame_1.extras.get(key), frame_2.extras.get(key)) {
            (Some(c1), Some(c2)) => {
                extras.insert(*key, interpolate_contour(c1, c2, t, c2.id, c2.original_frame));
            }
            (Some(c1), None) => {
                extras.insert(*key, c1.clone());
            }
            (None, Some(c2)) => {
                extras.insert(*key, c2.clone());
            }
            (None, None) => {}
        }
    }

    let reference_point = match (&frame_1.reference_point, &frame_2.reference_point) {
        (Some(p1), Some(p2)) => Some(interp_point(p1, p2, t, frame_2.id, 0)),
        (Some(p1), None) => Some(p1.clone()),
        (None, Some(p2)) => Some(p2.clone()),
        (None, None) => None,
    };

    Frame {
        id: frame_2.id, // placeholder; Geometry::insert_frame will reassign IDs
        centroid,
        lumen,
        extras,
        reference_point,
    }
}

/// Create two frames (1/3 and 2/3) between frame_1 and frame_2
fn fix_two_frame_hole(frame_1: &Frame, frame_2: &Frame) -> (Frame, Frame) {
    let f1 = create_interpolated_frame(frame_1, frame_2, 1.0 / 3.0);
    let f2 = create_interpolated_frame(frame_1, frame_2, 2.0 / 3.0);
    (f1, f2)
}

fn fix_spacing(geometry: &Geometry) -> Geometry {
    // TODO: implement spacing correction; currently return cloned input as placeholder
    geometry.clone()
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
