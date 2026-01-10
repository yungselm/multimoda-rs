use anyhow::anyhow;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

use super::wall::create_wall_frames;
use crate::intravascular::io::geometry::{Contour, ContourType, Frame, Geometry};
use crate::intravascular::io::input::ContourPoint;
use crate::intravascular::processing::process_utils::{
    downsample_contour_points, hausdorff_distance, search_range,
};

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
) -> anyhow::Result<(Geometry, Vec<AlignLog>, bool)> {
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

    let mut cumulative_rotation: f64 = 0.0;

    for i in 1..geometry.frames.len() {
        let prev_frame = geometry.frames[i - 1].clone();
        let current = &mut geometry.frames[i];

        println!(
            "Aligning Frame {} to previous Frame {}",
            current.id, prev_frame.id
        );

        if cumulative_rotation != 0.0 {
            current.rotate_frame(cumulative_rotation);
        }

        let translation = (
            prev_frame.centroid.0 - current.centroid.0,
            prev_frame.centroid.1 - current.centroid.1,
            0.0,
        );
        current.translate_frame(translation);

        let testing_points =
            catheter_lumen_vec_from_frames(current, sample_size, sample_size_catheter);
        let reference_points =
            catheter_lumen_vec_from_frames(&prev_frame, sample_size, sample_size_catheter);

        let best_rotation = if bruteforce {
            search_range(
                |angle: f64| {
                    let rotated: Vec<ContourPoint> = testing_points
                        .par_iter()
                        .map(|p| p.rotate_point(angle, (current.centroid.0, current.centroid.1)))
                        .collect();
                    hausdorff_distance(&reference_points, &rotated)
                },
                step_deg,
                range_deg,
                None,
                range_deg,
            )
        } else {
            find_best_rotation(
                &reference_points,
                &testing_points,
                step_deg,
                range_deg,
                &current.centroid,
            )
        };

        current.rotate_frame(best_rotation);
        cumulative_rotation += best_rotation;

        let new_log = AlignLog {
            contour_id: current.id,
            matched_to: prev_frame.id,
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

    geometry.rotate_geometry(additional_rotation);

    let mut final_geometry = if anomalous_bool {
        assign_aortic(geometry.clone())
    } else {
        geometry.clone()
    };

    let wall_frames = create_wall_frames(&final_geometry.frames, anomalous_bool, false);
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

    Ok((final_geometry, logs, anomalous_bool))
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
            search_range(
                cost_fn,
                step_deg,
                range_small,
                Some(medium_angle),
                range_deg,
            )
        }
        _ => {
            let coarse_angle = search_range(&cost_fn, 1.0, range_deg, None, range_deg);
            let range = if range_deg > 5.0 { 5.0 } else { range_deg };
            let medium_angle = search_range(&cost_fn, 0.1, range, Some(coarse_angle), range_deg);
            let range_small = if range_deg > 0.1 { 0.1 } else { range_deg };
            let fine_angle =
                search_range(&cost_fn, 0.01, range_small, Some(medium_angle), range_deg);
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

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

/// Detect whether there's any gap that's substantially larger than typical spacing.
/// Returns (has_hole, baseline_spacing)
fn detect_holes(geometry: &Geometry) -> (bool, f64) {
    let mut z_diffs: Vec<f64> = Vec::new();
    for i in 1..geometry.frames.len() {
        let z_prev = geometry.frames[i - 1].centroid.2;
        let z_curr = geometry.frames[i].centroid.2;
        z_diffs.push((z_curr - z_prev).abs());
    }
    if z_diffs.is_empty() {
        return (false, 0.0);
    }

    let mut sorted = z_diffs.clone();
    let baseline = median(&mut sorted);

    // avoid divide-by-zero later; if baseline is zero treat as "no hole"
    if baseline <= std::f64::EPSILON {
        return (false, baseline);
    }

    let has_hole = z_diffs.iter().any(|&d| d >= 1.5 * baseline);

    (has_hole, baseline)
}

/// Fill holes by inserting averaged / interpolated frames using Geometry::insert_frame.
/// Uses ratio thresholds against the baseline spacing (median):
///   ratio < 1.5 => OK, do nothing
///  [1.5,2.5) => one missing frame -> insert averaged frame
///  [2.5,3.5) => two missing frames -> insert two interpolated frames
///  >= 3.5 => error (too big to auto-fix)
pub fn fill_holes(geometry: &mut Geometry) -> anyhow::Result<Geometry> {
    let (hole, baseline) = detect_holes(geometry);

    if !hole {
        return Ok(geometry.clone());
    }

    if baseline <= std::f64::EPSILON {
        return Err(anyhow!("Baseline spacing is zero or too small to decide."));
    }

    println!("⚠️\tHole detected! Attempting to fix using Geometry::insert_frame(...) (baseline spacing = {:.3})", baseline);

    let mut i: usize = 1;
    while i < geometry.frames.len() {
        let prev = geometry.frames[i - 1].clone();
        let curr = geometry.frames[i].clone();

        let diff = (curr.centroid.2 - prev.centroid.2).abs();
        let ratio = diff / baseline;

        if ratio < 1.5 {
            // normal spacing
            i += 1;
            continue;
        } else if ratio >= 1.5 && ratio < 2.5 {
            // one missing frame: insert averaged frame at position i
            let mid = fix_one_frame_hole(&prev, &curr);
            geometry.insert_frame(mid, Some(i));
            i += 2;
            println!(
                "✅ Fixed one-frame hole between Frame {} and Frame {} (dz = {:.3}, ratio = {:.3})",
                prev.id, curr.id, diff, ratio
            );
        } else if ratio >= 2.5 && ratio < 3.5 {
            // two missing frames: insert two interpolated frames at position i
            let (f1, f2) = fix_two_frame_hole(&prev, &curr);
            geometry.insert_frame(f1, Some(i));
            geometry.insert_frame(f2, Some(i + 1));
            i += 3;
            println!(
                "✅ Fixed two-frame hole between Frame {} and Frame {} (dz = {:.3}, ratio = {:.3})",
                prev.id, curr.id, diff, ratio
            );
        } else {
            // Larger gaps - calculate how many frames to insert
            let missing_frames_count = (ratio - 1.0).floor().max(1.0) as usize;

            if ratio >= 10.0 {
                println!("🛑 WARNING: Very large gap detected between Frame {} and Frame {} (dz = {:.3}, baseline: {:.3}, ratio: {:.3}) - inserting {} frames but geometry may not be realistic!", 
                    prev.id, curr.id, diff, baseline, ratio, missing_frames_count);
            } else if ratio >= 5.0 {
                println!("⚠️\tLarge gap detected between Frame {} and Frame {} (dz = {:.3}, baseline: {:.3}, ratio: {:.3}) - inserting {} frames", 
                    prev.id, curr.id, diff, baseline, ratio, missing_frames_count);
            } else {
                println!("🔄 Fixing {}-frame gap between Frame {} and Frame {} (dz = {:.3}, ratio = {:.3})", 
                    missing_frames_count, prev.id, curr.id, diff, ratio);
            }

            // Insert the missing frames using interpolation
            for frame_idx in 1..=missing_frames_count {
                let t = frame_idx as f64 / (missing_frames_count + 1) as f64;
                let new_frame = create_interpolated_frame(&prev, &curr, t);
                geometry.insert_frame(new_frame, Some(i + frame_idx - 1));
            }

            i += missing_frames_count + 1;
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

fn avg_point(
    p1: &ContourPoint,
    p2: &ContourPoint,
    frame_index: u32,
    point_index: u32,
) -> ContourPoint {
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

    let lumen = avg_contour(
        &frame_1.lumen,
        &frame_2.lumen,
        frame_2.lumen.id,
        frame_2.lumen.original_frame,
    );

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

    // for other algorithms only one reference opint can exist
    let reference_point = None;

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

fn interp_point(
    p1: &ContourPoint,
    p2: &ContourPoint,
    t: f64,
    frame_index: u32,
    point_index: u32,
) -> ContourPoint {
    ContourPoint {
        frame_index,
        point_index,
        x: p1.x + (p2.x - p1.x) * t,
        y: p1.y + (p2.y - p1.y) * t,
        z: p1.z + (p2.z - p1.z) * t,
        aortic: p1.aortic || p2.aortic,
    }
}

fn interpolate_contour(
    c1: &Contour,
    c2: &Contour,
    t: f64,
    id: u32,
    original_frame: u32,
) -> Contour {
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

    let lumen = interpolate_contour(
        &frame_1.lumen,
        &frame_2.lumen,
        t,
        frame_2.lumen.id,
        frame_2.lumen.original_frame,
    );

    let mut extras = std::collections::HashMap::new();
    for key in frame_1.extras.keys().chain(frame_2.extras.keys()) {
        if extras.contains_key(key) {
            continue;
        }
        match (frame_1.extras.get(key), frame_2.extras.get(key)) {
            (Some(c1), Some(c2)) => {
                extras.insert(
                    *key,
                    interpolate_contour(c1, c2, t, c2.id, c2.original_frame),
                );
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

#[cfg(test)]
mod align_within_tests {
    use anyhow::Ok;
    use approx::assert_relative_eq;

    use super::*;
    use crate::intravascular::utils::test_utils::{
        dummy_geometry, dummy_geometry_aligned_long, dummy_geometry_center_reference,
    };

    #[test]
    fn test_simple_geometry() -> anyhow::Result<()> {
        let mut dummy = dummy_geometry();
        let ref_frame_idx = dummy.find_ref_frame_idx()?;

        assert_eq!(ref_frame_idx, 0);

        let (geom, logs, _) = align_frames_in_geometry(&mut dummy, 0.01, 30.0, false, false, 6)?;

        assert!(!geom.frames.is_empty());
        assert_relative_eq!(
            geom.frames[0].lumen.points[0].x,
            geom.frames[1].lumen.points[0].x,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            geom.frames[0].lumen.points[0].y,
            geom.frames[1].lumen.points[0].y,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            geom.frames[0].lumen.points[0].x,
            geom.frames[2].lumen.points[0].x,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            geom.frames[0].lumen.points[0].y,
            geom.frames[2].lumen.points[0].y,
            epsilon = 1e-6
        );
        for (i, log) in logs.iter().enumerate() {
            let idx = i as f64 + 1.0;
            let expected_tx = -1.0 * idx;
            let expected_ty = -1.0 * idx;
            assert_relative_eq!(log.rot_deg, -15.0, epsilon = 1e-6);
            assert_relative_eq!(log.tx, expected_tx, epsilon = 1e-6);
            assert_relative_eq!(log.ty, expected_ty, epsilon = 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_simple_geometry_middle_ref() -> anyhow::Result<()> {
        let mut dummy = dummy_geometry_center_reference();
        let ref_frame_idx = dummy.find_ref_frame_idx();
        println!("Reference idx: {:?}", ref_frame_idx);

        for frame in dummy.frames.iter() {
            println!("Frame {:?}:\nz.position:{:?}, point0 x: {:?}, point0 y: {:?}, point0 z: {:?}, ref_point present?: {:?}",
                frame.id,
                frame.centroid.2,
                frame.lumen.points[0].x,
                frame.lumen.points[0].y,
                frame.lumen.points[0].z,
                frame.reference_point.is_some(),
            );
        }

        let result = align_frames_in_geometry(&mut dummy, 0.01, 30.0, false, false, 6);

        assert!(result.is_ok());
        Ok(())
    }

    #[test]
    fn test_idealized_geometry() -> anyhow::Result<()> {
        use crate::intravascular::io::build_geometry_from_inputdata;
        use std::path::Path;

        let mut geometry = build_geometry_from_inputdata(
            None,
            Some(Path::new("data/fixtures/idealized_geometry")),
            "stress",
            true,
            (4.5, 4.5),
            0.5,
            20,
        )?;

        let (geom, logs, anomalous) =
            align_frames_in_geometry(&mut geometry, 0.01, 20.0, true, false, 200)?;

        assert!(!geom.frames.is_empty());
        assert_eq!(anomalous, true);

        for log in &logs {
            assert_relative_eq!(log.rot_deg.abs(), 15.0, epsilon = 1.0)
        }
        for (i, log) in logs.iter().enumerate() {
            let idx = i as f64 + 1.0;
            let expected_tx = -0.01 * idx;
            let expected_ty = 0.01 * idx;
            assert_relative_eq!(log.tx, expected_tx, epsilon = 0.001);
            assert_relative_eq!(log.ty, expected_ty, epsilon = 0.001);
        }
        Ok(())
    }

    #[test]
    fn test_detect_holes_and_fill_one_frame() -> anyhow::Result<()> {
        let mut geometry = dummy_geometry_aligned_long();
        geometry.frames[5].translate_frame((0.0, 0.0, 1.0));

        let (bool_hole, avg_dist) = detect_holes(&geometry);

        assert_eq!(bool_hole, true);
        assert_relative_eq!(avg_dist, 1.0, epsilon = 1e-6);

        let new_frame = fix_one_frame_hole(&geometry.frames[1], &geometry.frames[2]);

        assert_relative_eq!(new_frame.centroid.2, 1.5, epsilon = 1e-6);
        for point in new_frame.lumen.points {
            assert_relative_eq!(point.z, 1.5, epsilon = 1e-6);
        }

        let new_geom = fill_holes(&mut geometry)?;

        assert_eq!(new_geom.frames.len(), 7);
        for (i, frame) in new_geom.frames.iter().enumerate() {
            assert_eq!(frame.id, i as u32);
            assert_eq!(frame.lumen.id, i as u32);
            assert_eq!(frame.centroid.2, i as f64);
            assert_eq!(frame.lumen.centroid.unwrap().2, i as f64);
            for point in frame.lumen.points.iter() {
                assert_eq!(point.z, i as f64);
            }
        }
        Ok(())
    }

    #[test]
    fn test_detect_holes_and_fill_two_frame() -> anyhow::Result<()> {
        let mut geometry = dummy_geometry_aligned_long();
        geometry.frames[5].translate_frame((0.0, 0.0, 2.0));

        let new_geom = fill_holes(&mut geometry)?;

        assert_eq!(new_geom.frames.len(), 8);
        for (i, frame) in new_geom.frames.iter().enumerate() {
            println!("Frame id {:?} and z-coord {:?}", frame.id, frame.centroid.2);
            assert_eq!(frame.id, i as u32);
            assert_eq!(frame.lumen.id, i as u32);
            assert_eq!(frame.centroid.2, i as f64);
            assert_eq!(frame.lumen.centroid.unwrap().2, i as f64);
            for point in frame.lumen.points.iter() {
                assert_eq!(point.z, i as f64);
            }
        }
        Ok(())
    }

    #[test]
    fn test_smoothing_effect() -> anyhow::Result<()> {
        let mut geometry = dummy_geometry();

        let (geom_unsmoothed, _, _) =
            align_frames_in_geometry(&mut geometry.clone(), 0.1, 30.0, false, false, 10)?;

        let (geom_smoothed, _, _) =
            align_frames_in_geometry(&mut geometry, 0.1, 30.0, true, false, 10)?;

        // Smoothed geometry should have same number of frames but potentially different point coordinates
        assert_eq!(geom_unsmoothed.frames.len(), geom_smoothed.frames.len());
        Ok(())
    }

    #[test]
    fn test_with_and_without_catheter() -> anyhow::Result<()> {
        let mut geometry_with_cath = dummy_geometry();
        // Add catheter points to frames
        for frame in &mut geometry_with_cath.frames {
            let catheter_contour = Contour {
                id: frame.id + 100,
                original_frame: frame.id,
                points: vec![
                    ContourPoint {
                        frame_index: frame.id,
                        point_index: 0,
                        x: 0.0,
                        y: 0.0,
                        z: frame.centroid.2,
                        aortic: false,
                    },
                    ContourPoint {
                        frame_index: frame.id,
                        point_index: 1,
                        x: 1.0,
                        y: 0.0,
                        z: frame.centroid.2,
                        aortic: false,
                    },
                ],
                centroid: Some((0.5, 0.0, frame.centroid.2)),
                aortic_thickness: None,
                pulmonary_thickness: None,
                kind: ContourType::Catheter,
            };
            frame.extras.insert(ContourType::Catheter, catheter_contour);
        }

        let (geom_with_cath, _, _) =
            align_frames_in_geometry(&mut geometry_with_cath, 0.1, 30.0, false, false, 10)?;

        let mut geometry_without_cath = dummy_geometry();
        let (geom_without_cath, _, _) =
            align_frames_in_geometry(&mut geometry_without_cath, 0.1, 30.0, false, false, 10)?;

        // Both should complete successfully
        assert_eq!(geom_with_cath.frames.len(), geom_without_cath.frames.len());
        Ok(())
    }
}
