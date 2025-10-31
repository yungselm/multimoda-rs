use rayon::prelude::*;
use std::collections::HashMap;
use std::collections::HashSet;
use std::f64::consts::PI;
use std::sync::{Arc, Mutex};

use crate::intravascular::io::input::ContourPoint;
use crate::intravascular::io::geometry::{Contour, Frame, Geometry};
use crate::intravascular::neo_processing::process_utils::downsample_contour_points;

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

pub fn align_frames_in_geometry(
    geometry: Geometry,
    step_deg: f64,
    range_deg: f64,
    smooth: bool,
    bruteforce: bool,
    sample_size: usize,    
) -> (Geometry, Vec<AlignLog>) {
    let ref_idx = geometry.find_ref_frame_idx().unwrap_or(geometry.find_proximal_end_idx());

    todo!()
}

fn catheter_lumen_contourvec_from_frames() {
    todo!()
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