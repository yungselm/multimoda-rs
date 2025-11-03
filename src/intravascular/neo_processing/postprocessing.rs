use anyhow::Context;

use super::align_between::GeometryPair;
use crate::intravascular::io::input::ContourPoint;
use crate::intravascular::io::geometry::{Contour, ContourType, Frame, Geometry};
use std::collections::HashMap;

pub fn postprocess_geom_pair(geom_pair: &GeometryPair, tol: f64) -> anyhow::Result<GeometryPair> {
    let (same_sample_rate, avg_diff_a, avg_diff_b) = check_same_sample_rate_geompair(geom_pair, tol);
    let ref_idx_a = geom_pair.geom_a.find_ref_frame_idx()?;
    let ref_idx_b = geom_pair.geom_b.find_ref_frame_idx()?;
    let ref_z_a = geom_pair.geom_a.frames[ref_idx_a].centroid.2;
    let ref_z_b = geom_pair.geom_b.frames[ref_idx_b].centroid.2;

    let geom_pair_resampled = if same_sample_rate {
        let mean_diff = (avg_diff_a + avg_diff_b) / 2.0;
        let geom_a = resample_by_diff(&geom_pair.geom_a, mean_diff);
        let geom_b = resample_by_diff(&geom_pair.geom_b, mean_diff);
        GeometryPair {
            geom_a,
            geom_b,
            label: geom_pair.label.clone(),
        }
    } else if avg_diff_a < avg_diff_b {
        let n = geom_pair.geom_a.frames.len();
        let end_zero = geom_pair.geom_a.frames[0].centroid.2;
        let end_n = geom_pair.geom_a.frames[n - 1].centroid.2;
        let (start, stop) = if end_zero < end_n { (end_zero, end_n) } else { (end_n, end_zero) };

        let z_coords = predict_z_positions(ref_z_a, start, stop, avg_diff_b);
        let new_geom = new_frames_by_sample_rate(&geom_pair.geom_a, z_coords);
        let _new_resampled = resample_by_diff(&new_geom, avg_diff_b);
        GeometryPair {
            geom_a: new_geom,
            geom_b: geom_pair.geom_b.clone(),
            label: geom_pair.label.clone(),
        }
    } else {
        let n = geom_pair.geom_b.frames.len();
        let end_zero = geom_pair.geom_b.frames[0].centroid.2;
        let end_n = geom_pair.geom_b.frames[n - 1].centroid.2;
        let (start, stop) = if end_zero < end_n { (end_zero, end_n) } else { (end_n, end_zero) };

        let z_coords = predict_z_positions(ref_z_b, start, stop, avg_diff_b);
        let new_geom = new_frames_by_sample_rate(&geom_pair.geom_b, z_coords);
        let _new_resampled = resample_by_diff(&new_geom, avg_diff_b);
        GeometryPair {
            geom_a: geom_pair.geom_a.clone(),
            geom_b: new_geom,
            label: geom_pair.label.clone(),
        }
    };
    let trimmed_geom_pair = trim_geom_pair(&geom_pair_resampled);
    Ok(trimmed_geom_pair)
}

fn check_same_sample_rate_geompair(geom_pair: &GeometryPair, tol: f64) -> (bool, f64, f64) {
    let avg_diff_a = get_avg_z_diff(&geom_pair.geom_a);
    let avg_diff_b = get_avg_z_diff(&geom_pair.geom_b);

    if (avg_diff_a - avg_diff_b) < tol {
        (true, avg_diff_a, avg_diff_b)
    } else {
        (false, avg_diff_a, avg_diff_b)
    }
}

fn get_avg_z_diff(geometry: &Geometry) -> f64 {
    let mut diffs_geom = Vec::new();
    for i in 1..geometry.frames.len() {
        let curr = &geometry.frames[i];
        let prev = &geometry.frames[i - 1];
        let diff = curr.centroid.2 - prev.centroid.2;
        diffs_geom.push(diff)
    }
    let avg_diff = diffs_geom.iter().sum::<f64>() / diffs_geom.len() as f64;
    avg_diff  
}

// TODO: equalize spacing between frames to be same in both geoms
fn resample_by_diff(geometry: &Geometry, diff: f64) -> Geometry {
    // make sure that smallest z-coord is at position 0, otherwise rotate frames
    let mut geometry = geometry.clone();

    if !geometry.frames.is_empty() {
        if let Some((min_idx, _)) = geometry
            .frames
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.centroid
                    .2
                    .partial_cmp(&b.centroid.2)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        {
            if min_idx != 0 {
                geometry.frames.rotate_left(min_idx);
            }
        }
    }

    let start_z = geometry.frames[0].centroid.2;

    for i in 1..geometry.frames.len() {
        let z_value = start_z + i as f64 * diff;
        geometry.frames[i].set_value(
            None,
            None,
            None,
            None,
            Some(z_value));
    }
    geometry
}

fn new_frames_by_sample_rate(geometry: &Geometry, z_coords: Vec<f64>) -> Geometry {
    let mut new_frames = Vec::new();

    for z_coord in z_coords {
        // Try to find exact match
        if let Some(frame) = geometry.frames.iter().find(|f| (f.centroid.2 - z_coord).abs() < 1e-9) {
            new_frames.push(frame.clone());
            continue;
        }

        // Find two closest frames for interpolation
        let (lower, upper) = geometry.frames.iter()
            .zip(geometry.frames.iter().skip(1))
            .find(|(f1, f2)| f1.centroid.2 <= z_coord && f2.centroid.2 >= z_coord)
            .context("Cannot find frames to interpolate between")
            .unwrap();

        // Calculate interpolation factor
        let t = (z_coord - lower.centroid.2) / (upper.centroid.2 - lower.centroid.2);

        // Interpolate lumen contour
        let new_lumen = interpolate_contour(&lower.lumen, &upper.lumen, t);

        // Interpolate extra contours
        let mut new_extras = HashMap::new();
        for kind in [ContourType::Eem, ContourType::Calcification, 
                    ContourType::Sidebranch, ContourType::Catheter, ContourType::Wall].iter() {
            if let (Some(l_extra), Some(u_extra)) = (lower.extras.get(kind), upper.extras.get(kind)) {
                new_extras.insert(*kind, interpolate_contour(l_extra, u_extra, t));
            }
        }

        // Create interpolated frame
        new_frames.push(Frame {
            id: lower.id,
            centroid: (
                lower.centroid.0 + t * (upper.centroid.0 - lower.centroid.0),
                lower.centroid.1 + t * (upper.centroid.1 - lower.centroid.1),
                z_coord
            ),
            lumen: new_lumen,
            extras: new_extras,
            reference_point: lower.reference_point.as_ref().map(|p1| 
                upper.reference_point.as_ref().map_or(p1.clone(), |p2| 
                    ContourPoint {
                        x: p1.x + t * (p2.x - p1.x),
                        y: p1.y + t * (p2.y - p1.y),
                        z: p1.z,
                        frame_index: p1.frame_index,
                        point_index: p1.point_index,
                        aortic: p1.aortic,
                    }
                )
            ),
        });
    }

    Geometry {
        frames: new_frames,
        label: geometry.label.clone(),
    }
}

fn interpolate_contour(c1: &Contour, c2: &Contour, t: f64) -> Contour {
    let new_points: Vec<ContourPoint> = c1.points.iter()
        .zip(c2.points.iter())
        .map(|(p1, p2)| ContourPoint {
            x: p1.x + t * (p2.x - p1.x),
            y: p1.y + t * (p2.y - p1.y),
            z: p1.z,
            frame_index: p1.frame_index,
            point_index: p1.point_index,
            aortic: p1.aortic,
        })
        .collect();

    let new_centroid = c1.centroid.zip(c2.centroid).map(|(c1, c2)| (
        c1.0 + t * (c2.0 - c1.0),
        c1.1 + t * (c2.1 - c1.1),
        c1.2 + t * (c2.2 - c1.2),
    ));

    Contour {
        id: c1.id,
        original_frame: c1.original_frame,
        points: new_points,
        centroid: new_centroid,
        aortic_thickness: c1.aortic_thickness
            .zip(c2.aortic_thickness)
            .map(|(t1, t2)| t1 + t * (t2 - t1)),
        pulmonary_thickness: c1.pulmonary_thickness
            .zip(c2.pulmonary_thickness)
            .map(|(t1, t2)| t1 + t * (t2 - t1)),
        kind: c1.kind,
    }
}

fn predict_z_positions(ref_z: f64, start_z: f64, stop_z: f64, z_diff: f64) -> Vec<f64> {
    let mut z_coords = Vec::new();
    if !z_diff.is_finite() || z_diff == 0.0 {
        return z_coords;
    }

    let eps = 1e-9;

    // Handle reference position in the middle
    if (ref_z - start_z).abs() > eps && (ref_z - stop_z).abs() > eps {
        // Go backwards from ref to start
        let mut cur = ref_z;
        while cur >= start_z - eps {
            z_coords.push(cur);
            cur -= z_diff;
            if !cur.is_finite() { break; }
        }
        z_coords.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Go forward from ref to stop
        let mut cur = ref_z + z_diff;  // Start from next position after ref
        while cur <= stop_z + eps {
            z_coords.push(cur);
            cur += z_diff;
            if !cur.is_finite() { break; }
        }
    } else {
        // Original logic for when ref_z is at start or stop
        let mut cur = start_z;
        if stop_z >= start_z && z_diff > 0.0 {
            while cur <= stop_z + eps {
                z_coords.push(cur);
                cur += z_diff;
                if !cur.is_finite() { break; }
            }
        } else if stop_z <= start_z && z_diff < 0.0 {
            while cur >= stop_z - eps {
                z_coords.push(cur);
                cur += z_diff;
                if !cur.is_finite() { break; }
            }
        }
    }

    z_coords
}

// TODO: trim to same number of contours, by matching on the reference contour
fn trim_geom_pair(geom_pair: &GeometryPair) -> GeometryPair {
    todo!()
}

// TODO: Adjust walls in geometrypair.