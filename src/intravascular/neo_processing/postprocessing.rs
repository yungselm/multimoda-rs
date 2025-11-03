use super::align_between::GeometryPair;
use crate::intravascular::io::input::ContourPoint;
use crate::intravascular::io::geometry::{Contour, ContourType, Frame, Geometry};

// TODO: Check sample_rate of the two geoms, if different interpolate between frames
pub fn postprocess_geom_pair(geom_pair: &GeometryPair, tol: f64) -> anyhow::Result<GeometryPair> {
    let (same_sample_rate, avg_diff_a, avg_diff_b) = check_same_sample_rate_geompair(geom_pair, tol);

    let geom_pair_resampled = if same_sample_rate {
        let mean_diff = (avg_diff_a + avg_diff_b) / 2.0;
        let geom_a = resample_by_diff(&geom_pair.geom_a, mean_diff);
        let geom_b = resample_by_diff(&geom_pair.geom_b, mean_diff);
        GeometryPair {
            geom_a: geom_a,
            geom_b: geom_b,
            label: geom_pair.label.clone(),
        }
    } else {
        // TODO:
        // if avg_diff_a < avg_diff_b {
        //     new_frames_by_sample_rate(geom_a)
        //     resample_by_diff(geom_a)
            geom_pair.clone()
        };
        
    Ok(geom_pair_resampled)
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

fn new_frames_by_sample_rate(geometry: &Geometry, diff: f64) -> Geometry {
    todo!()
}

fn predict_z_positions(start_z: f64, stop_z: f64, z_diff: f64) -> Vec<f64> {
    let mut z_coords = Vec::new();
    if !z_diff.is_finite() || z_diff == 0.0 {
        return z_coords;
    }

    let eps = 1e-9;
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

    z_coords
}

// TODO: trim to same number of contours, by matching on the reference contour

// TODO: Adjust walls in geometrypair.