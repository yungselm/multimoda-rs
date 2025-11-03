use rayon::prelude::*;
use std::f64::consts::PI;
use crate::intravascular::io::input::ContourPoint;

pub fn downsample_contour_points(points: &[ContourPoint], n: usize) -> Vec<ContourPoint> {
    if points.len() <= n {
        return points.to_vec();
    }
    let step = points.len() as f64 / n as f64;
    (0..n)
        .map(|i| {
            let index = (i as f64 * step) as usize;
            points[index].clone()
        })
        .collect()
}

pub fn search_range<F>(
    cost_fn: F,
    step_deg: f64,
    range_deg: f64,
    center_angle: Option<f64>,
    limes_deg: f64,
) -> f64 
where
    F: Fn(f64) -> f64 + Sync + Send,
{
    let range_rad = range_deg.to_radians();
    let step_rad = step_deg.to_radians();
    if step_rad <= 0.0 {
        return 0.0;
    }

    let center = center_angle.unwrap_or(0.0);
    let limes = limes_deg.to_radians();
    let lower_limes = -limes;

    let mut start_angle = center - range_rad;
    let stop_angle = center + range_rad;
    start_angle = start_angle.max(lower_limes);
    let stop_angle = stop_angle.min(limes);

    if stop_angle <= start_angle {
        return ((center + PI).rem_euclid(2.0 * PI)) - PI;
    }

    let steps = (((stop_angle - start_angle) / step_rad).ceil() as usize).max(1);

    let mut angle_cost_pairs = Vec::with_capacity(steps);
    for i in 0..=steps {
        let angle_lin = start_angle + (i as f64) * step_rad;
        if angle_lin > stop_angle {
            break;
        }

        let angle = angle_lin.rem_euclid(2.0 * PI);
        let mapped_angle = ((angle + PI).rem_euclid(2.0 * PI)) - PI;

        let cost = cost_fn(angle);
        angle_cost_pairs.push((mapped_angle, cost));
    }

    if angle_cost_pairs.is_empty() {
        return ((center + PI).rem_euclid(2.0 * PI)) - PI;
    }

    let (min_angle, _min_cost) = angle_cost_pairs
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