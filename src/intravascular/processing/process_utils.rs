use crate::intravascular::io::input::ContourPoint;
use rayon::prelude::*;
use std::f64::consts::PI;

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

    // Handle edge case: zero or negative step
    if step_rad <= 0.0 {
        return center_angle.unwrap_or(0.0);
    }

    let center = center_angle.unwrap_or(0.0);
    let limes = limes_deg.to_radians();

    let start_angle = (center - range_rad).max(-limes);
    let stop_angle = (center + range_rad).min(limes);

    if stop_angle <= start_angle {
        return center;
    }

    let steps = (((stop_angle - start_angle) / step_rad).ceil() as usize).max(1);

    let mut best_angle = center;
    let mut best_cost = f64::INFINITY;

    for i in 0..=steps {
        let angle = start_angle + (i as f64) * step_rad;
        if angle > stop_angle {
            break;
        }

        // Map angle to [-π, π] range
        let mapped_angle = ((angle + PI).rem_euclid(2.0 * PI)) - PI;
        let cost = cost_fn(mapped_angle);

        if cost < best_cost {
            best_cost = cost;
            best_angle = mapped_angle;
        }
    }

    best_angle
}

/// Computes the Hausdorff distance between two point sets.
pub fn hausdorff_distance(set1: &[ContourPoint], set2: &[ContourPoint]) -> f64 {
    let forward = directed_hausdorff(set1, set2);
    let backward = directed_hausdorff(set2, set1);
    forward.max(backward)
}

fn directed_hausdorff(contour_a: &[ContourPoint], contour_b: &[ContourPoint]) -> f64 {
    // Keep behavior simple for empty inputs (match prior behavior -> 0.0)
    if contour_a.is_empty() || contour_b.is_empty() {
        return 0.0;
    }

    // Decide chunk size based on number of threads to create many tasks but not too many
    let threads = rayon::current_num_threads().max(1);
    // make several chunks per thread for load balancing
    let chunks_per_thread = 4;
    let chunk_size = ((contour_a.len() + threads * chunks_per_thread - 1)
        / (threads * chunks_per_thread))
        .max(1);

    // For each chunk, compute the local maximum of the minimum squared distances
    let max_sq = contour_a
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut local_max_sq = 0.0_f64;
            for pa in chunk {
                // find min squared distance from pa to any pb (sequential inside chunk)
                let mut min_sq = f64::INFINITY;
                for pb in contour_b.iter() {
                    let dx = pa.x - pb.x;
                    let dy = pa.y - pb.y;
                    let d2 = dx * dx + dy * dy;
                    if d2 < min_sq {
                        min_sq = d2;
                    }
                }
                if min_sq.is_finite() && min_sq > local_max_sq {
                    local_max_sq = min_sq;
                }
            }
            local_max_sq
        })
        .reduce(|| 0.0_f64, f64::max);

    max_sq.sqrt()
}

// TODO: Move the interpolation to process_utils

#[cfg(test)]
mod process_utils_tests {
    use super::*;
    use crate::intravascular::utils::test_utils::dummy_geometry;
    use approx::assert_relative_eq;

    #[test]
    fn test_downsample_geometry() {
        let dummy = dummy_geometry();
        let cont_points = dummy.frames[0].lumen.points.clone();
        let downsampled_contour = downsample_contour_points(&cont_points, 3);

        assert_eq!(downsampled_contour.len(), 3);
        assert_eq!(downsampled_contour[0].point_index, 0);
        assert_eq!(downsampled_contour[1].point_index, 2);

        let downsampled_contour = downsample_contour_points(&cont_points, 6);

        assert_eq!(downsampled_contour.len(), 6);
        assert_eq!(downsampled_contour[0].point_index, 0);
        assert_eq!(downsampled_contour[1].point_index, 1);

        let downsampled_contour = downsample_contour_points(&cont_points, 5);
        let n = downsampled_contour.len();
        assert_eq!(downsampled_contour[n - 1].point_index, 4);
    }

    // tests from here are AI generated!!
    #[test]
    fn test_downsample_edge_cases() {
        let points = vec![
            ContourPoint {
                frame_index: 1,
                point_index: 0,
                x: 1.0,
                y: 2.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 1,
                x: 3.0,
                y: 4.0,
                z: 0.0,
                aortic: false,
            },
        ];

        // Test n > points.len()
        let result = downsample_contour_points(&points, 5);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].point_index, 0);
        assert_eq!(result[1].point_index, 1);

        // Test n == points.len()
        let result = downsample_contour_points(&points, 2);
        assert_eq!(result.len(), 2);

        // Test n == 0
        let result = downsample_contour_points(&points, 0);
        assert_eq!(result.len(), 0);

        // Test empty input
        let empty: Vec<ContourPoint> = Vec::new();
        let result = downsample_contour_points(&empty, 3);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_search_range_quadratic_function() {
        // Test with a simple quadratic cost function: (angle - 0.5)^2
        let cost_fn = |angle: f64| (angle - 0.5).powi(2);

        let result = search_range(cost_fn, 1.0, 180.0, None, 180.0);

        // Should find minimum near 0.5 radians
        assert_relative_eq!(result, 0.5, epsilon = 1.0_f64.to_radians());
    }

    #[test]
    fn test_search_range_with_center_angle() {
        // Test with center angle provided
        let cost_fn = |angle: f64| (angle - 1.0).powi(2);

        let result = search_range(cost_fn, 0.5, 45.0, Some(0.8), 180.0);

        // Should find minimum near 1.0 radians, but constrained by center and range
        assert_relative_eq!(result, 1.0, epsilon = 0.5_f64.to_radians());
    }

    #[test]
    fn test_search_range_sine_function() {
        // Test with sine function - multiple minima, but should find one in range
        let cost_fn = |angle: f64| angle.sin();

        let result = search_range(cost_fn, 1.0, 90.0, None, 180.0);

        // Sine is minimized at -π/2, -5π/2, etc. Within [-π, π], minimum is -π/2 ≈ -1.57
        assert!(result <= 0.0); // Should find a negative value where sine is minimal
    }

    #[test]
    fn test_search_range_edge_cases() {
        // Test with zero step - should return center angle
        let cost_fn = |_angle: f64| 1.0;
        let result = search_range(cost_fn, 0.0, 90.0, Some(1.0), 180.0);
        assert_relative_eq!(result, 1.0, epsilon = 1e-10);

        // Test with very small range
        let cost_fn = |angle: f64| (angle - 0.1).powi(2);
        let result = search_range(cost_fn, 1.0, 1.0, Some(0.0), 180.0);
        // With small range (1°) around center 0.0, we can't reach 0.1 rad (~5.7°)
        // So should find minimum at the boundary closest to 0.1
        let expected_min = 1.0_f64.to_radians(); // 1° in radians ≈ 0.01745
        assert_relative_eq!(result, expected_min, epsilon = 0.5_f64.to_radians());

        // Test with range beyond limes
        let cost_fn = |angle: f64| (angle - 2.0).powi(2); // 2.0 rad ≈ 114.6°
        let result = search_range(cost_fn, 1.0, 180.0, None, 90.0); // limes = 90° ≈ 1.57 rad
                                                                    // Should be constrained by limes, so minimum at limes boundary (1.57)
        assert_relative_eq!(result, 1.57, epsilon = 1.0_f64.to_radians());

        // Test with negative step - should return center angle
        let result = search_range(cost_fn, -1.0, 90.0, Some(0.5), 180.0);
        assert_relative_eq!(result, 0.5, epsilon = 1e-10);

        // Test with no center angle provided
        let cost_fn = |angle: f64| (angle - 0.5).powi(2);
        let result = search_range(cost_fn, 0.0, 90.0, None, 180.0);
        assert_relative_eq!(result, 0.0, epsilon = 1e-10); // Should return default 0.0
    }

    #[test]
    fn test_search_range_small_range() {
        // Test with range too small to reach the true minimum
        let cost_fn = |angle: f64| (angle - 0.5).powi(2); // Minimum at 0.5 rad

        // Search with small range around 0.0 that doesn't include 0.5
        // Range: 0.2° ≈ 0.00349 rad, step: 0.1° ≈ 0.001745 rad
        let result = search_range(cost_fn, 0.1, 0.2, Some(0.0), 180.0);

        // Should find the best angle within the search range [0.0-0.00349, 0.0+0.00349]
        // The best in this range is the upper boundary 0.00349 rad
        let expected = 0.2_f64.to_radians(); // 0.0034906585 rad
        assert_relative_eq!(result, expected, epsilon = 0.1_f64.to_radians());

        // Test with range that exactly includes the minimum
        let result = search_range(cost_fn, 0.1, 30.0, Some(0.0), 180.0); // 30° ≈ 0.5236 rad
                                                                         // Should find something close to 0.5 (within the step size)
        assert_relative_eq!(result, 0.5, epsilon = 0.1_f64.to_radians());
    }

    #[test]
    fn test_hausdorff_distance_identical_sets() {
        let points = vec![
            ContourPoint {
                frame_index: 1,
                point_index: 0,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 1,
                x: 1.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 2,
                x: 0.0,
                y: 1.0,
                z: 0.0,
                aortic: false,
            },
        ];

        let distance = hausdorff_distance(&points, &points);
        assert_relative_eq!(distance, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hausdorff_distance_shifted_sets() {
        let set1 = vec![
            ContourPoint {
                frame_index: 1,
                point_index: 0,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 1,
                x: 1.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
        ];

        let set2 = vec![
            ContourPoint {
                frame_index: 1,
                point_index: 0,
                x: 2.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 1,
                x: 3.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
        ];

        let distance = hausdorff_distance(&set1, &set2);
        // The farthest points are (0,0) to (3,0) = 3.0, and (1,0) to (2,0) = 1.0
        // But Hausdorff takes max of directed distances, so should be 2.0
        // (from (0,0) to nearest in set2 is (2,0) = 2.0, from (1,0) to nearest is (2,0) = 1.0)
        // and similarly from set2 to set1 gives 2.0
        assert_relative_eq!(distance, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hausdorff_distance_different_sizes() {
        let set1 = vec![
            ContourPoint {
                frame_index: 1,
                point_index: 0,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 1,
                x: 3.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
        ];

        let set2 = vec![
            ContourPoint {
                frame_index: 1,
                point_index: 0,
                x: 1.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 1,
                x: 2.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 2,
                x: 4.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
        ];

        let distance = hausdorff_distance(&set1, &set2);
        // set1 to set2:
        //   (0,0) -> nearest is (1,0) = 1.0
        //   (3,0) -> nearest is (2,0) = 1.0
        // set2 to set1:
        //   (1,0) -> nearest is (0,0) = 1.0
        //   (2,0) -> nearest is (3,0) = 1.0
        //   (4,0) -> nearest is (3,0) = 1.0
        // So Hausdorff distance should be 1.0
        assert_relative_eq!(distance, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hausdorff_distance_empty_sets() {
        let empty: Vec<ContourPoint> = Vec::new();
        let points = vec![ContourPoint {
            frame_index: 1,
            point_index: 0,
            x: 1.0,
            y: 1.0,
            z: 0.0,
            aortic: false,
        }];

        // Empty set to non-empty should return 0.0 (as per current implementation)
        let distance1 = hausdorff_distance(&empty, &points);
        assert_relative_eq!(distance1, 0.0, epsilon = 1e-10);

        let distance2 = hausdorff_distance(&points, &empty);
        assert_relative_eq!(distance2, 0.0, epsilon = 1e-10);

        let distance3 = hausdorff_distance(&empty, &empty);
        assert_relative_eq!(distance3, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hausdorff_distance_complex_shapes() {
        // Create a square and a diamond that partially overlap
        let square = vec![
            ContourPoint {
                frame_index: 1,
                point_index: 0,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 1,
                x: 2.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 2,
                x: 2.0,
                y: 2.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 3,
                x: 0.0,
                y: 2.0,
                z: 0.0,
                aortic: false,
            },
        ];

        let diamond = vec![
            ContourPoint {
                frame_index: 1,
                point_index: 0,
                x: 1.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 1,
                x: 2.0,
                y: 1.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 2,
                x: 1.0,
                y: 2.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 3,
                x: 0.0,
                y: 1.0,
                z: 0.0,
                aortic: false,
            },
        ];

        let distance = hausdorff_distance(&square, &diamond);

        // The farthest points should be from square corners to diamond
        // Let's verify it's a reasonable value
        assert!(distance > 0.0);
        assert!(distance < 2.0); // Should be less than the diagonal
    }

    #[test]
    fn test_directed_hausdorff_consistency() {
        // Test that the directed Hausdorff is consistent with the full Hausdorff
        let set1 = vec![
            ContourPoint {
                frame_index: 1,
                point_index: 0,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 1,
                x: 1.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
        ];

        let set2 = vec![
            ContourPoint {
                frame_index: 1,
                point_index: 0,
                x: 2.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 1,
                x: 3.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
        ];

        let full_distance = hausdorff_distance(&set1, &set2);
        let directed_1_to_2 = directed_hausdorff(&set1, &set2);
        let directed_2_to_1 = directed_hausdorff(&set2, &set1);

        // Hausdorff distance should be the maximum of the two directed distances
        assert_relative_eq!(
            full_distance,
            directed_1_to_2.max(directed_2_to_1),
            epsilon = 1e-10
        );

        // For this case, both directed distances should be 2.0
        assert_relative_eq!(directed_1_to_2, 2.0, epsilon = 1e-10);
        assert_relative_eq!(directed_2_to_1, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_performance_hausdorff_large_sets() {
        // Create larger point sets to test performance
        let mut set1 = Vec::new();
        let mut set2 = Vec::new();

        for i in 0..100 {
            set1.push(ContourPoint {
                frame_index: 1,
                point_index: i,
                x: i as f64,
                y: 0.0,
                z: 0.0,
                aortic: false,
            });

            set2.push(ContourPoint {
                frame_index: 2,
                point_index: i,
                x: i as f64 + 0.5,
                y: 0.0,
                z: 0.0,
                aortic: false,
            });
        }

        // This should complete quickly with the parallel implementation
        let distance = hausdorff_distance(&set1, &set2);

        // Distance should be 0.5 (the constant offset)
        assert_relative_eq!(distance, 0.5, epsilon = 1e-10);
    }
}
