use super::calculate_squared_distance;
use crate::intravascular::io::geometry::Frame;
use crate::intravascular::io::input::{Centerline, CenterlinePoint};
use core::f64;
use std::collections::HashSet;

pub fn centerline_based_aortic_diameter_optimization(
    intramural_points: &[(f64, f64, f64)],
    reference_points: &[(f64, f64, f64)],
    centerline: &Centerline,
) -> f64 {
    let start = -2.0f64;
    let end = 2.0f64;
    let step = 0.1f64;
    let steps = ((end - start) / step).round() as i32; // (4.0 / 0.1) => 40

    let mut min_dist = f64::MAX;
    let mut scaling_best = f64::MAX;

    for i in 0..=steps {
        let x = start + i as f64 * step;
        let temp_points = centerline_based_diameter_morphing(centerline, &intramural_points, x);

        let dist = symmetric_nn_distance(&reference_points, &temp_points);

        if dist < min_dist {
            min_dist = dist;
            scaling_best = x;
        }
    }
    scaling_best
}

pub fn centerline_based_diameter_optimization(
    anomalous_points: &[(f64, f64, f64)],
    n_proximal: usize,
    n_distal: usize,
    centerline: &Centerline,
    proximal_reference: &[(f64, f64, f64)],
    distal_reference: &[(f64, f64, f64)],
) -> (f64, f64) {
    let (proximal_points, anomalous_points_new) =
        find_region_points(&anomalous_points, &proximal_reference, n_proximal);
    let (distal_points, _) = find_region_points(&anomalous_points_new, &distal_reference, n_distal);

    let start = -2.0f64;
    let end = 2.0f64;
    let step = 0.1f64;
    let steps = ((end - start) / step).round() as i32; // (4.0 / 0.1) => 40

    let mut min_dist_proximal = f64::MAX;
    let mut min_dist_distal = f64::MAX;
    let mut prox_scaling_best = f64::MAX;
    let mut dist_scaling_best = f64::MAX;

    for i in 0..=steps {
        let x = start + i as f64 * step;
        let temp_points = centerline_based_diameter_morphing(centerline, &proximal_points, x);

        let dist = symmetric_nn_distance(&proximal_reference, &temp_points);

        if dist < min_dist_proximal {
            min_dist_proximal = dist;
            prox_scaling_best = x;
        }
    }
    for i in 0..=steps {
        let x = start + i as f64 * step;
        let temp_points = centerline_based_diameter_morphing(centerline, &distal_points, x);

        let dist = symmetric_nn_distance(&distal_reference, &temp_points);

        if dist < min_dist_distal {
            min_dist_distal = dist;
            dist_scaling_best = x;
        }
    }
    (prox_scaling_best, dist_scaling_best)
}

fn find_region_points(
    anomalous_points: &[(f64, f64, f64)],
    reference_points: &[(f64, f64, f64)],
    n_points: usize,
) -> (Vec<(f64, f64, f64)>, Vec<(f64, f64, f64)>) {
    if anomalous_points.is_empty() || reference_points.is_empty() || n_points == 0 {
        return (Vec::new(), anomalous_points.to_vec());
    }

    let mut indexed_dists: Vec<(usize, f64)> = anomalous_points
        .iter()
        .enumerate()
        .map(|(i, a_pt)| {
            let min_sq = reference_points
                .iter()
                .map(|r_pt| calculate_squared_distance(a_pt, r_pt))
                .fold(f64::INFINITY, |m, v| if v < m { v } else { m });
            (i, min_sq)
        })
        .collect();

    indexed_dists.sort_by(|(i1, d1), (i2, d2)| {
        d1.partial_cmp(d2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| i1.cmp(i2))
    });

    let take = n_points.min(anomalous_points.len());
    let selected_slice = &indexed_dists[..take];

    let selected_indices: HashSet<usize> = selected_slice.iter().map(|(i, _)| *i).collect();

    let selected_points: Vec<(f64, f64, f64)> = selected_slice
        .iter()
        .map(|(i, _)| anomalous_points[*i])
        .collect();

    let remaining_points: Vec<(f64, f64, f64)> = anomalous_points
        .iter()
        .enumerate()
        .filter_map(|(i, pt)| {
            if selected_indices.contains(&i) {
                None
            } else {
                Some(*pt)
            }
        })
        .collect();

    (selected_points, remaining_points)
}

/// Symmetric average nearest-neighbor distance between two point sets.
/// Returns sqrt of average squared distance for readability (i.e., RMS of nearest neighbor distances).
/// If either set is empty returns f64::INFINITY.
fn symmetric_nn_distance(a: &[(f64, f64, f64)], b: &[(f64, f64, f64)]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return f64::INFINITY;
    }

    let sum_a_to_b: f64 = a
        .iter()
        .map(|pa| {
            b.iter()
                .map(|pb| calculate_squared_distance(pa, pb))
                .fold(f64::INFINITY, |m, v| if v < m { v } else { m })
        })
        .sum();

    let avg_a_to_b = sum_a_to_b / (a.len() as f64);

    let sum_b_to_a: f64 = b
        .iter()
        .map(|pb| {
            a.iter()
                .map(|pa| calculate_squared_distance(pb, pa))
                .fold(f64::INFINITY, |m, v| if v < m { v } else { m })
        })
        .sum();

    let avg_b_to_a = sum_b_to_a / (b.len() as f64);

    ((avg_a_to_b + avg_b_to_a) / 2.0).sqrt()
}

pub fn centerline_based_diameter_morphing(
    centerline: &Centerline,
    points: &[(f64, f64, f64)],
    diameter_adjustment_mm: f64,
) -> Vec<(f64, f64, f64)> {
    let mut result_points = Vec::with_capacity(points.len());

    for point in points.iter() {
        let closest_cl_point = find_closest_centerline_point_optimized(centerline, *point);

        let vector = (
            point.0 - closest_cl_point.contour_point.x,
            point.1 - closest_cl_point.contour_point.y,
            point.2 - closest_cl_point.contour_point.z,
        );

        let magnitude = (vector.0 * vector.0 + vector.1 * vector.1 + vector.2 * vector.2).sqrt();

        if magnitude > 0.0 {
            let normalized_vector = (
                vector.0 / magnitude,
                vector.1 / magnitude,
                vector.2 / magnitude,
            );

            let new_point = (
                point.0 + normalized_vector.0 * diameter_adjustment_mm,
                point.1 + normalized_vector.1 * diameter_adjustment_mm,
                point.2 + normalized_vector.2 * diameter_adjustment_mm,
            );

            result_points.push(new_point);
        } else {
            // If point is exactly on centerline, we can't determine direction
            // Just keep the original point
            result_points.push(*point);
        }
    }

    result_points
}

fn find_closest_centerline_point_optimized(
    centerline: &Centerline,
    point: (f64, f64, f64),
) -> &CenterlinePoint {
    let mut min_distance_squared = f64::MAX;
    let mut closest_point = &centerline.points[0];

    for centerline_point in &centerline.points {
        let distance_squared = calculate_squared_distance(&point, centerline_point);
        if distance_squared < min_distance_squared {
            min_distance_squared = distance_squared;
            closest_point = centerline_point;
        }
    }

    closest_point
}

pub fn find_points_by_cl_region_rs(
    centerline: &Centerline,
    frames: &[Frame],
    points: &[(f64, f64, f64)],
) -> (
    Vec<(f64, f64, f64)>,
    Vec<(f64, f64, f64)>,
    Vec<(f64, f64, f64)>,
) {
    let mut cumulative_z_dist_frames = 0.0;
    for i in 1..frames.len() {
        cumulative_z_dist_frames += (frames[i].centroid.2 - frames[i - 1].centroid.2).abs();
    }
    cumulative_z_dist_frames /= (frames.len() - 1) as f64;

    let centroids_to_match = frames
        .iter()
        .map(|f| f.centroid)
        .collect::<Vec<(f64, f64, f64)>>();
    let cl_points_indices: Vec<usize> =
        find_cl_points_in_range(centerline, &centroids_to_match, cumulative_z_dist_frames);

    // needed for proximal/distal classification
    let dist_ref = centroids_to_match[centroids_to_match.len() - 1];

    let mut proximal_points: Vec<(f64, f64, f64)> = Vec::new();
    let mut distal_points: Vec<(f64, f64, f64)> = Vec::new();
    let mut points_between: Vec<(f64, f64, f64)> = Vec::new();

    let mut remaining_points = points.to_vec();

    // First pass: find all points between centerline regions
    remaining_points.retain(|point| {
        let closest_cl_point = find_closest_centerline_point_optimized(&centerline, *point);
        let cl_idx = closest_cl_point.contour_point.frame_index as usize;

        if cl_points_indices.contains(&cl_idx) {
            points_between.push(*point);
            false // remove from remaining
        } else {
            true // keep in remaining
        }
    });

    // Second pass: classify remaining points as proximal or distal
    for point in remaining_points.iter() {
        if point.0 > dist_ref.0 && point.1 > dist_ref.1 && point.2 > dist_ref.2 {
            proximal_points.push(*point);
        } else {
            distal_points.push(*point);
        }
    }
    let (proximal_points, points_between) =
        clean_up_non_section_points(proximal_points, points_between, 1.0, 0.6);
    let (distal_points, points_between) =
        clean_up_non_section_points(distal_points, points_between, 1.0, 0.6);
    (proximal_points, distal_points, points_between)
}

fn find_cl_points_in_range(
    centerline: &Centerline,
    points: &[(f64, f64, f64)],
    search_radius: f64,
) -> Vec<usize> {
    let mut selected_points = Vec::new();

    for point in points.iter() {
        for cl_point in centerline.points.iter() {
            let dx = point.0 - cl_point.contour_point.x;
            let dy = point.1 - cl_point.contour_point.y;
            let dz = point.2 - cl_point.contour_point.z;
            let distance_squared = dx * dx + dy * dy + dz * dz;
            if distance_squared <= search_radius * search_radius {
                selected_points.push(cl_point);
            }
        }
    }

    // remove duplicates
    selected_points.sort_by_key(|p| p.contour_point.frame_index);
    selected_points.dedup_by_key(|p| p.contour_point.frame_index);
    let mut final_points = Vec::new();
    for p in selected_points {
        final_points.push(p.contour_point.frame_index as usize);
    }
    final_points
}

pub fn clean_up_non_section_points(
    points_to_cleanup: Vec<(f64, f64, f64)>,
    reference_points: Vec<(f64, f64, f64)>,
    neighborhood_radius: f64,
    min_neigbor_ratio: f64,
) -> (Vec<(f64, f64, f64)>, Vec<(f64, f64, f64)>) {
    let neighborhood_radius_sq = neighborhood_radius * neighborhood_radius;

    let mut cleaned_points = Vec::new();
    let mut reassigned_points = reference_points.clone();

    for point in points_to_cleanup.iter() {
        let mut ref_neighbors = 0;
        let mut total_neighbors = 0;

        for ref_point in reference_points.iter() {
            let dx = point.0 - ref_point.0;
            let dy = point.1 - ref_point.1;
            let dz = point.2 - ref_point.2;
            let distance_squared = dx * dx + dy * dy + dz * dz;

            if distance_squared <= neighborhood_radius_sq {
                ref_neighbors += 1;
                total_neighbors += 1;
            }
        }

        for other_point in points_to_cleanup.iter() {
            if std::ptr::eq(point, other_point) {
                continue; // Skip the point itself
            }

            let dx = point.0 - other_point.0;
            let dy = point.1 - other_point.1;
            let dz = point.2 - other_point.2;
            let distance_squared = dx * dx + dy * dy + dz * dz;

            if distance_squared <= neighborhood_radius_sq {
                total_neighbors += 1;
            }
        }

        // Decision logic: if most neighbors are reference points, reassign
        if total_neighbors > 0 {
            let ref_ratio = ref_neighbors as f64 / total_neighbors as f64;
            if ref_ratio >= min_neigbor_ratio {
                // Reassign to reference_points (anomalous)
                reassigned_points.push(*point);
            } else {
                // Keep in cleaned_points (proximal/distal)
                cleaned_points.push(*point);
            }
        } else {
            // If no neighbors in range, keep original classification
            cleaned_points.push(*point);
        }
    }

    (cleaned_points, reassigned_points)
}

#[cfg(test)]
mod tests {
    use crate::intravascular::io::input::{CenterlinePoint, ContourPoint};
    use nalgebra::Vector3;

    use super::*;

    #[test]
    fn test_centerline_based_diameter_morphing() {
        let centerline = Centerline {
            points: vec![
                CenterlinePoint {
                    contour_point: ContourPoint {
                        frame_index: 0,
                        point_index: 0,
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                        aortic: false,
                    },
                    normal: Vector3::new(1.0, 0.0, 0.0).into(),
                },
                CenterlinePoint {
                    contour_point: ContourPoint {
                        frame_index: 1,
                        point_index: 1,
                        x: 1.0,
                        y: 0.0,
                        z: 0.0,
                        aortic: false,
                    },
                    normal: Vector3::new(1.0, 0.0, 0.0).into(),
                },
            ],
        };

        let points = vec![
            (1.0, 1.0, 0.0), // Point at (1,1,0) - closest to centerline point (1,0,0)
        ];

        let result = centerline_based_diameter_morphing(&centerline, &points, 1.0);

        // The point should move from (1,1,0) to (1,2,0) - same direction but 1 unit further
        assert_eq!(result.len(), 1);
        let new_point = result[0];
        assert!((new_point.0 - 1.0).abs() < 1e-6);
        assert!((new_point.1 - 2.0).abs() < 1e-6);
        assert!((new_point.2 - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_negative_adjustment() {
        let centerline = Centerline {
            points: vec![CenterlinePoint {
                contour_point: ContourPoint {
                    frame_index: 0,
                    point_index: 0,
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    aortic: false,
                },
                normal: Vector3::new(1.0, 0.0, 0.0).into(),
            }],
        };

        let points = vec![(2.0, 0.0, 0.0)];

        let result = centerline_based_diameter_morphing(&centerline, &points, -0.5);

        // Should move toward centerline by 0.5 units
        let new_point = result[0];
        assert!((new_point.0 - 1.5).abs() < 1e-6);
        assert!((new_point.1 - 0.0).abs() < 1e-6);
        assert!((new_point.2 - 0.0).abs() < 1e-6);
    }

    // #[test]
    // fn test_centerline_based_diameter_optimization_basic() {
    //     let proximal_points = vec![
    //         (1.0, 0.0, 0.0),
    //         (1.0, 1.0, 0.0),
    //         (1.0, -1.0, 0.0),
    //     ];

    //     let distal_points = vec![
    //         (2.0, 0.0, 0.0),
    //         (2.0, 1.0, 0.0),
    //         (2.0, -1.0, 0.0),
    //     ];

    //     // Reference points exactly match both regions
    //     let reference_points = vec![
    //         (1.0, 0.0, 0.0),
    //         (1.0, 1.0, 0.0),
    //         (1.0, -1.0, 0.0),
    //         (2.0, 0.0, 0.0),
    //         (2.0, 1.0, 0.0),
    //         (2.0, -1.0, 0.0),
    //     ];

    //     let centerline = Centerline {
    //         points: vec![
    //             CenterlinePoint {
    //                 contour_point: ContourPoint {
    //                     frame_index: 0,
    //                     point_index: 0,
    //                     x: 0.0,
    //                     y: 0.0,
    //                     z: 0.0,
    //                     aortic: false,
    //                 },
    //                 normal: Vector3::new(1.0, 0.0, 0.0).into(),
    //             },
    //             CenterlinePoint {
    //                 contour_point: ContourPoint {
    //                     frame_index: 1,
    //                     point_index: 1,
    //                     x: 0.0,
    //                     y: 0.0,
    //                     z: 1.0,
    //                     aortic: false,
    //                 },
    //                 normal: Vector3::new(1.0, 0.0, 0.0).into(),
    //             },
    //         ],
    //     };

    //     let (min_prox, min_dist) = centerline_based_diameter_optimization(
    //         &proximal_points,
    //         &distal_points,
    //         &centerline,
    //         &reference_points,
    //         &reference_points,
    //     );

    //     // Best match occurs at diameter_adjustment_mm ≈ 0.0
    //     assert!(
    //         min_prox < 1e-6,
    //         "Expected proximal min distance ≈ 0, got {}",
    //         min_prox
    //     );

    //     assert!(
    //         min_dist < 1e-6,
    //         "Expected distal min distance ≈ 0, got {}",
    //         min_dist
    //     );
    // }
}
