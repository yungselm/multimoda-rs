use crate::types::native::{Centerline, CenterlinePoint};
use nalgebra::Point3;
use rayon::prelude::*;
use std::collections::HashSet;

type Coords3 = (f64, f64, f64);

pub fn centerline_based_wall_diameter_optimization(
    centerline: &Centerline,
    ref_point_coronary: &Coords3,
    aortic_points: &[Coords3],
) -> f64 {
    if centerline.points.is_empty() || aortic_points.is_empty() {
        return 0.0;
    }

    let closest_cl = match centerline.points.iter().min_by(|a, b| {
        let dist_a = super::calculate_squared_distance(*a, ref_point_coronary);
        let dist_b = super::calculate_squared_distance(*b, ref_point_coronary);
        dist_a
            .partial_cmp(&dist_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        Some(pt) => pt,
        None => return 0.0,
    };

    let closest_aortic: &Coords3 = match aortic_points.iter().min_by(|a, b| {
        let dist_a = super::calculate_squared_distance(*a, ref_point_coronary);
        let dist_b = super::calculate_squared_distance(*b, ref_point_coronary);
        dist_a
            .partial_cmp(&dist_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        Some(pt) => pt,
        None => return 0.0,
    };

    let ref_point = Point3::new(
        ref_point_coronary.0,
        ref_point_coronary.1,
        ref_point_coronary.2,
    );
    let closest_cl_point = Point3::new(
        closest_cl.contour_point.x,
        closest_cl.contour_point.y,
        closest_cl.contour_point.z,
    );
    let closest_aortic_point = Point3::new(closest_aortic.0, closest_aortic.1, closest_aortic.2);

    // vector centerline point to coronary point
    let vector = ref_point - closest_cl_point;
    let Some(unit) = vector.try_normalize(0.0) else {
        return 0.0;
    };

    // project (ref_point_coronary - closest_aortic) onto unit direction
    // minimizes ||closest_aortic + t*unit - ref_point_coronary||², constrained t >= 0
    let to_ref = ref_point - closest_aortic_point;
    let t = to_ref.dot(&unit);

    t.max(0.0)
}

pub fn centerline_based_aortic_diameter_optimization(
    intramural_points: &[Coords3],
    reference_points: &[Coords3],
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
        let temp_points = centerline_based_diameter_morphing(centerline, intramural_points, x);
        let dist = symmetric_nn_distance(reference_points, &temp_points);
        if dist < min_dist {
            min_dist = dist;
            scaling_best = x;
        }
    }
    scaling_best
}

pub fn centerline_based_diameter_optimization(
    anomalous_points: &[Coords3],
    n_proximal: usize,
    n_distal: usize,
    centerline: &Centerline,
    proximal_reference: &[Coords3],
    distal_reference: &[Coords3],
) -> (f64, f64) {
    let (proximal_points, anomalous_points_new) =
        find_region_points(anomalous_points, proximal_reference, n_proximal);
    let (distal_points, _) = find_region_points(&anomalous_points_new, distal_reference, n_distal);

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
        let dist = symmetric_nn_distance(proximal_reference, &temp_points);
        if dist < min_dist_proximal {
            min_dist_proximal = dist;
            prox_scaling_best = x;
        }
    }
    for i in 0..=steps {
        let x = start + i as f64 * step;
        let temp_points = centerline_based_diameter_morphing(centerline, &distal_points, x);
        let dist = symmetric_nn_distance(distal_reference, &temp_points);
        if dist < min_dist_distal {
            min_dist_distal = dist;
            dist_scaling_best = x;
        }
    }
    (prox_scaling_best, dist_scaling_best)
}

fn find_region_points(
    anomalous_points: &[Coords3],
    reference_points: &[Coords3],
    n_points: usize,
) -> (Vec<Coords3>, Vec<Coords3>) {
    if anomalous_points.is_empty() || reference_points.is_empty() || n_points == 0 {
        return (Vec::new(), anomalous_points.to_vec());
    }

    let mut indexed_dists: Vec<(usize, f64)> = anomalous_points
        .iter()
        .enumerate()
        .map(|(i, a_pt)| {
            let min_sq = reference_points
                .iter()
                .map(|r_pt| super::calculate_squared_distance(a_pt, r_pt))
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

    let selected_points: Vec<Coords3> = selected_slice
        .iter()
        .map(|(i, _)| anomalous_points[*i])
        .collect();

    let remaining_points: Vec<Coords3> = anomalous_points
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
fn symmetric_nn_distance(a: &[Coords3], b: &[Coords3]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return f64::INFINITY;
    }

    let sum_a_to_b: f64 = a
        .par_iter()
        .map(|pa| {
            b.iter()
                .map(|pb| super::calculate_squared_distance(pa, pb))
                .fold(f64::INFINITY, |m, v| if v < m { v } else { m })
        })
        .sum();

    let avg_a_to_b = sum_a_to_b / (a.len() as f64);

    let sum_b_to_a: f64 = b
        .par_iter()
        .map(|pb| {
            a.iter()
                .map(|pa| super::calculate_squared_distance(pb, pa))
                .fold(f64::INFINITY, |m, v| if v < m { v } else { m })
        })
        .sum();

    let avg_b_to_a = sum_b_to_a / (b.len() as f64);

    ((avg_a_to_b + avg_b_to_a) / 2.0).sqrt()
}

pub fn centerline_based_diameter_morphing(
    centerline: &Centerline,
    points: &[Coords3],
    diameter_adjustment_mm: f64,
) -> Vec<Coords3> {
    points
        .par_iter()
        .map(|point| {
            let closest_cl_point = find_closest_centerline_point_optimized(centerline, *point);
            let p = Point3::new(point.0, point.1, point.2);
            let cl_point = Point3::new(
                closest_cl_point.contour_point.x,
                closest_cl_point.contour_point.y,
                closest_cl_point.contour_point.z,
            );

            match (p - cl_point).try_normalize(0.0) {
                Some(unit) => {
                    let moved = p + unit * diameter_adjustment_mm;
                    (moved.x, moved.y, moved.z)
                }
                None => *point,
            }
        })
        .collect()
}

fn find_closest_centerline_point_optimized(
    centerline: &Centerline,
    point: Coords3,
) -> &CenterlinePoint {
    let mut min_distance_squared = f64::MAX;
    let mut closest_point = &centerline.points[0];

    for centerline_point in &centerline.points {
        let distance_squared = super::calculate_squared_distance(&point, centerline_point);
        if distance_squared < min_distance_squared {
            min_distance_squared = distance_squared;
            closest_point = centerline_point;
        }
    }

    closest_point
}

#[cfg(test)]
mod tests {
    use crate::types::native::{CenterlinePoint, ContourPoint};
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
                    tangent: Vector3::new(1.0, 0.0, 0.0),
                    branch_id: 0,
                    radius: 0.0,
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
                    tangent: Vector3::new(1.0, 0.0, 0.0),
                    branch_id: 0,
                    radius: 0.0,
                },
            ],
            branch_start_indices: vec![0],
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
                tangent: Vector3::new(1.0, 0.0, 0.0),
                branch_id: 0,
                radius: 0.0,
            }],
            branch_start_indices: vec![0],
        };

        let points = vec![(2.0, 0.0, 0.0)];

        let result = centerline_based_diameter_morphing(&centerline, &points, -0.5);

        // Should move toward centerline by 0.5 units
        let new_point = result[0];
        assert!((new_point.0 - 1.5).abs() < 1e-6);
        assert!((new_point.1 - 0.0).abs() < 1e-6);
        assert!((new_point.2 - 0.0).abs() < 1e-6);
    }
}
