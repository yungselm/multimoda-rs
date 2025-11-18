use anyhow::Result;
use rayon::prelude::*;

use crate::intravascular::io::geometry::Geometry;
use crate::intravascular::io::input::ContourPoint;
use crate::intravascular::processing::process_utils::{
    downsample_contour_points, hausdorff_distance, search_range,
};

#[derive(Clone, Debug)]
pub struct GeometryPair {
    pub geom_a: Geometry,
    pub geom_b: Geometry,
    pub label: String,
}

impl GeometryPair {
    pub fn new(geom_a: Geometry, geom_b: Geometry) -> Result<Self> {
        let label = format!("{} - {}", geom_a.label, geom_b.label);
        Ok(Self {
            geom_a,
            geom_b,
            label,
        })
    }
}

pub fn align_between_geometries(
    geom_a: &mut Geometry,
    geom_b: &mut Geometry,
    rot_deg: f64,
    step_rot_deg: f64,
    sample_size: usize,
) -> Result<GeometryPair> {
    println!("Aligning geometry '{}' to '{}'", geom_b.label, geom_a.label);

    // Find reference frames
    let ref_frame_a_idx = geom_a
        .find_ref_frame_idx()
        .unwrap_or(geom_a.find_proximal_end_idx());
    let ref_frame_b_idx = geom_b
        .find_ref_frame_idx()
        .unwrap_or(geom_b.find_proximal_end_idx());

    let ref_frame_a = &geom_a.frames[ref_frame_a_idx];
    let ref_frame_b = &geom_b.frames[ref_frame_b_idx];

    let ref_frame_a_centroid = ref_frame_a.centroid;
    let ref_frame_b_centroid = ref_frame_b.centroid;

    println!("Reference frame A centroid: {:?}", ref_frame_a_centroid);
    println!("Reference frame B centroid: {:?}", ref_frame_b_centroid);

    // Calculate initial translation
    let initial_translation = (
        ref_frame_a_centroid.0 - ref_frame_b_centroid.0,
        ref_frame_a_centroid.1 - ref_frame_b_centroid.1,
        ref_frame_a_centroid.2 - ref_frame_b_centroid.2,
    );

    // Apply initial translation
    geom_b.translate_geometry(initial_translation);
    println!("Applied initial translation: {:?}", initial_translation);

    // Extract points for alignment (after initial translation)
    let test_geom_a = extract_geometry_points_with_frame_info(geom_a, sample_size.max(500));
    let test_geom_b = extract_geometry_points_with_frame_info(geom_b, sample_size.max(500));

    let best_rotation =
        find_best_rotation_between(&test_geom_a, &test_geom_b, step_rot_deg, rot_deg);

    println!(
        "Applying rotation of {:.2}° to align '{}' to '{}'",
        best_rotation.to_degrees(),
        geom_b.label,
        geom_a.label
    );

    // Apply the rotation to the ENTIRE geometry around the reference frame A centroid
    rotate_geometry_around_point(geom_b, best_rotation, ref_frame_a_centroid);

    // After rotation, recalculate and apply final translation to ensure perfect alignment
    let ref_frame_a_idx = geom_a
        .find_ref_frame_idx()
        .unwrap_or(geom_a.find_proximal_end_idx());
    let ref_frame_b_idx = geom_b
        .find_ref_frame_idx()
        .unwrap_or(geom_b.find_proximal_end_idx());

    let final_ref_frame_b_centroid = geom_b.frames[ref_frame_b_idx].centroid;
    let final_ref_frame_a_centroid = geom_a.frames[ref_frame_a_idx].centroid;
    let final_translation = (
        final_ref_frame_a_centroid.0 - final_ref_frame_b_centroid.0,
        final_ref_frame_a_centroid.1 - final_ref_frame_b_centroid.1,
        final_ref_frame_a_centroid.2 - final_ref_frame_b_centroid.2,
    );

    geom_b.translate_geometry(final_translation);

    Ok(GeometryPair::new(geom_a.clone(), geom_b.clone())?)
}

/// Rotate entire geometry around a single reference point
fn rotate_geometry_around_point(geometry: &mut Geometry, angle_rad: f64, center: (f64, f64, f64)) {
    let cos_angle = angle_rad.cos();
    let sin_angle = angle_rad.sin();

    // Helper closure to rotate a point around the common center
    let rotate_point = |x: f64, y: f64| -> (f64, f64) {
        let translated_x = x - center.0;
        let translated_y = y - center.1;

        let rotated_x = translated_x * cos_angle - translated_y * sin_angle;
        let rotated_y = translated_x * sin_angle + translated_y * cos_angle;

        (rotated_x + center.0, rotated_y + center.1)
    };

    // Rotate ALL frames around the same reference point
    for frame in &mut geometry.frames {
        // Rotate lumen points
        for point in &mut frame.lumen.points {
            let (new_x, new_y) = rotate_point(point.x, point.y);
            point.x = new_x;
            point.y = new_y;
        }

        // Rotate frame centroid
        let (new_cx, new_cy) = rotate_point(frame.centroid.0, frame.centroid.1);
        frame.centroid.0 = new_cx;
        frame.centroid.1 = new_cy;

        // Rotate extras
        for contour in frame.extras.values_mut() {
            for point in &mut contour.points {
                let (new_x, new_y) = rotate_point(point.x, point.y);
                point.x = new_x;
                point.y = new_y;
            }
            // Update contour centroid if it exists
            if let Some(centroid) = contour.centroid {
                let (new_cx, new_cy) = rotate_point(centroid.0, centroid.1);
                contour.centroid = Some((new_cx, new_cy, centroid.2));
            }
        }

        // Rotate reference point if it exists
        if let Some(ref mut rp) = frame.reference_point {
            let (new_x, new_y) = rotate_point(rp.x, rp.y);
            rp.x = new_x;
            rp.y = new_y;
        }
    }
}

#[derive(Debug, Clone)]
struct PointWithFrameInfo {
    pub point: ContourPoint,
    pub _frame_id: u32,
    pub _frame_centroid: (f64, f64, f64),
}

fn extract_geometry_points_with_frame_info(
    geometry: &Geometry,
    sample_size: usize,
) -> Vec<PointWithFrameInfo> {
    let total_points: usize = geometry.frames.iter().map(|f| f.lumen.points.len()).sum();

    let sample_ratio = sample_size as f64 / total_points as f64;

    let mut all_points = Vec::new();
    for frame in &geometry.frames {
        let frame_sample_size = (frame.lumen.points.len() as f64 * sample_ratio).ceil() as usize;
        let sampled = downsample_contour_points(&frame.lumen.points, frame_sample_size.max(1));

        for point in sampled {
            all_points.push(PointWithFrameInfo {
                point,
                _frame_id: frame.id,
                _frame_centroid: frame.centroid,
            });
        }
    }

    all_points
}

fn find_best_rotation_between(
    reference: &[PointWithFrameInfo],
    target: &[PointWithFrameInfo],
    step_deg: f64,
    range_deg: f64,
) -> f64 {
    // Use the reference geometry's global centroid as the rotation center for BOTH geometries
    let ref_centroid = calculate_global_centroid(reference);

    let cost_fn = |angle: f64| -> f64 {
        // Rotate ALL target points around the SAME global centroid
        let rotated_target: Vec<ContourPoint> = target
            .par_iter()
            .map(|point_info| {
                let translated_x = point_info.point.x - ref_centroid.0;
                let translated_y = point_info.point.y - ref_centroid.1;

                let cos_angle = angle.cos();
                let sin_angle = angle.sin();

                let rotated_x = translated_x * cos_angle - translated_y * sin_angle;
                let rotated_y = translated_x * sin_angle + translated_y * cos_angle;

                // Translate back to original coordinate system
                ContourPoint {
                    x: rotated_x + ref_centroid.0,
                    y: rotated_y + ref_centroid.1,
                    z: point_info.point.z,
                    ..point_info.point
                }
            })
            .collect();

        let reference_points: Vec<ContourPoint> =
            reference.iter().map(|pi| pi.point.clone()).collect();

        hausdorff_distance(&reference_points, &rotated_target)
    };

    // Multi-resolution search
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

fn calculate_global_centroid(points: &[PointWithFrameInfo]) -> (f64, f64, f64) {
    if points.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let sum_x: f64 = points.iter().map(|p| p.point.x).sum();
    let sum_y: f64 = points.iter().map(|p| p.point.y).sum();
    let sum_z: f64 = points.iter().map(|p| p.point.z).sum();
    let count = points.len() as f64;

    (sum_x / count, sum_y / count, sum_z / count)
}

#[cfg(test)]
mod align_between_tests {
    use approx::assert_relative_eq;

    use super::*;
    use crate::intravascular::utils::test_utils::dummy_geometry_aligned_long;

    #[test]
    fn test_align_between_simple_geometries() -> anyhow::Result<()> {
        let mut geom_a = dummy_geometry_aligned_long();
        let mut geom_b = dummy_geometry_aligned_long();
        let rotation: f64 = 15.0;
        geom_b.rotate_geometry(rotation.to_radians());

        let geom_pair = align_between_geometries(&mut geom_a, &mut geom_b, 30.0, 0.01, 6)?;

        for (frame_a, frame_b) in geom_pair
            .geom_a
            .frames
            .iter()
            .zip(geom_pair.geom_b.frames.iter())
        {
            assert_relative_eq!(frame_a.centroid.2, frame_b.centroid.2, epsilon = 1e-6);
            for (point_a, point_b) in frame_a.lumen.points.iter().zip(frame_b.lumen.points.iter()) {
                assert_relative_eq!(point_a.x, point_b.x, epsilon = 1e-6);
                assert_relative_eq!(point_a.y, point_b.y, epsilon = 1e-6);
                assert_relative_eq!(point_a.z, point_b.z, epsilon = 1e-6);
            }
        }
        Ok(())
    }

    #[test]
    fn test_align_between_optimized_geometries() -> anyhow::Result<()> {
        use crate::intravascular::io::build_geometry_from_inputdata;
        use crate::intravascular::processing::align_within::align_frames_in_geometry;
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

        let (geom, _, _) = align_frames_in_geometry(&mut geometry, 0.01, 45.0, true, false, 200)?;

        let rotation: f64 = 15.0;
        let mut geom_a = geom.clone();
        let mut geom_b = geom.clone();

        // Apply rotation to geometry B
        let ref_frame_b_idx = geom_b.find_proximal_end_idx();
        let ref_frame_b_centroid = geom_b.frames[ref_frame_b_idx].centroid;
        rotate_geometry_around_point(&mut geom_b, rotation.to_radians(), ref_frame_b_centroid);

        let geom_pair = align_between_geometries(&mut geom_a, &mut geom_b, 30.0, 0.01, 500)?;

        let mut max_error = 0.0f64;
        let mut total_error = 0.0f64;
        let mut point_count = 0;

        for (frame_a, frame_b) in geom_pair
            .geom_a
            .frames
            .iter()
            .zip(geom_pair.geom_b.frames.iter())
        {
            assert_relative_eq!(frame_a.centroid.2, frame_b.centroid.2, epsilon = 1e-4);

            // Ensure we have the same number of points
            assert_eq!(frame_a.lumen.points.len(), frame_b.lumen.points.len());

            for (point_a, point_b) in frame_a.lumen.points.iter().zip(frame_b.lumen.points.iter()) {
                let error_x = (point_a.x - point_b.x).abs();
                let error_y = (point_a.y - point_b.y).abs();
                let max_point_error = error_x.max(error_y);

                max_error = max_error.max(max_point_error);
                total_error += error_x + error_y;
                point_count += 2;
            }
        }

        let avg_error = total_error / point_count as f64;

        // Verify alignment precision
        assert!(
            max_error < 0.01,
            "Maximum alignment error {} exceeds threshold",
            max_error
        );
        assert!(
            avg_error < 0.001,
            "Average alignment error {} exceeds threshold",
            avg_error
        );

        Ok(())
    }
}
