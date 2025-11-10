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
    println!(
        "Aligning geometry '{}' to geometry '{}'",
        geom_b.label, geom_a.label
    );
    let ref_frame_a_idx = geom_a
        .find_ref_frame_idx()
        .unwrap_or(geom_a.find_proximal_end_idx());
    let ref_frame_b_idx = geom_b
        .find_ref_frame_idx()
        .unwrap_or(geom_b.find_proximal_end_idx());

    let ref_frame_a = &geom_a.frames[ref_frame_a_idx];
    let ref_frame_b = &geom_b.frames[ref_frame_b_idx];

    let translation = (
        ref_frame_a.centroid.0 - ref_frame_b.centroid.0,
        ref_frame_a.centroid.1 - ref_frame_b.centroid.1,
        ref_frame_a.centroid.2 - ref_frame_b.centroid.2,
    );
    geom_b.translate_geometry(translation);

    let test_geom_a = extract_geometry_points_with_frame_info(geom_a, sample_size);
    let test_geom_b = extract_geometry_points_with_frame_info(geom_b, sample_size);

    let best_rotation =
        find_best_rotation_between(&test_geom_a, &test_geom_b, step_rot_deg, rot_deg);

    for frame in &mut geom_b.frames {
        frame.rotate_frame(best_rotation);
    }

    println!(
        "Best rotation to align '{}' to '{}' is {:.2} degrees",
        geom_b.label,
        geom_a.label,
        best_rotation.to_degrees(),
    );

    Ok(GeometryPair::new(geom_a.clone(), geom_b.clone())?)
}

#[derive(Debug, Clone)]
struct PointWithFrameInfo {
    pub point: ContourPoint,
    pub _frame_id: u32,
    pub frame_centroid: (f64, f64, f64),
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
                frame_centroid: frame.centroid,
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
    let cost_fn = |angle: f64| -> f64 {
        // Rotate each target point around its own frame's centroid
        let rotated_target: Vec<ContourPoint> = target
            .par_iter()
            .map(|point_info| {
                let centroid = point_info.frame_centroid;
                point_info
                    .point
                    .rotate_point(angle, (centroid.0, centroid.1))
            })
            .collect();

        let reference_points: Vec<ContourPoint> =
            reference.iter().map(|pi| pi.point.clone()).collect();

        hausdorff_distance(&reference_points, &rotated_target)
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
