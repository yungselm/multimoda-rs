pub mod align_algorithms;
pub mod preprocessing;

use crate::intravascular::centerline_align::align_algorithms::{
    get_transformations, FrameTransformation,
};
use crate::intravascular::io::{
    geometry::{ContourType, Geometry},
    input::Centerline,
};
use crate::intravascular::processing::align_between::GeometryPair;
use anyhow::anyhow;

use crate::intravascular::to_object::process_case;
use align_algorithms::best_rotation_three_point;
use preprocessing::preprocess_centerline;

pub fn align_three_point_rs(
    centerline: Centerline,
    mut geom_pair: GeometryPair,
    aortic_ref_pt: (f64, f64, f64),
    upper_ref_pt: (f64, f64, f64),
    lower_ref_pt: (f64, f64, f64),
    angle_step: f64,
    write: bool,
    watertight: bool,
    interpolation_steps: usize,
    output_dir: &str,
    contour_types: Vec<ContourType>,
    case_name: &str,
) -> anyhow::Result<(GeometryPair, Centerline)> {
    let resampled_centerline = preprocess_centerline(centerline, &geom_pair.geom_a)
        .map_err(|e| anyhow!("Couldn't resample the centerline: {}", e))?;

    let ref_idx = geom_pair
        .geom_a
        .find_ref_frame_idx()
        .map_err(|e| anyhow!("Couldn't find ref frame idx: {:?}", e))?;
    let ref_point = geom_pair.geom_a.frames[ref_idx]
        .reference_point
        .as_ref()
        .ok_or_else(|| anyhow!("missing reference point"))?;
    let cl_ref_idx = resampled_centerline.find_reference_cl_point_idx(&aortic_ref_pt);

    let best_rot = best_rotation_three_point(
        &geom_pair.geom_a.frames[ref_idx].lumen,
        ref_point,
        aortic_ref_pt,
        upper_ref_pt,
        lower_ref_pt,
        angle_step,
        &resampled_centerline.points[cl_ref_idx],
    );

    geom_pair = rotate_by_best_rotation(geom_pair, best_rot);
    geom_pair = apply_transformations(geom_pair, &resampled_centerline, &aortic_ref_pt);

    geom_pair = if write {
        process_case(
            case_name,
            geom_pair,
            output_dir,
            interpolation_steps,
            watertight,
            &contour_types,
        )
        .map_err(|e| anyhow!("Failed to write obj: {}", e))?
    } else {
        geom_pair
    };

    Ok((geom_pair, resampled_centerline))
}

pub fn align_manual_rs(
    centerline: Centerline,
    mut geom_pair: GeometryPair,
    rotation_angle: f64,
    ref_pt: (f64, f64, f64),
    write: bool,
    watertight: bool,
    interpolation_steps: usize,
    output_dir: &str,
    contour_types: Vec<ContourType>,
    case_name: &str,
) -> anyhow::Result<(GeometryPair, Centerline)> {
    let resampled_centerline = preprocess_centerline(centerline, &geom_pair.geom_a)
        .map_err(|e| anyhow!("Couldn't resample the centerline: {}", e))?;

    geom_pair = rotate_by_best_rotation(geom_pair, rotation_angle);

    geom_pair = apply_transformations(geom_pair, &resampled_centerline, &ref_pt);

    geom_pair = if write {
        process_case(
            case_name,
            geom_pair,
            output_dir,
            interpolation_steps,
            watertight,
            &contour_types,
        )
        .map_err(|e| anyhow!("Failed to write obj: {}", e))?
    } else {
        geom_pair
    };

    Ok((geom_pair, resampled_centerline))
}

// pub fn align_hausdorff() -> () {
//     todo!()
// }

fn rotate_by_best_rotation(mut geom_pair: GeometryPair, angle: f64) -> GeometryPair {
    geom_pair.geom_a.rotate_geometry(angle);
    geom_pair.geom_b.rotate_geometry(angle);

    geom_pair
}

fn apply_transformations(
    mut geom_pair: GeometryPair,
    centerline: &Centerline,
    ref_pt: &(f64, f64, f64),
) -> GeometryPair {
    // Create transformations using geom_a as reference
    let transformations = get_transformations(geom_pair.geom_a.clone(), centerline, ref_pt);

    // Helper function to apply transformations to a geometry
    fn transform_geometry(
        mut geometry: Geometry,
        transformations: &[FrameTransformation],
    ) -> Geometry {
        // We assume transformations are in the same order as geometry frames
        for (i, frame) in geometry.frames.iter_mut().enumerate() {
            if i < transformations.len() {
                let tr = &transformations[i];

                // Transform lumen contour
                align_algorithms::apply_transformation_to_contour(&mut frame.lumen, tr);

                // Transform extra contours
                for contour in frame.extras.values_mut() {
                    align_algorithms::apply_transformation_to_contour(contour, tr);
                }

                // Transform reference point if it exists
                if let Some(ref mut reference_pt) = frame.reference_point {
                    *reference_pt = tr.apply_to_point(reference_pt);
                }

                // Update frame centroid
                frame.centroid = frame.lumen.centroid.unwrap_or((0.0, 0.0, 0.0));
            }
        }
        geometry
    }

    geom_pair.geom_a = transform_geometry(geom_pair.geom_a, &transformations);
    geom_pair.geom_b = transform_geometry(geom_pair.geom_b, &transformations);

    geom_pair
}
