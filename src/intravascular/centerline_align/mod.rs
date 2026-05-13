pub mod align_algorithms;
pub mod preprocessing;

use crate::intravascular::io::input::ContourPoint;
use crate::intravascular::io::{
    geometry::{ContourType, Geometry},
    input::Centerline,
};
use crate::intravascular::processing::align_between::GeometryPair;
use anyhow::anyhow;

use crate::intravascular::to_object::{process_case, write_single_geometry};
use align_algorithms::{
    apply_transformations, best_rotation_three_point, refine_alignment_hausdorff,
    rotate_by_best_rotation, AlignTarget,
};
use preprocessing::preprocess_centerline;

/// Extends [`AlignTarget`] with the ability to write the processed case to disk.
///
/// `Geometry` is a no-op (writing requires a pair for mesh comparison);
/// `GeometryPair` delegates to [`process_case`].
pub trait Processable: AlignTarget {
    fn process_and_write(
        self,
        case_name: &str,
        output_dir: &str,
        interpolation_steps: usize,
        watertight: bool,
        contour_types: &[ContourType],
    ) -> anyhow::Result<Self>;
}

impl Processable for GeometryPair {
    fn process_and_write(
        self,
        case_name: &str,
        output_dir: &str,
        interpolation_steps: usize,
        watertight: bool,
        contour_types: &[ContourType],
    ) -> anyhow::Result<Self> {
        process_case(
            case_name,
            self,
            output_dir,
            interpolation_steps,
            watertight,
            contour_types,
        )
    }
}

impl Processable for Geometry {
    fn process_and_write(
        self,
        case_name: &str,
        output_dir: &str,
        _interpolation_steps: usize,
        watertight: bool,
        contour_types: &[ContourType],
    ) -> anyhow::Result<Self> {
        write_single_geometry(case_name, self, output_dir, watertight, contour_types)
    }
}

pub fn align_three_point_rs<T: Processable>(
    centerline: Centerline,
    mut target: T,
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
) -> anyhow::Result<(T, Centerline)> {
    let resampled_centerline = preprocess_centerline(centerline, target.primary_geometry())
        .map_err(|e| anyhow!("Couldn't resample the centerline: {}", e))?;

    let ref_idx = target
        .primary_geometry()
        .find_ref_frame_idx()
        .map_err(|e| anyhow!("Couldn't find ref frame idx: {:?}", e))?;
    let ref_point = target.primary_geometry().frames[ref_idx]
        .reference_point
        .as_ref()
        .ok_or_else(|| anyhow!("missing reference point"))?;
    let cl_ref_idx = resampled_centerline.find_reference_cl_point_idx(&aortic_ref_pt);

    let best_rot = best_rotation_three_point(
        &target.primary_geometry().frames[ref_idx].lumen,
        ref_point,
        aortic_ref_pt,
        upper_ref_pt,
        lower_ref_pt,
        angle_step,
        &resampled_centerline.points[cl_ref_idx],
    );

    target = rotate_by_best_rotation(target, best_rot);
    target = apply_transformations(target, &resampled_centerline, &aortic_ref_pt);

    target = if write {
        target
            .process_and_write(
                case_name,
                output_dir,
                interpolation_steps,
                watertight,
                &contour_types,
            )
            .map_err(|e| anyhow!("Failed to write obj: {}", e))?
    } else {
        target
    };

    Ok((target, resampled_centerline))
}

pub fn align_manual_rs<T: Processable>(
    centerline: Centerline,
    mut target: T,
    rotation_angle_deg: f64,
    ref_pt: (f64, f64, f64),
    write: bool,
    watertight: bool,
    interpolation_steps: usize,
    output_dir: &str,
    contour_types: Vec<ContourType>,
    case_name: &str,
) -> anyhow::Result<(T, Centerline)> {
    let resampled_centerline = preprocess_centerline(centerline, target.primary_geometry())
        .map_err(|e| anyhow!("Couldn't resample the centerline: {}", e))?;

    target = rotate_by_best_rotation(target, rotation_angle_deg.to_radians());
    target = apply_transformations(target, &resampled_centerline, &ref_pt);

    target = if write {
        target
            .process_and_write(
                case_name,
                output_dir,
                interpolation_steps,
                watertight,
                &contour_types,
            )
            .map_err(|e| anyhow!("Failed to write obj: {}", e))?
    } else {
        target
    };

    Ok((target, resampled_centerline))
}

/// Combined alignment with three-point initialization and Hausdorff refinement
pub fn align_combined_rs<T: Processable>(
    centerline: Centerline,
    target: T,
    aortic_ref_pt: (f64, f64, f64),
    upper_ref_pt: (f64, f64, f64),
    lower_ref_pt: (f64, f64, f64),
    points: &[(f64, f64, f64)],
    angle_step: f64,
    refine_angle_range: f64,
    refine_index_range: usize,
    write: bool,
    watertight: bool,
    interpolation_steps: usize,
    output_dir: &str,
    contour_types: Vec<ContourType>,
    case_name: &str,
) -> anyhow::Result<(T, Centerline)> {
    let original = target.clone();

    println!("\nStep 1: Finding initial rotation via three-point method");

    let resampled_centerline =
        preprocess_centerline(centerline.clone(), original.primary_geometry())
            .map_err(|e| anyhow!("Couldn't resample the centerline: {}", e))?;

    let ref_idx = original
        .primary_geometry()
        .find_ref_frame_idx()
        .map_err(|e| anyhow!("Couldn't find ref frame idx: {:?}", e))?;

    let ref_point = original.primary_geometry().frames[ref_idx]
        .reference_point
        .as_ref()
        .ok_or_else(|| anyhow!("missing reference point"))?;

    let initial_cl_ref_idx = resampled_centerline.find_reference_cl_point_idx(&aortic_ref_pt);

    let initial_rotation = best_rotation_three_point(
        &original.primary_geometry().frames[ref_idx].lumen,
        ref_point,
        aortic_ref_pt,
        upper_ref_pt,
        lower_ref_pt,
        angle_step,
        &resampled_centerline.points[initial_cl_ref_idx],
    );

    let aligned = apply_transformations(
        rotate_by_best_rotation(original, initial_rotation),
        &resampled_centerline,
        &aortic_ref_pt,
    );

    let mutated_points = transfrom_tuples_to_contourpoints(points);

    println!("Step 2: Refining with Hausdorff distance");
    let (refined_rotation_delta, refined_cl_ref_idx) = refine_alignment_hausdorff(
        &aligned,
        &resampled_centerline,
        initial_cl_ref_idx,
        0.0,
        &mutated_points,
        refine_angle_range,
        angle_step,
        refine_index_range,
    );

    let total_rotation = initial_rotation + refined_rotation_delta;
    println!("---------------------Applying final transformation---------------------");
    println!(
        "Total rotation (initial + delta): {:.2}°",
        total_rotation.to_degrees()
    );
    let diff = initial_cl_ref_idx as i32 - refined_cl_ref_idx as i32;
    println!("Moving ostium by {} centerline points", diff);

    let refined_ref_pt = (
        resampled_centerline.points[refined_cl_ref_idx]
            .contour_point
            .x,
        resampled_centerline.points[refined_cl_ref_idx]
            .contour_point
            .y,
        resampled_centerline.points[refined_cl_ref_idx]
            .contour_point
            .z,
    );

    let mut final_target = apply_transformations(
        rotate_by_best_rotation(target.clone(), total_rotation),
        &resampled_centerline,
        &refined_ref_pt,
    );

    final_target = if write {
        final_target
            .process_and_write(
                case_name,
                output_dir,
                interpolation_steps,
                watertight,
                &contour_types,
            )
            .map_err(|e| anyhow!("Failed to write obj: {}", e))?
    } else {
        final_target
    };

    Ok((final_target, resampled_centerline))
}

// /// Simpler combined alignment that doesn't double-rotate
// pub fn align_combined_simple_rs(
//     centerline: Centerline,
//     geom_pair: GeometryPair,
//     aortic_ref_pt: (f64, f64, f64),
//     upper_ref_pt: (f64, f64, f64),
//     lower_ref_pt: (f64, f64, f64),
//     points: &[(f64, f64, f64)],
//     angle_step: f64,
//     refine_angle_range: f64,
//     refine_index_range: usize,
//     write: bool,
//     watertight: bool,
//     interpolation_steps: usize,
//     output_dir: &str,
//     contour_types: Vec<ContourType>,
//     case_name: &str,
// ) -> anyhow::Result<(GeometryPair, Centerline)> {
//     // Get the initial rotation from three-point alignment
//     let resampled_centerline = preprocess_centerline(centerline, &geom_pair.geom_a)
//         .map_err(|e| anyhow!("Couldn't resample the centerline: {}", e))?;
//
//     let ref_idx = geom_pair
//         .geom_a
//         .find_ref_frame_idx()
//         .map_err(|e| anyhow!("Couldn't find ref frame idx: {:?}", e))?;
//
//     let ref_point = geom_pair.geom_a.frames[ref_idx]
//         .reference_point
//         .as_ref()
//         .ok_or_else(|| anyhow!("missing reference point"))?;
//
//     let initial_cl_ref_idx = resampled_centerline.find_reference_cl_point_idx(&aortic_ref_pt);
//
//     let initial_rotation = best_rotation_three_point(
//         &geom_pair.geom_a.frames[ref_idx].lumen,
//         ref_point,
//         aortic_ref_pt,
//         upper_ref_pt,
//         lower_ref_pt,
//         angle_step,
//         &resampled_centerline.points[initial_cl_ref_idx],
//     );
//
//     // Apply the initial rotation
//     let mut aligned_geom_pair = rotate_by_best_rotation(geom_pair, initial_rotation);
//     aligned_geom_pair = apply_transformations(aligned_geom_pair, &resampled_centerline, &aortic_ref_pt);
//
//     // Refine with Hausdorff
//     let mutated_points = transfrom_tuples_to_contourpoints(points);
//
//     let (rotation_delta, refined_cl_ref_idx) = refine_alignment_hausdorff(
//         &aligned_geom_pair,
//         &resampled_centerline,
//         initial_cl_ref_idx,
//         0.0, // Start from current rotation
//         &mutated_points,
//         refine_angle_range,
//         angle_step / 2.0, // Use finer step for refinement
//         refine_index_range,
//     );
//
//     // Apply the delta rotation
//     if rotation_delta.abs() > 1e-6 {
//         aligned_geom_pair = rotate_by_best_rotation(aligned_geom_pair, rotation_delta);
//
//         // Re-apply transformations with refined centerline point
//         let refined_ref_pt = (
//             resampled_centerline.points[refined_cl_ref_idx].contour_point.x,
//             resampled_centerline.points[refined_cl_ref_idx].contour_point.y,
//             resampled_centerline.points[refined_cl_ref_idx].contour_point.z,
//         );
//
//         aligned_geom_pair = apply_transformations(aligned_geom_pair, &resampled_centerline, &refined_ref_pt);
//     }
//
//     // Write if requested
//     let final_geom_pair = if write {
//         process_case(
//             case_name,
//             aligned_geom_pair,
//             output_dir,
//             interpolation_steps,
//             watertight,
//             &contour_types,
//         )
//         .map_err(|e| anyhow!("Failed to write obj: {}", e))?
//     } else {
//         aligned_geom_pair
//     };
//
//     Ok((final_geom_pair, resampled_centerline))
// }

fn transfrom_tuples_to_contourpoints(points: &[(f64, f64, f64)]) -> Vec<ContourPoint> {
    let mut contour_points = Vec::with_capacity(points.len());

    for (i, point) in points.iter().enumerate() {
        let contour_point = ContourPoint {
            frame_index: 999,
            point_index: i as u32,
            x: point.0,
            y: point.1,
            z: point.2,
            aortic: false,
        };
        contour_points.push(contour_point);
    }
    contour_points
}
