pub mod align_algorithms;
pub mod preprocessing;

use crate::intravascular::io::input::ContourPoint;
use crate::intravascular::io::{geometry::ContourType, input::Centerline};
use crate::intravascular::processing::align_between::GeometryPair;
use anyhow::anyhow;

use crate::intravascular::to_object::process_case;
use align_algorithms::{
    apply_transformations, best_rotation_three_point, refine_alignment_hausdorff,
    rotate_by_best_rotation,
};
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
    rotation_angle_deg: f64,
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

    geom_pair = rotate_by_best_rotation(geom_pair, rotation_angle_deg.to_radians());

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

/// Combined alignment with three-point initialization and Hausdorff refinement
pub fn align_combined_rs(
    centerline: Centerline,
    geom_pair: GeometryPair,
    aortic_ref_pt: (f64, f64, f64),
    upper_ref_pt: (f64, f64, f64),
    lower_ref_pt: (f64, f64, f64),
    points: &[(f64, f64, f64)],
    angle_step: f64,
    refine_angle_range: f64,   // e.g., 15° in radians
    refine_index_range: usize, // e.g., 2
    write: bool,
    watertight: bool,
    interpolation_steps: usize,
    output_dir: &str,
    contour_types: Vec<ContourType>,
    case_name: &str,
) -> anyhow::Result<(GeometryPair, Centerline)> {
    // Clone the original geometry pair for the three-point alignment
    let original_geom_pair = geom_pair.clone();

    // Step 1: Get the initial rotation from three-point alignment
    println!("Step 1: Finding initial rotation via three-point method");

    let resampled_centerline =
        preprocess_centerline(centerline.clone(), &original_geom_pair.geom_a)
            .map_err(|e| anyhow!("Couldn't resample the centerline: {}", e))?;

    let ref_idx = original_geom_pair
        .geom_a
        .find_ref_frame_idx()
        .map_err(|e| anyhow!("Couldn't find ref frame idx: {:?}", e))?;

    let ref_point = original_geom_pair.geom_a.frames[ref_idx]
        .reference_point
        .as_ref()
        .ok_or_else(|| anyhow!("missing reference point"))?;

    let initial_cl_ref_idx = resampled_centerline.find_reference_cl_point_idx(&aortic_ref_pt);

    let initial_rotation = best_rotation_three_point(
        &original_geom_pair.geom_a.frames[ref_idx].lumen,
        ref_point,
        aortic_ref_pt,
        upper_ref_pt,
        lower_ref_pt,
        angle_step,
        &resampled_centerline.points[initial_cl_ref_idx],
    );

    println!(
        "Initial rotation from three-point: {:.4} rad",
        initial_rotation
    );

    // Step 2: Apply the three-point rotation
    println!("Step 2: Applying three-point rotation");
    let mut aligned_geom_pair = rotate_by_best_rotation(original_geom_pair, initial_rotation);
    aligned_geom_pair =
        apply_transformations(aligned_geom_pair, &resampled_centerline, &aortic_ref_pt);

    // Step 3: Convert points to contour points for Hausdorff comparison
    let mutated_points = transfrom_tuples_to_contourpoints(points);

    // Step 4: Refine alignment using Hausdorff distance in limited search space
    println!("Step 3: Refining with Hausdorff distance");
    let (refined_rotation_delta, refined_cl_ref_idx) = refine_alignment_hausdorff(
        &aligned_geom_pair,
        &resampled_centerline,
        initial_cl_ref_idx,
        0.0, // Start from 0 because we're already at the initial_rotation
        &mutated_points,
        refine_angle_range, // Search ± this range
        angle_step,
        refine_index_range,
    );

    let total_rotation = initial_rotation + refined_rotation_delta;
    println!(
        "Total rotation (initial + delta): {:.4} rad",
        total_rotation
    );

    // Step 5: Create final geometry with combined rotation
    println!("Step 4: Applying refined transformation");

    // Create new geometry pair with the total rotation
    let mut final_geom_pair = rotate_by_best_rotation(geom_pair.clone(), total_rotation);

    // Apply transformations using refined centerline reference index
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

    final_geom_pair =
        apply_transformations(final_geom_pair, &resampled_centerline, &refined_ref_pt);

    // Step 6: Write output if requested
    let final_geom_pair = if write {
        process_case(
            case_name,
            final_geom_pair,
            output_dir,
            interpolation_steps,
            watertight,
            &contour_types,
        )
        .map_err(|e| anyhow!("Failed to write obj: {}", e))?
    } else {
        final_geom_pair
    };

    Ok((final_geom_pair, resampled_centerline))
}

/// Simpler combined alignment that doesn't double-rotate
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

//     let ref_idx = geom_pair
//         .geom_a
//         .find_ref_frame_idx()
//         .map_err(|e| anyhow!("Couldn't find ref frame idx: {:?}", e))?;

//     let ref_point = geom_pair.geom_a.frames[ref_idx]
//         .reference_point
//         .as_ref()
//         .ok_or_else(|| anyhow!("missing reference point"))?;

//     let initial_cl_ref_idx = resampled_centerline.find_reference_cl_point_idx(&aortic_ref_pt);

//     let initial_rotation = best_rotation_three_point(
//         &geom_pair.geom_a.frames[ref_idx].lumen,
//         ref_point,
//         aortic_ref_pt,
//         upper_ref_pt,
//         lower_ref_pt,
//         angle_step,
//         &resampled_centerline.points[initial_cl_ref_idx],
//     );

//     // Apply the initial rotation
//     let mut aligned_geom_pair = rotate_by_best_rotation(geom_pair, initial_rotation);
//     aligned_geom_pair = apply_transformations(aligned_geom_pair, &resampled_centerline, &aortic_ref_pt);

//     // Refine with Hausdorff
//     let mutated_points = transfrom_tuples_to_contourpoints(points);

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

//     // Apply the delta rotation
//     if rotation_delta.abs() > 1e-6 {
//         aligned_geom_pair = rotate_by_best_rotation(aligned_geom_pair, rotation_delta);

//         // Re-apply transformations with refined centerline point
//         let refined_ref_pt = (
//             resampled_centerline.points[refined_cl_ref_idx].contour_point.x,
//             resampled_centerline.points[refined_cl_ref_idx].contour_point.y,
//             resampled_centerline.points[refined_cl_ref_idx].contour_point.z,
//         );

//         aligned_geom_pair = apply_transformations(aligned_geom_pair, &resampled_centerline, &refined_ref_pt);
//     }

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
