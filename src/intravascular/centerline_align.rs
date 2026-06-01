pub mod align_algorithms;
pub mod preprocessing;

use crate::intravascular::io::input::ContourPoint;
use crate::intravascular::io::{
    geometry::{Contour, ContourType, Frame, Geometry},
    input::Centerline,
};
use crate::intravascular::processing::align_between::GeometryPair;
use anyhow::anyhow;
use nalgebra::{Point3, Rotation3, Unit, Vector3};

use crate::intravascular::to_object;
use align_algorithms::{
    apply_transformations, best_rotation_three_point, refine_alignment_hausdorff,
    rotate_by_best_rotation, AlignTarget,
};

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
        to_object::process_case(
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
        to_object::write_single_geometry(case_name, self, output_dir, watertight, contour_types)
    }
}

pub fn align_three_point_rs<T: Processable>(
    centerline: Centerline,
    mut target: T,
    main_ref_pt: (f64, f64, f64),
    counterclockwise_ref_pt: (f64, f64, f64),
    clockwise_ref_pt: (f64, f64, f64),
    angle_step: f64,
    write: bool,
    watertight: bool,
    interpolation_steps: usize,
    output_dir: &str,
    contour_types: Vec<ContourType>,
    case_name: &str,
    align_wall_anomalous: bool,
) -> anyhow::Result<(T, Centerline)> {
    let resampled_centerline =
        preprocessing::preprocess_centerline(centerline, target.primary_geometry())
            .map_err(|e| anyhow!("Couldn't resample the centerline: {e}"))?;

    let ref_idx = target
        .primary_geometry()
        .find_ref_frame_idx()
        .map_err(|e| anyhow!("Couldn't find ref frame idx: {e:?}"))?;
    let ref_point = target.primary_geometry().frames[ref_idx]
        .reference_point
        .as_ref()
        .ok_or_else(|| anyhow!("missing reference point"))?;
    let cl_ref_idx = resampled_centerline.find_reference_cl_point_idx(&main_ref_pt);

    let best_rot = best_rotation_three_point(
        &target.primary_geometry().frames[ref_idx].lumen,
        ref_point,
        main_ref_pt,
        counterclockwise_ref_pt,
        clockwise_ref_pt,
        angle_step,
        &resampled_centerline.points[cl_ref_idx],
    );

    target = rotate_by_best_rotation(target, best_rot);
    target = apply_transformations(target, &resampled_centerline, &main_ref_pt);

    if align_wall_anomalous {
        target = align_walls(target, true);
    }

    target = if write {
        target
            .process_and_write(
                case_name,
                output_dir,
                interpolation_steps,
                watertight,
                &contour_types,
            )
            .map_err(|e| anyhow!("Failed to write obj: {e}"))?
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
    align_wall_anomalous: bool,
) -> anyhow::Result<(T, Centerline)> {
    let resampled_centerline =
        preprocessing::preprocess_centerline(centerline, target.primary_geometry())
            .map_err(|e| anyhow!("Couldn't resample the centerline: {e}"))?;

    target = rotate_by_best_rotation(target, rotation_angle_deg.to_radians());
    target = apply_transformations(target, &resampled_centerline, &ref_pt);

    if align_wall_anomalous {
        target = align_walls(target, true);
    }

    target = if write {
        target
            .process_and_write(
                case_name,
                output_dir,
                interpolation_steps,
                watertight,
                &contour_types,
            )
            .map_err(|e| anyhow!("Failed to write obj: {e}"))?
    } else {
        target
    };

    Ok((target, resampled_centerline))
}

/// Combined alignment with three-point initialization and Hausdorff refinement
pub fn align_combined_rs<T: Processable>(
    centerline: Centerline,
    target: T,
    main_ref_pt: (f64, f64, f64),
    counterclockwise_ref_pt: (f64, f64, f64),
    clockwise_ref_pt: (f64, f64, f64),
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
    align_wall_anomalous: bool,
) -> anyhow::Result<(T, Centerline)> {
    let original = target.clone();

    println!("\nStep 1: Finding initial rotation via three-point method");

    let resampled_centerline =
        preprocessing::preprocess_centerline(centerline.clone(), original.primary_geometry())
            .map_err(|e| anyhow!("Couldn't resample the centerline: {e}"))?;

    let ref_idx = original
        .primary_geometry()
        .find_ref_frame_idx()
        .map_err(|e| anyhow!("Couldn't find ref frame idx: {e:?}"))?;

    let ref_point = original.primary_geometry().frames[ref_idx]
        .reference_point
        .as_ref()
        .ok_or_else(|| anyhow!("missing reference point"))?;

    let initial_cl_ref_idx = resampled_centerline.find_reference_cl_point_idx(&main_ref_pt);

    let initial_rotation = best_rotation_three_point(
        &original.primary_geometry().frames[ref_idx].lumen,
        ref_point,
        main_ref_pt,
        counterclockwise_ref_pt,
        clockwise_ref_pt,
        angle_step,
        &resampled_centerline.points[initial_cl_ref_idx],
    );

    let aligned = apply_transformations(
        rotate_by_best_rotation(original, initial_rotation),
        &resampled_centerline,
        &main_ref_pt,
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
    println!("Moving ostium by {diff} centerline points");

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

    if align_wall_anomalous {
        final_target = align_walls(final_target, true);
    }

    final_target = if write {
        final_target
            .process_and_write(
                case_name,
                output_dir,
                interpolation_steps,
                watertight,
                &contour_types,
            )
            .map_err(|e| anyhow!("Failed to write obj: {e}"))?
    } else {
        final_target
    };

    Ok((final_target, resampled_centerline))
}

// /// Simpler combined alignment that doesn't double-rotate
// pub fn align_combined_simple_rs(
//     centerline: Centerline,
//     geom_pair: GeometryPair,
//     main_ref_pt: (f64, f64, f64),
//     counterclockwise_ref_pt: (f64, f64, f64),
//     clockwise_ref_pt: (f64, f64, f64),
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
//     let resampled_centerline = preprocessing::preprocess_centerline(centerline, &geom_pair.geom_a)
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
//     let initial_cl_ref_idx = resampled_centerline.find_reference_cl_point_idx(&main_ref_pt);
//
//     let initial_rotation = best_rotation_three_point(
//         &geom_pair.geom_a.frames[ref_idx].lumen,
//         ref_point,
//         main_ref_pt,
//         counterclockwise_ref_pt,
//         clockwise_ref_pt,
//         angle_step,
//         &resampled_centerline.points[initial_cl_ref_idx],
//     );
//
//     // Apply the initial rotation
//     let mut aligned_geom_pair = rotate_by_best_rotation(geom_pair, initial_rotation);
//     aligned_geom_pair = apply_transformations(aligned_geom_pair, &resampled_centerline, &main_ref_pt);
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
//         to_object::process_case(
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

/// Direction from `frame_centroid` to the centroid of aortic-flagged wall points.
///
/// Unambiguous (always points toward the aortic side). Returns `None` when no
/// points carry the `aortic` flag.
fn aortic_centroid_direction(
    wall: &Contour,
    frame_centroid: (f64, f64, f64),
) -> Option<Vector3<f64>> {
    let pts: Vec<_> = wall.points.iter().filter(|p| p.aortic).collect();
    if pts.is_empty() {
        return None;
    }
    let n = pts.len() as f64;
    let cx = pts.iter().map(|p| p.x).sum::<f64>() / n;
    let cy = pts.iter().map(|p| p.y).sum::<f64>() / n;
    let cz = pts.iter().map(|p| p.z).sum::<f64>() / n;
    let dir = Vector3::new(
        cx - frame_centroid.0,
        cy - frame_centroid.1,
        cz - frame_centroid.2,
    );
    if dir.norm() < 1e-9 {
        None
    } else {
        Some(dir)
    }
}

/// Direction of a wall contour's major axis (farthest-point pair). Ambiguous in sign.
fn wall_major_axis(wall: &Contour) -> Option<Vector3<f64>> {
    let pts = &wall.points;
    if pts.len() < 2 {
        return None;
    }
    let mut max_dist_sq = 0.0_f64;
    let mut fa = &pts[0];
    let mut fb = &pts[0];
    for i in 0..pts.len() {
        for j in i + 1..pts.len() {
            let dx = pts[i].x - pts[j].x;
            let dy = pts[i].y - pts[j].y;
            let dz = pts[i].z - pts[j].z;
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 > max_dist_sq {
                max_dist_sq = d2;
                fa = &pts[i];
                fb = &pts[j];
            }
        }
    }
    let dir = Vector3::new(fb.x - fa.x, fb.y - fa.y, fb.z - fa.z);
    if dir.norm() < 1e-9 {
        None
    } else {
        Some(dir)
    }
}

/// Lumen plane normal for `frame` via Newell's method.
fn lumen_normal(frame: &Frame) -> Vector3<f64> {
    let c = frame.centroid;
    let pts = &frame.lumen.points;
    if pts.len() < 3 {
        return Vector3::new(0.0, 0.0, 1.0);
    }
    let n = pts.len();
    let mut normal = Vector3::<f64>::zeros();
    for i in 0..n {
        let curr = &pts[i];
        let next = &pts[(i + 1) % n];
        normal.x += (curr.y - c.1) * (next.z - c.2) - (curr.z - c.2) * (next.y - c.1);
        normal.y += (curr.z - c.2) * (next.x - c.0) - (curr.x - c.0) * (next.z - c.2);
        normal.z += (curr.x - c.0) * (next.y - c.1) - (curr.y - c.1) * (next.x - c.0);
    }
    let norm = normal.norm();
    if norm > 1e-12 {
        normal / norm
    } else {
        Vector3::new(0.0, 0.0, 1.0)
    }
}

/// Projects `v` onto the plane perpendicular to `tangent` and returns the normalised result.
/// Returns `None` when the projection is degenerate (v nearly parallel to tangent).
fn project_onto_plane_normalized(v: Vector3<f64>, tangent: Vector3<f64>) -> Option<Vector3<f64>> {
    let proj = v - tangent * v.dot(&tangent);
    if proj.norm() < 1e-9 {
        None
    } else {
        Some(proj.normalize())
    }
}

/// Parallel-transports `v` from the frame with tangent `t_from` into the frame with tangent
/// `t_to` using the minimum (geodesic) rotation.
fn parallel_transport(v: Vector3<f64>, t_from: Vector3<f64>, t_to: Vector3<f64>) -> Vector3<f64> {
    let angle = t_from.angle(&t_to);
    if angle < 1e-9 {
        return v;
    }
    let axis = t_from.cross(&t_to);
    if axis.norm() < 1e-9 {
        // Anti-parallel tangents: rotate 180° around any axis perpendicular to t_from.
        let perp = if t_from.x.abs() < 0.9 {
            (Vector3::new(1.0, 0.0, 0.0) - t_from * t_from.x).normalize()
        } else {
            (Vector3::new(0.0, 1.0, 0.0) - t_from * t_from.y).normalize()
        };
        return Rotation3::from_axis_angle(&Unit::new_normalize(perp), std::f64::consts::PI) * v;
    }
    Rotation3::from_axis_angle(&Unit::new_normalize(axis), angle) * v
}

/// Signed angle (in radians) to rotate `from` towards `to` around `axis` (right-hand rule).
fn signed_angle_around_axis(from: Vector3<f64>, to: Vector3<f64>, axis: Vector3<f64>) -> f64 {
    from.cross(&to).dot(&axis).atan2(from.dot(&to))
}

/// Rotates only the `Wall` extra contour in every frame so its major axis tracks the
/// parallel-transported reference direction established at frame 0.
///
/// When aortic-flagged points are present their centroid provides an unambiguous
/// direction (no 180° flip risk). Otherwise the major axis is used with a
/// min-magnitude disambiguation fallback.
///
/// All other contours (lumen, catheter, …) are left untouched.
fn align_walls_on_geometry(geom: &mut Geometry) {
    let frame0 = &geom.frames[0];
    let t0 = lumen_normal(frame0);

    // Initialize the reference direction from frame 0's wall.
    let wall0 = match frame0.extras.get(&ContourType::Wall) {
        Some(w) => w.clone(),
        None => return,
    };
    let dir0 =
        aortic_centroid_direction(&wall0, frame0.centroid).or_else(|| wall_major_axis(&wall0));
    let mut u = match dir0.and_then(|d| project_onto_plane_normalized(d, t0)) {
        Some(v) => v,
        None => return,
    };

    for i in 1..geom.frames.len() {
        let t_prev = lumen_normal(&geom.frames[i - 1]);
        let t_curr = lumen_normal(&geom.frames[i]);

        // Parallel-transport the reference direction into the current tangent plane.
        u = parallel_transport(u, t_prev, t_curr);
        u = match project_onto_plane_normalized(u, t_curr) {
            Some(v) => v,
            None => continue,
        };

        let center = geom.frames[i].centroid;

        // Prefer unambiguous aortic centroid; fall back to major axis.
        let (wall_dir, has_aortic) = match geom.frames[i].extras.get(&ContourType::Wall) {
            Some(w) => match aortic_centroid_direction(w, center) {
                Some(d) => (d, true),
                None => match wall_major_axis(w) {
                    Some(d) => (d, false),
                    None => continue,
                },
            },
            None => continue,
        };

        let v = match project_onto_plane_normalized(wall_dir, t_curr) {
            Some(v) => v,
            None => continue,
        };

        let angle = if has_aortic {
            // Aortic centroid is unambiguous — compute angle directly.
            signed_angle_around_axis(v, u, t_curr)
        } else {
            // Major axis is ambiguous in sign — pick the rotation with smaller magnitude.
            let a1 = signed_angle_around_axis(v, u, t_curr);
            let a2 = signed_angle_around_axis(-v, u, t_curr);
            if a1.abs() <= a2.abs() {
                a1
            } else {
                a2
            }
        };

        if angle.abs() < 1e-6 {
            continue;
        }

        let rotation = Rotation3::from_axis_angle(&Unit::new_normalize(t_curr), angle);
        let pivot = Point3::new(center.0, center.1, center.2);

        if let Some(wall) = geom.frames[i].extras.get_mut(&ContourType::Wall) {
            for pt in &mut wall.points {
                let p = Point3::new(pt.x, pt.y, pt.z);
                let rotated = pivot + rotation * (p - pivot);
                pt.x = rotated.x;
                pt.y = rotated.y;
                pt.z = rotated.z;
            }
        }
    }
}

/// Compensates each frame's `Wall` contour for centerline twist via parallel transport.
/// Lumen and all other extras are left unchanged.
/// Does nothing when `anomalous` is `false` or fewer than 2 frames exist.
pub fn align_walls<T: AlignTarget>(mut target: T, anomalous: bool) -> T {
    if !anomalous || target.primary_geometry().frames.len() < 2 {
        return target;
    }
    target.for_each_geometry_mut(align_walls_on_geometry);
    target
}

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
