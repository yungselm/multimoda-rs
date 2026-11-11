use crate::types::native::contour::{Contour, ContourType};
use crate::types::native::frame::Frame;
use crate::types::native::geometry::Geometry;
use crate::types::native::GeometryPair;
use crate::types::native::{Centerline, ContourPoint};
use anyhow::anyhow;
use nalgebra::{Point3, Rotation3, Unit, Vector3};

use super::align_algorithms::{
    apply_transformations, best_rotation_three_point, refine_alignment_nn_distance,
    rotate_by_best_rotation, AlignTarget,
};
use super::preprocessing::{prepare_dense_centerline, resample_anchored_at};
use crate::intravascular::to_object;

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

/// Places the reference frame on the centerline point nearest `main_ref_pt` and rotates the
/// frames so the reference and the lumen points at index 0 and n/2 come closest to the three
/// reference points.
///
/// With `reference_frame` the primary geometry is re-anchored on that frame's reference first
/// ([`AlignTarget::set_reference_frame`]), so that frame lands on the centerline and its
/// reference drives the rotation.
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
    reference_frame: Option<u32>,
) -> anyhow::Result<(T, f64, f64)> {
    if let Some(frame) = reference_frame {
        target
            .set_reference_frame(frame)
            .map_err(|e| anyhow!("Couldn't re-anchor on frame {frame}: {e}"))?;
    }
    let (resampled_centerline, cl_ref_idx, spacing_mm) =
        super::preprocessing::preprocess_centerline(
            centerline,
            target.primary_geometry(),
            &main_ref_pt,
        )
        .map_err(|e| anyhow!("Couldn't resample the centerline: {e}"))?;

    let ref_idx = target
        .primary_geometry()
        .find_ref_frame_idx()
        .map_err(|e| anyhow!("Couldn't find ref frame idx: {e:?}"))?;
    let ref_point = target.primary_geometry().frames[ref_idx]
        .reference_point
        .as_ref()
        .ok_or_else(|| anyhow!("missing reference point"))?;

    let total_rotation = best_rotation_three_point(
        &target.primary_geometry().frames[ref_idx].lumen,
        ref_point,
        main_ref_pt,
        counterclockwise_ref_pt,
        clockwise_ref_pt,
        angle_step,
        &resampled_centerline.points[cl_ref_idx],
    );

    target = rotate_by_best_rotation(target, total_rotation);
    target = apply_transformations(target, &resampled_centerline, cl_ref_idx);

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

    Ok((target, spacing_mm, total_rotation))
}

/// Rotates the frames by `rotation_angle_deg` and places the reference frame on the centerline
/// point nearest `ref_pt`. With `reference_frame` the primary geometry is re-anchored first.
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
    reference_frame: Option<u32>,
) -> anyhow::Result<(T, f64, f64)> {
    if let Some(frame) = reference_frame {
        target
            .set_reference_frame(frame)
            .map_err(|e| anyhow!("Couldn't re-anchor on frame {frame}: {e}"))?;
    }
    let (resampled_centerline, cl_ref_idx, spacing_mm) =
        super::preprocessing::preprocess_centerline(centerline, target.primary_geometry(), &ref_pt)
            .map_err(|e| anyhow!("Couldn't resample the centerline: {e}"))?;

    let total_rotation = rotation_angle_deg.to_radians();
    target = rotate_by_best_rotation(target, total_rotation);
    target = apply_transformations(target, &resampled_centerline, cl_ref_idx);

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

    Ok((target, spacing_mm, total_rotation))
}

/// Combined alignment with three-point initialization and point-cloud refinement.
///
/// With `reference_frame` the primary geometry is re-anchored on that frame's reference first
/// ([`AlignTarget::set_reference_frame`]), as in [`align_three_point_rs`] and [`align_manual_rs`].
///
/// `refine_index_range` counts points of the input `centerline` (branch 0): the refinement
/// tries every input point within that many points of the initial reference as the ostium
/// position, with the frames resampled to the frame spacing around it.
pub fn align_combined_rs<T: Processable>(
    centerline: Centerline,
    mut target: T,
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
    reference_frame: Option<u32>,
) -> anyhow::Result<(T, f64, f64)> {
    if let Some(frame) = reference_frame {
        target
            .set_reference_frame(frame)
            .map_err(|e| anyhow!("Couldn't re-anchor on frame {frame}: {e}"))?;
    }
    let original = target.clone();

    println!("\nStep 1: Finding initial rotation via three-point method");

    // Indices into the dense (input-resolution) centerline; frames are placed on a copy
    // resampled to the frame spacing and anchored on the chosen dense point.
    let (dense_centerline, initial_cl_ref_idx, spacing_mm) =
        prepare_dense_centerline(centerline, original.primary_geometry(), &main_ref_pt)
            .map_err(|e| anyhow!("Couldn't resample the centerline: {e}"))?;
    let (resampled_centerline, initial_grid_ref_idx) =
        resample_anchored_at(&dense_centerline, spacing_mm, initial_cl_ref_idx);

    let ref_idx = original
        .primary_geometry()
        .find_ref_frame_idx()
        .map_err(|e| anyhow!("Couldn't find ref frame idx: {e:?}"))?;

    let ref_point = original.primary_geometry().frames[ref_idx]
        .reference_point
        .as_ref()
        .ok_or_else(|| anyhow!("missing reference point"))?;

    let initial_rotation = best_rotation_three_point(
        &original.primary_geometry().frames[ref_idx].lumen,
        ref_point,
        main_ref_pt,
        counterclockwise_ref_pt,
        clockwise_ref_pt,
        angle_step,
        &resampled_centerline.points[initial_grid_ref_idx],
    );

    // The refinement places frames itself, so hand it the rotated but unplaced frames:
    // rotations are about z through each frame centroid, which is only the in-plane
    // rotation the final placement applies while the frames are still unplaced.
    let rotated = rotate_by_best_rotation(original, initial_rotation);

    let mutated_points = transfrom_tuples_to_contourpoints(points);

    println!("Step 2: Refining with mean point-cloud distance");
    let (refined_rotation_delta, refined_cl_ref_idx) = refine_alignment_nn_distance(
        &rotated,
        &dense_centerline,
        spacing_mm,
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
    let diff = refined_cl_ref_idx as isize - initial_cl_ref_idx as isize;
    println!(
        "Moving ostium by {diff} centerline points (index {initial_cl_ref_idx} -> {refined_cl_ref_idx})"
    );

    // Same grid the winning candidate was scored on.
    let (refined_centerline, refined_grid_ref_idx) =
        resample_anchored_at(&dense_centerline, spacing_mm, refined_cl_ref_idx);
    let mut final_target = apply_transformations(
        rotate_by_best_rotation(target, total_rotation),
        &refined_centerline,
        refined_grid_ref_idx,
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

    Ok((final_target, spacing_mm, total_rotation))
}

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

#[cfg(test)]
mod reference_frame_tests {
    use super::*;
    use crate::types::native::{Annotation, AnnotationKind, AnnotationRole, CenterlinePoint};
    use crate::types::native::{Point3D, Transform};
    use std::collections::HashMap;
    use std::f64::consts::{FRAC_PI_2, PI, TAU};

    const N_POINTS: u32 = 8;

    fn reference(frame: u32, x: f64, y: f64, z: f64, auto: bool) -> Annotation {
        let point = ContourPoint {
            frame_index: frame,
            point_index: 0,
            x,
            y,
            z,
            aortic: false,
        };
        Annotation::new(
            AnnotationKind::Reference,
            AnnotationRole::Point,
            auto as u32,
            vec![point],
        )
    }

    /// Five unit circles 1 mm apart, original frames 10..=14. Anchor on frame 10 at lumen
    /// point 0, an automatic reference on frame 13 at lumen point 2 (90 degrees).
    fn circle_geometry() -> Geometry {
        let frames = (0..5u32)
            .map(|i| {
                let original = 10 + i;
                let z = i as f64;
                let points = (0..N_POINTS)
                    .map(|p| {
                        let theta = p as f64 * TAU / N_POINTS as f64;
                        ContourPoint {
                            frame_index: original,
                            point_index: p,
                            x: theta.cos(),
                            y: theta.sin(),
                            z,
                            aortic: false,
                        }
                    })
                    .collect();
                let mut lumen = Contour {
                    id: i,
                    original_frame: original,
                    points,
                    centroid: None,
                    aortic_thickness: None,
                    pulmonary_thickness: None,
                    kind: ContourType::Lumen,
                };
                lumen.compute_centroid();
                let mut annotations = Vec::new();
                if original == 10 {
                    annotations.push(reference(original, 1.0, 0.0, z, false));
                }
                if original == 13 {
                    annotations.push(reference(original, 0.0, 1.0, z, true));
                }
                Frame {
                    id: i,
                    centroid: lumen.centroid.unwrap(),
                    lumen,
                    extras: HashMap::new(),
                    reference_point: (original == 10).then_some(ContourPoint {
                        frame_index: original,
                        point_index: 0,
                        x: 1.0,
                        y: 0.0,
                        z,
                        aortic: false,
                    }),
                    annotations,
                }
            })
            .collect();
        Geometry {
            frames,
            label: "circles".to_string(),
        }
    }

    /// Straight centerline on the z axis from z = 30 down to 0, 0.1 mm apart.
    fn z_axis_centerline() -> Centerline {
        let points = (0..=300u32)
            .map(|k| CenterlinePoint {
                contour_point: ContourPoint {
                    frame_index: k,
                    point_index: k,
                    x: 0.0,
                    y: 0.0,
                    z: 30.0 - 0.1 * k as f64,
                    aortic: false,
                },
                tangent: Vector3::new(0.0, 0.0, 1.0),
                radius: 0.0,
                branch_id: 0,
            })
            .collect();
        Centerline {
            points,
            branch_start_indices: vec![0],
        }
    }

    fn frame_of(geometry: &Geometry, original: u32) -> &Frame {
        geometry
            .frames
            .iter()
            .find(|f| f.lumen.original_frame == original)
            .unwrap()
    }

    fn close(a: (f64, f64, f64), b: (f64, f64, f64)) -> bool {
        a.distance_to(&b) < 1e-6
    }

    #[test]
    fn test_set_reference_frame_moves_the_anchor_with_its_frame() -> anyhow::Result<()> {
        // transform first, the anchor takes the moved annotation coordinates
        let mut geometry = circle_geometry().rotate(FRAC_PI_2, (0.0, 0.0));
        geometry = geometry.translate(1.0, 2.0, 3.0);
        assert_eq!(geometry.anchor_frame(), Some(10));
        assert_eq!(
            geometry
                .references()
                .iter()
                .map(|(f, auto, _)| (*f, *auto))
                .collect::<Vec<_>>(),
            vec![(10, false), (13, true)]
        );

        geometry.set_reference_frame(13)?;
        assert_eq!(geometry.anchor_frame(), Some(13));
        let anchored: Vec<u32> = geometry
            .frames
            .iter()
            .filter(|f| f.reference_point.is_some())
            .map(|f| f.lumen.original_frame)
            .collect();
        assert_eq!(anchored, vec![13]);

        // (0, 1, 3) rotated by 90 degrees and translated
        let rp = frame_of(&geometry, 13).reference_point.unwrap();
        assert!(close((rp.x, rp.y, rp.z), (0.0, 2.0, 6.0)), "{rp:?}");
        assert_eq!(rp.frame_index, 13);
        // the lumen point it sits on, the one three-point alignment takes as reference
        assert_eq!(rp.point_index, 2);

        let err = geometry.set_reference_frame(12).unwrap_err().to_string();
        assert!(
            err.contains("frame 12 has no reference annotation"),
            "{err}"
        );
        let err = geometry.set_reference_frame(99).unwrap_err().to_string();
        assert!(err.contains("no frame with original frame 99"), "{err}");
        assert_eq!(
            geometry.anchor_frame(),
            Some(13),
            "a failed call keeps the anchor"
        );

        let mut pair = GeometryPair::new(circle_geometry(), circle_geometry())?;
        pair.set_reference_frame(13)?;
        assert_eq!(pair.geom_a.anchor_frame(), Some(13));
        assert_eq!(pair.geom_b.anchor_frame(), Some(10));
        Ok(())
    }

    #[test]
    fn test_align_manual_reference_frame_places_the_chosen_frame() -> anyhow::Result<()> {
        let align = |reference_frame: Option<u32>| {
            align_manual_rs(
                z_axis_centerline(),
                circle_geometry(),
                0.0,
                (0.0, 0.0, 15.0),
                false,
                true,
                0,
                "",
                vec![ContourType::Lumen],
                "",
                false,
                reference_frame,
            )
            .map(|(g, ..)| g)
        };
        let default = align(None)?;
        let chosen = align(Some(13))?;

        assert!(close(frame_of(&default, 10).centroid, (0.0, 0.0, 15.0)));
        assert!(close(frame_of(&chosen, 13).centroid, (0.0, 0.0, 15.0)));
        // frame 10 now sits three frames (3 mm) further along the centerline
        assert!(close(frame_of(&chosen, 10).centroid, (0.0, 0.0, 18.0)));
        assert_eq!(chosen.anchor_frame(), Some(13));
        assert!(align(Some(12)).is_err());
        Ok(())
    }

    #[test]
    fn test_align_three_point_rotates_with_the_chosen_reference() -> anyhow::Result<()> {
        // ccw and cw targets on the axis are equally far from every lumen point, so only the
        // main target decides the rotation: the reference must turn towards +x.
        let align = |reference_frame: Option<u32>| {
            align_three_point_rs(
                z_axis_centerline(),
                circle_geometry(),
                (1.0, 0.0, 15.0),
                (0.0, 0.0, 15.0),
                (0.0, 0.0, 15.0),
                1f64.to_radians(),
                false,
                true,
                0,
                "",
                vec![ContourType::Lumen],
                "",
                false,
                reference_frame,
            )
        };
        let step = 1f64.to_radians();

        let (default, _, rotation) = align(None)?;
        assert!(rotation.abs() < step, "{}", rotation.to_degrees());
        assert!(close(frame_of(&default, 10).centroid, (0.0, 0.0, 15.0)));

        // the reference on frame 13 sits at 90 degrees, a 270 degree turn brings it to +x
        let (chosen, _, rotation) = align(Some(13))?;
        assert!(
            (rotation - 1.5 * PI).abs() < step,
            "{}",
            rotation.to_degrees()
        );
        let frame = frame_of(&chosen, 13);
        assert!(close(frame.centroid, (0.0, 0.0, 15.0)));
        let rp = frame.reference_point.unwrap();
        assert!(rp.distance_to(&(1.0, 0.0, 15.0)) < 0.05, "{rp:?}");
        let annotated = frame.annotations[0].points[0];
        assert!(rp.distance_to(&annotated) < 1e-9);
        Ok(())
    }
}
