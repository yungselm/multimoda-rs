use crate::intravascular::io::input::ContourPoint;
use crate::intravascular::io::input::{Centerline, CenterlinePoint};
use crate::intravascular::processing::align_between::GeometryPair;
use crate::intravascular::processing::process_utils::hausdorff_distance;
use crate::types::native;
use crate::types::native::contour::Contour;
use crate::types::native::geometry::Geometry;
use nalgebra::{Point3, Rotation3, Unit, Vector3};

/// Allows alignment algorithms to operate on either a single [`Geometry`] or a [`GeometryPair`].
pub trait AlignTarget: Sized + Clone {
    /// Returns the primary geometry used as the reference for computing transformations.
    fn primary_geometry(&self) -> &Geometry;
    /// Applies a closure mutably to every geometry contained in this target.
    fn for_each_geometry_mut<F: FnMut(&mut Geometry)>(&mut self, f: F);
    /// Applies frame transformations to all contained geometries.
    fn apply_frame_transforms(self, transformations: &[FrameTransformation]) -> Self;
    /// Rotates all contained geometries by `angle` radians around each frame's normal.
    fn rotate_all(self, angle: f64) -> Self;
}

impl AlignTarget for Geometry {
    fn primary_geometry(&self) -> &Geometry {
        self
    }

    fn for_each_geometry_mut<F: FnMut(&mut Geometry)>(&mut self, mut f: F) {
        f(self);
    }

    fn apply_frame_transforms(mut self, transformations: &[FrameTransformation]) -> Self {
        apply_transforms_to_geometry(&mut self, transformations);
        self
    }

    fn rotate_all(mut self, angle: f64) -> Self {
        self.rotate_geometry(angle);
        self
    }
}

impl AlignTarget for GeometryPair {
    fn primary_geometry(&self) -> &Geometry {
        &self.geom_a
    }

    fn for_each_geometry_mut<F: FnMut(&mut Geometry)>(&mut self, mut f: F) {
        f(&mut self.geom_a);
        f(&mut self.geom_b);
    }

    fn apply_frame_transforms(mut self, transformations: &[FrameTransformation]) -> Self {
        apply_transforms_to_geometry(&mut self.geom_a, transformations);
        apply_transforms_to_geometry(&mut self.geom_b, transformations);
        self
    }

    fn rotate_all(mut self, angle: f64) -> Self {
        self.geom_a.rotate_geometry(angle);
        self.geom_b.rotate_geometry(angle);
        self
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FrameTransformation {
    pub frame_index: u32,
    pub translation: Vector3<f64>,
    pub rotation: Rotation3<f64>,
    pub pivot: Point3<f64>,
}

impl FrameTransformation {
    pub fn apply_to_point(&self, point: &ContourPoint) -> ContourPoint {
        let translated_x = point.x + self.translation.x;
        let translated_y = point.y + self.translation.y;
        let translated_z = point.z + self.translation.z;

        let current_point = Point3::new(translated_x, translated_y, translated_z);
        let relative_vector = current_point - self.pivot;
        let rotated_relative = self.rotation * relative_vector;
        let rotated_point = self.pivot + rotated_relative;

        // Preserve other fields from the original ContourPoint
        ContourPoint {
            frame_index: point.frame_index,
            point_index: point.point_index,
            x: rotated_point.x,
            y: rotated_point.y,
            z: rotated_point.z,
            aortic: point.aortic,
        }
    }
}

pub fn get_transformations(
    geometry: Geometry,
    centerline: &Centerline,
    ref_pt: &(f64, f64, f64),
) -> Vec<FrameTransformation> {
    let mut transformations = Vec::new();

    // Find the reference point in the centerline
    let ref_idx_cl = centerline.find_reference_cl_point_idx(ref_pt);

    // The geometry frames are ordered, and we assume they correspond to centerline points
    // starting from the reference point and moving in the same direction
    for (i, frame) in geometry.frames.into_iter().enumerate() {
        // Calculate which centerline point corresponds to this geometry frame
        // We start from the reference centerline point and move through the centerline
        // based on the geometry frame's position relative to the reference frame
        let cl_index = ref_idx_cl as isize + (i as isize);

        if cl_index >= 0 && cl_index < centerline.points.len() as isize {
            let cl_point = &centerline.points[cl_index as usize];
            let transformation = align_frame(&frame.lumen, cl_point);
            transformations.push(transformation);
        } else {
            eprintln!(
                "Centerline index {} out of bounds for geometry frame {}",
                cl_index, frame.id
            );
        }
    }
    transformations
}

fn align_frame(frame: &Contour, cl_point: &CenterlinePoint) -> FrameTransformation {
    // Get centroid or compute if None
    let centroid = frame.centroid.unwrap_or_else(|| {
        let x_avg = frame.points.iter().map(|p| p.x).sum::<f64>() / frame.points.len() as f64;
        let y_avg = frame.points.iter().map(|p| p.y).sum::<f64>() / frame.points.len() as f64;
        let z_avg = frame.points.iter().map(|p| p.z).sum::<f64>() / frame.points.len() as f64;
        (x_avg, y_avg, z_avg)
    });

    // === Translation Step ===
    let translation_vec = Vector3::new(
        cl_point.contour_point.x - centroid.0,
        cl_point.contour_point.y - centroid.1,
        cl_point.contour_point.z - centroid.2,
    );

    // === Rotation Step ===
    let current_normal = calculate_normal(&frame.points, &centroid);
    let desired_normal = cl_point.tangent;
    let angle = current_normal.angle(&desired_normal);
    let rotation: Rotation3<f64> = if angle.abs() < 1e-6 {
        Rotation3::identity()
    } else {
        let rotation_axis = current_normal.cross(&desired_normal);
        if rotation_axis.norm() < 1e-6 {
            Rotation3::identity()
        } else {
            let rotation_axis_unit = Unit::new_normalize(rotation_axis);
            Rotation3::from_axis_angle(&rotation_axis_unit, angle)
        }
    };

    // Define the pivot as the centerline point
    let pivot = Point3::new(
        cl_point.contour_point.x,
        cl_point.contour_point.y,
        cl_point.contour_point.z,
    );

    FrameTransformation {
        frame_index: frame.original_frame, // Keep original frame index for tracking
        translation: translation_vec,
        rotation,
        pivot,
    }
}

/// Applies transformation to a contour (mutable version)
pub fn apply_transformation_to_contour(
    contour: &mut Contour,
    transformation: &FrameTransformation,
) {
    for point in contour.points.iter_mut() {
        let transformed_point = transformation.apply_to_point(point);
        *point = transformed_point;
    }

    // Update centroid if it exists
    if let Some(centroid) = contour.centroid.as_mut() {
        let centroid_point = ContourPoint {
            frame_index: transformation.frame_index,
            point_index: 0,
            x: centroid.0,
            y: centroid.1,
            z: centroid.2,
            aortic: false,
        };
        let transformed_centroid = transformation.apply_to_point(&centroid_point);
        *centroid = (
            transformed_centroid.x,
            transformed_centroid.y,
            transformed_centroid.z,
        );
    }
}

/// Calculates the normal vector using a stable cross product method.
/// Calculates the normal vector using a more robust method
fn calculate_normal(points: &[ContourPoint], centroid: &(f64, f64, f64)) -> Vector3<f64> {
    if points.len() < 3 {
        return Vector3::new(0.0, 0.0, 1.0); // Default to Z-axis for degenerate cases
    }

    // Use a more stable method: Newell's method for polygon normal
    let mut normal = Vector3::zeros();

    for i in 0..points.len() {
        let current = &points[i];
        let next = &points[(i + 1) % points.len()];

        normal.x += (current.y - centroid.1) * (next.z - centroid.2)
            - (current.z - centroid.2) * (next.y - centroid.1);
        normal.y += (current.z - centroid.2) * (next.x - centroid.0)
            - (current.x - centroid.0) * (next.z - centroid.2);
        normal.z += (current.x - centroid.0) * (next.y - centroid.1)
            - (current.y - centroid.1) * (next.x - centroid.0);
    }

    // Normalize the result
    let norm = normal.norm();
    if norm > 1e-12 {
        normal /= norm;
    } else {
        normal = Vector3::new(0.0, 0.0, 1.0);
    }

    normal
}

/// Rotates a contour around its centroid (for use in best_rotation_three_point)
fn rotate_contour_around_centroid(contour: &mut Contour, angle: f64) {
    let centroid = contour.centroid.unwrap_or_else(|| {
        let x_avg = contour.points.iter().map(|p| p.x).sum::<f64>() / contour.points.len() as f64;
        let y_avg = contour.points.iter().map(|p| p.y).sum::<f64>() / contour.points.len() as f64;
        let z_avg = contour.points.iter().map(|p| p.z).sum::<f64>() / contour.points.len() as f64;
        (x_avg, y_avg, z_avg)
    });

    let rotation_axis = calculate_normal(&contour.points, &centroid);
    let rotation = Rotation3::from_axis_angle(&Unit::new_normalize(rotation_axis), angle);
    let pivot = Point3::new(centroid.0, centroid.1, centroid.2);

    for point in contour.points.iter_mut() {
        let current_point = Point3::new(point.x, point.y, point.z);
        let relative_vector = current_point - pivot;
        let rotated_relative = rotation * relative_vector;
        let rotated_point = pivot + rotated_relative;
        point.x = rotated_point.x;
        point.y = rotated_point.y;
        point.z = rotated_point.z;
    }
}

/// Finds the optimal rotation angle by minimizing the distance between the closest opposite point
/// and the reference coordinate.
pub fn best_rotation_three_point(
    contour: &Contour,
    reference_point: &ContourPoint,
    main_ref_pt: (f64, f64, f64),
    counterclockwise_ref_pt: (f64, f64, f64),
    clockwise_ref_pt: (f64, f64, f64),
    angle_step: f64,
    centerline_point: &CenterlinePoint,
) -> f64 {
    let index_reference = reference_point.point_index;

    let [target_main, target_ccw, target_cw]: [Point3<f64>; 3] =
        [main_ref_pt, counterclockwise_ref_pt, clockwise_ref_pt]
            .map(|(x, y, z)| Point3::new(x, y, z));

    let mut best_angle = 0.0;
    let mut min_total_error = f64::MAX;

    let mut angle = 0.0;
    println!(
        "---------------------Centerline alignment: Finding optimal rotation---------------------"
    );

    while angle < core::f64::consts::TAU {
        // approx 360°
        let mut temp_contour = contour.clone();

        // Rotate around centroid
        rotate_contour_around_centroid(&mut temp_contour, angle);

        // Apply centerline alignment transformation
        let transformation = align_frame(&temp_contour, centerline_point);
        apply_transformation_to_contour(&mut temp_contour, &transformation);

        let temp_points = &temp_contour.points;

        let n_points = temp_points.len() as u32;

        let p_main = temp_points
            .iter()
            .find(|p| p.point_index == index_reference)
            .unwrap();
        // point_index == 0 is the highest-Y point → counterclockwise side (viewed proximal → distal)
        let cont_p_ccw = temp_points.iter().find(|p| p.point_index == 0).unwrap();
        // point_index == n/2 is the diametrically opposite point → clockwise side
        let cont_p_cw = temp_points
            .iter()
            .find(|p| p.point_index == (n_points / 2))
            .unwrap();

        let d_main = nalgebra::distance(&Point3::new(p_main.x, p_main.y, p_main.z), &target_main);

        let d_ccw = nalgebra::distance(
            &Point3::new(cont_p_ccw.x, cont_p_ccw.y, cont_p_ccw.z),
            &target_ccw,
        );

        let d_cw = nalgebra::distance(
            &Point3::new(cont_p_cw.x, cont_p_cw.y, cont_p_cw.z),
            &target_cw,
        );

        // Calculate sum of squared errors
        let total_error = d_main.powi(2) + d_ccw.powi(2) + d_cw.powi(2);

        if total_error < min_total_error {
            min_total_error = total_error;
            best_angle = angle;
        }
        angle += angle_step;
    }
    println!("✅ Best angle found: {:.2}°", best_angle.to_degrees());
    best_angle
}

/// Refines alignment using Hausdorff distance within a limited search space
pub fn refine_alignment_hausdorff<T: AlignTarget>(
    target: &T,
    centerline: &Centerline,
    initial_cl_ref_idx: usize,
    initial_rotation: f64,
    mutated_points: &[ContourPoint],
    angle_search_range: f64,
    angle_step: f64,
    index_search_range: usize,
) -> (f64, usize) {
    let len_frames = target.primary_geometry().frames.len();

    let mut best_angle = initial_rotation;
    let mut best_cl_ref_idx = initial_cl_ref_idx;
    let mut min_hausdorff = f64::MAX;

    println!("---------------------Refining alignment with Hausdorff---------------------");
    println!(
        "Initial rotation: {:.2}°, Initial CL index: {}",
        initial_rotation.to_degrees(),
        initial_cl_ref_idx
    );

    // When index_search_range == 0 only test the initial index; no loop over candidates needed.
    let delta_range = if index_search_range == 0 {
        0isize..=0isize
    } else {
        -(index_search_range as isize)..=(index_search_range as isize)
    };

    for delta_idx in delta_range {
        let signed = initial_cl_ref_idx as isize + delta_idx;
        if signed < 0 {
            continue;
        }
        let current_cl_ref_idx = signed as usize;

        if current_cl_ref_idx + len_frames >= centerline.points.len() {
            continue;
        }

        let cl_end_idx = current_cl_ref_idx + len_frames;
        let cl_segment = Centerline {
            points: centerline.points[current_cl_ref_idx..cl_end_idx].to_vec(),
            branch_start_indices: vec![0],
        };

        let mut angle = initial_rotation - angle_search_range;
        while angle <= initial_rotation + angle_search_range {
            let ref_pt = (
                centerline.points[current_cl_ref_idx].contour_point.x,
                centerline.points[current_cl_ref_idx].contour_point.y,
                centerline.points[current_cl_ref_idx].contour_point.z,
            );

            let transformed = apply_transformations(
                rotate_by_best_rotation(target.clone(), angle),
                &cl_segment,
                &ref_pt,
            );

            let filtered_points = filter_points_in_region(
                mutated_points,
                &centerline.points[current_cl_ref_idx],
                &centerline.points[cl_end_idx - 1],
            );

            if filtered_points.is_empty() {
                angle += angle_step;
                continue;
            }

            let frames = &transformed.primary_geometry().frames;
            let n_points_per_frame = frames[0].lumen.points.len();
            let mut nested: Vec<Vec<ContourPoint>> = Vec::with_capacity(len_frames);

            let ratio =
                filtered_points.len() as f64 / (n_points_per_frame as f64 * len_frames as f64);
            let mut n_downsample = (ratio * n_points_per_frame as f64).ceil() as usize;
            n_downsample = n_downsample.clamp(1, n_points_per_frame);

            for frame in frames.iter() {
                if n_downsample < n_points_per_frame {
                    let downsampled =
                        native::downsample_contour_points(&frame.lumen.points, n_downsample);
                    nested.push(downsampled);
                } else {
                    nested.push(frame.lumen.points.clone());
                }
            }

            let flat_geometry_points: Vec<ContourPoint> = nested.into_iter().flatten().collect();
            let hausdorff_dist = hausdorff_distance(&filtered_points, &flat_geometry_points);

            if hausdorff_dist < min_hausdorff {
                min_hausdorff = hausdorff_dist;
                best_angle = angle;
                best_cl_ref_idx = current_cl_ref_idx;
            }

            angle += angle_step;
        }
    }

    println!(
        "Refined rotation: {:.2}°, Refined CL index: {}, Hausdorff: {:.2}",
        best_angle.to_degrees(),
        best_cl_ref_idx,
        min_hausdorff
    );

    (best_angle, best_cl_ref_idx)
}

/// Filter points to region between two centerline points
fn filter_points_in_region(
    points: &[ContourPoint],
    start_cl_point: &CenterlinePoint,
    end_cl_point: &CenterlinePoint,
) -> Vec<ContourPoint> {
    // Simple bounding box filter
    let margin = 5.0; // Small margin

    let min_x = start_cl_point
        .contour_point
        .x
        .min(end_cl_point.contour_point.x)
        - margin;
    let max_x = start_cl_point
        .contour_point
        .x
        .max(end_cl_point.contour_point.x)
        + margin;
    let min_y = start_cl_point
        .contour_point
        .y
        .min(end_cl_point.contour_point.y)
        - margin;
    let max_y = start_cl_point
        .contour_point
        .y
        .max(end_cl_point.contour_point.y)
        + margin;
    let min_z = start_cl_point
        .contour_point
        .z
        .min(end_cl_point.contour_point.z)
        - margin;
    let max_z = start_cl_point
        .contour_point
        .z
        .max(end_cl_point.contour_point.z)
        + margin;

    points
        .iter()
        .filter(|p| {
            p.x >= min_x
                && p.x <= max_x
                && p.y >= min_y
                && p.y <= max_y
                && p.z >= min_z
                && p.z <= max_z
        })
        .cloned()
        .collect()
}

pub fn rotate_by_best_rotation<T: AlignTarget>(target: T, angle: f64) -> T {
    target.rotate_all(angle)
}

pub fn apply_transformations<T: AlignTarget>(
    target: T,
    centerline: &Centerline,
    ref_pt: &(f64, f64, f64),
) -> T {
    let transformations =
        get_transformations(target.primary_geometry().clone(), centerline, ref_pt);
    target.apply_frame_transforms(&transformations)
}

fn apply_transforms_to_geometry(geometry: &mut Geometry, transformations: &[FrameTransformation]) {
    for (i, frame) in geometry.frames.iter_mut().enumerate() {
        if i < transformations.len() {
            let tr = &transformations[i];
            apply_transformation_to_contour(&mut frame.lumen, tr);
            for contour in frame.extras.values_mut() {
                apply_transformation_to_contour(contour, tr);
            }
            if let Some(ref mut reference_pt) = frame.reference_point {
                *reference_pt = tr.apply_to_point(reference_pt);
            }
            frame.centroid = frame.lumen.centroid.unwrap_or((0.0, 0.0, 0.0));
        }
    }
}

#[cfg(test)]
mod align_algorithms_tests {
    use super::*;
    use crate::types::native::contour::ContourType;
    use crate::types::native::frame::Frame;
    use crate::types::native::geometry::Geometry;
    use std::collections::HashMap;

    fn create_test_contour(id: u32, original_frame: u32, points: Vec<ContourPoint>) -> Contour {
        Contour {
            id,
            original_frame,
            points,
            centroid: None,
            aortic_thickness: None,
            pulmonary_thickness: None,
            kind: ContourType::Lumen,
        }
    }

    fn create_test_centerline_point(x: f64, y: f64, z: f64, frame_index: u32) -> CenterlinePoint {
        CenterlinePoint {
            contour_point: ContourPoint {
                frame_index,
                point_index: 0,
                x,
                y,
                z,
                aortic: false,
            },
            tangent: Vector3::new(0.0, 0.0, 1.0), // Default tangent pointing up
            branch_id: 0,
            radius: 0.0,
        }
    }

    #[test]
    fn test_frame_transformation_apply_to_point() {
        let transformation = FrameTransformation {
            frame_index: 0,
            translation: Vector3::new(1.0, 2.0, 3.0),
            rotation: Rotation3::identity(),
            pivot: Point3::new(0.0, 0.0, 0.0),
        };

        let point = ContourPoint {
            frame_index: 0,
            point_index: 0,
            x: 1.0,
            y: 1.0,
            z: 1.0,
            aortic: false,
        };

        let transformed = transformation.apply_to_point(&point);

        // Only translation should be applied (rotation is identity)
        assert_eq!(transformed.x, 2.0);
        assert_eq!(transformed.y, 3.0);
        assert_eq!(transformed.z, 4.0);
        assert_eq!(transformed.frame_index, point.frame_index);
        assert_eq!(transformed.point_index, point.point_index);
        assert_eq!(transformed.aortic, point.aortic);
    }

    #[test]
    fn test_frame_transformation_with_rotation() {
        // 90 degree rotation around Z axis
        let rotation = Rotation3::from_axis_angle(&Vector3::z_axis(), std::f64::consts::FRAC_PI_2);
        let transformation = FrameTransformation {
            frame_index: 0,
            translation: Vector3::new(0.0, 0.0, 0.0),
            rotation,
            pivot: Point3::new(0.0, 0.0, 0.0),
        };

        let point = ContourPoint {
            frame_index: 0,
            point_index: 0,
            x: 1.0,
            y: 0.0,
            z: 0.0,
            aortic: false,
        };

        let transformed = transformation.apply_to_point(&point);

        // Point (1, 0, 0) rotated 90° around Z should become (0, 1, 0)
        assert!((transformed.x - 0.0).abs() < 1e-12);
        assert!((transformed.y - 1.0).abs() < 1e-12);
        assert!((transformed.z - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_align_frame() {
        // Create a simple square contour in XY plane
        let points = vec![
            ContourPoint {
                frame_index: 0,
                point_index: 0,
                x: -1.0,
                y: -1.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 0,
                point_index: 1,
                x: 1.0,
                y: -1.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 0,
                point_index: 2,
                x: 1.0,
                y: 1.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 0,
                point_index: 3,
                x: -1.0,
                y: 1.0,
                z: 0.0,
                aortic: false,
            },
        ];

        let contour = create_test_contour(0, 0, points);
        let centerline_point = create_test_centerline_point(10.0, 10.0, 10.0, 0);

        let transformation = align_frame(&contour, &centerline_point);

        assert_eq!(transformation.frame_index, 0);

        // The transformation should move the centroid (0,0,0) to (10,10,10)
        assert!((transformation.translation.x - 10.0).abs() < 1e-12);
        assert!((transformation.translation.y - 10.0).abs() < 1e-12);
        assert!((transformation.translation.z - 10.0).abs() < 1e-12);

        // Pivot should be the centerline point
        assert!((transformation.pivot.x - 10.0).abs() < 1e-12);
        assert!((transformation.pivot.y - 10.0).abs() < 1e-12);
        assert!((transformation.pivot.z - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_apply_transformation_to_contour() {
        let points = vec![
            ContourPoint {
                frame_index: 0,
                point_index: 0,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 0,
                point_index: 1,
                x: 1.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
        ];

        let mut contour = create_test_contour(0, 0, points);
        contour.centroid = Some((0.5, 0.0, 0.0));

        let transformation = FrameTransformation {
            frame_index: 0,
            translation: Vector3::new(2.0, 3.0, 4.0),
            rotation: Rotation3::identity(),
            pivot: Point3::new(0.0, 0.0, 0.0),
        };

        apply_transformation_to_contour(&mut contour, &transformation);

        // Check points were transformed
        assert!((contour.points[0].x - 2.0).abs() < 1e-12);
        assert!((contour.points[0].y - 3.0).abs() < 1e-12);
        assert!((contour.points[0].z - 4.0).abs() < 1e-12);

        assert!((contour.points[1].x - 3.0).abs() < 1e-12);
        assert!((contour.points[1].y - 3.0).abs() < 1e-12);
        assert!((contour.points[1].z - 4.0).abs() < 1e-12);

        // Check centroid was transformed
        let centroid = contour.centroid.unwrap();
        assert!((centroid.0 - 2.5).abs() < 1e-12);
        assert!((centroid.1 - 3.0).abs() < 1e-12);
        assert!((centroid.2 - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_calculate_normal() {
        // Create points in XY plane (normal should be +Z or -Z)
        let points = vec![
            ContourPoint {
                frame_index: 0,
                point_index: 0,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 0,
                point_index: 1,
                x: 1.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 0,
                point_index: 2,
                x: 0.0,
                y: 1.0,
                z: 0.0,
                aortic: false,
            },
        ];

        let centroid = (0.0, 0.0, 0.0);
        let normal = calculate_normal(&points, &centroid);

        // Normal should be approximately in Z direction (up or down)
        // The function takes negative of the computed normal, so we need to check magnitude
        assert!(normal.norm() > 0.0);

        // The cross product of (1,0,0) and (0,1,0) is (0,0,1) or (0,0,-1)
        // After normalization and negation, it should be unit length
        assert!((normal.norm() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_rotate_contour_around_centroid() {
        // Create a contour with clear normal direction
        let points = vec![
            ContourPoint {
                frame_index: 0,
                point_index: 0,
                x: 1.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 0,
                point_index: 1,
                x: 0.0,
                y: 1.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 0,
                point_index: 2,
                x: -1.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 0,
                point_index: 3,
                x: 0.0,
                y: -1.0,
                z: 0.0,
                aortic: false,
            },
        ];

        let mut contour = create_test_contour(0, 0, points);
        contour.centroid = Some((0.0, 0.0, 0.0));

        // 90 degree rotation around Z axis
        rotate_contour_around_centroid(&mut contour, std::f64::consts::FRAC_PI_2);

        // With improved normal calculation, check the first point
        // Point (1,0,0) should rotate to approximately (0,1,0)
        assert!((contour.points[0].x - 0.0).abs() < 1e-6);
        assert!((contour.points[0].y - 1.0).abs() < 1e-6);
        assert!((contour.points[0].z - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_get_transformations() {
        // Create a simple geometry with one frame
        let contour_points = vec![
            ContourPoint {
                frame_index: 0,
                point_index: 0,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 0,
                point_index: 1,
                x: 1.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
        ];

        let lumen = create_test_contour(0, 0, contour_points);

        let frame = Frame {
            id: 0,
            centroid: (0.5, 0.0, 0.0),
            lumen,
            extras: HashMap::new(),
            reference_point: None,
        };

        let geometry = Geometry {
            frames: vec![frame],
            label: "test".to_string(),
        };

        // Create a centerline with matching frame indices
        let centerline_points = vec![
            create_test_centerline_point(10.0, 10.0, 10.0, 0),
            create_test_centerline_point(11.0, 10.0, 10.0, 1),
        ];

        let centerline = Centerline {
            points: centerline_points,
            branch_start_indices: vec![0],
        };
        let ref_pt = (10.0, 10.0, 10.0);

        let transformations = get_transformations(geometry, &centerline, &ref_pt);

        // Should get one transformation for the one frame
        assert_eq!(transformations.len(), 1);
        assert_eq!(transformations[0].frame_index, 0);
    }

    #[test]
    fn test_best_rotation_three_point_simple_case() {
        // Create a simple circular contour
        let points: Vec<ContourPoint> = (0..8)
            .map(|i| {
                let angle = (i as f64) * std::f64::consts::FRAC_PI_4;
                ContourPoint {
                    frame_index: 0,
                    point_index: i as u32,
                    x: angle.cos(),
                    y: angle.sin(),
                    z: 0.0,
                    aortic: false,
                }
            })
            .collect();

        let mut contour = create_test_contour(0, 0, points);
        contour.centroid = Some((0.0, 0.0, 0.0));

        let reference_point = ContourPoint {
            frame_index: 0,
            point_index: 0, // first point as reference
            x: 1.0,
            y: 0.0,
            z: 0.0,
            aortic: false,
        };

        // Set targets to match the current positions (so best rotation should be 0)
        let main_ref_pt = (1.0, 0.0, 0.0);
        let counterclockwise_ref_pt = (0.0, 1.0, 0.0);
        let clockwise_ref_pt = (-1.0, 0.0, 0.0);
        let angle_step = std::f64::consts::FRAC_PI_8; // 22.5 degree steps

        let centerline_point = create_test_centerline_point(0.0, 0.0, 0.0, 0);

        let best_angle = best_rotation_three_point(
            &contour,
            &reference_point,
            main_ref_pt,
            counterclockwise_ref_pt,
            clockwise_ref_pt,
            angle_step,
            &centerline_point,
        );

        // With targets matching current positions, best rotation should be near 0
        // Allow some tolerance due to discrete angle steps
        assert!(best_angle.abs() < angle_step + 1e-6);
    }
}
