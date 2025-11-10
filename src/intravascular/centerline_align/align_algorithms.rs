use crate::intravascular::io::geometry::{Contour, Geometry};
use crate::intravascular::io::input::ContourPoint;
use crate::intravascular::io::input::{Centerline, CenterlinePoint};
use nalgebra::{Point3, Rotation3, Unit, Vector3};

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

pub fn get_transformations(geometry: Geometry, centerline: &Centerline, ref_pt: &(f64, f64, f64)) -> Vec<FrameTransformation> {
    let mut transformations = Vec::new();

    let ref_id_cl = centerline.find_reference_cl_point_idx(ref_pt) as u32;

    for frame in geometry.frames.into_iter() {
        if let Some(cl_point) = centerline.get_by_frame(frame.id + ref_id_cl) {
            // Create one transformation per frame using the lumen as reference
            let transformation = align_frame(&frame.lumen, cl_point);
            transformations.push(transformation);
        }
    }
    transformations
}

/// Modified align_frame returns the transformation applied (doesn't mutate input)
fn align_frame(frame: &Contour, cl_point: &CenterlinePoint) -> FrameTransformation {
    if frame.original_frame != cl_point.contour_point.frame_index {
        panic!(
            "Frame Index {} does not match Centerline Point Frame Index {}",
            frame.original_frame, cl_point.contour_point.frame_index
        );
    }

    // Get centroid or compute if None
    let centroid = frame.centroid.unwrap_or_else(|| {
        let x_avg = frame.points.iter().map(|p| p.x).sum::<f64>() / frame.points.len() as f64;
        let y_avg = frame.points.iter().map(|p| p.y).sum::<f64>() / frame.points.len() as f64;
        let z_avg = frame.points.iter().map(|p| p.z).sum::<f64>() / frame.points.len() as f64;
        (x_avg, y_avg, z_avg)
    });

    // === Translation Step ===
    // Compute the translation vector to bring the frame's centroid to the centerline point.
    let translation_vec = Vector3::new(
        cl_point.contour_point.x - centroid.0,
        cl_point.contour_point.y - centroid.1,
        cl_point.contour_point.z - centroid.2,
    );

    // === Rotation Step ===
    // Compute the rotation needed to align the frame's normal with the centerline normal.
    let current_normal = calculate_normal(&frame.points, &centroid);
    let desired_normal = cl_point.normal;
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

    // Define the pivot as the centerline point.
    let pivot = Point3::new(
        cl_point.contour_point.x,
        cl_point.contour_point.y,
        cl_point.contour_point.z,
    );

    // Return the transformation details for later use
    FrameTransformation {
        frame_index: frame.original_frame,
        translation: translation_vec,
        rotation,
        pivot,
    }
}

/// Applies transformation to a contour (mutable version)
pub fn apply_transformation_to_contour(contour: &mut Contour, transformation: &FrameTransformation) {
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
        *centroid = (transformed_centroid.x, transformed_centroid.y, transformed_centroid.z);
    }
}

/// Calculates the normal vector using a stable cross product method.
fn calculate_normal(points: &[ContourPoint], centroid: &(f64, f64, f64)) -> Vector3<f64> {
    let p1 = &points[0];
    let p2 = &points[points.len() / 3];
    let p3 = &points[2 * points.len() / 3];

    let v1 = Vector3::new(p1.x - centroid.0, p1.y - centroid.1, p1.z - centroid.2);
    let v2 = Vector3::new(p2.x - centroid.0, p2.y - centroid.1, p2.z - centroid.2);
    let v3 = Vector3::new(p3.x - centroid.0, p3.y - centroid.1, p3.z - centroid.2);

    // need to take the negative normal, since centerline "appears backwards"
    -(v1.cross(&v2) + v2.cross(&v3)).normalize()
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
    aortic_ref_pt: (f64, f64, f64),
    upper_ref_pt: (f64, f64, f64),
    lower_ref_pt: (f64, f64, f64),
    angle_step: f64,
    centerline_point: &CenterlinePoint,
) -> f64 {
    let index_reference = reference_point.point_index;

    let [target_aortic, target_upper, target_lower]: [Point3<f64>; 3] =
        [aortic_ref_pt, upper_ref_pt, lower_ref_pt].map(|(x, y, z)| Point3::new(x, y, z));

    let mut best_angle = 0.0;
    let mut min_total_error = f64::MAX;

    let mut angle = 0.0;
    println!(
        "---------------------Centerline alignment: Finding optimal rotation---------------------"
    );
    
    while angle < 6.283185 { // approx 360°
        let mut temp_contour = contour.clone();

        // Rotate around centroid
        rotate_contour_around_centroid(&mut temp_contour, angle);

        // Apply centerline alignment transformation
        let transformation = align_frame(&temp_contour, centerline_point);
        apply_transformation_to_contour(&mut temp_contour, &transformation);
        
        let temp_points = &temp_contour.points;

        let n_points = temp_points.len() as u32;

        let p_aortic = temp_points
            .iter()
            .find(|p| p.point_index == index_reference)
            .unwrap();
        let cont_p_upper = temp_points.iter().find(|p| p.point_index == 0).unwrap();
        let cont_p_lower = temp_points
            .iter()
            .find(|p| p.point_index == (n_points / 2))
            .unwrap();

        let d_aortic = nalgebra::distance(
            &Point3::new(p_aortic.x, p_aortic.y, p_aortic.z),
            &target_aortic,
        );

        let d_upper = nalgebra::distance(
            &Point3::new(cont_p_upper.x, cont_p_upper.y, cont_p_upper.z),
            &target_upper,
        );

        let d_lower = nalgebra::distance(
            &Point3::new(cont_p_lower.x, cont_p_lower.y, cont_p_lower.z),
            &target_lower,
        );

        // Calculate sum of squared errors
        let total_error = d_aortic.powi(2) + d_upper.powi(2) + d_lower.powi(2);

        if total_error < min_total_error {
            min_total_error = total_error;
            best_angle = angle;
        }
        angle += angle_step;
    }

    best_angle
}
