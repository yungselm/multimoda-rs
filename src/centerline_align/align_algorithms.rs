use crate::io::input::{Centerline, CenterlinePoint};
use crate::io::input::{Contour, ContourPoint};
use crate::io::Geometry;
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

pub fn get_transformations(mesh: Geometry, centerline: &Centerline) -> Vec<FrameTransformation> {
    let mut transformations = Vec::new();

    for mut contour in mesh.contours.into_iter() {
        if let Some(cl_point) = centerline.get_by_frame(contour.id) {
            let transformation = align_frame(&mut contour, cl_point);
            transformations.push(transformation)
        }
    }
    transformations
}

/// Modified align_frame returns the transformation applied.
fn align_frame(frame: &mut Contour, cl_point: &CenterlinePoint) -> FrameTransformation {
    if frame.id != cl_point.contour_point.frame_index {
        panic!(
            "Frame Index {} does not match Centerline Point Frame Index {}",
            frame.id, cl_point.contour_point.frame_index
        );
    }

    // === Translation Step ===
    // Compute the translation vector to bring the frame's centroid to the centerline point.
    let translation_vec = Vector3::new(
        cl_point.contour_point.x - frame.centroid.0,
        cl_point.contour_point.y - frame.centroid.1,
        cl_point.contour_point.z - frame.centroid.2,
    );
    for point in frame.points.iter_mut() {
        point.x += translation_vec.x;
        point.y += translation_vec.y;
        point.z += translation_vec.z;
    }
    frame.centroid.0 += translation_vec.x;
    frame.centroid.1 += translation_vec.y;
    frame.centroid.2 += translation_vec.z;

    // === Rotation Step ===
    // Compute the rotation needed to align the frame's normal with the centerline normal.
    let current_normal = calculate_normal(&frame.points, &frame.centroid);
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
    if angle.abs() >= 1e-6 {
        for point in frame.points.iter_mut() {
            let current_point = Point3::new(point.x, point.y, point.z);
            let relative_vector = current_point - pivot;
            let rotated_relative = rotation * relative_vector;
            let rotated_point = pivot + rotated_relative;
            point.x = rotated_point.x;
            point.y = rotated_point.y;
            point.z = rotated_point.z;
        }
    }

    // Return the transformation details for later use (speed up).
    FrameTransformation {
        frame_index: frame.id,
        translation: translation_vec,
        rotation,
        pivot,
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
    while angle < 6.283185 {
        //aprox 360°
        // maybe better approach then bruteforce, fix later, still fast enough
        let mut temp_frame = contour.clone();

        temp_frame.rotate_contour(angle);

        align_frame(&mut temp_frame, centerline_point);
        let temp_contour = &temp_frame.points;

        let n_points = temp_contour.len() as u32;

        let p_aortic = temp_contour
            .iter()
            .find(|p| p.point_index == index_reference)
            .unwrap();
        let cont_p_upper = temp_contour.iter().find(|p| p.point_index == 0).unwrap();
        let cont_p_lower = temp_contour
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

        // println!("angle: {:?}, error: {:?}", angle, total_error);

        if total_error < min_total_error {
            min_total_error = total_error;
            best_angle = angle;
        }
        angle += angle_step;
    }

    best_angle
}
