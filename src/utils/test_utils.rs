use crate::io::input::{Contour, ContourPoint};
use std::f64::consts::PI;

/// Generates ellipse contour points for testing
pub fn generate_ellipse_points(
    major: f64,
    minor: f64,
    num_points: usize,
    rotation: f64,
    translation: (f64, f64),
    frame_idx: u32,
) -> Vec<ContourPoint> {
    let mut points = Vec::with_capacity(num_points);
    for i in 0..num_points {
        let theta = 2.0 * PI * (i as f64) / (num_points as f64);
        let x = major * theta.cos();
        let y = minor * theta.sin();
        let (x_rot, y_rot) = rotate_point((x, y), rotation);
        points.push(ContourPoint {
            frame_index: frame_idx,
            point_index: i as u32,
            x: x_rot + translation.0,
            y: y_rot + translation.1,
            z: 0.0,
            aortic: false,
        });
    }
    points
}

/// Rotates a point around origin
pub fn rotate_point(point: (f64, f64), angle: f64) -> (f64, f64) {
    let (x, y) = point;
    let cos = angle.cos();
    let sin = angle.sin();
    (x * cos - y * sin, x * sin + y * cos)
}

/// Creates a dummy Contour with given id
pub fn new_dummy_contour(id: u32) -> Contour {
    Contour {
        id,
        points: Vec::new(),
        centroid: (0.0, 0.0, 0.0),
        aortic_thickness: None,
        pulmonary_thickness: None,
    }
}
