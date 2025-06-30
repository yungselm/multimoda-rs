use pyo3::prelude::*;
use crate::geometry::{ContourPoint, Contour};

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyContourPoint {
    #[pyo3(get, set)]
    pub frame_index: u32,
    #[pyo3(get, set)]
    pub point_index: u32,
    #[pyo3(get, set)]
    pub x: f64,
    #[pyo3(get, set)]
    pub y: f64,
    #[pyo3(get, set)]
    pub z: f64,
    #[pyo3(get, set)]
    pub aortic: bool,
}

impl From<&ContourPoint> for PyContourPoint {
    fn from(point: &ContourPoint) -> Self {
        Self {
            frame_index: point.frame_index,
            point_index: point.point_index,
            x: point.x,
            y: point.y,
            z: point.z,
            aortic: point.aortic,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyContour {
    #[pyo3(get, set)]
    pub id: u32,
    #[pyo3(get, set)]
    pub points: Vec<PyContourPoint>,
    #[pyo3(get, set)]
    pub centroid: (f64, f64, f64),
}

#[pymethods]
impl PyContour {
    #[new]
    pub fn new(id: u32, points: Vec<PyContourPoint>, centroid: (f64, f64, f64)) -> Self {
        Self { id, points, centroid }
    }

    pub fn find_farthest_points(&self) -> ((PyContourPoint, PyContourPoint), f64) {
        let rust_points: Vec<ContourPoint> = self.points.iter().map(|p| ContourPoint {
            frame_index: p.frame_index,
            point_index: p.point_index,
            x: p.x,
            y: p.y,
            z: p.z,
            aortic: p.aortic,
        }).collect();

        let rust_contour = Contour {
            id: self.id,
            points: rust_points,
            centroid: self.centroid,
            aortic_thickness: None,
            pulmonary_thickness: None,
        };

        let ((p1, p2), distance) = rust_contour.find_farthest_points();

        ((PyContourPoint::from(p1), PyContourPoint::from(p2)), distance)
    }

    
    /// Find the pair of points whose chord is the smallest diameter,
    /// by matching each point to the one whose angle (about the centroid)
    /// differs by as close to π radians as possible.
    pub fn find_closest_opposite(&self) -> ((PyContourPoint, PyContourPoint), f64) {
        let n = self.points.len();
        assert!(n > 2, "Need at least 3 points");

        // 1) Compute centroid (x0,y0)
        let (cx, cy, _) = self.centroid;

        // 2) Precompute angles
        let thetas: Vec<f64> = self.points.iter()
            .map(|p| {
                let mut t = (p.y - cy).atan2(p.x - cx);
                if t < 0.0 { t += 2.0 * std::f64::consts::PI; }
                t
            })
            .collect();

        let mut min_dist = f64::MAX;
        let mut best_pair = (&self.points[0], &self.points[1]);

        // 3) Brute‐force: for each i, find j that best approximates θi+π
        for i in 0..n {
            // let target = (thetas[i] + std::f64::consts::PI) % (2.0 * std::f64::consts::PI);
            let mut best_angle_diff = f64::MAX;
            let mut best_j = i;

            for j in 0..n {
                if j == i { continue; }
                // compute angular separation in [0,2π)
                let mut delta = (thetas[j] - thetas[i]).abs();
                if delta > std::f64::consts::PI {
                    delta = 2.0 * std::f64::consts::PI - delta;
                }
                let diff = (delta - std::f64::consts::PI).abs();
                if diff < best_angle_diff {
                    best_angle_diff = diff;
                    best_j = j;
                }
            }

            // 4) Compute chord length between i and best_j
            let pi = &self.points[i];
            let pj = &self.points[best_j];
            let dx = pi.x - pj.x;
            let dy = pi.y - pj.y;
            let dist = (dx*dx + dy*dy).sqrt();
            if dist < min_dist {
                min_dist = dist;
                best_pair = (pi, pj);
            }
        }

        (best_pair, min_dist)
    }
}
