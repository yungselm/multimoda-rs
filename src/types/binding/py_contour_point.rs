use crate::types::native::{ContourPoint, Point3D};
use pyo3::prelude::*;

/// Python representation of a 3D contour point.
///
/// Attributes
/// ----------
/// frame_index : int
///     Frame number in the acquisition sequence.
/// point_index : int
///     Index of this point within its contour.
/// x : float
///     X-coordinate in mm.
/// y : float
///     Y-coordinate in mm.
/// z : float
///     Z-coordinate (depth) in mm.
/// aortic : bool
///     ``True`` when the point is at an aortic position (relevant for
///     intramural vessel courses).
///
/// Examples
/// --------
/// >>> point = PyContourPoint(
/// ...     frame_index=0,
/// ...     point_index=1,
/// ...     x=1.23,
/// ...     y=4.56,
/// ...     z=7.89,
/// ...     aortic=True
/// ... )
#[pyclass(from_py_object)]
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

#[pymethods]
impl PyContourPoint {
    #[new]
    fn new(frame_index: u32, point_index: u32, x: f64, y: f64, z: f64, aortic: bool) -> Self {
        Self {
            frame_index,
            point_index,
            x,
            y,
            z,
            aortic,
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "Point(frame_id={}, pt_id={}, x={:.2}, y={:.2}, z={:.2}, aortic={})",
            self.frame_index, self.point_index, self.x, self.y, self.z, self.aortic
        )
    }

    fn __str__(&self) -> String {
        format!(
            "Point(frame_id={}, pt_id={}, x={:.2}, y={:.2}, z={:.2}, aortic={})",
            self.frame_index, self.point_index, self.x, self.y, self.z, self.aortic
        )
    }

    /// Euclidean distance to another PyContourPoint.
    ///
    /// Parameters
    /// ----------
    /// other : PyContourPoint
    ///     The target point to measure the distance to.
    ///
    /// Returns
    /// -------
    /// float
    ///     Euclidean distance between the two points in mm.
    ///
    /// Examples
    /// --------
    /// >>> p1.distance(p2)
    pub fn distance(&self, other: &PyContourPoint) -> f64 {
        let p1: ContourPoint = ContourPoint::from(self);
        let p2: ContourPoint = ContourPoint::from(other);
        p1.distance_to(&p2)
    }
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

impl From<&PyContourPoint> for ContourPoint {
    fn from(point: &PyContourPoint) -> Self {
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

impl From<&&ContourPoint> for PyContourPoint {
    fn from(point: &&ContourPoint) -> Self {
        (*point).into()
    }
}
