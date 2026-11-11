use super::py_contour_point::PyContourPoint;
use crate::types::native::{Annotation, ContourPoint, Transform};
use pyo3::prelude::*;

/// Python representation of free-form per-frame annotation points.
///
/// Annotations (side-branch outlines, plaque regions, landmarks) follow
/// every rigid transform of their frame. They are never sorted, smoothed,
/// interpolated, meshed or used for alignment.
///
/// Attributes
/// ----------
/// label : str
///     Free-form kind, e.g. ``"sidebranch"`` or ``"calcium"``.
/// group : int
///     Distinguishes several annotations with the same label in one frame.
/// points : list of PyContourPoint
///     Points in input order.
///
/// Examples
/// --------
/// >>> ann = PyAnnotation(label="sidebranch", group=0, points=[p1, p2, p3])
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct PyAnnotation {
    #[pyo3(get, set)]
    pub label: String,
    #[pyo3(get, set)]
    pub group: u32,
    #[pyo3(get, set)]
    pub points: Vec<PyContourPoint>,
}

#[pymethods]
impl PyAnnotation {
    #[new]
    #[pyo3(signature = (label, group=0, points=Vec::new()))]
    fn new(label: String, group: u32, points: Vec<PyContourPoint>) -> Self {
        Self {
            label,
            group,
            points,
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "Annotation(label='{}', group={}, points={})",
            self.label,
            self.group,
            self.points.len()
        )
    }

    fn __len__(&self) -> usize {
        self.points.len()
    }

    /// Rotate the annotation in the xy-plane by an angle in degrees.
    ///
    /// Parameters
    /// ----------
    /// angle_deg : float
    ///     Rotation angle in degrees.
    /// center : tuple of float, optional
    ///     ``(x, y)`` rotation center. Defaults to the mean of the points.
    ///
    /// Returns
    /// -------
    /// PyAnnotation
    ///     New rotated annotation.
    #[pyo3(signature = (angle_deg, center=None))]
    pub fn rotate(&self, angle_deg: f64, center: Option<(f64, f64)>) -> PyAnnotation {
        let center = center.unwrap_or_else(|| {
            let n = self.points.len().max(1) as f64;
            let sx: f64 = self.points.iter().map(|p| p.x).sum();
            let sy: f64 = self.points.iter().map(|p| p.y).sum();
            (sx / n, sy / n)
        });
        let rotated = self
            .to_rust_annotation()
            .rotate(angle_deg.to_radians(), center);
        PyAnnotation::from(&rotated)
    }

    /// Translate the annotation by ``(dx, dy, dz)``.
    ///
    /// Returns
    /// -------
    /// PyAnnotation
    ///     New translated annotation.
    #[pyo3(signature = (dx, dy, dz))]
    pub fn translate(&self, dx: f64, dy: f64, dz: f64) -> PyAnnotation {
        let moved = self.to_rust_annotation().translate(dx, dy, dz);
        PyAnnotation::from(&moved)
    }
}

impl PyAnnotation {
    pub fn to_rust_annotation(&self) -> Annotation {
        Annotation {
            label: self.label.clone(),
            group: self.group,
            points: self.points.iter().map(ContourPoint::from).collect(),
        }
    }
}

impl From<&Annotation> for PyAnnotation {
    fn from(annotation: &Annotation) -> Self {
        PyAnnotation {
            label: annotation.label.clone(),
            group: annotation.group,
            points: annotation.points.iter().map(PyContourPoint::from).collect(),
        }
    }
}
