use super::py_contour_point::PyContourPoint;
use crate::types::native::{Annotation, AnnotationKind, AnnotationRole, ContourPoint, Transform};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Python representation of non-pipe per-frame annotation points.
///
/// Annotations (side-branch outlines, plaque regions, landmarks) follow
/// every rigid transform of their frame. They are never sorted, smoothed,
/// interpolated, meshed or used for alignment.
///
/// Attributes
/// ----------
/// kind : str
///     ``"sidebranch"``, ``"calcification"``, ``"lipid"``, ``"macrophage"``,
///     ``"reference"``, or any other name for a custom kind (e.g. ``"fibrous_cap"``).
/// role : str
///     ``"region"``, ``"landmark"``, ``"axis_tip"``, ``"diameter_end"`` or ``"point"``.
/// group : int
///     Distinguishes several annotations of the same kind and role in one frame.
/// points : list of PyContourPoint
///     Points in input order.
///
/// Examples
/// --------
/// >>> ann = PyAnnotation("sidebranch", "region", group=0, points=[p1, p2, p3])
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct PyAnnotation {
    pub kind: AnnotationKind,
    pub role: AnnotationRole,
    #[pyo3(get, set)]
    pub group: u32,
    #[pyo3(get, set)]
    pub points: Vec<PyContourPoint>,
}

fn parse_role(role: &str) -> PyResult<AnnotationRole> {
    AnnotationRole::from_name(role).ok_or_else(|| {
        PyValueError::new_err(format!(
            "Unknown annotation role: '{role}'. Valid roles are: {}",
            AnnotationRole::ALL.map(|r| r.name()).join(", ")
        ))
    })
}

#[pymethods]
impl PyAnnotation {
    #[new]
    #[pyo3(signature = (kind, role, group=0, points=Vec::new()))]
    fn new(kind: &str, role: &str, group: u32, points: Vec<PyContourPoint>) -> PyResult<Self> {
        Ok(Self {
            kind: AnnotationKind::from_name(kind),
            role: parse_role(role)?,
            group,
            points,
        })
    }

    /// Kind name, unknown names are kept as custom kinds.
    #[getter(kind)]
    fn get_kind(&self) -> String {
        self.kind.name().to_string()
    }

    #[setter(kind)]
    fn set_kind(&mut self, kind: &str) {
        self.kind = AnnotationKind::from_name(kind);
    }

    /// Role name, unknown names raise ``ValueError``.
    #[getter(role)]
    fn get_role(&self) -> &'static str {
        self.role.name()
    }

    #[setter(role)]
    fn set_role(&mut self, role: &str) -> PyResult<()> {
        self.role = parse_role(role)?;
        Ok(())
    }

    /// ``True`` when the kind is not one of the built-in kinds.
    #[getter]
    fn is_custom(&self) -> bool {
        matches!(self.kind, AnnotationKind::Custom(_))
    }

    /// Built-in kind names.
    #[staticmethod]
    fn known_kinds() -> Vec<&'static str> {
        vec![
            "sidebranch",
            "calcification",
            "lipid",
            "macrophage",
            "reference",
        ]
    }

    /// Valid role names.
    #[staticmethod]
    fn roles() -> Vec<&'static str> {
        AnnotationRole::ALL.map(|r| r.name()).to_vec()
    }

    pub fn __repr__(&self) -> String {
        format!(
            "Annotation(kind='{}', role='{}', group={}, points={})",
            self.kind.name(),
            self.role.name(),
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
            kind: self.kind.clone(),
            role: self.role,
            group: self.group,
            points: self.points.iter().map(ContourPoint::from).collect(),
        }
    }
}

impl From<&Annotation> for PyAnnotation {
    fn from(annotation: &Annotation) -> Self {
        PyAnnotation {
            kind: annotation.kind.clone(),
            role: annotation.role,
            group: annotation.group,
            points: annotation.points.iter().map(PyContourPoint::from).collect(),
        }
    }
}
