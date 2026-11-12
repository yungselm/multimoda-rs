use super::py_contour_point::PyContourPoint;
use crate::types::native::{ContourPoint, Structure, StructureKind, Transform};
use pyo3::prelude::*;

/// Python representation of a non-pipe structure of one frame.
///
/// In :class:`PyInputData` the points are the drawn contour. On a frame of a
/// :class:`PyGeometry` they are the point cloud generated from it, which follows
/// every rigid transform of its frame. A closed contour is jittered, an open
/// side-branch arc (ends on the lumen border) is turned around its own axis,
/// the in-plane line from the lumen centroid to the middle of the arc.
///
/// Attributes
/// ----------
/// kind : str
///     ``"sidebranch"``, ``"calcification"``, ``"lipid"``, ``"macrophage"``, or
///     any other name for a custom kind (e.g. ``"fibrous_cap"``). ``"branch"``
///     and ``"calcium"`` are read as ``"sidebranch"`` and ``"calcification"``.
/// closed : bool
///     ``True`` for a closed contour, ``False`` for an open arc (side branches
///     leaving the lumen).
/// points : list of PyContourPoint
///     Contour points in drawing order, or the generated cloud.
///
/// Examples
/// --------
/// >>> s = PyStructure("calcification", points=[p1, p2, p3])
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct PyStructure {
    pub kind: StructureKind,
    #[pyo3(get, set)]
    pub closed: bool,
    #[pyo3(get, set)]
    pub points: Vec<PyContourPoint>,
}

#[pymethods]
impl PyStructure {
    #[new]
    #[pyo3(signature = (kind, points=Vec::new(), closed=true))]
    fn new(kind: &str, points: Vec<PyContourPoint>, closed: bool) -> Self {
        Self {
            kind: StructureKind::from_name(kind),
            closed,
            points,
        }
    }

    /// Kind name, unknown names are kept as custom kinds.
    #[getter(kind)]
    fn get_kind(&self) -> String {
        self.kind.name().to_string()
    }

    #[setter(kind)]
    fn set_kind(&mut self, kind: &str) {
        self.kind = StructureKind::from_name(kind);
    }

    /// ``True`` when the kind is not one of the built-in kinds.
    #[getter]
    fn is_custom(&self) -> bool {
        matches!(self.kind, StructureKind::Custom(_))
    }

    /// Built-in kind names.
    #[staticmethod]
    fn known_kinds() -> Vec<&'static str> {
        vec!["sidebranch", "calcification", "lipid", "macrophage"]
    }

    pub fn __repr__(&self) -> String {
        format!(
            "Structure(kind='{}', closed={}, points={})",
            self.kind.name(),
            if self.closed { "True" } else { "False" },
            self.points.len()
        )
    }

    fn __len__(&self) -> usize {
        self.points.len()
    }

    /// Rotate the structure in the xy-plane by an angle in degrees.
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
    /// PyStructure
    ///     New rotated structure.
    #[pyo3(signature = (angle_deg, center=None))]
    pub fn rotate(&self, angle_deg: f64, center: Option<(f64, f64)>) -> PyStructure {
        let center = center.unwrap_or_else(|| {
            let n = self.points.len().max(1) as f64;
            let sx: f64 = self.points.iter().map(|p| p.x).sum();
            let sy: f64 = self.points.iter().map(|p| p.y).sum();
            (sx / n, sy / n)
        });
        let rotated = self
            .to_rust_structure()
            .rotate(angle_deg.to_radians(), center);
        PyStructure::from(&rotated)
    }

    /// Translate the structure by ``(dx, dy, dz)``.
    ///
    /// Returns
    /// -------
    /// PyStructure
    ///     New translated structure.
    #[pyo3(signature = (dx, dy, dz))]
    pub fn translate(&self, dx: f64, dy: f64, dz: f64) -> PyStructure {
        let moved = self.to_rust_structure().translate(dx, dy, dz);
        PyStructure::from(&moved)
    }
}

impl PyStructure {
    pub fn to_rust_structure(&self) -> Structure {
        Structure {
            kind: self.kind.clone(),
            closed: self.closed,
            points: self.points.iter().map(ContourPoint::from).collect(),
        }
    }
}

impl From<&Structure> for PyStructure {
    fn from(structure: &Structure) -> Self {
        PyStructure {
            kind: structure.kind.clone(),
            closed: structure.closed,
            points: structure.points.iter().map(PyContourPoint::from).collect(),
        }
    }
}
