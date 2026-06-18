use super::py_contour_point::PyContourPoint;
use crate::types::native::{CenterlinePoint, ContourPoint};
use nalgebra::Vector3;
use pyo3::prelude::*;

/// Python representation of a centerline point.
///
/// Combines a contour point with its local tangent vector.
///
/// Attributes
/// ----------
/// contour_point : PyContourPoint
///     Position of the centerline point in 3D space.
/// tangent : tuple of float
///     Tangent vector ``(tx, ty, tz)`` at this centerline position.
/// branch_id : int
///     Branch identifier.  ``0`` = main vessel; ``1+`` = side branches
///     ordered by descending length.
///
/// Examples
/// --------
/// >>> cl_point = PyCenterlinePoint(
/// ...     contour_point=point,
/// ...     tangent=(0.0, 1.0, 0.0)
/// ... )
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct PyCenterlinePoint {
    #[pyo3(get, set)]
    pub contour_point: PyContourPoint,
    #[pyo3(get, set)]
    pub tangent: (f64, f64, f64),
    #[pyo3(get, set)]
    pub branch_id: u32,
}

#[pymethods]
impl PyCenterlinePoint {
    #[new]
    #[pyo3(signature = (contour_point, tangent, branch_id = 0))]
    fn new(contour_point: PyContourPoint, tangent: (f64, f64, f64), branch_id: u32) -> Self {
        Self {
            contour_point,
            tangent,
            branch_id,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "CenterlinePoint(point={}, tangent=({:.3}, {:.3}, {:.3}), branch={})",
            self.contour_point.__repr__(),
            self.tangent.0,
            self.tangent.1,
            self.tangent.2,
            self.branch_id,
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

impl From<&CenterlinePoint> for PyCenterlinePoint {
    fn from(p: &CenterlinePoint) -> Self {
        PyCenterlinePoint {
            contour_point: PyContourPoint::from(&p.contour_point),
            tangent: (p.tangent[0], p.tangent[1], p.tangent[2]),
            branch_id: p.branch_id,
        }
    }
}

impl From<&PyCenterlinePoint> for CenterlinePoint {
    fn from(p: &PyCenterlinePoint) -> Self {
        CenterlinePoint {
            contour_point: ContourPoint::from(&p.contour_point),
            tangent: Vector3::new(p.tangent.0, p.tangent.1, p.tangent.2),
            branch_id: p.branch_id,
        }
    }
}
