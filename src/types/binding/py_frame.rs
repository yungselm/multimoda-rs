use super::py_contour::PyContour;
use super::py_contour_point::PyContourPoint;
use crate::types::native::{ContourPoint, ContourType, Frame, Transform};
use pyo3::prelude::*;
use std::collections::HashMap;

/// Python representation of a single intravascular imaging frame.
///
/// Attributes
/// ----------
/// id : int
///     Frame identifier.
/// centroid : tuple of float
///     ``(x, y, z)`` centroid of the frame.
/// lumen : PyContour
///     Lumen contour for this frame.
/// extras : dict of str to PyContour
///     Additional contour types keyed by name: ``"Eem"``,
///     ``"Calcification"``, ``"Sidebranch"``, ``"Catheter"``, ``"Wall"``.
/// reference_point : PyContourPoint or None
///     Reference position used for alignment, if available.
///
/// Examples
/// --------
/// >>> geom = PyFrame(
/// ...     id=0,
/// ...     centroid=(0.0, 0.0, 0.0),
/// ...     lumen=lumen_contour,
/// ...     extras={"Eem": eem_contour},
/// ...     reference_point=ref_point
/// ... )
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct PyFrame {
    #[pyo3(get, set)]
    pub id: u32,
    #[pyo3(get, set)]
    pub centroid: (f64, f64, f64),
    #[pyo3(get, set)]
    pub lumen: PyContour,
    #[pyo3(get, set)]
    pub extras: HashMap<String, PyContour>, // String keys for ContourType
    #[pyo3(get, set)]
    pub reference_point: Option<PyContourPoint>,
}

#[pymethods]
impl PyFrame {
    #[new]
    fn new(
        id: u32,
        centroid: (f64, f64, f64),
        lumen: PyContour,
        extras: HashMap<String, PyContour>,
        reference_point: Option<PyContourPoint>,
    ) -> Self {
        Self {
            id,
            centroid,
            lumen,
            extras,
            reference_point,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Frame(id={}, centroid=({:.2}, {:.2}, {:.2}), lumen={}, extras={})",
            self.id,
            self.centroid.0,
            self.centroid.1,
            self.centroid.2,
            self.lumen.__repr__(),
            self.extras.len()
        )
    }

    /// Rotate all contour points in the frame by the given angle.
    ///
    /// Parameters
    /// ----------
    /// angle_deg : float
    ///     Rotation angle in degrees.
    ///
    /// Returns
    /// -------
    /// frame : PyFrame
    ///     New frame with all contours rotated.
    #[pyo3(signature = (angle_deg))]
    pub fn rotate(&self, angle_deg: f64) -> PyResult<PyFrame> {
        let rust_frame = self.to_rust_frame()?;
        let center = (rust_frame.centroid.0, rust_frame.centroid.1);
        let rust_frame = rust_frame.rotate(angle_deg.to_radians(), center);
        Ok(PyFrame::from(&rust_frame))
    }

    /// Translate all contour points in the frame by the given offsets.
    ///
    /// Parameters
    /// ----------
    /// dx : float
    ///     Translation along the x-axis in mm.
    /// dy : float
    ///     Translation along the y-axis in mm.
    /// dz : float
    ///     Translation along the z-axis in mm.
    ///
    /// Returns
    /// -------
    /// frame : PyFrame
    ///     New frame with all contours translated.
    #[pyo3(signature = (dx, dy, dz))]
    pub fn translate(&self, dx: f64, dy: f64, dz: f64) -> PyResult<PyFrame> {
        let rust_frame = self.to_rust_frame()?.translate(dx, dy, dz);
        Ok(PyFrame::from(&rust_frame))
    }

    /// Sort contour points in the frame by their angular position.
    ///
    /// Returns
    /// -------
    /// frame : PyFrame
    ///     New frame with contour points sorted angularly.
    pub fn sort_frame_points(&self) -> PyResult<PyFrame> {
        let mut rust_frame = self.to_rust_frame()?;
        rust_frame.sort_frame_points();
        Ok(PyFrame::from(&rust_frame))
    }
}

impl From<Frame> for PyFrame {
    fn from(frame: Frame) -> Self {
        PyFrame {
            id: frame.id,
            centroid: frame.centroid,
            lumen: PyContour::from(&frame.lumen),
            extras: frame
                .extras
                .into_iter()
                .map(|(key, value)| (key.to_string(), PyContour::from(&value)))
                .collect(),
            reference_point: frame.reference_point.map(|p| PyContourPoint::from(&p)),
        }
    }
}

impl From<&Frame> for PyFrame {
    fn from(frame: &Frame) -> Self {
        let mut extras = HashMap::new();
        for (contour_type, contour) in &frame.extras {
            let type_str = match contour_type {
                ContourType::Eem => "Eem",
                ContourType::Calcification => "Calcification",
                ContourType::Sidebranch => "Sidebranch",
                ContourType::Catheter => "Catheter",
                ContourType::Wall => "Wall",
                _ => continue,
            };
            extras.insert(type_str.to_string(), PyContour::from(contour));
        }

        PyFrame {
            id: frame.id,
            centroid: frame.centroid,
            lumen: PyContour::from(&frame.lumen),
            extras,
            reference_point: frame.reference_point.as_ref().map(PyContourPoint::from),
        }
    }
}

impl PyFrame {
    pub fn to_rust_frame(&self) -> PyResult<Frame> {
        let lumen = self.lumen.to_rust_contour()?;

        let mut extras = HashMap::new();
        for (key, py_contour) in &self.extras {
            let contour_type = match key.as_str() {
                "Eem" => ContourType::Eem,
                "Calcification" => ContourType::Calcification,
                "Sidebranch" => ContourType::Sidebranch,
                "Catheter" => ContourType::Catheter,
                "Wall" => ContourType::Wall,
                _ => continue, // Skip unknown types
            };
            let mut contour = py_contour.to_rust_contour()?;
            contour.kind = contour_type;
            extras.insert(contour_type, contour);
        }

        let reference_point = self.reference_point.as_ref().map(ContourPoint::from);

        Ok(Frame {
            id: self.id,
            centroid: self.centroid,
            lumen,
            extras,
            reference_point,
        })
    }
}
