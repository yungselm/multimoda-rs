use super::py_contour_point::PyContourPoint;
use crate::types::native::{Contour, ContourPoint, ContourType, Transform};
use pyo3::prelude::*;

/// Python representation of a 3D contour.
///
/// Attributes
/// ----------
/// id : int
///     Contour identifier (sequence number).
/// original_frame : int
///     Frame index from which this contour originates.
/// points : list of PyContourPoint
///     Ordered list of contour points.
/// centroid : tuple of float
///     ``(x, y, z)`` centroid coordinates of the contour.
/// aortic_thickness : float or None
///     Aortic wall thickness at this contour, if available.
/// pulmonary_thickness : float or None
///     Pulmonary wall thickness at this contour, if available.
/// kind : str
///     String representation of the contour type (e.g. ``"Lumen"``).
///
/// Examples
/// --------
/// >>> contour = PyContour(
/// ...     id=0,
/// ...     points=[point1, point2, ...],
/// ...     centroid=(1.0, 1.0, 1.0)
/// ... )
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct PyContour {
    #[pyo3(get, set)]
    pub id: u32,
    #[pyo3(get, set)]
    pub original_frame: u32,
    #[pyo3(get, set)]
    pub points: Vec<PyContourPoint>,
    #[pyo3(get, set)]
    pub centroid: (f64, f64, f64),
    #[pyo3(get, set)]
    pub aortic_thickness: Option<f64>,
    #[pyo3(get, set)]
    pub pulmonary_thickness: Option<f64>,
    #[pyo3(get, set)]
    pub kind: String, // String representation of ContourType
}

#[pymethods]
impl PyContour {
    /// Create a new PyContour instance.
    ///
    /// Parameters
    /// ----------
    /// id : int
    ///     Contour identifier.
    /// points : list of PyContourPoint
    ///     List of contour points.
    #[new]
    fn new(
        id: u32,
        original_frame: u32,
        points: Vec<PyContourPoint>,
        centroid: (f64, f64, f64),
        aortic_thickness: Option<f64>,
        pulmonary_thickness: Option<f64>,
        kind: String,
    ) -> Self {
        Self {
            id,
            original_frame,
            points,
            centroid,
            aortic_thickness,
            pulmonary_thickness,
            kind,
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "Contour(id={}, frame={}, points={}, centroid=({:.2}, {:.2}, {:.2}), kind={})",
            self.id,
            self.original_frame,
            self.points.len(),
            self.centroid.0,
            self.centroid.1,
            self.centroid.2,
            self.kind
        )
    }

    fn __len__(&self) -> usize {
        self.points.len()
    }

    /// Calculate the contour centroid by averaging all point coordinates.
    ///
    /// Examples
    /// --------
    /// >>> contour.compute_centroid()
    pub fn compute_centroid(&mut self) {
        if self.points.is_empty() {
            self.centroid = (0.0, 0.0, 0.0);
            return;
        }

        let (sum_x, sum_y, sum_z) = self.points.iter().fold((0.0, 0.0, 0.0), |(sx, sy, sz), p| {
            (sx + p.x, sy + p.y, sz + p.z)
        });

        let n = self.points.len() as f64;
        self.centroid = (sum_x / n, sum_y / n, sum_z / n);
    }

    /// Return contour points as a list of ``(x, y, z)`` tuples.
    ///
    /// Returns
    /// -------
    /// list of tuple of float
    ///     Each element is ``(x, y, z)`` coordinates of one contour point.
    ///
    /// Examples
    /// --------
    /// >>> contour.points_as_tuples()
    /// [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
    fn points_as_tuples(&self) -> Vec<(f64, f64, f64)> {
        self.points.iter().map(|p| (p.x, p.y, p.z)).collect()
    }

    /// Find the two farthest points in the contour.
    ///
    /// Returns
    /// -------
    /// points : tuple of PyContourPoint
    ///     Pair ``(p1, p2)`` of the two most distant points.
    /// distance : float
    ///     Euclidean distance between *p1* and *p2*.
    ///
    /// Examples
    /// --------
    /// >>> (p1, p2), distance = contour.find_farthest_points()
    pub fn find_farthest_points(&self) -> PyResult<((PyContourPoint, PyContourPoint), f64)> {
        let rust_contour = self.to_rust_contour()?;
        let ((p1, p2), distance) = rust_contour.find_farthest_points();
        Ok((
            (PyContourPoint::from(&p1), PyContourPoint::from(&p2)),
            distance,
        ))
    }

    /// Find the closest points on opposite sides of the contour.
    ///
    /// Returns
    /// -------
    /// points : tuple of PyContourPoint
    ///     Pair ``(p1, p2)`` of opposing contour points with minimum distance.
    /// distance : float
    ///     Euclidean distance between *p1* and *p2*.
    ///
    /// Examples
    /// --------
    /// >>> (p1, p2), distance = contour.find_closest_opposite()
    pub fn find_closest_opposite(&self) -> PyResult<((PyContourPoint, PyContourPoint), f64)> {
        let rust_contour = self.to_rust_contour()?;
        let ((p1, p2), distance) = rust_contour.find_closest_opposite();
        Ok((
            (PyContourPoint::from(&p1), PyContourPoint::from(&p2)),
            distance,
        ))
    }

    /// Get the elliptic ratio of the current contour.
    ///
    /// Returns
    /// -------
    /// float
    ///     Ratio of the farthest-points distance divided by the
    ///     closest-opposite-points distance.
    ///
    /// Examples
    /// --------
    /// >>> elliptic_ratio = contour.get_elliptic_ratio()
    pub fn get_elliptic_ratio(&self) -> PyResult<f64> {
        let rust_contour = self.to_rust_contour()?;
        Ok(rust_contour.elliptic_ratio())
    }

    /// Get the area of the current contour using the shoelace formula.
    ///
    /// Returns
    /// -------
    /// float
    ///     Area of the contour in the units of the original data (e.g. mm²).
    ///
    /// Examples
    /// --------
    /// >>> area = contour.get_area()
    pub fn get_area(&self) -> PyResult<f64> {
        let rust_contour = self.to_rust_contour()?;
        Ok(rust_contour.area())
    }

    /// Rotate the contour around its own centroid by an angle in degrees.
    ///
    /// Returns
    /// -------
    /// PyContour
    ///     New contour rotated around its centroid.
    ///
    /// Examples
    /// --------
    /// >>> contour = contour.rotate(20)
    #[pyo3(signature = (angle_deg))]
    pub fn rotate(&self, angle_deg: f64) -> PyResult<PyContour> {
        let angle_rad = angle_deg.to_radians();
        let mut rust_contour = self.to_rust_contour()?;
        rust_contour.compute_centroid();
        let (cx, cy, _) = rust_contour.centroid.unwrap_or((0.0, 0.0, 0.0));
        rust_contour.rotate_mut(angle_rad, (cx, cy));

        Ok(PyContour::from(&rust_contour))
    }

    /// Translate the contour by ``(dx, dy, dz)`` coordinates.
    ///
    /// Parameters
    /// ----------
    /// dx : float
    ///     Translation in the x-direction.
    /// dy : float
    ///     Translation in the y-direction.
    /// dz : float
    ///     Translation in the z-direction.
    ///
    /// Returns
    /// -------
    /// PyContour
    ///     New contour translated by ``(dx, dy, dz)``.
    ///
    /// Examples
    /// --------
    /// >>> contour = contour.translate((0.0, 1.0, 2.0))
    #[pyo3(signature = (dx, dy, dz))]
    pub fn translate(&self, dx: f64, dy: f64, dz: f64) -> PyResult<PyContour> {
        let rust_contour = self.to_rust_contour()?.translate(dx, dy, dz);

        Ok(PyContour::from(&rust_contour))
    }

    /// Sort points within the contour in counterclockwise order.
    ///
    /// The point with the highest y-coordinate receives index 0; all
    /// remaining points are ordered counterclockwise.
    ///
    /// Returns
    /// -------
    /// PyContour
    ///     New contour with rearranged point indices.
    ///
    /// Examples
    /// --------
    /// >>> contour = contour.sort_contour_points()
    pub fn sort_contour_points(&self) -> PyResult<PyContour> {
        let mut rust_contour = self.to_rust_contour()?;
        rust_contour.sort_contour_points();
        Ok(PyContour::from(&rust_contour))
    }
}

impl PyContour {
    pub fn to_rust_contour(&self) -> PyResult<Contour> {
        let points = self.points.iter().map(ContourPoint::from).collect();
        let kind = match self.kind.as_str() {
            "Lumen" => ContourType::Lumen,
            "Eem" => ContourType::Eem,
            "Calcification" => ContourType::Calcification,
            "Sidebranch" => ContourType::Sidebranch,
            "Catheter" => ContourType::Catheter,
            "Wall" => ContourType::Wall,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown contour type: {}",
                    self.kind
                )))
            }
        };

        Ok(Contour {
            id: self.id,
            original_frame: self.original_frame,
            points,
            centroid: Some(self.centroid),
            aortic_thickness: self.aortic_thickness,
            pulmonary_thickness: self.pulmonary_thickness,
            kind,
        })
    }
}

/// Python representation of the available intravascular contour types.
///
/// Examples
/// --------
/// >>> from multimodars import PyContourType
/// >>> contour_type = PyContourType.Lumen
/// >>> contour_type.name
/// 'Lumen'
#[pyclass(from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PyContourType {
    Lumen,
    Eem,
    Calcification,
    Sidebranch,
    Catheter,
    Wall,
}

#[pymethods]
impl PyContourType {
    #[new]
    fn new() -> Self {
        PyContourType::Lumen // Default to Lumen
    }

    /// Create from string name
    #[staticmethod]
    fn from_string(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "lumen" => Ok(PyContourType::Lumen),
            "eem" => Ok(PyContourType::Eem),
            "calcification" => Ok(PyContourType::Calcification),
            "sidebranch" => Ok(PyContourType::Sidebranch),
            "catheter" => Ok(PyContourType::Catheter),
            "wall" => Ok(PyContourType::Wall),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown contour type: '{name}'. Valid types are: lumen, eem, calcification, sidebranch, catheter, wall"
            ))),
        }
    }

    /// Get the string name of the contour type
    #[getter]
    fn name(&self) -> &'static str {
        match self {
            PyContourType::Lumen => "Lumen",
            PyContourType::Eem => "Eem",
            PyContourType::Calcification => "Calcification",
            PyContourType::Sidebranch => "Sidebranch",
            PyContourType::Catheter => "Catheter",
            PyContourType::Wall => "Wall",
        }
    }

    fn __repr__(&self) -> String {
        format!("PyContourType.{}", self.name())
    }

    fn __str__(&self) -> String {
        self.name().to_string()
    }

    /// Get all available contour types
    #[staticmethod]
    fn all_types() -> Vec<PyContourType> {
        vec![
            PyContourType::Lumen,
            PyContourType::Eem,
            PyContourType::Calcification,
            PyContourType::Sidebranch,
            PyContourType::Catheter,
            PyContourType::Wall,
        ]
    }
}

impl From<ContourType> for PyContourType {
    fn from(contour_type: ContourType) -> Self {
        match contour_type {
            ContourType::Lumen => PyContourType::Lumen,
            ContourType::Eem => PyContourType::Eem,
            ContourType::Calcification => PyContourType::Calcification,
            ContourType::Sidebranch => PyContourType::Sidebranch,
            ContourType::Catheter => PyContourType::Catheter,
            ContourType::Wall => PyContourType::Wall,
        }
    }
}

impl From<PyContourType> for ContourType {
    fn from(py_contour_type: PyContourType) -> Self {
        match py_contour_type {
            PyContourType::Lumen => ContourType::Lumen,
            PyContourType::Eem => ContourType::Eem,
            PyContourType::Calcification => ContourType::Calcification,
            PyContourType::Sidebranch => ContourType::Sidebranch,
            PyContourType::Catheter => ContourType::Catheter,
            PyContourType::Wall => ContourType::Wall,
        }
    }
}

impl From<&PyContourType> for ContourType {
    fn from(py_contour_type: &PyContourType) -> Self {
        (*py_contour_type).into()
    }
}

impl From<&Contour> for PyContour {
    fn from(contour: &Contour) -> Self {
        let kind_str = match contour.kind {
            ContourType::Lumen => "Lumen",
            ContourType::Eem => "Eem",
            ContourType::Calcification => "Calcification",
            ContourType::Sidebranch => "Sidebranch",
            ContourType::Catheter => "Catheter",
            ContourType::Wall => "Wall",
        }
        .to_string();

        PyContour {
            id: contour.id,
            original_frame: contour.original_frame,
            points: contour.points.iter().map(PyContourPoint::from).collect(),
            centroid: contour.centroid.unwrap_or((0.0, 0.0, 0.0)),
            aortic_thickness: contour.aortic_thickness,
            pulmonary_thickness: contour.pulmonary_thickness,
            kind: kind_str,
        }
    }
}
