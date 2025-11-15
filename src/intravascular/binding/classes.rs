use crate::intravascular::io::geometry::{Contour, ContourType, Frame, Geometry};
use crate::intravascular::io::input::{
    Centerline, CenterlinePoint, ContourPoint, InputData, Record,
};
use crate::intravascular::processing::align_between::GeometryPair;
use anyhow::{anyhow, Result};
use nalgebra::Vector3;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::convert::TryFrom;

/// Python representation of InputData
///
/// Attributes:
///    lumen (List[PyContour]): Vessel lumen contours
///    eem (List[PyContour] | None): Vessel EEM contours
///    calcification (List[PyContour] | None): Vessel calcification contours
///    sidebranch (List[PyContor] | None): Vessel sidebranch contours
///    record (PyRecord): Metadata about the input data
///    ref_point (PyContourPoint): Reference point for alignment
///    diastole (bool): Flag indicating if data is diastolic
///    label (str): label for the input data
/// Example:
///     >>> input_data = PyInputData(
///     ...     lumen=[lumen_contour1, lumen_contour2, ...],
///     ...     eem=[eem_contour1, eem_contour2, ...],
///     ...     calcification=[],
///     ...     sidebranch=[],
///     ...     record=record,
///     ...     diastole=True,
///     ...     lablel="Pat00_diastole_rest"
///     ... )
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyInputData {
    #[pyo3(get, set)]
    pub lumen: Vec<PyContour>,
    #[pyo3(get, set)]
    pub eem: Option<Vec<PyContour>>,
    #[pyo3(get, set)]
    pub calcification: Option<Vec<PyContour>>,
    #[pyo3(get, set)]
    pub sidebranch: Option<Vec<PyContour>>,
    #[pyo3(get, set)]
    pub record: Option<Vec<PyRecord>>,
    #[pyo3(get, set)]
    pub ref_point: PyContourPoint,
    #[pyo3(get, set)]
    pub diastole: bool,
    #[pyo3(get, set)]
    pub label: String,
}

#[pymethods]
impl PyInputData {
    #[new]
    fn new(
        lumen: Vec<PyContour>,
        eem: Option<Vec<PyContour>>,
        calcification: Option<Vec<PyContour>>,
        sidebranch: Option<Vec<PyContour>>,
        record: Option<Vec<PyRecord>>,
        ref_point: PyContourPoint,
        diastole: bool,
        label: String,
    ) -> Self {
        Self {
            lumen,
            eem,
            calcification,
            sidebranch,
            record,
            ref_point,
            diastole,
            label,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "InputData(lumen={}, eem={}, calcification={}, sidebranch={}, record={}, ref_point={}, diastole={}, label='{}')",
            self.lumen.len(),
            self.eem.as_ref().map_or(0, |v| v.len()),
            self.calcification.as_ref().map_or(0, |v| v.len()),
            self.sidebranch.as_ref().map_or(0, |v| v.len()),
            self.record.as_ref().map_or(0, |v| v.len()),
            self.ref_point.__repr__(),
            self.diastole,
            self.label,
        )
    }
}

impl TryFrom<&PyInputData> for InputData {
    type Error = anyhow::Error;

    fn try_from(py_in: &PyInputData) -> Result<Self> {
        // Flatten Vec<PyContour> -> Vec<ContourPoint>
        let flatten = |contours_opt: &Option<Vec<PyContour>>| -> Result<Option<Vec<ContourPoint>>> {
            if let Some(contours) = contours_opt {
                let mut acc: Vec<ContourPoint> = Vec::new();
                for c in contours {
                    // convert PyContour -> Contour (may return PyErr), use to_rust_contour()
                    let rust_contour = c.to_rust_contour().map_err(|e| {
                        anyhow!(
                            "failed to convert PyContour(id={}) to Contour: {:?}",
                            c.id,
                            e
                        )
                    })?;
                    acc.extend(rust_contour.points.into_iter());
                }
                Ok(Some(acc))
            } else {
                Ok(None)
            }
        };

        // Lumen (required)
        let mut lumen_points: Vec<ContourPoint> = Vec::new();
        for c in &py_in.lumen {
            let rust_contour = c.to_rust_contour().map_err(|e| {
                anyhow!(
                    "failed to convert lumen PyContour(id={}) to Contour: {:?}",
                    c.id,
                    e
                )
            })?;
            lumen_points.extend(rust_contour.points.into_iter());
        }

        let eem_points = flatten(&py_in.eem)?;
        let calc_points = flatten(&py_in.calcification)?;
        let sidebranch_points = flatten(&py_in.sidebranch)?;

        // Records: Option<Vec<PyRecord>> -> Option<Vec<Record>>
        let records_rust: Option<Vec<Record>> = match &py_in.record {
            Some(py_records) => {
                let mut out = Vec::with_capacity(py_records.len());
                for r in py_records {
                    out.push(r.to_rust_record());
                }
                Some(out)
            }
            None => None,
        };

        // ref_point: PyContourPoint -> ContourPoint
        let ref_point_rust: ContourPoint = ContourPoint::from(&py_in.ref_point);

        // Build InputData via its constructor to preserve validations
        InputData::new(
            lumen_points,
            eem_points,
            calc_points,
            sidebranch_points,
            records_rust,
            ref_point_rust,
            py_in.diastole,
            py_in.label.clone(),
        )
        .map_err(|e| anyhow!("InputData::new failed: {:?}", e))
    }
}

impl TryFrom<PyInputData> for InputData {
    type Error = anyhow::Error;

    fn try_from(py_in: PyInputData) -> Result<Self> {
        InputData::try_from(&py_in)
    }
}

impl From<&InputData> for PyInputData {
    fn from(input: &InputData) -> Self {
        // helper: build a single PyContour from a flattened Vec<ContourPoint>
        fn make_pycontour_from_points(
            points: &Vec<ContourPoint>,
            id: u32,
            original_frame: u32,
        ) -> PyContour {
            // compute centroid (average) if points non-empty
            let centroid = if points.is_empty() {
                (0.0, 0.0, 0.0)
            } else {
                let (sx, sy, sz) = points
                    .iter()
                    .fold((0.0f64, 0.0f64, 0.0f64), |(sx, sy, sz), p| {
                        (sx + p.x, sy + p.y, sz + p.z)
                    });
                let n = points.len() as f64;
                (sx / n, sy / n, sz / n)
            };

            let py_points: Vec<PyContourPoint> = points.iter().map(PyContourPoint::from).collect();

            PyContour {
                id,
                original_frame,
                points: py_points,
                centroid,
                aortic_thickness: None,
                pulmonary_thickness: None,
                kind: "Lumen".to_string(),
            }
        }

        // Build lumen as a single PyContour (id = 0)
        let original_frame = input.ref_point.frame_index;
        let lumen_py = make_pycontour_from_points(&input.lumen, 0, original_frame);
        let lumen_vec = vec![lumen_py];

        // Optional groups: wrap each existing flattened vec into a single PyContour (if present)
        let wrap_opt =
            |opt_pts: &Option<Vec<ContourPoint>>, id_start: u32| -> Option<Vec<PyContour>> {
                opt_pts
                    .as_ref()
                    .map(|pts| vec![make_pycontour_from_points(pts, id_start, original_frame)])
            };

        let eem_py = wrap_opt(&input.eem, 0);
        let calc_py = wrap_opt(&input.calcification, 0);
        let sb_py = wrap_opt(&input.sidebranch, 0);

        // Records
        let record_py: Option<Vec<PyRecord>> = input
            .record
            .as_ref()
            .map(|records| records.iter().map(|r| PyRecord::from(r)).collect());

        let ref_point_py = PyContourPoint::from(&input.ref_point);

        PyInputData {
            lumen: lumen_vec,
            eem: eem_py,
            calcification: calc_py,
            sidebranch: sb_py,
            record: record_py,
            ref_point: ref_point_py,
            diastole: input.diastole,
            label: input.label.clone(),
        }
    }
}

// Also provide owned conversion if you want to convert InputData (by value)
impl From<InputData> for PyInputData {
    fn from(input: InputData) -> Self {
        PyInputData::from(&input)
    }
}

/// Python representation of a 3D contour point
///
/// Attributes:
///     frame_index (int): Frame number in sequence
///     point_index (int): Index within contour
///     x (float): X-coordinate in mm
///     y (float): Y-coordinate in mm
///     z (float): Z-coordinate (depth) in mm
///     aortic (bool): Flag indicating aortic position (in case of intramural course)
///
/// Example:
///     >>> point = PyContourPoint(
///     ...     frame_index=0,
///     ...     point_index=1,
///     ...     x=1.23,
///     ...     y=4.56,
///     ...     z=7.89,
///     ...     aortic=True
///     ... )
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

    // Add a __repr__ method
    fn __repr__(&self) -> String {
        format!(
            "Point(frame_id={}, pt_id={}, x={:.2}, y={:.2}, z={:.2}, aortic={})",
            self.frame_index, self.point_index, self.x, self.y, self.z, self.aortic
        )
    }

    // Add a __str__ method for human-readable output
    fn __str__(&self) -> String {
        format!(
            "Point(frame_id={}, pt_id={}, x={:.2}, y={:.2}, z={:.2}, aortic={})",
            self.frame_index, self.point_index, self.x, self.y, self.z, self.aortic
        )
    }

    /// Euclidean distance to another PyContourPoint
    ///
    /// Args:
    ///     point (PyContourPoint): Any other PyContourPoint.
    ///
    /// Example:
    ///     >>> p1.distance(p2)
    pub fn distance(&self, other: &PyContourPoint) -> f64 {
        let p1: ContourPoint = ContourPoint::from(self);
        let p2: ContourPoint = ContourPoint::from(other);
        p1.distance_to(&p2)
    }
}

// Conversion from Rust type to Python type
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

// Conversion from Python type to Rust type
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

// Implement conversion for references
impl From<&&ContourPoint> for PyContourPoint {
    fn from(point: &&ContourPoint) -> Self {
        (*point).into()
    }
}

/// Python representation of a 3D contour
///
/// Attributes:
///     id (int): Contour number in sequence
///     points ([PyContourPoint]): Vector of ContourPoints
///     centroid (float, float, float): Tuple containing x-, y-, z-coordinates
///
/// Example:
///     >>> contour = PyContour(
///     ...     id=0,
///     ...     points=[point1, point2, ...],
///     ...     centroid=(1.0, 1.0, 1.0)
///     ... )
#[pyclass]
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
    /// Creates a new PyContour instance, automatically calculates centroid
    ///
    /// Args:
    ///     id (int): Contour identifier
    ///     points (List[PyContourPoint]): List of contour points
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

    fn __repr__(&self) -> String {
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

    /// Calculates the contours centroid by averaging over all coordinates
    ///
    /// Example:
    ///     >>> contour.compute_centroid()
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

    /// Returns contour points as list of (x, y, z) tuples
    ///
    /// Example:
    ///     >>> contour.points_as_tuples()
    ///     [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
    fn points_as_tuples(&self) -> Vec<(f64, f64, f64)> {
        self.points.iter().map(|p| (p.x, p.y, p.z)).collect()
    }

    /// Finds the two farthest points in the contour
    ///
    /// Returns:
    ///     Tuple[Tuple[PyContourPoint, PyContourPoint], float]:
    ///         Pair of points and their Euclidean distance
    ///
    /// Example:
    ///     >>> (p1, p2), distance = contour.find_farthest_points()
    pub fn find_farthest_points(&self) -> PyResult<((PyContourPoint, PyContourPoint), f64)> {
        let rust_contour = self.to_rust_contour()?;
        let ((p1, p2), distance) = rust_contour.find_farthest_points();
        Ok((
            (PyContourPoint::from(&p1), PyContourPoint::from(&p2)),
            distance,
        ))
    }

    /// Finds closest points on opposite sides of the contour
    ///
    /// Returns:
    ///     Tuple[Tuple[PyContourPoint, PyContourPoint], float]:
    ///         Pair of points and their Euclidean distance
    /// Example:
    ///     >>> (p1, p2), distance = contour.find_closest_opposite()
    pub fn find_closest_opposite(&self) -> PyResult<((PyContourPoint, PyContourPoint), f64)> {
        let rust_contour = self.to_rust_contour()?;
        let ((p1, p2), distance) = rust_contour.find_closest_opposite();
        Ok((
            (PyContourPoint::from(&p1), PyContourPoint::from(&p2)),
            distance,
        ))
    }

    /// Get the elliptic ratio of the current contour
    ///
    /// Returns:
    ///     float:
    ///         Ratio of farthest points distance divided by closest
    ///         opposite points distance.
    /// Example:
    ///     >>> elliptic_ratio = contour.get_elliptic_ratio()
    pub fn get_elliptic_ratio(&self) -> PyResult<f64> {
        let rust_contour = self.to_rust_contour()?;
        Ok(rust_contour.elliptic_ratio())
    }

    /// Get the area of the current contour using shoelace formula
    ///
    /// Returns:
    ///     float:
    ///         Area of the current contour in the unit that the original
    ///         contour data was provided (e.g. mm2).
    /// Example:
    ///     >>> area = contour.get_area()    
    pub fn get_area(&self) -> PyResult<f64> {
        let rust_contour = self.to_rust_contour()?;
        Ok(rust_contour.area())
    }

    /// Rotate a given contour around it's own centroid by an angle
    /// in degrees.
    ///
    /// Returns:
    ///     PyContour:
    ///         Original Contour rotated around it's centroid
    /// Example:
    ///     >>> contour = contour.rotate(20)
    #[pyo3(signature = (angle_deg))]
    pub fn rotate(&self, angle_deg: f64) -> PyResult<PyContour> {
        let angle_rad = angle_deg.to_radians();
        let mut rust_contour = self.to_rust_contour()?;
        rust_contour.rotate_contour(angle_rad);

        Ok(PyContour::from(&rust_contour))
    }

    /// translate a given contour by x, y, z coordinates
    ///
    /// Args:
    ///     dx (float): Translation in x-direction.
    ///     dy (float): Translation in y-direction.
    ///     dz (float): Translation in z-direction.
    ///
    /// Returns:
    ///     PyContour:
    ///         Original Contour translated to (x, y, z)
    /// Example:
    ///     >>> contour = contour.translate((0.0, 1.0, 2.0))
    #[pyo3(signature = (dx, dy, dz))]
    pub fn translate(&self, dx: f64, dy: f64, dz: f64) -> PyResult<PyContour> {
        let mut rust_contour = self.to_rust_contour()?;
        rust_contour.translate_contour((dx, dy, dz));

        Ok(PyContour::from(&rust_contour))
    }

    /// Sort points within a contour, so highest y-coord point
    /// has index 0 and all the others are sorted counterclockwise
    ///
    /// Returns:
    ///     PyContour:
    ///         Original Contour rearranged points.point_idx
    /// Example:
    ///     >>> contour = contour.sort_contour_points()
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

/// Python representation of contour types
///
/// Example:
///     >>> from multimodars import PyContourType
///     >>> contour_type = PyContourType.Lumen
///     >>> contour_type.name
///     'Lumen'
#[pyclass]
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
                "Unknown contour type: '{}'. Valid types are: lumen, eem, calcification, sidebranch, catheter, wall",
                name
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

// Conversion between Rust ContourType and PyContourType
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

// Also implement for references
impl From<&PyContourType> for ContourType {
    fn from(py_contour_type: &PyContourType) -> Self {
        (*py_contour_type).into()
    }
}

#[pyclass]
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

/// Python representation of a full geometry set
///
/// Contains:
///     - Vessel contours
///     - Catheter points
///     - Wall contours
///     - Reference point
///
/// Attributes:
///     id /int): Frame id
///     centroid ((float, float, float)): (x, y, z)
///     extras (Dict[str, PyContour]): "Eem", "Calcification", "Sidebranch", "Catheter", "Wall"
///     reference_point (PyContourPoint): Reference position
///
/// Example:
///     >>> geom = PyFrame(
///     ...     id=0,
///     ...     centroid=(0.0, 0.0, 0.0),
///     ...     lumen=lumen_contour,
///     ...     extras={"Eem": eem_contour}
///     ...     reference_point=ref_point
///     ... )
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

    #[pyo3(signature = (angle_deg))]
    pub fn rotate(&self, angle_deg: f64) -> PyResult<PyFrame> {
        let mut rust_frame = self.to_rust_frame()?;
        rust_frame.rotate_frame(angle_deg.to_radians());
        Ok(PyFrame::from(&rust_frame))
    }

    #[pyo3(signature = (dx, dy, dz))]
    pub fn translate(&self, dx: f64, dy: f64, dz: f64) -> PyResult<PyFrame> {
        let mut rust_frame = self.to_rust_frame()?;
        rust_frame.translate_frame((dx, dy, dz));
        Ok(PyFrame::from(&rust_frame))
    }

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

/// Python representation of a full geometry set
///
/// Contains:
///     - Vessel contours
///     - Catheter points
///     - Wall contours
///     - Reference point
///
/// Attributes:
///     contours (List[PyContour]): Vessel contours
///     catheter (List[PyContour]): Catheter points
///     walls (List[PyContour]): Wall contours
///     reference_point (PyContourPoint): Reference position
///
/// Example:
///     >>> geom = PyGeometry(
///     ...     contours=[contour1, contour2],
///     ...     catheter=[catheter_points],
///     ...     walls=[wall1, wall2],
///     ...     reference_point=ref_point
///     ... )
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyGeometry {
    #[pyo3(get, set)]
    pub frames: Vec<PyFrame>,
    #[pyo3(get, set)]
    pub label: String,
}

#[pymethods]
impl PyGeometry {
    /// Creates a new PyGeometry instance
    ///
    /// Args:
    ///     contours (List[PyContour]): Vessel contours
    ///     catheter (List[PyContour]): Catheter points
    ///     walls (List[PyContour]): Wall contours
    ///     reference_point (PyContourPoint): Reference position
    #[new]
    fn new(frames: Vec<PyFrame>, label: String) -> Self {
        Self { frames, label }
    }

    fn __repr__(&self) -> String {
        format!(
            "Geometry({} frames, label='{}')",
            self.frames.len(),
            self.label
        )
    }

    fn __len__(&self) -> usize {
        self.frames.len()
    }

    /// Get all contours of a specific type
    pub fn get_contours_by_type(&self, contour_type: &str) -> Vec<PyContour> {
        let target_type = match contour_type {
            "Lumen" => ContourType::Lumen,
            "Eem" => ContourType::Eem,
            "Calcification" => ContourType::Calcification,
            "Sidebranch" => ContourType::Sidebranch,
            "Catheter" => ContourType::Catheter,
            "Wall" => ContourType::Wall,
            _ => return Vec::new(),
        };

        self.frames
            .iter()
            .filter_map(|frame| {
                if target_type == ContourType::Lumen {
                    Some(frame.lumen.clone())
                } else {
                    let type_str = match target_type {
                        ContourType::Eem => "Eem",
                        ContourType::Calcification => "Calcification",
                        ContourType::Sidebranch => "Sidebranch",
                        ContourType::Catheter => "Catheter",
                        ContourType::Wall => "Wall",
                        _ => return None,
                    };
                    frame.extras.get(type_str).cloned()
                }
            })
            .collect()
    }

    /// Get lumen contours (convenience method)
    pub fn get_lumen_contours(&self) -> Vec<PyContour> {
        self.frames.iter().map(|f| f.lumen.clone()).collect()
    }

    /// Get contours by type string
    pub fn get_contours(&self, contour_type: &str) -> Vec<PyContour> {
        self.get_contours_by_type(contour_type)
    }

    /// Rotate all contours/walls/catheters of a given geometry
    /// around it's own centroid by an angle in degrees. Catheters are rotated
    /// around the same centroid as contour.
    ///
    /// Returns:
    ///     PyGeometry:
    ///         Original Geometry rotated around it's centroid
    /// Example:
    ///     >>> geometry = geometry.rotate(20)
    #[pyo3(signature = (angle_deg))]
    pub fn rotate(&self, angle_deg: f64) -> PyResult<PyGeometry> {
        let mut rust_geometry = self.to_rust_geometry()?;
        for frame in &mut rust_geometry.frames {
            frame.rotate_frame(angle_deg.to_radians());
        }
        Ok(PyGeometry::from(&rust_geometry))
    }

    /// Translates all contours, walls, and catheters in a geometry by (dx, dy, dz).
    ///
    /// Args:
    ///     dx (float): translation in x-direction.
    ///     dy (float): translation in y-direction.
    ///     dz (float): translation in z-direction.
    ///
    /// Returns:
    ///     A new PyGeometry with all elements translated.
    #[pyo3(signature = (dx, dy, dz))]
    pub fn translate(&self, dx: f64, dy: f64, dz: f64) -> PyResult<PyGeometry> {
        let mut rust_geometry = self.to_rust_geometry()?;
        for frame in &mut rust_geometry.frames {
            frame.translate_frame((dx, dy, dz));
        }
        Ok(PyGeometry::from(&rust_geometry))
    }

    /// Applies smoothing to all contours using a threepoint moving average
    ///
    /// Example:
    ///     >>> geom.smooth_frames()
    pub fn smooth_frames(&self) -> PyResult<PyGeometry> {
        let rust_geometry = self.to_rust_geometry()?;
        let smoothed = rust_geometry.smooth_frames();
        Ok(PyGeometry::from(&smoothed))
    }

    /// Get a compact summary of lumen properties for this geometry.
    ///
    /// Returns:
    ///     tuple: (mla, max_stenosis, stenosis_length_mm)
    ///         mla (float): minimal lumen area (same units as contour.area(), e.g. mm^2)
    ///         max_stenosis (float): 1 - (mla / biggest_area)
    ///         stenosis_length_mm (float): length (in mm) of the longest contiguous region
    ///         where contour area < threshold.
    ///
    /// Threshold logic (implemented by assumption):
    ///     If ALL contours have elliptic_ratio < 1.3 we treat the vessel as "elliptic"
    ///     and use a more lenient threshold of 0.70 * biggest_area.
    ///     Otherwise we use a stricter threshold of 0.50 * biggest_area (50%).
    pub fn get_summary(&self) -> PyResult<(f64, f64, f64)> {
        let geometry = self.to_rust_geometry()?;

        if geometry.frames.is_empty() {
            return Ok((0.0, 0.0, 0.0));
        }

        // Compute areas for all lumen contours
        let areas: Vec<f64> = geometry.frames.iter().map(|f| f.lumen.area()).collect();

        let biggest = areas.iter().cloned().fold(f64::NAN, f64::max);
        let mla = areas.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_stenosis = if biggest > 0.0 {
            1.0 - (mla / biggest)
        } else {
            0.0
        };

        // Compute elliptic ratios to decide threshold
        let all_elliptic = geometry
            .frames
            .iter()
            .all(|f| f.lumen.elliptic_ratio() < 1.3);

        let threshold = if all_elliptic {
            0.70 * biggest
        } else {
            0.50 * biggest
        };

        // Compute stenosis length using frame centroids
        let centroids: Vec<(f64, f64, f64)> = geometry.frames.iter().map(|f| f.centroid).collect();

        let mut longest_mm = 0.0;
        let mut i = 0;
        while i < areas.len() {
            if areas[i] < threshold {
                let start = i;
                let mut end = i;
                while end + 1 < areas.len() && areas[end + 1] < threshold {
                    end += 1;
                }

                let mut run_len = 0.0;
                for k in start..end {
                    let a = centroids[k];
                    let b = centroids[k + 1];
                    let dx = a.0 - b.0;
                    let dy = a.1 - b.1;
                    let dz = a.2 - b.2;
                    run_len += (dx * dx + dy * dy + dz * dz).sqrt();
                }

                if run_len > longest_mm {
                    longest_mm = run_len;
                }
                i = end + 1;
            } else {
                i += 1;
            }
        }

        Ok((mla, max_stenosis, longest_mm))
    }

    /// Centers the entire geometry to a specific contour type
    ///
    /// Args:
    ///     contour_type (str): Type of contour to center on ("Lumen", "Eem", "Wall", etc.)
    ///
    /// Returns:
    ///     PyGeometry: A new geometry centered on the specified contour type
    #[pyo3(signature = (contour_type))]
    pub fn center_to_contour(&self, contour_type: PyContourType) -> PyResult<PyGeometry> {
        let rust_contour_type: crate::intravascular::io::geometry::ContourType =
            contour_type.into();

        let mut rust_geometry = self.to_rust_geometry()?;
        rust_geometry.center_to_contour(rust_contour_type);
        Ok(PyGeometry::from(&rust_geometry))
    }
}

impl PyGeometry {
    pub fn to_rust_geometry(&self) -> PyResult<Geometry> {
        let mut frames = Vec::new();
        for py_frame in &self.frames {
            frames.push(py_frame.to_rust_frame()?);
        }

        Ok(Geometry {
            frames,
            label: self.label.clone(),
        })
    }
}

/// Python representation of a diastolic/systolic geometry pair
///
/// Attributes:
///     dia_geom (PyGeometry): Diastolic geometry
///     sys_geom (PyGeometry): Systolic geometry
///
/// Example:
///     >>> pair = PyGeometryPair(
///     ...     dia_geom=diastole,
///     ...     sys_geom=systole
///     ... )
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyGeometryPair {
    #[pyo3(get, set)]
    pub geom_a: PyGeometry,
    #[pyo3(get, set)]
    pub geom_b: PyGeometry,
    #[pyo3(get, set)]
    pub label: String,
}

#[pymethods]
impl PyGeometryPair {
    #[new]
    fn new(geom_a: PyGeometry, geom_b: PyGeometry, label: String) -> Self {
        Self {
            geom_a,
            geom_b,
            label,
        }
    }

    // Add a __repr__ method
    fn __repr__(&self) -> String {
        format!(
            "GeometryPair {} (diastolic: {} frames, systolic: {} frames)",
            self.label,
            self.geom_a.frames.len(),
            self.geom_b.frames.len()
        )
    }

    /// Get summaries for both diastolic and systolic geometries.
    ///
    /// Returns a tuple: ((dia_mla, dia_max_stenosis, dia_len_mm), (sys_mla, sys_max_stenosis, sys_len_mm))
    /// and a matrix (N, 6): (contour id, area_dia, ellip_dia, area_sys, ellip_sys, z-coordinate)
    ///
    /// This calls ``get_summary()`` on each contained PyGeometry and returns both results.
    /// and additionally assesses dynamic between the two PyGeometry object (area, elliptic ratio)
    pub fn get_summary(&self) -> PyResult<(((f64, f64, f64), (f64, f64, f64)), Vec<[f64; 6]>)> {
        let dia = self.geom_a.get_summary()?;
        let sys = self.geom_b.get_summary()?;
        let map = self.create_deformation_table();
        Ok(((dia, sys), map))
    }

    fn create_deformation_table(&self) -> Vec<[f64; 6]> {
        let dia_lumen = self.geom_a.get_lumen_contours();
        let sys_lumen = self.geom_b.get_lumen_contours();

        let areas_dia: Vec<f64> = dia_lumen.iter().map(|c| c.get_area().unwrap()).collect();
        let areas_sys: Vec<f64> = sys_lumen.iter().map(|c| c.get_area().unwrap()).collect();

        let ellip_dia: Vec<f64> = dia_lumen
            .iter()
            .map(|c| c.get_elliptic_ratio().unwrap())
            .collect();
        let ellip_sys: Vec<f64> = sys_lumen
            .iter()
            .map(|c| c.get_elliptic_ratio().unwrap())
            .collect();

        let ids: Vec<u32> = dia_lumen.iter().map(|c| c.id).collect();
        let z_coords: Vec<f64> = dia_lumen.iter().map(|c| c.centroid.2).collect();

        // Ensure all vectors have same length
        let n = ids.len();
        if areas_dia.len() != n
            || ellip_dia.len() != n
            || areas_sys.len() != n
            || ellip_sys.len() != n
            || z_coords.len() != n
        {
            eprintln!("ERROR: mismatched lengths between contour vectors");
        }

        // Build numeric matrix: each row is [id, area_dia, ellip_dia, area_sys, ellip_sys, z]
        let mut mat: Vec<[f64; 6]> = Vec::with_capacity(n);
        for i in 0..n {
            mat.push([
                ids[i] as f64,
                areas_dia[i],
                ellip_dia[i],
                areas_sys[i],
                ellip_sys[i],
                z_coords[i],
            ]);
        }

        // Prepare printable rows (format floats to 6 decimal places)
        let headers = ["id", "area_dia", "ellip_dia", "area_sys", "ellip_sys", "z"];
        let rows: Vec<[String; 6]> = (0..n)
            .map(|i| {
                [
                    ids[i].to_string(),          // id as integer
                    format!("{:.2}", mat[i][1]), // area_dia
                    format!("{:.2}", mat[i][2]), // ellip_dia
                    format!("{:.2}", mat[i][3]), // area_sys
                    format!("{:.2}", mat[i][4]), // ellip_sys
                    format!("{:.2}", mat[i][5]), // z
                ]
            })
            .collect();

        // Compute max width for each of the 6 columns
        let mut widths = [0usize; 6];
        for (i, &h) in headers.iter().enumerate() {
            widths[i] = h.len();
        }
        for row in &rows {
            for (i, cell) in row.iter().enumerate() {
                widths[i] = widths[i].max(cell.len());
            }
        }

        // Print a left-aligned data row (same style as your dump_table)
        fn print_row(cells: &[String], widths: &[usize]) {
            print!("|");
            for (i, cell) in cells.iter().enumerate() {
                let pad = widths[i] - cell.len();
                print!(" {}{} |", cell, " ".repeat(pad));
            }
            println!();
        }

        // Print a centered header row
        fn print_header(cells: &[String], widths: &[usize]) {
            print!("|");
            for (i, cell) in cells.iter().enumerate() {
                let total_pad = widths[i] - cell.len();
                let left = total_pad / 2;
                let right = total_pad - left;
                print!(" {}{}{} |", " ".repeat(left), cell, " ".repeat(right));
            }
            println!();
        }

        // Top border
        print!("+");
        for w in &widths {
            print!("{}+", "-".repeat(w + 2));
        }
        println!();

        // Header row
        let header_cells: Vec<String> = headers.iter().map(|&s| s.to_string()).collect();
        print_header(&header_cells, &widths);

        // Separator
        print!("+");
        for w in &widths {
            print!("{}+", "-".repeat(w + 2));
        }
        println!();

        // Data rows
        for row in &rows {
            print_row(&row.to_vec(), &widths);
        }

        // Bottom border
        print!("+");
        for w in &widths {
            print!("{}+", "-".repeat(w + 2));
        }
        println!();

        mat
    }
}

impl PyGeometryPair {
    pub fn to_rust_geometry_pair(&self) -> GeometryPair {
        GeometryPair {
            geom_a: self
                .geom_a
                .to_rust_geometry()
                .expect("could not convert geom_a"),
            geom_b: self
                .geom_b
                .to_rust_geometry()
                .expect("could not convert geom_b"),
            label: self.label.clone(),
        }
    }
}

/// Python representation of a centerline point
///
/// Combines a contour point with its normal vector
///
/// Attributes:
///     contour_point (PyContourPoint): Position in 3D space
///     normal (Tuple[float, float, float]): Normal vector (nx, ny, nz)
///
/// Example:
///     >>> cl_point = PyCenterlinePoint(
///     ...     contour_point=point,
///     ...     normal=(0.0, 1.0, 0.0)
///     ... )
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyCenterlinePoint {
    #[pyo3(get, set)]
    pub contour_point: PyContourPoint,
    #[pyo3(get, set)]
    pub normal: (f64, f64, f64),
}

#[pymethods]
impl PyCenterlinePoint {
    #[new]
    fn new(contour_point: PyContourPoint, normal: (f64, f64, f64)) -> Self {
        Self {
            contour_point,
            normal,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "CenterlinePoint(point={}, normal=({:.3}, {:.3}, {:.3}))",
            self.contour_point.__repr__(),
            self.normal.0,
            self.normal.1,
            self.normal.2
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
            normal: (p.normal[0], p.normal[1], p.normal[2]),
        }
    }
}

// Conversion from PyCenterlinePoint to Rust CenterlinePoint
impl From<&PyCenterlinePoint> for CenterlinePoint {
    fn from(p: &PyCenterlinePoint) -> Self {
        CenterlinePoint {
            contour_point: ContourPoint::from(&p.contour_point),
            normal: Vector3::new(p.normal.0, p.normal.1, p.normal.2),
        }
    }
}

/// Python representation of a vessel centerline
///
/// Attributes:
///     points (List[PyCenterlinePoint]): Ordered points along centerline
///
/// Example:
///     >>> centerline = PyCenterline(points=[p1, p2, p3])
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyCenterline {
    #[pyo3(get, set)]
    pub points: Vec<PyCenterlinePoint>,
}

#[pymethods]
impl PyCenterline {
    #[new]
    fn new(points: Vec<PyCenterlinePoint>) -> Self {
        Self { points }
    }

    /// Build a Centerline from a flat list of PyContourPoint.
    ///
    /// Args:
    ///     contour_points (List[PyContourPoint]): sequence of points in order.
    ///
    /// Returns:
    ///     PyCenterline
    ///
    /// Example:
    ///     >>> pts = [PyContourPoint(...), PyContourPoint(...), ...]
    ///     >>> cl = PyCenterline.from_contour_points(pts)
    #[staticmethod]
    fn from_contour_points(contour_points: Vec<PyContourPoint>) -> PyResult<Self> {
        // convert Python points → Rust ContourPoint
        let rust_pts: Vec<ContourPoint> = contour_points.iter().map(|p| p.into()).collect();

        // call your existing Rust constructor
        let rust_cl = Centerline::from_contour_points(rust_pts);

        // use your From<&Centerline> impl to go back into PyCenterline
        Ok(PyCenterline::from(&rust_cl))
    }

    fn __repr__(&self) -> String {
        format!(
            "Centerline(len={}, spacing={:.2} mm)",
            self.points.len(),
            self._spacing()
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn __len__(&self) -> usize {
        self.points.len()
    }

    fn _spacing(&self) -> f64 {
        // if fewer than 2 points there is no spacing
        if self.points.len() < 2 {
            return 0.0;
        }

        // sum distances between consecutive contour points
        let mut total: f64 = 0.0;
        let mut count: usize = 0;
        for pair in self.points.windows(2) {
            let a = &pair[0].contour_point;
            let b = &pair[1].contour_point;

            let dx = a.x - b.x;
            let dy = a.y - b.y;
            let dz = a.z - b.z;
            total += (dx * dx + dy * dy + dz * dz).sqrt();
            count += 1;
        }

        total / (count as f64)
    }

    fn points_as_tuples(&self) -> Vec<(f64, f64, f64)> {
        self.points
            .iter()
            .map(|p| (p.contour_point.x, p.contour_point.y, p.contour_point.z))
            .collect()
    }
}

// Moved out of pymethods since it's for internal use
impl PyCenterline {
    pub fn to_rust_centerline(&self) -> Centerline {
        Centerline {
            points: self.points.iter().map(|p| p.into()).collect(),
        }
    }
}

// Conversion from Python to Rust for entire back-and-forth
impl From<&Centerline> for PyCenterline {
    fn from(cl: &Centerline) -> Self {
        let points = cl.points.iter().map(|p| p.into()).collect();
        PyCenterline { points }
    }
}

// Conversion from Python to Rust for entire back-and-forth
impl From<Centerline> for PyCenterline {
    fn from(cl: Centerline) -> Self {
        let points = cl.points.iter().map(|p| p.into()).collect();
        PyCenterline { points }
    }
}

/// Python representation of a measurement record
///
/// Attributes:
///     frame (int): Frame number
///     phase (str): Cardiac phase ('D'/'S') for diastole or systole
///     measurement_1 (float, optional): Primary measurement. In coronary artery anomalies thickness between aorta and coronary.
///     measurement_2 (float, optional): Secondary measurement. In coronary artery anomalies thickness between pulmonary artery and coronary.
///
/// Example:
///     >>> record = PyRecord(
///     ...     frame=5,
///     ...     phase="D",
///     ...     measurement_1=1.4,
///     ...     measurement_2=2.1
///     ... )
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyRecord {
    #[pyo3(get, set)]
    pub frame: u32,
    #[pyo3(get, set)]
    pub phase: String,
    #[pyo3(get, set)]
    pub measurement_1: Option<f64>,
    #[pyo3(get, set)]
    pub measurement_2: Option<f64>,
}

#[pymethods]
impl PyRecord {
    /// Python constructor
    #[new]
    fn new(
        frame: u32,
        phase: String,
        measurement_1: Option<f64>,
        measurement_2: Option<f64>,
    ) -> Self {
        PyRecord {
            frame,
            phase,
            measurement_1,
            measurement_2,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Record(frame={}, phase={}, m1={:?}, m2={:?})",
            self.frame, self.phase, self.measurement_1, self.measurement_2
        )
    }
}

// Convert PyRecord → Record (for passing into your Rust core)
impl PyRecord {
    pub fn to_rust_record(&self) -> Record {
        Record {
            frame: self.frame,
            phase: self.phase.clone(),
            measurement_1: self.measurement_1,
            measurement_2: self.measurement_2,
        }
    }
}

// Convert &Record → PyRecord (for returning back out)
impl From<&Record> for PyRecord {
    fn from(r: &Record) -> Self {
        PyRecord {
            frame: r.frame,
            phase: r.phase.clone(),
            measurement_1: r.measurement_1,
            measurement_2: r.measurement_2,
        }
    }
}

// Conversion from Rust to Python types
impl From<&Geometry> for PyGeometry {
    fn from(geometry: &Geometry) -> Self {
        PyGeometry {
            frames: geometry.frames.iter().map(PyFrame::from).collect(),
            label: geometry.label.clone(),
        }
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

impl From<Geometry> for PyGeometry {
    fn from(geometry: Geometry) -> Self {
        PyGeometry::from(&geometry)
    }
}

impl From<GeometryPair> for PyGeometryPair {
    fn from(pair: GeometryPair) -> Self {
        PyGeometryPair {
            geom_a: pair.geom_a.into(),
            geom_b: pair.geom_b.into(),
            label: pair.label.clone(),
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
