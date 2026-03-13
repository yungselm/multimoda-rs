use crate::intravascular::io::geometry::{Contour, ContourType, Frame, Geometry};
use crate::intravascular::io::input::{
    Centerline, CenterlinePoint, ContourPoint, InputData, Record,
};
use crate::intravascular::processing::align_between::GeometryPair;
use crate::intravascular::processing::process_utils::downsample_contour_points;
use anyhow::{anyhow, Result};
use nalgebra::Vector3;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::convert::TryFrom;

/// Python representation of the intravascular imaging input data for one cardiac phase.
///
/// Attributes
/// ----------
/// lumen : list of PyContour
///     Vessel lumen contours.
/// eem : list of PyContour or None
///     Vessel EEM (external elastic membrane) contours.
/// calcification : list of PyContour or None
///     Vessel calcification contours.
/// sidebranch : list of PyContour or None
///     Vessel sidebranch contours.
/// record : list of PyRecord or None
///     Metadata records about the input data.
/// ref_point : PyContourPoint
///     Reference point used for alignment.
/// diastole : bool
///     ``True`` when the data corresponds to the diastolic phase.
/// label : str
///     Human-readable label for this input dataset.
///
/// Examples
/// --------
/// >>> input_data = PyInputData(
/// ...     lumen=[lumen_contour1, lumen_contour2, ...],
/// ...     eem=[eem_contour1, eem_contour2, ...],
/// ...     calcification=[],
/// ...     sidebranch=[],
/// ...     record=record,
/// ...     diastole=True,
/// ...     lablel="Pat00_diastole_rest"
/// ... )
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

impl From<InputData> for PyInputData {
    fn from(input: InputData) -> Self {
        PyInputData::from(&input)
    }
}

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

    fn __repr__(&self) -> String {
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
        rust_contour.rotate_contour(angle_rad);

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
        let mut rust_contour = self.to_rust_contour()?;
        rust_contour.translate_contour((dx, dy, dz));

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
        let mut rust_frame = self.to_rust_frame()?;
        rust_frame.rotate_frame(angle_deg.to_radians());
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
        let mut rust_frame = self.to_rust_frame()?;
        rust_frame.translate_frame((dx, dy, dz));
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

/// Python representation of a full intravascular imaging geometry (sequence of frames).
///
/// Attributes
/// ----------
/// frames : list of PyFrame
///     Ordered list of imaging frames constituting the geometry.
/// label : str
///     Human-readable label for this geometry.
///
/// Examples
/// --------
/// >>> geom = PyGeometry(
/// ...     frames=[frame1, frame2, ...],
/// ...     label="Pat00_diastole"
/// ... )
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
    /// Create a new PyGeometry instance.
    ///
    /// Parameters
    /// ----------
    /// frames : list of PyFrame
    ///     Ordered list of imaging frames.
    /// label : str
    ///     Human-readable label for this geometry.
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

    /// Rotate the entire geometry around its centroid by an angle in degrees.
    ///
    /// All frames (lumen, extras) are rotated around the same centroid.
    ///
    /// Returns
    /// -------
    /// PyGeometry
    ///     New geometry rotated around its centroid.
    ///
    /// Examples
    /// --------
    /// >>> geometry = geometry.rotate(20)
    #[pyo3(signature = (angle_deg))]
    pub fn rotate(&self, angle_deg: f64) -> PyResult<PyGeometry> {
        let mut rust_geometry = self.to_rust_geometry()?;
        for frame in &mut rust_geometry.frames {
            frame.rotate_frame(angle_deg.to_radians());
        }
        Ok(PyGeometry::from(&rust_geometry))
    }

    /// Translate all frames in the geometry by ``(dx, dy, dz)``.
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
    /// PyGeometry
    ///     New geometry with all frames translated.
    #[pyo3(signature = (dx, dy, dz))]
    pub fn translate(&self, dx: f64, dy: f64, dz: f64) -> PyResult<PyGeometry> {
        let mut rust_geometry = self.to_rust_geometry()?;
        for frame in &mut rust_geometry.frames {
            frame.translate_frame((dx, dy, dz));
        }
        Ok(PyGeometry::from(&rust_geometry))
    }

    /// Re-index all frame contour points so the point with the highest Z-value
    /// in frame 0's lumen gets ``point_index = 0``.  The same index offset is
    /// applied to every contour in every frame.  Physical point positions are
    /// unchanged — only the ``point_index`` fields are reassigned.
    ///
    /// Returns
    /// -------
    /// PyGeometry
    ///     New geometry with re-indexed frames.
    pub fn sort_frame_points(&self) -> PyResult<PyGeometry> {
        let mut rust_geometry = self.to_rust_geometry()?;
        rust_geometry.sort_frame_points_by_z();
        Ok(PyGeometry::from(&rust_geometry))
    }

    /// Apply smoothing to all frames using a three-point moving average.
    ///
    /// Returns
    /// -------
    /// PyGeometry
    ///     New geometry with smoothed frames.
    ///
    /// Examples
    /// --------
    /// >>> geom.smooth_frames()
    pub fn smooth_frames(&self) -> PyResult<PyGeometry> {
        let rust_geometry = self.to_rust_geometry()?;
        let smoothed = rust_geometry.smooth_frames();
        Ok(PyGeometry::from(&smoothed))
    }

    /// Get a compact summary of lumen properties for this geometry.
    ///
    /// When all contours have an elliptic ratio below 1.3 the vessel is
    /// treated as elliptic and a lenient threshold of 70 % of the maximum
    /// area is used to identify stenotic segments; otherwise a stricter
    /// 50 % threshold is applied.
    ///
    /// Returns
    /// -------
    /// mla : float
    ///     Minimal lumen area in the units of the input data (e.g. mm²).
    /// max_stenosis : float
    ///     ``1 - (mla / max_area)``.
    /// stenosis_length_mm : float
    ///     Length in mm of the longest contiguous region where the contour
    ///     area falls below the threshold.
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

    /// Center the entire geometry on a specific contour type.
    ///
    /// Parameters
    /// ----------
    /// contour_type : PyContourType
    ///     Contour type to center on (e.g. ``PyContourType.Lumen``).
    ///
    /// Returns
    /// -------
    /// PyGeometry
    ///     New geometry centered on the specified contour type.
    #[pyo3(signature = (contour_type))]
    pub fn center_to_contour(&self, contour_type: PyContourType) -> PyResult<PyGeometry> {
        let rust_contour_type: crate::intravascular::io::geometry::ContourType =
            contour_type.into();

        let mut rust_geometry = self.to_rust_geometry()?;
        rust_geometry.center_to_contour(rust_contour_type);
        Ok(PyGeometry::from(&rust_geometry))
    }

    /// Return the frame whose centroid z-coordinate is closest to ``z``.
    ///
    /// Parameters
    /// ----------
    /// z : float
    ///     Target z position in the same units as the geometry.
    ///
    /// Returns
    /// -------
    /// PyFrame
    ///     Frame with centroid z nearest to ``z``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the geometry contains no frames.
    ///
    /// Examples
    /// --------
    /// >>> frame = geometry.get_frame_at_z(34.8)
    #[pyo3(signature = (z))]
    pub fn get_frame_at_z(&self, z: f64) -> PyResult<PyFrame> {
        self.frames
            .iter()
            .min_by(|a, b| {
                let da = (a.centroid.2 - z).abs();
                let db = (b.centroid.2 - z).abs();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("geometry contains no frames"))
    }

    /// Return the frame at position ``index``.
    ///
    /// Parameters
    /// ----------
    /// index : int
    ///     Zero-based index of the frame to retrieve.
    ///
    /// Returns
    /// -------
    /// PyFrame
    ///     Frame at the given index.
    ///
    /// Raises
    /// ------
    /// IndexError
    ///     If ``index`` is out of range.
    ///
    /// Examples
    /// --------
    /// >>> frame = geometry.get_frame_at_index(0)
    #[pyo3(signature = (index))]
    pub fn get_frame_at_index(&self, index: usize) -> PyResult<PyFrame> {
        self.frames
            .get(index)
            .cloned()
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err(format!(
                "index {} out of range for geometry with {} frames",
                index,
                self.frames.len()
            )))
    }

    /// Return a new geometry with the frame at ``index`` replaced by ``frame``.
    ///
    /// Parameters
    /// ----------
    /// index : int
    ///     Zero-based index of the frame to replace.
    /// frame : PyFrame
    ///     Replacement frame.
    ///
    /// Returns
    /// -------
    /// PyGeometry
    ///     New geometry with the specified frame replaced.
    ///
    /// Raises
    /// ------
    /// IndexError
    ///     If ``index`` is out of range.
    ///
    /// Examples
    /// --------
    /// >>> new_geom = geometry.replace_frame(5, other_frame)
    #[pyo3(signature = (index, frame))]
    pub fn replace_frame(&self, index: usize, frame: PyFrame) -> PyResult<PyGeometry> {
        if index >= self.frames.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "index {} is out of range for geometry with {} frames",
                index,
                self.frames.len()
            )));
        }
        let mut new_frames = self.frames.clone();
        new_frames[index] = frame;
        Ok(PyGeometry {
            frames: new_frames,
            label: self.label.clone(),
        })
    }

    /// Return a new geometry with ``n_points`` per ContourType.
    ///
    /// Parameters
    /// ----------
    /// n_points : int
    ///     Number of points remaining per Contour.
    ///
    /// Returns
    /// -------
    /// PyGeometry
    ///     New downsampled geometry.
    ///
    /// Examples
    /// --------
    /// >>> new_geom = geometry.downsample(100)
    #[pyo3(signature = (n_points))]
    pub fn downsample(&self, n_points: usize) -> PyResult<PyGeometry> {
        let downsample_contour = |contour: &PyContour| -> PyContour {
            let rust_points: Vec<ContourPoint> =
                contour.points.iter().map(ContourPoint::from).collect();
            let downsampled = downsample_contour_points(&rust_points, n_points);
            PyContour {
                points: downsampled.iter().map(PyContourPoint::from).collect(),
                ..contour.clone()
            }
        };

        let new_frames = self
            .frames
            .iter()
            .map(|frame| {
                let new_lumen = downsample_contour(&frame.lumen);
                let new_extras = frame
                    .extras
                    .iter()
                    .map(|(key, contour)| {
                        if key == "Catheter" {
                            (key.clone(), contour.clone())
                        } else {
                            (key.clone(), downsample_contour(contour))
                        }
                    })
                    .collect();
                PyFrame {
                    lumen: new_lumen,
                    extras: new_extras,
                    ..frame.clone()
                }
            })
            .collect();

        Ok(PyGeometry {
            frames: new_frames,
            label: self.label.clone(),
        })
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

/// Python representation of a diastolic/systolic geometry pair.
///
/// Attributes
/// ----------
/// geom_a : PyGeometry
///     First geometry (typically diastolic).
/// geom_b : PyGeometry
///     Second geometry (typically systolic).
/// label : str
///     Human-readable label for this geometry pair.
///
/// Examples
/// --------
/// >>> pair = PyGeometryPair(
/// ...     geom_a=diastole,
/// ...     geom_b=systole,
/// ...     label="Pat00_rest"
/// ... )
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

    fn __repr__(&self) -> String {
        format!(
            "GeometryPair {} (diastolic: {} frames, systolic: {} frames)",
            self.label,
            self.geom_a.frames.len(),
            self.geom_b.frames.len()
        )
    }

    /// Get summaries for both geometries and a per-frame deformation table.
    ///
    /// Calls :meth:`PyGeometry.get_summary` on each contained geometry and
    /// additionally computes per-frame area and elliptic ratio for both
    /// phases.
    ///
    /// Returns
    /// -------
    /// summaries : tuple
    ///     ``((dia_mla, dia_max_stenosis, dia_len_mm), (sys_mla, sys_max_stenosis, sys_len_mm))``.
    /// table : list of list of float
    ///     Matrix of shape ``(N, 6)`` with columns
    ///     ``[id, area_dia, ellip_dia, area_sys, ellip_sys, z]``.
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

/// Python representation of a centerline point.
///
/// Combines a contour point with its local normal vector.
///
/// Attributes
/// ----------
/// contour_point : PyContourPoint
///     Position of the centerline point in 3D space.
/// normal : tuple of float
///     Normal vector ``(nx, ny, nz)`` at this centerline position.
///
/// Examples
/// --------
/// >>> cl_point = PyCenterlinePoint(
/// ...     contour_point=point,
/// ...     normal=(0.0, 1.0, 0.0)
/// ... )
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

impl From<&PyCenterlinePoint> for CenterlinePoint {
    fn from(p: &PyCenterlinePoint) -> Self {
        CenterlinePoint {
            contour_point: ContourPoint::from(&p.contour_point),
            normal: Vector3::new(p.normal.0, p.normal.1, p.normal.2),
        }
    }
}

/// Python representation of a vessel centerline.
///
/// Attributes
/// ----------
/// points : list of PyCenterlinePoint
///     Ordered list of centerline points.
///
/// Examples
/// --------
/// >>> centerline = PyCenterline(points=[p1, p2, p3])
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

    /// Build a centerline from a flat list of ``PyContourPoint`` objects.
    ///
    /// Parameters
    /// ----------
    /// contour_points : list of PyContourPoint
    ///     Ordered sequence of contour points.
    ///
    /// Returns
    /// -------
    /// PyCenterline
    ///     Centerline constructed from the provided points.
    ///
    /// Examples
    /// --------
    /// >>> pts = [PyContourPoint(...), PyContourPoint(...), ...]
    /// >>> cl = PyCenterline.from_contour_points(pts)
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

impl From<&Centerline> for PyCenterline {
    fn from(cl: &Centerline) -> Self {
        let points = cl.points.iter().map(|p| p.into()).collect();
        PyCenterline { points }
    }
}

impl From<Centerline> for PyCenterline {
    fn from(cl: Centerline) -> Self {
        let points = cl.points.iter().map(|p| p.into()).collect();
        PyCenterline { points }
    }
}

/// Python representation of a per-frame measurement record.
///
/// Attributes
/// ----------
/// frame : int
///     Frame number within the acquisition sequence.
/// phase : str
///     Cardiac phase identifier: ``"D"`` for diastole or ``"S"`` for
///     systole.
/// measurement_1 : float or None
///     Primary measurement value.  In coronary artery anomalies this is
///     the wall thickness between the aorta and the coronary artery.
/// measurement_2 : float or None
///     Secondary measurement value.  In coronary artery anomalies this is
///     the wall thickness between the pulmonary artery and the coronary
///     artery.
///
/// Examples
/// --------
/// >>> record = PyRecord(
/// ...     frame=5,
/// ...     phase="D",
/// ...     measurement_1=1.4,
/// ...     measurement_2=2.1
/// ... )
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
