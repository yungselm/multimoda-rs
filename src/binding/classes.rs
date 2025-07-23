// File: src/python_bind.rs
use crate::io::input::{Centerline, CenterlinePoint, Contour, ContourPoint, Record};
use crate::io::Geometry;
use crate::processing::geometries::GeometryPair;
use crate::entry_arr::refine_ordering;
use nalgebra::Vector3;
use pyo3::prelude::*;

/// Python representation of a 3D contour point
///
/// Attributes:
///     frame_index (int): Frame number in sequence
///     point_index (int): Index within contour
///     x (float): X-coordinate in mm
///     y (float): Y-coordinate in mm
///     z (float): Z-coordinate (depth) in mm
///     aortic (bool): Flag indicating aortic position
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
            "Point(f={}, p={}, x={:.2}, y={:.2}, z={:.2}, aortic={})",
            self.frame_index, self.point_index, self.x, self.y, self.z, self.aortic
        )
    }

    // Add a __str__ method for human-readable output
    fn __str__(&self) -> String {
        format!(
            "Point(f={}, p={}, x={:.2}, y={:.2}, z={:.2}, aortic={})",
            self.frame_index, self.point_index, self.x, self.y, self.z, self.aortic
        )
    }

    /// Euclidean distance to another PyContourPoint
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
    pub points: Vec<PyContourPoint>,
    #[pyo3(get, set)]
    pub centroid: (f64, f64, f64),
}

#[pymethods]
impl PyContour {
    /// Creates a new PyContour instance
    ///
    /// Args:
    ///     id (int): Contour identifier
    ///     points (List[PyContourPoint]): List of contour points
    ///     centroid (Tuple[float, float, float]): (x, y, z) centroid position
    #[new]
    fn new(id: u32, points: Vec<PyContourPoint>, centroid: (f64, f64, f64)) -> Self {
        Self {
            id,
            points,
            centroid,
        }
    }

    /// Returns human-readable representation of contour
    fn __repr__(&self) -> String {
        format!(
            "Contour(id={}, points={}, centroid=({:.2}, {:.2}, {:.2}))",
            self.id,
            self.points.len(),
            self.centroid.0,
            self.centroid.1,
            self.centroid.2
        )
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
        Ok(((p1.into(), p2.into()), distance))
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
        Ok(((p1.into(), p2.into()), distance))
    }

    /// Get the elliptic ratio of the current contour
    ///
    /// Returns:
    ///     float:
    ///         Ratio of farthest points distance divided by closest
    ///         opposite points distance.
    /// Example:
    ///     elliptic_ratio = contour.get_elliptic_ratio()
    pub fn get_elliptic_ratio(&self) -> PyResult<f64> {
        let rust_contour = self.to_rust_contour()?;
        let elliptic_ratio = rust_contour.elliptic_ratio();
        Ok(elliptic_ratio)
    }

    /// Get the area of the current contour
    ///
    /// Returns:
    ///     float:
    ///         Area of the current contour in the unit that the original
    ///         contour data was provided (e.g. mm2).
    /// Example:
    ///     elliptic_ratio = contour.get_elliptic_ratio()    
    pub fn get_area(&self) -> PyResult<f64> {
        let rust_contour = self.to_rust_contour()?;
        let area = rust_contour.area();
        Ok(area)
    }

    /// Rotate a given contour around it's own centroid by an angle
    /// in degrees.
    ///
    /// Returns:
    ///     PyContour:
    ///         Original Contour rotated around it's centroid
    /// Example:
    ///     contour = contour.rotate(20)
    pub fn rotate(&mut self, angle_deg: f64) -> PyResult<PyContour> {
        let angle_rad = angle_deg.to_radians();
        let mut rust_contour = self.to_rust_contour()?;
        rust_contour.rotate_contour(angle_rad);
        let contour: PyContour = rust_contour.into();
        Ok(contour)
    }

    /// translate a given contour by x, y, z coordinates
    ///
    /// Returns:
    ///     PyContour:
    ///         Original Contour translated to (x, y, z)
    /// Example:
    ///     contour = contour.translate((0.0, 1.0, 2.0))
    pub fn translate(&mut self, translation: (f64, f64, f64)) -> PyResult<PyContour> {
        let mut rust_contour = self.to_rust_contour()?;
        rust_contour.translate_contour(translation);
        let contour: PyContour = rust_contour.into();
        Ok(contour)
    }
}

impl PyContour {
    pub fn to_rust_contour(&self) -> PyResult<Contour> {
        let points = self.points.iter().map(|p| ContourPoint::from(p)).collect();

        Ok(Contour {
            id: self.id,
            points,
            centroid: self.centroid,
            aortic_thickness: None,
            pulmonary_thickness: None,
        })
    }
}

// Implement conversion for references
impl From<&&ContourPoint> for PyContourPoint {
    fn from(point: &&ContourPoint) -> Self {
        (*point).into()
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
    pub contours: Vec<PyContour>,
    #[pyo3(get, set)]
    pub catheter: Vec<PyContour>,
    #[pyo3(get, set)]
    pub walls: Vec<PyContour>,
    #[pyo3(get, set)]
    pub reference_point: PyContourPoint,
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
    fn new(
        contours: Vec<PyContour>,
        catheter: Vec<PyContour>,
        walls: Vec<PyContour>,
        reference_point: PyContourPoint,
    ) -> Self {
        Self {
            contours,
            catheter,
            walls,
            reference_point,
        }
    }

    // Add a __repr__ method for better printing
    fn __repr__(&self) -> String {
        format!(
            "Geometry({} contours, {} walls), Catheter({} catheter), Reference Point: {}",
            self.contours.len(),
            self.walls.len(),
            self.catheter.len(),
            self.reference_point.__repr__()
        )
    }

    // Add a __str__ method for human-readable output
    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Applies smoothing to all contours using moving average
    ///
    /// Args:
    ///     window_size (int): Number of points in smoothing window
    ///
    /// Note:
    ///     Larger windows create smoother contours but may lose detail
    ///
    /// Example:
    ///     >>> geom.smooth_contours(window_size=5)
    pub fn smooth_contours(&mut self, window_size: usize) {
        for contour in &mut self.contours {
            // Simple smoothing implementation
            let n = contour.points.len();
            if window_size == 0 || n < window_size {
                continue;
            }

            let mut smoothed = Vec::with_capacity(n);
            for i in 0..n {
                let start = i.saturating_sub(window_size / 2);
                let end = (i + window_size / 2 + 1).min(n);

                let mut sum_x = 0.0;
                let mut sum_y = 0.0;
                let mut sum_z = 0.0;
                let count = (end - start) as f64;

                for j in start..end {
                    sum_x += contour.points[j].x;
                    sum_y += contour.points[j].y;
                    sum_z += contour.points[j].z;
                }

                smoothed.push(PyContourPoint {
                    frame_index: contour.points[i].frame_index,
                    point_index: contour.points[i].point_index,
                    x: sum_x / count,
                    y: sum_y / count,
                    z: sum_z / count,
                    aortic: contour.points[i].aortic,
                });
            }
            contour.points = smoothed;
        }
    }

    /// Re‑orders and realigns the sequence of contours to minimize a combined spatial + index‐jump cost.
    ///
    /// Args:
    ///     delta (float): Jump penalty weight between contour IDs.
    ///     max_rounds (int): Maximum refinement iterations.
    ///     steps (int): Number of steps for frame alignment.
    ///     range (float): Range parameter for frame alignment.
    ///
    /// Returns:
    ///     PyGeometry: A new geometry with contours and catheter re‑ordered and aligned.
    pub fn reorder(&mut self, delta: f64, max_rounds: usize, steps: usize, range:f64) -> PyGeometry {
        let mut rust_geometry = self.to_rust_geometry();
        rust_geometry = refine_ordering(rust_geometry, delta, max_rounds, steps, range);
        rust_geometry.into()
    }
}

impl PyGeometry {
    /// Rust‐only: convert this Python wrapper into the core Geometry.
    pub fn to_rust_geometry(&self) -> Geometry {
        Geometry {
            contours: self
                .contours
                .iter()
                .map(|c| c.to_rust_contour().unwrap())
                .collect(),
            catheter: self
                .catheter
                .iter()
                .map(|c| c.to_rust_contour().unwrap())
                .collect(),
            walls: self
                .walls
                .iter()
                .map(|c| c.to_rust_contour().unwrap())
                .collect(),
            reference_point: (&self.reference_point).into(),
            label: String::new(),
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyGeometryPair {
    #[pyo3(get, set)]
    pub dia_geom: PyGeometry,
    #[pyo3(get, set)]
    pub sys_geom: PyGeometry,
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
#[pymethods]
impl PyGeometryPair {
    #[new]
    fn new(dia_geom: PyGeometry, sys_geom: PyGeometry) -> Self {
        Self { dia_geom, sys_geom }
    }

    // Add a __repr__ method
    fn __repr__(&self) -> String {
        format!(
            "Diastolic Geometry({} contours), ({} catheter), Reference Point: {} \n\
            Systolic Geometry({} contours), ({} catheter), Reference Point: {}",
            self.dia_geom.contours.len(),
            self.dia_geom.catheter.len(),
            self.dia_geom.reference_point.__repr__(),
            self.sys_geom.contours.len(),
            self.sys_geom.catheter.len(),
            self.sys_geom.reference_point.__repr__()
        )
    }

    // Add a __str__ method
    fn __str__(&self) -> String {
        self.__repr__()
    }
}

impl PyGeometryPair {
    pub fn to_rust_geometry_pair(&self) -> GeometryPair {
        GeometryPair {
            dia_geom: self.dia_geom.to_rust_geometry(),
            sys_geom: self.sys_geom.to_rust_geometry(),
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
        let rust_pts: Vec<ContourPoint> = contour_points
            .iter()
            .map(|p| p.into())
            .collect();

        // call your existing Rust constructor
        let rust_cl = Centerline::from_contour_points(rust_pts);

        // use your From<&Centerline> impl to go back into PyCenterline
        Ok(PyCenterline::from(&rust_cl))
    }

    fn __repr__(&self) -> String {
        format!("Centerline(len={})", self.points.len())
    }

    fn __str__(&self) -> String {
        self.__repr__()
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

/// Python wrapper for your Rust `Record`
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

/// Python representation of a measurement record
///
/// Attributes:
///     frame (int): Frame number
///     phase (str): Cardiac phase ('Diastole'/'Systole')
///     measurement_1 (float, optional): Primary measurement
///     measurement_2 (float, optional): Secondary measurement
///
/// Example:
///     >>> record = PyRecord(
///     ...     frame=5,
///     ...     phase="Diastole",
///     ...     measurement_1=42.0,
///     ...     measurement_2=38.5
///     ... )
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
    fn from(geom: &Geometry) -> Self {
        PyGeometry {
            contours: geom.contours.iter().map(|c| c.into()).collect(),
            catheter: geom.catheter.iter().map(|c| c.into()).collect(),
            walls: geom.walls.iter().map(|c| c.into()).collect(),
            reference_point: PyContourPoint::from(&geom.reference_point),
        }
    }
}

impl From<Contour> for PyContour {
    fn from(contour: Contour) -> Self {
        PyContour {
            id: contour.id,
            points: contour
                .points
                .into_iter()
                .map(|p| PyContourPoint::from(&p))
                .collect(),
            centroid: contour.centroid,
        }
    }
}


impl From<&Contour> for PyContour {
    fn from(contour: &Contour) -> Self {
        PyContour {
            id: contour.id,
            points: contour.points.iter().map(|p| p.into()).collect(),
            centroid: contour.centroid,
        }
    }
}

impl From<Geometry> for PyGeometry {
    fn from(geom: Geometry) -> Self {
        PyGeometry {
            contours: geom.contours.iter().map(|c| c.into()).collect(),
            catheter: geom.catheter.iter().map(|c| c.into()).collect(),
            walls: geom.walls.iter().map(|c| c.into()).collect(),
            reference_point: PyContourPoint::from(&geom.reference_point),
        }
    }
}

impl From<GeometryPair> for PyGeometryPair {
    fn from(pair: GeometryPair) -> Self {
        PyGeometryPair {
            dia_geom: pair.dia_geom.into(),
            sys_geom: pair.sys_geom.into(),
        }
    }
}
