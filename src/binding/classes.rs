// File: src/python_bind.rs
use crate::io::input::{Contour, ContourPoint, Record};
use crate::io::Geometry;
use crate::processing::geometries::GeometryPair;
use pyo3::prelude::*;

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
    #[new]
    fn new(id: u32, points: Vec<PyContourPoint>, centroid: (f64, f64, f64)) -> Self {
        Self {
            id,
            points,
            centroid,
        }
    }

    // Add a __repr__ method
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

    // Add method to get points as tuples
    fn points_as_tuples(&self) -> Vec<(f64, f64, f64)> {
        self.points.iter().map(|p| (p.x, p.y, p.z)).collect()
    }

    pub fn find_farthest_points(&self) -> PyResult<((PyContourPoint, PyContourPoint), f64)> {
        let rust_contour = self.to_rust_contour()?;
        let ((p1, p2), distance) = rust_contour.find_farthest_points();
        Ok(((p1.into(), p2.into()), distance))
    }

    pub fn find_closest_opposite(&self) -> PyResult<((PyContourPoint, PyContourPoint), f64)> {
        let rust_contour = self.to_rust_contour()?;
        let ((p1, p2), distance) = rust_contour.find_closest_opposite();
        Ok(((p1.into(), p2.into()), distance))
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
}

impl PyGeometry {
    /// Rust‐only: convert this Python wrapper into the core Geometry.
    pub fn to_rust_geometry(&self) -> Geometry {
        Geometry {
            contours: self.contours.iter().map(|c| c.to_rust_contour().unwrap()).collect(),
            catheter: self.catheter.iter().map(|c| c.to_rust_contour().unwrap()).collect(),
            walls: self.walls.iter().map(|c| c.to_rust_contour().unwrap()).collect(),
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

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyCenterlinePoint {
    #[pyo3(get, set)]
    pub contour_point: PyContourPoint,
    #[pyo3(get, set)]
    pub normal: (f64, f64, f64),
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyCenterline {
    #[pyo3(get, set)]
    pub points: Vec<PyCenterlinePoint>,
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
