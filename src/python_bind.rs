// File: src/python_bind.rs
use pyo3::prelude::*;
use crate::io::input::{ContourPoint, Contour};

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
    pub fn find_farthest_points(&self) -> PyResult<((PyContourPoint, PyContourPoint), f64)> {
        let rust_contour = self.to_rust_contour()?;
        let ((p1, p2), distance) = rust_contour.find_farthest_points();
        Ok((
            (p1.into(), 
             p2.into()),
            distance
        ))
    }

    pub fn find_closest_opposite(&self) -> PyResult<((PyContourPoint, PyContourPoint), f64)> {
        let rust_contour = self.to_rust_contour()?;
        let ((p1, p2), distance) = rust_contour.find_closest_opposite();
        Ok((
            (p1.into(), 
             p2.into()),
            distance
        ))
    }
    
    #[new]
    fn new(id: u32, points: Vec<PyContourPoint>, centroid: (f64, f64, f64)) -> Self {
        Self { id, points, centroid }
    }
}

impl PyContour {
    fn to_rust_contour(&self) -> PyResult<Contour> {
        let points = self.points
            .iter()
            .map(|p| ContourPoint::from(p))
            .collect();
            
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