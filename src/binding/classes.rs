// File: src/python_bind.rs
use crate::entry_arr::refine_ordering;
use crate::io::input::{Centerline, CenterlinePoint, Contour, ContourPoint, Record};
use crate::io::Geometry;
use crate::processing::align_between::GeometryPair;
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
    /// Creates a new PyContour instance, automatically calculates centroid
    ///
    /// Args:
    ///     id (int): Contour identifier
    ///     points (List[PyContourPoint]): List of contour points
    #[new]
    fn new(id: u32, points: Vec<PyContourPoint>) -> Self {
        let mut contour = Self {
            id,
            points,
            centroid: (0.0, 0.0, 0.0),
        };
        contour.compute_centroid();
        contour
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

    /// Returns the len of points
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
    ///     >>> elliptic_ratio = contour.get_elliptic_ratio()
    pub fn get_elliptic_ratio(&self) -> PyResult<f64> {
        let rust_contour = self.to_rust_contour()?;
        let elliptic_ratio = rust_contour.elliptic_ratio();
        Ok(elliptic_ratio)
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
    ///     >>> contour = contour.rotate(20)
    #[pyo3(signature = (angle_deg))]
    pub fn rotate(&mut self, angle_deg: f64) -> PyResult<PyContour> {
        let angle_rad = angle_deg.to_radians();
        let mut rust_contour = self.to_rust_contour()?;
        rust_contour.rotate_contour(angle_rad);
        let contour: PyContour = rust_contour.into();
        Ok(contour)
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
    pub fn translate(&mut self, dx: f64, dy: f64, dz: f64) -> PyResult<PyContour> {
        let translation = (dx, dy, dz);
        let mut rust_contour = self.to_rust_contour()?;
        rust_contour.translate_contour(translation);
        let contour: PyContour = rust_contour.into();
        Ok(contour)
    }

    /// Sort points within a contour, so highest y-coord point
    /// has index 0 and all the others are sorted counterclockwise
    ///
    /// Returns:
    ///     PyContour:
    ///         Original Contour rearranged points.point_idx
    /// Example:
    ///     >>> contour = contour.sort_contour_points()
    pub fn sort_contour_points(&mut self) -> PyResult<PyContour> {
        let mut rust_contour = self.to_rust_contour()?;
        rust_contour.sort_contour_points();
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
    pub catheters: Vec<PyContour>,
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
        catheters: Vec<PyContour>,
        walls: Vec<PyContour>,
        reference_point: PyContourPoint,
    ) -> Self {
        Self {
            contours,
            catheters,
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
            self.catheters.len(),
            self.reference_point.__repr__()
        )
    }

    // Add a __str__ method for human-readable output
    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Replace the contour at `idx` (can be negative).
    /// 
    /// Args:
    ///     idx (float): Target index to replace.
    ///     contour (PyContour): Contour to set to target index.
    /// Example:
    ///     >>> contour = geom.contours[0].copy()
    ///     >>> geom.set_contour(10, contour)
    #[pyo3(signature = (idx, contour))]
    fn set_contour(&mut self, idx: isize, contour: PyContour) -> PyResult<()> {
        let len = self.contours.len() as isize;
        let i = if idx < 0 { len + idx } else { idx };
        if i < 0 || i >= len {
            Err(pyo3::exceptions::PyIndexError::new_err(
                "index out of range",
            ))
        } else {
            self.contours[i as usize] = contour;
            Ok(())
        }
    }

    /// Replace the contour at `idx` (can be negative).
    /// 
    /// Args:
    ///     idx (float): Target index to replace.
    ///     wall (PyContour): Wall-contour to set to target index.
    /// Example:
    ///     >>> wall = geom.walls[0].copy()
    ///     >>> geom.set_wall(10, wall)
    #[pyo3(signature = (idx, wall))]
    fn set_wall(&mut self, idx: isize, wall: PyContour) -> PyResult<()> {
        let len = self.walls.len() as isize;
        let i = if idx < 0 { len + idx } else { idx };
        if i < 0 || i >= len {
            Err(pyo3::exceptions::PyIndexError::new_err(
                "index out of range",
            ))
        } else {
            self.walls[i as usize] = wall;
            Ok(())
        }
    }

    /// Replace the contour at `idx` (can be negative).
    /// 
    /// Args:
    ///     idx (float): Target index to replace.
    ///     catheter (PyContour): Catheter to set to target index.
    /// Example:
    ///     >>> catheter = geom.catheters[0].copy()
    ///     >>> geom.set_catheter(10, catheter)
    #[pyo3(signature = (idx, catheter))]
    fn set_catheter(&mut self, idx: isize, catheter: PyContour) -> PyResult<()> {
        let len = self.catheters.len() as isize;
        let i = if idx < 0 { len + idx } else { idx };
        if i < 0 || i >= len {
            Err(pyo3::exceptions::PyIndexError::new_err(
                "index out of range",
            ))
        } else {
            self.catheters[i as usize] = catheter;
            Ok(())
        }
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
    pub fn rotate(&self, angle_deg: f64) -> PyGeometry {
        let angle_rad = angle_deg.to_radians();
        let mut rust_geometry = self.to_rust_geometry();

        let mut python_contours = Vec::with_capacity(rust_geometry.contours.len());
        let mut python_catheters = Vec::with_capacity(rust_geometry.catheter.len());

        // Rotate contours and corresponding catheter around contour centroid
        for (i, mut contour) in rust_geometry.contours.into_iter().enumerate() {
            let centroid = contour.centroid;
            contour.rotate_contour(angle_rad);

            let py_contour = PyContour::from(&contour);
            python_contours.push(py_contour);

            // If catheter exists for this contour, rotate it too
            if let Some(catheter) = rust_geometry.catheter.get_mut(i) {
                catheter.rotate_contour_around_point(angle_rad, (centroid.0, centroid.1)); // only in x-, y-plane
                python_catheters.push(PyContour::from(&*catheter));
            }
        }

        // Rotate walls normally around their own centroids
        let python_walls: Vec<PyContour> = rust_geometry
            .walls
            .into_iter()
            .map(|mut wall| {
                wall.rotate_contour(angle_rad);
                PyContour::from(wall)
            })
            .collect();

        PyGeometry {
            contours: python_contours,
            catheters: python_catheters,
            walls: python_walls,
            reference_point: self.reference_point.clone(),
        }
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
    pub fn translate(&mut self, dx: f64, dy: f64, dz: f64) -> PyGeometry {
        let rust_geometry = self.to_rust_geometry();
        let translation = (dx, dy, dz);

        let mut python_contours: Vec<PyContour> = Vec::new();
        for mut contour in rust_geometry.contours {
            contour.translate_contour(translation);
            python_contours.push(PyContour::from(contour));
        }

        let mut python_walls: Vec<PyContour> = Vec::new();
        for mut wall in rust_geometry.walls {
            wall.translate_contour(translation);
            python_walls.push(PyContour::from(wall));
        }

        let mut python_catheters: Vec<PyContour> = Vec::new();
        for mut cath in rust_geometry.catheter {
            cath.translate_contour(translation);
            python_catheters.push(PyContour::from(cath));
        }

        PyGeometry {
            contours: python_contours,
            walls: python_walls,
            catheters: python_catheters,
            reference_point: PyContourPoint {
                x: self.reference_point.x + dx,
                y: self.reference_point.y + dy,
                z: self.reference_point.z + dz,
                ..self.reference_point.clone()
            },
        }
    }

    /// Applies smoothing to all contours using a threepoint moving average
    /// 
    /// Example:
    ///     >>> geom.smooth_contours()
    pub fn smooth_contours(&self) -> PyGeometry {
        // take &self, build the Rust Geometry, run smoothing, convert back
        let geometry = self.to_rust_geometry();
        let smoothed = geometry.smooth_contours();
        smoothed.into()
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
    #[pyo3(signature = (delta, max_rounds))]
    pub fn reorder(&mut self, delta: f64, max_rounds: usize) -> PyGeometry {
        let mut rust_geometry = self.to_rust_geometry();
        rust_geometry = refine_ordering(rust_geometry, delta, max_rounds);
        rust_geometry.into()
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
    #[pyo3(signature = ())]
    pub fn get_summary(&self) -> PyResult<(f64, f64, f64)> {
        let geom = self.to_rust_geometry();

        if geom.contours.is_empty() {
            return Ok((0.0, 0.0, 0.0));
        }

        // compute areas and elliptic ratios
        let areas: Vec<f64> = geom.contours.iter().map(|c| c.area()).collect();
        let biggest = areas
                    .iter()
                    .cloned()
                    .fold(f64::NAN, |a, b| if a.is_nan() || b > a { b } else { a });
        let mla = areas
            .iter()
            .cloned()
            .fold(f64::INFINITY, |a, b| if b < a { b } else { a });

        let max_stenosis = if biggest > 0.0 { 1.0 - (mla / biggest) } else { 0.0 };

        // elliptic decision
        let all_elliptic = geom
            .contours
            .iter()
            .all(|c| c.elliptic_ratio() < 1.3);

        // threshold selection (see note above)
        let threshold = if all_elliptic { 0.70 * biggest } else { 0.50 * biggest };

        // build centroid list for distance computations
        let centroids: Vec<(f64, f64, f64)> = geom.contours.iter().map(|c| c.centroid).collect();

        // find longest contiguous run where area < threshold
        let mut longest_mm: f64 = 0.0;
        let mut i = 0usize;
        while i < areas.len() {
            if areas[i] < threshold {
                // start run
                let start = i;
                let mut end = i;
                while end + 1 < areas.len() && areas[end + 1] < threshold {
                    end += 1;
                }

                // compute length along centroids from start..=end
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

        println!("Geometry {:?}:\nMLA [mm²]: {:.2}\nMax. stenosis [%]: {:.0}\nStenosis length [mm]: {:.2}\n", geom.label, mla, max_stenosis * 100.0, longest_mm);
        Ok((mla, max_stenosis, longest_mm))
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
                .catheters
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
            self.dia_geom.catheters.len(),
            self.dia_geom.reference_point.__repr__(),
            self.sys_geom.contours.len(),
            self.sys_geom.catheters.len(),
            self.sys_geom.reference_point.__repr__()
        )
    }

    // Add a __str__ method
    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Get summaries for both diastolic and systolic geometries.
    ///
    /// Returns a tuple: ((dia_mla, dia_max_stenosis, dia_len_mm), (sys_mla, sys_max_stenosis, sys_len_mm))
    /// and a matrix (N, 6): (contour id, area_dia, ellip_dia, area_sys, ellip_sys, z-coordinate)
    ///
    /// This calls ``get_summary()`` on each contained PyGeometry and returns both results.
    /// and additionally assesses dynamic between the two PyGeometry object (area, elliptic ratio)
    #[pyo3(signature = ())]
    pub fn get_summary(&self) -> PyResult<(((f64, f64, f64), (f64, f64, f64)), Vec<[f64; 6]>)> {
        let dia = self.dia_geom.get_summary()?;
        let sys = self.sys_geom.get_summary()?;

        let map = self.create_deformation_table();
        Ok(((dia, sys), map))
    }

    fn create_deformation_table(&self) -> Vec<[f64; 6]> {
        let areas_dia: Vec<f64> =
            self.dia_geom.contours.iter().map(|c| c.get_area().unwrap()).collect();
        let areas_sys: Vec<f64> =
            self.sys_geom.contours.iter().map(|c| c.get_area().unwrap()).collect();

        let ellip_dia: Vec<f64> =
            self.dia_geom.contours.iter().map(|c| c.get_elliptic_ratio().unwrap()).collect();
        let ellip_sys: Vec<f64> =
            self.sys_geom.contours.iter().map(|c| c.get_elliptic_ratio().unwrap()).collect();

        let ids: Vec<u32> = self.dia_geom.contours.iter().map(|c| c.id).collect();
        let z_coords: Vec<f64> =
            self.dia_geom.contours.iter().map(|c| c.centroid.2).collect();

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
                    ids[i].to_string(),                      // id as integer
                    format!("{:.2}", mat[i][1]),             // area_dia
                    format!("{:.2}", mat[i][2]),             // ellip_dia
                    format!("{:.2}", mat[i][3]),             // area_sys
                    format!("{:.2}", mat[i][4]),             // ellip_sys
                    format!("{:.2}", mat[i][5]),             // z
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
        let rust_pts: Vec<ContourPoint> = contour_points.iter().map(|p| p.into()).collect();

        // call your existing Rust constructor
        let rust_cl = Centerline::from_contour_points(rust_pts);

        // use your From<&Centerline> impl to go back into PyCenterline
        Ok(PyCenterline::from(&rust_cl))
    }

    fn __repr__(&self) -> String {
        format!("Centerline(len={}, spacing={:.2} mm)", self.points.len(), self._spacing())
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
    fn from(geom: &Geometry) -> Self {
        PyGeometry {
            contours: geom.contours.iter().map(|c| c.into()).collect(),
            catheters: geom.catheter.iter().map(|c| c.into()).collect(),
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
            catheters: geom.catheter.iter().map(|c| c.into()).collect(),
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
