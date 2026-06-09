use super::py_contour::{PyContour, PyContourType};
use super::py_contour_point::PyContourPoint;
use super::py_frame::PyFrame;
use crate::types::native::{self as native, ContourPoint, ContourType, Geometry};
use pyo3::prelude::*;

type GeomSummary = (f64, f64, f64);

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
#[pyclass(from_py_object)]
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
        rust_geometry.rotate_geometry(angle_deg.to_radians());
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
        rust_geometry.translate_geometry((dx, dy, dz));
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
    pub fn get_summary(&self) -> PyResult<GeomSummary> {
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
        let rust_contour_type: ContourType = contour_type.into();

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
        self.frames.get(index).cloned().ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err(format!(
                "index {} out of range for geometry with {} frames",
                index,
                self.frames.len()
            ))
        })
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
            let downsampled = native::downsample_contour_points(&rust_points, n_points);
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

impl From<&Geometry> for PyGeometry {
    fn from(geometry: &Geometry) -> Self {
        PyGeometry {
            frames: geometry.frames.iter().map(PyFrame::from).collect(),
            label: geometry.label.clone(),
        }
    }
}

impl From<Geometry> for PyGeometry {
    fn from(geometry: Geometry) -> Self {
        PyGeometry::from(&geometry)
    }
}
