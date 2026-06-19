use super::py_centerline_point::PyCenterlinePoint;
use super::py_contour_point::PyContourPoint;
use crate::types::native::{Centerline, ContourPoint};
use pyo3::prelude::*;

/// Python representation of a vessel centerline.
///
/// Attributes
/// ----------
/// points : list of PyCenterlinePoint
///     Ordered list of centerline points.
/// branch_start_indices : list of int
///     Index into ``points`` where each branch begins.  Entry 0 is always 0
///     (the main vessel); subsequent entries mark the start of side branches.
///     Read-only — recomputed by ``calculate_branches``.
///
/// Examples
/// --------
/// >>> centerline = PyCenterline(points=[p1, p2, p3])
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct PyCenterline {
    #[pyo3(get, set)]
    pub points: Vec<PyCenterlinePoint>,
    #[pyo3(get)]
    pub branch_start_indices: Vec<usize>,
}

#[pymethods]
impl PyCenterline {
    #[new]
    fn new(points: Vec<PyCenterlinePoint>) -> Self {
        let branch_start_indices = if points.is_empty() { vec![] } else { vec![0] };
        Self {
            points,
            branch_start_indices,
        }
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
            "Centerline(len={}, spacing={:.2} mm, branches={:?})",
            self.points.len(),
            self.mean_spacing(),
            self.branch_start_indices.len(),
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn __len__(&self) -> usize {
        self.points.len()
    }

    fn points_as_tuples(&self) -> Vec<(f64, f64, f64)> {
        self.points
            .iter()
            .map(|p| (p.contour_point.x, p.contour_point.y, p.contour_point.z))
            .collect()
    }

    /// Detect branches by spatial proximity and return a new centerline with
    /// ``branch_id`` assigned on every point.
    ///
    /// Points whose mutual distance is ≤ ``spacing_tolerance × median_nn_spacing``
    /// are considered spatially consecutive regardless of their original array
    /// order.  The largest connected group becomes branch 0 (main vessel);
    /// further groups are numbered by descending size.
    ///
    /// Parameters
    /// ----------
    /// spacing_tolerance : float
    ///     Multiplier on the median nearest-neighbour spacing used as the
    ///     adjacency threshold.  ``1.5`` is a reasonable starting value;
    ///     increase it if branches are incorrectly split, decrease it if
    ///     distinct branches are incorrectly merged.
    ///
    /// Returns
    /// -------
    /// PyCenterline
    ///     New centerline with ``branch_id`` set on every point and
    ///     ``branch_start_indices`` populated.
    ///
    /// Examples
    /// --------
    /// >>> cl = centerline.calculate_branches(1.5)
    /// >>> main = [p for p in cl.points if p.branch_id == 0]
    #[pyo3(signature = (spacing_tolerance = 1.0))]
    pub fn calculate_branches(&self, spacing_tolerance: f64) -> PyResult<PyCenterline> {
        let mut cl = self.to_rust_centerline();
        cl.calculate_branches(spacing_tolerance);
        Ok(PyCenterline::from(&cl))
    }

    /// Return local positions (0-indexed within the branch) of interior points
    /// where the opening angle is sharper than `cos_threshold`.
    ///
    /// Parameters
    /// ----------
    /// branch_id : int
    ///     Branch to inspect (0 = main vessel).
    /// cos_threshold : float
    ///     Cosine of the opening angle above which a point is considered sharp.
    ///     Use 0.0 for < 90°, 0.5 for < 60°, 0.866 for < 30°, etc.
    ///
    /// Returns
    /// -------
    /// list[int]
    ///     Local positions within the branch where sharp angles were found.
    pub fn find_sharp_angles(&self, branch_id: u32, cos_threshold: f64) -> Vec<usize> {
        self.to_rust_centerline()
            .find_sharp_angles(branch_id, cos_threshold)
    }

    /// Split a branch at a local position and return the updated centerline.
    ///
    /// Both resulting segments include the split point. When splitting the main
    /// branch (``branch_id=0``) the longer segment stays as branch 0; for side
    /// branches the first segment keeps its slot and the second is appended.
    ///
    /// Parameters
    /// ----------
    /// branch_id : int
    ///     Branch to split.
    /// local_pos : int
    ///     0-indexed position within the branch (as returned by
    ///     ``find_sharp_angles``).
    ///
    /// Returns
    /// -------
    /// PyCenterline
    ///     New centerline with the branch split and all IDs reassigned.
    pub fn split_branch(&self, branch_id: u32, local_pos: usize) -> PyResult<PyCenterline> {
        let mut cl = self.to_rust_centerline();
        cl.split_branch(branch_id, local_pos);
        Ok(PyCenterline::from(&cl))
    }

    /// Merge two branches and return the updated centerline.
    ///
    /// Segments are joined at the closest endpoint pair. If either branch is
    /// the main branch (id 0) the merged result becomes branch 0.
    ///
    /// Parameters
    /// ----------
    /// branch_id_a : int
    /// branch_id_b : int
    ///
    /// Returns
    /// -------
    /// PyCenterline
    ///     New centerline with the two branches merged and all IDs reassigned.
    pub fn merge_branches(&self, branch_id_a: u32, branch_id_b: u32) -> PyResult<PyCenterline> {
        let mut cl = self.to_rust_centerline();
        cl.merge_branches(branch_id_a, branch_id_b);
        Ok(PyCenterline::from(&cl))
    }

    /// Return a new centerline containing only the points of one branch.
    ///
    /// All retained points are reassigned to ``branch_id = 0`` and
    /// ``branch_start_indices`` is reset to ``[0]``.
    ///
    /// Parameters
    /// ----------
    /// branch_id : int
    ///     Branch to extract.
    ///
    /// Returns
    /// -------
    /// PyCenterline
    ///     Single-branch centerline with the requested points.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``branch_id`` does not exist in this centerline.
    pub fn get_branch(&self, branch_id: u32) -> PyResult<PyCenterline> {
        let points: Vec<PyCenterlinePoint> = self
            .points
            .iter()
            .filter(|p| p.branch_id == branch_id)
            .cloned()
            .map(|mut p| {
                p.branch_id = 0;
                p
            })
            .collect();
        if points.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "branch_id {branch_id} not found in centerline"
            )));
        }
        Ok(PyCenterline {
            points,
            branch_start_indices: vec![0],
        })
    }

    /// Remove the run-alongside-main-branch prefix from every side branch and
    /// optionally strip the inlet region from branch 0.
    ///
    /// VTP files export every branch starting from the vessel origin, so side
    /// branches share a common prefix with branch 0.  This method trims that
    /// prefix from each side branch, keeping only the bifurcation junction and
    /// the diverged portion.  Branches that overlap with branch 0 entirely are
    /// dropped.  The trim threshold is one mean inter-point spacing of branch 0.
    ///
    /// Parameters
    /// ----------
    /// rm_start_mm : float, optional
    ///     Arc-length in mm to remove from the start of branch 0 (the inlet
    ///     region).  Set to ``0.0`` (default) to leave branch 0 untouched.
    ///
    /// Returns
    /// -------
    /// PyCenterline
    ///     New centerline with overlapping prefixes removed from all side
    ///     branches and the inlet trimmed from branch 0 if requested.
    #[pyo3(signature = (rm_start_mm = 5.0))]
    pub fn cleanup_vtp_data(&self, rm_start_mm: f64) -> PyResult<PyCenterline> {
        let mut cl = self.to_rust_centerline();
        cl.cleanup_vtp_data(rm_start_mm);
        Ok(PyCenterline::from(&cl))
    }

    /// Normalise branch ordering so that downstream processing is consistent.
    ///
    /// * **Branch 0** – the point with the highest z-coordinate is moved to
    ///   index 0 (the whole branch is reversed if necessary).
    /// * **Side branches** – the endpoint closest to branch 0 becomes index 0
    ///   (the branch is reversed if necessary).
    ///
    /// Returns
    /// -------
    /// PyCenterline
    ///     New centerline with all branches in canonical order.
    pub fn check_centerline(&self) -> PyResult<PyCenterline> {
        let mut cl = self.to_rust_centerline();
        cl.check_centerline();
        Ok(PyCenterline::from(&cl))
    }
}

impl PyCenterline {
    pub fn to_rust_centerline(&self) -> Centerline {
        Centerline {
            points: self.points.iter().map(|p| p.into()).collect(),
            branch_start_indices: self.branch_start_indices.clone(),
        }
    }

    pub(crate) fn mean_spacing(&self) -> f64 {
        self.to_rust_centerline().mean_spacing()
    }
}

impl From<&Centerline> for PyCenterline {
    fn from(cl: &Centerline) -> Self {
        PyCenterline {
            points: cl.points.iter().map(|p| p.into()).collect(),
            branch_start_indices: cl.branch_start_indices.clone(),
        }
    }
}

impl From<Centerline> for PyCenterline {
    fn from(cl: Centerline) -> Self {
        PyCenterline::from(&cl)
    }
}
