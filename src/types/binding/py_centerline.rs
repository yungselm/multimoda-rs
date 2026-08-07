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
    /// order.  The longest path through the resulting tree (by arc length, via
    /// a double-BFS diameter search) becomes branch 0 (main vessel); remaining
    /// connected components are numbered by descending point count.
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

    /// Return global `point_index` values (indices into ``points``) of interior
    /// points on `branch_id` where the opening angle is sharper than `cos_threshold`.
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
    ///     ``point_index`` values where sharp angles were found, suitable for
    ///     ``split_branch``.
    pub fn find_sharp_angles(&self, branch_id: u32, cos_threshold: f64) -> Vec<usize> {
        self.to_rust_centerline()
            .find_sharp_angles(branch_id, cos_threshold)
    }

    /// Split a branch at a point and return the updated centerline.
    ///
    /// Both resulting segments include the split point. Branches are re-sorted
    /// by descending length afterwards, so branch 0 is always the longest
    /// overall - the same invariant ``calculate_branches`` establishes.
    ///
    /// Parameters
    /// ----------
    /// branch_id : int
    ///     Branch to split.
    /// point_index : int
    ///     Global index into ``points`` (as returned by ``find_sharp_angles``)
    ///     where the split occurs. Must fall within `branch_id`'s own range.
    ///
    /// Returns
    /// -------
    /// PyCenterline
    ///     New centerline with the branch split and all IDs reassigned.
    pub fn split_branch(&self, branch_id: u32, point_index: usize) -> PyResult<PyCenterline> {
        let mut cl = self.to_rust_centerline();
        cl.split_branch(branch_id, point_index);
        Ok(PyCenterline::from(&cl))
    }

    /// Merge two branches and return the updated centerline.
    ///
    /// Segments are joined at the closest endpoint pair. Branches are re-sorted
    /// by descending length afterwards, so branch 0 is always the longest
    /// overall - the same invariant ``calculate_branches`` establishes.
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

    /// Remove the run-alongside-main-branch prefix duplicated by every side branch.
    ///
    /// Some centerline export formats (e.g. VTP) write every branch starting from
    /// the vessel origin, so side branches share a common prefix with branch 0.
    /// This trims that prefix from each side branch, keeping only the bifurcation
    /// junction and the diverged portion. Branches that overlap with branch 0
    /// entirely are dropped. The trim threshold is one mean inter-point spacing
    /// of branch 0.
    ///
    /// Returns
    /// -------
    /// PyCenterline
    ///     New centerline with overlapping prefixes removed from all side branches.
    pub fn remove_branch_overlap(&self) -> PyResult<PyCenterline> {
        let mut cl = self.to_rust_centerline();
        cl.remove_branch_overlap();
        Ok(PyCenterline::from(&cl))
    }

    /// Trim `mm` of arc length off the start of branch 0.
    ///
    /// Useful when the main branch starts at the aortic inlet and the proximal
    /// region is outside the region of interest.
    ///
    /// Parameters
    /// ----------
    /// mm : float
    ///     Arc-length in mm to remove from the start of branch 0.
    ///
    /// Returns
    /// -------
    /// PyCenterline
    ///     New centerline with the inlet trimmed from branch 0.
    pub fn trim_start(&self, mm: f64) -> PyResult<PyCenterline> {
        let mut cl = self.to_rust_centerline();
        cl.trim_start(mm);
        Ok(PyCenterline::from(&cl))
    }

    /// Resample every branch independently to even arc-length spacing.
    ///
    /// Interior points are linearly interpolated (position and radius) between
    /// the two nearest original points; tangents are recomputed afterwards. No
    /// interpolation occurs across a bifurcation.
    ///
    /// Parameters
    /// ----------
    /// spacing_mm : float
    ///     Target arc-length spacing in mm between consecutive points.
    ///
    /// Returns
    /// -------
    /// PyCenterline
    ///     New centerline resampled to even spacing per branch.
    pub fn resample(&self, spacing_mm: f64) -> PyResult<PyCenterline> {
        let mut cl = self.to_rust_centerline();
        cl.resample(spacing_mm);
        Ok(PyCenterline::from(&cl))
    }

    /// Smooth centerline positions with a Gaussian kernel (per branch).
    ///
    /// `sigma` is the half-width in number of centerline points. A value of
    /// ``1.0`` is a gentle neighbourhood average; ``2–5`` removes noise while
    /// keeping the overall vessel path; larger values heavily round corners.
    /// Branches are processed independently so no smoothing bleeds across a
    /// bifurcation.
    ///
    /// Parameters
    /// ----------
    /// sigma : float
    ///     Half-width of the Gaussian kernel in number of centerline points.
    ///
    /// Returns
    /// -------
    /// PyCenterline
    ///     New centerline with smoothed positions and recomputed tangents.
    pub fn smooth(&self, sigma: f64) -> PyResult<PyCenterline> {
        let mut cl = self.to_rust_centerline();
        cl.smooth(sigma);
        Ok(PyCenterline::from(&cl))
    }

    /// Reverse branch 0 if its highest-z point is not already at its start,
    /// then apply the same "closer end goes first" rule to every side branch,
    /// using branch 0 (post-reversal) as the reference.
    ///
    /// For centerlines with no anatomical reference to orient against, e.g. the
    /// aorta — use ``orient_to_reference`` instead whenever one is available.
    /// Only correct under the standard CT/DICOM convention where z increases
    /// toward the head, so the aortic root/valve is the highest-z point.
    ///
    /// Returns
    /// -------
    /// PyCenterline
    ///     New centerline with all branches in canonical order.
    pub fn orient_by_max_z(&self) -> PyResult<PyCenterline> {
        let mut cl = self.to_rust_centerline();
        cl.orient_by_max_z();
        Ok(PyCenterline::from(&cl))
    }

    /// Reverse branch 0 if its last point is closer to `reference`'s branch 0
    /// than its first point is, then apply the same rule to every side branch —
    /// so every branch starts at whichever end is nearer `reference`.
    ///
    /// Any side branches `reference` has are ignored so a stray one can't skew
    /// the distance check — only `reference`'s branch 0 is ever measured
    /// against. Distance to `reference` is the minimum distance to any point of
    /// its branch 0, not a single fixed point — e.g. for a coronary centerline,
    /// `reference` would be the aorta centerline, not one ostium point.
    ///
    /// Parameters
    /// ----------
    /// reference : PyCenterline
    ///     Centerline to orient towards (e.g. the aorta, for a coronary centerline).
    ///
    /// Returns
    /// -------
    /// PyCenterline
    ///     New centerline with all branches in canonical order.
    pub fn orient_to_reference(&self, reference: &PyCenterline) -> PyResult<PyCenterline> {
        let mut cl = self.to_rust_centerline();
        cl.orient_to_reference(&reference.to_rust_centerline());
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
