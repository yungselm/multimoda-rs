use crate::types::binding::PyContour;
use crate::types::native::DiscretizedVesselTree;
use pyo3::prelude::*;

type Point3D = (f64, f64, f64);
/// (main_ref, clock_ref, counter_clock_ref) — all three as plain xyz tuples.
type RefTriplet = (Point3D, Point3D, Point3D);

/// Fully discretized coronary vessel tree (aorta + RCA + LCA + side branches).
///
/// Attributes
/// ----------
/// discretized_aorta : list of PyContour
///     Cross-sectional contours along the aortic centerline.
/// discretized_rca_main : list of PyContour
///     Cross-sectional contours along the RCA main vessel.
/// discretized_lca_main : list of PyContour
///     Cross-sectional contours along the LCA main vessel.
/// rca_branches : list of list of PyContour
///     Per-side-branch contour lists for the RCA.  ``rca_branches[i]``
///     corresponds to RCA branch_id ``i + 1``.
/// lca_branches : list of list of PyContour
///     Per-side-branch contour lists for the LCA.
/// rca_references : list of tuple
///     Orientation triplets ``(main_ref, clock_ref, counter_clock_ref)`` along
///     the RCA, sorted proximal → distal.  Each element is a 3-tuple of
///     ``(x, y, z)`` coordinate tuples.
/// lca_references : list of tuple
///     Same structure for the LCA.
/// ao_rca : tuple of float
///     Centroid ``(x, y, z)`` of the aorta slice closest to the RCA ostium.
/// ao_lca : tuple of float
///     Centroid ``(x, y, z)`` of the aorta slice closest to the LCA ostium.
#[pyclass(skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct PyDiscretizedVesselTree {
    #[pyo3(get, set)]
    pub discretized_aorta: Vec<PyContour>,
    #[pyo3(get, set)]
    pub discretized_rca_main: Vec<PyContour>,
    #[pyo3(get, set)]
    pub discretized_lca_main: Vec<PyContour>,
    #[pyo3(get, set)]
    pub rca_branches: Vec<Vec<PyContour>>,
    #[pyo3(get, set)]
    pub lca_branches: Vec<Vec<PyContour>>,
    #[pyo3(get)]
    pub rca_references: Vec<RefTriplet>,
    #[pyo3(get)]
    pub lca_references: Vec<RefTriplet>,
    #[pyo3(get, set)]
    pub ao_rca: Point3D,
    #[pyo3(get, set)]
    pub ao_lca: Point3D,
}

#[pymethods]
impl PyDiscretizedVesselTree {
    fn __repr__(&self) -> String {
        format!(
            "DiscretizedVesselTree(ao={}, rca_main={}, lca_main={}, rca_branches={}, lca_branches={}, rca_refs={}, lca_refs={})",
            self.discretized_aorta.len(),
            self.discretized_rca_main.len(),
            self.discretized_lca_main.len(),
            self.rca_branches.len(),
            self.lca_branches.len(),
            self.rca_references.len(),
            self.lca_references.len(),
        )
    }

    /// Recompute orientation reference triplets and aortic ostium centroids
    /// from the current contour data.
    ///
    /// Call this after replacing contours (e.g. with B-spline fits) so that
    /// ``rca_references``, ``lca_references``, ``ao_rca``, and ``ao_lca``
    /// reflect the updated geometry.
    pub fn calculate_ref_pts(&mut self) -> PyResult<()> {
        let convert = |contours: &[PyContour]| -> PyResult<Vec<crate::types::native::Contour>> {
            contours.iter().map(|c| c.to_rust_contour()).collect()
        };

        let rust_aorta = convert(&self.discretized_aorta)?;
        let rust_rca = convert(&self.discretized_rca_main)?;
        let rust_lca = convert(&self.discretized_lca_main)?;
        let rust_rca_branches: PyResult<Vec<Vec<_>>> =
            self.rca_branches.iter().map(|b| convert(b)).collect();
        let rust_lca_branches: PyResult<Vec<Vec<_>>> =
            self.lca_branches.iter().map(|b| convert(b)).collect();

        let tree = DiscretizedVesselTree {
            discretized_aorta: rust_aorta,
            discretized_rca_main: rust_rca,
            discretized_lca_main: rust_lca,
            rca_branches: rust_rca_branches?,
            lca_branches: rust_lca_branches?,
            spacing: 1.0,
            rca_references: vec![],
            lca_references: vec![],
            ao_rca: (0.0, 0.0, 0.0),
            ao_lca: (0.0, 0.0, 0.0),
            pts_cusp_rcc: None,
            pts_cusp_lcc: None,
            pts_cusp_acc: None,
            index_stj_slice: None,
            index_aa: None,
        };

        let updated = tree.calculate_ref_pts();
        self.rca_references = updated
            .rca_references
            .into_iter()
            .map(|r| (r.main_ref, r.clock_ref, r.counter_clock_ref))
            .collect();
        self.lca_references = updated
            .lca_references
            .into_iter()
            .map(|r| (r.main_ref, r.clock_ref, r.counter_clock_ref))
            .collect();
        self.ao_rca = updated.ao_rca;
        self.ao_lca = updated.ao_lca;

        Ok(())
    }
}

impl From<DiscretizedVesselTree> for PyDiscretizedVesselTree {
    fn from(t: DiscretizedVesselTree) -> Self {
        Self {
            discretized_aorta: t.discretized_aorta.iter().map(PyContour::from).collect(),
            discretized_rca_main: t.discretized_rca_main.iter().map(PyContour::from).collect(),
            discretized_lca_main: t.discretized_lca_main.iter().map(PyContour::from).collect(),
            rca_branches: t
                .rca_branches
                .iter()
                .map(|b| b.iter().map(PyContour::from).collect())
                .collect(),
            lca_branches: t
                .lca_branches
                .iter()
                .map(|b| b.iter().map(PyContour::from).collect())
                .collect(),
            rca_references: t
                .rca_references
                .into_iter()
                .map(|r| (r.main_ref, r.clock_ref, r.counter_clock_ref))
                .collect(),
            lca_references: t
                .lca_references
                .into_iter()
                .map(|r| (r.main_ref, r.clock_ref, r.counter_clock_ref))
                .collect(),
            ao_rca: t.ao_rca,
            ao_lca: t.ao_lca,
        }
    }
}
