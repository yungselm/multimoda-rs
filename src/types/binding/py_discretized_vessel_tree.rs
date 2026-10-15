use crate::types::binding::PyContour;
use crate::types::native::{Contour, DiscretizedBranch, DiscretizedVesselTree, ReferenceTriplet};
use pyo3::prelude::*;
use std::collections::BTreeMap;

type Point3D = (f64, f64, f64);
/// (main_ref, clock_ref, counter_clock_ref) as xyz tuples.
type RefTriplet = (Point3D, Point3D, Point3D);
type PyBranches = BTreeMap<u32, Py<PyDiscretizedBranch>>;

/// One discretized branch of a coronary.
///
/// Attributes
/// ----------
/// contours : list of PyContour
///     Cross-sections along the branch, proximal → distal.
/// parent : tuple of (int, int) or None
///     Parent branch id (from the centerline) and the index of the parent
///     contour nearest this branch's start. ``None`` for the main vessel.
/// ostium : tuple or None
///     Triplet ``(main_ref, clock_ref, counter_clock_ref)`` of ``(x, y, z)``
///     tuples on the first contour. Its ``main_ref`` faces the vessel the
///     branch leaves.
/// bifurcations : dict of int to tuple
///     Triplet on this branch where each child branch leaves it, by child
///     branch id. Its ``main_ref`` is the child's start.
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyDiscretizedBranch {
    #[pyo3(get, set)]
    pub contours: Vec<PyContour>,
    #[pyo3(get)]
    pub parent: Option<(u32, usize)>,
    #[pyo3(get)]
    pub ostium: Option<RefTriplet>,
    #[pyo3(get)]
    pub bifurcations: BTreeMap<u32, RefTriplet>,
}

#[pymethods]
impl PyDiscretizedBranch {
    fn __repr__(&self) -> String {
        format!(
            "DiscretizedBranch(contours={}, parent={:?}, bifurcations={:?})",
            self.contours.len(),
            self.parent,
            self.bifurcations.keys().collect::<Vec<_>>(),
        )
    }
}

impl PyDiscretizedBranch {
    fn from_native(b: &DiscretizedBranch) -> Self {
        Self {
            contours: b.contours.iter().map(PyContour::from).collect(),
            parent: b.parent,
            ostium: b.ostium.clone().map(to_py_triplet),
            bifurcations: b
                .bifurcations
                .iter()
                .map(|(&id, r)| (id, to_py_triplet(r.clone())))
                .collect(),
        }
    }

    fn to_native(&self) -> PyResult<DiscretizedBranch> {
        Ok(DiscretizedBranch {
            contours: to_rust_contours(&self.contours)?,
            parent: self.parent,
            ..Default::default()
        })
    }
}

/// Discretized coronary vessel tree: the aorta, and every RCA and LCA branch by branch id.
///
/// Attributes
/// ----------
/// discretized_aorta : list of PyContour
///     Cross-sections along the aorta.
/// rca, lca : dict of int to PyDiscretizedBranch
///     Branches by branch id, 0 being the main vessel. Ids are the centerline's
///     branch ids. A removed or failed branch is missing, the others keep their
///     id. ``tree.rca[0].ostium`` is the RCA ostium triplet,
///     ``tree.lca[0].bifurcations[1]`` the triplet where LCA branch 1 leaves the
///     main vessel.
/// ao_rca, ao_lca : tuple of float
///     Centroid of the aorta slice closest to each ostium.
/// spacing : float
///     Arc length between cross-sections in mm.
#[pyclass]
#[derive(Debug)]
pub struct PyDiscretizedVesselTree {
    #[pyo3(get, set)]
    pub discretized_aorta: Vec<PyContour>,
    #[pyo3(get, set)]
    pub rca: PyBranches,
    #[pyo3(get, set)]
    pub lca: PyBranches,
    #[pyo3(get, set)]
    pub ao_rca: Point3D,
    #[pyo3(get, set)]
    pub ao_lca: Point3D,
    #[pyo3(get)]
    pub spacing: f64,
}

#[pymethods]
impl PyDiscretizedVesselTree {
    fn __repr__(&self) -> String {
        format!(
            "DiscretizedVesselTree(ao={}, rca_branches={:?}, lca_branches={:?})",
            self.discretized_aorta.len(),
            self.rca.keys().collect::<Vec<_>>(),
            self.lca.keys().collect::<Vec<_>>(),
        )
    }

    /// Recompute the junction contours, the reference triplets, ``ao_rca`` and
    /// ``ao_lca`` after editing contours. Branch parents are kept.
    pub fn calculate_ref_pts(&mut self, py: Python<'_>) -> PyResult<()> {
        let tree = DiscretizedVesselTree {
            discretized_aorta: to_rust_contours(&self.discretized_aorta)?,
            spacing: self.spacing,
            rca: to_native_branches(py, &self.rca)?,
            lca: to_native_branches(py, &self.lca)?,
            ao_rca: (0.0, 0.0, 0.0),
            ao_lca: (0.0, 0.0, 0.0),
            pts_cusp_rcc: None,
            pts_cusp_lcc: None,
            pts_cusp_acc: None,
            index_stj_slice: None,
            index_aa: None,
        }
        .calculate_ref_pts();

        // Write into the existing branch objects, so references held in Python see the update.
        for (py_branches, branches) in [(&self.rca, &tree.rca), (&self.lca, &tree.lca)] {
            for (id, b) in branches {
                let mut target = py_branches[id].borrow_mut(py);
                let updated = PyDiscretizedBranch::from_native(b);
                target.parent = updated.parent;
                target.ostium = updated.ostium;
                target.bifurcations = updated.bifurcations;
            }
        }
        self.ao_rca = tree.ao_rca;
        self.ao_lca = tree.ao_lca;
        Ok(())
    }
}

impl PyDiscretizedVesselTree {
    pub fn from_native(py: Python<'_>, t: &DiscretizedVesselTree) -> PyResult<Self> {
        Ok(Self {
            discretized_aorta: t.discretized_aorta.iter().map(PyContour::from).collect(),
            rca: to_py_branches(py, &t.rca)?,
            lca: to_py_branches(py, &t.lca)?,
            ao_rca: t.ao_rca,
            ao_lca: t.ao_lca,
            spacing: t.spacing,
        })
    }
}

fn to_py_branches(
    py: Python<'_>,
    branches: &BTreeMap<u32, DiscretizedBranch>,
) -> PyResult<PyBranches> {
    branches
        .iter()
        .map(|(&id, b)| Ok((id, Py::new(py, PyDiscretizedBranch::from_native(b))?)))
        .collect()
}

fn to_native_branches(
    py: Python<'_>,
    branches: &PyBranches,
) -> PyResult<BTreeMap<u32, DiscretizedBranch>> {
    branches
        .iter()
        .map(|(&id, b)| Ok((id, b.borrow(py).to_native()?)))
        .collect()
}

fn to_rust_contours(contours: &[PyContour]) -> PyResult<Vec<Contour>> {
    contours.iter().map(|c| c.to_rust_contour()).collect()
}

fn to_py_triplet(r: ReferenceTriplet) -> RefTriplet {
    (r.main_ref, r.clock_ref, r.counter_clock_ref)
}
