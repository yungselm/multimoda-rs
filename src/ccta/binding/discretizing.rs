//! Bindings for discretizing a mesh into the vessel tree, the skeleton of evenly spaced
//! cross-sections with its reference points, and for cutting the mesh at its planes.
use std::collections::BTreeMap;

use crate::ccta::discretizing;
use crate::ccta::discretizing::cutting;
use crate::types::binding::{PyCenterline, PyContour, PyDiscretizedVesselTree};
use crate::types::native::DiscretizedVesselTree;
use pyo3::prelude::*;

type Point3D = (f64, f64, f64);

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(discretize_vessel, m)?)?;
    m.add_function(wrap_pyfunction!(discretize_vessel_tree, m)?)?;
    m.add_function(wrap_pyfunction!(cut_mesh_at_contour, m)?)?;
    Ok(())
}

/// Discretize a vessel into evenly sampled cross-sections by cutting its surface mesh.
///
/// Cuts the mesh with planes perpendicular to branch ``branch_id`` every
/// ``step_size``, drops empty and incomplete end slices and resamples each
/// outline to ``n_points`` evenly spaced points. The cut keeps lumens of any
/// shape (eccentric, notched, crescent-shaped). Gaps in the cut region, such as
/// a side-branch ostium excluded by ``region_points``, are bridged with
/// straight edges. ``centerline`` must already be smoothed and resampled.
///
/// Parameters
/// ----------
/// centerline : PyCenterline
///     Prepared centerline of the vessel.
/// vertices : list of tuple of float
///     Mesh vertices, e.g. ``[tuple(v) for v in mesh.vertices.tolist()]``.
/// faces : list of list of int
///     Triangles as vertex-index triples, e.g. ``mesh.faces.tolist()``.
/// branch_id : int
///     Branch to walk (0 = main vessel).
/// step_size : float
///     Arc length between cross-sections in mm.
/// n_points : int
///     Points per output contour.
/// region_points : list of tuple of float, optional
///     Vertices labelling the region to cut (e.g. ``results["rca_branches"][0]``).
///     A face is cut when at least two of its vertices are labelled. ``None``
///     cuts the whole mesh.
///
/// Returns
/// -------
/// contours : list of PyContour
///     One closed contour per valid cross-section.
///
/// Raises
/// ------
/// ValueError
///     If a face references a vertex index outside ``vertices``, ``step_size``
///     is not positive, ``n_points`` < 3, ``branch_id`` does not exist, or
///     ``region_points`` select no faces.
///
/// Examples
/// --------
/// >>> vertices = [tuple(v) for v in mesh.vertices.tolist()]
/// >>> contours = mm.discretize_vessel(centerline, vertices, mesh.faces.tolist(), 0, 0.5, 200)
#[pyfunction]
#[pyo3(signature = (
    centerline, vertices, faces,
    branch_id = 0, step_size = 0.5, n_points = 200,
    region_points = None,
))]
pub fn discretize_vessel(
    centerline: PyCenterline,
    vertices: Vec<Point3D>,
    faces: Vec<[usize; 3]>,
    branch_id: u32,
    step_size: f64,
    n_points: usize,
    region_points: Option<Vec<Point3D>>,
) -> PyResult<Vec<PyContour>> {
    let mesh = discretizing::SurfaceMesh::new(&vertices, faces)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let contours = discretizing::discretize_vessel_rs(
        &centerline.to_rust_centerline(),
        &mesh,
        region_points.as_deref(),
        branch_id,
        step_size,
        n_points,
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(contours.iter().map(PyContour::from).collect())
}

/// Discretize the full coronary vessel tree and compute orientation references.
///
/// Runs :func:`discretize_vessel` for the aorta and every labelled RCA and LCA
/// branch, each on its own labelled faces, then computes the reference
/// triplets. Branch parents come from the centerlines. The centerlines must
/// already be smoothed and resampled.
///
/// Parameters
/// ----------
/// ao_cl, rca_cl, lca_cl : PyCenterline
///     Prepared centerlines, coronaries with their branches calculated.
/// mesh_vertices : list of tuple of float
///     Vertices of the labelled mesh (``results["mesh"]``).
/// mesh_faces : list of list of int
///     Its triangles as vertex-index triples.
/// points_ao : list of tuple of float
///     Vertices labelling the aorta.
/// rca_branches, lca_branches : dict of int to list of tuple of float
///     Vertices per branch id (``results["rca_branches"]``,
///     ``results["lca_branches"]``). Every id must be a branch of its
///     centerline, and 0 is the main vessel.
/// step_size : float
///     Arc length between cross-sections in mm.
/// n_points : int
///     Points per output contour.
///
/// Returns
/// -------
/// PyDiscretizedVesselTree
///     Vessel tree with the branches by id, including the reference triplets.
///
/// Raises
/// ------
/// ValueError
///     If the aorta or a main vessel cannot be discretized (see
///     :func:`discretize_vessel`), or a labelled branch is not in its
///     centerline. A failing side branch is left out with a warning.
///
/// Examples
/// --------
/// >>> mesh = results["mesh"]
/// >>> tree = mm.discretize_vessel_tree(
/// ...     ao_cl, rca_cl, lca_cl,
/// ...     [tuple(v) for v in mesh.vertices.tolist()], mesh.faces.tolist(),
/// ...     results["aorta_points"], results["rca_branches"], results["lca_branches"],
/// ... )
#[pyfunction]
#[pyo3(signature = (
    ao_cl, rca_cl, lca_cl,
    mesh_vertices, mesh_faces,
    points_ao, rca_branches, lca_branches,
    step_size = 1.0, n_points = 100,
))]
pub fn discretize_vessel_tree(
    py: Python<'_>,
    ao_cl: PyCenterline,
    rca_cl: PyCenterline,
    lca_cl: PyCenterline,
    mesh_vertices: Vec<Point3D>,
    mesh_faces: Vec<[usize; 3]>,
    points_ao: Vec<Point3D>,
    rca_branches: BTreeMap<u32, Vec<Point3D>>,
    lca_branches: BTreeMap<u32, Vec<Point3D>>,
    step_size: f64,
    n_points: usize,
) -> PyResult<PyDiscretizedVesselTree> {
    let mesh = discretizing::SurfaceMesh::new(&mesh_vertices, mesh_faces)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let tree = DiscretizedVesselTree::from_results_dict(
        &ao_cl.to_rust_centerline(),
        &rca_cl.to_rust_centerline(),
        &lca_cl.to_rust_centerline(),
        &mesh,
        &points_ao,
        &rca_branches,
        &lca_branches,
        step_size,
        n_points,
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
    .calculate_ref_pts();

    PyDiscretizedVesselTree::from_native(py, &tree)
}

/// Cut a mesh with the plane of a discretized contour and close the cut flat.
///
/// The plane runs through the contour's centroid, normal to its outline
/// (pointing distally). Only the closed ring of the mesh around the centroid
/// is cut, so other vessels crossing the plane are untouched. The ring lies
/// exactly on the plane: vertices closer to it than *snap_fraction* of a
/// crossed edge are moved onto it, the other crossed edges are split. The
/// part of the surface the ring separates off on the dropped side is removed
/// (with any branch leaving it), and the ring is closed with a flat fan
/// around its area centroid.
///
/// Parameters
/// ----------
/// vertices : list of (float, float, float)
///     Mesh vertices.
/// faces : list of [int, int, int]
///     Mesh faces.
/// contour : PyContour
///     A contour of a discretized tree. Only its plane is used, so the mesh
///     may have changed (smoothed, remeshed) since discretizing.
/// keep : {"larger", "proximal", "distal"}
///     Side of the plane that stays: the one with the larger surface area
///     (default), or the side against / along the centerline direction.
/// search_radius_factor : float
///     Only faces within this multiple of the contour radius from its
///     centroid are cut.
/// snap_fraction : float
///     Fraction of a crossed edge within which its end vertex is snapped onto
///     the plane, in ``[0, 0.5)``.
///
/// Returns
/// -------
/// (vertices, faces, ring, centroid, cap_faces, old_to_new, kept)
///     The cut and capped mesh, the ring vertex indices in cap order, the
///     cap centre's index, the cap faces (also in *faces*) and each input
///     vertex's new index (``None`` where it was cut away), and the side kept
///     (``"proximal"`` or ``"distal"``, also when *keep* is ``"larger"``).
///
/// Raises
/// ------
/// ValueError
///     If no closed ring encloses the centroid (the plane cuts a junction),
///     the ring does not separate the mesh, or the fan would fold over.
#[pyfunction]
#[pyo3(signature = (
    vertices, faces, contour,
    keep = "larger", search_radius_factor = 2.0, snap_fraction = 0.2,
))]
#[allow(clippy::type_complexity)]
pub fn cut_mesh_at_contour(
    vertices: Vec<Point3D>,
    faces: Vec<[usize; 3]>,
    contour: PyContour,
    keep: &str,
    search_radius_factor: f64,
    snap_fraction: f64,
) -> PyResult<(
    Vec<Point3D>,
    Vec<[usize; 3]>,
    Vec<usize>,
    usize,
    Vec<[usize; 3]>,
    Vec<Option<usize>>,
    &'static str,
)> {
    let value_error = pyo3::exceptions::PyValueError::new_err;
    let keep = match keep {
        "proximal" => cutting::Keep::Proximal,
        "distal" => cutting::Keep::Distal,
        "larger" => cutting::Keep::Larger,
        other => {
            return Err(value_error(format!(
                "keep must be 'larger', 'proximal' or 'distal', got {other:?}"
            )))
        }
    };
    let mesh = discretizing::SurfaceMesh::new(&vertices, faces).map_err(value_error)?;
    let plane =
        cutting::CutPlane::from_contour(&contour.to_rust_contour()?).map_err(value_error)?;
    let cut = cutting::cut_mesh_at_plane(
        &mesh,
        &plane,
        search_radius_factor * plane.radius,
        keep,
        snap_fraction,
    )
    .map_err(value_error)?;
    Ok((
        cut.vertices.iter().map(|p| (p.x, p.y, p.z)).collect(),
        cut.faces,
        cut.ring,
        cut.centroid,
        cut.cap_faces,
        cut.old_to_new,
        if cut.kept_distal {
            "distal"
        } else {
            "proximal"
        },
    ))
}
