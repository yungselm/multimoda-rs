//! Bindings for labeling a mesh: which vertices belong to the aorta, the coronaries, their
//! branches and the pullback segment. The mesh itself is not changed.
use std::collections::{BTreeMap, HashMap, HashSet};

use crate::ccta::labeling::{coronary, pullback, topology};
use crate::types::binding::{PyCenterline, PyFrame};
use pyo3::prelude::*;

type Point3D = (f64, f64, f64);
/// (aorta, lca, rca, lca_removed, rca_removed) point lists.
type FiveRegionPointLists = (
    Vec<Point3D>,
    Vec<Point3D>,
    Vec<Point3D>,
    Vec<Point3D>,
    Vec<Point3D>,
);
/// (proximal, distal, anomalous, off_path) point lists.
type PointLists4 = (Vec<Point3D>, Vec<Point3D>, Vec<Point3D>, Vec<Point3D>);

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_centerline_bounded_points, m)?)?;
    m.add_function(wrap_pyfunction!(remove_occluded_points, m)?)?;
    m.add_function(wrap_pyfunction!(find_aortic_points, m)?)?;
    m.add_function(wrap_pyfunction!(final_reclassification, m)?)?;
    m.add_function(wrap_pyfunction!(find_points_by_cl_region, m)?)?;
    m.add_function(wrap_pyfunction!(clean_outlier_points, m)?)?;
    m.add_function(wrap_pyfunction!(build_adjacency_map, m)?)?;
    m.add_function(wrap_pyfunction!(keep_largest_connected_component, m)?)?;
    Ok(())
}

/// Find points bounded by spheres along a coronary vessel centerline.
///
/// Parameters
/// ----------
/// centerline : PyCenterline
///     Centerline of the vessel.
/// points : list of tuple of float
///     List of ``(x, y, z)`` point coordinates.
/// radius : float
///     Radius of the bounding spheres around each centerline point.
///
/// Returns
/// -------
/// bounded_points : list of tuple of float
///     Filtered points that are inside the bounding spheres.
///
/// Examples
/// --------
/// >>> import multimodars as mm
/// >>>
/// >>> # Load centerline and point cloud
/// >>> centerline = mm.load_centerline("path/to/centerline.json")
/// >>> points = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), ...]  # or mesh.vertices.tolist()
/// >>>
/// >>> # Find points bounded by centerline spheres
/// >>> bounded_points = mm.find_centerline_bounded_points(centerline, points, 2.0)
/// >>> print(f"Found {len(bounded_points)} points inside vessel bounds")
#[pyfunction]
#[pyo3(signature = (centerline, points, radius))]
pub fn find_centerline_bounded_points(
    centerline: PyCenterline,
    points: Vec<Point3D>,
    radius: f64,
) -> PyResult<Vec<Point3D>> {
    let rust_centerline = centerline.to_rust_centerline();

    let result_points = coronary::find_centerline_bounded_points(rust_centerline, &points, radius)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    Ok(result_points)
}

/// Split coronary points into the visible ones and the ones hidden by the mesh.
///
/// For an acute take-off the coronary runs along the aortic wall, where the
/// bounding spheres also pick up aortic wall vertices. Rays are cast from the
/// proximal *range_mm* of the coronary centerline and the aortic centerline;
/// points hidden behind the mesh faces touching *points* are removed.
///
/// Parameters
/// ----------
/// vertices : list of tuple of float
///     Mesh vertex coordinates.
/// faces : list of list of int
///     Mesh faces as vertex-index triples.
/// points : list of tuple of float
///     Coronary points, vertices of the mesh (e.g. output of
///     :func:`find_centerline_bounded_points`).
/// centerline_coronary : PyCenterline
///     Coronary centerline, oriented from the ostium.
/// centerline_aorta : PyCenterline
///     Aortic centerline.
/// range_mm : float
///     Arc length along the coronary centerline, from its proximal end, that is checked.
/// step_size_mm : float, optional
///     Step between the centerline points rays are cast from. Default 1.0.
/// tol : float, optional
///     Distance within which a point matches a mesh vertex. Default 1e-6.
///
/// Returns
/// -------
/// kept, removed : list of tuple of float
///     The visible and the hidden points.
#[pyfunction]
#[pyo3(signature = (
    vertices, faces, points, centerline_coronary, centerline_aorta, range_mm,
    step_size_mm = 1.0, tol = 1e-6,
))]
pub fn remove_occluded_points(
    vertices: Vec<Point3D>,
    faces: Vec<[usize; 3]>,
    points: Vec<Point3D>,
    centerline_coronary: PyCenterline,
    centerline_aorta: PyCenterline,
    range_mm: f64,
    step_size_mm: f64,
    tol: f64,
) -> (Vec<Point3D>, Vec<Point3D>) {
    coronary::split_occluded_points(
        &vertices,
        &faces,
        &points,
        &centerline_coronary.to_rust_centerline(),
        &centerline_aorta.to_rust_centerline(),
        range_mm,
        step_size_mm,
        tol,
    )
}

/// Vertices present in neither `points_a` nor `points_b` (exact-match set difference).
///
/// Parameters
/// ----------
/// vertices : list of tuple of float
///     Mesh vertex coordinates, e.g. ``mesh.vertices.tolist()``.
/// points_a : list of tuple of float
///     First set of vertex-derived points to exclude.
/// points_b : list of tuple of float
///     Second set of vertex-derived points to exclude.
///
/// Returns
/// -------
/// aortic_points : list of tuple of float
///     Vertices from *vertices* that are not present in *points_a* or *points_b*.
#[pyfunction]
pub fn find_aortic_points(
    vertices: Vec<Point3D>,
    points_a: Vec<Point3D>,
    points_b: Vec<Point3D>,
) -> PyResult<Vec<Point3D>> {
    Ok(coronary::find_aortic_points(
        &vertices, &points_a, &points_b,
    ))
}

/// Refine vertex labels using a mesh adjacency map.
///
/// Applies two adjacency-based correction rules:
///
/// * Logic A - an isolated RCA/LCA vertex (no same-label neighbours) is
///   reassigned to the aorta class.
/// * Logic B - a vertex removed by occlusion detection but whose neighbours are
///   predominantly (> 70%) the corresponding coronary label is restored to that
///   label.
///
/// Parameters
/// ----------
/// vertices : list of tuple of float
///     Mesh vertex coordinates, e.g. ``mesh.vertices.tolist()``.
/// faces : list of list of int
///     Mesh face vertex-index triples, e.g. ``mesh.faces.tolist()``.
/// rca_points, lca_points, rca_removed_points, lca_removed_points : list of tuple of float
///     Current per-region vertex-derived point lists.
///
/// Returns
/// -------
/// aorta_points, rca_points, lca_points, rca_removed_points, lca_removed_points : list of tuple of float
///     The five region point lists after adjacency-based label smoothing.
#[pyfunction]
pub fn final_reclassification(
    vertices: Vec<Point3D>,
    faces: Vec<[usize; 3]>,
    rca_points: Vec<Point3D>,
    lca_points: Vec<Point3D>,
    rca_removed_points: Vec<Point3D>,
    lca_removed_points: Vec<Point3D>,
) -> PyResult<FiveRegionPointLists> {
    let result = coronary::final_reclassification(
        &vertices,
        &faces,
        &rca_points,
        &lca_points,
        &rca_removed_points,
        &lca_removed_points,
    );
    Ok((
        result.aorta_points,
        result.rca_points,
        result.lca_points,
        result.rca_removed_points,
        result.lca_removed_points,
    ))
}

/// Split the labelled points of a vessel along the pullback path of one branch.
///
/// The pullback path runs from the ostium through the parent branches, each up
/// to the junction the path leaves it at, to the tip of ``branch_id``. Only
/// vertices labelled for a branch on that path are placed on it, at the nearest
/// point of their own branch, and the frames snapped to the path bound the segment.
/// Every other vertex (other branches, parent vessels beyond the junction) is
/// off the path.
///
/// Parameters
/// ----------
/// centerline : PyCenterline
///     Centerline of the whole vessel with all its branches, branch 0 running
///     from the ostium distally.
/// branch_id : int
///     Branch the frames are aligned to.
/// frames : list of PyFrame
///     Aligned frames defining the segment along the path.
/// branch_points : dict of int to list of tuple of float
///     Vertices labelled for each branch id (``results["rca_branches"]``).
///
/// Returns
/// -------
/// proximal_points, distal_points, anomalous_points, off_path_points : list of tuple of float
///     Path vertices before, after and inside the segment, and the vertices off
///     the path.
///
/// Raises
/// ------
/// ValueError
///     If *frames* or *centerline* is empty, or *branch_id* is not in
///     *centerline*.
#[pyfunction]
pub fn find_points_by_cl_region(
    centerline: PyCenterline,
    branch_id: u32,
    frames: Vec<PyFrame>,
    branch_points: BTreeMap<u32, Vec<Point3D>>,
) -> PyResult<PointLists4> {
    let rust_centerline = centerline.to_rust_centerline();
    let rust_frames: Vec<crate::types::native::frame::Frame> = frames
        .into_iter()
        .map(|f| f.to_rust_frame())
        .collect::<Result<_, _>>()?;

    let regions = pullback::find_points_by_cl_region_rs(
        &rust_centerline,
        branch_id,
        &rust_frames,
        &branch_points,
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)?;

    Ok((
        regions.proximal,
        regions.distal,
        regions.anomalous,
        regions.off_path,
    ))
}

/// Clean up outlier points based on neighbourhood density and reference points.
///
/// Parameters
/// ----------
/// points_to_cleanup : list of tuple of float
///     Input ``(x, y, z)`` points to be filtered.
/// reference_points : list of tuple of float
///     Reference ``(x, y, z)`` points used to absorb outliers.
/// neighborhood_radius : float
///     Radius used to determine the neighbourhood for density estimation.
/// min_neigbor_ratio : float
///     Minimum fraction of neighbours required to keep a point.
///
/// Returns
/// -------
/// cleaned_points : list of tuple of float
///     Points from ``points_to_cleanup`` with outliers removed.
/// augmented_reference : list of tuple of float
///     Reference points augmented with the removed outliers.
///
/// Examples
/// --------
/// >>> import multimodars as mm
#[pyfunction]
pub fn clean_outlier_points(
    points_to_cleanup: Vec<Point3D>,
    reference_points: Vec<Point3D>,
    neighborhood_radius: f64,
    min_neigbor_ratio: f64,
) -> PyResult<(Vec<Point3D>, Vec<Point3D>)> {
    let result_points = pullback::clean_up_non_section_points(
        points_to_cleanup,
        reference_points,
        neighborhood_radius,
        min_neigbor_ratio,
    );

    Ok(result_points)
}

/// Build a vertex adjacency map from a triangle mesh face list.
///
/// For each triangle face, all three undirected edges are recorded so that
/// every vertex maps to the set of vertices it shares an edge with.
///
/// Parameters
/// ----------
/// faces : list of list of int
///     Triangle faces, each represented as a three-element array of vertex
///     indices ``[v0, v1, v2]``.
///
/// Returns
/// -------
/// adjacency_map : dict of int to set of int
///     Mapping from each vertex index to the set of its directly connected
///     neighbour vertex indices.
///
/// Examples
/// --------
/// >>> import multimodars as mm
/// >>>
/// >>> faces = [[0, 1, 2], [1, 2, 3]]
/// >>> adj = mm.build_adjacency_map(faces)
/// >>> print(adj[1])  # {0, 2, 3}
#[pyfunction]
pub fn build_adjacency_map(faces: Vec<[usize; 3]>) -> HashMap<usize, HashSet<usize>> {
    topology::build_adjacency_map(&faces)
}

/// Keep only the largest mesh-connected component of `points`.
///
/// Points split into regions by distance alone can include a few points that
/// are not mesh-connected to the region's main cluster ("islands"). These are
/// dropped. With fewer than 2 points, or none matching a mesh vertex, *points*
/// comes back unchanged.
///
/// Parameters
/// ----------
/// vertices : list of tuple of float
///     Mesh vertex coordinates.
/// faces : list of list of int
///     Mesh faces as vertex-index triples.
/// points : list of tuple of float
///     Vertices of one region.
///
/// Returns
/// -------
/// list of tuple of float
///     The points of the largest connected component.
#[pyfunction]
pub fn keep_largest_connected_component(
    vertices: Vec<Point3D>,
    faces: Vec<[usize; 3]>,
    points: Vec<Point3D>,
) -> Vec<Point3D> {
    topology::keep_largest_connected_component(&vertices, &faces, points)
}
