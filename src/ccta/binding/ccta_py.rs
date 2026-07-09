// src/ccta/binding/ccta_py.rs
use std::collections::{HashMap, HashSet, VecDeque};

use crate::ccta::adjust_mesh::label_coronary;
use crate::ccta::adjust_mesh::label_coronary::Triangle;
use crate::ccta::adjust_mesh::scale_coronary;
use crate::ccta::discretizing;
use crate::types::binding::{PyCenterline, PyContour, PyDiscretizedVesselTree, PyFrame};
use crate::types::native::DiscretizedVesselTree;
use pyo3::prelude::*;

type Point3D = (f64, f64, f64);
type TriangleTuple = (Point3D, Point3D, Point3D);
type FiveRegionPointLists = (
    Vec<Point3D>,
    Vec<Point3D>,
    Vec<Point3D>,
    Vec<Point3D>,
    Vec<Point3D>,
);

/// Find points bounded by spheres along a coronary vessel centerline.
///
/// This version accepts and returns simple Python lists of tuples.
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
pub fn find_centerline_bounded_points_simple(
    centerline: PyCenterline,
    points: Vec<Point3D>,
    radius: f64,
) -> PyResult<Vec<Point3D>> {
    let rust_centerline = centerline.to_rust_centerline();

    // Call Rust function directly - no complex conversions needed
    let result_points =
        label_coronary::find_centerline_bounded_points(rust_centerline, &points, radius);

    Ok(result_points)
}

/// Remove occluded points using ray-triangle intersection testing.
///
/// For each centerline point, candidate points that are occluded by the mesh
/// surface (represented as triangles) are removed via ray casting.  This is
/// intended to run as a cleanup step after bounding-sphere selection.
///
/// Parameters
/// ----------
/// centerline_coronary : PyCenterline
///     Centerline of the coronary vessel used as vantage points.
/// centerline_aorta : PyCenterline
///     Centerline of the aorta used as additional vantage points.
/// range_coronary : int
///     Number of centerline points considered around each coronary point.
/// points : list of tuple of float
///     Candidate ``(x, y, z)`` points, e.g. output of
///     :func:`find_centerline_bounded_points`.
/// faces : list of tuple of tuple of float
///     Triangle faces as ``((v0x, v0y, v0z), (v1x, v1y, v1z), (v2x, v2y, v2z))``
///     triples representing the mesh surface.
/// step_size_mm : float, optional
///    Step size in mm for iterating over coronary centerline points.  Default is
///     1.0 mm, which is usually sufficient for most CCTA datasets.  Larger values
///     may speed up processing at the cost of potentially missing occlusions in very tight vessel regions.
///
/// Returns
/// -------
/// filtered_points : list of tuple of float
///     Points with occluded entries removed.
#[pyfunction]
#[pyo3(
    signature = (
        centerline_coronary,
        centerline_aorta,
        range_coronary,
        points,
        faces,
        step_size_mm = 1.0,
    )
)]
pub fn remove_occluded_points_ray_triangle(
    centerline_coronary: PyCenterline,
    centerline_aorta: PyCenterline,
    range_coronary: usize,
    points: Vec<Point3D>,
    faces: Vec<TriangleTuple>,
    step_size_mm: f64,
) -> PyResult<Vec<Point3D>> {
    let rust_centerline_coronary = centerline_coronary.to_rust_centerline();
    let rust_centerline_aorta = centerline_aorta.to_rust_centerline();

    let triangles: Vec<Triangle> = faces
        .into_iter()
        .map(|(v0, v1, v2)| Triangle::new(v0, v1, v2))
        .collect();

    let result = label_coronary::remove_occluded_points_ray_triangle_rust(
        &rust_centerline_coronary,
        &rust_centerline_aorta,
        range_coronary,
        &points,
        &triangles,
        step_size_mm,
    );
    Ok(result)
}

/// Find mesh faces that reference any vertex coincident (within `tol`) with one of
/// `points`.
///
/// Parameters
/// ----------
/// vertices : list of tuple of float
///     Mesh vertex coordinates, e.g. ``mesh.vertices.tolist()``.
/// faces : list of list of int
///     Mesh face vertex-index triples, e.g. ``mesh.faces.tolist()``.
/// points : list of tuple of float
///     Query points to match against mesh vertices (exact/near-exact matches
///     expected, within `tol`), e.g. output of
///     :func:`find_centerline_bounded_points`.
/// tol : float, optional
///     Distance tolerance for vertex matching.  Default is ``1e-6``.
///
/// Returns
/// -------
/// faces_found : list of tuple of tuple of float
///     Triangle faces as ``((v0x, v0y, v0z), (v1x, v1y, v1z), (v2x, v2y, v2z))``
///     triples, ready to pass to :func:`remove_occluded_points_ray_triangle`.
#[pyfunction]
#[pyo3(signature = (vertices, faces, points, tol = 1e-6))]
pub fn find_faces_near_points(
    vertices: Vec<Point3D>,
    faces: Vec<[usize; 3]>,
    points: Vec<Point3D>,
    tol: f64,
) -> PyResult<Vec<TriangleTuple>> {
    let triangles = label_coronary::find_faces_near_points(&vertices, &faces, &points, tol);
    let result: Vec<TriangleTuple> = triangles.into_iter().map(|t| (t.v0, t.v1, t.v2)).collect();
    Ok(result)
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
    Ok(label_coronary::find_aortic_points(
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
    let result = label_coronary::final_reclassification(
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

/// Adjust the vessel diameter by morphing points outward or inward along the centerline.
///
/// Parameters
/// ----------
/// centerline : PyCenterline
///     Centerline of the vessel.
/// points : list of tuple of float
///     Input ``(x, y, z)`` point coordinates.
/// diameter_adjustment_mm : float
///     Amount to adjust the diameter in mm.  Positive values expand the
///     vessel; negative values contract it.
///
/// Returns
/// -------
/// adjusted_points : list of tuple of float
///     Morphed ``(x, y, z)`` point coordinates.
#[pyfunction]
pub fn adjust_diameter_centerline_morphing_simple(
    centerline: PyCenterline,
    points: Vec<Point3D>,
    diameter_adjustment_mm: f64,
) -> PyResult<Vec<Point3D>> {
    let rust_centerline = centerline.to_rust_centerline();

    let result_points = scale_coronary::centerline_based_diameter_morphing(
        &rust_centerline,
        &points,
        diameter_adjustment_mm,
    );

    Ok(result_points)
}

/// Find points that lie within a specified frame region along the centerline.
///
/// Parameters
/// ----------
/// centerline : PyCenterline
///     Centerline of the vessel.
/// frames : list of PyFrame
///     Frames defining the region of interest along the centerline.
/// points : list of tuple of float
///     Input ``(x, y, z)`` point coordinates.
///
/// Returns
/// -------
/// proximal_points : list of tuple of float
///     Points located before the first frame in the region.
/// distal_points : list of tuple of float
///     Points located after the last frame in the region.
/// points_between : list of tuple of float
///     Points located within the frame region.
///
/// Examples
/// --------
/// >>> import multimodars as mm
#[pyfunction]
pub fn find_points_by_cl_region(
    centerline: PyCenterline,
    frames: Vec<PyFrame>,
    points: Vec<Point3D>,
) -> PyResult<(Vec<Point3D>, Vec<Point3D>, Vec<Point3D>)> {
    let rust_centerline = centerline.to_rust_centerline();
    let rust_frames: Vec<crate::types::native::frame::Frame> = frames
        .into_iter()
        .map(|f| f.to_rust_frame())
        .collect::<Result<_, _>>()?;

    let result_points =
        scale_coronary::find_points_by_cl_region_rs(&rust_centerline, &rust_frames, &points);

    Ok(result_points)
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
    let result_points = scale_coronary::clean_up_non_section_points(
        points_to_cleanup,
        reference_points,
        neighborhood_radius,
        min_neigbor_ratio,
    );

    Ok(result_points)
}

/// Find the optimal diameter scaling for the proximal and distal regions.
///
/// Parameters
/// ----------
/// anomalous_points : list of tuple of float
///     ``(x, y, z)`` coordinates of the anomalous vessel region.
/// n_proximal : int
///     Number of proximal points used for comparison.
/// n_distal : int
///     Number of distal points used for comparison.
/// centerline : PyCenterline
///     Centerline of the vessel region.
/// proximal_reference : list of tuple of float
///     Reference ``(x, y, z)`` points from the CCTA mesh for the proximal region.
/// distal_reference : list of tuple of float
///     Reference ``(x, y, z)`` points for the distal region.
///
/// Returns
/// -------
/// proximal_scaling : float
///     Optimal scaling distance for the proximal region.
/// distal_scaling : float
///     Optimal scaling distance for the distal region.
///
/// Examples
/// --------
/// >>> import multimodars as mm
#[pyfunction]
pub fn find_proximal_distal_scaling(
    anomalous_points: Vec<Point3D>,
    n_proximal: usize,
    n_distal: usize,
    centerline: PyCenterline,
    proximal_reference: Vec<Point3D>,
    distal_reference: Vec<Point3D>,
) -> PyResult<(f64, f64)> {
    let rust_centerline = centerline.to_rust_centerline();
    let (prox_dist, distal_dist) = scale_coronary::centerline_based_diameter_optimization(
        &anomalous_points,
        n_proximal,
        n_distal,
        &rust_centerline,
        &proximal_reference,
        &distal_reference,
    );
    Ok((prox_dist, distal_dist))
}

/// Find the optimal aortic diameter scaling using the intramural region.
///
/// Parameters
/// ----------
/// intramural_points : list of tuple of float
///     ``(x, y, z)`` coordinates of the intramural vessel region.
/// reference_points : list of tuple of float
///     Reference ``(x, y, z)`` points from the CCTA mesh.
/// centerline : PyCenterline
///     Centerline of the aorta.
///
/// Returns
/// -------
/// scaling : float
///     Optimal scaling distance for the aortic region.
///
/// Examples
/// --------
/// >>> import multimodars as mm
#[pyfunction]
pub fn find_aortic_scaling(
    intramural_points: Vec<Point3D>,
    reference_points: Vec<Point3D>,
    centerline: PyCenterline,
) -> PyResult<f64> {
    let rust_centerline = centerline.to_rust_centerline();
    let dist = scale_coronary::centerline_based_aortic_diameter_optimization(
        &intramural_points,
        &reference_points,
        &rust_centerline,
    );
    Ok(dist)
}

/// Find the optimal aortic wall scaling for coronary artery anomalies
/// using aortic points and the aortic centerline.
/// Additionally needs a reference point on the first quarter of the first round lumen.
/// Projects a vector from the centerline point to the reference point on the coronary
/// and finds the best scaling along this vector
///
/// Parameters
/// ----------
/// centerline : PyCenterline
///     Centerline of the aorta.
/// ref_pt_coronary : tuple of float
///     ``(x, y, z)`` coordinates of the first round ref point.
/// aortic_pts : list of tuple of float
///     Reference ``(x, y, z)`` points from the CCTA mesh.
///
/// Returns
/// -------
/// scaling : float
///     Optimal scaling distance for the aortic wall.
///
/// Examples
/// --------
/// >>> import multimodars as mm
/// >>>
#[pyfunction]
pub fn find_aortic_wall_scaling(
    cl_aorta: PyCenterline,
    ref_pt_coronary: Point3D,
    aortic_pts: Vec<Point3D>,
) -> PyResult<f64> {
    let rust_centerline = cl_aorta.to_rust_centerline();
    let dist = scale_coronary::centerline_based_wall_diameter_optimization(
        &rust_centerline,
        &ref_pt_coronary,
        &aortic_pts,
    );
    Ok(dist)
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
    let mut adjacency: HashMap<usize, HashSet<usize>> = HashMap::new();

    for face in faces {
        let v0 = face[0];
        let v1 = face[1];
        let v2 = face[2];

        // Add connections for all three edges of the triangle
        let edges = [(v0, v1), (v1, v2), (v2, v0)];

        for &(a, b) in &edges {
            adjacency.entry(a).or_default().insert(b);
            adjacency.entry(b).or_default().insert(a);
        }
    }

    adjacency
}

/// Fix face winding so that every pair of adjacent faces traverses their
/// shared edge in opposite directions (consistent orientation).
///
/// Faithful port of `trimesh.repair.fix_winding` — a BFS over the
/// face-adjacency graph, once per connected component, propagating a
/// consistent orientation outward from an arbitrary root face per component.
/// trimesh's own implementation does this as a Python/NetworkX loop over
/// every face-adjacency edge with several small numpy allocations per
/// iteration; this Rust version does the same traversal without that
/// per-edge interpreter overhead.
///
/// Parameters
/// ----------
/// faces : list of list of int
///     Mesh face vertex-index triples, e.g. ``mesh.faces.tolist()``.
///
/// Returns
/// -------
/// faces : list of list of int
///     The same faces, each either left as-is or with its vertex order
///     reversed so that winding is consistent within each connected
///     component of the mesh.
#[pyfunction]
pub fn fix_mesh_winding(faces: Vec<[usize; 3]>) -> Vec<[usize; 3]> {
    let n_faces = faces.len();
    if n_faces == 0 {
        return faces;
    }

    // (face_idx, edge_start, edge_end) - a face's own directed traversal of an edge.
    type FaceDirectedEdge = (usize, usize, usize);
    // (neighbor_face, this_face's_directed_edge, neighbor's_directed_edge)
    type FaceAdjacencyEntry = (usize, (usize, usize), (usize, usize));

    // For each undirected edge, collect the owning faces along with each
    // face's own directed (original-order) traversal of that edge.
    let mut edge_owners: HashMap<(usize, usize), Vec<FaceDirectedEdge>> = HashMap::new();
    for (fi, face) in faces.iter().enumerate() {
        let directed_edges = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])];
        for &(u, v) in &directed_edges {
            let key = if u < v { (u, v) } else { (v, u) };
            edge_owners.entry(key).or_default().push((fi, u, v));
        }
    }

    // Face adjacency: only edges owned by exactly 2 faces count (matches
    // trimesh's `face_adjacency`, which drops boundary/non-manifold edges).
    let mut adjacency: HashMap<usize, Vec<FaceAdjacencyEntry>> = HashMap::new();
    for owners in edge_owners.values() {
        if owners.len() != 2 {
            continue;
        }
        let (fa, ua, va) = owners[0];
        let (fb, ub, vb) = owners[1];
        adjacency
            .entry(fa)
            .or_default()
            .push((fb, (ua, va), (ub, vb)));
        adjacency
            .entry(fb)
            .or_default()
            .push((fa, (ub, vb), (ua, va)));
    }

    let mut flipped = vec![false; n_faces];
    let mut visited = vec![false; n_faces];

    for start in 0..n_faces {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let mut queue: VecDeque<usize> = VecDeque::new();
        queue.push_back(start);

        while let Some(current) = queue.pop_front() {
            let Some(neighbors) = adjacency.get(&current) else {
                continue;
            };
            for &(neighbor, current_edge, neighbor_edge) in neighbors {
                if visited[neighbor] {
                    continue;
                }
                // Current face's directed edge, accounting for its own flip state.
                let current_dir = if flipped[current] {
                    (current_edge.1, current_edge.0)
                } else {
                    current_edge
                };
                // Same starting vertex => both faces traverse the shared edge
                // in the same direction => inconsistent => flip the neighbor.
                if current_dir.0 == neighbor_edge.0 {
                    flipped[neighbor] = true;
                }
                visited[neighbor] = true;
                queue.push_back(neighbor);
            }
        }
    }

    faces
        .into_iter()
        .enumerate()
        .map(|(i, face)| {
            if flipped[i] {
                [face[2], face[1], face[0]]
            } else {
                face
            }
        })
        .collect()
}

/// Discretize a coronary vessel into uniform cross-sectional contours.
///
/// Walks ``branch_id`` of ``centerline`` at uniform arc-length intervals of
/// ``step_size``, projects the supplied mesh ``points`` onto the perpendicular
/// plane at each position, filters out empty and incomplete (half-circle) slices,
/// and resamples each surviving slice to exactly ``n_points`` evenly-spaced
/// points via a closed Catmull-Rom spline.
///
/// Parameters
/// ----------
/// centerline : PyCenterline
///     Centerline of the vessel to discretize.
/// points : list of tuple of float
///     ``(x, y, z)`` surface point cloud of the vessel (e.g. mesh vertices).
/// branch_id : int
///     Branch of the centerline to walk (0 = main vessel, 1+ = side branches).
/// step_size : float
///     Arc-length step between successive cross-sections in mm.  The slab
///     half-thickness used for point selection is ``step_size / 2``.
/// n_points : int
///     Number of evenly-spaced points on each output contour.
///
/// Returns
/// -------
/// contours : list of PyContour
///     One uniformly-sampled closed contour per valid cross-section.
///
/// Examples
/// --------
/// >>> import multimodars as mm
/// >>> centerline = mm.load_centerline("vessel.json")
/// >>> vertices = mesh.vertices.tolist()
/// >>> contours = mm.discretize_vessel(centerline, vertices, 0, 0.5, 200)
/// >>> print(f"Got {len(contours)} cross-sections")
#[pyfunction]
pub fn discretize_vessel(
    centerline: PyCenterline,
    points: Vec<Point3D>,
    branch_id: u32,
    step_size: f64,
    n_points: usize,
) -> PyResult<Vec<PyContour>> {
    let rust_centerline = centerline.to_rust_centerline();
    let contours = discretizing::discretize_vessel_rs(
        &rust_centerline,
        &points,
        branch_id,
        step_size,
        n_points,
    );
    Ok(contours.iter().map(PyContour::from).collect())
}

/// Smooth per-vertex mesh labels by majority voting over neighbours.
///
/// In each iteration every vertex whose label differs from the unanimous
/// majority of its neighbours is reassigned to that majority label.  Only
/// flips where *all* neighbours agree are applied, making the smoothing
/// conservative.
///
/// Parameters
/// ----------
/// labels : list of int
///     Per-vertex label values (``u8``), indexed by vertex index.
/// adjacency_map : dict of int to set of int
///     Vertex adjacency map as returned by :func:`build_adjacency_map`.
/// iterations : int
///     Number of smoothing passes to perform.
///
/// Returns
/// -------
/// smoothed_labels : list of int
///     Updated per-vertex label values after smoothing.
///
/// Examples
/// --------
/// >>> import multimodars as mm
/// >>>
/// >>> faces = [[0, 1, 2], [1, 2, 3]]
/// >>> adj = mm.build_adjacency_map(faces)
/// >>> labels = [0, 1, 0, 0]
/// >>> smoothed = mm.smooth_mesh_labels(labels, adj, iterations=3)
#[pyfunction]
pub fn smooth_mesh_labels(
    labels: Vec<u8>,
    adjacency_map: HashMap<usize, HashSet<usize>>,
    iterations: usize,
) -> Vec<u8> {
    let mut current_labels = labels;
    let n = current_labels.len();

    for _ in 0..iterations {
        let mut next_labels = current_labels.clone();

        for i in 0..n {
            if let Some(neighbors) = adjacency_map.get(&i) {
                if neighbors.is_empty() {
                    continue;
                }

                let my_label = current_labels[i];

                // Count occurrences of neighbor labels
                let mut counts = HashMap::new();
                for &neighbor_idx in neighbors {
                    let label = current_labels[neighbor_idx];
                    *counts.entry(label).or_insert(0) += 1;
                }

                // Find the most frequent label among neighbors
                let (&majority_label, &max_count) =
                    counts.iter().max_by_key(|&(_, count)| count).unwrap();

                // If I am different from the majority and the majority is strong
                // (You can adjust this logic: e.g., only flip if 100% of neighbors are different)
                if max_count == neighbors.len() && my_label != majority_label {
                    next_labels[i] = majority_label;
                }
            }
        }
        current_labels = next_labels;
    }

    current_labels
}

/// Discretize the full coronary vessel tree and compute orientation references.
///
/// Runs :func:`discretize_vessel` for every branch (aorta, RCA main, LCA main,
/// and each side branch), smoothes all centerlines with a Gaussian kernel
/// (σ = 2.5 points) beforehand, and then computes orientation reference triplets
/// at the ostium and every side-branch bifurcation.
///
/// Parameters
/// ----------
/// ao_cl : PyCenterline
///     Aortic centerline (branch 0 only).
/// rca_cl : PyCenterline
///     RCA centerline with all branches calculated.
/// lca_cl : PyCenterline
///     LCA centerline with all branches calculated.
/// points_ao : list of tuple of float
///     Surface mesh points ``(x, y, z)`` of the aorta.
/// points_rca_main : list of tuple of float
///     Surface mesh points for the RCA main vessel.
/// points_lca_main : list of tuple of float
///     Surface mesh points for the LCA main vessel.
/// side_branches_rca : list of list of tuple of float
///     One point list per RCA side branch, ordered by branch_id
///     (``side_branches_rca[0]`` → branch_id 1, etc.).
///     Pass ``results["rca_points_side_1"]``, ``["rca_points_side_2"]``, … in order.
/// side_branches_lca : list of list of tuple of float
///     Same structure for LCA.
/// branch_id_rca : int
///     Branch ID of the RCA main vessel (almost always ``0``).
/// branch_id_lca : int
///     Branch ID of the LCA main vessel (almost always ``0``).
/// step_size : float
///     Arc-length step between cross-sections in mm.
/// n_points : int
///     Number of evenly-spaced points per output contour.
///
/// Returns
/// -------
/// PyDiscretizedVesselTree
///     Fully populated vessel tree including orientation references.
///
/// Examples
/// --------
/// >>> import multimodars as mm
/// >>> results = mm.label_branches(rca_cl, results)
/// >>> results = mm.label_branches(lca_cl, results, results_key="lca_points")
/// >>> side_rca = [results["rca_points_side_1"], results["rca_points_side_2"]]
/// >>> side_lca = [results["lca_points_side_1"]]
/// >>> tree = mm.discretize_vessel_tree(
/// ...     ao_cl, rca_cl, lca_cl,
/// ...     results["aorta_points"],
/// ...     results["rca_points_main"],
/// ...     results["lca_points_main"],
/// ...     side_rca, side_lca,
/// ...     branch_id_rca=0, branch_id_lca=0,
/// ...     step_size=1.0, n_points=100,
/// ... )
#[pyfunction]
#[pyo3(signature = (
    ao_cl, rca_cl, lca_cl,
    points_ao, points_rca_main, points_lca_main,
    side_branches_rca, side_branches_lca,
    branch_id_rca = 0, branch_id_lca = 0,
    step_size = 1.0, n_points = 100,
    calculate_ref_pts=true,
))]
pub fn discretize_vessel_tree(
    ao_cl: PyCenterline,
    rca_cl: PyCenterline,
    lca_cl: PyCenterline,
    points_ao: Vec<Point3D>,
    points_rca_main: Vec<Point3D>,
    points_lca_main: Vec<Point3D>,
    side_branches_rca: Vec<Vec<Point3D>>,
    side_branches_lca: Vec<Vec<Point3D>>,
    branch_id_rca: u32,
    branch_id_lca: u32,
    step_size: f64,
    n_points: usize,
    calculate_ref_pts: bool,
) -> PyResult<PyDiscretizedVesselTree> {
    let mut tree = DiscretizedVesselTree::from_results_dict(
        &ao_cl.to_rust_centerline(),
        &rca_cl.to_rust_centerline(),
        &lca_cl.to_rust_centerline(),
        &points_ao,
        &points_rca_main,
        &points_lca_main,
        side_branches_rca,
        side_branches_lca,
        branch_id_rca,
        branch_id_lca,
        step_size,
        n_points,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    tree = if calculate_ref_pts {
        tree.calculate_ref_pts()
    } else {
        tree
    };

    Ok(PyDiscretizedVesselTree::from(tree))
}

#[cfg(test)]
mod fix_mesh_winding_tests {
    use super::*;

    #[test]
    fn test_already_consistent_quad_is_unchanged() {
        // Quad 0,1,2,3 split into two triangles with standard CCW winding:
        // shared edge {0,2} is (2,0) in face 0 and (0,2) in face 1 - opposite
        // directions, already consistent.
        let faces = vec![[0, 1, 2], [0, 2, 3]];
        let result = fix_mesh_winding(faces.clone());
        assert_eq!(result, faces);
    }

    #[test]
    fn test_inconsistent_pair_gets_second_face_flipped() {
        // Face 1 traverses the shared edge {0,2} as (2,0), same direction as
        // face 0's (2,0) -> inconsistent -> face 1 must be flipped.
        let faces = vec![[0, 1, 2], [2, 0, 3]];
        let result = fix_mesh_winding(faces);
        assert_eq!(result[0], [0, 1, 2]); // root face untouched
        assert_eq!(result[1], [3, 0, 2]); // reversed
    }

    #[test]
    fn test_isolated_faces_with_no_shared_edges_unchanged() {
        let faces = vec![[0, 1, 2], [5, 6, 7]];
        let result = fix_mesh_winding(faces.clone());
        assert_eq!(result, faces);
    }

    #[test]
    fn test_empty_input_returns_empty() {
        let result = fix_mesh_winding(vec![]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_bfs_propagates_across_a_triangle_fan() {
        // Fan of 3 triangles around a shared centre vertex 0, all originally
        // wound the same way as seen from +z (0,1,2), (0,2,3), (0,3,4) - i.e.
        // each shares an edge with the next in a CONSISTENT fan, so nothing
        // should need flipping; this checks BFS visits every face in one
        // component without spuriously flipping consistent faces.
        let faces = vec![[0, 1, 2], [0, 2, 3], [0, 3, 4]];
        let result = fix_mesh_winding(faces.clone());
        assert_eq!(result, faces);
    }
}
