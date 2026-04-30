// src/ccta/binding/label_py.rs
use std::collections::{HashMap, HashSet};

use crate::ccta::adjust_mesh::label_coronary::find_centerline_bounded_points;
use crate::ccta::adjust_mesh::label_coronary::{
    remove_occluded_points_ray_triangle_rust, Triangle,
};
use crate::ccta::adjust_mesh::scale_coronary::{
    centerline_based_aortic_diameter_optimization, centerline_based_diameter_morphing,
    centerline_based_diameter_optimization, centerline_based_wall_diameter_optimization,
    clean_up_non_section_points, find_points_by_cl_region_rs,
};
use crate::intravascular::binding::classes::{PyCenterline, PyFrame};
use pyo3::prelude::*;

type Point3D = (f64, f64, f64);
type TriangleTuple = (Point3D, Point3D, Point3D);

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
    let result_points = find_centerline_bounded_points(rust_centerline, &points, radius);

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
///
/// Returns
/// -------
/// filtered_points : list of tuple of float
///     Points with occluded entries removed.
#[pyfunction]
pub fn remove_occluded_points_ray_triangle(
    centerline_coronary: PyCenterline,
    centerline_aorta: PyCenterline,
    range_coronary: usize,
    points: Vec<Point3D>,
    faces: Vec<TriangleTuple>,
) -> PyResult<Vec<Point3D>> {
    let rust_centerline_coronary = centerline_coronary.to_rust_centerline();
    let rust_centerline_aorta = centerline_aorta.to_rust_centerline();

    // Convert faces to Triangle format
    let triangles: Vec<Triangle> = faces
        .into_iter()
        .map(|(v0, v1, v2)| Triangle::new(v0, v1, v2))
        .collect();

    let result = remove_occluded_points_ray_triangle_rust(
        &rust_centerline_coronary,
        &rust_centerline_aorta,
        range_coronary,
        &points,
        &triangles,
    );
    Ok(result)
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

    let result_points =
        centerline_based_diameter_morphing(&rust_centerline, &points, diameter_adjustment_mm);

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
    let rust_frames: Vec<crate::intravascular::io::geometry::Frame> = frames
        .into_iter()
        .map(|f| f.to_rust_frame())
        .collect::<Result<_, _>>()?;

    let result_points = find_points_by_cl_region_rs(&rust_centerline, &rust_frames, &points);

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
    let result_points = clean_up_non_section_points(
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
    let (prox_dist, distal_dist) = centerline_based_diameter_optimization(
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
    let dist = centerline_based_aortic_diameter_optimization(
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
    let dist = centerline_based_wall_diameter_optimization(
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
