// src/ccta/binding/label_py.rs
use std::collections::{HashMap, HashSet};

use crate::ccta::adjust_mesh::label_coronary::find_centerline_bounded_points;
use crate::ccta::adjust_mesh::label_coronary::{
    remove_occluded_points_ray_triangle_rust, Triangle,
};
use crate::ccta::adjust_mesh::scale_coronary::{
    centerline_based_aortic_diameter_optimization, centerline_based_diameter_morphing,
    centerline_based_diameter_optimization, clean_up_non_section_points,
    find_points_by_cl_region_rs,
};
use crate::intravascular::binding::classes::{PyCenterline, PyFrame};
use pyo3::prelude::*;

/// Finds points that are bounded by spheres along a coronary vessel centerline.
/// This version accepts and returns simple Python lists of tuples.
///
/// Args:
///     centerline: PyCenterline object representing the vessel centerline
///     points: List of (x, y, z) tuples containing point coordinates
///     radius: Radius of the bounding spheres around each centerline point
///
/// Returns:
///     List of (x, y, z) tuples: Filtered points that are inside the bounding spheres
///
/// Example:
///     >>> import multimodars as mm
///     >>>
///     >>> # Load centerline and point cloud  
///     >>> centerline = mm.load_centerline("path/to/centerline.json")
///     >>> points = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), ...]  # or mesh.vertices.tolist()
///     >>>
///     >>> # Find points bounded by centerline spheres
///     >>> bounded_points = mm.find_centerline_bounded_points(centerline, points, 2.0)
///     >>> print(f"Found {len(bounded_points)} points inside vessel bounds")
#[pyfunction]
pub fn find_centerline_bounded_points_simple(
    centerline: PyCenterline,
    points: Vec<(f64, f64, f64)>,
    radius: f64,
) -> PyResult<Vec<(f64, f64, f64)>> {
    let rust_centerline = centerline.to_rust_centerline();

    // Call Rust function directly - no complex conversions needed
    let result_points = find_centerline_bounded_points(rust_centerline, &points, radius);

    Ok(result_points)
}

/// Remove occluded / overlapping points seen from the centerline.
///
/// For each centerline point, we keep at most one point per angular bin
/// (quantized theta/phi) — the point with the smallest radius (closest to the center).
///
/// Args:
///     `centerline`: centerline used as vantage points
///     `points`: candidate points (e.g., output of find_centerline_bounded_points)
///     `radius`: same bounding sphere radius used to gather points (we only consider points within this distance from each cl point)
///     `angular_tolerance_deg`: angular bin size in degrees (e.g. 8.0). Smaller = stricter separation.
///
/// Returns:
///    Filtered list of points with occluded points removed.
///
/// This is intended to run as a separate cleanup step after your existing selection.
#[pyfunction]
pub fn remove_occluded_points_ray_triangle(
    centerline_coronary: PyCenterline,
    centerline_aorta: PyCenterline,
    range_coronary: usize,
    points: Vec<(f64, f64, f64)>,
    faces: Vec<((f64, f64, f64), (f64, f64, f64), (f64, f64, f64))>,
) -> PyResult<Vec<(f64, f64, f64)>> {
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

/// Adjust centerline-based diameter by morphing points outward or inward.
///
/// Args:
///     centerline: PyCenterline object representing the vessel centerline
///     points: List of (x, y, z) tuples containing point coordinates
///     diameter_adjustment_mm: Amount to adjust diameter (positive to expand, negative to contract)
///
/// Returns:
///     List of (x, y, z) tuples: Adjusted point coordinates
#[pyfunction]
pub fn adjust_diameter_centerline_morphing_simple(
    centerline: PyCenterline,
    points: Vec<(f64, f64, f64)>,
    diameter_adjustment_mm: f64,
) -> PyResult<Vec<(f64, f64, f64)>> {
    let rust_centerline = centerline.to_rust_centerline();

    let result_points =
        centerline_based_diameter_morphing(&rust_centerline, &points, diameter_adjustment_mm);

    Ok(result_points)
}

/// Find points that are within a specified frame region along the centerline.
///
/// Args:
///     centerline: PyCenterline object representing the vessel centerline
///     start_frame: PyFrame
///     end_frame: PyFrame
///     points: list of (x, y, z) tuples containing point coordinates
///
/// Returns:
///     Tuple of three lists of (x, y, z) tuples:
///         - proximal_points: points before the start frame region
///         - distal_points: points after the end frame region
///         - points_between: points within the frame region
/// Example:
///     >>> import multimodars as mm
#[pyfunction]
pub fn find_points_by_cl_region(
    centerline: PyCenterline,
    frames: Vec<PyFrame>,
    points: Vec<(f64, f64, f64)>,
) -> PyResult<(
    Vec<(f64, f64, f64)>,
    Vec<(f64, f64, f64)>,
    Vec<(f64, f64, f64)>,
)> {
    let rust_centerline = centerline.to_rust_centerline();
    let rust_frames: Vec<crate::intravascular::io::geometry::Frame> = frames
        .into_iter()
        .map(|f| f.to_rust_frame())
        .collect::<Result<_, _>>()?;

    let result_points = find_points_by_cl_region_rs(&rust_centerline, &rust_frames, &points);

    Ok(result_points)
}

/// Clean up points based on their neigbouring points and a list of reference points.
///
/// Args:
///     points_to_cleanup: list of (x, y, z) tuples containing point coordinates
///     reference_points: list of (x, y, z) tuples containing point coordinates
///
/// Returns:
///     Tuple of two lists of (x, y, z) tuples:
///         - cleaned_points: removed outliers
///         - reference_points: added outliers
/// Example:
///     >>> import multimodars as mm
#[pyfunction]
pub fn clean_outlier_points(
    points_to_cleanup: Vec<(f64, f64, f64)>,
    reference_points: Vec<(f64, f64, f64)>,
    neighborhood_radius: f64,
    min_neigbor_ratio: f64,
) -> PyResult<(Vec<(f64, f64, f64)>, Vec<(f64, f64, f64)>)> {
    let result_points = clean_up_non_section_points(
        points_to_cleanup,
        reference_points,
        neighborhood_radius,
        min_neigbor_ratio,
    );

    Ok(result_points)
}

/// Find optimal scaling for distal and proximal region.
///
/// Args:
///     anomalous_points: list of (x, y, z) tuples containing proximal frame coordinates
///     n_proximal: number of proximal points to compare
///     n_distal: number of distal points to compare
///     centerline: PyCenterline for the region
///     proximal_reference: list of (x, y, z) with the region of interest points from the CCTA mesh
///     distal_reference: list of (x, y, z) tuples containing distal frame coordinates
///
/// Returns:
///     Tuple of two floats with proximal and distal scaling distance
/// Example:
///     >>> import multimodars as mm
#[pyfunction]
pub fn find_proximal_distal_scaling(
    anomalous_points: Vec<(f64, f64, f64)>,
    n_proximal: usize,
    n_distal: usize,
    centerline: PyCenterline,
    proximal_reference: Vec<(f64, f64, f64)>,
    distal_reference: Vec<(f64, f64, f64)>,
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

/// Find optimal scaling for aorta using intramural region.
///
/// Args:
///     intramural_points: list of (x, y, z) tuples containing proximal frame coordinates
///     reference_opints: list of (x, y, z) with the region of interest points from the CCTA mesh
///     centerline: PyCenterline of the aorta
///
/// Returns:
///     float with best scaling distance
/// Example:
///     >>> import multimodars as mm
#[pyfunction]
pub fn find_aortic_scaling(
    intramural_points: Vec<(f64, f64, f64)>,
    reference_points: Vec<(f64, f64, f64)>,
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
