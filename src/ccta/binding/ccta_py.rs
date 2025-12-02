// src/ccta/binding/label_py.rs
use crate::intravascular::binding::classes::{PyCenterline, PyFrame};
use crate::ccta::adjust_mesh::label_coronary::find_centerline_bounded_points;
use crate::ccta::adjust_mesh::label_coronary::{Triangle, remove_occluded_points_ray_triangle_rust};
use crate::ccta::adjust_mesh::scale_coronary::{centerline_based_diameter_morphing, find_points_by_cl_region_rs};
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
    let result_points = find_centerline_bounded_points(
        rust_centerline, 
        &points, 
        radius
    );
    
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

    let result_points = centerline_based_diameter_morphing(
        &rust_centerline,
        &points,
        diameter_adjustment_mm,
    );

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
) -> PyResult<(Vec<(f64, f64, f64)>, Vec<(f64, f64, f64)>, Vec<(f64, f64, f64)>)> {
    let rust_centerline = centerline.to_rust_centerline();
    let rust_frames: Vec<crate::intravascular::io::geometry::Frame> = frames
        .into_iter()
        .map(|f| f.to_rust_frame())
        .collect::<Result<_, _>>()?;

    let result_points = find_points_by_cl_region_rs(
        &rust_centerline,
        &rust_frames,
        &points,
    );

    Ok(result_points)
}