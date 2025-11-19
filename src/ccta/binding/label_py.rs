// src/ccta/binding/label_py.rs
use crate::intravascular::binding::classes::PyCenterline;
use crate::ccta::label_mesh::label_coronary::find_centerline_bounded_points;
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