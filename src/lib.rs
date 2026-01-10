// #![deny(clippy::unwrap_used)]

mod ccta;
mod intravascular;

use crate::ccta::binding::ccta_py::{
    adjust_diameter_centerline_morphing_simple, build_adjacency_map, clean_outlier_points,
    find_aortic_scaling, find_centerline_bounded_points_simple, find_points_by_cl_region,
    find_proximal_distal_scaling, remove_occluded_points_ray_triangle, smooth_mesh_labels,
};
use crate::intravascular::binding::align::{align_combined, align_manual, align_three_point};
use crate::intravascular::binding::classes::*;
use crate::intravascular::binding::{
    from_array_doublepair, from_array_full, from_array_single, from_array_singlepair,
    from_file_doublepair, from_file_full, from_file_single, from_file_singlepair, to_obj,
};

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

/// This is the module importable from Python
///
/// Test with the provided example data:
/// ```python
/// import multimodars as mm
/// rest, stress, dia, sys = mm.from_file_full(
///     "data/ivus_rest", "data/ivus_stress"
/// )
/// print(rest)
/// ```
#[pymodule]
fn multimodars(_py: Python, m: pyo3::prelude::Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(from_file_full, m.clone())?)?;
    m.add_function(wrap_pyfunction!(from_file_doublepair, m.clone())?)?;
    m.add_function(wrap_pyfunction!(from_file_singlepair, m.clone())?)?;
    m.add_function(wrap_pyfunction!(from_file_single, m.clone())?)?;
    m.add_function(wrap_pyfunction!(align_three_point, m.clone())?)?;
    m.add_function(wrap_pyfunction!(align_manual, m.clone())?)?;
    m.add_function(wrap_pyfunction!(align_combined, m.clone())?)?;
    m.add_function(wrap_pyfunction!(from_array_full, m.clone())?)?;
    m.add_function(wrap_pyfunction!(from_array_doublepair, m.clone())?)?;
    m.add_function(wrap_pyfunction!(from_array_singlepair, m.clone())?)?;
    m.add_function(wrap_pyfunction!(from_array_single, m.clone())?)?;
    m.add_function(wrap_pyfunction!(to_obj, m.clone())?)?;
    m.add_function(wrap_pyfunction!(
        find_centerline_bounded_points_simple,
        m.clone()
    )?)?;
    m.add_function(wrap_pyfunction!(
        remove_occluded_points_ray_triangle,
        m.clone()
    )?)?;
    m.add_function(wrap_pyfunction!(
        adjust_diameter_centerline_morphing_simple,
        m.clone()
    )?)?;
    m.add_function(wrap_pyfunction!(find_points_by_cl_region, m.clone())?)?;
    m.add_function(wrap_pyfunction!(clean_outlier_points, m.clone())?)?;
    m.add_function(wrap_pyfunction!(find_proximal_distal_scaling, m.clone())?)?;
    m.add_function(wrap_pyfunction!(find_aortic_scaling, m.clone())?)?;
    m.add_function(wrap_pyfunction!(build_adjacency_map, m.clone())?)?;
    m.add_function(wrap_pyfunction!(smooth_mesh_labels, m.clone())?)?;

    // Updated class registration
    m.add_class::<PyInputData>()?;
    m.add_class::<PyContourPoint>()?;
    m.add_class::<PyContour>()?;
    m.add_class::<PyContourType>()?;
    m.add_class::<PyFrame>()?;
    m.add_class::<PyGeometry>()?;
    m.add_class::<PyGeometryPair>()?;
    m.add_class::<PyCenterlinePoint>()?;
    m.add_class::<PyCenterline>()?;
    m.add_class::<PyRecord>()?;
    Ok(())
}
