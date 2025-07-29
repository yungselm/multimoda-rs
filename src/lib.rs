mod binding;

mod centerline_align;
mod io;
mod processing;
mod texture;
mod utils;

use binding::align::{align_manual, align_three_point};
use binding::classes::*;
use binding::*;
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
    // align hausdorff missing
    m.add_function(wrap_pyfunction!(create_catheter_geometry, m.clone())?)?;
    m.add_function(wrap_pyfunction!(geometry_from_array, m.clone())?)?;
    m.add_function(wrap_pyfunction!(from_array_full, m.clone())?)?;
    m.add_function(wrap_pyfunction!(from_array_doublepair, m.clone())?)?;
    m.add_function(wrap_pyfunction!(from_array_singlepair, m.clone())?)?;
    m.add_function(wrap_pyfunction!(to_obj, m.clone())?)?;

    // Updated class registration
    m.add_class::<PyContourPoint>()?;
    m.add_class::<PyContour>()?;
    m.add_class::<PyGeometry>()?;
    m.add_class::<PyGeometryPair>()?;
    m.add_class::<PyCenterlinePoint>()?;
    m.add_class::<PyCenterline>()?;
    m.add_class::<PyRecord>()?;
    Ok(())
}
