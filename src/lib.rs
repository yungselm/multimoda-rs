// #![deny(clippy::unwrap_used)]
#![allow(clippy::too_many_arguments)]

mod ccta;
mod intravascular;
pub mod types;

use crate::intravascular::binding;
use crate::types::binding::*;

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
    m.add_function(wrap_pyfunction!(binding::from_file_full, m.clone())?)?;
    m.add_function(wrap_pyfunction!(binding::from_file_doublepair, m.clone())?)?;
    m.add_function(wrap_pyfunction!(binding::from_file_singlepair, m.clone())?)?;
    m.add_function(wrap_pyfunction!(binding::from_file_single, m.clone())?)?;
    m.add_function(wrap_pyfunction!(binding::align_three_point, m.clone())?)?;
    m.add_function(wrap_pyfunction!(binding::align_manual, m.clone())?)?;
    m.add_function(wrap_pyfunction!(binding::align_combined, m.clone())?)?;
    m.add_function(wrap_pyfunction!(binding::from_array_full, m.clone())?)?;
    m.add_function(wrap_pyfunction!(binding::from_array_doublepair, m.clone())?)?;
    m.add_function(wrap_pyfunction!(binding::from_array_singlepair, m.clone())?)?;
    m.add_function(wrap_pyfunction!(binding::from_array_single, m.clone())?)?;
    m.add_function(wrap_pyfunction!(binding::to_obj, m.clone())?)?;
    m.add_function(wrap_pyfunction!(binding::read_centerline_vtp, m.clone())?)?;
    m.add_function(wrap_pyfunction!(binding::write_centerline_vtp, m.clone())?)?;
    ccta::binding::labeling::register(&m)?;
    ccta::binding::morphing::register(&m)?;
    ccta::binding::discretizing::register(&m)?;

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
    m.add_class::<PyDiscretizedBranch>()?;
    m.add_class::<PyDiscretizedVesselTree>()?;
    Ok(())
}
