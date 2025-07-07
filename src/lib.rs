mod binding;

mod io;
mod processing;
mod texture;
mod utils;
mod mesh_to_centerline;

use pyo3::prelude::*;
use binding::{from_file_full_py, from_file_state_py, from_file_single_py};
use pyo3::wrap_pyfunction;
use binding::classes::{PyContour, PyContourPoint, PyGeometry, PyGeometryPair};

// fn from_file()

/// This is the module importable from Python:
///
/// ```python
/// import multimodars as mm
/// mm.run_process_case_py(
///     "test_data/rest_csv_files", "test_data/stress_csv_files",
///     "output/dia.csv", "out/sys.csv"
/// )
/// ```
#[pymodule]
fn multimodars(_py: Python, m: pyo3::prelude::Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(from_file_full_py, m.clone())?)?;
    m.add_function(wrap_pyfunction!(from_file_state_py, m.clone())?)?;
    m.add_function(wrap_pyfunction!(from_file_single_py, m.clone())?)?;

    // Updated class registration
    m.add_class::<PyContourPoint>()?;
    m.add_class::<PyContour>()?;
    m.add_class::<PyGeometry>()?;
    m.add_class::<PyGeometryPair>()?;
    Ok(())
}
