mod entry;

mod io;
mod processing;
mod texture;
mod utils;

use pyo3::prelude::*;
use entry::{run_process_case, run_rest_stress_only};

/// Python wrapper around Rust pipeline.
///
/// Uses a Python‐friendly signature to allow defaults.
#[pyfunction]
#[pyo3(
    signature = (
        rest_input_path,
        stress_input_path,
        diastole_comparison_path,
        systole_comparison_path,
        // these four get defaults if not passed
        steps_best_rotation = 300usize,
        range_rotation_rad = 1.57f64,
        rest_output_path = "output/rest",
        stress_output_path = "output/stress",
        interpolation_steps = 28usize
    )
)]
fn run_process_case_py(
    rest_input_path: &str,
    stress_input_path: &str,
    diastole_comparison_path: &str,
    systole_comparison_path: &str,
    steps_best_rotation: usize,
    range_rotation_rad: f64,
    rest_output_path: &str,
    stress_output_path: &str,
    interpolation_steps: usize,
) -> PyResult<()> {
    run_process_case(
        rest_input_path,
        steps_best_rotation,
        range_rotation_rad,
        rest_output_path,
        interpolation_steps,
        stress_input_path,
        stress_output_path,
        diastole_comparison_path,
        systole_comparison_path,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (
    rest_input_path,
    stress_input_path,
    // defaults for the rest:
    steps_best_rotation = 300usize,
    range_rotation_rad = 1.57f64,
    rest_output_path = "output/rest",
    stress_output_path = "output/stress",
    interpolation_steps = 28usize
))]
fn rest_stress_py(
    rest_input_path: &str,
    stress_input_path: &str,
    steps_best_rotation: usize,
    range_rotation_rad: f64,
    rest_output_path: &str,
    stress_output_path: &str,
    interpolation_steps: usize,
) -> PyResult<()> {
    run_rest_stress_only(
        rest_input_path,
        steps_best_rotation,
        range_rotation_rad,
        rest_output_path,
        interpolation_steps,
        stress_input_path,
        stress_output_path,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// This is the module importable from Python:
///
/// ```python
/// import aaoca_package
/// aaoca_package.run_process_case_py(
///     "input/rest.csv", "input/stress.csv",
///     "out/dia.csv", "out/sys.csv"
/// )
/// ```
#[pymodule]
fn multimodars(_py: Python, m: pyo3::prelude::Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(run_process_case_py, m.clone())?)?;
    m.add_function(wrap_pyfunction!(rest_stress_py, m.clone())?)?;
    Ok(())
}
