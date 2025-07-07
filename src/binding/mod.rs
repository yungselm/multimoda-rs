pub mod entry;
pub mod classes;

use entry::{from_file_full, from_file_state, from_file_single};
use pyo3::prelude::*;
use classes::{PyContour, PyContourPoint, PyGeometry, PyGeometryPair};

/// Python wrapper around Rust pipeline.
///
/// Uses a Python‐friendly signature to allow defaults.
#[pyfunction]
#[pyo3(
    signature = (
        rest_input_path,
        stress_input_path,
        // these four get defaults if not passed
        steps_best_rotation = 300usize,
        range_rotation_rad = 1.57f64,
        rest_output_path = "output/rest",
        stress_output_path = "output/stress",
        diastole_output_path = "output/diastole",
        systole_output_path = "output/systole",
        interpolation_steps = 28usize
    )
)]
pub fn from_file_full_py(
    rest_input_path: &str,
    stress_input_path: &str,
    steps_best_rotation: usize,
    range_rotation_rad: f64,
    rest_output_path: &str,
    stress_output_path: &str,
    diastole_output_path: &str,
    systole_output_path: &str,
    interpolation_steps: usize,
) -> PyResult<(PyGeometryPair, PyGeometryPair, PyGeometryPair, PyGeometryPair)> {
    let (rest_pair, stress_pair, dia_pair, sys_pair) = from_file_full(
        rest_input_path,
        steps_best_rotation,
        range_rotation_rad,
        rest_output_path,
        interpolation_steps,
        stress_input_path,
        stress_output_path,
        diastole_output_path,
        systole_output_path,
    ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let py_rest = rest_pair.into();
    let py_stress = stress_pair.into();
    let py_dia = dia_pair.into();
    let py_sys = sys_pair.into();

    Ok((py_rest, py_stress, py_dia, py_sys))
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
pub fn from_file_state_py(
    rest_input_path: &str,
    stress_input_path: &str,
    steps_best_rotation: usize,
    range_rotation_rad: f64,
    rest_output_path: &str,
    stress_output_path: &str,
    interpolation_steps: usize,
) -> PyResult<(PyGeometryPair, PyGeometryPair)> {
    let (rest_pair, stress_pair) = from_file_state(
        rest_input_path,
        steps_best_rotation,
        range_rotation_rad,
        rest_output_path,
        interpolation_steps,
        stress_input_path,
        stress_output_path,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let py_rest = rest_pair.into();
    let py_stress = stress_pair.into();

    Ok((py_rest, py_stress))
}

#[pyfunction]
#[pyo3(signature = (
    input_path,
    // defaults for the rest:
    steps_best_rotation = 300usize,
    range_rotation_rad = 1.57f64,
    output_path = "output/single",
))]
pub fn from_file_single_py(
    input_path: &str,
    steps_best_rotation: usize,
    range_rotation_rad: f64,
    output_path: &str,
) -> PyResult<(PyGeometry)> {
    let geom = from_file_single(
        input_path,
        steps_best_rotation,
        range_rotation_rad,
        output_path,
        true,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let py_geom = geom.into();

    Ok(py_geom)
}