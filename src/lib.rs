mod binding;

mod io;
mod mesh_to_centerline;
mod processing;
mod texture;
mod utils;

use binding::classes::{PyContour, PyContourPoint, PyGeometry, PyGeometryPair, PyRecord};
use binding::*;
use mesh_to_centerline::create_centerline_aligned_meshes;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pyfunction]
#[pyo3(
    signature = (
        centerline_path,
        aortic_ref_pt,
        upper_ref_pt,
        lower_ref_pt,
        state="neutral",
        input_dir = "output/rest",
        output_dir = "output/rest_aligned",
        interpolation_steps = 28usize,
    )
)]
pub fn centerline_align(
    centerline_path: &str,
    aortic_ref_pt: (f64, f64, f64),
    upper_ref_pt: (f64, f64, f64),
    lower_ref_pt: (f64, f64, f64),
    state: &str,
    input_dir: &str,
    output_dir: &str,
    interpolation_steps: usize,
) -> Result<(PyGeometry, PyGeometry), PyErr> {
    let (dia_geom, sys_geom) = create_centerline_aligned_meshes(
        state,
        centerline_path,
        input_dir,
        output_dir,
        interpolation_steps,
        aortic_ref_pt,
        upper_ref_pt,
        lower_ref_pt,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let py_dia_geom = PyGeometry::from(dia_geom);
    let py_sys_geom = PyGeometry::from(sys_geom);
    Ok((py_dia_geom, py_sys_geom))
}

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
    m.add_function(wrap_pyfunction!(centerline_align, m.clone())?)?;
    m.add_function(wrap_pyfunction!(create_catheter_contours, m.clone())?)?;
    m.add_function(wrap_pyfunction!(geometry_from_array, m.clone())?)?;
    m.add_function(wrap_pyfunction!(from_array_full, m.clone())?)?;
    m.add_function(wrap_pyfunction!(from_array_doublepair, m.clone())?)?;
    m.add_function(wrap_pyfunction!(from_array_singlepair, m.clone())?)?;    

    // Updated class registration
    m.add_class::<PyContourPoint>()?;
    m.add_class::<PyContour>()?;
    m.add_class::<PyGeometry>()?;
    m.add_class::<PyGeometryPair>()?;
    m.add_class::<PyRecord>()?;
    Ok(())
}
