mod binding;

mod io;
mod processing;
mod texture;
mod utils;
mod mesh_to_centerline;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use binding::{from_file_full, from_file_doublepair, from_file_singlepair, from_file_single};
use binding::classes::{PyContour, PyContourPoint, PyGeometry, PyGeometryPair};
use mesh_to_centerline::create_centerline_aligned_meshes;

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
        lower_ref_pt)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let py_dia_geom = PyGeometry::from(dia_geom);
    let py_sys_geom = PyGeometry::from(sys_geom);
    Ok((py_dia_geom, py_sys_geom))
}

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
    m.add_function(wrap_pyfunction!(from_file_full, m.clone())?)?;
    m.add_function(wrap_pyfunction!(from_file_doublepair, m.clone())?)?;
    m.add_function(wrap_pyfunction!(from_file_singlepair, m.clone())?)?;    
    m.add_function(wrap_pyfunction!(from_file_single, m.clone())?)?;
    m.add_function(wrap_pyfunction!(centerline_align, m.clone())?)?;

    // Updated class registration
    m.add_class::<PyContourPoint>()?;
    m.add_class::<PyContour>()?;
    m.add_class::<PyGeometry>()?;
    m.add_class::<PyGeometryPair>()?;
    Ok(())
}
