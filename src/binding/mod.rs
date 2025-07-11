pub mod classes;
pub mod entry_arr;
pub mod entry_file;

use crate::io::input::{Contour, ContourPoint, Record};
use classes::{PyContour, PyContourPoint, PyGeometry, PyGeometryPair, PyRecord};
use entry_file::{
    from_file_doublepair_rs, from_file_full_rs, from_file_single_rs, from_file_singlepair_rs,
};
use entry_arr::*;
use pyo3::prelude::*;

/// This function processes four geometries (rest, stress, diastole, systole) in parallel,
/// based on the given input csv files and configuration parameters. It returns
/// four geometry pairs corresponding to each phase.
///
///         `Rest`:                             `Stress`:
///         diastole  ---------------------->   diastole
///            |                                   |
///            |                                   |
///            v                                   v
///         systole   ---------------------->   systole
///
/// # Arguments
/// * `rest_input_path` - Path to the input file for the REST image.
/// * `stress_input_path` - Path to the input file for the STRESS image.
/// * `steps_best_rotation` - Number of steps for finding the best rotation (default: 300).
/// * `range_rotation_rad` - Max rotation range in radians (default: 1.57).
/// * `rest_output_path` - Output path for processed REST geometry (default: "output/rest").
/// * `stress_output_path` - Output path for processed STRESS geometry (default: "output/stress").
/// * `diastole_output_path` - Output path for DIASTOLE geometry (default: "output/diastole").
/// * `systole_output_path` - Output path for SYSTOLE geometry (default: "output/systole").
/// * `interpolation_steps` - Number of interpolation steps to use (default: 28).
/// * `image_center` - Center of the image as a tuple (x, y) (default: (4.5, 4.5)).
/// * `radius` - Radius to use for processing (default: 0.5).
/// * `n_points` - Number of boundary points to generate (default: 20).
///
/// # Expected format .csv file, e.g.:
///  --------------------------------------------------------------------
///  |      185     |       5.32     |      2.37       |        0.0     |
///  |      ...     |       ...      |      ...        |        ...     |
///  No headers -> frame index, x-coord [mm], y-coord [mm], z-coord [mm]           
///
/// # Returns
/// A tuple of four `PyGeometryPair` objects: `(rest, stress, diastole, systole)`.
///
/// # Raises
/// `RuntimeError` if any part of the pipeline fails.
///
/// # Example usage
/// import multimodars as mm
/// rest, stress, diastole, systole = mm.from_file_full("data/ivus_rest", "data/ivus_stress")
///
/// This function is a Python wrapper around the internal Rust implementation.
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
        interpolation_steps = 28usize,
        image_center = (4.5f64, 4.5f64),
        radius = 0.5f64,
        n_points = 20u32,
    )
)]
pub fn from_file_full(
    rest_input_path: &str,
    stress_input_path: &str,
    steps_best_rotation: usize,
    range_rotation_rad: f64,
    rest_output_path: &str,
    stress_output_path: &str,
    diastole_output_path: &str,
    systole_output_path: &str,
    interpolation_steps: usize,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
) -> PyResult<(
    PyGeometryPair,
    PyGeometryPair,
    PyGeometryPair,
    PyGeometryPair,
)> {
    let (rest_pair, stress_pair, dia_pair, sys_pair) = from_file_full_rs(
        rest_input_path,
        steps_best_rotation,
        range_rotation_rad,
        rest_output_path,
        interpolation_steps,
        stress_input_path,
        stress_output_path,
        diastole_output_path,
        systole_output_path,
        image_center,
        radius,
        n_points,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let py_rest = rest_pair.into();
    let py_stress = stress_pair.into();
    let py_dia = dia_pair.into();
    let py_sys = sys_pair.into();

    Ok((py_rest, py_stress, py_dia, py_sys))
}

/// This function processes four geometries (rest, stress, diastole, systole) in parallel,
/// based on the given input csv files and configuration parameters. It returns
/// two geometry pairs corresponding to each phase.
///
///         `Rest`:                             `Stress`:
///         diastole                            diastole
///            |                                   |
///            |                                   |
///            v                                   v
///         systole                             systole
///
/// # Arguments
/// * `rest_input_path` - Path to the input file for the REST image.
/// * `stress_input_path` - Path to the input file for the STRESS image.
/// * `steps_best_rotation` - Number of steps for finding the best rotation (default: 300).
/// * `range_rotation_rad` - Max rotation range in radians (default: 1.57).
/// * `rest_output_path` - Output path for processed REST geometry (default: "output/rest").
/// * `stress_output_path` - Output path for processed STRESS geometry (default: "output/stress").
/// * `interpolation_steps` - Number of interpolation steps to use (default: 28).
/// * `image_center` - Center of the image as a tuple (x, y) (default: (4.5, 4.5)).
/// * `radius` - Radius to use for processing (default: 0.5).
/// * `n_points` - Number of boundary points to generate (default: 20).
///
/// # Expected format .csv file, e.g.:
///  --------------------------------------------------------------------
///  |      185     |       5.32     |      2.37       |        0.0     |
///  |      ...     |       ...      |      ...        |        ...     |
///  No headers -> frame index, x-coord [mm], y-coord [mm], z-coord [mm]           
///
/// # Returns
/// A tuple of two `PyGeometryPair` objects: `(rest, stress)`.
///
/// # Raises
/// `RuntimeError` if any part of the pipeline fails.
///
/// # Example usage
/// import multimodars as mm
/// rest, stress, diastole, systole = mm.from_file_doublepair("data/ivus_rest", "data/ivus_stress")
///
/// This function is a Python wrapper around the internal Rust implementation.
#[pyfunction]
#[pyo3(signature = (
    rest_input_path,
    stress_input_path,
    // defaults for the rest:
    steps_best_rotation = 300usize,
    range_rotation_rad = 1.57f64,
    rest_output_path = "output/rest",
    stress_output_path = "output/stress",
    interpolation_steps = 28usize,
    image_center = (4.5f64, 4.5f64),
    radius = 0.5f64,
    n_points = 20u32,
))]
pub fn from_file_doublepair(
    rest_input_path: &str,
    stress_input_path: &str,
    steps_best_rotation: usize,
    range_rotation_rad: f64,
    rest_output_path: &str,
    stress_output_path: &str,
    interpolation_steps: usize,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
) -> PyResult<(PyGeometryPair, PyGeometryPair)> {
    let (rest_pair, stress_pair) = from_file_doublepair_rs(
        rest_input_path,
        steps_best_rotation,
        range_rotation_rad,
        rest_output_path,
        interpolation_steps,
        stress_input_path,
        stress_output_path,
        image_center,
        radius,
        n_points,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let py_rest = rest_pair.into();
    let py_stress = stress_pair.into();

    Ok((py_rest, py_stress))
}

/// This function processes four geometries (rest, stress, diastole, systole) in parallel,
/// based on the given input csv files and configuration parameters. It returns
/// a geometry pair corresponding to the phase.
///
///                           `Rest`/`Stress`:
///                              diastole
///                                  |
///                                  |
///                                  v
///                               systole
///
/// # Arguments
/// * `input_path` - Path to the input file for the REST image.
/// * `steps_best_rotation` - Number of steps for finding the best rotation (default: 300).
/// * `range_rotation_rad` - Max rotation range in radians (default: 1.57).
/// * `output_path` - Output path for processed STRESS geometry.
/// * `interpolation_steps` - Number of interpolation steps to use (default: 28).
/// * `image_center` - Center of the image as a tuple (x, y) (default: (4.5, 4.5)).
/// * `radius` - Radius to use for processing (default: 0.5).
/// * `n_points` - Number of boundary points to generate (default: 20).
///
/// # Expected format .csv file, e.g.:
///  --------------------------------------------------------------------
///  |      185     |       5.32     |      2.37       |        0.0     |
///  |      ...     |       ...      |      ...        |        ...     |
///  No headers -> frame index, x-coord [mm], y-coord [mm], z-coord [mm]           
///
/// # Returns
/// A `PyGeometryPair` object: `(rest or `stress`)`.
///
/// # Raises
/// `RuntimeError` if any part of the pipeline fails.
///
/// # Example usage
/// import multimodars as mm
/// rest, stress, diastole, systole = mm.from_file_singlepair("data/ivus_rest", "output/rest")
///
/// This function is a Python wrapper around the internal Rust implementation.
#[pyfunction]
#[pyo3(signature = (
    input_path,
    output_path,
    // defaults for the rest:
    steps_best_rotation = 300usize,
    range_rotation_rad = 1.57f64,
    interpolation_steps = 28usize,
    image_center = (4.5f64, 4.5f64),
    radius = 0.5f64,
    n_points = 20u32,
))]
pub fn from_file_singlepair(
    input_path: &str,
    output_path: &str,
    steps_best_rotation: usize,
    range_rotation_rad: f64,
    interpolation_steps: usize,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
) -> PyResult<PyGeometryPair> {
    let geom_pair = from_file_singlepair_rs(
        input_path,
        steps_best_rotation,
        range_rotation_rad,
        output_path,
        interpolation_steps,
        image_center,
        radius,
        n_points,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let py_pair = geom_pair.into();

    Ok(py_pair)
}

#[pyfunction]
#[pyo3(signature = (
    input_path,
    // defaults for the rest:
    steps_best_rotation = 300usize,
    range_rotation_rad = 1.57f64,
    output_path = "output/single",
    diastole = true,
    image_center = (4.5f64, 4.5f64),
    radius = 0.5f64,
    n_points = 20u32,
))]
pub fn from_file_single(
    input_path: &str,
    steps_best_rotation: usize,
    range_rotation_rad: f64,
    output_path: &str,
    diastole: bool,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
) -> PyResult<PyGeometry> {
    let geom = from_file_single_rs(
        input_path,
        steps_best_rotation,
        range_rotation_rad,
        output_path,
        diastole,
        image_center,
        radius,
        n_points,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let py_geom = geom.into();

    Ok(py_geom)
}

#[pyfunction]
#[pyo3(signature = (points, image_center = (4.5f64, 4.5f64), radius = 0.5f64, n_points = 20u32))]
pub fn create_catheter_contours(
    points: Vec<PyContourPoint>,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
) -> PyResult<Vec<PyContour>> {
    // 1) Convert Vec<PyContourPoint> → Vec<ContourPoint> using your &PyContourPoint → ContourPoint impl
    let rust_pts: Vec<ContourPoint> = points.iter().map(ContourPoint::from).collect();

    // 2) Call the Rust-level function
    let rust_contours: Vec<Contour> =
        Contour::create_catheter_contours(&rust_pts, image_center, radius, n_points)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    // 3) Convert Vec<Contour> → Vec<PyContour> using your &Contour → PyContour impl
    let py_contours: Vec<PyContour> = rust_contours.iter().map(PyContour::from).collect();

    Ok(py_contours)
}

#[pyfunction]
#[pyo3(signature = (
    contours,
    reference_point,
    steps_best_rotation = 300usize,
    range_rotation_rad = 1.57f64,
    image_center = (4.5f64, 4.5f64),
    radius = 0.5f64,
    n_points = 20u32,
    label = "None",
    records = None,
    delta = 0.1f64,
    max_rounds = 5,
    diastole = true,
    sort = true,
    write_obj=false,
    output_path="output/single"
))]
pub fn geometry_from_array(
    contours: Vec<PyContour>,
    reference_point: PyContourPoint,
    steps_best_rotation: usize,
    range_rotation_rad: f64,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
    label: &str,
    records: Option<Vec<PyRecord>>,
    delta: f64,
    max_rounds: usize,
    diastole: bool,
    sort: bool,
    write_obj: bool,
    output_path: &str,
) -> PyResult<PyGeometry> {
    let contours_rs: Vec<Contour> = contours
        .iter()
        .map(|pyc| pyc.to_rust_contour().unwrap())
        .collect();

    let reference_point_rs: ContourPoint = (&reference_point).into();

    let records_rs: Option<Vec<Record>> =
        records.map(|vec_py| vec_py.into_iter().map(|py| py.to_rust_record()).collect());

    let geom_rs = geometry_from_array_rs(
        contours_rs,
        reference_point_rs,
        steps_best_rotation,
        range_rotation_rad,
        image_center,
        radius,
        n_points,
        label,
        records_rs,
        delta,
        max_rounds,
        diastole,
        sort,
        write_obj,
        output_path,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let py_geom = PyGeometry::from(geom_rs);
    Ok(py_geom)
}

#[pyfunction]
#[pyo3(
    signature = (
        rest_geometry_dia,
        rest_geometry_sys,
        stress_geometry_dia,
        stress_geometry_sys,
        steps_best_rotation = 300usize,
        range_rotation_rad = 1.57f64,
        interpolation_steps = 28usize,
        rest_output_path = "output/rest",
        stress_output_path = "output/stress",
        diastole_output_path = "output/diastole",
        systole_output_path = "output/systole",
    )
)]
pub fn from_array_full(
    rest_geometry_dia: PyGeometry,
    rest_geometry_sys: PyGeometry,
    stress_geometry_dia: PyGeometry,
    stress_geometry_sys: PyGeometry,
    steps_best_rotation: usize,
    range_rotation_rad: f64,
    interpolation_steps: usize,
    rest_output_path: &str,
    stress_output_path: &str,
    diastole_output_path: &str,
    systole_output_path: &str,
) -> PyResult<(
    PyGeometryPair,
    PyGeometryPair,
    PyGeometryPair,
    PyGeometryPair,
)> {
    let (rest_pair, stress_pair, dia_pair, sys_pair) = from_array_full_rs(
        rest_geometry_dia.to_rust_geometry(),
        rest_geometry_sys.to_rust_geometry(),
        stress_geometry_dia.to_rust_geometry(),
        stress_geometry_sys.to_rust_geometry(),
        steps_best_rotation,
        range_rotation_rad,
        interpolation_steps,
        rest_output_path,
        stress_output_path,
        diastole_output_path,
        systole_output_path,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let py_rest = rest_pair.into();
    let py_stress = stress_pair.into();
    let py_dia = dia_pair.into();
    let py_sys = sys_pair.into();

    Ok((py_rest, py_stress, py_dia, py_sys))
}

#[pyfunction]
#[pyo3(
    signature = (
        rest_geometry_dia,
        rest_geometry_sys,
        stress_geometry_dia,
        stress_geometry_sys,
        steps_best_rotation = 300usize,
        range_rotation_rad = 1.57f64,
        interpolation_steps = 28usize,
        rest_output_path = "output/rest",
        stress_output_path = "output/stress",
    )
)]
pub fn from_array_doublepair(
    rest_geometry_dia: PyGeometry,
    rest_geometry_sys: PyGeometry,
    stress_geometry_dia: PyGeometry,
    stress_geometry_sys: PyGeometry,
    steps_best_rotation: usize,
    range_rotation_rad: f64,
    interpolation_steps: usize,
    rest_output_path: &str,
    stress_output_path: &str,
) -> PyResult<(
    PyGeometryPair,
    PyGeometryPair,
)> {
    let (rest_pair, stress_pair) = from_array_doublepair_rs(
        rest_geometry_dia.to_rust_geometry(),
        rest_geometry_sys.to_rust_geometry(),
        stress_geometry_dia.to_rust_geometry(),
        stress_geometry_sys.to_rust_geometry(),
        steps_best_rotation,
        range_rotation_rad,
        interpolation_steps,
        rest_output_path,
        stress_output_path,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let py_rest = rest_pair.into();
    let py_stress = stress_pair.into();

    Ok((py_rest, py_stress))
}

#[pyfunction]
#[pyo3(
    signature = (
        geometry_dia,
        geometry_sys,
        output_path,
        steps_best_rotation = 300usize,
        range_rotation_rad = 1.57f64,
        interpolation_steps = 28usize,
    )
)]
pub fn from_array_singlepair(
    geometry_dia: PyGeometry,
    geometry_sys: PyGeometry,
    output_path: &str,
    steps_best_rotation: usize,
    range_rotation_rad: f64,
    interpolation_steps: usize,
) -> PyResult<PyGeometryPair> {
    let pair = from_array_singlepair_rs(
        geometry_dia.to_rust_geometry(),
        geometry_sys.to_rust_geometry(),
        output_path,
        steps_best_rotation,
        range_rotation_rad,
        interpolation_steps,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let py_pair = pair.into();

    Ok(py_pair)
}