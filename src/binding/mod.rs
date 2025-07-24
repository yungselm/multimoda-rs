pub mod classes;
pub mod entry_arr;
pub mod entry_file;
pub mod align;

use crate::io::input::{Contour, ContourPoint, Record};
use classes::{PyContour, PyContourPoint, PyGeometry, PyGeometryPair, PyRecord};
use entry_arr::*;
use entry_file::{
    from_file_doublepair_rs, from_file_full_rs, from_file_single_rs, from_file_singlepair_rs,
};
use pyo3::prelude::*;

/// Processes four geometries in parallel.
///
/// Pipeline:
///
/// .. code-block:: text
///
///    Rest:                    Stress:
///    diastole  -------------->  diastole
///       |                         |
///       v                         v
///    systole  -------------->  systole
///
/// Arguments:
///
/// - ``rest_input_path`` – Path to REST input file
/// - ``stress_input_path`` – Path to STRESS input file
/// - ``steps_best_rotation`` (default: 300) – Rotation steps
/// - ``range_rotation_rad`` (default: 1.57) – Rotation range in radians
/// - ``rest_output_path`` (default: "output/rest")
/// - ``stress_output_path`` (default: "output/stress")
/// - ``diastole_output_path`` (default: "output/diastole")
/// - ``systole_output_path`` (default: "output/systole")
/// - ``interpolation_steps`` (default: 28)
/// - ``image_center`` (default: (4.5, 4.5))
/// - ``radius`` (default: 0.5)
/// - ``n_points`` (default: 20)
///
/// CSV format:
///
/// .. code-block:: text
///
///    Frame Index, X-coord (mm), Y-coord (mm), Z-coord (mm)
///    185, 5.32, 2.37, 0.0
///    ...
///
/// Returns:
///
/// A 4‑tuple of ``PyGeometryPair`` for (rest, stress, diastole, systole).
///
/// Example:
///
/// .. code-block:: python
///
///    import multimodars as mm
///    rest, stress, dia, sys = mm.from_file_full(
///        "data/ivus_rest", "data/ivus_stress"
///    )
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

/// Processes two geometries in parallel.
///
/// Pipeline:
///
/// .. code-block:: text
///
///    Rest:                    Stress:
///    diastole                  diastole
///       |                         |
///       v                         v
///    systole                   systole
///
/// Arguments:
///
/// - ``rest_input_path`` – Path to REST input file
/// - ``stress_input_path`` – Path to STRESS input file
/// - ``steps_best_rotation`` (default: 300) – Rotation steps
/// - ``range_rotation_rad`` (default: 1.57) – Rotation range in radians
/// - ``rest_output_path`` (default: "output/rest")
/// - ``stress_output_path`` (default: "output/stress")
/// - ``interpolation_steps`` (default: 28)
/// - ``image_center`` (default: (4.5, 4.5))
/// - ``radius`` (default: 0.5)
/// - ``n_points`` (default: 20)
///
/// CSV format:
///
/// .. code-block:: text
///
///    Frame Index, X-coord (mm), Y-coord (mm), Z-coord (mm)
///    185, 5.32, 2.37, 0.0
///    ...
///
/// Returns:
///
/// A 2‑tuple of ``PyGeometryPair`` for (rest, stress).
///
/// Example:
///
/// .. code-block:: python
///
///    import multimodars as mm
///    rest, stress, dia, sys = mm.from_file_full(
///        "data/ivus_rest", "data/ivus_stress"
///    )
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

/// Processes two geometries (rest and stress) in parallel from an input CSV,
/// returning a single ``PyGeometryPair`` for the chosen phase.
///
/// .. code-block:: text
///
///    Rest/Stress pipeline:
///       diastole
///         │
///         ▼
///       systole
///
/// Arguments
/// ---------
///
/// - ``input_path``: Path to the input CSV file.  
/// - ``output_path``: Path to write the processed geometry.  
/// - ``steps_best_rotation`` (default: 300): Number of rotation steps.  
/// - ``range_rotation_rad`` (default: 1.57): Rotation range in radians.  
/// - ``interpolation_steps`` (default: 28): Number of interpolation steps.  
/// - ``image_center`` (default: (4.5, 4.5)): Center coordinates (x, y).  
/// - ``radius`` (default: 0.5): Processing radius.  
/// - ``n_points`` (default: 20): Number of boundary points.  
///
/// CSV Format
/// ----------
///
/// The CSV must have **no header**. Each row is:
///
/// .. code-block:: text
///
///    185, 5.32, 2.37, 0.0
///    ...
///
/// Returns
/// -------
///
/// A single ``PyGeometryPair`` for (rest or stress) geometry.
///
/// Raises
/// ------
///
/// ``RuntimeError`` if the Rust pipeline fails.
///
/// Example
/// -------
///
/// .. code-block:: python
///
///    import multimodars as mm
///    pair = mm.from_file_singlepair(
///        "data/ivus_rest.csv",
///        "output/rest"
///    )
///
/// This is a thin Python wrapper around the Rust implementation.
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

/// Processes a single geometry (either diastole or systole) from an IVUS CSV file.
///
/// .. code-block:: text
///
///    Rest/Stress pipeline (choose phase via `diastole` flag):
///      e.g. diastole
///
/// Arguments
/// ---------
///
/// - ``input_path``: Path to the input CSV (no header; columns: frame, x, y, z).  
/// - ``steps_best_rotation`` (default: 300): Number of steps to search for best rotation.  
/// - ``range_rotation_rad`` (default: 1.57): Max rotation in radians.  
/// - ``output_path`` (default: "output/single"): Where to write the processed geometry.  
/// - ``diastole`` (default: true): If true, process the diastole phase; otherwise systole.  
/// - ``image_center`` (default: (4.5, 4.5)): (x, y) center for processing.  
/// - ``radius`` (default: 0.5): Radius around center to consider.  
/// - ``n_points`` (default: 20): Number of boundary points to generate.  
///
/// Returns
/// -------
///
/// A ``PyGeometry`` containing the processed contour for the chosen phase.
///
/// Raises
/// ------
///
/// ``RuntimeError`` if the underlying Rust pipeline fails.
///
/// Example
/// -------
///
/// .. code-block:: python
///
///    import multimodars as mm
///    geom = mm.from_file_single(
///        "data/ivus.csv",
///        steps_best_rotation=300,
///        range_rotation_rad=1.57,
///        output_path="out/single",
///        diastole=False
///    )
///
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

/// Generate circular catheter contour approximations from boundary points.
///
/// Given a list of boundary ``PyContourPoint``, this function computes
/// smoothed circular contours approximating the catheter wall around
/// those points.
///
/// .. code-block:: text
///
///    Input: list of (x, y) points around catheter boundary
///    Output: list of closed contours at evenly spaced angles
///
/// Arguments
/// ---------
///
/// - ``points``: Vector of ``PyContourPoint`` defining the catheter outline.  
/// - ``image_center`` (default: (4.5, 4.5)): Center of the image (x, y).  
/// - ``radius`` (default: 0.5): Approximation radius around each point.  
/// - ``n_points`` (default: 20): Number of points per generated contour.  
///
/// Returns
/// -------
///
/// A ``Vec[PyContour]`` containing the generated closed contours.
///
/// Raises
/// ------
///
/// ``RuntimeError`` if contour generation fails in Rust.
///
/// Example
/// -------
///
/// .. code-block:: python
///
///    import multimodars as mm
///    points = [mm.PyContourPoint(x, y, z) for x, y, z in data]
///    contours = mm.create_catheter_contours(points, image_center=(4.5,4.5))
///
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

/// Process an existing ``PyGeometry`` by optionally reordering, aligning,
/// and refining its contours, walls, and catheter data based on various criteria.
///
/// This wraps the internal Rust function ``geometry_from_array_rs``, which:
/// 1. Builds catheter contours (if ``n_points > 0``),  
/// 2. Optionally reorders contours using provided ``records`` and z‑coordinate sorting,  
/// 3. Aligns frames and refines the contour ordering via dynamic programming or 2‑opt,  
/// 4. Smooths the final geometry,  
/// 5. Optionally writes OBJ meshes.
///
/// Arguments
/// ---------
///
/// - ``geometry``: The input ``PyGeometry`` (with ``contours``, ``walls``, and a ``reference_point``).  
/// - ``steps_best_rotation`` (default: 300): Number of rotation steps for frame alignment.  
/// - ``range_rotation_rad`` (default: 1.57): Angular range (in radians) for alignment.  
/// - ``image_center`` (default: (4.5, 4.5)): Center (x, y) for catheter contour generation.  
/// - ``radius`` (default: 0.5): Radius around ``image_center`` for catheter contours.  
/// - ``n_points`` (default: 20): Number of points per catheter contour; set to 0 to skip.  
/// - ``label`` (default: "None"): Label tag applied to the output geometry.  
/// - ``records`` (default: None): Optional list of ``PyRecord`` entries; if provided, contours are reordered to match the chronological record order using z‑coordinates.  
/// - ``delta`` (default: 0.1): Penalty weight for non‑sequential jumps when building the cost matrix.  
/// - ``max_rounds`` (default: 5): Maximum iterations for the ``refine_ordering`` loop.  
/// - ``diastole`` (default: true): Phase flag used during initial reorder by `records`.  
/// - ``sort`` (default: true): If true, applies ``refine_ordering`` after an initial alignment; otherwise only aligns once.  
/// - ``write_obj`` (default: false): If true, exports OBJ meshes to ``output_path``.  
/// - ``output_path`` (default: "output/single"): Directory path for OBJ exports (if enabled).
///
/// Returns
/// -------
///
/// A new ``PyGeometry`` instance containing reordered, aligned, and smoothed contours.
///
/// Raises
/// ------
///
/// ``RuntimeError`` if any Rust‑side processing step fails.
///
/// Example
/// -------
///
/// .. code-block:: python
///
///    import multimodars as mm
///    # Suppose ``geo`` is an existing PyGeometry from earlier processing:
///    refined = mm.geometry_from_array(
///        geo,
///        steps_best_rotation=200,
///        range_rotation_rad=1.0,
///        records=my_records,
///        delta=0.2,
///        max_rounds=3,
///        sort=True,
///        write_obj=True,
///        output_path="out/mesh"
///    )
///
#[pyfunction]
#[pyo3(signature = (
    geometry,
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
    geometry: PyGeometry,
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
    let contours_rs: Vec<Contour> = geometry
        .contours
        .iter()
        .map(|pyc| pyc.to_rust_contour().unwrap())
        .collect();

    let walls_rs: Vec<Contour> = geometry
        .walls
        .iter()
        .map(|pyc| pyc.to_rust_contour().unwrap())
        .collect();

    let reference_point_rs: ContourPoint = (&geometry.reference_point).into();

    let records_rs: Option<Vec<Record>> =
        records.map(|vec_py| vec_py.into_iter().map(|py| py.to_rust_record()).collect());

    let geom_rs = geometry_from_array_rs(
        contours_rs,
        walls_rs,
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

/// Process four existing ``PyGeometry`` objects (rest‑dia, rest‑sys, stress‑dia, stress‑sys)
/// in parallel, aligning and interpolating between phases.
///
/// .. code-block:: text
///
///    Rest           Stress
///    diastole ──▶ diastole
///       │            │
///       ▼            ▼
///    systole ──▶ systole
///
/// Arguments
/// ---------
///
/// - ``rest_geometry_dia``: Input ``PyGeometry`` at diastole for REST.  
/// - ``rest_geometry_sys``: Input ``PyGeometry`` at systole for REST.  
/// - ``stress_geometry_dia``: Input ``PyGeometry`` at diastole for STRESS.  
/// - ``stress_geometry_sys``: Input ``PyGeometry`` at systole for STRESS.  
/// - ``steps_best_rotation`` (default: 300): Number of rotation steps for alignment.  
/// - ``range_rotation_rad`` (default: 1.57): Angular range in radians for alignment.  
/// - ``interpolation_steps`` (default: 28): Number of interpolation steps between phases.  
/// - ``rest_output_path`` (default: "output/rest"): Output directory for REST results.  
/// - ``stress_output_path`` (default: "output/stress"): Output directory for STRESS results.  
/// - ``diastole_output_path`` (default: "output/diastole"): Output for interpolated diastole.  
/// - ``systole_output_path`` (default: "output/systole"): Output for interpolated systole.  
///
/// Returns
/// -------
///
/// A 4‑tuple of ``PyGeometryPair`` corresponding to  
/// (rest, stress, diastole, systole) geometries.
///
/// Raises
/// ------
///
/// ``RuntimeError`` if any Rust‑side processing fails.
///
/// Example
/// -------
///
/// .. code-block:: python
///
///    import multimodars as mm
///    # Assume you have four PyGeometry objects from earlier:
///    full = mm.from_array_full(
///        rest_dia, rest_sys, stress_dia, stress_sys,
///        steps_best_rotation=200,
///        interpolation_steps=32,
///        rest_output_path="out/rest",
///        stress_output_path="out/stress"
///    )
///    rest_pair, stress_pair, dia_pair, sys_pair = full
///
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

/// Processes two geometries in parallel.
///
/// Pipeline:
///
/// .. code-block:: text
///
///    Rest:                    Stress:
///    diastole                  diastole
///       |                         |
///       v                         v
///    systole                   systole
///
/// Arguments
/// ---------
///
/// - ``rest_geometry_dia``: Input ``PyGeometry`` at diastole for REST.  
/// - ``rest_geometry_sys``: Input ``PyGeometry`` at systole for REST.  
/// - ``stress_geometry_dia``: Input ``PyGeometry`` at diastole for STRESS.  
/// - ``stress_geometry_sys``: Input ``PyGeometry`` at systole for STRESS.  
/// - ``steps_best_rotation`` (default: 300): Number of rotation steps for alignment.  
/// - ``range_rotation_rad`` (default: 1.57): Angular range in radians for alignment.  
/// - ``interpolation_steps`` (default: 28): Number of interpolation steps between phases.  
/// - ``rest_output_path`` (default: "output/rest"): Output directory for REST results.  
/// - ``stress_output_path`` (default: "output/stress"): Output directory for STRESS results.  
///
/// Returns
/// -------
///
/// A tuple ``(rest_pair, stress_pair)`` of type ``(PyGeometryPair, PyGeometryPair)``,
/// containing the interpolated diastole/systole geometries for REST and STRESS.
///
/// Raises
/// ------
///
/// ``RuntimeError`` if any Rust‑side processing fails.
///
/// Example
/// -------
///
/// .. code-block:: python
///
///    import multimodars as mm
///    rest_pair, stress_pair = mm.from_array_doublepair(
///        rest_dia, rest_sys,
///        stress_dia, stress_sys,
///        steps_best_rotation=250,
///        interpolation_steps=32,
///        rest_output_path="out/rest",
///        stress_output_path="out/stress"
///    )
///
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
) -> PyResult<(PyGeometryPair, PyGeometryPair)> {
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

/// Interpolate between two existing ``PyGeometry`` objects (diastole and systole)
/// and return a ``PyGeometryPair`` containing both phases.
///
/// .. code-block:: text
///
///    Geometry pipeline:
///      diastole ──▶ systole
///
/// Arguments
/// ---------
///
/// - ``geometry_dia``: Input ``PyGeometry`` at diastole.  
/// - ``geometry_sys``: Input ``PyGeometry`` at systole.  
/// - ``output_path``: Directory path to write interpolated output files.  
/// - ``steps_best_rotation`` (default: 300): Number of rotation steps used for alignment.  
/// - ``range_rotation_rad`` (default: 1.57): Angular range in radians for the alignment search.  
/// - ``interpolation_steps`` (default: 28): Number of steps to interpolate between diastole and systole.  
///
/// Returns
/// -------
///
/// A ``PyGeometryPair`` tuple containing the diastole and systole geometries with interpolation applied.
///
/// Raises
/// ------
///
/// ``RuntimeError`` if the underlying Rust function fails.
///
/// Example
/// -------
///
/// .. code-block:: python
///
///    import multimodars as mm
///    pair = mm.from_array_singlepair(
///        rest_dia, rest_sys,
///        output_path="out/single",
///        steps_best_rotation=250,
///        interpolation_steps=30
///    )
///
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
