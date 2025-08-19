pub mod align;
pub mod classes;
pub mod entry_arr;
pub mod entry_file;

use crate::io::{
    input::{Contour, ContourPoint, Record},
    output::write_obj_mesh_without_uv,
};
use crate::processing::align_within::AlignLog;
use classes::{PyContour, PyGeometry, PyGeometryPair, PyRecord};
use entry_arr::*;
use entry_file::{
    from_file_doublepair_rs, from_file_full_rs, from_file_single_rs, from_file_singlepair_rs,
};
use pyo3::prelude::*;

fn logs_to_tuples(logs: Vec<AlignLog>) -> Vec<(u32, u32, f64, f64, f64, f64, f64, f64)> {
    logs.into_iter()
        .map(|l| {
            (
                l.contour_id,
                l.matched_to,
                l.rel_rot_deg,
                l.total_rot_deg,
                l.tx,
                l.ty,
                l.centroid.0,
                l.centroid.1,
            )
        })
        .collect()
}

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
/// - ``step_rotation_deg`` (default 0.5°) – Rotation step in degree
/// - ``range_rotation_deg`` (default 90°) – Rotation (+/-) range in degree, for 90° total range 180°
/// - ``image_center`` (default (4.5mm, 4.5mm)) in mm
/// - ``radius`` (default 0.5mm) in mm for catheter
/// - ``n_points`` (default 20) number of points for catheter, more points stronger influence of image center
/// - ``write_obj`` (default true)
/// - ``rest_output_path`` (default "output/rest")
/// - ``stress_output_path`` (default "output/stress")
/// - ``diastole_output_path`` (default "output/diastole")
/// - ``systole_output_path`` (default "output/systole")
/// - ``interpolation_steps`` (default 28)
/// - ``bruteforce`` (default false)
/// - ``sample_size`` (default 200) number of points to downsample to
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
/// A ``PyGeometryPair`` for rest, stress, diastole, systole.
/// A 4-tuple of ``Vec<id, matched_to, rel_rot_deg, total_rot_deg, tx, ty, centroid_x, centroid_y>``
/// for (diastole logs, systole logs, diastole stress logs, systole stress logs).
///
/// Example:
///
/// .. code-block:: python
///
///    import multimodars as mm
///    rest, stress, dia, sys, _ = mm.from_file_full(
///        "data/ivus_rest", "data/ivus_stress"
///    )
#[pyfunction]
#[pyo3(
    signature = (
        rest_input_path,
        stress_input_path,
        // these four get defaults if not passed
        step_rotation_deg = 0.5f64,
        range_rotation_deg = 90.0f64,
        image_center = (4.5f64, 4.5f64),
        radius = 0.5f64,
        n_points = 20u32,
        write_obj = true,
        rest_output_path = "output/rest",
        stress_output_path = "output/stress",
        diastole_output_path = "output/diastole",
        systole_output_path = "output/systole",
        interpolation_steps = 28usize,
        bruteforce = false,
        sample_size = 500,
    )
)]
pub fn from_file_full(
    rest_input_path: &str,
    stress_input_path: &str,
    step_rotation_deg: f64,
    range_rotation_deg: f64,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
    write_obj: bool,
    rest_output_path: &str,
    stress_output_path: &str,
    diastole_output_path: &str,
    systole_output_path: &str,
    interpolation_steps: usize,
    bruteforce: bool,
    sample_size: usize,
) -> PyResult<(
    PyGeometryPair,
    PyGeometryPair,
    PyGeometryPair,
    PyGeometryPair,
    (
        Vec<(u32, u32, f64, f64, f64, f64, f64, f64)>,
        Vec<(u32, u32, f64, f64, f64, f64, f64, f64)>,
        Vec<(u32, u32, f64, f64, f64, f64, f64, f64)>,
        Vec<(u32, u32, f64, f64, f64, f64, f64, f64)>,
    ),
)> {
    let (
        (rest_pair, stress_pair, dia_pair, sys_pair),
        (dia_logs, sys_logs, dia_logs_stress, sys_logs_stress),
    ) = from_file_full_rs(
        rest_input_path,
        stress_input_path,
        step_rotation_deg,
        range_rotation_deg,
        image_center,
        radius,
        n_points,
        write_obj,
        rest_output_path,
        stress_output_path,
        diastole_output_path,
        systole_output_path,
        interpolation_steps,
        bruteforce,
        sample_size,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let py_rest = rest_pair.into();
    let py_stress = stress_pair.into();
    let py_dia = dia_pair.into();
    let py_sys = sys_pair.into();
    let py_dia_log = logs_to_tuples(dia_logs);
    let py_sys_log = logs_to_tuples(sys_logs);
    let py_dia_log_stress = logs_to_tuples(dia_logs_stress);
    let py_sys_log_stress = logs_to_tuples(sys_logs_stress);
    Ok((
        py_rest,
        py_stress,
        py_dia,
        py_sys,
        (py_dia_log, py_sys_log, py_dia_log_stress, py_sys_log_stress),
    ))
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
/// - ``step_rotation_deg`` (default 0.5°) – Rotation step in degree
/// - ``range_rotation_deg`` (default 90°) – Rotation (+/-) range in degree, for 90° total range 180°
/// - ``image_center`` (default (4.5mm, 4.5mm)) in mm
/// - ``radius`` (default 0.5mm) in mm for catheter
/// - ``n_points`` (default 20) number of points for catheter, more points stronger influence of image center
/// - ``write_obj`` (default true)
/// - ``rest_output_path`` (default "output/rest")
/// - ``stress_output_path`` (default "output/stress")
/// - ``interpolation_steps`` (default 28)
/// - ``bruteforce`` (default false)
/// - ``sample_size`` (default 200) number of points to downsample to
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
/// A ``PyGeometryPair`` for rest, stress.
/// A 4-tuple of ``Vec<id, matched_to, rel_rot_deg, total_rot_deg, tx, ty, centroid_x, centroid_y>``
/// for (diastole logs, systole logs, diastole stress logs, systole stress logs).
///
/// Example:
///
/// .. code-block:: python
///
///    import multimodars as mm
///    rest, stress, _ = mm.from_file_doublepair(
///        "data/ivus_rest", "data/ivus_stress"
///    )
#[pyfunction]
#[pyo3(signature = (
    rest_input_path,
    stress_input_path,
    // defaults for the rest:
    step_rotation_deg = 0.5f64,
    range_rotation_deg = 90.0f64,
    image_center = (4.5f64, 4.5f64),
    radius = 0.5f64,
    n_points = 20u32,
    write_obj = true,
    rest_output_path = "output/rest",
    stress_output_path = "output/stress",
    interpolation_steps = 28usize,
    bruteforce = false,
    sample_size = 500,
))]
pub fn from_file_doublepair(
    rest_input_path: &str,
    stress_input_path: &str,
    step_rotation_deg: f64,
    range_rotation_deg: f64,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
    write_obj: bool,
    rest_output_path: &str,
    stress_output_path: &str,
    interpolation_steps: usize,
    bruteforce: bool,
    sample_size: usize,
) -> PyResult<(
    PyGeometryPair,
    PyGeometryPair,
    (
        Vec<(u32, u32, f64, f64, f64, f64, f64, f64)>,
        Vec<(u32, u32, f64, f64, f64, f64, f64, f64)>,
        Vec<(u32, u32, f64, f64, f64, f64, f64, f64)>,
        Vec<(u32, u32, f64, f64, f64, f64, f64, f64)>,
    ),
)> {
    let ((rest_pair, stress_pair), (dia_logs, sys_logs, dia_logs_stress, sys_logs_stress)) =
        from_file_doublepair_rs(
            rest_input_path,
            stress_input_path,
            step_rotation_deg,
            range_rotation_deg,
            image_center,
            radius,
            n_points,
            write_obj,
            rest_output_path,
            stress_output_path,
            interpolation_steps,
            bruteforce,
            sample_size,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let py_rest = rest_pair.into();
    let py_stress = stress_pair.into();
    let py_dia_log = logs_to_tuples(dia_logs);
    let py_sys_log = logs_to_tuples(sys_logs);
    let py_dia_log_stress = logs_to_tuples(dia_logs_stress);
    let py_sys_log_stress = logs_to_tuples(sys_logs_stress);

    Ok((
        py_rest,
        py_stress,
        (py_dia_log, py_sys_log, py_dia_log_stress, py_sys_log_stress),
    ))
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
/// - ``step_rotation_deg`` (default 0.5°) – Rotation step in degree
/// - ``range_rotation_deg`` (default 90°) – Rotation (+/-) range in degree, for 90° total range 180°
/// - ``image_center`` (default (4.5mm, 4.5mm)): Center coordinates (x, y).  
/// - ``radius`` (default 0.5mm) Processing radius.  
/// - ``n_points`` (default 20) Number of boundary points.  
/// - ``write_obj`` (default true)
/// - ``output_path``: Path to write the processed geometry. 
/// - ``interpolation_steps`` (default 28) Number of interpolation steps.  
/// - ``bruteforce`` (default false)
/// - ``sample_size`` (default 200) number of points to downsample to
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
/// A 2-tuple of ``Vec<id, matched_to, rel_rot_deg, total_rot_deg, tx, ty, centroid_x, centroid_y>``
/// for (diastole logs, systole logs).
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
///    pair, _ = mm.from_file_singlepair(
///        "data/ivus_rest.csv",
///        "output/rest"
///    )
///
/// This is a thin Python wrapper around the Rust implementation.
#[pyfunction]
#[pyo3(signature = (
    input_path,
    // defaults for the rest:
    step_rotation_deg = 0.5f64,
    range_rotation_deg = 90.0f64,
    image_center = (4.5f64, 4.5f64),
    radius = 0.5f64,
    n_points = 20u32,
    write_obj = true,
    output_path = "output/singlepair",
    interpolation_steps = 28usize,
    bruteforce = false,
    sample_size = 500,
))]
pub fn from_file_singlepair(
    input_path: &str,
    step_rotation_deg: f64,
    range_rotation_deg: f64,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
    write_obj: bool,
    output_path: &str,
    interpolation_steps: usize,
    bruteforce: bool,
    sample_size: usize,
) -> PyResult<(
    PyGeometryPair,
    (
        Vec<(u32, u32, f64, f64, f64, f64, f64, f64)>,
        Vec<(u32, u32, f64, f64, f64, f64, f64, f64)>,
    ),
)> {
    let (geom_pair, (dia_logs, sys_logs)) = from_file_singlepair_rs(
        input_path,
        step_rotation_deg,
        range_rotation_deg,
        image_center,
        radius,
        n_points,
        write_obj,
        output_path,
        interpolation_steps,
        bruteforce,
        sample_size,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let py_pair = geom_pair.into();
    let py_dia_log = logs_to_tuples(dia_logs);
    let py_sys_log = logs_to_tuples(sys_logs);
    Ok((py_pair, (py_dia_log, py_sys_log)))
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
/// - ``step_rotation_deg`` (default 0.5°) – Rotation step in degree
/// - ``range_rotation_deg`` (default 90°) – Rotation (+/-) range in degree, for 90° total range 180°
/// - ``diastole`` (default true): If true, process the diastole phase; otherwise systole.  
/// - ``image_center`` (default (4.5mm, 4.5mm)): (x, y) center for processing.  
/// - ``radius`` (default: 0.5mm): Radius around center to consider for catheter.  
/// - ``n_points`` (default: 20): Number of boundary points to generate.
/// - ``write_obj`` (default true)
/// - ``output_path`` (default: "output/single"): Where to write the processed geometry.  
/// - ``bruteforce`` (default false)
/// - ``sample_size`` (default 200) number of points to downsample to
///
/// Returns
/// -------
///
/// A ``PyGeometry`` containing the processed contour for the chosen phase.
/// A ``Vec<id, matched_to, rel_rot_deg, total_rot_deg, tx, ty, centroid_x, centroid_y>``.
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
///    geom, _ = mm.from_file_single(
///        "data/ivus.csv",
///        steps_best_rotation=0.5,
///        range_rotation_rad=90,
///        output_path="out/single",
///        diastole=False
///    )
///
#[pyfunction]
#[pyo3(signature = (
    input_path,
    // defaults for the rest:
    step_rotation_deg = 0.5f64,
    range_rotation_deg = 90.0f64,
    diastole = true,
    image_center = (4.5f64, 4.5f64),
    radius = 0.5f64,
    n_points = 20u32,
    write_obj = true,
    output_path = "output/single",
    bruteforce = false,
    sample_size = 500,
))]
pub fn from_file_single(
    input_path: &str,
    step_rotation_deg: f64,
    range_rotation_deg: f64,
    diastole: bool,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
    write_obj: bool,
    output_path: &str,
    bruteforce: bool,
    sample_size: usize,
) -> PyResult<(PyGeometry, Vec<(u32, u32, f64, f64, f64, f64, f64, f64)>)> {
    let (geom, logs) = from_file_single_rs(
        input_path,
        step_rotation_deg,
        range_rotation_deg,
        diastole,
        image_center,
        radius,
        n_points,
        write_obj,
        output_path,
        bruteforce,
        sample_size,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let py_geom = geom.into();
    let py_logs = logs_to_tuples(logs);

    Ok((py_geom, py_logs))
}

/// Generate catheter contours and return a new PyGeometry with them filled in.
///
/// This function takes an existing PyGeometry, extracts all its contour points,
/// computes catheter contours around those points, and returns a new PyGeometry
/// with the catheter field populated.
///
/// Arguments
/// ---------
/// - ``geometry``: The original geometry with contours but no catheters.
/// - ``image_center``: Center of the image (default = (4.5mm, 4.5mm)).
/// - ``radius``: Radius of the generated catheter contours (default = 0.5mm).
/// - ``n_points``: Number of points per catheter contour (default = 20mm).
///
/// Returns
/// -------
/// A new ``PyGeometry`` with the same data but `catheter` field filled.
///
#[pyfunction]
#[pyo3(signature = (geometry, image_center = (4.5, 4.5), radius = 0.5, n_points = 20))]
pub fn create_catheter_geometry(
    geometry: PyGeometry,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
) -> PyResult<PyGeometry> {
    // 1. Extract all contour points
    let all_points: Vec<ContourPoint> = geometry
        .contours
        .iter()
        .flat_map(|contour| contour.points.iter().map(ContourPoint::from))
        .collect();

    // 2. Generate catheter contours
    let rust_catheters: Vec<Contour> =
        Contour::create_catheter_contours(&all_points, image_center, radius, n_points)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    // 3. Convert to Python-compatible PyContour
    let py_catheters: Vec<PyContour> = rust_catheters.into_iter().map(PyContour::from).collect();

    // 4. Return a new PyGeometry with the catheter field filled
    Ok(PyGeometry {
        contours: geometry.contours,
        catheters: py_catheters,
        walls: geometry.walls,
        reference_point: geometry.reference_point,
    })
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
/// - ``step_rotation_deg`` (default 0.5°) – Rotation step in degree
/// - ``range_rotation_deg`` (default 90°) – Rotation (+/-) range in degree, for 90° total range 180°  
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
/// - ``bruteforce`` (default false)
/// - ``sample_size`` (default 200) number of points to downsample to
///
/// Returns
/// -------
///
/// A new ``PyGeometry`` instance containing reordered, aligned, and smoothed contours.
/// A ``Vec<id, matched_to, rel_rot_deg, total_rot_deg, tx, ty, centroid_x, centroid_y>``.
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
///    refined, _ = mm.geometry_from_array(
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
    step_rotation_deg = 0.5f64,
    range_rotation_deg = 90.0f64,
    image_center = (4.5f64, 4.5f64),
    radius = 0.5f64,
    n_points = 20u32,
    label = "None",
    records = None,
    delta = 0.0f64,
    max_rounds = 5,
    diastole = true,
    sort = true,
    write_obj=false,
    output_path="output/single",
    bruteforce = false,
    sample_size = 500,
))]
pub fn geometry_from_array(
    geometry: PyGeometry,
    step_rotation_deg: f64,
    range_rotation_deg: f64,
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
    bruteforce: bool,
    sample_size: usize,
) -> PyResult<(PyGeometry, Vec<(u32, u32, f64, f64, f64, f64, f64, f64)>)> {
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

    let (geom_rs, logs) = geometry_from_array_rs(
        contours_rs,
        walls_rs,
        reference_point_rs,
        step_rotation_deg,
        range_rotation_deg,
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
        bruteforce,
        sample_size,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let py_geom = PyGeometry::from(geom_rs);
    let py_logs = logs_to_tuples(logs);
    Ok((py_geom, py_logs))
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
/// - ``step_rotation_deg`` (default 0.5°) – Rotation step in degree
/// - ``range_rotation_deg`` (default 90°) – Rotation (+/-) range in degree, for 90° total range 180° 
/// - ``write_obj`` (default true)
/// - ``rest_output_path`` (default: "output/rest"): Output directory for REST results.  
/// - ``stress_output_path`` (default: "output/stress"): Output directory for STRESS results.  
/// - ``diastole_output_path`` (default: "output/diastole"): Output for interpolated diastole.  
/// - ``systole_output_path`` (default: "output/systole"): Output for interpolated systole.  
/// - ``interpolation_steps`` (default: 28): Number of interpolation steps between phases.  
/// - ``bruteforce`` (default false)
/// - ``sample_size`` (default 200) number of points to downsample to
///
/// Returns
/// -------
///
/// A ``PyGeometryPair`` for rest, stress, diastole, systole.
/// A 4-tuple of ``Vec<id, matched_to, rel_rot_deg, total_rot_deg, tx, ty, centroid_x, centroid_y>``
/// for (diastole logs, systole logs, diastole stress logs, systole stress logs).
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
///    rest, stress, dia, sys, _ = mm.from_array_full(
///        rest_dia, rest_sys, stress_dia, stress_sys,
///        steps_best_rotation=0.1,
///        interpolation_steps=28,
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
        step_rotation_deg = 0.5f64,
        range_rotation_deg = 90.0f64,
        image_center = (4.5f64, 4.5f64),
        radius = 0.5f64,
        n_points = 20u32,
        write_obj = true,
        rest_output_path = "output/rest",
        stress_output_path = "output/stress",
        diastole_output_path = "output/diastole",
        systole_output_path = "output/systole",
        interpolation_steps = 28usize,
        bruteforce = false,
        sample_size= 200,
    )
)]
pub fn from_array_full(
    rest_geometry_dia: PyGeometry,
    rest_geometry_sys: PyGeometry,
    stress_geometry_dia: PyGeometry,
    stress_geometry_sys: PyGeometry,
    step_rotation_deg: f64,
    range_rotation_deg: f64,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
    write_obj: bool,
    rest_output_path: &str,
    stress_output_path: &str,
    diastole_output_path: &str,
    systole_output_path: &str,
    interpolation_steps: usize,
    bruteforce: bool,
    sample_size: usize,
) -> PyResult<(
    PyGeometryPair,
    PyGeometryPair,
    PyGeometryPair,
    PyGeometryPair,
    (
        Vec<(u32, u32, f64, f64, f64, f64, f64, f64)>,
        Vec<(u32, u32, f64, f64, f64, f64, f64, f64)>,
        Vec<(u32, u32, f64, f64, f64, f64, f64, f64)>,
        Vec<(u32, u32, f64, f64, f64, f64, f64, f64)>,
    ),
)> {
    let (
        (rest_pair, stress_pair, dia_pair, sys_pair),
        (dia_logs, sys_logs, dia_logs_stress, sys_logs_stress),
    ) = from_array_full_rs(
        rest_geometry_dia.to_rust_geometry(),
        rest_geometry_sys.to_rust_geometry(),
        stress_geometry_dia.to_rust_geometry(),
        stress_geometry_sys.to_rust_geometry(),
        step_rotation_deg,
        range_rotation_deg,
        image_center,
        radius,
        n_points,
        write_obj,
        rest_output_path,
        stress_output_path,
        diastole_output_path,
        systole_output_path,
        interpolation_steps,
        bruteforce,
        sample_size,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let py_rest = rest_pair.into();
    let py_stress = stress_pair.into();
    let py_dia = dia_pair.into();
    let py_sys = sys_pair.into();
    let py_dia_log = logs_to_tuples(dia_logs);
    let py_sys_log = logs_to_tuples(sys_logs);
    let py_dia_log_stress = logs_to_tuples(dia_logs_stress);
    let py_sys_log_stress = logs_to_tuples(sys_logs_stress);
    Ok((
        py_rest,
        py_stress,
        py_dia,
        py_sys,
        (py_dia_log, py_sys_log, py_dia_log_stress, py_sys_log_stress),
    ))
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
/// - ``step_rotation_deg`` (default 0.5°) – Rotation step in degree.
/// - ``range_rotation_deg`` (default 90°) – Rotation (+/-) range in degree, for 90° total range 180°. 
/// - ``write_obj`` (default true)
/// - ``rest_output_path`` (default: "output/rest"): Output directory for REST results.  
/// - ``stress_output_path`` (default: "output/stress"): Output directory for STRESS results.  
/// - ``interpolation_steps`` (default: 28): Number of interpolation steps between phases.  
/// - ``bruteforce`` (default false)
/// - ``sample_size`` (default 200) number of points to downsample to
///
/// Returns
/// -------
///
/// A tuple ``(rest_pair, stress_pair)`` of type ``(PyGeometryPair, PyGeometryPair)``,
/// containing the interpolated diastole/systole geometries for REST and STRESS.
/// A 4-tuple of ``Vec<id, matched_to, rel_rot_deg, total_rot_deg, tx, ty, centroid_x, centroid_y>``
/// for (diastole logs, systole logs, diastole stress logs, systole stress logs).
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
///    rest_pair, stress_pair, _ = mm.from_array_doublepair(
///        rest_dia, rest_sys,
///        stress_dia, stress_sys,
///        steps_best_rotation=0.2,
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
        step_rotation_deg = 0.5f64,
        range_rotation_deg = 90.0f64,
        image_center = (4.5f64, 4.5f64),
        radius = 0.5f64,
        n_points = 20u32,
        write_obj = true,
        rest_output_path = "output/rest",
        stress_output_path = "output/stress",
        interpolation_steps = 28usize,
        bruteforce = false,
        sample_size = 500,
    )
)]
pub fn from_array_doublepair(
    rest_geometry_dia: PyGeometry,
    rest_geometry_sys: PyGeometry,
    stress_geometry_dia: PyGeometry,
    stress_geometry_sys: PyGeometry,
    step_rotation_deg: f64,
    range_rotation_deg: f64,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
    write_obj: bool,
    rest_output_path: &str,
    stress_output_path: &str,
    interpolation_steps: usize,
    bruteforce: bool,
    sample_size: usize,
) -> PyResult<(
    PyGeometryPair,
    PyGeometryPair,
    (
        Vec<(u32, u32, f64, f64, f64, f64, f64, f64)>,
        Vec<(u32, u32, f64, f64, f64, f64, f64, f64)>,
        Vec<(u32, u32, f64, f64, f64, f64, f64, f64)>,
        Vec<(u32, u32, f64, f64, f64, f64, f64, f64)>,
    ),
)> {
    let ((rest_pair, stress_pair), (dia_logs, sys_logs, dia_logs_stress, sys_logs_stress)) =
        from_array_doublepair_rs(
            rest_geometry_dia.to_rust_geometry(),
            rest_geometry_sys.to_rust_geometry(),
            stress_geometry_dia.to_rust_geometry(),
            stress_geometry_sys.to_rust_geometry(),
            step_rotation_deg,
            range_rotation_deg,
            image_center,
            radius,
            n_points,
            write_obj,
            rest_output_path,
            stress_output_path,
            interpolation_steps,
            bruteforce,
            sample_size,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let py_rest = rest_pair.into();
    let py_stress = stress_pair.into();
    let py_dia_log = logs_to_tuples(dia_logs);
    let py_sys_log = logs_to_tuples(sys_logs);
    let py_dia_log_stress = logs_to_tuples(dia_logs_stress);
    let py_sys_log_stress = logs_to_tuples(sys_logs_stress);

    Ok((
        py_rest,
        py_stress,
        (py_dia_log, py_sys_log, py_dia_log_stress, py_sys_log_stress),
    ))
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
/// - ``step_rotation_deg`` (default 0.5°) – Rotation step in degree
/// - ``range_rotation_deg`` (default 90°) – Rotation (+/-) range in degree, for 90° total range 180°
/// - ``write_obj`` (default true)
/// - ``output_path``: Directory path to write interpolated output files.  
/// - ``interpolation_steps`` (default: 28): Number of steps to interpolate between diastole and systole.  
/// - ``bruteforce`` (default false)
/// - ``sample_size`` (default 200) number of points to downsample to
///
/// Returns
/// -------
///
/// A ``PyGeometryPair`` tuple containing the diastole and systole geometries with interpolation applied.
/// A 2-tuple of ``Vec<id, matched_to, rel_rot_deg, total_rot_deg, tx, ty, centroid_x, centroid_y>``
/// for (diastole logs, systole logs).
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
///    pair, _ = mm.from_array_singlepair(
///        rest_dia, rest_sys,
///        output_path="out/single",
///        steps_best_rotation=0.1,
///        interpolation_steps=30
///    )
///
#[pyfunction]
#[pyo3(
    signature = (
        geometry_dia,
        geometry_sys,
        step_rotation_deg = 0.5f64,
        range_rotation_deg = 90.0f64,
        image_center = (4.5f64, 4.5f64),
        radius = 0.5f64,
        n_points = 20u32,
        write_obj = true,
        output_path = "output/singlepair",
        interpolation_steps = 28usize,
        bruteforce = false,
        sample_size = 500,
    )
)]
pub fn from_array_singlepair(
    geometry_dia: PyGeometry,
    geometry_sys: PyGeometry,
    step_rotation_deg: f64,
    range_rotation_deg: f64,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
    write_obj: bool,
    output_path: &str,
    interpolation_steps: usize,
    bruteforce: bool,
    sample_size: usize,
) -> PyResult<(
    PyGeometryPair,
    (
        Vec<(u32, u32, f64, f64, f64, f64, f64, f64)>,
        Vec<(u32, u32, f64, f64, f64, f64, f64, f64)>,
    ),
)> {
    let (pair, (dia_logs, sys_logs)) = from_array_singlepair_rs(
        geometry_dia.to_rust_geometry(),
        geometry_sys.to_rust_geometry(),
        step_rotation_deg,
        range_rotation_deg,
        image_center,
        radius,
        n_points,
        write_obj,
        output_path,
        interpolation_steps,
        bruteforce,
        sample_size,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let py_pair = pair.into();
    let py_dia_log = logs_to_tuples(dia_logs);
    let py_sys_log = logs_to_tuples(sys_logs);
    Ok((py_pair, (py_dia_log, py_sys_log)))
}

/// Convert a ``PyGeometry`` object into one or more OBJ files and write them to disk.
///
/// This function takes a Python-exposed geometry (``PyGeometry``), converts it into the
/// corresponding Rust geometry, and writes the specified components (contours, walls,
/// catheter) as OBJ meshes without UV coordinates. Each component is written to its own
/// file, with a corresponding MTL material file.
///
/// # Arguments
///
/// * ``geometry`` - Input ``PyGeometry`` instance containing the mesh data.
/// * ``output_path`` - Directory path where the OBJ and MTL files will be written.
/// * ``contours`` - Whether to export the contour mesh (default: ``true``).
/// * ``walls`` - Whether to export the wall mesh (default: ``true``).
/// * ``catheter`` - Whether to export the catheter mesh (default: ``true``).
/// * ``filename_contours`` - Filename for the contour OBJ (default: "contours.obj").
/// * ``material_contours`` - Filename for the contour MTL (default: "contours.mtl").
/// * ``filename_catheter`` - Filename for the catheter OBJ (default: "catheter.obj").
/// * ``material_catheter`` - Filename for the catheter MTL (default: "catheter.mtl").
/// * ``filename_walls`` - Filename for the walls OBJ (default: "walls.obj").
/// * ``material_walls`` - Filename for the walls MTL (default: "walls.mtl").
///
/// # Errors
///
/// Returns a `PyRuntimeError` if any of the underlying file writes fail.
#[pyfunction(
    signature = (
        geometry,
        output_path,
        contours = true,
        walls = true,
        catheter = true,
        filename_contours = "contours.obj",
        material_contours = "contours.mtl",
        filename_catheter = "catheter.obj",
        material_catheter = "catheter.mtl",
        filename_walls = "walls.obj",
        material_walls = "walls.mtl",
    )
)]
pub fn to_obj(
    geometry: PyGeometry,
    output_path: &str,
    contours: bool,
    walls: bool,
    catheter: bool,
    filename_contours: &str,
    material_contours: &str,
    filename_catheter: &str,
    material_catheter: &str,
    filename_walls: &str,
    material_walls: &str,
) -> PyResult<()> {
    // Convert the Python geometry to Rust representation
    let geometry_rs = geometry.to_rust_geometry();

    // Ensure output directory exists
    std::fs::create_dir_all(output_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Could not create output directory '{}': {}",
            output_path, e
        ))
    })?;

    // Write each component if requested
    if contours {
        write_obj_mesh_without_uv(
            &geometry_rs.contours,
            &format!("{}/{}", output_path, filename_contours),
            material_contours,
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to write contours OBJ: {}",
                e
            ))
        })?;
    }
    if catheter {
        write_obj_mesh_without_uv(
            &geometry_rs.catheter,
            &format!("{}/{}", output_path, filename_catheter),
            material_catheter,
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to write catheter OBJ: {}",
                e
            ))
        })?;
    }
    if walls {
        write_obj_mesh_without_uv(
            &geometry_rs.walls,
            &format!("{}/{}", output_path, filename_walls),
            material_walls,
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to write walls OBJ: {}",
                e
            ))
        })?;
    }
    Ok(())
}
