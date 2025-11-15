pub mod align;
pub mod classes;
pub mod entry;

use crate::intravascular::io::{input::InputData, output::write_obj_mesh_without_uv};
use crate::intravascular::processing::align_within::AlignLog;
use classes::{PyContourType, PyGeometry, PyGeometryPair, PyInputData};
use entry::*;
use pyo3::prelude::*;
use std::path::Path;

fn logs_to_tuples(logs: Vec<AlignLog>) -> Vec<(u32, u32, f64, f64, f64, f64, f64)> {
    logs.into_iter()
        .map(|l| {
            (
                l.contour_id,
                l.matched_to,
                l.rot_deg,
                l.tx,
                l.ty,
                l.centroid.0,
                l.centroid.1,
            )
        })
        .collect()
}

/// Processes four geometries in parallel from folders.
///
/// .. code-block:: text
///
///    Rest           Stress
///    diastole ──▶ diastole
///       │            │
///       ▼            ▼
///    systole ──▶ systole
///
///
/// Args:
///     input_path_a: Path to e.g. REST input file
///     input_path_b: Path to e.g. STRESS input file
///     step_rotation_deg (default 0.5°): Rotation step in degree
///     range_rotation_deg (default 90°): Rotation (+/-) range in degree, for 90° total range 180°
///     image_center (default (4.5mm, 4.5mm)): in mm
///     radius (default 0.5mm): in mm for catheter
///     n_points (default 20): number of points for catheter, more points stronger influence of image center
///     write_obj (default true): Wether to write OBJ files
///     watertight (default true): Wether to write shell or watertight mesh to OBJ.
///     output_path_a (default "output/rest"):
///     output_path_b (default "output/stress"):
///     output_path_c (default "output/diastole"):
///     output_path_d (default "output/systole"):
///     interpolation_steps (default 28): Number of interpolated meshes
///     bruteforce (default false): Wether to use bruteforce alignment (comparison for every step size)
///     sample_size (default 200): number of points to downsample to
///     contour_type (default [PyContourType.Lumen, PyContourType.Catheter, PyContourType.Wall])
///     smooth (default true): bool smooth after alignment with 3-point moving average
///     postprocessing (default true): adjusts spacing within/between geometry/geometries to have equal offsets
///
/// .. warning::
///
///    The CSV must have **no header**. Each row is (frame index, x-coord (mm), y-coord (mm), z-coord (mm)):
///
/// .. code-block:: text
///
///    185, 5.32, 2.37, 0.0
///    ...
///
/// Returns:
///     ``PyGeometryPair`` for rest, stress, diastole, systole.
///     A 4-tuple of ``Vec<id, matched_to, rel_rot_deg, total_rot_deg, tx, ty, centroid_x, centroid_y>``
///     for (diastole logs, systole logs, diastole stress logs, systole stress logs).
///
/// Example:
///     >>> import multimodars as mm
///     >>> rest, stress, dia, sys, _ = mm.from_file_full(
///     ...     "data/ivus_rest", "data/ivus_stress"
///     ... )
#[pyfunction]
#[pyo3(
    signature = (
        input_path_a,
        input_path_b,
        label = "full",
        diastole = true,
        step_rotation_deg = 0.5f64,
        range_rotation_deg = 90.0f64,
        image_center = (4.5f64, 4.5f64),
        radius = 0.5f64,
        n_points = 20u32,
        write_obj = true,
        watertight = true,
        output_path_a = "output/rest",
        output_path_b = "output/stress",
        output_path_c = "output/diastole",
        output_path_d = "output/systole",
        interpolation_steps = 28usize,
        bruteforce = false,
        sample_size = 500,
        contour_types = vec![PyContourType::Lumen, PyContourType::Catheter, PyContourType::Wall],
        smooth = true,
        postprocessing = true,
    )
)]
pub fn from_file_full(
    input_path_a: &str,
    input_path_b: &str,
    label: &str,
    diastole: bool,
    step_rotation_deg: f64,
    range_rotation_deg: f64,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
    write_obj: bool,
    watertight: bool,
    output_path_a: &str,
    output_path_b: &str,
    output_path_c: &str,
    output_path_d: &str,
    interpolation_steps: usize,
    bruteforce: bool,
    sample_size: usize,
    contour_types: Vec<PyContourType>,
    smooth: bool,
    postprocessing: bool,
) -> PyResult<(
    PyGeometryPair,
    PyGeometryPair,
    PyGeometryPair,
    PyGeometryPair,
    (
        Vec<(u32, u32, f64, f64, f64, f64, f64)>,
        Vec<(u32, u32, f64, f64, f64, f64, f64)>,
        Vec<(u32, u32, f64, f64, f64, f64, f64)>,
        Vec<(u32, u32, f64, f64, f64, f64, f64)>,
    ),
)> {
    let rust_contour_types: Vec<crate::intravascular::io::geometry::ContourType> =
        contour_types.iter().map(|ct| ct.into()).collect();

    let (
        geom_ab_final,
        geom_cd_final,
        geom_ac_final,
        geom_bd_final,
        logs_a,
        logs_b,
        logs_c,
        logs_d,
    ) = full_processing_rs(
        label.to_string(),
        image_center,
        radius,
        n_points,
        Some(input_path_a),
        Some(input_path_b),
        None,
        None,
        None,
        None,
        diastole,
        write_obj,
        interpolation_steps,
        rust_contour_types, // Use converted types
        watertight,
        output_path_a,
        output_path_b,
        output_path_c,
        output_path_d,
        step_rotation_deg,
        range_rotation_deg,
        smooth,
        bruteforce,
        sample_size,
        postprocessing,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let py_geom_ab = geom_ab_final.into();
    let py_geom_cd = geom_cd_final.into();
    let py_geom_ac = geom_ac_final.into();
    let py_geom_bd = geom_bd_final.into();
    let py_logs_a = logs_to_tuples(logs_a);
    let py_logs_b = logs_to_tuples(logs_b);
    let py_logs_c = logs_to_tuples(logs_c);
    let py_logs_d = logs_to_tuples(logs_d);
    Ok((
        py_geom_ab,
        py_geom_cd,
        py_geom_ac,
        py_geom_bd,
        (py_logs_a, py_logs_b, py_logs_c, py_logs_d),
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
///        │                         │
///        ▼                         ▼
///    systole                   systole
///
/// Args:
///     input_path_a: Path to e.g. REST input file
///     input_path_b: Path to e.g. STRESS input file
///     step_rotation_deg (default 0.5°): Rotation step in degree
///     range_rotation_deg (default 90°): Rotation (+/-) range in degree, for 90° total range 180°
///     image_center (default (4.5mm, 4.5mm)): in mm
///     radius (default 0.5mm): in mm for catheter
///     n_points (default 20): number of points for catheter, more points stronger influence of image center
///     write_obj (default true): Wether to write OBJ files
///     watertight (default true): Wether to write shell or watertight mesh to OBJ.
///     output_path_a (default "output/rest"):
///     output_path_b (default "output/stress"):
///     interpolation_steps (default 28): Number of interpolated meshes
///     bruteforce (default false): Wether to use bruteforce alignment (comparison for every step size)
///     sample_size (default 200): number of points to downsample to
///     contour_type (default [PyContourType.Lumen, PyContourType.Catheter, PyContourType.Wall])
///     smooth (default true): bool smooth after alignment with 3-point moving average
///     postprocessing (default true): adjusts spacing within/between geometry/geometries to have equal offsets
///
/// .. warning::
///
///    The CSV must have **no header**. Each row is (frame index, x-coord (mm), y-coord (mm), z-coord (mm)):
///
/// .. code-block:: text
///
///    185, 5.32, 2.37, 0.0
///    ...
///
/// Returns:
///     A ``PyGeometryPair`` for rest, stress.
///     A 4-tuple of ``Vec<id, matched_to, rel_rot_deg, total_rot_deg, tx, ty, centroid_x, centroid_y>``
///     for (diastole logs, systole logs, diastole stress logs, systole stress logs).
///
/// Example:
///     >>> import multimodars as mm
///     >>> rest, stress, _ = mm.from_file_doublepair(
///     ...     "data/ivus_rest", "data/ivus_stress"
///     ... )
#[pyfunction]
#[pyo3(signature = (
    input_path_a,
    input_path_b,
    label = "double_pair",
    diastole = true,
    step_rotation_deg = 0.5f64,
    range_rotation_deg = 90.0f64,
    image_center = (4.5f64, 4.5f64),
    radius = 0.5f64,
    n_points = 20u32,
    write_obj = true,
    watertight = true,
    output_path_a = "output/rest",
    output_path_b = "output/stress",
    interpolation_steps = 28usize,
    bruteforce = false,
    sample_size = 500,
    contour_types = vec![PyContourType::Lumen, PyContourType::Catheter, PyContourType::Wall],
    smooth = true,
    postprocessing = true,
))]
pub fn from_file_doublepair(
    input_path_a: &str,
    input_path_b: &str,
    label: &str,
    diastole: bool,
    step_rotation_deg: f64,
    range_rotation_deg: f64,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
    write_obj: bool,
    watertight: bool,
    output_path_a: &str,
    output_path_b: &str,
    interpolation_steps: usize,
    bruteforce: bool,
    sample_size: usize,
    contour_types: Vec<PyContourType>,
    smooth: bool,
    postprocessing: bool,
) -> PyResult<(
    PyGeometryPair,
    PyGeometryPair,
    (
        Vec<(u32, u32, f64, f64, f64, f64, f64)>,
        Vec<(u32, u32, f64, f64, f64, f64, f64)>,
        Vec<(u32, u32, f64, f64, f64, f64, f64)>,
        Vec<(u32, u32, f64, f64, f64, f64, f64)>,
    ),
)> {
    let rust_contour_types: Vec<crate::intravascular::io::geometry::ContourType> =
        contour_types.iter().map(|ct| ct.into()).collect();

    let (geom_ab_final, geom_cd_final, logs_a, logs_b, logs_c, logs_d) = double_pair_processing_rs(
        label.to_string(),
        image_center,
        radius,
        n_points,
        Some(input_path_a),
        Some(input_path_b),
        None,
        None,
        None,
        None,
        diastole,
        write_obj,
        interpolation_steps,
        rust_contour_types,
        watertight,
        output_path_a,
        output_path_b,
        step_rotation_deg,
        range_rotation_deg,
        smooth,
        bruteforce,
        sample_size,
        postprocessing,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let py_geom_ab = geom_ab_final.into();
    let py_geom_cd = geom_cd_final.into();
    let py_logs_a = logs_to_tuples(logs_a);
    let py_logs_b = logs_to_tuples(logs_b);
    let py_logs_c = logs_to_tuples(logs_c);
    let py_logs_d = logs_to_tuples(logs_d);
    Ok((
        py_geom_ab,
        py_geom_cd,
        (py_logs_a, py_logs_b, py_logs_c, py_logs_d),
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
/// Args:
///     input_path: Path to the input CSV file.  
///     step_rotation_deg (default 0.5°) – Rotation step in degree
///     range_rotation_deg (default 90°) – Rotation (+/-) range in degree, for 90° total range 180°
///     image_center (default (4.5mm, 4.5mm)): Center coordinates (x, y).  
///     radius (default 0.5mm): in mm for catheter
///     n_points (default 20): number of points for catheter, more points stronger influence of image center
///     write_obj (default true): Wether to write OBJ files
///     watertight (default true): Wether to write shell or watertight mesh to OBJ.
///     output_path: Path to write the processed geometry.
///     interpolation_steps (default 28): Number of interpolated meshes
///     bruteforce (default false): Wether to use bruteforce alignment (comparison for every step size)
///     sample_size (default 200): number of points to downsample to
///     contour_type (default [PyContourType.Lumen, PyContourType.Catheter, PyContourType.Wall])
///     smooth (default true): bool smooth after alignment with 3-point moving average
///     postprocessing (default true): adjusts spacing within/between geometry/geometries to have equal offsets
///
/// .. warning::
///
///    The CSV must have **no header**. Each row is:
///
/// .. code-block:: text
///
///    185, 5.32, 2.37, 0.0
///    ...
///
/// Returns:
///     A single ``PyGeometryPair`` for (rest or stress) geometry.
///     A 2-tuple of ``Vec<id, matched_to, rel_rot_deg, total_rot_deg, tx, ty, centroid_x, centroid_y>``
///     for (diastole logs, systole logs).
///
/// Raises:
///     **RuntimeError** if the Rust pipeline fails.
///
/// Example:
///     >>> import multimodars as mm
///     >>> pair, _ = mm.from_file_singlepair(
///     ...     "data/ivus_rest.csv",
///     ...     "output/rest"
///     ... )
#[pyfunction]
#[pyo3(signature = (
    input_path,
    label = "single_pair",
    diastole = true,
    step_rotation_deg = 0.5f64,
    range_rotation_deg = 90.0f64,
    image_center = (4.5f64, 4.5f64),
    radius = 0.5f64,
    n_points = 20u32,
    write_obj = true,
    watertight = true,
    output_path = "output/singlepair",
    interpolation_steps = 28usize,
    bruteforce = false,
    sample_size = 500,
    contour_types = vec![PyContourType::Lumen, PyContourType::Catheter, PyContourType::Wall],
    smooth = true,
    postprocessing = true,
))]
pub fn from_file_singlepair(
    input_path: &str,
    label: &str,
    diastole: bool,
    step_rotation_deg: f64,
    range_rotation_deg: f64,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
    write_obj: bool,
    watertight: bool,
    output_path: &str,
    interpolation_steps: usize,
    bruteforce: bool,
    sample_size: usize,
    contour_types: Vec<PyContourType>,
    smooth: bool,
    postprocessing: bool,
) -> PyResult<(
    PyGeometryPair,
    (
        Vec<(u32, u32, f64, f64, f64, f64, f64)>,
        Vec<(u32, u32, f64, f64, f64, f64, f64)>,
    ),
)> {
    let rust_contour_types: Vec<crate::intravascular::io::geometry::ContourType> =
        contour_types.iter().map(|ct| ct.into()).collect();

    let (geom_pair_final, logs_a, logs_b) = pair_processing_rs(
        label.to_string(),
        image_center,
        radius,
        n_points,
        Some(input_path),
        None,
        None,
        diastole,
        write_obj,
        interpolation_steps,
        rust_contour_types,
        watertight,
        output_path,
        step_rotation_deg,
        range_rotation_deg,
        smooth,
        bruteforce,
        sample_size,
        postprocessing,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let py_geom_ab = geom_pair_final.into();
    let py_logs_a = logs_to_tuples(logs_a);
    let py_logs_b = logs_to_tuples(logs_b);
    Ok((py_geom_ab, (py_logs_a, py_logs_b)))
}

/// Processes a single geometry (either diastole or systole) from an IVUS CSV file.
///
/// .. code-block:: text
///
///    Rest/Stress pipeline (choose phase via `diastole` flag):
///      e.g. diastole
///
/// Args:
///     input_path: Path to the input CSV (no header; columns: frame, x, y, z).  
///     step_rotation_deg (default 0.5°): Rotation step in degree
///     range_rotation_deg (default 90°): Rotation (+/-) range in degree, for 90° total range 180°
///     diastole (default true): If true, process the diastole phase; otherwise systole.  
///     image_center (default (4.5mm, 4.5mm)): (x, y) center for processing.  
///     radius (default: 0.5mm): Radius around center to consider for catheter.  
///     n_points (default 20): number of points for catheter, more points stronger influence of image center.
///     write_obj (default true): Wether to write OBJ files.
///     watertight (default true): Wether to write shell or watertight mesh to OBJ.
///     output_path (default "output/single"): Where to write the processed geometry.  
///     bruteforce (default false): Wether to use bruteforce alignment (comparison for every step size).
///     sample_size (default 200): number of points to downsample to
///     contour_type (default [PyContourType.Lumen, PyContourType.Catheter, PyContourType.Wall])
///     smooth (default true): bool smooth after alignment with 3-point moving average
///     postprocessing (default true): adjusts spacing within/between geometry/geometries to have equal offsets
///
/// Returns:
///     A ``PyGeometry`` containing the processed contour for the chosen phase.
///     A ``Vec<id, matched_to, rel_rot_deg, total_rot_deg, tx, ty, centroid_x, centroid_y>``.
///
/// Raises:
///     **RuntimeError** if the underlying Rust pipeline fails.
///
/// Example:
///     >>> import multimodars as mm
///     >>> geom, _ = mm.from_file_single(
///     ...     "data/ivus.csv",
///     ...     steps_best_rotation=0.5,
///     ...     range_rotation_rad=90,
///     ...     output_path="out/single",
///     ...     diastole=False
///     ... )
///
#[pyfunction]
#[pyo3(signature = (
    input_path,
    label = "single",
    diastole = true,
    step_rotation_deg = 0.5f64,
    range_rotation_deg = 90.0f64,
    image_center = (4.5f64, 4.5f64),
    radius = 0.5f64,
    n_points = 20u32,
    write_obj = true,
    watertight = true,
    output_path = "output/single",
    bruteforce = false,
    sample_size = 200,
    contour_types = vec![PyContourType::Lumen, PyContourType::Catheter, PyContourType::Wall],
    smooth = true,
))]
pub fn from_file_single(
    input_path: &str,
    label: &str,
    diastole: bool,
    step_rotation_deg: f64,
    range_rotation_deg: f64,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
    write_obj: bool,
    watertight: bool,
    output_path: &str,
    bruteforce: bool,
    sample_size: usize,
    contour_types: Vec<PyContourType>,
    smooth: bool,
) -> PyResult<(PyGeometry, Vec<(u32, u32, f64, f64, f64, f64, f64)>)> {
    let rust_contour_types: Vec<crate::intravascular::io::geometry::ContourType> =
        contour_types.iter().map(|ct| ct.into()).collect();

    let (geom, logs) = single_processing_rs(
        label.to_string(),
        image_center,
        radius,
        n_points,
        Some(input_path),
        None,
        diastole,
        write_obj,
        watertight,
        rust_contour_types,
        output_path,
        step_rotation_deg,
        range_rotation_deg,
        smooth,
        bruteforce,
        sample_size,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let py_geom = geom.into();
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
/// Args:
///     input_data_a: Input ``PyInputData`` at e.g diastole for REST.  
///     input_data_b: Input ``PyInputData`` at e.g. systole for REST.  
///     input_data_c: Input ``PyInputData`` at e.g. diastole for STRESS.  
///     input_data_d: Input ``PyInputData`` at e.g. systole for STRESS.  
///     step_rotation_deg (default 0.5°): Rotation step in degree
///     range_rotation_deg (default 90°): Rotation (+/-) range in degree, for 90° total range 180°
///     write_obj (default True): Wether to write OBJ files.
///     watertight (default True): Wether to write shell or watertight mesh to OBJ.
///     output_path_a (default "output/rest"): Output directory for e.g. REST results.  
///     output_path_b (default "output/stress"): Output directory for e.g. STRESS results.  
///     output_path_c (default "output/diastole"): Output for e.g. DIASTOLE results.  
///     output_path_d (default "output/systole"): Output for e.g. SYSTOLE results.  
///     interpolation_steps (default 28): Number of interpolation steps between phases.  
///     bruteforce (default False): Wether to use bruteforce alignment (comparison for every step size).
///     sample_size (default 200): number of points to downsample to
///     contour_type (default [PyContourType.Lumen, PyContourType.Catheter, PyContourType.Wall])
///     smooth (default true): bool smooth after alignment with 3-point moving average
///     postprocessing (default true): adjusts spacing within/between geometry/geometries to have equal offsets
///
/// Returns:
///     A ``PyGeometryPair`` for rest, stress, diastole, systole.
///     A 4-tuple of ``Vec<id, matched_to, rel_rot_deg, total_rot_deg, tx, ty, centroid_x, centroid_y>``
///     for (diastole logs, systole logs, diastole stress logs, systole stress logs).
///
/// Raises:
///     **RuntimeError** if any Rust‑side processing fails.
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
        input_data_a,
        input_data_b,
        input_data_c,
        input_data_d,
        label = "full",
        diastole = true,
        step_rotation_deg = 0.5f64,
        range_rotation_deg = 90.0f64,
        image_center = (4.5f64, 4.5f64),
        radius = 0.5f64,
        n_points = 20u32,
        write_obj = true,
        watertight = true,
        output_path_a = "output/rest",
        output_path_b = "output/stress",
        output_path_c = "output/diastole",
        output_path_d = "output/systole",
        interpolation_steps = 28usize,
        bruteforce = false,
        sample_size= 200,
        contour_types = vec![PyContourType::Lumen, PyContourType::Catheter, PyContourType::Wall],
        smooth = true,
        postprocessing = true,
    )
)]
pub fn from_array_full(
    input_data_a: PyInputData,
    input_data_b: PyInputData,
    input_data_c: PyInputData,
    input_data_d: PyInputData,
    label: &str,
    diastole: bool,
    step_rotation_deg: f64,
    range_rotation_deg: f64,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
    write_obj: bool,
    watertight: bool,
    output_path_a: &str,
    output_path_b: &str,
    output_path_c: &str,
    output_path_d: &str,
    interpolation_steps: usize,
    bruteforce: bool,
    sample_size: usize,
    contour_types: Vec<PyContourType>,
    smooth: bool,
    postprocessing: bool,
) -> PyResult<(
    PyGeometryPair,
    PyGeometryPair,
    PyGeometryPair,
    PyGeometryPair,
    (
        Vec<(u32, u32, f64, f64, f64, f64, f64)>,
        Vec<(u32, u32, f64, f64, f64, f64, f64)>,
        Vec<(u32, u32, f64, f64, f64, f64, f64)>,
        Vec<(u32, u32, f64, f64, f64, f64, f64)>,
    ),
)> {
    let rust_contour_types: Vec<crate::intravascular::io::geometry::ContourType> =
        contour_types.iter().map(|ct| ct.into()).collect();

    let input_data_a_rust: InputData = input_data_a.try_into().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to convert input_data_a: {}", e))
    })?;
    let input_data_b_rust: InputData = input_data_b.try_into().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to convert input_data_b: {}", e))
    })?;
    let input_data_c_rust: InputData = input_data_c.try_into().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to convert input_data_c: {}", e))
    })?;
    let input_data_d_rust: InputData = input_data_d.try_into().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to convert input_data_d: {}", e))
    })?;

    let (
        geom_ab_final,
        geom_cd_final,
        geom_ac_final,
        geom_bd_final,
        logs_a,
        logs_b,
        logs_c,
        logs_d,
    ) = full_processing_rs(
        label.to_string(),
        image_center,
        radius,
        n_points,
        None, // No file paths for array version
        None,
        Some(input_data_a_rust),
        Some(input_data_b_rust),
        Some(input_data_c_rust),
        Some(input_data_d_rust),
        diastole,
        write_obj,
        interpolation_steps,
        rust_contour_types,
        watertight,
        output_path_a,
        output_path_b,
        output_path_c,
        output_path_d,
        step_rotation_deg,
        range_rotation_deg,
        smooth,
        bruteforce,
        sample_size,
        postprocessing,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let py_geom_ab = geom_ab_final.into();
    let py_geom_cd = geom_cd_final.into();
    let py_geom_ac = geom_ac_final.into();
    let py_geom_bd = geom_bd_final.into();
    let py_logs_a = logs_to_tuples(logs_a);
    let py_logs_b = logs_to_tuples(logs_b);
    let py_logs_c = logs_to_tuples(logs_c);
    let py_logs_d = logs_to_tuples(logs_d);

    Ok((
        py_geom_ab,
        py_geom_cd,
        py_geom_ac,
        py_geom_bd,
        (py_logs_a, py_logs_b, py_logs_c, py_logs_d),
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
///        │                         │
///        ▼                         ▼
///    systole                   systole
///
/// Args:
///     input_data_a: Input ``PyInputData`` at e.g diastole for REST.  
///     input_data_b: Input ``PyInputData`` at e.g. systole for REST.  
///     input_data_c: Input ``PyInputData`` at e.g. diastole for STRESS.  
///     input_data_d: Input ``PyInputData`` at e.g. systole for STRESS.  
///     step_rotation_deg (default 0.5°): Rotation step in degree.
///     range_rotation_deg (default 90°): Rotation (+/-) range in degree, for 90° total range 180°.
///     write_obj (default True): Wether to write OBJ files.
///     watertight (default True): Wether to write shell or watertight mesh to OBJ.
///     output_path_a (default "output/rest"): Output directory for e.g. REST results.  
///     output_path_b (default "output/stress"): Output directory for e.g. STRESS results.  
///     interpolation_steps (default 28): Number of interpolation steps between phases.  
///     bruteforce (default False): Wether to use bruteforce alignment (comparison for every step size).
///     sample_size (default 200) number of points to downsample to
///     contour_type (default [PyContourType.Lumen, PyContourType.Catheter, PyContourType.Wall])
///     smooth (default true): bool smooth after alignment with 3-point moving average
///     postprocessing (default true): adjusts spacing within/between geometry/geometries to have equal offsets
///
/// Returns:
///     A tuple ``(rest_pair, stress_pair)`` of type ``(PyGeometryPair, PyGeometryPair)``,
///     containing the interpolated diastole/systole geometries for REST and STRESS.
///     A 4-tuple of ``Vec<id, matched_to, rel_rot_deg, total_rot_deg, tx, ty, centroid_x, centroid_y>``
///     for (diastole logs, systole logs, diastole stress logs, systole stress logs).
///
/// Raises:
///     **RuntimeError** if any Rust‑side processing fails.
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
///        output_path_a="out/rest",
///        output_path_b="out/stress"
///    )
///
#[pyfunction]
#[pyo3(
    signature = (
        input_data_a,
        input_data_b,
        input_data_c,
        input_data_d,
        label = "double_pair",
        diastole = true,
        step_rotation_deg = 0.5f64,
        range_rotation_deg = 90.0f64,
        image_center = (4.5f64, 4.5f64),
        radius = 0.5f64,
        n_points = 20u32,
        write_obj = true,
        watertight = true,
        output_path_a = "output/rest",
        output_path_b = "output/stress",
        interpolation_steps = 28usize,
        bruteforce = false,
        sample_size= 200,
        contour_types = vec![PyContourType::Lumen, PyContourType::Catheter, PyContourType::Wall],
        smooth = true,
        postprocessing = true,
    )
)]
pub fn from_array_doublepair(
    input_data_a: PyInputData,
    input_data_b: PyInputData,
    input_data_c: PyInputData,
    input_data_d: PyInputData,
    label: &str,
    diastole: bool,
    step_rotation_deg: f64,
    range_rotation_deg: f64,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
    write_obj: bool,
    watertight: bool,
    output_path_a: &str,
    output_path_b: &str,
    interpolation_steps: usize,
    bruteforce: bool,
    sample_size: usize,
    contour_types: Vec<PyContourType>,
    smooth: bool,
    postprocessing: bool,
) -> PyResult<(
    PyGeometryPair,
    PyGeometryPair,
    (
        Vec<(u32, u32, f64, f64, f64, f64, f64)>,
        Vec<(u32, u32, f64, f64, f64, f64, f64)>,
        Vec<(u32, u32, f64, f64, f64, f64, f64)>,
        Vec<(u32, u32, f64, f64, f64, f64, f64)>,
    ),
)> {
    let rust_contour_types: Vec<crate::intravascular::io::geometry::ContourType> =
        contour_types.iter().map(|ct| ct.into()).collect();

    let input_data_a_rust: InputData = input_data_a.try_into().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to convert input_data_a: {}", e))
    })?;
    let input_data_b_rust: InputData = input_data_b.try_into().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to convert input_data_b: {}", e))
    })?;
    let input_data_c_rust: InputData = input_data_c.try_into().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to convert input_data_c: {}", e))
    })?;
    let input_data_d_rust: InputData = input_data_d.try_into().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to convert input_data_d: {}", e))
    })?;

    let (geom_ab_final, geom_cd_final, logs_a, logs_b, logs_c, logs_d) = double_pair_processing_rs(
        label.to_string(),
        image_center,
        radius,
        n_points,
        None,
        None,
        Some(input_data_a_rust),
        Some(input_data_b_rust),
        Some(input_data_c_rust),
        Some(input_data_d_rust),
        diastole,
        write_obj,
        interpolation_steps,
        rust_contour_types,
        watertight,
        output_path_a,
        output_path_b,
        step_rotation_deg,
        range_rotation_deg,
        smooth,
        bruteforce,
        sample_size,
        postprocessing,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let py_geom_ab = geom_ab_final.into();
    let py_geom_cd = geom_cd_final.into();
    let py_logs_a = logs_to_tuples(logs_a);
    let py_logs_b = logs_to_tuples(logs_b);
    let py_logs_c = logs_to_tuples(logs_c);
    let py_logs_d = logs_to_tuples(logs_d);

    Ok((
        py_geom_ab,
        py_geom_cd,
        (py_logs_a, py_logs_b, py_logs_c, py_logs_d),
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
/// Args:
///     input_data_a: Input ``PyInputData`` at e.g diastole for REST.  
///     input_data_b: Input ``PyInputData`` at e.g. systole for REST.  
///     step_rotation_deg (default 0.5°): Rotation step in degree
///     range_rotation_deg (default 90°): Rotation (+/-) range in degree, for 90° total range 180°
///     write_obj (default True): Wether to write OBJ files.
///     watertight (default True): Wether to write shell or watertight mesh to OBJ.
///     output_path: Directory path to write interpolated output files.  
///     interpolation_steps (default 28): Number of steps to interpolate between diastole and systole.  
///     bruteforce (default False): Wether to use bruteforce alignment (comparison for every step size).
///     sample_size (default 200): number of points to downsample to
///     contour_type (default [PyContourType.Lumen, PyContourType.Catheter, PyContourType.Wall])
///     smooth (default true): bool smooth after alignment with 3-point moving average
///     postprocessing (default true): adjusts spacing within/between geometry/geometries to have equal offsets
///
/// Returns:
///     A ``PyGeometryPair`` tuple containing the diastole and systole geometries with interpolation applied.
///     A 2-tuple of ``Vec<id, matched_to, rel_rot_deg, total_rot_deg, tx, ty, centroid_x, centroid_y>``
///     for (diastole logs, systole logs).
///
/// Raises:
///     **RuntimeError** if the underlying Rust function fails.
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
        input_data_a,
        input_data_b,
        label = "single_pair",
        diastole = true,
        step_rotation_deg = 0.5f64,
        range_rotation_deg = 90.0f64,
        image_center = (4.5f64, 4.5f64),
        radius = 0.5f64,
        n_points = 20u32,
        write_obj = true,
        watertight = true,
        output_path = "output/singlepair",
        interpolation_steps = 28usize,
        bruteforce = false,
        sample_size= 200,
        contour_types = vec![PyContourType::Lumen, PyContourType::Catheter, PyContourType::Wall],
        smooth = true,
        postprocessing = true,
    )
)]
pub fn from_array_singlepair(
    input_data_a: PyInputData,
    input_data_b: PyInputData,
    label: &str,
    diastole: bool,
    step_rotation_deg: f64,
    range_rotation_deg: f64,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
    write_obj: bool,
    watertight: bool,
    output_path: &str,
    interpolation_steps: usize,
    bruteforce: bool,
    sample_size: usize,
    contour_types: Vec<PyContourType>,
    smooth: bool,
    postprocessing: bool,
) -> PyResult<(
    PyGeometryPair,
    (
        Vec<(u32, u32, f64, f64, f64, f64, f64)>,
        Vec<(u32, u32, f64, f64, f64, f64, f64)>,
    ),
)> {
    let rust_contour_types: Vec<crate::intravascular::io::geometry::ContourType> =
        contour_types.iter().map(|ct| ct.into()).collect();

    let input_data_a_rust: InputData = input_data_a.try_into().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to convert input_data_a: {}", e))
    })?;
    let input_data_b_rust: InputData = input_data_b.try_into().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to convert input_data_b: {}", e))
    })?;

    let (geom_ab_final, logs_a, logs_b) = pair_processing_rs(
        label.to_string(),
        image_center,
        radius,
        n_points,
        None,
        Some(input_data_a_rust),
        Some(input_data_b_rust),
        diastole,
        write_obj,
        interpolation_steps,
        rust_contour_types,
        watertight,
        output_path,
        step_rotation_deg,
        range_rotation_deg,
        smooth,
        bruteforce,
        sample_size,
        postprocessing,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let py_geom_ab = geom_ab_final.into();
    let py_logs_a = logs_to_tuples(logs_a);
    let py_logs_b = logs_to_tuples(logs_b);

    Ok((py_geom_ab, (py_logs_a, py_logs_b)))
}

/// Processes a single geometry (either diastole or systole) from an IVUS CSV file.
///
/// .. code-block:: text
///
///    Rest/Stress pipeline (choose phase via `diastole` flag):
///      e.g. diastole
///
/// Args:
///     input_data_a: Input ``PyInputData`` at e.g diastole for REST. ///     step_rotation_deg (default 0.5°): Rotation step in degree
///     range_rotation_deg (default 90°): Rotation (+/-) range in degree, for 90° total range 180°
///     diastole (default true): If true, process the diastole phase; otherwise systole.  
///     image_center (default (4.5mm, 4.5mm)): (x, y) center for processing.  
///     radius (default: 0.5mm): Radius around center to consider for catheter.  
///     n_points (default 20): number of points for catheter, more points stronger influence of image center.
///     write_obj (default true): Wether to write OBJ files.
///     watertight (default true): Wether to write shell or watertight mesh to OBJ.
///     output_path (default "output/single"): Where to write the processed geometry.  
///     bruteforce (default false): Wether to use bruteforce alignment (comparison for every step size).
///     sample_size (default 200): number of points to downsample to
///     contour_type (default [PyContourType.Lumen, PyContourType.Catheter, PyContourType.Wall])
///     smooth (default true): bool smooth after alignment with 3-point moving average
///     postprocessing (default true): adjusts spacing within/between geometry/geometries to have equal offsets
///
/// Returns:
///     A ``PyGeometry`` containing the processed contour for the chosen phase.
///     A ``Vec<id, matched_to, rel_rot_deg, total_rot_deg, tx, ty, centroid_x, centroid_y>``.
///
/// Raises:
///     **RuntimeError** if the underlying Rust pipeline fails.
///
/// Example:
///     >>> import multimodars as mm
///     >>> geom, _ = mm.from_array_single(
///     ...     input_data,
///     ...     steps_best_rotation=0.5,
///     ...     range_rotation_rad=90,
///     ...     output_path="out/single",
///     ...     diastole=False
///     ... )
///
#[pyfunction]
#[pyo3(signature = (
    input_data,
    label = "single",
    diastole = true,
    step_rotation_deg = 0.5f64,
    range_rotation_deg = 90.0f64,
    image_center = (4.5f64, 4.5f64),
    radius = 0.5f64,
    n_points = 20u32,
    write_obj=false,
    watertight = true,
    output_path="output/single",
    bruteforce = false,
    sample_size = 200,
    contour_types = vec![PyContourType::Lumen, PyContourType::Catheter, PyContourType::Wall],
    smooth = true,
))]
pub fn from_array_single(
    input_data: PyInputData,
    label: &str,
    diastole: bool,
    step_rotation_deg: f64,
    range_rotation_deg: f64,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
    write_obj: bool,
    watertight: bool,
    output_path: &str,
    bruteforce: bool,
    sample_size: usize,
    contour_types: Vec<PyContourType>,
    smooth: bool,
) -> PyResult<(PyGeometry, Vec<(u32, u32, f64, f64, f64, f64, f64)>)> {
    let rust_contour_types: Vec<crate::intravascular::io::geometry::ContourType> =
        contour_types.iter().map(|ct| ct.into()).collect();

    let input_data_rust: InputData = input_data.try_into().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to convert input_data_a: {}", e))
    })?;

    let (geom_rs, logs) = single_processing_rs(
        label.to_string(),
        image_center,
        radius,
        n_points,
        None,
        Some(input_data_rust),
        diastole,
        write_obj,
        watertight,
        rust_contour_types,
        output_path,
        step_rotation_deg,
        range_rotation_deg,
        smooth,
        bruteforce,
        sample_size,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let py_geom = PyGeometry::from(geom_rs);
    let py_logs = logs_to_tuples(logs);
    Ok((py_geom, py_logs))
}

/// Convert a ``PyGeometry`` object into one or more OBJ files and write them to disk.
///
/// This function takes a Python-exposed geometry (``PyGeometry``), converts it into the
/// corresponding Rust geometry, and writes the specified contour types as OBJ meshes
/// without UV coordinates. Each contour type is written to its own file, with a
/// corresponding MTL material file.
///
/// Args:
///     geometry: Input ``PyGeometry`` instance containing the mesh data.
///     output_path: Directory path where the OBJ and MTL files will be written.
///     watertight (default True): Whether to write shell or watertight mesh to OBJ.
///     contour_types (default [Lumen, Catheter, Wall]): Which contour types to export.
///     filename_prefix (default ""): Optional prefix for all filenames.
///
/// Returns:
///     Returns a `PyRuntimeError` if any of the underlying file writes fail.
#[pyfunction]
#[pyo3(
    signature = (
        geometry,
        output_path,
        watertight = true,
        contour_types = vec![PyContourType::Lumen, PyContourType::Catheter, PyContourType::Wall],
        filename_prefix = "",
    )
)]
pub fn to_obj(
    geometry: PyGeometry,
    output_path: &str,
    watertight: bool,
    contour_types: Vec<PyContourType>,
    filename_prefix: &str,
) -> PyResult<()> {
    // Convert the Python geometry to Rust representation
    let geometry_rs = geometry.to_rust_geometry().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to convert geometry: {}", e))
    })?;

    // Ensure output directory exists
    std::fs::create_dir_all(output_path).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Could not create output directory '{}': {}",
            output_path, e
        ))
    })?;

    // Write each requested contour type
    for contour_type in contour_types {
        let contours = extract_contours_by_type(&geometry_rs, contour_type.into());

        if contours.is_empty() {
            eprintln!(
                "Warning: No contours found for type {:?}, skipping",
                contour_type
            );
            continue;
        }

        let type_name = get_contour_type_name(contour_type.into());
        let filename = if filename_prefix.is_empty() {
            format!("{}.obj", type_name)
        } else {
            format!("{}_{}.obj", filename_prefix, type_name)
        };
        let material_name = if filename_prefix.is_empty() {
            format!("{}.mtl", type_name)
        } else {
            format!("{}_{}.mtl", filename_prefix, type_name)
        };

        let obj_path = Path::new(output_path).join(&filename);
        let mtl_path = Path::new(output_path).join(&material_name);

        // Create appropriate MTL file based on contour type
        create_mtl_for_contour_type(contour_type.into(), &mtl_path, &filename).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to create mtl for geometry: {}",
                e
            ))
        })?;

        // Write OBJ without UV coordinates
        write_obj_mesh_without_uv(
            &contours,
            obj_path.to_str().unwrap(),
            mtl_path.to_str().unwrap(),
            watertight,
        )
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to write {} OBJ: {}",
                type_name, e
            ))
        })?;

        println!("Successfully wrote {} to {}", type_name, obj_path.display());
    }

    Ok(())
}
