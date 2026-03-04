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

/// Process four intravascular imaging geometries in parallel from CSV folders.
///
/// Reads REST and STRESS acquisitions from two input folders, aligns frames
/// within and between each cardiac phase in parallel, and writes interpolated
/// OBJ meshes.
///
/// .. code-block:: text
///
///    Rest           Stress
///    diastole ──▶ diastole
///       │            │
///       ▼            ▼
///    systole ──▶ systole
///
/// .. warning::
///
///    The CSV must have **no header**. Each row is
///    ``(frame index, x-coord (mm), y-coord (mm), z-coord (mm))``:
///
/// .. code-block:: text
///
///    185, 5.32, 2.37, 0.0
///    ...
///
/// Parameters
/// ----------
/// input_path_a : str
///     Path to the REST input folder.
/// input_path_b : str
///     Path to the STRESS input folder.
/// label : str, optional
///     Label used for output filenames.  Default is ``"full"``.
/// diastole : bool, optional
///     Whether to process the diastolic phase.  Default is ``True``.
/// step_rotation_deg : float, optional
///     Rotation step in degrees.  Default is ``0.5``.
/// range_rotation_deg : float, optional
///     Rotation search range (±) in degrees; a range of 90° gives 180°
///     total.  Default is ``90.0``.
/// image_center : tuple of float, optional
///     Image center ``(x, y)`` in mm.  Default is ``(4.5, 4.5)``.
/// radius : float, optional
///     Catheter radius in mm.  Default is ``0.5``.
/// n_points : int, optional
///     Number of catheter points; more points increases the influence of
///     the image center.  Default is ``20``.
/// write_obj : bool, optional
///     Whether to write OBJ files to disk.  Default is ``True``.
/// watertight : bool, optional
///     Whether to write a watertight or shell mesh.  Default is ``True``.
/// output_path_a : str, optional
///     Output directory for REST results.  Default is ``"output/rest"``.
/// output_path_b : str, optional
///     Output directory for STRESS results.  Default is ``"output/stress"``.
/// output_path_c : str, optional
///     Output directory for DIASTOLE results.  Default is
///     ``"output/diastole"``.
/// output_path_d : str, optional
///     Output directory for SYSTOLE results.  Default is
///     ``"output/systole"``.
/// interpolation_steps : int, optional
///     Number of interpolated meshes between phases.  Default is ``28``.
/// bruteforce : bool, optional
///     Whether to use brute-force alignment (one comparison per step).
///     Default is ``False``.
/// sample_size : int, optional
///     Number of points to downsample to during alignment.  Default is
///     ``500``.
/// contour_types : list of PyContourType, optional
///     Contour types to export.  Default is
///     ``[PyContourType.Lumen, PyContourType.Catheter, PyContourType.Wall]``.
/// smooth : bool, optional
///     Whether to smooth frames after alignment using a 3-point moving
///     average.  Default is ``True``.
/// postprocessing : bool, optional
///     Whether to adjust spacing within/between geometries to equal
///     offsets.  Default is ``True``.
///
/// Returns
/// -------
/// rest : PyGeometryPair
///     Aligned geometry pair for the REST condition.
/// stress : PyGeometryPair
///     Aligned geometry pair for the STRESS condition.
/// diastole : PyGeometryPair
///     Aligned geometry pair for the diastolic phase.
/// systole : PyGeometryPair
///     Aligned geometry pair for the systolic phase.
/// logs : tuple of list
///     4-tuple of alignment logs ``(logs_a, logs_b, logs_c, logs_d)``;
///     each entry is a list of
///     ``(id, matched_to, rel_rot_deg, total_rot_deg, tx, ty, centroid_x, centroid_y)``.
///
/// Examples
/// --------
/// >>> import multimodars as mm
/// >>> rest, stress, dia, sys, _ = mm.from_file_full(
/// ...     "data/ivus_rest", "data/ivus_stress"
/// ... )
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

/// Process two intravascular imaging geometry pairs in parallel from CSV folders.
///
/// Reads REST and STRESS acquisitions and aligns frames within each pair
/// independently (diastole → systole), then writes interpolated OBJ meshes.
///
/// .. code-block:: text
///
///    Rest:                    Stress:
///    diastole                  diastole
///        │                         │
///        ▼                         ▼
///    systole                   systole
///
/// .. warning::
///
///    The CSV must have **no header**. Each row is
///    ``(frame index, x-coord (mm), y-coord (mm), z-coord (mm))``:
///
/// .. code-block:: text
///
///    185, 5.32, 2.37, 0.0
///    ...
///
/// Parameters
/// ----------
/// input_path_a : str
///     Path to the REST input folder.
/// input_path_b : str
///     Path to the STRESS input folder.
/// label : str, optional
///     Label used for output filenames.  Default is ``"double_pair"``.
/// diastole : bool, optional
///     Whether to process the diastolic phase.  Default is ``True``.
/// step_rotation_deg : float, optional
///     Rotation step in degrees.  Default is ``0.5``.
/// range_rotation_deg : float, optional
///     Rotation search range (±) in degrees.  Default is ``90.0``.
/// image_center : tuple of float, optional
///     Image center ``(x, y)`` in mm.  Default is ``(4.5, 4.5)``.
/// radius : float, optional
///     Catheter radius in mm.  Default is ``0.5``.
/// n_points : int, optional
///     Number of catheter points.  Default is ``20``.
/// write_obj : bool, optional
///     Whether to write OBJ files.  Default is ``True``.
/// watertight : bool, optional
///     Whether to write a watertight or shell mesh.  Default is ``True``.
/// output_path_a : str, optional
///     Output directory for REST results.  Default is ``"output/rest"``.
/// output_path_b : str, optional
///     Output directory for STRESS results.  Default is ``"output/stress"``.
/// interpolation_steps : int, optional
///     Number of interpolated meshes.  Default is ``28``.
/// bruteforce : bool, optional
///     Whether to use brute-force alignment.  Default is ``False``.
/// sample_size : int, optional
///     Number of points to downsample to.  Default is ``500``.
/// contour_types : list of PyContourType, optional
///     Contour types to export.  Default is
///     ``[PyContourType.Lumen, PyContourType.Catheter, PyContourType.Wall]``.
/// smooth : bool, optional
///     Whether to smooth frames after alignment.  Default is ``True``.
/// postprocessing : bool, optional
///     Whether to equalise spacing within/between geometries.  Default is
///     ``True``.
///
/// Returns
/// -------
/// rest : PyGeometryPair
///     Aligned geometry pair for the REST condition.
/// stress : PyGeometryPair
///     Aligned geometry pair for the STRESS condition.
/// logs : tuple of list
///     4-tuple of alignment logs ``(logs_a, logs_b, logs_c, logs_d)``;
///     each entry is a list of
///     ``(id, matched_to, rel_rot_deg, total_rot_deg, tx, ty, centroid_x, centroid_y)``.
///
/// Examples
/// --------
/// >>> import multimodars as mm
/// >>> rest, stress, _ = mm.from_file_doublepair(
/// ...     "data/ivus_rest", "data/ivus_stress"
/// ... )
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

/// Process a single diastole/systole pair from an input CSV folder.
///
/// Reads one acquisition folder, aligns the diastolic and systolic frames,
/// and returns a single ``PyGeometryPair``.
///
/// .. code-block:: text
///
///    Rest/Stress pipeline:
///       diastole
///         │
///         ▼
///       systole
///
/// .. warning::
///
///    The CSV must have **no header**. Each row is
///    ``(frame index, x-coord (mm), y-coord (mm), z-coord (mm))``:
///
/// .. code-block:: text
///
///    185, 5.32, 2.37, 0.0
///    ...
///
/// Parameters
/// ----------
/// input_path : str
///     Path to the input CSV file.
/// label : str, optional
///     Label used for output filenames.  Default is ``"single_pair"``.
/// diastole : bool, optional
///     Whether to process the diastolic phase.  Default is ``True``.
/// step_rotation_deg : float, optional
///     Rotation step in degrees.  Default is ``0.5``.
/// range_rotation_deg : float, optional
///     Rotation search range (±) in degrees.  Default is ``90.0``.
/// image_center : tuple of float, optional
///     Image center ``(x, y)`` in mm.  Default is ``(4.5, 4.5)``.
/// radius : float, optional
///     Catheter radius in mm.  Default is ``0.5``.
/// n_points : int, optional
///     Number of catheter points.  Default is ``20``.
/// write_obj : bool, optional
///     Whether to write OBJ files.  Default is ``True``.
/// watertight : bool, optional
///     Whether to write a watertight or shell mesh.  Default is ``True``.
/// output_path : str, optional
///     Directory path to write the processed geometry.  Default is
///     ``"output/singlepair"``.
/// interpolation_steps : int, optional
///     Number of interpolated meshes.  Default is ``28``.
/// bruteforce : bool, optional
///     Whether to use brute-force alignment.  Default is ``False``.
/// sample_size : int, optional
///     Number of points to downsample to.  Default is ``500``.
/// contour_types : list of PyContourType, optional
///     Contour types to export.  Default is
///     ``[PyContourType.Lumen, PyContourType.Catheter, PyContourType.Wall]``.
/// smooth : bool, optional
///     Whether to smooth frames after alignment.  Default is ``True``.
/// postprocessing : bool, optional
///     Whether to equalise spacing within/between geometries.  Default is
///     ``True``.
///
/// Returns
/// -------
/// pair : PyGeometryPair
///     Aligned diastole/systole geometry pair.
/// logs : tuple of list
///     2-tuple of alignment logs ``(logs_a, logs_b)``; each entry is a list
///     of ``(id, matched_to, rel_rot_deg, total_rot_deg, tx, ty, centroid_x, centroid_y)``.
///
/// Raises
/// ------
/// RuntimeError
///     If the Rust processing pipeline fails.
///
/// Examples
/// --------
/// >>> import multimodars as mm
/// >>> pair, _ = mm.from_file_singlepair(
/// ...     "data/ivus_rest.csv",
/// ...     "output/rest"
/// ... )
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

/// Process a single intravascular imaging geometry from a CSV file.
///
/// Reads one phase (diastole or systole) from an IVUS CSV file, aligns frames
/// within the geometry, and optionally writes OBJ output.
///
/// .. code-block:: text
///
///    Rest/Stress pipeline (choose phase via `diastole` flag):
///      e.g. diastole
///
/// Parameters
/// ----------
/// input_path : str
///     Path to the input CSV (no header; columns: frame, x, y, z).
/// label : str, optional
///     Label used for output filenames.  Default is ``"single"``.
/// diastole : bool, optional
///     When ``True`` process the diastolic phase; otherwise systole.
///     Default is ``True``.
/// step_rotation_deg : float, optional
///     Rotation step in degrees.  Default is ``0.5``.
/// range_rotation_deg : float, optional
///     Rotation search range (±) in degrees.  Default is ``90.0``.
/// image_center : tuple of float, optional
///     Image center ``(x, y)`` in mm.  Default is ``(4.5, 4.5)``.
/// radius : float, optional
///     Catheter radius in mm.  Default is ``0.5``.
/// n_points : int, optional
///     Number of catheter points.  Default is ``20``.
/// write_obj : bool, optional
///     Whether to write OBJ files.  Default is ``True``.
/// watertight : bool, optional
///     Whether to write a watertight or shell mesh.  Default is ``True``.
/// output_path : str, optional
///     Directory path to write the processed geometry.  Default is
///     ``"output/single"``.
/// bruteforce : bool, optional
///     Whether to use brute-force alignment.  Default is ``False``.
/// sample_size : int, optional
///     Number of points to downsample to.  Default is ``200``.
/// contour_types : list of PyContourType, optional
///     Contour types to export.  Default is
///     ``[PyContourType.Lumen, PyContourType.Catheter, PyContourType.Wall]``.
/// smooth : bool, optional
///     Whether to smooth frames after alignment.  Default is ``True``.
///
/// Returns
/// -------
/// geom : PyGeometry
///     Processed geometry for the chosen phase.
/// logs : list
///     Alignment log entries; each entry is
///     ``(id, matched_to, rel_rot_deg, total_rot_deg, tx, ty, centroid_x, centroid_y)``.
///
/// Raises
/// ------
/// RuntimeError
///     If the underlying Rust pipeline fails.
///
/// Examples
/// --------
/// >>> import multimodars as mm
/// >>> geom, _ = mm.from_file_single(
/// ...     "data/ivus.csv",
/// ...     steps_best_rotation=0.5,
/// ...     range_rotation_rad=90,
/// ...     output_path="out/single",
/// ...     diastole=False
/// ... )
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

/// Process four ``PyInputData`` objects in parallel, aligning and interpolating between phases.
///
/// Accepts pre-loaded input data for REST diastole, REST systole, STRESS
/// diastole, and STRESS systole, then aligns frames within and between
/// each cardiac phase.
///
/// .. code-block:: text
///
///    Rest           Stress
///    diastole ──▶ diastole
///       │            │
///       ▼            ▼
///    systole ──▶ systole
///
/// Parameters
/// ----------
/// input_data_a : PyInputData
///     Diastolic REST input data.
/// input_data_b : PyInputData
///     Systolic REST input data.
/// input_data_c : PyInputData
///     Diastolic STRESS input data.
/// input_data_d : PyInputData
///     Systolic STRESS input data.
/// label : str, optional
///     Label used for output filenames.  Default is ``"full"``.
/// diastole : bool, optional
///     Whether to process the diastolic phase.  Default is ``True``.
/// step_rotation_deg : float, optional
///     Rotation step in degrees.  Default is ``0.5``.
/// range_rotation_deg : float, optional
///     Rotation search range (±) in degrees.  Default is ``90.0``.
/// image_center : tuple of float, optional
///     Image center ``(x, y)`` in mm.  Default is ``(4.5, 4.5)``.
/// radius : float, optional
///     Catheter radius in mm.  Default is ``0.5``.
/// n_points : int, optional
///     Number of catheter points.  Default is ``20``.
/// write_obj : bool, optional
///     Whether to write OBJ files.  Default is ``True``.
/// watertight : bool, optional
///     Whether to write a watertight or shell mesh.  Default is ``True``.
/// output_path_a : str, optional
///     Output directory for REST results.  Default is ``"output/rest"``.
/// output_path_b : str, optional
///     Output directory for STRESS results.  Default is ``"output/stress"``.
/// output_path_c : str, optional
///     Output directory for DIASTOLE results.  Default is
///     ``"output/diastole"``.
/// output_path_d : str, optional
///     Output directory for SYSTOLE results.  Default is
///     ``"output/systole"``.
/// interpolation_steps : int, optional
///     Number of interpolation steps between phases.  Default is ``28``.
/// bruteforce : bool, optional
///     Whether to use brute-force alignment.  Default is ``False``.
/// sample_size : int, optional
///     Number of points to downsample to.  Default is ``200``.
/// contour_types : list of PyContourType, optional
///     Contour types to export.  Default is
///     ``[PyContourType.Lumen, PyContourType.Catheter, PyContourType.Wall]``.
/// smooth : bool, optional
///     Whether to smooth frames after alignment.  Default is ``True``.
/// postprocessing : bool, optional
///     Whether to equalise spacing within/between geometries.  Default is
///     ``True``.
///
/// Returns
/// -------
/// rest : PyGeometryPair
///     Aligned geometry pair for the REST condition.
/// stress : PyGeometryPair
///     Aligned geometry pair for the STRESS condition.
/// diastole : PyGeometryPair
///     Aligned geometry pair for the diastolic phase.
/// systole : PyGeometryPair
///     Aligned geometry pair for the systolic phase.
/// logs : tuple of list
///     4-tuple of alignment logs ``(logs_a, logs_b, logs_c, logs_d)``;
///     each entry is a list of
///     ``(id, matched_to, rel_rot_deg, total_rot_deg, tx, ty, centroid_x, centroid_y)``.
///
/// Raises
/// ------
/// RuntimeError
///     If any Rust-side processing fails.
///
/// Examples
/// --------
///
/// .. code-block:: python
///
///    import multimodars as mm
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

/// Process two ``PyInputData`` pairs in parallel, aligning frames within each pair independently.
///
/// Accepts pre-loaded data for REST (diastole + systole) and STRESS
/// (diastole + systole), aligns each pair independently, and writes
/// interpolated OBJ meshes.
///
/// .. code-block:: text
///
///    Rest:                    Stress:
///    diastole                  diastole
///        │                         │
///        ▼                         ▼
///    systole                   systole
///
/// Parameters
/// ----------
/// input_data_a : PyInputData
///     Diastolic REST input data.
/// input_data_b : PyInputData
///     Systolic REST input data.
/// input_data_c : PyInputData
///     Diastolic STRESS input data.
/// input_data_d : PyInputData
///     Systolic STRESS input data.
/// label : str, optional
///     Label used for output filenames.  Default is ``"double_pair"``.
/// diastole : bool, optional
///     Whether to process the diastolic phase.  Default is ``True``.
/// step_rotation_deg : float, optional
///     Rotation step in degrees.  Default is ``0.5``.
/// range_rotation_deg : float, optional
///     Rotation search range (±) in degrees.  Default is ``90.0``.
/// image_center : tuple of float, optional
///     Image center ``(x, y)`` in mm.  Default is ``(4.5, 4.5)``.
/// radius : float, optional
///     Catheter radius in mm.  Default is ``0.5``.
/// n_points : int, optional
///     Number of catheter points.  Default is ``20``.
/// write_obj : bool, optional
///     Whether to write OBJ files.  Default is ``True``.
/// watertight : bool, optional
///     Whether to write a watertight or shell mesh.  Default is ``True``.
/// output_path_a : str, optional
///     Output directory for REST results.  Default is ``"output/rest"``.
/// output_path_b : str, optional
///     Output directory for STRESS results.  Default is ``"output/stress"``.
/// interpolation_steps : int, optional
///     Number of interpolation steps between phases.  Default is ``28``.
/// bruteforce : bool, optional
///     Whether to use brute-force alignment.  Default is ``False``.
/// sample_size : int, optional
///     Number of points to downsample to.  Default is ``200``.
/// contour_types : list of PyContourType, optional
///     Contour types to export.  Default is
///     ``[PyContourType.Lumen, PyContourType.Catheter, PyContourType.Wall]``.
/// smooth : bool, optional
///     Whether to smooth frames after alignment.  Default is ``True``.
/// postprocessing : bool, optional
///     Whether to equalise spacing within/between geometries.  Default is
///     ``True``.
///
/// Returns
/// -------
/// rest : PyGeometryPair
///     Aligned geometry pair for the REST condition.
/// stress : PyGeometryPair
///     Aligned geometry pair for the STRESS condition.
/// logs : tuple of list
///     4-tuple of alignment logs ``(logs_a, logs_b, logs_c, logs_d)``;
///     each entry is a list of
///     ``(id, matched_to, rel_rot_deg, total_rot_deg, tx, ty, centroid_x, centroid_y)``.
///
/// Raises
/// ------
/// RuntimeError
///     If any Rust-side processing fails.
///
/// Examples
/// --------
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

/// Align and interpolate between two ``PyInputData`` objects (diastole and systole).
///
/// Accepts pre-loaded diastolic and systolic input data, aligns frames
/// between the two phases, and returns a single ``PyGeometryPair``.
///
/// .. code-block:: text
///
///    Geometry pipeline:
///      diastole ──▶ systole
///
/// Parameters
/// ----------
/// input_data_a : PyInputData
///     Diastolic input data.
/// input_data_b : PyInputData
///     Systolic input data.
/// label : str, optional
///     Label used for output filenames.  Default is ``"single_pair"``.
/// diastole : bool, optional
///     Whether to process the diastolic phase.  Default is ``True``.
/// step_rotation_deg : float, optional
///     Rotation step in degrees.  Default is ``0.5``.
/// range_rotation_deg : float, optional
///     Rotation search range (±) in degrees.  Default is ``90.0``.
/// image_center : tuple of float, optional
///     Image center ``(x, y)`` in mm.  Default is ``(4.5, 4.5)``.
/// radius : float, optional
///     Catheter radius in mm.  Default is ``0.5``.
/// n_points : int, optional
///     Number of catheter points.  Default is ``20``.
/// write_obj : bool, optional
///     Whether to write OBJ files.  Default is ``True``.
/// watertight : bool, optional
///     Whether to write a watertight or shell mesh.  Default is ``True``.
/// output_path : str, optional
///     Directory path to write interpolated output files.  Default is
///     ``"output/singlepair"``.
/// interpolation_steps : int, optional
///     Number of interpolation steps between phases.  Default is ``28``.
/// bruteforce : bool, optional
///     Whether to use brute-force alignment.  Default is ``False``.
/// sample_size : int, optional
///     Number of points to downsample to.  Default is ``200``.
/// contour_types : list of PyContourType, optional
///     Contour types to export.  Default is
///     ``[PyContourType.Lumen, PyContourType.Catheter, PyContourType.Wall]``.
/// smooth : bool, optional
///     Whether to smooth frames after alignment.  Default is ``True``.
/// postprocessing : bool, optional
///     Whether to equalise spacing within/between geometries.  Default is
///     ``True``.
///
/// Returns
/// -------
/// pair : PyGeometryPair
///     Aligned diastole/systole geometry pair with interpolation applied.
/// logs : tuple of list
///     2-tuple of alignment logs ``(logs_a, logs_b)``; each entry is a list
///     of ``(id, matched_to, rel_rot_deg, total_rot_deg, tx, ty, centroid_x, centroid_y)``.
///
/// Raises
/// ------
/// RuntimeError
///     If the underlying Rust function fails.
///
/// Examples
/// --------
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

/// Process a single geometry phase from a ``PyInputData`` object.
///
/// Accepts pre-loaded input data for one cardiac phase, aligns frames
/// within the geometry, and optionally writes OBJ output.
///
/// .. code-block:: text
///
///    Rest/Stress pipeline (choose phase via `diastole` flag):
///      e.g. diastole
///
/// Parameters
/// ----------
/// input_data : PyInputData
///     Input data for a single cardiac phase (e.g. diastolic REST).
/// label : str, optional
///     Label used for output filenames.  Default is ``"single"``.
/// diastole : bool, optional
///     When ``True`` process the diastolic phase; otherwise systole.
///     Default is ``True``.
/// step_rotation_deg : float, optional
///     Rotation step in degrees.  Default is ``0.5``.
/// range_rotation_deg : float, optional
///     Rotation search range (±) in degrees.  Default is ``90.0``.
/// image_center : tuple of float, optional
///     Image center ``(x, y)`` in mm.  Default is ``(4.5, 4.5)``.
/// radius : float, optional
///     Catheter radius in mm.  Default is ``0.5``.
/// n_points : int, optional
///     Number of catheter points.  Default is ``20``.
/// write_obj : bool, optional
///     Whether to write OBJ files.  Default is ``False``.
/// watertight : bool, optional
///     Whether to write a watertight or shell mesh.  Default is ``True``.
/// output_path : str, optional
///     Directory path to write the processed geometry.  Default is
///     ``"output/single"``.
/// bruteforce : bool, optional
///     Whether to use brute-force alignment.  Default is ``False``.
/// sample_size : int, optional
///     Number of points to downsample to.  Default is ``200``.
/// contour_types : list of PyContourType, optional
///     Contour types to export.  Default is
///     ``[PyContourType.Lumen, PyContourType.Catheter, PyContourType.Wall]``.
/// smooth : bool, optional
///     Whether to smooth frames after alignment.  Default is ``True``.
///
/// Returns
/// -------
/// geom : PyGeometry
///     Processed geometry for the chosen phase.
/// logs : list
///     Alignment log entries; each entry is
///     ``(id, matched_to, rel_rot_deg, total_rot_deg, tx, ty, centroid_x, centroid_y)``.
///
/// Raises
/// ------
/// RuntimeError
///     If the underlying Rust pipeline fails.
///
/// Examples
/// --------
/// >>> import multimodars as mm
/// >>> geom, _ = mm.from_array_single(
/// ...     input_data,
/// ...     steps_best_rotation=0.5,
/// ...     range_rotation_rad=90,
/// ...     output_path="out/single",
/// ...     diastole=False
/// ... )
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
/// Converts the Python-exposed geometry into the corresponding Rust
/// representation and writes the specified contour types as OBJ meshes
/// without UV coordinates.  Each contour type is written to its own file
/// together with a corresponding MTL material file.
///
/// Parameters
/// ----------
/// geometry : PyGeometry
///     Input geometry instance containing the mesh data.
/// output_path : str
///     Directory path where the OBJ and MTL files will be written.
/// watertight : bool, optional
///     Whether to write a watertight or shell mesh.  Default is ``True``.
/// contour_types : list of PyContourType, optional
///     Contour types to export.  Default is
///     ``[PyContourType.Lumen, PyContourType.Catheter, PyContourType.Wall]``.
/// filename_prefix : str, optional
///     Optional prefix prepended to all output filenames.  Default is
///     ``""``.
///
/// Raises
/// ------
/// RuntimeError
///     If any of the underlying file writes fail.
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
