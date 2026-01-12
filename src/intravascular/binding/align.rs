use crate::intravascular::{
    binding::classes::{PyCenterline, PyContourType, PyGeometryPair},
    centerline_align::{align_combined_rs, align_manual_rs, align_three_point_rs},
};
use pyo3::prelude::*;

/// Creates centerline-aligned meshes for diastolic and systolic geometries
/// based on three reference points (aorta, upper section, lower section).
/// Only works for elliptic vessels e.g. coronary artery anomalies.
///
/// Args:
///     centerline: PyCenterline object
///     geometry_pair: PyGeometryPair object
///     aortic_ref_pt: Reference point for aortic position
///     upper_ref_pt: Upper reference point
///     lower_ref_pt: Lower reference point
///     angle_step_deg (default 1.0): step size in degrees for rotation search
///     write (default false): Wether to write aligned meshes to OBJ
///     watertight (default true): Wether to write shell or watertight mesh to OBJ.
///     interpolation_steps: Number of interpolation steps
///     output_dir (default "output/aligned"): Output directory for aligned meshes
///     case_name (default "None"): Case name for output files
///
/// Returns:
///     PyGeometryPair, PyCenterline (resampled)
///
/// Example:
///     >>> import multimodars as mm
///     >>> dia, sys = mm.align_three_point(
///     ...     centerline,
///     ...     geometry_pair,
///     ...     (12.2605, -201.3643, 1751.0554),
///     ...     (11.7567, -202.1920, 1754.7975),
///     ...     (15.6605, -202.1920, 1749.9655),
///     ... )
#[pyfunction]
#[pyo3(
    signature = (
        centerline,
        geometry_pair,
        aortic_ref_pt,
        upper_ref_pt,
        lower_ref_pt,
        angle_step_deg=1.0,
        write=false,
        watertight=true,
        interpolation_steps=28usize,
        output_dir="output/aligned",
        contour_types=vec![PyContourType::Lumen, PyContourType::Catheter, PyContourType::Wall],
        case_name="None",
    )
)]
pub fn align_three_point(
    centerline: PyCenterline,
    geometry_pair: PyGeometryPair,
    aortic_ref_pt: (f64, f64, f64),
    upper_ref_pt: (f64, f64, f64),
    lower_ref_pt: (f64, f64, f64),
    angle_step_deg: f64,
    write: bool,
    watertight: bool,
    interpolation_steps: usize,
    output_dir: &str,
    contour_types: Vec<PyContourType>,
    case_name: &str,
) -> PyResult<(PyGeometryPair, PyCenterline)> {
    let rust_contour_types: Vec<crate::intravascular::io::geometry::ContourType> =
        contour_types.iter().map(|ct| ct.into()).collect();
    let cl_rs = centerline.to_rust_centerline();
    let geom_pair_rs = geometry_pair.to_rust_geometry_pair();
    let angle_step = angle_step_deg.to_radians();

    let (geom_pair, cl) = align_three_point_rs(
        cl_rs,
        geom_pair_rs,
        aortic_ref_pt,
        upper_ref_pt,
        lower_ref_pt,
        angle_step,
        write,
        watertight,
        interpolation_steps,
        output_dir,
        rust_contour_types,
        case_name,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let aligned_py: PyGeometryPair = geom_pair.into();
    let resampled_cl: PyCenterline = cl.into();

    Ok((aligned_py, resampled_cl))
}

/// Creates centerline-aligned meshes for diastolic and systolic geometries
/// based on three reference points (aorta, upper section, lower section).
/// Only works for elliptic vessels e.g. coronary artery anomalies.
///
/// Args:
///     centerline: PyCenterline object
///     geometry_pair: PyGeometryPair object
///     rotation_angle: Rotation angle in radians
///     start_point: Index of centerline point to use as reference point
///     write (default false): Wether to write aligned meshes to OBJ
///     watertight (default true): Wether to write shell or watertight mesh to OBJ.
///     interpolation_steps: Number of interpolation steps
///     output_dir (default "output/aligned"): Output directory for aligned meshes
///     case_name (default "None"): Case name for output files
///
/// Returns:
///     PyGeometryPair
///
/// Example:
///     >>> import multimodars as mm
///     >>> dia, sys = mm.centerline_align(
///     ...     "path/to/centerline.csv",
///     ...     (1.0, 2.0, 3.0),
///     ...     (4.0, 5.0, 6.0),
///     ...     (7.0, 8.0, 9.0),
///     ...     "rest"
///     ... )
#[pyfunction]
#[pyo3(
    signature = (
        centerline,
        geometry_pair,
        rotation_angle,
        ref_point,
        write=false,
        watertight=true,
        interpolation_steps=28usize,
        output_dir="output/aligned",
        contour_types=vec![PyContourType::Lumen, PyContourType::Catheter, PyContourType::Wall],
        case_name="None",
    )
)]
pub fn align_manual(
    centerline: PyCenterline,
    geometry_pair: PyGeometryPair,
    rotation_angle: f64,
    ref_point: (f64, f64, f64),
    write: bool,
    watertight: bool,
    interpolation_steps: usize,
    output_dir: &str,
    contour_types: Vec<PyContourType>,
    case_name: &str,
) -> PyResult<(PyGeometryPair, PyCenterline)> {
    let rust_contour_types: Vec<crate::intravascular::io::geometry::ContourType> =
        contour_types.iter().map(|ct| ct.into()).collect();
    let cl_rs = centerline.to_rust_centerline();
    let geom_pair_rs = geometry_pair.to_rust_geometry_pair();

    let (geom_pair, cl) = align_manual_rs(
        cl_rs,
        geom_pair_rs,
        rotation_angle,
        ref_point,
        write,
        watertight,
        interpolation_steps,
        output_dir,
        rust_contour_types,
        case_name,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let aligned_py: PyGeometryPair = geom_pair.into();
    let resampled_cl: PyCenterline = cl.into();

    Ok((aligned_py, resampled_cl))
}

/// Creates centerline-aligned meshes for diastolic and systolic geometries
/// based on one reference point (e.g. aorta ostium, upper, lower) and Hausdorff distances.
///
/// Args:
///     centerline: PyCenterline object
///     geometry_pair: PyGeometryPair object
///     aortic_ref_pt: Reference point for aortic position
///     points: List of points to use for Hausdorff distance calculation
///     angle_step_deg (default 1.0): step size in degrees for rotation search
///     write (default false): Wether to write aligned meshes to OBJ
///     watertight (default true): Wether to write shell or watertight mesh to OBJ.
///     interpolation_steps: Number of interpolation steps
///     output_dir (default "output/aligned"): Output directory for aligned meshes
///     case_name (default "None"): Case name for output files
///
/// Returns:
///     PyGeometryPair, PyCenterline (resampled)
///
/// Example:
///     >>> import multimodars as mm
///     >>> dia, sys = mm.align_three_point(
///     ...     centerline,
///     ...     geometry_pair,
///     ...     (12.2605, -201.3643, 1751.0554),
///     ...     [(11.7567, -202.1920, 1754.7975), ... ],
///     ... )
#[pyfunction]
#[pyo3(
    signature = (
        centerline,
        geom_pair,
        aortic_ref_pt,
        upper_ref_pt,
        lower_ref_pt,
        points,
        angle_step_deg=1.0,
        angle_range_deg=15.0,
        index_range=2,
        write=false,
        watertight=true,
        interpolation_steps=28usize,
        output_dir="output/aligned",
        contour_types=vec![PyContourType::Lumen, PyContourType::Catheter, PyContourType::Wall],
        case_name="None",
    )
)]
pub fn align_combined(
    centerline: PyCenterline,
    geom_pair: PyGeometryPair,
    aortic_ref_pt: (f64, f64, f64),
    upper_ref_pt: (f64, f64, f64),
    lower_ref_pt: (f64, f64, f64),
    points: Vec<(f64, f64, f64)>,
    angle_step_deg: f64,
    angle_range_deg: f64, // e.g., 15° in radians
    index_range: usize,   // e.g., 2
    write: bool,
    watertight: bool,
    interpolation_steps: usize,
    output_dir: &str,
    contour_types: Vec<PyContourType>,
    case_name: &str,
) -> PyResult<(PyGeometryPair, PyCenterline)> {
    let rust_contour_types: Vec<crate::intravascular::io::geometry::ContourType> =
        contour_types.iter().map(|ct| ct.into()).collect();
    let cl_rs = centerline.to_rust_centerline();
    let geom_pair_rs = geom_pair.to_rust_geometry_pair();
    let angle_step = angle_step_deg.to_radians();
    let angle_range = angle_range_deg.to_radians();

    let (geom_pair, cl) = align_combined_rs(
        cl_rs,
        geom_pair_rs,
        aortic_ref_pt,
        upper_ref_pt,
        lower_ref_pt,
        points.as_ref(),
        angle_step,
        angle_range,
        index_range,
        write,
        watertight,
        interpolation_steps,
        output_dir,
        rust_contour_types,
        case_name,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let aligned_py: PyGeometryPair = geom_pair.into();
    let resampled_cl: PyCenterline = cl.into();

    Ok((aligned_py, resampled_cl))
}
