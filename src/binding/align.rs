use pyo3::prelude::*;

use crate::{
    binding::classes::{PyCenterline, PyGeometryPair},
    centerline_align::{align_manual_rs, align_three_point_rs},
};

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
///     state: Physiological state ("rest" or "stress")
///     input_dir: Input directory for raw geometries
///     output_dir: Output directory for aligned meshes
///     interpolation_steps: Number of interpolation steps
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
        interpolation_steps=28usize,
        output_dir="output/aligned",
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
    interpolation_steps: usize,
    output_dir: &str,
    case_name: &str,
) -> PyResult<(PyGeometryPair, PyCenterline)> {
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
        interpolation_steps,
        output_dir,
        case_name,
    );

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
///     aortic_ref_pt: Reference point for aortic position
///     upper_ref_pt: Upper reference point
///     lower_ref_pt: Lower reference point
///     state: Physiological state ("rest" or "stress")
///     input_dir: Input directory for raw geometries
///     output_dir: Output directory for aligned meshes
///     interpolation_steps: Number of interpolation steps
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
        start_point,
        write=false,
        interpolation_steps=28usize,
        output_dir="output/aligned",
        case_name="None",
    )
)]
pub fn align_manual(
    centerline: PyCenterline,
    geometry_pair: PyGeometryPair,
    rotation_angle: f64,
    start_point: usize,
    write: bool,
    interpolation_steps: usize,
    output_dir: &str,
    case_name: &str,
) -> PyResult<(PyGeometryPair, PyCenterline)> {
    let cl_rs = centerline.to_rust_centerline();
    let geom_pair_rs = geometry_pair.to_rust_geometry_pair();

    let (geom_pair, cl) = align_manual_rs(
        cl_rs,
        geom_pair_rs,
        rotation_angle,
        start_point,
        write,
        interpolation_steps,
        output_dir,
        case_name,
    );

    let aligned_py: PyGeometryPair = geom_pair.into();
    let resampled_cl: PyCenterline = cl.into();

    Ok((aligned_py, resampled_cl))
}
