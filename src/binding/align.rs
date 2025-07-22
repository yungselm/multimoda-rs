use pyo3::prelude::*;

use crate::{
    binding::classes::{PyCenterline, PyGeometryPair},
    centerline_align::align_three_point_rs,
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
        aortic_ref_pt,
        upper_ref_pt,
        lower_ref_pt,
        angle_step=0.01745329,
        write=false,
        interpolation_steps=28usize,
        output_dir="output/aligned",
    )
)]
pub fn align_three_point(
    centerline: PyCenterline,
    geometry_pair: PyGeometryPair,
    aortic_ref_pt: (f64, f64, f64),
    upper_ref_pt: (f64, f64, f64),
    lower_ref_pt: (f64, f64, f64),
    angle_step: f64,
    write: bool,
    interpolation_steps: usize,
    output_dir: &str,
) -> PyResult<PyGeometryPair> {
    let cl_rs = centerline.to_rust_centerline();
    let geom_pair_rs = geometry_pair.to_rust_geometry_pair();

    let geom_pair = align_three_point_rs(
        cl_rs,
        geom_pair_rs,
        aortic_ref_pt,
        upper_ref_pt,
        lower_ref_pt,
        angle_step,
        write,
        interpolation_steps,
        output_dir,
    );

    let aligned_py: PyGeometryPair = geom_pair.into();

    Ok(aligned_py)
}
