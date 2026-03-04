use crate::intravascular::{
    binding::classes::{PyCenterline, PyContourType, PyGeometryPair},
    centerline_align::{align_combined_rs, align_manual_rs, align_three_point_rs},
};
use pyo3::prelude::*;

/// Align diastolic and systolic meshes to the centerline using three reference points.
///
/// Creates centerline-aligned meshes for a diastolic/systolic geometry pair
/// based on three anatomical reference points (aorta, upper section, lower
/// section).  Only works for elliptic vessels such as coronary artery
/// anomalies.
///
/// Parameters
/// ----------
/// centerline : PyCenterline
///     Centerline of the vessel.
/// geometry_pair : PyGeometryPair
///     Diastolic/systolic geometry pair to align.
/// aortic_ref_pt : tuple of float
///     ``(x, y, z)`` reference point at the aortic ostium.
/// upper_ref_pt : tuple of float
///     ``(x, y, z)`` upper section reference point.
/// lower_ref_pt : tuple of float
///     ``(x, y, z)`` lower section reference point.
/// angle_step_deg : float, optional
///     Step size in degrees for the rotation search.  Default is ``1.0``.
/// write : bool, optional
///     Whether to write the aligned meshes to OBJ files.  Default is ``False``.
/// watertight : bool, optional
///     Whether to write a watertight or shell mesh to OBJ.  Default is ``True``.
/// interpolation_steps : int, optional
///     Number of interpolation steps between phases.  Default is ``28``.
/// output_dir : str, optional
///     Output directory for aligned meshes.  Default is ``"output/aligned"``.
/// case_name : str, optional
///     Case name used as a filename prefix.  Default is ``"None"``.
///
/// Returns
/// -------
/// geom_pair : PyGeometryPair
///     Aligned geometry pair.
/// centerline : PyCenterline
///     Resampled centerline.
///
/// Examples
/// --------
/// >>> import multimodars as mm
/// >>> dia, sys = mm.align_three_point(
/// ...     centerline,
/// ...     geometry_pair,
/// ...     (12.2605, -201.3643, 1751.0554),
/// ...     (11.7567, -202.1920, 1754.7975),
/// ...     (15.6605, -202.1920, 1749.9655),
/// ... )
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

/// Align diastolic and systolic meshes to the centerline using a manual rotation angle.
///
/// Creates centerline-aligned meshes for a diastolic/systolic geometry pair
/// using an explicit rotation angle and a single reference point on the
/// centerline.  Only works for elliptic vessels such as coronary artery
/// anomalies.
///
/// Parameters
/// ----------
/// centerline : PyCenterline
///     Centerline of the vessel.
/// geometry_pair : PyGeometryPair
///     Diastolic/systolic geometry pair to align.
/// rotation_angle : float
///     Rotation angle in radians to apply.
/// ref_point : tuple of float
///     ``(x, y, z)`` reference point on the centerline.
/// write : bool, optional
///     Whether to write the aligned meshes to OBJ files.  Default is ``False``.
/// watertight : bool, optional
///     Whether to write a watertight or shell mesh to OBJ.  Default is ``True``.
/// interpolation_steps : int, optional
///     Number of interpolation steps between phases.  Default is ``28``.
/// output_dir : str, optional
///     Output directory for aligned meshes.  Default is ``"output/aligned"``.
/// case_name : str, optional
///     Case name used as a filename prefix.  Default is ``"None"``.
///
/// Returns
/// -------
/// geom_pair : PyGeometryPair
///     Aligned geometry pair.
/// centerline : PyCenterline
///     Resampled centerline.
///
/// Examples
/// --------
/// >>> import multimodars as mm
/// >>> dia, sys = mm.centerline_align(
/// ...     "path/to/centerline.csv",
/// ...     (1.0, 2.0, 3.0),
/// ...     (4.0, 5.0, 6.0),
/// ...     (7.0, 8.0, 9.0),
/// ...     "rest"
/// ... )
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

/// Align meshes using three reference points combined with Hausdorff distance refinement.
///
/// Creates centerline-aligned meshes for a diastolic/systolic geometry pair
/// using three anatomical reference points for an initial orientation and a
/// set of additional points for Hausdorff distance-based fine-tuning of the
/// rotation.
///
/// Parameters
/// ----------
/// centerline : PyCenterline
///     Centerline of the vessel.
/// geom_pair : PyGeometryPair
///     Diastolic/systolic geometry pair to align.
/// aortic_ref_pt : tuple of float
///     ``(x, y, z)`` reference point at the aortic ostium.
/// upper_ref_pt : tuple of float
///     ``(x, y, z)`` upper section reference point.
/// lower_ref_pt : tuple of float
///     ``(x, y, z)`` lower section reference point.
/// points : list of tuple of float
///     Point cloud used for Hausdorff distance calculation during rotation
///     refinement.
/// angle_step_deg : float, optional
///     Step size in degrees for the rotation search.  Default is ``1.0``.
/// angle_range_deg : float, optional
///     Total rotation search range in degrees.  Default is ``15.0``.
/// index_range : int, optional
///     Number of centerline indices considered around the reference.
///     Default is ``2``.
/// write : bool, optional
///     Whether to write the aligned meshes to OBJ files.  Default is ``False``.
/// watertight : bool, optional
///     Whether to write a watertight or shell mesh to OBJ.  Default is ``True``.
/// interpolation_steps : int, optional
///     Number of interpolation steps between phases.  Default is ``28``.
/// output_dir : str, optional
///     Output directory for aligned meshes.  Default is ``"output/aligned"``.
/// case_name : str, optional
///     Case name used as a filename prefix.  Default is ``"None"``.
///
/// Returns
/// -------
/// geom_pair : PyGeometryPair
///     Aligned geometry pair.
/// centerline : PyCenterline
///     Resampled centerline.
///
/// Examples
/// --------
/// >>> import multimodars as mm
/// >>> dia, sys = mm.align_three_point(
/// ...     centerline,
/// ...     geometry_pair,
/// ...     (12.2605, -201.3643, 1751.0554),
/// ...     [(11.7567, -202.1920, 1754.7975), ... ],
/// ... )
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
