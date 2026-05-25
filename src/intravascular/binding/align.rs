use crate::intravascular::{
    binding::classes::{PyCenterline, PyContourType, PyGeometry, PyGeometryPair},
    centerline_align::{align_combined_rs, align_manual_rs, align_three_point_rs},
};
use pyo3::prelude::*;

/// Align a geometry (or geometry pair) to the centerline using three reference points.
///
/// Creates centerline-aligned meshes based on three anatomical reference points
/// (aorta, upper section, lower section).  Only works for elliptic vessels such
/// as coronary artery anomalies.
///
/// Parameters
/// ----------
/// centerline : PyCenterline
///     Centerline of the vessel.
/// geometry : PyGeometry or PyGeometryPair
///     Single geometry or diastolic/systolic geometry pair to align.
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
///     Number of interpolation steps between phases.  Only used when *geometry*
///     is a ``PyGeometryPair``.  Default is ``0``.
/// output_dir : str, optional
///     Output directory for aligned meshes.  Default is ``"output/aligned"``.
/// case_name : str, optional
///     Case name used as a filename prefix.  Default is ``"None"``.
/// align_wall_anomalous : bool, optional
///     When ``True``, rotate the Wall contour in every frame (from frame 2 onward)
///     so its aortic straight portion aligns to the plane defined by frames 0 and 1.
///     Only meaningful for anomalous vessels.  Default is ``False``.
///
/// Returns
/// -------
/// geometry : PyGeometry or PyGeometryPair
///     Aligned geometry, matching the type of the input.
/// centerline : PyCenterline
///     Resampled centerline.
///
/// Examples
/// --------
/// >>> import multimodars as mm
/// >>> result, cl = mm.align_three_point(
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
        geometry,
        aortic_ref_pt,
        upper_ref_pt,
        lower_ref_pt,
        angle_step_deg=1.0,
        write=false,
        watertight=true,
        interpolation_steps=0usize,
        output_dir="output/aligned",
        contour_types=vec![PyContourType::Lumen, PyContourType::Catheter, PyContourType::Wall],
        case_name="None",
        align_wall_anomalous=false,
    )
)]
pub fn align_three_point(
    py: Python<'_>,
    centerline: PyCenterline,
    geometry: Bound<'_, PyAny>,
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
    align_wall_anomalous: bool,
) -> PyResult<(Py<PyAny>, PyCenterline)> {
    let rust_contour_types: Vec<crate::intravascular::io::geometry::ContourType> =
        contour_types.iter().map(|ct| ct.into()).collect();
    let cl_rs = centerline.to_rust_centerline();
    let angle_step = angle_step_deg.to_radians();

    if let Ok(geom_pair) = geometry.extract::<PyGeometryPair>() {
        let (result_rs, cl) = align_three_point_rs(
            cl_rs,
            geom_pair.to_rust_geometry_pair(),
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
            align_wall_anomalous,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let py_result: PyGeometryPair = result_rs.into();
        Ok((Py::new(py, py_result)?.into_any(), cl.into()))
    } else if let Ok(geom) = geometry.extract::<PyGeometry>() {
        let (result_rs, cl) = align_three_point_rs(
            cl_rs,
            geom.to_rust_geometry()?,
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
            align_wall_anomalous,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let py_result: PyGeometry = result_rs.into();
        Ok((Py::new(py, py_result)?.into_any(), cl.into()))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "geometry must be a PyGeometry or PyGeometryPair",
        ))
    }
}

/// Align a geometry (or geometry pair) to the centerline using a manual rotation angle.
///
/// Creates centerline-aligned meshes using an explicit rotation angle and a
/// single reference point on the centerline.  Only works for elliptic vessels
/// such as coronary artery anomalies.
///
/// Parameters
/// ----------
/// centerline : PyCenterline
///     Centerline of the vessel.
/// geometry : PyGeometry or PyGeometryPair
///     Single geometry or diastolic/systolic geometry pair to align.
/// rotation_angle : float
///     Rotation angle in radians to apply.
/// ref_point : tuple of float
///     ``(x, y, z)`` reference point on the centerline.
/// write : bool, optional
///     Whether to write the aligned meshes to OBJ files.  Default is ``False``.
/// watertight : bool, optional
///     Whether to write a watertight or shell mesh to OBJ.  Default is ``True``.
/// interpolation_steps : int, optional
///     Number of interpolation steps between phases.  Only used when *geometry*
///     is a ``PyGeometryPair``.  Default is ``0``.
/// output_dir : str, optional
///     Output directory for aligned meshes.  Default is ``"output/aligned"``.
/// case_name : str, optional
///     Case name used as a filename prefix.  Default is ``"None"``.
/// align_wall_anomalous : bool, optional
///     When ``True``, rotate the Wall contour in every frame (from frame 2 onward)
///     so its aortic straight portion aligns to the plane defined by frames 0 and 1.
///     Only meaningful for anomalous vessels.  Default is ``False``.
///
/// Returns
/// -------
/// geometry : PyGeometry or PyGeometryPair
///     Aligned geometry, matching the type of the input.
/// centerline : PyCenterline
///     Resampled centerline.
///
/// Examples
/// --------
/// >>> import multimodars as mm
/// >>> result, cl = mm.align_manual(
/// ...     centerline, geometry_pair, rotation_angle=1.57, ref_point=(1.0, 2.0, 3.0)
/// ... )
#[pyfunction]
#[pyo3(
    signature = (
        centerline,
        geometry,
        rotation_angle,
        ref_point,
        write=false,
        watertight=true,
        interpolation_steps=0usize,
        output_dir="output/aligned",
        contour_types=vec![PyContourType::Lumen, PyContourType::Catheter, PyContourType::Wall],
        case_name="None",
        align_wall_anomalous=false,
    )
)]
pub fn align_manual(
    py: Python<'_>,
    centerline: PyCenterline,
    geometry: Bound<'_, PyAny>,
    rotation_angle: f64,
    ref_point: (f64, f64, f64),
    write: bool,
    watertight: bool,
    interpolation_steps: usize,
    output_dir: &str,
    contour_types: Vec<PyContourType>,
    case_name: &str,
    align_wall_anomalous: bool,
) -> PyResult<(Py<PyAny>, PyCenterline)> {
    let rust_contour_types: Vec<crate::intravascular::io::geometry::ContourType> =
        contour_types.iter().map(|ct| ct.into()).collect();
    let cl_rs = centerline.to_rust_centerline();

    if let Ok(geom_pair) = geometry.extract::<PyGeometryPair>() {
        let (result_rs, cl) = align_manual_rs(
            cl_rs,
            geom_pair.to_rust_geometry_pair(),
            rotation_angle,
            ref_point,
            write,
            watertight,
            interpolation_steps,
            output_dir,
            rust_contour_types,
            case_name,
            align_wall_anomalous,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let py_result: PyGeometryPair = result_rs.into();
        Ok((Py::new(py, py_result)?.into_any(), cl.into()))
    } else if let Ok(geom) = geometry.extract::<PyGeometry>() {
        let (result_rs, cl) = align_manual_rs(
            cl_rs,
            geom.to_rust_geometry()?,
            rotation_angle,
            ref_point,
            write,
            watertight,
            interpolation_steps,
            output_dir,
            rust_contour_types,
            case_name,
            align_wall_anomalous,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let py_result: PyGeometry = result_rs.into();
        Ok((Py::new(py, py_result)?.into_any(), cl.into()))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "geometry must be a PyGeometry or PyGeometryPair",
        ))
    }
}

/// Align a geometry (or geometry pair) using three reference points and Hausdorff refinement.
///
/// Creates centerline-aligned meshes using three anatomical reference points
/// for an initial orientation and a set of additional points for
/// Hausdorff distance-based fine-tuning of the rotation.
///
/// Parameters
/// ----------
/// centerline : PyCenterline
///     Centerline of the vessel.
/// geometry : PyGeometry or PyGeometryPair
///     Single geometry or diastolic/systolic geometry pair to align.
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
///     Number of interpolation steps between phases.  Only used when *geometry*
///     is a ``PyGeometryPair``.  Default is ``0``.
/// output_dir : str, optional
///     Output directory for aligned meshes.  Default is ``"output/aligned"``.
/// case_name : str, optional
///     Case name used as a filename prefix.  Default is ``"None"``.
/// align_wall_anomalous : bool, optional
///     When ``True``, rotate the Wall contour in every frame (from frame 2 onward)
///     so its aortic straight portion aligns to the plane defined by frames 0 and 1.
///     Only meaningful for anomalous vessels.  Default is ``False``.
///
/// Returns
/// -------
/// geometry : PyGeometry or PyGeometryPair
///     Aligned geometry, matching the type of the input.
/// centerline : PyCenterline
///     Resampled centerline.
///
/// Examples
/// --------
/// >>> import multimodars as mm
/// >>> result, cl = mm.align_combined(
/// ...     centerline,
/// ...     geometry_pair,
/// ...     (12.2605, -201.3643, 1751.0554),
/// ...     (11.7567, -202.1920, 1754.7975),
/// ...     (15.6605, -202.1920, 1749.9655),
/// ...     point_cloud,
/// ... )
#[pyfunction]
#[pyo3(
    signature = (
        centerline,
        geometry,
        aortic_ref_pt,
        upper_ref_pt,
        lower_ref_pt,
        points,
        angle_step_deg=1.0,
        angle_range_deg=15.0,
        index_range=2,
        write=false,
        watertight=true,
        interpolation_steps=0usize,
        output_dir="output/aligned",
        contour_types=vec![PyContourType::Lumen, PyContourType::Catheter, PyContourType::Wall],
        case_name="None",
        align_wall_anomalous=false,
    )
)]
pub fn align_combined(
    py: Python<'_>,
    centerline: PyCenterline,
    geometry: Bound<'_, PyAny>,
    aortic_ref_pt: (f64, f64, f64),
    upper_ref_pt: (f64, f64, f64),
    lower_ref_pt: (f64, f64, f64),
    points: Vec<(f64, f64, f64)>,
    angle_step_deg: f64,
    angle_range_deg: f64,
    index_range: usize,
    write: bool,
    watertight: bool,
    interpolation_steps: usize,
    output_dir: &str,
    contour_types: Vec<PyContourType>,
    case_name: &str,
    align_wall_anomalous: bool,
) -> PyResult<(Py<PyAny>, PyCenterline)> {
    let rust_contour_types: Vec<crate::intravascular::io::geometry::ContourType> =
        contour_types.iter().map(|ct| ct.into()).collect();
    let cl_rs = centerline.to_rust_centerline();
    let angle_step = angle_step_deg.to_radians();
    let angle_range = angle_range_deg.to_radians();

    if let Ok(geom_pair) = geometry.extract::<PyGeometryPair>() {
        let (result_rs, cl) = align_combined_rs(
            cl_rs,
            geom_pair.to_rust_geometry_pair(),
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
            align_wall_anomalous,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let py_result: PyGeometryPair = result_rs.into();
        Ok((Py::new(py, py_result)?.into_any(), cl.into()))
    } else if let Ok(geom) = geometry.extract::<PyGeometry>() {
        let (result_rs, cl) = align_combined_rs(
            cl_rs,
            geom.to_rust_geometry()?,
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
            align_wall_anomalous,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let py_result: PyGeometry = result_rs.into();
        Ok((Py::new(py, py_result)?.into_any(), cl.into()))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "geometry must be a PyGeometry or PyGeometryPair",
        ))
    }
}
