import numpy as np
from typing import Union, Tuple, List, Optional, Dict
import multimodars as mm


def to_array(generic) -> Union[np.ndarray, dict, Tuple[dict, dict]]:
    """
    Convert various multimodars Py* objects into numpy array(s) or dictionaries of arrays.

    Parameters
    ----------
    generic : PyContour, PyCenterline, PyGeometry, PyGeometryPair, PyFrame, or PyInputData
        The object to be converted to numpy representation.

    Returns
    -------
    np.ndarray
        For PyContour or PyCenterline:
        A 2D array of shape (N, 4), where each row is (frame_index, x, y, z).

    dict[str, np.ndarray]
        For PyGeometry:
        A dictionary with keys for each contour type and "reference",
        each containing a 2D array of shape (M, 4), where M is the number of points in that layer.

    Tuple[dict[str, np.ndarray], dict[str, np.ndarray]]
        For PyGeometryPair:
        A tuple of two dictionaries (one for geom_a, one for geom_b), each in the same format
        as returned for a single PyGeometry.

    dict[str, Union[np.ndarray, List[str], bool]]
        For PyInputData:
        A dictionary containing arrays for each contour type and metadata.

    Raises
    ------
    TypeError
        If the input type is not one of the supported multimodars types.
    """
    from . import (
        PyContour,
        PyCenterline,
        PyGeometry,
        PyGeometryPair,
        PyFrame,
        PyInputData,
    )

    if isinstance(generic, PyContour):
        pts = [(p.frame_index, p.x, p.y, p.z) for p in generic.points]
        return np.array(pts, dtype=float)

    if isinstance(generic, PyCenterline):
        pts = [
            (
                p.contour_point.frame_index,
                p.contour_point.x,
                p.contour_point.y,
                p.contour_point.z,
            )
            for p in generic.points
        ]
        return np.array(pts, dtype=float)

    if isinstance(generic, PyFrame):
        return _frame_to_numpy(generic)

    if isinstance(generic, PyGeometry):
        return _geometry_to_numpy(generic)

    if isinstance(generic, PyGeometryPair):
        geom_a_dict = _geometry_to_numpy(generic.geom_a)
        geom_b_dict = _geometry_to_numpy(generic.geom_b)
        return geom_a_dict, geom_b_dict

    if isinstance(generic, PyInputData):
        return _input_data_to_numpy(generic)

    raise TypeError(f"Unsupported type for to_array: {type(generic)}")


def _frame_to_numpy(frame) -> dict[str, np.ndarray]:
    """Convert PyFrame to dictionary of numpy arrays."""
    result = {}

    # Add lumen
    lumen_pts = [(p.frame_index, p.x, p.y, p.z) for p in frame.lumen.points]
    result["lumen"] = (
        np.array(lumen_pts, dtype=float) if lumen_pts else np.zeros((0, 4), dtype=float)
    )

    # Add extras
    for contour_type, contour in frame.extras.items():
        pts = [(p.frame_index, p.x, p.y, p.z) for p in contour.points]
        result[contour_type.lower()] = (
            np.array(pts, dtype=float) if pts else np.zeros((0, 4), dtype=float)
        )

    # Add reference point
    if frame.reference_point:
        ref = frame.reference_point
        result["reference"] = np.array(
            [[ref.frame_index, ref.x, ref.y, ref.z]], dtype=float
        )
    else:
        result["reference"] = np.zeros((0, 4), dtype=float)

    return result


def _geometry_to_numpy(geom) -> dict[str, np.ndarray]:
    """Convert PyGeometry to dictionary of numpy arrays."""
    result = {
        "lumen": np.zeros((0, 4), dtype=float),
        "eem": np.zeros((0, 4), dtype=float),
        "calcification": np.zeros((0, 4), dtype=float),
        "sidebranch": np.zeros((0, 4), dtype=float),
        "catheter": np.zeros((0, 4), dtype=float),
        "wall": np.zeros((0, 4), dtype=float),
        "reference": np.zeros((0, 4), dtype=float),
    }

    # Collect all points from all frames
    for frame in geom.frames:
        frame_data = _frame_to_numpy(frame)
        for key in result:
            if key in frame_data and len(frame_data[key]) > 0:
                if len(result[key]) == 0:
                    result[key] = frame_data[key]
                else:
                    result[key] = np.vstack([result[key], frame_data[key]])

    return result


def _input_data_to_numpy(input_data) -> dict[str, Union[np.ndarray, List[str], bool]]:
    """Convert PyInputData to dictionary of numpy arrays and metadata."""
    result = {
        "lumen": np.zeros((0, 4), dtype=float),
        "eem": np.zeros((0, 4), dtype=float),
        "calcification": np.zeros((0, 4), dtype=float),
        "sidebranch": np.zeros((0, 4), dtype=float),
        "reference": np.zeros((0, 4), dtype=float),
        "diastole": input_data.diastole,
        "label": input_data.label,
    }

    # Process lumen (required)
    if input_data.lumen:
        lumen_pts = []
        for contour in input_data.lumen:
            lumen_pts.extend([(p.frame_index, p.x, p.y, p.z) for p in contour.points])
        if lumen_pts:
            result["lumen"] = np.array(lumen_pts, dtype=float)

    # Process optional contour types
    for contour_type in ["eem", "calcification", "sidebranch"]:
        contours = getattr(input_data, contour_type)
        if contours:
            pts = []
            for contour in contours:
                pts.extend([(p.frame_index, p.x, p.y, p.z) for p in contour.points])
            if pts:
                result[contour_type] = np.array(pts, dtype=float)

    # Process reference point
    ref = input_data.ref_point
    result["reference"] = np.array(
        [[ref.frame_index, ref.x, ref.y, ref.z]], dtype=float
    )

    # Process records if available
    if input_data.record:
        record_data = []
        for record in input_data.record:
            row = [record.frame, record.phase]
            row.append(
                record.measurement_1 if record.measurement_1 is not None else np.nan
            )
            row.append(
                record.measurement_2 if record.measurement_2 is not None else np.nan
            )
            record_data.append(row)
        result["records"] = np.array(record_data, dtype=object)

    return result


def numpy_to_inputdata(
    lumen_arr: np.ndarray,
    record: Optional[np.ndarray],
    ref_point: np.ndarray,
    diastole: bool,
    eem_arr: Optional[np.ndarray] = None,
    calcification: Optional[np.ndarray] = None,
    sidebranch: Optional[np.ndarray] = None,
    label: str = "",
) -> mm.PyInputData:
    """
    Build a PyInputData from numpy arrays, grouping by frame_index into frames.

    Each row in the ``*_arr`` is [frame_index, x, y, z].

    Returns a PyInputData containing:
      - lumen: list of PyContour (one per frame found in lumen_arr)
      - eem: optional list of PyContour (only contours that exist)
      - calcification: optional list of PyContour (only contours that exist)
      - sidebranch: optional list of PyContour (only contours that exist)
      - record: optional list of PyRecord (converted from record array)
      - ref_point: PyContourPoint (first reference row, or default)
      - diastole: boolean
      - label: str
    """
    import numpy as np
    from . import PyContour, PyContourPoint, PyRecord, PyInputData

    def _to_numeric_array(arr: Optional[np.ndarray], name: str) -> np.ndarray:
        if arr is None:
            return np.zeros((0, 4), dtype=float)
        # Handle structured arrays with named fields
        if arr.ndim == 1 and getattr(arr, "dtype", None).names:
            try:
                arr = np.vstack([arr[n] for n in arr.dtype.names]).T
            except Exception:
                raise ValueError(f"Could not convert structured array for {name}")
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            # Single row -> make it 2D
            arr = arr.reshape(1, -1)
        return arr

    def build_contour_from_array(arr: np.ndarray, frame_id: int, contour_type: str):
        """Return PyContour for given frame_id or None if not present."""
        if arr.size == 0:
            return None
        mask = arr[:, 0].astype(int) == int(frame_id)
        pts_arr = arr[mask]
        if pts_arr.shape[0] == 0:
            return None

        pts = [
            PyContourPoint(
                frame_index=int(fr),
                point_index=i,
                x=float(x),
                y=float(y),
                z=float(z),
                aortic=False,
            )
            for i, (fr, x, y, z) in enumerate(pts_arr)
        ]

        centroid = (
            float(np.mean(pts_arr[:, 1])),
            float(np.mean(pts_arr[:, 2])),
            float(np.mean(pts_arr[:, 3])),
        )

        return PyContour(
            id=int(frame_id),
            original_frame=int(frame_id),
            points=pts,
            centroid=centroid,
            aortic_thickness=None,
            pulmonary_thickness=None,
            kind=contour_type,
        )

    def _records_from_array(arr: Optional[np.ndarray]):
        if arr is None:
            return None
        # If structured with fields, try to coerce to (N,4) or (N,3)
        if arr.ndim == 1 and getattr(arr, "dtype", None).names:
            # Try to create a 2D array with numeric fields where appropriate
            try:
                arr = np.vstack([arr[n] for n in arr.dtype.names]).T
            except Exception:
                # fallback to treating each element as a row-like object
                arr = np.asarray(arr)
        arr = np.asarray(arr)
        if arr.size == 0:
            return None
        # If 1D single-row, reshape
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        recs = []
        for row in arr:
            # Attempt to extract fields robustly:
            # Expecting [frame, phase, measurement_1, measurement_2] or at least [frame, phase]
            frame = int(row[0])
            phase_val = row[1] if row.shape[0] > 1 else ""
            # Normalize phase to string
            if isinstance(phase_val, (bytes, bytearray)):
                try:
                    phase = phase_val.decode("utf-8")
                except Exception:
                    phase = str(phase_val)
            else:
                # numeric -> map 0 -> "D", 1 -> "S", otherwise string-ify
                if np.issubdtype(type(phase_val), np.number):
                    phase = "D" if int(phase_val) == 0 else "S"
                else:
                    phase = str(phase_val)

            def _to_optional_float(v):
                try:
                    fv = float(v)
                    if np.isnan(fv):
                        return None
                    return fv
                except Exception:
                    return None

            m1 = _to_optional_float(row[2]) if row.shape[0] > 2 else None
            m2 = _to_optional_float(row[3]) if row.shape[0] > 3 else None

            recs.append(
                PyRecord(frame=frame, phase=phase, measurement_1=m1, measurement_2=m2)
            )

        return recs if len(recs) > 0 else None

    # Convert arrays
    lumen_arr = _to_numeric_array(lumen_arr, "lumen_arr")
    eem_arr = _to_numeric_array(eem_arr, "eem_arr")
    calcification_arr = _to_numeric_array(calcification, "calcification")
    sidebranch_arr = _to_numeric_array(sidebranch, "sidebranch")

    # Reference point: prefer provided ref_point, otherwise a default
    global_ref = None
    if ref_point is not None:
        try:
            ref_arr = np.asarray(ref_point)
            if ref_arr.ndim == 1:
                fr, x, y, z = ref_arr[:4]
            else:
                fr, x, y, z = ref_arr[0, :4]
            global_ref = PyContourPoint(
                frame_index=int(fr),
                point_index=0,
                x=float(x),
                y=float(y),
                z=float(z),
                aortic=False,
            )
        except Exception:
            global_ref = None

    if global_ref is None:
        # default fallback (required by PyInputData ctor)
        global_ref = PyContourPoint(
            frame_index=0, point_index=0, x=0.0, y=0.0, z=0.0, aortic=False
        )

    # Build lists of contours
    if lumen_arr.size == 0:
        raise ValueError("lumen_arr cannot be empty")

    all_lumen_frames = sorted(set(lumen_arr[:, 0].astype(int)))

    lumen_list = []
    eem_list = []
    calc_list = []
    sidebranch_list = []

    for frame_id in all_lumen_frames:
        lumen_contour = build_contour_from_array(lumen_arr, frame_id, "Lumen")
        if lumen_contour is None:
            # If no lumen for this frame, skip (shouldn't usually happen since frames come from lumen)
            continue
        lumen_list.append(lumen_contour)

        eem_contour = build_contour_from_array(eem_arr, frame_id, "Eem")
        if eem_contour is not None:
            eem_list.append(eem_contour)

        calc_contour = build_contour_from_array(
            calcification_arr, frame_id, "Calcification"
        )
        if calc_contour is not None:
            calc_list.append(calc_contour)

        sb_contour = build_contour_from_array(sidebranch_arr, frame_id, "Sidebranch")
        if sb_contour is not None:
            sidebranch_list.append(sb_contour)

    # Convert records (if any)
    record_list = _records_from_array(record)

    # Convert empty lists to None for optional fields like eem/calcification/sidebranch/record
    eem_final = eem_list if len(eem_list) > 0 else None
    calc_final = calc_list if len(calc_list) > 0 else None
    sidebranch_final = sidebranch_list if len(sidebranch_list) > 0 else None

    return PyInputData(
        lumen=lumen_list,
        eem=eem_final,
        calcification=calc_final,
        sidebranch=sidebranch_final,
        record=record_list,
        ref_point=global_ref,
        diastole=bool(diastole),
        label=label or "",
    )


def numpy_to_geometry(
    lumen_arr: np.ndarray,
    eem_arr: Optional[np.ndarray] = None,
    catheter_arr: Optional[np.ndarray] = None,
    wall_arr: Optional[np.ndarray] = None,
    reference_arr: Optional[np.ndarray] = None,
    label: str = "",
) -> mm.PyGeometry:
    """
    Build a PyGeometry from numpy arrays, grouping by frame_index into frames.

    Each row in the ``*_arr`` is [frame_index, x, y, z].

    Returns a PyGeometry containing frames with:
      - lumen: PyContour for each frame
      - extras: dictionary with optional EEM, Catheter, Wall contours
      - reference_point: PyContourPoint from reference_arr (if provided)

    Parameters
    ----------
    lumen_arr : np.ndarray
        (N,4) array of lumen points [frame_index, x, y, z]
    eem_arr : np.ndarray, optional
        (M,4) array of EEM points [frame_index, x, y, z]
    catheter_arr : np.ndarray, optional
        (K,4) array of catheter points [frame_index, x, y, z]
    wall_arr : np.ndarray, optional
        (L,4) array of wall points [frame_index, x, y, z]
    reference_arr : np.ndarray, optional
        (1,4) or (4,) array [frame_index, x, y, z] for reference point
    label : str
        Label for the geometry

    Returns
    -------
    PyGeometry
    """
    from . import PyContour, PyContourPoint, PyFrame, PyGeometry

    def _to_numeric_array(arr: Optional[np.ndarray], layer_name: str) -> np.ndarray:
        if arr is None:
            return np.zeros((0, 4), dtype=float)
        # Handle structured arrays
        if arr.ndim == 1 and arr.dtype.names:
            try:
                arr = np.vstack([arr[name] for name in arr.dtype.names]).T
            except Exception:
                raise ValueError(f"Could not convert structured array for {layer_name}")
        arr = np.asarray(arr, dtype=float)
        return arr

    def build_contour_from_array(
        arr: np.ndarray, frame_id: int, contour_type: str
    ) -> PyContour:
        """Build a PyContour from array points for a specific frame."""
        if arr.size == 0:
            return None

        mask = arr[:, 0].astype(int) == frame_id
        pts_arr = arr[mask]

        if len(pts_arr) == 0:
            return None

        pts = [
            PyContourPoint(
                frame_index=int(fr),
                point_index=i,
                x=float(x),
                y=float(y),
                z=float(z),
                aortic=False,
            )
            for i, (fr, x, y, z) in enumerate(pts_arr)
        ]

        # Compute centroid
        centroid = (
            np.mean(pts_arr[:, 1]),
            np.mean(pts_arr[:, 2]),
            np.mean(pts_arr[:, 3]),
        )

        return PyContour(
            id=frame_id,
            original_frame=frame_id,
            points=pts,
            centroid=centroid,
            aortic_thickness=None,
            pulmonary_thickness=None,
            kind=contour_type,
        )

    # Convert arrays to numeric format
    lumen_arr = _to_numeric_array(lumen_arr, "lumen_arr")
    eem_arr = _to_numeric_array(eem_arr, "eem_arr")
    catheter_arr = _to_numeric_array(catheter_arr, "catheter_arr")
    wall_arr = _to_numeric_array(wall_arr, "wall_arr")
    reference_arr = _to_numeric_array(reference_arr, "reference_arr")

    if lumen_arr.size == 0:
        raise ValueError("lumen_arr cannot be empty")

    # Handle reference array - always take the first valid reference point
    global_reference = None
    if reference_arr.size > 0:
        # If 1D array, use it directly
        if reference_arr.ndim == 1:
            fr, x, y, z = reference_arr[:4]
        else:
            # If 2D array, take the first row
            fr, x, y, z = reference_arr[0, :4]

        global_reference = PyContourPoint(
            frame_index=int(fr),
            point_index=0,
            x=float(x),
            y=float(y),
            z=float(z),
            aortic=False,
        )

    # Get all unique frame indices from all arrays
    all_frames = set()
    for arr in [lumen_arr, eem_arr, catheter_arr, wall_arr]:
        if arr.size > 0:
            all_frames.update(arr[:, 0].astype(int))

    frames = []
    for frame_id in sorted(all_frames):
        # Build lumen contour (required)
        lumen_contour = build_contour_from_array(lumen_arr, frame_id, "Lumen")
        if not lumen_contour:
            continue

        # Build extras
        extras = {}
        eem_contour = build_contour_from_array(eem_arr, frame_id, "Eem")
        if eem_contour:
            extras["Eem"] = eem_contour

        catheter_contour = build_contour_from_array(catheter_arr, frame_id, "Catheter")
        if catheter_contour:
            extras["Catheter"] = catheter_contour

        wall_contour = build_contour_from_array(wall_arr, frame_id, "Wall")
        if wall_contour:
            extras["Wall"] = wall_contour

        # Use the global reference point for all frames
        frame_reference = global_reference

        frame = PyFrame(
            id=frame_id,
            centroid=lumen_contour.centroid,
            lumen=lumen_contour,
            extras=extras,
            reference_point=frame_reference,
        )
        frames.append(frame)

    return PyGeometry(frames=frames, label=label)


def numpy_to_centerline(
    arr: np.ndarray,
    aortic: bool = False,
) -> mm.PyCenterline:
    """
    Build a PyCenterline from a numpy array of shape (N,3),
    where each row is (x, y, z).

    This function will linearly interpolate NaN values along each coordinate
    axis. If an entire coordinate column is NaN, or result has fewer than 2
    points after processing, a ValueError is raised.

    Args:
        arr: np.ndarray of shape (N,3)
        aortic: whether to mark each point as aortic

    Returns:
        PyCenterline
    """
    import numpy as np
    from . import PyContourPoint, PyCenterline

    arr = np.asarray(arr, dtype=float)

    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("Input must be a (N,3) array")

    n = arr.shape[0]
    if n == 0:
        raise ValueError("Input array must contain at least one point")

    # If there are NaNs, try linear interpolation along the index axis.
    if np.isnan(arr).any():
        idx = np.arange(n)
        arr_interp = arr.copy()
        for col in range(3):
            col_vals = arr[:, col]
            valid_mask = ~np.isnan(col_vals)
            if valid_mask.sum() == 0:
                # Can't interpolate if whole column is missing
                raise ValueError(
                    f"All values are NaN for coordinate column {col}; cannot build centerline."
                )
            if valid_mask.sum() < n:
                # np.interp will fill leading and trailing NaNs by extrapolating the first/last valid values
                arr_interp[:, col] = np.interp(
                    idx, idx[valid_mask], col_vals[valid_mask]
                )
        arr = arr_interp

    # After interpolation, ensure we have at least two distinct points to form a centerline
    if arr.shape[0] < 2:
        raise ValueError(
            "Centerline must contain at least two points after cleaning/interpolation."
        )

    pts = []
    for i, (x, y, z) in enumerate(arr.tolist()):
        pts.append(
            PyContourPoint(
                frame_index=i,
                point_index=i,  # point_index can be meaningful; set to i instead of 0
                x=float(x),
                y=float(y),
                z=float(z),
                aortic=aortic,
            )
        )

    # Optionally validate that no NaNs remain
    for p in pts:
        if any(np.isnan((p.x, p.y, p.z))):
            raise ValueError("NaN coordinate found after interpolation — aborting.")

    return PyCenterline.from_contour_points(pts)


def array_to_pyinputdata(
    lumen=None,
    eem=None,
    calcification=None,
    sidebranch=None,
    records=None,
    reference=None,
    diastole: bool = True,
    label: str = "",
) -> mm.PyInputData:
    """
    Create a PyInputData from either Py* objects or NumPy arrays.

    Parameters mirror PyInputData fields. For layer arrays each row must be
    (frame_index, x, y, z). `records` accepts structured array or list/array
    of rows (frame, phase, m1, m2) or existing PyRecord instances. `reference`
    is (1,4) or (4,) row with frame,x,y,z.

    Returns
    -------
    PyInputData
    """
    from . import (
        PyContour,
        PyContourPoint,
        PyRecord,
        PyInputData,
    )

    def _to_numeric_array(arr, layer_name: str):
        if arr is None:
            return np.zeros((0, 4), dtype=float)
        if isinstance(arr, (list, tuple)):
            arr = np.asarray(arr, dtype=object)
        if isinstance(arr, np.ndarray) and arr.dtype.names:
            try:
                arr = np.vstack([arr[name] for name in arr.dtype.names]).T
            except Exception as e:
                raise ValueError(
                    f"Could not convert structured array for {layer_name}: {e}"
                )
        arr = np.asarray(arr)
        if arr.size == 0:
            return np.zeros((0, 4), dtype=float)
        if arr.ndim == 1:
            if arr.shape[0] == 4:
                arr = arr[np.newaxis, :]
            else:
                raise ValueError(
                    f"{layer_name} 1D array must have length 4, got {arr.shape}"
                )
        return arr.astype(object)

    def build_layer_from_array(arr, layer_name: str, kind: str):
        """Return list[PyContour] from a numeric array (frame,x,y,z) grouped by frame."""
        arr = _to_numeric_array(arr, layer_name)
        if arr.size == 0:
            return []

        if arr.ndim != 2 or arr.shape[1] < 4:
            raise ValueError(f"{layer_name} must be (N,4)-like, got shape {arr.shape}")

        frames = np.unique(arr[:, 0].astype(int))
        contours = []
        for frame in frames:
            mask = arr[:, 0].astype(int) == frame
            pts_arr = arr[mask]
            pts = []
            for i, row in enumerate(pts_arr):
                fr = int(row[0])
                x = float(row[1])
                y = float(row[2])
                z = float(row[3])
                pts.append(
                    PyContourPoint(
                        frame_index=fr, point_index=i, x=x, y=y, z=z, aortic=False
                    )
                )

            # Compute centroid
            if len(pts_arr) > 0:
                centroid = (
                    np.mean(pts_arr[:, 1]),
                    np.mean(pts_arr[:, 2]),
                    np.mean(pts_arr[:, 3]),
                )
            else:
                centroid = (0.0, 0.0, 0.0)

            contour = PyContour(
                id=frame,
                original_frame=frame,
                points=pts,
                centroid=centroid,
                aortic_thickness=None,
                pulmonary_thickness=None,
                kind=kind,
            )
            contours.append(contour)
        return contours

    def ensure_contours(maybe, kind: str):
        """Accept a list of PyContour already, or numpy arrays, or None."""
        if maybe is None:
            return []
        if (
            isinstance(maybe, list)
            and maybe
            and hasattr(maybe[0], "points")
            and hasattr(maybe[0], "id")
        ):
            return maybe
        return build_layer_from_array(maybe, "layer", kind)

    lumen_contours = ensure_contours(lumen, "Lumen")
    eem_contours = ensure_contours(eem, "Eem")
    calc_contours = ensure_contours(calcification, "Calcification")
    sidebranch_contours = ensure_contours(sidebranch, "Sidebranch")

    def parse_records(recs):
        if recs is None:
            return None
        if (
            isinstance(recs, (list, tuple))
            and recs
            and hasattr(recs[0], "frame")
            and hasattr(recs[0], "phase")
        ):
            return list(recs)
        if isinstance(recs, np.ndarray):
            if recs.dtype.names:
                names = recs.dtype.names

                def get_field(name, default=None):
                    for cand in [name, name.lower(), name.upper()]:
                        if cand in names:
                            return recs[cand]
                    return None

                frames = get_field("frame")
                phases = get_field("phase")
                m1 = get_field("measurement_1") or get_field("m1")
                m2 = get_field("measurement_2") or get_field("m2")
                if frames is None or phases is None:
                    raise ValueError(
                        "Structured records must contain 'frame' and 'phase'"
                    )
                out = []
                for fr, ph, mm1, mm2 in zip(
                    frames,
                    phases,
                    m1 if m1 is not None else [None] * len(frames),
                    m2 if m2 is not None else [None] * len(frames),
                ):
                    out.append(
                        PyRecord(
                            int(fr),
                            str(ph),
                            None if mm1 is None else float(mm1),
                            None if mm2 is None else float(mm2),
                        )
                    )
                return out
            else:
                arr = np.asarray(recs)
                if arr.ndim == 1:
                    arr = arr[np.newaxis, :]
                out = []
                for row in arr:
                    fr = int(row[0])
                    ph = str(row[1])
                    m1 = (
                        None
                        if (
                            len(row) < 3
                            or row[2] is None
                            or (isinstance(row[2], float) and np.isnan(row[2]))
                        )
                        else float(row[2])
                    )
                    m2 = (
                        None
                        if (
                            len(row) < 4
                            or row[3] is None
                            or (isinstance(row[3], float) and np.isnan(row[3]))
                        )
                        else float(row[3])
                    )
                    out.append(PyRecord(fr, ph, m1, m2))
                return out

        if isinstance(recs, (list, tuple)):
            out = []
            for item in recs:
                if hasattr(item, "frame") and hasattr(item, "phase"):
                    out.append(item)
                else:
                    fr = int(item[0])
                    ph = str(item[1])
                    m1 = None if len(item) < 3 or item[2] is None else float(item[2])
                    m2 = None if len(item) < 4 or item[3] is None else float(item[3])
                    out.append(PyRecord(fr, ph, m1, m2))
            return out
        raise ValueError("Unsupported records format")

    parsed_records = parse_records(records)

    def parse_reference(ref):
        if ref is None:
            return PyContourPoint(
                frame_index=0, point_index=0, x=0.0, y=0.0, z=0.0, aortic=False
            )
        arr = np.asarray(ref)
        if arr.ndim == 1:
            if arr.shape[0] >= 4:
                fr, x, y, z = arr[:4]
            else:
                raise ValueError("reference must be length 4 or shape (1,4)")
        else:
            if arr.shape[1] < 4:
                raise ValueError("reference must be (N,4)-like")
            nonzero = np.any(arr != 0, axis=1)
            if np.any(nonzero):
                row = arr[nonzero][0]
            else:
                row = arr[0]
            fr, x, y, z = row[:4]
        return PyContourPoint(
            frame_index=int(fr),
            point_index=0,
            x=float(x),
            y=float(y),
            z=float(z),
            aortic=False,
        )

    ref_point = parse_reference(reference)

    def none_if_empty(lst):
        return None if not lst else lst

    pyinput = PyInputData(
        lumen=lumen_contours,
        eem=none_if_empty(eem_contours),
        calcification=none_if_empty(calc_contours),
        sidebranch=none_if_empty(sidebranch_contours),
        record=parsed_records,
        ref_point=ref_point,
        diastole=bool(diastole),
        label=str(label),
    )

    return pyinput


def geometry_to_frames_array(geometry: mm.PyGeometry) -> Dict[str, np.ndarray]:
    """
    Convert PyGeometry to dictionary of numpy arrays organized by frame.

    Returns a dictionary where each key is a frame ID and the value is another
    dictionary with contour types as keys and numpy arrays as values.

    Parameters
    ----------
    geometry : PyGeometry
        The geometry to convert

    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        Dictionary mapping frame IDs to dictionaries of contour arrays
    """
    result = {}

    for frame in geometry.frames:
        frame_data = {}

        # Add lumen
        lumen_pts = [(p.frame_index, p.x, p.y, p.z) for p in frame.lumen.points]
        frame_data["lumen"] = (
            np.array(lumen_pts, dtype=float)
            if lumen_pts
            else np.zeros((0, 4), dtype=float)
        )

        # Add extras
        for contour_type, contour in frame.extras.items():
            pts = [(p.frame_index, p.x, p.y, p.z) for p in contour.points]
            frame_data[contour_type.lower()] = (
                np.array(pts, dtype=float) if pts else np.zeros((0, 4), dtype=float)
            )

        # Add reference point
        if frame.reference_point:
            ref = frame.reference_point
            frame_data["reference"] = np.array(
                [[ref.frame_index, ref.x, ref.y, ref.z]], dtype=float
            )
        else:
            frame_data["reference"] = np.zeros((0, 4), dtype=float)

        result[str(frame.id)] = frame_data

    return result
