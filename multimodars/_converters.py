import numpy as np
from typing import Union, Tuple


def to_array(generic) -> Union[np.ndarray, dict, Tuple[dict, dict]]:
    """
    Convert various multimodars Py* objects into numpy array(s) or dictionaries of arrays.

    Parameters
    ----------
    generic : PyContour, PyCenterline, PyGeometry, or PyGeometryPair
        The object to be converted to numpy representation.

    Returns
    -------
    np.ndarray
        For PyContour or PyCenterline:
        A 2D array of shape (N, 4), where each row is (frame_index, x, y, z).

    dict[str, np.ndarray]
        For PyGeometry:
        A dictionary with keys ["contours", "catheters", "walls", "reference"],
        each containing a 2D array of shape (M, 4), where M is the number of points in that layer.
        "reference" is a (1, 4) array or (0, 4) if missing.

    Tuple[dict[str, np.ndarray], dict[str, np.ndarray]]
        For PyGeometryPair:
        A tuple of two dictionaries (one for diastolic, one for systolic), each in the same format
        as returned for a single PyGeometry.

    Raises
    ------
    TypeError
        If the input type is not one of the supported multimodars types.
    """
    from . import PyContour, PyCenterline, PyGeometry, PyGeometryPair

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

    if isinstance(generic, PyGeometry):
        return _geometry_to_numpy(generic)

    if isinstance(generic, PyGeometryPair):
        dia_dict = _geometry_to_numpy(generic.dia_geom)
        sys_dict = _geometry_to_numpy(generic.sys_geom)
        return dia_dict, sys_dict

    raise TypeError(f"Unsupported type for to_array: {type(generic)}")


def _geometry_to_numpy(geom) -> dict[str, np.ndarray]:
    def extract_points(items, label: str) -> np.ndarray:
        result = []
        for obj in items:
            pts = [(p.frame_index, p.x, p.y, p.z) for p in obj.points]
            result.extend(pts)
        return (
            np.array(result, dtype=float) if result else np.zeros((0, 4), dtype=float)
        )

    contours = extract_points(geom.contours, "contours")
    catheters = extract_points(geom.catheters, "catheters")
    walls = extract_points(geom.walls, "walls")

    reference = np.array(
        [
            [
                geom.reference_point.frame_index,
                geom.reference_point.x,
                geom.reference_point.y,
                geom.reference_point.z,
            ]
        ],
        dtype=float,
    )

    return {
        "contours": contours,  # (N1, 4)
        "catheters": catheters,  # (N2, 4)
        "walls": walls,  # (N3, 4)
        "reference": reference,  # (1, 4)
    }


def numpy_to_geometry(
    contours_arr: np.ndarray,
    catheters_arr: np.ndarray,
    walls_arr: np.ndarray,
    reference_arr: np.ndarray,
) -> "PyGeometry":
    """
    Build a PyGeometry from four (M, 4) NumPy arrays or structured arrays, one per layer, grouping by frame_index.

    Each row in the ``*_arr`` is [frame_index, x, y, z].

    Returns a PyGeometry containing:
      - contours: list of PyContour (one per frame in contours_arr)
      - catheters: list of PyContour (one per frame in catheters_arr)
      - walls:    list of PyContour (one per frame in walls_arr)
      - reference: single PyContourPoint from reference_arr[0]
    """
    from . import PyContour, PyContourPoint, PyGeometry

    def _to_numeric_array(arr: np.ndarray, layer_name: str) -> np.ndarray:
        # Handle structured arrays (e.g. from np.genfromtxt with names)
        if arr.ndim == 1 and arr.dtype.names:
            try:
                arr = np.vstack([arr[name] for name in arr.dtype.names]).T
            except Exception:
                raise ValueError(f"Could not convert structured array for {layer_name}")
        arr = np.asarray(arr, dtype=float)
        return arr

    def build_layer(arr: np.ndarray, layer_name: str) -> list[PyContour]:
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
            contours.append(PyContour(frame, pts))
        return contours

    contours = build_layer(contours_arr, "contours_arr")
    catheters = build_layer(catheters_arr, "catheters_arr")
    walls = build_layer(walls_arr, "walls_arr")

    reference_arr = _to_numeric_array(reference_arr, "reference_arr")
    # if there's exactly one point, make it into a (1,4) row
    if reference_arr.ndim == 1:
        reference_arr = reference_arr[np.newaxis, :]
    non_zero_mask = np.any(reference_arr != 0, axis=1)
    if np.any(non_zero_mask):
        ref_row = reference_arr[non_zero_mask][0]
        fr, x, y, z = ref_row[:4]
    else:
        fr, x, y, z = 0, 0, 0, 0

    reference = PyContourPoint(
        frame_index=int(fr),
        point_index=0,
        x=float(x),
        y=float(y),
        z=float(z),
        aortic=False,
    )

    return PyGeometry(contours, catheters, walls, reference)


def numpy_to_centerline(
    arr: np.ndarray,
    aortic: bool = False,
) -> "PyCenterline":
    """
    Build a PyCenterline from a numpy array of shape (N,3),
    where each row is (x, y, z).

    Args:
        arr: np.ndarray of shape (N,3)
        aortic: whether to mark each point as aortic

    Returns:
        PyCenterline
    """
    from . import PyContourPoint, PyCenterline

    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("Input must be a (N,3) array")

    pts = []
    for i, (x, y, z) in enumerate(arr.tolist()):
        # point_index here is meaningless for a centerline; set to 0
        pts.append(
            PyContourPoint(
                frame_index=i,
                point_index=0,
                x=float(x),
                y=float(y),
                z=float(z),
                aortic=aortic,
            )
        )

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
) -> "PyInputData":
    """
    Create a PyInputData from either Py* objects (no-op) or NumPy arrays.

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

    # Reuse the internal helpers already present in your file.
    # If you keep the functions _to_numeric_array and build_layer, use them.
    # Otherwise include the minimal versions below:

    def _to_numeric_array(arr, layer_name: str):
        if arr is None:
            return np.zeros((0, 4), dtype=float)
        if isinstance(arr, (list, tuple)):
            arr = np.asarray(arr, dtype=object)
        if isinstance(arr, np.ndarray) and arr.dtype.names:
            try:
                arr = np.vstack([arr[name] for name in arr.dtype.names]).T
            except Exception as e:
                raise ValueError(f"Could not convert structured array for {layer_name}: {e}")
        arr = np.asarray(arr)
        if arr.size == 0:
            return np.zeros((0, 4), dtype=float)
        if arr.ndim == 1:
            if arr.shape[0] == 4:
                arr = arr[np.newaxis, :]
            else:
                raise ValueError(f"{layer_name} 1D array must have length 4, got {arr.shape}")
        return arr.astype(object)

    def build_layer_from_array(arr, layer_name: str):
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
                pts.append(PyContourPoint(frame_index=fr, point_index=i, x=x, y=y, z=z, aortic=False))
            contours.append(PyContour(frame, pts))
        return contours

    def ensure_contours(maybe):
        """Accept a list of PyContour already, or numpy arrays, or None."""
        if maybe is None:
            return []
        if isinstance(maybe, list) and maybe and hasattr(maybe[0], "points") and hasattr(maybe[0], "id"):
            return maybe
        return build_layer_from_array(maybe, "layer")

    lumen_contours = ensure_contours(lumen)
    eem_contours = ensure_contours(eem)
    calc_contours = ensure_contours(calcification)
    sidebranch_contours = ensure_contours(sidebranch)

    def parse_records(recs):
        if recs is None:
            return None
        if isinstance(recs, (list, tuple)) and recs and hasattr(recs[0], "frame") and hasattr(recs[0], "phase"):
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
                    raise ValueError("Structured records must contain 'frame' and 'phase'")
                out = []
                for fr, ph, mm1, mm2 in zip(frames, phases, m1 if m1 is not None else [None]*len(frames), m2 if m2 is not None else [None]*len(frames)):
                    out.append(PyRecord(int(fr), str(ph), None if mm1 is None else float(mm1), None if mm2 is None else float(mm2)))
                return out
            else:
                arr = np.asarray(recs)
                if arr.ndim == 1:
                    arr = arr[np.newaxis, :]
                out = []
                for row in arr:
                    fr = int(row[0])
                    ph = str(row[1])
                    m1 = None if (len(row) < 3 or row[2] is None or (isinstance(row[2], float) and np.isnan(row[2]))) else float(row[2])
                    m2 = None if (len(row) < 4 or row[3] is None or (isinstance(row[3], float) and np.isnan(row[3]))) else float(row[3])
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
            return PyContourPoint(frame_index=0, point_index=0, x=0.0, y=0.0, z=0.0, aortic=False)
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
        return PyContourPoint(frame_index=int(fr), point_index=0, x=float(x), y=float(y), z=float(z), aortic=False)

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