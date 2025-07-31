import numpy as np
from typing import Union, Tuple
from multimodars import PyGeometry, PyCenterline


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
    # Import here to avoid circular imports
    from multimodars import PyContour, PyCenterline, PyGeometry, PyGeometryPair

    # Contour: flatten points
    if isinstance(generic, PyContour):
        pts = [(p.frame_index, p.x, p.y, p.z) for p in generic.points]
        return np.array(pts, dtype=float)

    # Centerline: use contour_point attrs
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

    # Geometry: use helper
    if isinstance(generic, PyGeometry):
        return _geometry_to_numpy(generic)

    # Geometry Pair: two geometries
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
) -> PyGeometry:
    """
    Build a PyGeometry from four (M, 4) NumPy arrays or structured arrays, one per layer, grouping by frame_index.

    Each row in the ``*_arr`` is [frame_index, x, y, z].

    Returns a PyGeometry containing:
      - contours: list of PyContour (one per frame in contours_arr)
      - catheters: list of PyContour (one per frame in catheters_arr)
      - walls:    list of PyContour (one per frame in walls_arr)
      - reference: single PyContourPoint from reference_arr[0]
    """
    from multimodars import PyContour, PyContourPoint

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
) -> PyCenterline:
    """
    Build a PyCenterline from a numpy array of shape (N,3),
    where each row is (x, y, z).

    Args:
        arr: np.ndarray of shape (N,3)
        aortic: whether to mark each point as aortic

    Returns:
        PyCenterline
    """
    from multimodars import PyContourPoint, PyCenterline

    # sanity check
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

    # Use your static constructor to build a PyCenterline
    return PyCenterline.from_contour_points(pts)
