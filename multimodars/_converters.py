import numpy as np
from typing import Union, Tuple
from multimodars import PyGeometry, PyCenterline

def to_array(generic) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Convert various Py* objects into numpy arrays.

    - PyContour: returns array of shape (N, 4) as (frame_index, x, y, z)
    - PyCenterline: returns array of shape (N, 4) as (frame_index, x, y, z)
    - PyGeometry: returns array of shape (max_len, 4, 4) [contours, catheter, walls, reference]
    - PyGeometryPair: returns tuple of two arrays, each as PyGeometry

    Returns
    -------
    np.ndarray
        A 3D array of shape (max_len, 4 layers, 4 coords) for PyGeometry
        A 3D array of shape (frame_index, x, y, z) for PyContour, PyCenterline
    Tuple[np.ndarray, np.ndarray]
        Two A 3D arrays of shape (max_len, 4 layers, 4 coords) for PyGeometryPair
    """
    # Import here to avoid circular imports
    from multimodars import PyContour, PyCenterline, PyGeometry, PyGeometryPair

    # Contour: flatten points
    if isinstance(generic, PyContour):
        pts = [(p.frame_index, p.x, p.y, p.z) for p in generic.points]
        return np.array(pts, dtype=float)

    # Centerline: use contour_point attrs
    if isinstance(generic, PyCenterline):
        pts = [(
            p.contour_point.frame_index,
            p.contour_point.x,
            p.contour_point.y,
            p.contour_point.z
        ) for p in generic.points]
        return np.array(pts, dtype=float)

    # Geometry: use helper
    if isinstance(generic, PyGeometry):
        return _geometry_to_numpy(generic)

    # Geometry Pair: two geometries
    if isinstance(generic, PyGeometryPair):
        dia_arr = _geometry_to_numpy(generic.dia_geom)
        sys_arr = _geometry_to_numpy(generic.sys_geom)
        return dia_arr, sys_arr

    raise TypeError(f"Unsupported type for to_array: {type(generic)}")


def _geometry_to_numpy(geom) -> np.ndarray:
    """
    Flatten all contours + catheter + walls + reference point
    into a single array of shape (max_len, 4, 4), where:

      - axis 1 layers: contours, catheter, walls, reference
      - axis 2 coords: [frame_index, x, y, z]

    Returns
    -------
    np.ndarray
        A 3D array of shape (max_len, 4 layers, 4 coords) ready for downstream processing.
    """
    layers = {
        "contours": [
            (pt.frame_index, pt.x, pt.y, pt.z)
            for contour in geom.contours
            for pt in contour.points
        ],
        "catheter": [
            (pt.frame_index, pt.x, pt.y, pt.z)
            for contour in geom.catheter
            for pt in contour.points
        ],
        "walls": [
            (pt.frame_index, pt.x, pt.y, pt.z)
            for contour in geom.walls
            for pt in contour.points
        ],
        "reference": [
            (geom.reference_point.frame_index,
             geom.reference_point.x,
             geom.reference_point.y,
             geom.reference_point.z)
        ]
    }
    max_len = max(len(lst) for lst in layers.values())
    arr = np.zeros((4, max_len, 4), dtype=float)
    for i, key in enumerate(["contours", "catheter", "walls", "reference"]):
        pts_list = layers[key]
        if len(pts_list) == 0:
            # make an empty (0,4) array so broadcasting works
            pts = np.empty((0, 4), dtype=float)
        else:
            pts = np.array(pts_list, dtype=float).reshape(-1, 4)
        arr[i, : pts.shape[0], :] = pts
    # transpose to (max_len, 4 layers, 4 coords)
    return arr.transpose(1, 0, 2)


def numpy_to_geometry_layers(
    contours_arr: np.ndarray,
    catheter_arr: np.ndarray,
    walls_arr: np.ndarray,
    reference_arr: np.ndarray,
) -> PyGeometry:
    """
    Build a PyGeometry from four (M, 4) NumPy arrays or structured arrays, one per layer, grouping by frame_index.

    Each row in the *_arr is [frame_index, x, y, z].

    Returns a PyGeometry containing:
      - contours: list of PyContour (one per frame in contours_arr)
      - catheter: list of PyContour (one per frame in catheter_arr)
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
            pts_arr = arr[arr[:, 0].astype(int) == frame]
            pts = [
                PyContourPoint(
                    frame_index=int(fr), point_index=i,
                    x=float(x), y=float(y), z=float(z),
                    aortic=False
                )
                for i, (fr, x, y, z) in enumerate(pts_arr)
            ]
            contours.append(PyContour(frame, pts))
        return contours

    contours = build_layer(contours_arr, "contours_arr")
    catheter = build_layer(catheter_arr, "catheter_arr")
    walls    = build_layer(walls_arr,    "walls_arr")

    # Reference: structured or numeric
    if reference_arr.dtype.names and reference_arr.ndim == 1:
        ref0 = reference_arr[0]
        fr, x, y, z = (float(ref0[name]) for name in reference_arr.dtype.names[:4])
    else:
        ref_flat = np.asarray(reference_arr, dtype=float).flatten()
        if ref_flat.size < 4:
            raise ValueError(f"reference_arr must contain at least 4 values, got {ref_flat.size}")
        fr, x, y, z = ref_flat[:4]

    reference = PyContourPoint(
        frame_index=int(fr), point_index=0,
        x=float(x), y=float(y), z=float(z),
        aortic=False
    )

    return PyGeometry(contours, catheter, walls, reference)


def numpy_to_geometry(
    arr: np.ndarray,
    layer_names=("contours","catheter","walls","reference")
) -> PyGeometry:
    """
    Build a PyGeometry from a single array of shape (N,4,4), where:
      - arr[:,i,:] is the (M,4) block for layer layer_names[i]
    """
    from multimodars import PyGeometry

    contours_arr  = arr[:, layer_names.index("contours"), :]
    catheter_arr  = arr[:, layer_names.index("catheter"), :]
    walls_arr     = arr[:, layer_names.index("walls"), :]
    reference_arr = arr[:, layer_names.index("reference"), :]

    return numpy_to_geometry_layers(
        contours_arr, catheter_arr, walls_arr, reference_arr
    )


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
        pts.append(PyContourPoint(
            frame_index=i,
            point_index=0,
            x=float(x),
            y=float(y),
            z=float(z),
            aortic=aortic,
        ))

    # Use your static constructor to build a PyCenterline
    return PyCenterline.from_contour_points(pts)
