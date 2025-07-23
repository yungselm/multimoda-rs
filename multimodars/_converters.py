import numpy as np
from multimodars import PyContour, PyContourPoint, PyGeometry, PyCenterline

def geometry_to_numpy(geom) -> np.ndarray:
    """
    Flatten all contours + catheter + walls + reference point
    into a single array of shape (N, 4, 4), where::

      - axis 1 layers: contours, catheter, walls, reference
      - axis 2 coords: [frame_index, x, y, z]

    Returns
    -------
    np.ndarray
        A 3D array of shape (max_frames, 4 layers, 4 coords) ready for downstream processing.
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

    # Allocate zeros: shape = (4 layers, max_len points, 4 coords)
    arr = np.zeros((4, max_len, 4), dtype=float)

    # Each layer i gets its own slice:
    for i, key in enumerate(["contours","catheter","walls","reference"]):
        pts = np.array(layers[key], dtype=float)  # shape (M,4)
        arr[i, : pts.shape[0], :] = pts

    return arr.transpose(1, 0, 2)  # now shape = (max_len, 4, 4)


def numpy_to_geometry_layers(
    contours_arr: np.ndarray,
    catheter_arr: np.ndarray,
    walls_arr: np.ndarray,
    reference_arr: np.ndarray,
) -> PyGeometry:
    """
    Build a PyGeometry from four (M, 4) NumPy arrays, one per layer, where::

      - each row: [frame_index, x, y, z]
      - contours_arr, catheter_arr, walls_arr: shape (Nc,4), (Ncat,4), (Nw,4)
      - reference_arr: shape (1,4); only the first row is used

    Parameters
    ----------
    contours_arr : np.ndarray
        (Nc,4) array of contour points.
    catheter_arr : np.ndarray
        (Ncat,4) array of catheter points.
    walls_arr : np.ndarray
        (Nw,4) array of wall points.
    reference_arr : np.ndarray
        (1,4) or (Nr,4) array; only row 0 is read as the reference.

    Returns
    -------
    PyGeometry
    """
    def build_contour(points_arr: np.ndarray, contour_id: int) -> PyContour:
        pts = [
            PyContourPoint(
                frame_index=int(fr), point_index=i,  # random point index setting, gets corrected later
                x=float(x), y=float(y), z=float(z),
                aortic=False  # also gets set later
            )
            for i, (fr, x, y, z) in enumerate(points_arr)
        ]
        centroid = (
            pts[0].x if pts else 0.0,
            pts[0].y if pts else 0.0,
            pts[0].z if pts else 0.0,
        )
        return PyContour(contour_id, pts, centroid)

    # In this simplest version, we treat each entire layer as a single contour.
    contours = [build_contour(contours_arr, 0)]
    catheter = [build_contour(catheter_arr, 0)]
    walls    = [build_contour(walls_arr,    0)]

    # Reference point should be only one with the current setup
    fr, x, y, z = reference_arr[0]
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
