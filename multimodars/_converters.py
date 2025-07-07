import numpy as np
from multimodars import PyGeometryPair, PyGeometry, PyContour, PyContourPoint

def geometry_to_numpy(geom: PyGeometry, mode="contours") -> np.ndarray:
    """
    Flatten all contours+points into a single (N,3) array of XYZ,
    or return a list of per-contour arrays, whichever you prefer.
    mode: "contours" (default), "catheter", or "walls"
    """
    if mode == "contours":
        arrays = [np.array(cnt.points_as_tuples(), dtype=float)
                  for cnt in geom.contours]
    elif mode == "catheter":
        arrays = [np.array(cat.points_as_tuples(), dtype=float)
                  for cat in geom.catheter]
    elif mode == "walls":
        arrays = [np.array(wall.points_as_tuples(), dtype=float)
                  for wall in getattr(geom, "walls", [])]
    else:
        raise ValueError(f"Unknown mode: {mode}")
    if not arrays:
        return np.empty((0, 3), dtype=float)
    return np.concatenate(arrays, axis=0)


def numpy_to_geometry(arr: np.ndarray, reference_point=None) -> PyGeometry:
    """
    Build a new PyGeometry from an (N,3) array.
    """
    # simple toy: treat entire array as one contour, index everything
    from multimodars import PyContour, PyContourPoint
    pts = [
        PyContourPoint(
          frame_index=0,
          point_index=i,
          x=float(x), y=float(y), z=float(z),
          aortic=False 
        )
        for i, (x,y,z) in enumerate(arr)
    ]
    contour = PyContour(id=0, points=pts, centroid=(0,0,0))
    # pick reference_point if given, else first point
    ref = reference_point or pts[0]
    return PyGeometry(contours=[contour], catheter=[], reference_point=ref)