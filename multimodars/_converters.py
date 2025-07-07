import numpy as np
from multimodars import PyGeometryPair, PyGeometry, PyContour, PyContourPoint

def geometry_to_numpy(geom: PyGeometry) -> np.ndarray:
    """
    Flatten all contours+points into a single (N,3) array of XYZ,
    or return a list of per-contour arrays, whichever you prefer.
    """
    # example: stack all contours end‑to‑end
    arrays = [ np.array(cnt.points_as_tuples(), dtype=float)
               for cnt in geom.contours ]
    # concatenate along the 0‑axis
    return np.concatenate(arrays, axis=0)


def numpy_to_geometry(arr: np.ndarray, reference_point=None) -> PyGeometry:
    """
    Build a new PyGeometry from an (N,3) array.  You’ll need to
    decide how to split into contours (fixed length? markers?)
    and how to generate ContourPoints from raw floats.
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