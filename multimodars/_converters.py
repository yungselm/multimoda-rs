import numpy as np
from multimodars import PyGeometry, PyContourPoint

def geometry_to_numpy(geom: PyGeometry, mode="contours") -> np.ndarray:
    """
    Flatten all contours+points into a single (N, 4) array of
    [frame_index, x, y, z], concatenated in the requested mode.
    mode: "contours" (default), "catheter", or "walls"
    """
    if mode == "contours":
        sequences = geom.contours
    elif mode == "catheter":
        sequences = geom.catheter
    elif mode == "walls":
        # if you later add walls to PyGeometry
        sequences = getattr(geom, "walls", [])
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    arrays = []
    for seq in sequences:
        # seq.points is List[PyContourPoint]
        pts = seq.points
        if not pts:
            continue
        # build an (M,4) Python list
        block = [
            (p.frame_index, p.x, p.y, p.z)
            for p in pts
        ]
        # convert to float; frame_index will cast to float
        arrays.append(np.array(block, dtype=float))

    if not arrays:
        # no points at all
        return np.empty((0, 4), dtype=float)

    # concatenate into one (N,4) array
    return np.concatenate(arrays, axis=0)


def numpy_to_geometry(arr: np.ndarray, *, reference_point=None) -> PyGeometry:
    """
    Build a new PyGeometry from an (N,4) array of [frame, x, y, z].
    Packs *all* points into a single contour (toy example).
    """
    from multimodars import PyContour, PyContourPoint

    pts = [
        PyContourPoint(
            frame_index=int(row[0]),
            point_index=i,
            x=float(row[1]),
            y=float(row[2]),
            z=float(row[3]),
            aortic=False,
        )
        for i, row in enumerate(arr)
    ]

    # one big contour; you could split by frame or by some delimiter if you like
    contour = PyContour(id=0, points=pts, centroid=(0.0, 0.0, 0.0))

    # pick your reference point
    ref = reference_point if reference_point is not None else pts[0]

    return PyGeometry(contours=[contour], catheter=[], reference_point=ref)
