from __future__ import annotations

from .multimodars import *
from ._converters import (
    to_array, 
    numpy_to_geometry, 
    numpy_to_centerline
)
from ._wrappers import (
    from_file,
    from_array,
    to_centerline,
)
from .io.write_geometries import centerline_to_obj

__all__ = [
    # rust‑backend functions
    # (fill these in manually, or use `multimodars = dir()` hack to auto‑generate)
    "align_three_point",
    "align_manual",
    "from_file_full",
    "from_file_doublepair",
    "from_file_singlepair",
    "from_file_single",
    "create_catheter_geometry",
    "geometry_from_array",
    "from_array_full",
    "from_array_doublepair",
    "from_array_singlepair",
    # rust class bound methods
    ## PyContourPoint
    "distance",
    ## PyContour
    "compute_centroid",
    "points_as_tuples",
    "find_farthest_points",
    "find_closest_opposite",
    "get_elliptic_ratio",
    "get_area",
    "rotate",
    "translate",
    "sort_contour_points",
    ## PyGeometry
    "rotate",
    "translate",
    "smooth_contours",
    "reorder",
    ## PyCenterline
    "from_contour_points",
    # converters
    "to_array",
    "numpy_to_geometry",
    "numpy_to_centerline",
    # wrappers
    "from_file",
    "from_array",
    "to_centerline",
    "to_obj",
    # io
    "centerline_to_obj"
]
