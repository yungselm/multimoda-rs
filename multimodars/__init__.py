from __future__ import annotations

from .multimodars import *
from ._converters import (
    to_array,
    numpy_to_geometry,
    numpy_to_centerline,
    numpy_to_inputdata,
)
from ._wrappers import (
    from_file,
    from_array,
)
from .io import read_geometrical, write_geometries
from .ccta.adjust_ccta import (
    label_geometry,
    label_anomalous_region,
    scale_region_centerline_morphing,
    find_distal_and_proximal_scaling,
    find_aorta_scaling,
)

__all__ = [
    # Core classes
    "PyContourPoint",
    "PyContour",
    "PyFrame",
    "PyGeometry",
    "PyGeometryPair",
    "PyCenterline",
    "PyCenterlinePoint",
    "PyInputData",
    "PyRecord",
    "PyContourType",
    # Converter functions
    "to_array",
    "numpy_to_geometry",
    "numpy_to_centerline",
    "numpy_to_inputdata",
    # Wrapper functions
    "from_file",
    "from_array",
    # Processing functions (from Rust)
    "from_file_full",
    "from_file_doublepair",
    "from_file_singlepair",
    "from_file_single",
    "from_array_full",
    "from_array_doublepair",
    "from_array_singlepair",
    "from_array_single",
    "align_three_point",
    "align_manual",
    "to_obj",
    "find_centerline_bounded_points_simple",
    "find_proximal_distal_scaling",
    # CCTA module
    "label_geometry",
    "label_anomalous_region",
    "scale_region_centerline_morphing",
    "find_distal_and_proximal_scaling",
    "find_aorta_scaling",
]
