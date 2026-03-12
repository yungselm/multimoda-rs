from __future__ import annotations

# import os
# from importlib.metadata import version

from .multimodars import (
    PyContourPoint,
    PyContour,
    PyFrame,
    PyGeometry,
    PyGeometryPair,
    PyCenterline,
    PyCenterlinePoint,
    PyInputData,
    PyRecord,
    PyContourType,
    from_file_full,
    from_file_doublepair,
    from_file_singlepair,
    from_file_single,
    from_array_full,
    from_array_doublepair,
    from_array_singlepair,
    from_array_single,
    align_three_point,
    align_manual,
    align_combined,
    to_obj,
    find_centerline_bounded_points_simple,
    find_proximal_distal_scaling,
)
from ._converters import (
    to_array,
    numpy_to_geometry,
    numpy_to_centerline,
    numpy_to_inputdata,
)
from .io import read_geometrical, write_geometries
from .ccta.labeling import (
    label_geometry,
    label_anomalous_region,
)
from .ccta.manipulating import (
    scale_region_centerline_morphing,
    find_distal_and_proximal_scaling,
    find_aorta_scaling,
    remove_labeled_points_from_mesh,
    keep_labeled_points_from_mesh,
    sync_results_to_mesh,
    stitch_ccta_to_intravascular,
)
from .ccta.fixing_functions import fix_and_remesh_stitched_mesh, manual_hole_fill
from .ccta.debug_plots import plot_results_key

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
    "align_combined",
    "to_obj",
    "find_centerline_bounded_points_simple",
    "find_proximal_distal_scaling",
    # I/O
    "read_geometrical",
    "write_geometries",
    # CCTA module
    "label_geometry",
    "label_anomalous_region",
    "scale_region_centerline_morphing",
    "find_distal_and_proximal_scaling",
    "find_aorta_scaling",
    "remove_labeled_points_from_mesh",
    "keep_labeled_points_from_mesh",
    "sync_results_to_mesh",
    "stitch_ccta_to_intravascular",
    "fix_and_remesh_stitched_mesh",
    "manual_hole_fill",
    "plot_results_key",
]

# def _print_banner():
#     v = version("multimodars")
#     print(r"""
#   .__   __  .__                   .___                    
#   _____  __ __|  |_/  |_|__| _____   ____   __| _/____ _______  ______
#  /     \|  |  \  |\   __\  |/     \ /  _ \ / __ |\__  \\_  __ \/  ___/
# |  Y Y  \  |  /  |_|  | |  |  Y Y  (  <_> ) /_/ | / __ \|  | \/\___ \ 
# |__|_|  /____/|____/__| |__|__|_|  /\____/\____ |(____  /__|  /____  >
#       \/                         \/            \/     \/           \/  
# """)
#     print(f"  version  : {v}")
#     print(f"  docs     : https://multimoda-rs.readthedocs.io")
#     print(f"  license  : MIT\n")

# if os.environ.get("MULTIMODARS_SILENT", "0") == "0":
#     _print_banner()