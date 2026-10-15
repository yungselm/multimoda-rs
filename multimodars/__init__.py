from __future__ import annotations

from ._converters import (
    numpy_to_centerline,
    numpy_to_geometry,
    numpy_to_inputdata,
    to_array,
)
from ._processing import (
    align_combined,
    align_manual,
    align_three_point,
    build_adjacency_map,
    discretize_vessel,
    find_centerline_bounded_points,
    find_proximal_distal_scaling,
    from_array_doublepair,
    from_array_full,
    from_array_single,
    from_array_singlepair,
    from_file_doublepair,
    from_file_full,
    from_file_single,
    from_file_singlepair,
    read_centerline_vtp,
    to_obj,
    write_centerline_vtp,
)
from .ccta import create_wall_mesh, export_section_stl, label, scale, stitch
from .ccta.centerline_prep import (
    load_centerline,
    prepare_centerline,
)
from .ccta.debug_plots import (
    plot_boundary_edges,
    plot_centerline_edges,
    plot_results_key,
    plot_sharp_angles,
)
from .ccta.discretization_map import (
    discretize_vessel_tree,
    find_sharp_angles,
    label_branches_pair,
)
from .ccta.labeling import (
    label_anomalous_region,
    label_branches,
    label_geometry,
)
from .ccta.mesh_regions import (
    cut_at_slice,
    keep_labeled_points_from_mesh,
    remove_branch,
    remove_labeled_points_from_mesh,
)
from .ccta.postprocessing import fix_and_remesh_stitched_mesh, manual_hole_fill
from .ccta.scaling import (
    find_aorta_scaling,
    find_aortic_wall_scaling,
    find_distal_and_proximal_scaling,
    scale_region_centerline_morphing,
    sync_results_to_mesh,
)
from .ccta.stitching import (
    stitch_ccta_to_intravascular,
)
from .io import read_geometrical, write_geometries

# import os
# from importlib.metadata import version
from .multimodars import (
    PyCenterline,
    PyCenterlinePoint,
    PyContour,
    PyContourPoint,
    PyContourType,
    PyDiscretizedBranch,
    PyDiscretizedVesselTree,
    PyFrame,
    PyGeometry,
    PyGeometryPair,
    PyInputData,
    PyRecord,
)

__all__ = [
    "PyCenterline",
    "PyCenterlinePoint",
    "PyContour",
    "PyContourPoint",
    "PyContourType",
    "PyDiscretizedBranch",
    "PyDiscretizedVesselTree",
    "PyFrame",
    "PyGeometry",
    "PyGeometryPair",
    "PyInputData",
    "PyRecord",
    "align_combined",
    "align_manual",
    "align_three_point",
    "build_adjacency_map",
    "create_wall_mesh",
    "cut_at_slice",
    "discretize_vessel",
    "discretize_vessel_tree",
    "export_section_stl",
    "find_aorta_scaling",
    "find_aortic_wall_scaling",
    "find_centerline_bounded_points",
    "find_distal_and_proximal_scaling",
    "find_proximal_distal_scaling",
    "find_sharp_angles",
    "fix_and_remesh_stitched_mesh",
    "from_array_doublepair",
    "from_array_full",
    "from_array_single",
    "from_array_singlepair",
    "from_file_doublepair",
    "from_file_full",
    "from_file_single",
    "from_file_singlepair",
    "keep_labeled_points_from_mesh",
    "label",
    "label_anomalous_region",
    "label_branches",
    "label_branches_pair",
    "label_geometry",
    "load_centerline",
    "manual_hole_fill",
    "numpy_to_centerline",
    "numpy_to_geometry",
    "numpy_to_inputdata",
    "plot_boundary_edges",
    "plot_centerline_edges",
    "plot_results_key",
    "plot_sharp_angles",
    "postprocess_stitched_mesh",
    "prepare_centerline",
    "read_centerline_vtp",
    "read_geometrical",
    "remove_branch",
    "remove_labeled_points_from_mesh",
    "scale",
    "scale_region_centerline_morphing",
    "stitch",
    "stitch_ccta_to_intravascular",
    "sync_results_to_mesh",
    "to_array",
    "to_obj",
    "write_centerline_vtp",
    "write_geometries",
]

# def _print_banner():
#     v = version("multimodars")
#     print(r"""
#               .__   __  .__                   .___
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
