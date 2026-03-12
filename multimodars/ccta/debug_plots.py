from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import trimesh
from trimesh.points import PointCloud

if TYPE_CHECKING:
    from ..multimodars import PyCenterline


def plot_results_key(
    results: dict,
    aorta_points: bool = True,
    rca_points: bool = False,
    lca_points: bool = False,
    rca_removed_points: bool = False,
    proximal_points: bool = False,
    distal_points: bool = False,
    anomalous_points: bool = False,
    cl_rca: PyCenterline | None = None,
    cl_lca: PyCenterline | None = None,
    cl_aorta: PyCenterline | None = None,
):
    """Open an interactive 3-D scene visualising selected regions from a results dict.

    Toggle which regions are shown by passing ``True``/``False`` for each flag.

    Colour coding:

    * **Yellow** - aortic points.
    * **Blue** - RCA coronary points.
    * **Green** - LCA coronary points.
    * **Red** - removed / reassigned points (RCA + LCA combined).
    * **Cyan** - proximal points.
    * **Magenta** - distal points.
    * **Orange** - anomalous (intramural) points.

    Parameters
    ----------
    results : dict
        Dictionary with keys ``"aorta_points"``, ``"rca_points"``,
        ``"lca_points"``, ``"rca_removed_points"``, ``"proximal_points"``,
        ``"distal_points"``, ``"anomalous_points"`` - each a list of
        ``(x, y, z)`` tuples.
    aorta_points : bool
        Show aortic points (yellow).
    rca_points : bool
        Show RCA coronary points (blue).
    lca_points : bool
        Show LCA coronary points (green).
    rca_removed_points : bool
        Show removed RCA points (red).
    proximal_points : bool
        Show proximal points (cyan).
    distal_points : bool
        Show distal points (magenta).
    anomalous_points : bool
        Show anomalous points (orange).
    """
    print("\n=== RESULTS KEY PLOT ===")

    region_config = [
        ("aorta_points",     aorta_points,     [255, 255,   0, 255], "Yellow  = Aorta"),
        ("rca_points",       rca_points,       [  0,   0, 255, 255], "Blue    = RCA"),
        ("lca_points",       lca_points,       [  0, 255,   0, 255], "Green   = LCA"),
        ("rca_removed_points", rca_removed_points, [255, 0, 0, 255], "Red     = Removed"),
        ("proximal_points",  proximal_points,  [  0, 255, 255, 255], "Cyan    = Proximal"),
        ("distal_points",    distal_points,    [255,   0, 255, 255], "Magenta = Distal"),
        ("anomalous_points", anomalous_points, [255, 165,   0, 255], "Orange  = Anomalous"),
    ]

    scene_geoms = []
    for key, enabled, color, label in region_config:
        pts = results.get(key, [])
        print(f"  {label:30s}  n={len(pts):6d}  {'[shown]' if enabled and pts else '[hidden]'}")
        if enabled and pts:
            arr = np.array(pts, dtype=np.float64)
            colors = np.tile(color, (len(pts), 1))
            scene_geoms.append(PointCloud(arr, colors=colors))

    if not scene_geoms:
        print("Nothing to show - all regions are disabled or empty.")
        return
    
    mesh_visual = results["mesh"]
    mesh_visual.visual.face_colors = [200, 200, 200, 100]
    scene_geoms.append(mesh_visual)

    if cl_rca:
        rca = _get_cl_arry(cl_rca)
        scene_geoms.append(PointCloud(rca, colors=[0, 100, 200, 255]))
    if cl_lca:
        lca = _get_cl_arry(cl_lca)
        scene_geoms.append(PointCloud(lca, colors=[0, 150, 0, 255]))
    if cl_aorta:
        ao = _get_cl_arry(cl_aorta)
        scene_geoms.append(PointCloud(ao, colors=[200, 200, 0, 255]))

    scene = trimesh.Scene(scene_geoms)
    scene.show()


def _get_cl_arry(cl: PyCenterline) -> np.array:
    cl = np.array(
        [
            (p.contour_point.x, p.contour_point.y, p.contour_point.z)
            for p in cl.points
        ],
        dtype=np.float64,
    )
    return cl


def compare_centerline_scaling(
    original_mesh: trimesh.Trimesh,
    scaled_mesh: trimesh.Trimesh,
    region_points: list,
    centerline=None,
):
    """Open three interactive scenes comparing the original and scaled meshes.

    * **Scene 1** - original mesh with the scaled region highlighted in red and
      the centerline in cyan.
    * **Scene 2** - scaled mesh with the same overlays.
    * **Scene 3** - side-by-side view: original (blue-ish) shifted 150 mm along
      the x-axis next to the scaled mesh (orange-ish).

    Parameters
    ----------
    original_mesh : trimesh.Trimesh
        Mesh before scaling.
    scaled_mesh : trimesh.Trimesh
        Mesh after scaling (e.g. from
        :func:`~multimodars.ccta.adjust_ccta.scale_region_centerline_morphing`).
    region_points : list of tuple
        ``(x, y, z)`` coordinates of the scaled region, highlighted in red.
    centerline : PyCenterline, optional
        Centerline overlaid in cyan; skipped when ``None``.
    """
    print(f"\n=== CENTERLINE SCALING COMPARISON ===")

    region_array = np.array(region_points, dtype=np.float64)
    region_colors = np.tile([255, 0, 0, 255], (len(region_points), 1))  # Red

    # Scene 1: Original mesh with region highlighted
    scene1 = trimesh.Scene(
        [original_mesh, PointCloud(region_array, colors=region_colors)]
    )
    original_mesh.visual.face_colors = [200, 200, 200, 100]  # Semi-transparent

    centerline_points = None
    centerline_colors = None
    if centerline is not None:
        centerline_points = np.array(
            [
                (p.contour_point.x, p.contour_point.y, p.contour_point.z)
                for p in centerline.points
            ],
            dtype=np.float64,
        )
        centerline_colors = np.tile(
            [0, 255, 255, 255], (len(centerline_points), 1)
        )  # Cyan
        scene1.add_geometry(PointCloud(centerline_points, colors=centerline_colors))

    print("Showing Scene 1: Original mesh with RCA region (red) and centerline (cyan)")
    scene1.show()

    # Scene 2: Scaled mesh with region highlighted
    scene2 = trimesh.Scene(
        [scaled_mesh, PointCloud(region_array, colors=region_colors)]
    )
    scaled_mesh.visual.face_colors = [200, 200, 200, 100]  # Semi-transparent

    if centerline_points is not None:
        scene2.add_geometry(PointCloud(centerline_points, colors=centerline_colors))

    print("Showing Scene 2: Scaled mesh with RCA region (red) and centerline (cyan)")
    scene2.show()

    # Scene 3: Side-by-side comparison
    scaled_mesh_shifted = scaled_mesh.copy()
    shift_amount = np.array([150, 0, 0])  # Adjust based on your mesh size
    scaled_mesh_shifted.apply_translation(shift_amount)

    scene3 = trimesh.Scene([original_mesh, scaled_mesh_shifted])

    original_mesh.visual.face_colors = [0, 100, 200, 100]  # Blue-ish
    scaled_mesh_shifted.visual.face_colors = [200, 100, 0, 100]  # Orange-ish

    if centerline_points is not None:
        centerline_shifted = centerline_points + shift_amount
        scene3.add_geometry(PointCloud(centerline_shifted, colors=centerline_colors))

    print("Showing Scene 3: Side-by-side comparison (Blue=Original, Orange=Scaled)")
    scene3.show()
