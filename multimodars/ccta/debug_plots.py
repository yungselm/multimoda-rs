from __future__ import annotations

import numpy as np
import trimesh
from trimesh.points import PointCloud


def labeled_geometry_plot(
    mesh: trimesh.Trimesh,
    rca_points: list,
    lca_points: list,
    rca_removed_points: list,
    lca_removed_points: list,
    cl_rca=None,
    cl_lca=None,
    cl_aorta=None,
):
    """Open an interactive 3-D scene visualising the labelled mesh regions.

    Colour coding:

    * **Yellow** - aortic vertices (not in RCA or LCA).
    * **Blue** - RCA coronary vertices.
    * **Green** - LCA coronary vertices.
    * **Red** - vertices removed by occlusion detection.
    * **Dark blue / dark green / dark yellow** - RCA / LCA / aorta centerline
      points.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The surface mesh displayed semi-transparently in the background.
    rca_points : list of tuple
        RCA vertex coordinates.
    lca_points : list of tuple
        LCA vertex coordinates.
    rca_removed_points : list of tuple
        RCA vertices that were flagged by occlusion removal.
    lca_removed_points : list of tuple
        LCA vertices that were flagged by occlusion removal.
    cl_rca : PyCenterline, optional
        RCA centerline object; plotted when provided.
    cl_lca : PyCenterline, optional
        LCA centerline object; plotted when provided.
    cl_aorta : PyCenterline, optional
        Aorta centerline object; plotted when provided.
    """
    rca_set = set(rca_points)
    lca_set = set(lca_points)
    aortic_points = [
        tuple(vertex)
        for vertex in mesh.vertices
        if tuple(vertex) not in rca_set and tuple(vertex) not in lca_set
    ]

    print(f"\n=== DEBUG PLOT STATISTICS ===")
    print(f"Total mesh vertices: {len(mesh.vertices)}")
    print(f"Aortic points (yellow): {len(aortic_points)}")
    print(f"RCA coronary points (blue): {len(rca_points)}")
    print(f"LCA coronary points (green): {len(lca_points)}")
    print(f"RCA reassigned points (red): {len(rca_removed_points)}")
    print(f"LCA reassigned points (red): {len(lca_removed_points)}")

    scene_geoms = []

    # Add mesh (semi-transparent)
    mesh_visual = mesh.copy()
    mesh_visual.visual.face_colors = [200, 200, 200, 100]  # Semi-transparent gray
    scene_geoms.append(mesh_visual)

    if aortic_points:
        aortic_array = np.array(aortic_points, dtype=np.float64)
        aortic_colors = np.tile([255, 255, 0, 255], (len(aortic_points), 1))  # Yellow
        scene_geoms.append(PointCloud(aortic_array, colors=aortic_colors))

    if rca_points:
        rca_array = np.array(rca_points, dtype=np.float64)
        rca_colors = np.tile([0, 0, 255, 255], (len(rca_points), 1))  # Blue
        scene_geoms.append(PointCloud(rca_array, colors=rca_colors))

    if lca_points:
        lca_array = np.array(lca_points, dtype=np.float64)
        lca_colors = np.tile([0, 255, 0, 255], (len(lca_points), 1))  # Green
        scene_geoms.append(PointCloud(lca_array, colors=lca_colors))

    all_removed = rca_removed_points + lca_removed_points
    if all_removed:
        removed_array = np.array(all_removed, dtype=np.float64)
        removed_colors = np.tile([255, 0, 0, 255], (len(all_removed), 1))  # Red
        scene_geoms.append(PointCloud(removed_array, colors=removed_colors))

    if cl_rca is not None:
        rca_centerline_points = np.array(
            [
                (p.contour_point.x, p.contour_point.y, p.contour_point.z)
                for p in cl_rca.points
            ],
            dtype=np.float64,
        )
        scene_geoms.append(
            PointCloud(rca_centerline_points, colors=[0, 100, 200, 255])
        )  # Dark blue

    if cl_lca is not None:
        lca_centerline_points = np.array(
            [
                (p.contour_point.x, p.contour_point.y, p.contour_point.z)
                for p in cl_lca.points
            ],
            dtype=np.float64,
        )
        scene_geoms.append(
            PointCloud(lca_centerline_points, colors=[0, 150, 0, 255])
        )  # Dark green

    if cl_aorta is not None:
        aorta_centerline_points = np.array(
            [
                (p.contour_point.x, p.contour_point.y, p.contour_point.z)
                for p in cl_aorta.points
            ],
            dtype=np.float64,
        )
        scene_geoms.append(
            PointCloud(aorta_centerline_points, colors=[200, 200, 0, 255])
        )  # Dark yellow

    scene1 = trimesh.Scene(scene_geoms)

    print("\nShowing Scene 1: Mesh with colored points")
    print("Colors: Yellow=Aorta, Blue=RCA, Green=LCA, Red=Removed")
    scene1.show()


def plot_anomalous_region(results, centerline):
    """Open an interactive 3-D scene visualising the anomalous region sub-classes.

    Colour coding:

    * **Blue** - proximal points.
    * **Red** - anomalous (intramural) points.
    * **Green** - distal points.
    * **Yellow** - centerline points.

    Parameters
    ----------
    results : dict
        Dictionary containing ``"proximal_points"``, ``"anomalous_points"``,
        and ``"distal_points"`` keys (as returned by
        :func:`~multimodars.ccta.adjust_ccta.label_anomalous_region`).
    centerline : PyCenterline or None
        Centerline to overlay; skipped when ``None``.
    """
    print(f"\n=== ANOMALOUS REGION VISUALIZATION ===")

    scene_geoms = []

    if results["proximal_points"]:
        proximal_array = np.array(results["proximal_points"])
        proximal_cloud = PointCloud(proximal_array, colors=[0, 0, 255, 255])  # Blue
        scene_geoms.append(proximal_cloud)

    if results["anomalous_points"]:
        anomalous_array = np.array(results["anomalous_points"])
        anomalous_cloud = PointCloud(anomalous_array, colors=[255, 0, 0, 255])  # Red
        scene_geoms.append(anomalous_cloud)

    if results["distal_points"]:
        distal_array = np.array(results["distal_points"])
        distal_cloud = PointCloud(distal_array, colors=[0, 255, 0, 255])  # Green
        scene_geoms.append(distal_cloud)

    if centerline is not None:
        centerline_points = np.array(
            [
                (p.contour_point.x, p.contour_point.y, p.contour_point.z)
                for p in centerline.points
            ],
            dtype=np.float64,
        )
        centerline_cloud = PointCloud(
            centerline_points, colors=[255, 255, 0, 255]
        )  # Yellow
        scene_geoms.append(centerline_cloud)

    scene = trimesh.Scene(scene_geoms)
    scene.show()


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
