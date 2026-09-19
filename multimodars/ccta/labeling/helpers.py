from __future__ import annotations

import trimesh

from ...multimodars import (
    find_faces_near_points,
    remove_occluded_points_ray_triangle,
    keep_largest_connected_component,
    PyCenterline,
)


def _apply_occlusion_removal(
    range_mm_takeoff_rca: float,
    step_size_mm: float,
    tolerance_float: float,
    cl_aorta: PyCenterline,
    cl_rca: PyCenterline,
    points_list: list[tuple[float, float, float]],
    mesh_faces_list: list[list[int]],
    rca_points_found: list[tuple[float, float, float]],
    name: str,
):
    print(f"Applying occlusion removal for acute-takeoff {name}...")
    rca_faces_for_rust = find_faces_near_points(
        points_list, mesh_faces_list, rca_points_found, tolerance_float
    )
    final_rca_points_found = remove_occluded_points_ray_triangle(
        centerline_coronary=cl_rca,
        centerline_aorta=cl_aorta,
        range_mm=range_mm_takeoff_rca,
        points=rca_points_found,
        faces=rca_faces_for_rust,
        step_size_mm=step_size_mm,
    )
    final_rca_points_found_set = set(final_rca_points_found)
    rca_removed_points = [
        p for p in rca_points_found if p not in final_rca_points_found_set
    ]
    print(f"{name}: relabeled {len(rca_removed_points)} points in intramual course")
    return rca_removed_points, final_rca_points_found


def _keep_largest_connected_component(
    mesh: trimesh.Trimesh, points: list[tuple[float, float, float]]
) -> list[tuple[float, float, float]]:
    """Keep only the largest mesh-connected component of *points*.

    ``find_points_by_cl_region`` classifies points using coordinate-only
    heuristics (nearest 3-D centerline point, axis-aligned proximal/distal
    split) with no notion of mesh topology.  That can leave a handful of
    points assigned to a region despite not being mesh-connected to its main
    cluster ("islands") - e.g. a point geometrically close to the anomalous
    segment but on a different, unconnected part of the vessel surface.

    This restricts the mesh's face-adjacency graph to *points* and keeps
    only the single largest connected component; the rest are dropped
    (callers typically let dropped points fall through to a different
    region via a complement/set-difference step, mirroring how
    :func:`final_reclassification` reassigns isolated vertices to aorta).
    """
    return keep_largest_connected_component(
        [tuple(v) for v in mesh.vertices], mesh.faces.tolist(), points
    )
