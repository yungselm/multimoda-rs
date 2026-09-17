from __future__ import annotations

import numpy as np
import trimesh

from ...multimodars import PyGeometry
from ..._converters import geometry_to_trimesh
from .boundary import (
    _adjust_start_point_by_z,
    _fix_ring_direction_by_distance,
    _fix_ring_direction_by_winding,
    _prepare_prox_dist_boundary_pts,
    _rotate_to_nearest_iv,
)
from .helpers import _fast_fix_normals


def stitch_ccta_to_intravascular(
    iv_mesh: PyGeometry,
    mesh: trimesh.Trimesh,
    results: dict,
    n_points_iv_cont: int = 100,
    prox_start_mode: str = "nearest_iv",
    dist_start_mode: str = "nearest_iv",
    proximal_is_ostium: bool = True,
    clamp_overshoot: float = 0.5,
    boundary_point_ratio: float = 1.0,
) -> dict:
    """Stitch an aligned intravascular mesh to a CCTA mesh.

    *results* must carry two boundary rings (see
    :func:`~multimodars.ccta.mesh_regions.remove_labeled_points_from_mesh` with
    ``target_boundaries=2``).  Each ring is assigned to an IV end as a whole, by
    whichever pairing of rings to the proximal and distal frame centroids is
    closest overall.

    ``prox_start_mode`` / ``dist_start_mode`` control how index 0 of each
    boundary ring is chosen before stitching:

    * ``"nearest_iv"`` (default) - rotate to the point closest to IV point 0.
    * ``"highest_z"`` - rotate to the point with the largest z-coordinate.

    ``clamp_overshoot`` sets the minimum distance (mm) that every proximal
    boundary point must sit away from the IV plane after clamping.  Points
    that land too close are pushed further until they are exactly
    ``clamp_overshoot`` mm from the plane, creating a slight inward step that
    softens the stitching angle.  The two mesh rings adjacent to the boundary
    are also pushed radially outward (ring 1: 0.1 mm, ring 2: 0.2 mm) within
    the IV plane to avoid ridges at the clamping zone.  Only active when the
    boundary-ring plane and the IV plane form an angle ≥ ``ostium_angle_threshold_deg``
    (default 45°).
    """
    iv_mesh = iv_mesh.downsample(n_points_iv_cont)
    iv_mesh_points = [
        (p.x, p.y, p.z) for frame in iv_mesh.frames for p in frame.lumen.points
    ]
    proximal_centroid = iv_mesh.frames[0].centroid
    distal_centroid = iv_mesh.frames[-1].centroid
    proximal_points = iv_mesh.frames[0].lumen.points
    distal_points = iv_mesh.frames[-1].lumen.points

    # Vessel axis: outward for the proximal patch points toward frames[0], and
    # vice-versa for the distal patch.  Needed before the boundary prep, since
    # the ostium plane is slid along the proximal outward direction.
    prox_c = np.array(iv_mesh.frames[0].centroid)
    dist_c = np.array(iv_mesh.frames[-1].centroid)
    prox_outward = prox_c - dist_c  # points toward the proximal end
    dist_outward = dist_c - prox_c  # points toward the distal end

    target_n = max(3, round(boundary_point_ratio * len(proximal_points)))

    prox_boundary_pts, dist_boundary_pts, mesh = _prepare_prox_dist_boundary_pts(
        mesh,
        results,
        proximal_centroid,
        distal_centroid,
        proximal_is_ostium=proximal_is_ostium,
        proximal_iv_frame_pts=iv_mesh.frames[0].lumen.points,
        clamp_overshoot=clamp_overshoot,
        target_n=target_n,
        prox_outward=prox_outward,
        prox_start_mode=prox_start_mode,
        dist_start_mode=dist_start_mode,
    )
    prox_point_step = max(1, len(proximal_points) // len(prox_boundary_pts))
    dist_point_step = max(1, len(distal_points) // len(dist_boundary_pts))

    # Adjust start point
    if prox_start_mode == "highest_z" or dist_start_mode == "highest_z":
        iv_mesh = iv_mesh.sort_frame_points()
        proximal_points = iv_mesh.frames[0].lumen.points
        distal_points = iv_mesh.frames[-1].lumen.points
    if prox_start_mode == "highest_z":
        prox_boundary_pts = _adjust_start_point_by_z(prox_boundary_pts)
    else:
        prox_boundary_pts = _rotate_to_nearest_iv(prox_boundary_pts, proximal_points[0])
    if dist_start_mode == "highest_z":
        dist_boundary_pts = _adjust_start_point_by_z(dist_boundary_pts)
    else:
        dist_boundary_pts = _rotate_to_nearest_iv(dist_boundary_pts, distal_points[0])

    # Check / fix winding direction of each boundary ring vs its IV ring
    # independently, using the method that matches the start-point strategy.
    if prox_start_mode == "highest_z":
        prox_boundary_pts = _fix_ring_direction_by_winding(
            prox_boundary_pts, proximal_points
        )
    else:
        prox_boundary_pts = _fix_ring_direction_by_distance(
            prox_boundary_pts, proximal_points, prox_point_step
        )

    if dist_start_mode == "highest_z":
        dist_boundary_pts = _fix_ring_direction_by_winding(
            dist_boundary_pts, distal_points
        )
    else:
        dist_boundary_pts = _fix_ring_direction_by_distance(
            dist_boundary_pts, distal_points, dist_point_step
        )

    # Step 3: stitch each boundary ring to its IV ring
    prox_patch = _stitch_rings(prox_boundary_pts, proximal_points, prox_outward)
    dist_patch = _stitch_rings(dist_boundary_pts, distal_points, dist_outward)
    test_mesh = geometry_to_trimesh(iv_mesh)
    test_mesh.update_faces(test_mesh.unique_faces())
    test_mesh.update_faces(test_mesh.nondegenerate_faces())
    _fast_fix_normals(test_mesh)
    mesh = trimesh.util.concatenate([mesh, prox_patch, dist_patch, test_mesh])
    trimesh.tol.merge = 0.001
    mesh.merge_vertices()
    if not mesh.is_watertight:
        mesh.fill_holes()
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    _fast_fix_normals(mesh)

    results["prox_boundary_points"] = prox_boundary_pts
    results["dist_boundary_points"] = dist_boundary_pts
    results["anomalous_points"] = iv_mesh_points
    results["rca_points"] = (
        iv_mesh_points + results["distal_points"] + results["proximal_points"]
    )
    results["mesh"] = mesh

    return results


def _stitch_rings(
    boundary_pts: list,
    iv_pts,
    outward_direction: np.ndarray | None = None,
) -> trimesh.Trimesh:
    """Stitch an IV lumen ring to a CCTA boundary ring as a closed triangle strip.

    Walks both rings together, each step advancing whichever ring is further
    behind in normalised perimeter position and emitting one triangle for that
    advance.  This produces exactly ``len(boundary_pts) + len(iv_pts)``
    triangles - a complete annulus with no gaps - for any ratio between the two
    counts.  Equal counts give the obvious quad strip (two triangles per
    segment); unequal counts spread the extra triangles evenly around the ring
    instead of bunching them.

    Parameters
    ----------
    boundary_pts : list of tuple
        Ordered CCTA boundary vertices.
    iv_pts : list of Point
        Ordered IV lumen points (with ``.x`` / ``.y`` / ``.z``).
    outward_direction : np.ndarray, optional
        Vessel-axis direction that should point outward for this patch; the
        whole patch is flipped when its average normal disagrees.

    Returns
    -------
    trimesh.Trimesh
        Patch mesh with the boundary vertices first, then the IV vertices.
    """
    n_b = len(boundary_pts)
    n_iv = len(iv_pts)
    if n_b < 3 or n_iv < 3:
        raise ValueError(
            f"Need at least 3 points per ring to stitch (got boundary={n_b}, iv={n_iv})."
        )

    b_arr = np.asarray(boundary_pts, dtype=np.float64)
    iv_arr = np.array([(p.x, p.y, p.z) for p in iv_pts], dtype=np.float64)
    vertices = np.vstack([b_arr, iv_arr])

    faces: list[tuple[int, int, int]] = []
    i = j = 0
    while i < n_b or j < n_iv:
        # Advance whichever ring is behind; ties go to the boundary ring.
        take_boundary = j >= n_iv or (i < n_b and (i + 1) / n_b <= (j + 1) / n_iv)
        if take_boundary:
            faces.append((i % n_b, (i + 1) % n_b, n_b + j % n_iv))
            i += 1
        else:
            faces.append((i % n_b, n_b + (j + 1) % n_iv, n_b + j % n_iv))
            j += 1

    patch = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )

    if outward_direction is not None:
        # The strip is internally consistent, but may face inward as a whole -
        # the proximal IV lumen winds opposite the distal one seen from a fixed
        # direction.  For a roughly flat annulus the average normal is a reliable
        # indicator, so compare it against the known outward axis.
        face_normals = patch.face_normals
        valid = ~np.isnan(face_normals).any(axis=1)
        if (
            valid.any()
            and np.dot(face_normals[valid].mean(axis=0), outward_direction) < 0
        ):
            patch.faces = patch.faces[:, ::-1]

    return patch
