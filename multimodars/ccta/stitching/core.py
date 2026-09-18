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
    fillet_bulge: float = 0.0,
    fillet_layers: int = 2,
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

    ``fillet_bulge`` > 0 replaces the sharp, direct seam between each CCTA
    boundary point and its corresponding IV point with a small rounded arc
    (see :func:`_stitch_rings_rounded`), purely cosmetic - it doesn't move
    either endpoint, so the aortic-thickness correction stays exact.  ``0``
    (default) keeps the original direct strip.  Only used when the boundary
    ring and IV ring have the same point count (true by default, since
    ``boundary_point_ratio=1.0``); otherwise falls back to the direct strip.
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
        proximal_aortic_thickness=iv_mesh.frames[0].lumen.aortic_thickness,
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
    prox_patch = _stitch_rings(
        prox_boundary_pts,
        proximal_points,
        prox_outward,
        fillet_bulge=fillet_bulge,
        fillet_layers=fillet_layers,
    )
    dist_patch = _stitch_rings(
        dist_boundary_pts,
        distal_points,
        dist_outward,
        fillet_bulge=fillet_bulge,
        fillet_layers=fillet_layers,
    )
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
    fillet_bulge: float = 0.0,
    fillet_layers: int = 2,
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
    fillet_bulge : float, optional
        When > 0 and the two rings have equal point count, delegates to
        :func:`_stitch_rings_rounded` for a cosmetically rounded seam instead
        of the sharp direct strip.  ``0`` (default) always uses the direct
        strip below.
    fillet_layers : int, optional
        Passed through to :func:`_stitch_rings_rounded`.

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

    if fillet_bulge > 0.0 and n_b == n_iv:
        return _stitch_rings_rounded(
            boundary_pts,
            iv_pts,
            outward_direction,
            bulge_fraction=fillet_bulge,
            n_layers=fillet_layers,
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

    _orient_patch(patch, outward_direction)
    return patch


def _orient_patch(patch: trimesh.Trimesh, outward_direction: np.ndarray | None) -> None:
    """Flip *patch*'s faces in place if its average normal disagrees with
    *outward_direction* - the strip/collar is internally consistent, but may
    face inward as a whole (the proximal IV lumen winds opposite the distal
    one seen from a fixed direction).  For a roughly flat annulus the average
    normal is a reliable indicator, so it's compared against the known
    outward axis.
    """
    if outward_direction is None:
        return
    face_normals = patch.face_normals
    valid = ~np.isnan(face_normals).any(axis=1)
    if valid.any() and np.dot(face_normals[valid].mean(axis=0), outward_direction) < 0:
        patch.faces = patch.faces[:, ::-1]


def _bulge_arc_points(
    p1: np.ndarray,
    p2: np.ndarray,
    centroid: np.ndarray,
    n_layers: int,
    bulge_fraction: float,
) -> list[np.ndarray]:
    """*n_layers* points on a smooth arc from *p1* to *p2*, bulging away from
    *centroid*.

    Used to round the direct seam between a CCTA boundary point and its IV
    point into a small fillet instead of a sharp straight strut.  The bulge
    direction is perpendicular to the ``p1 -> p2`` chord and points away from
    *centroid* (in practice, radially outward from the vessel), with a
    sine profile so it's zero at both endpoints (which stay exactly at *p1*
    and *p2*) and maximal at the midpoint.
    """
    chord = p2 - p1
    chord_len = float(np.linalg.norm(chord))
    if chord_len < 1e-9 or n_layers <= 0:
        return []
    axis = chord / chord_len

    outward = (p1 + p2) / 2.0 - centroid
    outward -= float(np.dot(outward, axis)) * axis
    norm = float(np.linalg.norm(outward))
    if norm < 1e-9:
        ref = (
            np.array([1.0, 0.0, 0.0])
            if abs(axis[0]) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
        outward = np.cross(axis, ref)
        norm = float(np.linalg.norm(outward))
    outward /= norm

    bulge = bulge_fraction * chord_len
    points = []
    for k in range(1, n_layers + 1):
        t = k / (n_layers + 1)
        height = bulge * np.sin(np.pi * t)
        points.append(p1 + t * chord + height * outward)
    return points


def _stitch_rings_rounded(
    boundary_pts: list,
    iv_pts,
    outward_direction: np.ndarray | None,
    bulge_fraction: float,
    n_layers: int,
) -> trimesh.Trimesh:
    """Like :func:`_stitch_rings`, but with a rounded fillet instead of a
    direct strut between each boundary point and its corresponding IV point.

    Requires ``len(boundary_pts) == len(iv_pts)`` - a 1:1 correspondence, so
    every point pair gets its own arc.  Builds *n_layers* intermediate rings
    between the boundary ring and the IV ring (see :func:`_bulge_arc_points`)
    and triangulates each consecutive pair of rings as an ordinary quad
    strip, so the two original rings' points stay exactly where they were
    (the aortic-thickness correction is untouched) and only the surface
    between them curves.
    """
    n = len(boundary_pts)
    b_arr = np.asarray(boundary_pts, dtype=np.float64)
    iv_arr = np.array([(p.x, p.y, p.z) for p in iv_pts], dtype=np.float64)
    centroid = np.vstack([b_arr, iv_arr]).mean(axis=0)

    arc_layers: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
    for i in range(n):
        for k, p in enumerate(
            _bulge_arc_points(b_arr[i], iv_arr[i], centroid, n_layers, bulge_fraction)
        ):
            arc_layers[k].append(p)

    rings = (
        [b_arr]
        + [np.asarray(layer, dtype=np.float64) for layer in arc_layers]
        + [iv_arr]
    )
    vertices = np.vstack(rings)

    faces: list[tuple[int, int, int]] = []
    for layer in range(len(rings) - 1):
        base0 = layer * n
        base1 = (layer + 1) * n
        for i in range(n):
            i2 = (i + 1) % n
            faces.append((base0 + i, base0 + i2, base1 + i))
            faces.append((base0 + i2, base1 + i2, base1 + i))

    patch = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    _orient_patch(patch, outward_direction)
    return patch
