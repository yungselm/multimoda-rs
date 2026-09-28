from __future__ import annotations

from typing import cast

import numpy as np
import trimesh

from ..._converters import geometry_to_trimesh
from ...multimodars import PyGeometry
from .boundary import (
    _adjust_start_point_by_z,
    _carry_ring_weights,
    _fix_ring_direction_by_distance,
    _fix_ring_direction_by_winding,
    _prepare_prox_dist_boundary_pts,
    _rotate_to_nearest_iv,
)
from .helpers import _fast_fix_normals, _plane_normal_svd


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
    seam_points_a: int = 2,
    seam_points_b: int = 4,
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

    With ``prox_start_mode="highest_z"`` and ``proximal_is_ostium``, the
    ostial ring is conditioned as two halves - an aorta-facing Half A offset
    from the IV ostium by the aortic wall thickness, and a coronary-facing
    Half B (see :func:`~.boundary._condition_ostium_ring_two_half`).
    ``seam_points_a`` / ``seam_points_b`` set how many Half A / Half B points
    either side of each of the two seams where the halves meet are replaced
    by a smooth arc - the same counts at both seams, counted on the CCTA ring
    as cut from the mesh (before it is densified to the IV point count).
    Everything outside those ranges stays exactly in place, so Half A's
    mid-section keeps its distance from the IV ostium.  ``0`` / ``0`` leaves
    the seams sharp.

    ``fillet_bulge`` > 0 rounds that ostial Half A seam - the strip between
    each CCTA boundary point and its IV point - into a small arc instead of a
    sharp, direct strut (see :func:`_stitch_rings_rounded`).  The arcs bulge
    away from the IV geometry and fade out along the Half A / Half B seam
    arcs, so Half B and the distal seam always stay direct strips.  It is
    purely cosmetic - neither endpoint moves, so the aortic-thickness
    correction stays exact - and only applies with the two-half ostium
    above.  It also needs the boundary ring and IV ring to have the same
    point count (true by default, since ``boundary_point_ratio=1.0``);
    otherwise the direct strip is used.  ``0`` (default) keeps the direct
    strip.
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

    prox_boundary_pts, dist_boundary_pts, mesh, prox_half_a_weight = (
        _prepare_prox_dist_boundary_pts(
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
            seam_points_a=seam_points_a,
            seam_points_b=seam_points_b,
        )
    )
    conditioned_prox = prox_boundary_pts
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

    # Step 3: stitch each boundary ring to its IV ring.  The fillet only ever
    # rounds the ostium's Half A, so the distal seam is always a direct strip.
    prox_fillet_weight = None
    if prox_half_a_weight is not None:
        prox_fillet_weight = _carry_ring_weights(
            conditioned_prox, prox_half_a_weight, prox_boundary_pts
        )
    elif fillet_bulge > 0.0:
        print(
            "Warning: fillet_bulge only rounds the ostium's Half A, which needs "
            "prox_start_mode='highest_z' and proximal_is_ostium=True (and a ring "
            "that splits into two halves); stitching without it."
        )
    prox_patch = _stitch_rings(
        prox_boundary_pts,
        proximal_points,
        prox_outward,
        fillet_bulge=fillet_bulge if prox_fillet_weight is not None else 0.0,
        fillet_layers=fillet_layers,
        fillet_weight=prox_fillet_weight,
        fillet_direction=_ostium_away_direction(iv_mesh),
    )
    dist_patch = _stitch_rings(dist_boundary_pts, distal_points, dist_outward)
    test_mesh = geometry_to_trimesh(iv_mesh)
    test_mesh.update_faces(test_mesh.unique_faces())
    test_mesh.update_faces(test_mesh.nondegenerate_faces())
    _fast_fix_normals(test_mesh)
    # Concatenating Trimeshes gives a Trimesh; trimesh only types it as Geometry.
    mesh = cast(
        trimesh.Trimesh,
        trimesh.util.concatenate([mesh, prox_patch, dist_patch, test_mesh]),
    )
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


def _ostium_away_direction(iv_mesh: PyGeometry) -> np.ndarray:
    """Unit normal of the IV ostial frame, pointing away from the rest of the
    IV geometry - the side the ostial fillet bulges to, so it rounds the seam
    off toward the aorta instead of folding back into the vessel.
    """
    frame = iv_mesh.frames[0]
    pts = np.array([(p.x, p.y, p.z) for p in frame.lumen.points], dtype=np.float64)
    normal = _plane_normal_svd(pts)
    rest = np.array([f.centroid for f in iv_mesh.frames[1:]], dtype=np.float64)
    if len(rest) and np.dot(normal, np.asarray(frame.centroid) - rest.mean(axis=0)) < 0:
        normal = -normal
    return normal


def _stitch_rings(
    boundary_pts: list,
    iv_pts,
    outward_direction: np.ndarray | None = None,
    fillet_bulge: float = 0.0,
    fillet_layers: int = 2,
    fillet_weight: np.ndarray | None = None,
    fillet_direction: np.ndarray | None = None,
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
    fillet_weight : np.ndarray, optional
        How much of *fillet_bulge* each strut gets, one value per boundary
        point: strut ``i`` bulges ``fillet_bulge * fillet_weight[i]`` of its
        length (``0`` = straight, ``1`` = the full bulge).
        :func:`stitch_ccta_to_intravascular` passes the ostium's Half A
        weights here - that is what confines the fillet to Half A.  ``None``
        gives every strut the full bulge.
    fillet_direction : np.ndarray, optional
        The side the arcs bulge to (see :func:`_fillet_arc_layers`).
        Defaults to *outward_direction*; with neither, the direct strip is
        used.

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

    direction = outward_direction if fillet_direction is None else fillet_direction
    weight = np.ones(n_b) if fillet_weight is None else np.asarray(fillet_weight, float)
    if fillet_bulge > 0.0 and n_b == n_iv and direction is not None and weight.any():
        return _stitch_rings_rounded(
            boundary_pts,
            iv_pts,
            outward_direction,
            bulge_fraction=fillet_bulge * weight,
            n_layers=fillet_layers,
            bulge_direction=direction,
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


def _fillet_arc_layers(
    start: np.ndarray,
    end: np.ndarray,
    bulge_direction: np.ndarray,
    bulge_fraction: np.ndarray,
    n_layers: int,
) -> np.ndarray:
    """The *n_layers* intermediate rings of a rounded seam from *start* to *end*.

    Each pair ``start[i] -> end[i]`` gets its own arc, used to round the
    direct seam between a CCTA boundary point and its IV point into a small
    fillet instead of a sharp straight strut.  The arc bulges perpendicular
    to the pair's chord, on the side *bulge_direction* points to (its part
    perpendicular to the chord, so the side follows each chord's own
    orientation), with a sine profile: zero at both endpoints, which stay
    exactly where they are, and ``bulge_fraction[i]`` times the chord length
    at the midpoint.  A chord parallel to *bulge_direction* has no such side
    and stays straight.

    Returns an ``(n_layers, len(start), 3)`` array, ordered from *start* to
    *end*.
    """
    chord = end - start
    length = np.linalg.norm(chord, axis=1)
    axis = np.divide(
        chord, length[:, None], out=np.zeros_like(chord), where=length[:, None] > 1e-9
    )
    away = np.asarray(bulge_direction, dtype=np.float64)
    away = away / np.linalg.norm(away)
    side = away - (axis @ away)[:, None] * axis
    side_len = np.linalg.norm(side, axis=1)
    side = np.divide(
        side, side_len[:, None], out=np.zeros_like(side), where=side_len[:, None] > 1e-6
    )

    t = np.arange(1, n_layers + 1) / (n_layers + 1)
    height = np.sin(np.pi * t)[:, None] * (bulge_fraction * length)[None, :]
    return start[None] + t[:, None, None] * chord[None] + height[..., None] * side[None]


def _stitch_rings_rounded(
    boundary_pts: list,
    iv_pts,
    outward_direction: np.ndarray | None,
    bulge_fraction: np.ndarray,
    n_layers: int,
    bulge_direction: np.ndarray,
) -> trimesh.Trimesh:
    """Like :func:`_stitch_rings`, but with a rounded fillet instead of a
    direct strut between each boundary point and its corresponding IV point.

    Requires ``len(boundary_pts) == len(iv_pts)`` - a 1:1 correspondence, so
    every point pair gets its own arc, bulging by its own entry of
    *bulge_fraction* to the *bulge_direction* side.  Builds *n_layers*
    intermediate rings between the boundary ring and the IV ring (see
    :func:`_fillet_arc_layers`) and triangulates each consecutive pair of
    rings as an ordinary quad strip, so the two original rings' points stay
    exactly where they were (the aortic-thickness correction is untouched)
    and only the surface between them curves.
    """
    n = len(boundary_pts)
    b_arr = np.asarray(boundary_pts, dtype=np.float64)
    iv_arr = np.array([(p.x, p.y, p.z) for p in iv_pts], dtype=np.float64)

    arcs = _fillet_arc_layers(b_arr, iv_arr, bulge_direction, bulge_fraction, n_layers)
    rings = [b_arr, *arcs, iv_arr]
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
