from __future__ import annotations

import numpy as np
import trimesh

from ...multimodars import build_adjacency_map, fix_mesh_winding


def _project_to_best_fit_plane(
    points: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    """Project a ring of boundary points onto their best-fit plane.

    Fits a plane via SVD (the plane normal is the direction of minimum variance)
    and orthogonally projects every point onto it, flattening noise perpendicular
    to the ring.
    """
    if len(points) < 3:
        return points
    pts = np.array(points, dtype=np.float64)
    centroid = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - centroid, full_matrices=False)
    normal = Vt[-1]
    distances = (pts - centroid) @ normal
    projected = pts - np.outer(distances, normal)
    return [tuple(p) for p in projected]


def _project_onto_plane(
    points: list[tuple[float, float, float]],
    origin: np.ndarray,
    normal: np.ndarray,
) -> list[tuple[float, float, float]]:
    """Orthogonally project *points* onto the plane through *origin*."""
    pts = np.asarray(points, dtype=np.float64)
    return [tuple(p) for p in pts - np.outer((pts - origin) @ normal, normal)]


def _plane_normal_svd(pts: np.ndarray) -> np.ndarray:
    """Best-fit plane normal for a point cloud via SVD (minimum-variance axis)."""
    centroid = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - centroid, full_matrices=False)
    return Vt[-1]


def _angle_between_planes_deg(n1: np.ndarray, n2: np.ndarray) -> float:
    """Acute angle in degrees between two planes given their normals."""
    cos = np.clip(np.abs(np.dot(n1, n2)), 0.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def _clamp_to_plane(
    points: list[tuple[float, float, float]],
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    overshoot: float = 0.0,
) -> list[tuple[float, float, float]]:
    """Clamp wrong-side points to the IV plane, then enforce a minimum gap.

    Step 1: project any point on the wrong side of the plane onto it.
    Step 2: if ``overshoot`` > 0, every point (including freshly clamped ones
    that now sit exactly on the plane) that is within ``overshoot`` mm of the
    plane on the correct side is pushed further away until it is exactly
    ``overshoot`` mm from the plane.  This creates a clean buffer zone between
    the aortic boundary ring and the IV ostium plane, avoiding the sharp angle
    that would otherwise form.
    """
    pts = np.array(points, dtype=np.float64)
    dists = (pts - plane_origin) @ plane_normal
    correct_sign = np.sign(np.median(dists))

    # Step 1: project wrong-side points onto the plane
    wrong = (np.sign(dists) != correct_sign) & (dists != 0.0)
    pts[wrong] -= np.outer(dists[wrong], plane_normal)

    if overshoot > 0.0:
        # Step 2: recompute distances and push any point within the buffer zone
        # further away on the aortic (correct) side
        dists2 = (pts - plane_origin) @ plane_normal
        signed_dist = correct_sign * dists2  # positive = on correct side
        too_close = signed_dist < overshoot
        deficit = overshoot - signed_dist[too_close]
        pts[too_close] += np.outer(deficit * correct_sign, plane_normal)

    return [tuple(p) for p in pts]


def _shift_plane_clear_of(
    origin: np.ndarray,
    normal: np.ndarray,
    points,
    outward: np.ndarray,
    overshoot: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Slide a plane along its normal until it clears a set of points.

    The boundary plane of an anomalous ostium can cut straight through the IV
    ostial frame.  This re-orients *normal* to point along *outward* (toward the
    aorta), then translates the plane far enough that every point in *points*
    sits at least *overshoot* mm behind it.

    Returns the shifted origin, the outward-oriented normal, and how far the
    plane moved (``0.0`` when it already cleared the points).
    """
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    if float(np.dot(n, np.asarray(outward, dtype=np.float64))) < 0.0:
        n = -n
    o = np.asarray(origin, dtype=np.float64)

    signed = (np.asarray(points, dtype=np.float64) - o) @ n
    worst = float(signed.max())
    if worst <= -overshoot:
        return o, n, 0.0
    shift = worst + overshoot
    return o + shift * n, n, shift


def _write_ring_to_mesh(
    mesh: trimesh.Trimesh,
    old_pts: list,
    new_pts: list,
) -> tuple[trimesh.Trimesh, set[int]]:
    """Move the mesh vertices sitting at *old_pts* to *new_pts*.

    Returns the rebuilt mesh and the indices that actually moved, which callers
    need as seeds for :func:`_enforce_layer_gap_from_plane`.
    """
    coord_to_idx = {tuple(v): i for i, v in enumerate(mesh.vertices)}
    verts = mesh.vertices.copy()
    moved: set[int] = set()
    for old, new in zip(old_pts, new_pts):
        idx = coord_to_idx.get(tuple(old))
        if idx is not None:
            verts[idx] = new
            moved.add(idx)
    return trimesh.Trimesh(vertices=verts, faces=mesh.faces, process=False), moved


def _enforce_layer_gap_from_plane(
    mesh: trimesh.Trimesh,
    seed_indices: set[int],
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    layer_step_mm: float = 0.1,
    n_rings: int = 2,
) -> trimesh.Trimesh:
    """Push neighboring mesh rings radially away from the IV ring center.

    The boundary ring was clamped toward the IV plane, which can leave second-
    and third-layer aortic vertices sitting closer to the coronary axis than
    the boundary ring itself — creating a visible ridge.  The fix is to push
    those vertices outward *within* the IV plane (i.e. along the aortic
    surface, away from the coronary center), not perpendicular to it.

    Ring k is displaced by ``k * layer_step_mm`` in the radial direction:
    the projection of the vertex onto the IV plane, measured from the IV
    ring centre (``plane_origin``), gives the outward direction.
    """
    adj_map = build_adjacency_map(mesh.faces.tolist())
    new_vertices = mesh.vertices.copy()

    frontier = set(seed_indices)
    visited = set(seed_indices)

    for ring in range(1, n_rings + 1):
        push_dist = ring * layer_step_mm
        next_frontier = set()
        for vi in frontier:
            for nb in adj_map.get(vi, []):
                if nb not in visited:
                    next_frontier.add(nb)

        for vi in next_frontier:
            p = new_vertices[vi]
            # Project the vertex onto the IV plane to get its lateral position
            p_proj = p - float(np.dot(p - plane_origin, plane_normal)) * plane_normal
            # Radial direction: from IV ring centre outward, within the IV plane
            radial = p_proj - plane_origin
            r_norm = np.linalg.norm(radial)
            if r_norm < 1e-10:
                continue
            new_vertices[vi] = p + (push_dist / r_norm) * radial

        visited.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break

    return trimesh.Trimesh(vertices=new_vertices, faces=mesh.faces, process=False)


def _fast_fix_normals(mesh: trimesh.Trimesh) -> None:
    """Drop-in replacement for ``trimesh.Trimesh.fix_normals()``.

    trimesh's own ``fix_winding`` does a Python/NetworkX BFS over the
    face-adjacency graph with several small numpy allocations per edge -
    O(n_edges) with heavy per-iteration overhead (e.g. ~3.9s on a ~52k-face
    mesh). ``fix_mesh_winding`` is a Rust port of the same BFS-consistency
    algorithm; ``fix_inversion`` (the volume-sign flip check) is already
    vectorized numpy in trimesh, so it's left as-is.
    """
    mesh.faces = np.array(fix_mesh_winding(mesh.faces.tolist()), dtype=mesh.faces.dtype)
    trimesh.repair.fix_inversion(mesh, multibody=False)


def _signed_area_projected(pts: list, normal: np.ndarray) -> float:
    """Signed area of a polygon projected onto the plane with the given normal.

    Positive = CCW when viewed from the normal direction.
    """
    ref = (
        np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    )
    u = np.cross(normal, ref)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    arr = np.array(pts)
    us = arr @ u
    vs = arr @ v
    return float(0.5 * np.sum(us * np.roll(vs, -1) - np.roll(us, -1) * vs))


def _newell_normal(pts: list) -> np.ndarray:
    """Compute a polygon's outward normal via Newell's method.

    The resulting normal points in the direction from which the polygon
    appears CCW — so ``_signed_area_projected(pts, _newell_normal(pts))``
    is always positive for any non-degenerate polygon.
    """
    normal = np.zeros(3)
    n = len(pts)
    arr = np.array(pts)
    for i in range(n):
        c = arr[i]
        nx = arr[(i + 1) % n]
        normal[0] += (c[1] - nx[1]) * (c[2] + nx[2])
        normal[1] += (c[2] - nx[2]) * (c[0] + nx[0])
        normal[2] += (c[0] - nx[0]) * (c[1] + nx[1])
    length = np.linalg.norm(normal)
    return normal / length if length > 1e-10 else np.array([0.0, 0.0, 1.0])
