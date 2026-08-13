from __future__ import annotations

import numpy as np
import trimesh

from ..multimodars import (
    PyGeometry,
    build_adjacency_map,
    fix_mesh_winding,
)
from .._converters import geometry_to_trimesh
from .boundary import clean_open_boundary, order_boundary_rings

# Per-ring boundary keys: "boundary_points_1", "boundary_points_2", ...
_BOUNDARY_RING_PREFIX = "boundary_points_"


def _store_boundary_rings(
    updated: dict,
    vertices: np.ndarray,
    rings: list[list[int]],
) -> None:
    """Store *rings* in *updated* as one list per ring, plus the flat total.

    Each ring becomes ``boundary_points_1``, ``boundary_points_2``, ... in walk
    order, so downstream code can treat a rim as a unit instead of guessing the
    split back out of a flat list.  ``boundary_points`` still holds the
    concatenation of all rings for callers that just want every boundary vertex.
    Per-ring keys from an earlier call are cleared first, so a stale ring from a
    previous removal cannot linger and be mistaken for a current one.
    """
    for key in [k for k in updated if k.startswith(_BOUNDARY_RING_PREFIX)]:
        del updated[key]
    per_ring = [[tuple(vertices[i]) for i in ring] for ring in rings]
    for n, pts in enumerate(per_ring, start=1):
        updated[f"{_BOUNDARY_RING_PREFIX}{n}"] = pts
    updated["boundary_points"] = [pt for pts in per_ring for pt in pts]


def _boundary_rings(results: dict, mesh: trimesh.Trimesh) -> list[list[tuple]]:
    """Return the boundary rings stored in *results*, each in walk order.

    Prefers the per-ring ``boundary_points_<n>`` lists written by
    :func:`_store_boundary_rings`.  If only the flat ``boundary_points`` is
    present - a results dict built before ring grouping, or by hand - the rings
    are recovered from the mesh's open edges instead, so callers always get
    properly ordered rings rather than an arbitrary point order.
    """
    rings: list[list[tuple]] = []
    n = 1
    while (key := f"{_BOUNDARY_RING_PREFIX}{n}") in results:
        if results[key]:
            rings.append([tuple(p) for p in results[key]])
        n += 1
    if rings:
        return rings

    flat = results.get("boundary_points") or []
    if not flat:
        return []
    coord_to_idx = {tuple(v): i for i, v in enumerate(mesh.vertices)}
    seeds = {idx for pt in flat if (idx := coord_to_idx.get(tuple(pt))) is not None}
    return [
        [tuple(mesh.vertices[i]) for i in ring]
        for ring in order_boundary_rings(mesh.faces, mesh.vertices, seeds)
    ]


def _assign_rings_to_ends(
    rings: list[list[tuple]],
    prox_centroid: tuple[float, float, float],
    dist_centroid: tuple[float, float, float],
) -> tuple[list[tuple], list[tuple], list[int]]:
    """Choose which whole ring stitches to each end of the intravascular mesh.

    Of every way to hand two distinct rings to the proximal and distal ends, the
    pairing with the smallest total centroid distance wins.  Assigning whole
    rings - rather than each boundary point independently - means a stray point
    that happens to sit nearer the far centroid can no longer tear one ring
    across both seams.

    Returns
    -------
    (list, list, list[int])
        The proximal ring, the distal ring, and the indices of any rings left
        over.
    """
    prox = np.asarray(prox_centroid, dtype=np.float64)
    dist = np.asarray(dist_centroid, dtype=np.float64)
    centroids = [np.asarray(r, dtype=np.float64).mean(axis=0) for r in rings]

    best_cost = float("inf")
    best = (0, 1)
    for i in range(len(rings)):
        for j in range(len(rings)):
            if i == j:
                continue
            cost = float(
                np.linalg.norm(centroids[i] - prox)
                + np.linalg.norm(centroids[j] - dist)
            )
            if cost < best_cost:
                best_cost, best = cost, (i, j)

    i, j = best
    leftover = [k for k in range(len(rings)) if k not in (i, j)]
    return rings[i], rings[j], leftover


def remove_labeled_points_from_mesh(
    results: dict,
    region_keys: list[str] | str = "anomalous_points",
    target_boundaries: int = 1,
) -> dict:
    """Remove one or more labeled regions of vertices from the mesh.

    Collects all points stored under *region_keys*, deletes the corresponding
    vertices (and any faces referencing them) from the mesh, remaps the
    remaining faces, and rebuilds every coordinate list in *results* to
    reflect the new vertex indices.

    Parameters
    ----------
    results : dict
        Dictionary containing at minimum the key ``"mesh"``.  Any of
        ``"aorta_points"``, ``"rca_points"``, ``"lca_points"``,
        ``"rca_removed_points"``, ``"lca_removed_points"``,
        ``"proximal_points"``, and ``"distal_points"`` are also updated if
        present.
    region_keys : str or list of str
        Key(s) in *results* whose point lists should be removed from the mesh.
        Defaults to ``"anomalous_points"`` for backwards compatibility.
    target_boundaries : int
        Number of open-boundary rings the removal is expected to create.
        Removing a single blob leaves one ring (default); removing a region
        that splits the surface (e.g. aorta + intramural wall) can leave two.
        The rim is cleaned and reduced to this many rings before it is stored.

    Returns
    -------
    dict
        Updated *results* dict with ``"mesh"`` replaced by the trimmed mesh,
        all *region_keys* cleared, and all other coordinate lists remapped to
        the new vertex set.  The open boundary exposed by the removal is stored
        both per ring - ``"boundary_points_1"``, ``"boundary_points_2"``, ... in
        walk order - and flattened into ``"boundary_points"``.
    """
    if isinstance(region_keys, str):
        region_keys = [region_keys]

    mesh: trimesh.Trimesh = results["mesh"]

    points_to_remove = [pt for key in region_keys for pt in results.get(key, [])]

    if not points_to_remove:
        return results

    # 1. Map coordinates -> vertex index
    coord_to_idx = {tuple(coord): i for i, coord in enumerate(mesh.vertices)}

    # 2. Collect vertex indices to remove
    remove_indices = set()
    for pt in points_to_remove:
        idx = coord_to_idx.get(tuple(pt))
        if idx is not None:
            remove_indices.add(idx)

    if not remove_indices:
        return results

    n_vertices = len(mesh.vertices)
    keep_mask = np.ones(n_vertices, dtype=bool)
    keep_mask[list(remove_indices)] = False

    # 3. Mark the removal rim: kept vertices that had at least one removed
    #    neighbour.  This seeds which open boundaries to clean, so the mesh's
    #    unrelated rims (aorta inlet, vessel ends) are left alone.
    adj_map = build_adjacency_map(mesh.faces.tolist())
    boundary_indices = {
        i
        for i in range(n_vertices)
        if keep_mask[i] and any(j in remove_indices for j in adj_map.get(i, []))
    }

    # 4. Drop faces that reference any removed vertex, then clean the exposed
    #    rim.  Vertices that cannot form a clean ring are deleted from the mesh
    #    too - not just skipped in the ring - so the stored boundary really is
    #    the mesh's open edge.
    face_keep_mask = np.all(keep_mask[mesh.faces], axis=1)
    extra_drop, components = clean_open_boundary(
        mesh.faces[face_keep_mask],
        mesh.vertices,
        boundary_indices,
        target_n=target_boundaries,
    )
    if extra_drop:
        keep_mask[np.fromiter(extra_drop, dtype=np.int64)] = False
        face_keep_mask = np.all(keep_mask[mesh.faces], axis=1)
    new_faces = mesh.faces[face_keep_mask]

    # 5. Remap vertex indices in the kept faces
    new_index = np.full(n_vertices, -1, dtype=np.int64)
    new_index[keep_mask] = np.arange(keep_mask.sum(), dtype=np.int64)
    new_faces = new_index[new_faces]

    new_vertices = mesh.vertices[keep_mask]
    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)

    # 6. Rebuild the results dict with updated coordinate lists
    new_coord_set = {tuple(v) for v in new_vertices}

    updated = dict(results)
    updated["mesh"] = new_mesh
    _store_boundary_rings(updated, mesh.vertices, components)

    print(f"Applying removal of '{region_keys}'")
    print(f"Removed {len(points_to_remove)}")
    if extra_drop:
        print(f"Culled {len(extra_drop)} unclean boundary vertices from the mesh")
    print(
        f"Created {len(updated['boundary_points'])} boundary points "
        f"in {len(components)} ring(s): {[len(c) for c in components]}"
    )

    for key in region_keys:
        updated[key] = []

    for key in (
        "aorta_points",
        "rca_points",
        "lca_points",
        "rca_removed_points",
        "lca_removed_points",
        "proximal_points",
        "distal_points",
    ):
        if key in updated and key not in region_keys:
            updated[key] = [p for p in updated[key] if tuple(p) in new_coord_set]

    return updated


def keep_labeled_points_from_mesh(
    results: dict,
    region_key: str | list[str],
    target_boundaries: int = 1,
) -> dict:
    """Keep only the labeled region of vertices and remove everything else.

    Retains only the vertices stored under *region_key* (and the faces that
    reference exclusively those vertices), remaps faces, and rebuilds every
    coordinate list in *results* to reflect the new vertex indices.

    Parameters
    ----------
    results : dict
        Dictionary containing at minimum the key ``"mesh"``.  Any of
        ``"aorta_points"``, ``"rca_points"``, ``"lca_points"``,
        ``"rca_removed_points"``, ``"lca_removed_points"``,
        ``"proximal_points"``, and ``"distal_points"`` are also updated if
        present.
    region_key : str or list[str]
        Key (or list of keys) in *results* whose point lists define the
        vertices to *keep*.  When multiple keys are given the union of all
        their point sets is kept.
    target_boundaries : int
        Number of open-boundary rings the trim is expected to leave.  The rim
        is cleaned and reduced to this many rings before it is stored.

    Returns
    -------
    dict
        Updated *results* dict with ``"mesh"`` replaced by the trimmed mesh
        and all other coordinate lists filtered to the surviving vertex set.
        The open boundary is stored both per ring - ``"boundary_points_1"``,
        ``"boundary_points_2"``, ... in walk order - and flattened into
        ``"boundary_points"``.
    """
    mesh: trimesh.Trimesh = results["mesh"]

    region_keys = [region_key] if isinstance(region_key, str) else list(region_key)

    points_to_keep = []
    for key in region_keys:
        points_to_keep.extend(results.get(key, []))
    if not points_to_keep:
        return results

    coord_to_idx = {tuple(coord): i for i, coord in enumerate(mesh.vertices)}

    keep_indices = set()
    for pt in points_to_keep:
        idx = coord_to_idx.get(tuple(pt))
        if idx is not None:
            keep_indices.add(idx)

    if not keep_indices:
        return results

    n_vertices = len(mesh.vertices)
    keep_mask = np.zeros(n_vertices, dtype=bool)
    keep_mask[list(keep_indices)] = True
    remove_indices = set(range(n_vertices)) - keep_indices

    # Boundary: kept vertices that had at least one removed neighbour
    adj_map = build_adjacency_map(mesh.faces.tolist())
    boundary_indices = {
        i for i in keep_indices if any(j in remove_indices for j in adj_map.get(i, []))
    }

    # Drop faces that reference any removed vertex, then clean the exposed rim,
    # deleting unclean boundary vertices from the mesh as well as the ring.
    face_keep_mask = np.all(keep_mask[mesh.faces], axis=1)
    extra_drop, components = clean_open_boundary(
        mesh.faces[face_keep_mask],
        mesh.vertices,
        boundary_indices,
        target_n=target_boundaries,
    )
    if extra_drop:
        keep_mask[np.fromiter(extra_drop, dtype=np.int64)] = False
        face_keep_mask = np.all(keep_mask[mesh.faces], axis=1)
    new_faces = mesh.faces[face_keep_mask]

    # Remap vertex indices
    new_index = np.full(n_vertices, -1, dtype=np.int64)
    new_index[keep_mask] = np.arange(keep_mask.sum(), dtype=np.int64)
    new_faces = new_index[new_faces]

    new_vertices = mesh.vertices[keep_mask]
    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)

    new_coord_set = {tuple(v) for v in new_vertices}

    updated = dict(results)
    updated["mesh"] = new_mesh
    _store_boundary_rings(updated, mesh.vertices, components)

    for key in (
        "aorta_points",
        "rca_points",
        "lca_points",
        "rca_removed_points",
        "lca_removed_points",
        "proximal_points",
        "distal_points",
        *region_keys,
    ):
        if key in updated:
            updated[key] = [p for p in updated[key] if tuple(p) in new_coord_set]

    return updated


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
    :func:`remove_labeled_points_from_mesh` with ``target_boundaries=2``).  Each
    ring is assigned to an IV end as a whole, by whichever pairing of rings to
    the proximal and distal frame centroids is closest overall.

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


def _prepare_prox_dist_boundary_pts(
    mesh: trimesh.Trimesh,
    results: dict,
    prox_centroid: tuple[float, float, float],
    dist_centroid: tuple[float, float, float],
    proximal_is_ostium: bool = True,
    proximal_iv_frame_pts=None,
    ostium_angle_threshold_deg: float = 45.0,
    clamp_overshoot: float = 1.0,
    target_n: int | None = None,
    prox_outward: np.ndarray | None = None,
) -> tuple[list, list, trimesh.Trimesh]:
    """Pick and condition the two boundary rings that will be stitched.

    Both rims get the same treatment: the ring is flattened onto its own
    best-fit plane, smoothed, respaced evenly along its perimeter, and finally
    densified to *target_n* points so the stitch is a clean strip.  Every one of
    those steps is written back into the mesh, so the returned rings are the
    mesh's real open edge.  An ostial proximal ring gets the extra plane
    handling in :func:`_condition_ostium_ring` before densification.
    """
    rings = _boundary_rings(results, mesh)
    if len(rings) < 2:
        raise ValueError(
            f"Stitching needs a proximal and a distal boundary ring, but "
            f"{len(rings)} were found. Re-run the removal with "
            f"target_boundaries=2 so both rims are kept as separate rings."
        )

    prox_ring, dist_ring, leftover = _assign_rings_to_ends(
        rings, prox_centroid, dist_centroid
    )
    if leftover:
        print(
            f"Warning: {len(leftover)} boundary ring(s) "
            f"{[len(rings[k]) for k in leftover]} are not adjacent to either IV "
            f"end and are left unstitched."
        )

    # Flatten + even out both rims.  Smoothing removes the in-plane jaggedness
    # that would otherwise show up as ragged stitch triangles; respacing then
    # makes the interpolated points land uniformly around the ring.  The
    # size-preserving smoother matters here: plain Laplacian smoothing shrinks a
    # coarse ring badly (~16 % at 17 points), which showed up as a distal seam
    # pinched well inside the vessel.
    prox_pts = _redistribute_ring_evenly(
        _smooth_ring_preserving_size(_project_to_best_fit_plane(prox_ring))
    )
    mesh, _ = _write_ring_to_mesh(mesh, prox_ring, prox_pts)
    dist_pts = _redistribute_ring_evenly(
        _smooth_ring_preserving_size(_project_to_best_fit_plane(dist_ring))
    )
    mesh, _ = _write_ring_to_mesh(mesh, dist_ring, dist_pts)

    if proximal_is_ostium:
        prox_pts, mesh = _condition_ostium_ring(
            mesh,
            prox_pts,
            prox_centroid,
            proximal_iv_frame_pts,
            prox_outward,
            ostium_angle_threshold_deg,
            clamp_overshoot,
            aorta_pts=results.get("aorta_points"),
        )

    # Densify last, so the inserted points interpolate between final positions
    # and inherit the ring's planarity for free.
    if target_n:
        mesh, prox_pts = _densify_boundary(mesh, prox_pts, target_n)
        mesh, dist_pts = _densify_boundary(mesh, dist_pts, target_n)

    return prox_pts, dist_pts, mesh


def _toward_aorta(
    ring_centroid: np.ndarray,
    aorta_pts,
    fallback: np.ndarray | None,
) -> tuple[np.ndarray | None, str]:
    """Direction from the ostial ring into the aorta, and how it was derived.

    Taken from the labelled aortic surface: the vector from the ring centroid to
    the aortic centroid points into the aortic lumen whatever the take-off angle.
    The vessel axis is *not* a usable substitute here - an anomalous coronary runs
    inside the aortic wall, so the lumen lies roughly perpendicular to the axis
    and the axis can point away from the aorta entirely.  *fallback* is only used
    when no aortic points are available.
    """
    if aorta_pts is not None and len(aorta_pts) > 0:
        direction = np.asarray(aorta_pts, dtype=np.float64).mean(axis=0) - ring_centroid
        if np.any(direction):
            return direction, "aorta_points centroid"
    if fallback is not None and np.any(fallback):
        return np.asarray(fallback, dtype=np.float64), "vessel axis (no aorta_points)"
    return None, "unavailable"


def _condition_ostium_ring(
    mesh: trimesh.Trimesh,
    ring: list,
    prox_centroid: tuple[float, float, float],
    iv_frame_pts,
    outward: np.ndarray | None,
    angle_threshold_deg: float,
    overshoot: float,
    aorta_pts=None,
) -> tuple[list, trimesh.Trimesh]:
    """Keep an ostial boundary ring clear of the IV ostial frame.

    Two corrections, both along a plane normal:

    1. *Whole-plane shift.*  An anomalous ostium can leave the ring's own plane
       cutting straight through the IV ostial frame.  When that happens the plane
       is slid toward the aorta - the direction coming from :func:`_toward_aorta`
       - until it clears every frame point by *overshoot* mm, and the ring is
       re-projected onto the shifted plane.
    2. *Per-point clamp.*  The existing correction: where the two planes meet at
       a steep angle, individual points on the wrong side of - or too close to -
       the IV plane are clamped, and the two mesh layers behind them are pushed
       out to avoid a ridge.
    """
    if iv_frame_pts is None or len(ring) < 3:
        return ring, mesh

    iv_arr = np.array([[p.x, p.y, p.z] for p in iv_frame_pts], dtype=np.float64)
    ring_arr = np.asarray(ring, dtype=np.float64)
    original = list(ring)

    # 1. Slide the ring's own plane clear of the ostial frame.
    aorta_dir, dir_source = _toward_aorta(ring_arr.mean(axis=0), aorta_pts, outward)
    if aorta_dir is not None:
        shifted_origin, shifted_normal, moved = _shift_plane_clear_of(
            ring_arr.mean(axis=0),
            _plane_normal_svd(ring_arr),
            iv_arr,
            aorta_dir,
            overshoot,
        )
        if moved > 0.0:
            print(
                f"Ostium: boundary plane cut the IV frame; moved it {moved:.2f} mm "
                f"toward the aorta (direction from {dir_source})."
            )
            ring = _project_onto_plane(ring, shifted_origin, shifted_normal)
            ring_arr = np.asarray(ring, dtype=np.float64)

    # 2. Clamp individual points against the IV plane when the planes are steep.
    iv_normal = _plane_normal_svd(iv_arr)
    clamped = False
    iv_origin = np.asarray(prox_centroid, dtype=np.float64)
    if (
        _angle_between_planes_deg(_plane_normal_svd(ring_arr), iv_normal)
        >= angle_threshold_deg
    ):
        ring = _clamp_to_plane(ring, iv_origin, iv_normal, overshoot=overshoot)
        clamped = True

    mesh, moved_indices = _write_ring_to_mesh(mesh, original, ring)
    if clamped and moved_indices:
        mesh = _enforce_layer_gap_from_plane(mesh, moved_indices, iv_origin, iv_normal)
    return ring, mesh


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


def _smooth_ring_laplacian(
    points: list[tuple[float, float, float]],
    iterations: int = 5,
    alpha: float = 0.5,
) -> list[tuple[float, float, float]]:
    """Laplacian smoothing of a closed boundary ring.

    Each vertex is blended toward the midpoint of its two ring neighbors.
    Since the input is already coplanar, the result stays on the same plane
    (a linear combination of coplanar points is coplanar).

    Parameters
    ----------
    iterations : int
        Number of smoothing passes.
    alpha : float
        Weight kept on the original position (0 = full Laplacian, 1 = no-op).
    """
    if len(points) < 3:
        return points
    pts = np.array(points, dtype=np.float64)
    for _ in range(iterations):
        prev = pts.copy()
        neighbor_avg = (np.roll(prev, 1, axis=0) + np.roll(prev, -1, axis=0)) / 2.0
        pts = alpha * prev + (1.0 - alpha) * neighbor_avg
    return [tuple(p) for p in pts]


def _ring_calibre(pts: np.ndarray) -> float:
    """Mean distance from a ring's centroid - its effective radius.

    Preferred over perimeter as a size measure for a noisy ring: radial noise
    averages out here, whereas it inflates path length, so perimeter would read a
    jagged ring as much larger than a smooth one of the same diameter.
    """
    return float(np.linalg.norm(pts - pts.mean(axis=0), axis=1).mean())


def _smooth_ring_preserving_size(
    points: list[tuple[float, float, float]],
    iterations: int = 5,
    alpha: float = 0.5,
) -> list[tuple[float, float, float]]:
    """Laplacian-smooth a closed ring without shrinking it.

    Plain Laplacian smoothing pulls every vertex toward the midpoint of its two
    neighbours, which contracts a closed ring on every pass.  For an evenly
    spaced ring of *n* points the calibre drops by roughly
    ``(alpha + (1 - alpha) * cos(2*pi/n)) ** iterations``, so coarse rings lose
    the most - about 16 % at 17 points versus 6 % at 29 and under 1 % at 100.
    That is far too much for a boundary ring, whose diameter has to keep matching
    the vessel.

    This smooths as before, then scales the result about its centroid to restore
    the original calibre, removing jaggedness without losing diameter.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return [tuple(p) for p in pts]

    before = _ring_calibre(pts)
    smoothed = np.asarray(
        _smooth_ring_laplacian([tuple(p) for p in pts], iterations, alpha),
        dtype=np.float64,
    )
    after = _ring_calibre(smoothed)
    if before <= 0.0 or after <= 0.0:
        return [tuple(p) for p in smoothed]

    centroid = smoothed.mean(axis=0)
    restored = centroid + (smoothed - centroid) * (before / after)
    return [tuple(p) for p in restored]


def _redistribute_ring_evenly(
    points: list[tuple[float, float, float]],
    n_out: int | None = None,
) -> list[tuple[float, float, float]]:
    """Resample a closed ring to evenly spaced points along its own perimeter.

    Walks the ring as a closed polyline and places *n_out* samples at equal
    arc-length intervals, so clustered vertices spread out and sparse stretches
    fill in.  Index 0 stays exactly where it was, preserving any start point the
    caller already chose.  Unlike :func:`_smooth_ring_laplacian` this does not
    shrink the ring - every sample lands on the original polygon.
    """
    pts = np.asarray(points, dtype=np.float64)
    count = len(pts) if n_out is None else n_out
    if len(pts) < 3 or count < 3:
        return [tuple(p) for p in pts]

    loop = np.vstack([pts, pts[:1]])
    seg_len = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    perimeter = float(cum[-1])
    if perimeter <= 0.0:
        return [tuple(p) for p in pts]

    out: list[tuple[float, float, float]] = []
    for target in np.linspace(0.0, perimeter, count, endpoint=False):
        k = min(int(np.searchsorted(cum, target, side="right") - 1), len(seg_len) - 1)
        span = float(seg_len[k])
        frac = 0.0 if span <= 0.0 else (float(target) - float(cum[k])) / span
        out.append(tuple(loop[k] + frac * (loop[k + 1] - loop[k])))
    return out


def _project_onto_plane(
    points: list[tuple[float, float, float]],
    origin: np.ndarray,
    normal: np.ndarray,
) -> list[tuple[float, float, float]]:
    """Orthogonally project *points* onto the plane through *origin*."""
    pts = np.asarray(points, dtype=np.float64)
    return [tuple(p) for p in pts - np.outer((pts - origin) @ normal, normal)]


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


def _densify_boundary(
    mesh: trimesh.Trimesh,
    ring: list[tuple[float, float, float]],
    target_n: int,
) -> tuple[trimesh.Trimesh, list[tuple[float, float, float]]]:
    r"""Insert points along a boundary ring until it holds *target_n* vertices.

    New points are placed on the ring's own edges, so the rim keeps its shape,
    and the surplus goes to the longest edges first to keep spacing even.

    A subdivided edge belongs to exactly one surviving face, so that face is
    replaced by a fan onto its opposite ("third") vertex::

            C                       C
           / \                    / | \  \
          /   \        ->        /  |  \    \
         A-----B                A--P1---P2---B

    Without this the new points would be T-junctions on an edge no face knows
    about, and the stitched mesh could not close.  A face carrying two rim edges
    has no corner off the subdivision, so it fans from the polygon centroid
    instead, avoiding zero-area triangles.

    Returns the rebuilt mesh and the densified ring in walk order.
    """
    n = len(ring)
    extra = target_n - n
    if n < 3 or extra <= 0:
        if extra < 0:
            print(
                f"Warning: boundary ring has {n} points, more than the target "
                f"{target_n}; leaving it as it is (reducing it would need edge "
                f"collapses on the CCTA mesh)."
            )
        return mesh, list(ring)

    coord_to_idx = {tuple(v): i for i, v in enumerate(mesh.vertices)}
    lookup = [coord_to_idx.get(tuple(p)) for p in ring]
    if any(i is None for i in lookup):
        print("Warning: boundary ring is not on the mesh; skipping densification.")
        return mesh, list(ring)
    idx: list[int] = [int(i) for i in lookup if i is not None]

    # Undirected edge -> incident faces, so each ring edge can find its face(s).
    edge_faces: dict[frozenset, list[int]] = {}
    for fi, f in enumerate(mesh.faces):
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            edge_faces.setdefault(frozenset((int(a), int(b))), []).append(fi)

    # Spread the extra points over the ring, longest edge first.
    verts = mesh.vertices
    edges = [(idx[i], idx[(i + 1) % n]) for i in range(n)]
    lengths = [float(np.linalg.norm(verts[b] - verts[a])) for a, b in edges]
    counts = [extra // n] * n
    for e in sorted(range(n), key=lambda k: lengths[k], reverse=True)[: extra % n]:
        counts[e] += 1

    new_pts: list[np.ndarray] = []
    inserted: dict[tuple[int, int], list[int]] = {}
    next_idx = len(verts)
    for (a, b), count in zip(edges, counts):
        ids: list[int] = []
        if count:
            pa, pb = verts[a], verts[b]
            for j in range(1, count + 1):
                new_pts.append(pa + (j / (count + 1)) * (pb - pa))
                ids.append(next_idx)
                next_idx += 1
        inserted[(a, b)] = ids

    all_vertices = np.vstack([verts, np.asarray(new_pts, dtype=np.float64)])

    def points_on(a: int, b: int) -> list[int]:
        """Inserted ids for the directed edge a->b (reversed if stored b->a)."""
        if inserted.get((a, b)):
            return inserted[(a, b)]
        if inserted.get((b, a)):
            return list(reversed(inserted[(b, a)]))
        return []

    touched: set[int] = set()
    for (a, b), ids in inserted.items():
        if ids:
            touched.update(edge_faces.get(frozenset((a, b)), []))

    faces: list[tuple[int, int, int]] = [
        tuple(int(v) for v in f)  # type: ignore[misc]
        for fi, f in enumerate(mesh.faces)
        if fi not in touched
    ]
    for fi in touched:
        f = [int(v) for v in mesh.faces[fi]]
        poly: list[int] = []
        on_subdivided: set[int] = set()
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            poly.append(a)
            mids = points_on(a, b)
            poly.extend(mids)
            if mids:
                on_subdivided.update((a, b))

        apex = next((v for v in f if v not in on_subdivided), None)
        if apex is not None:
            r = poly.index(apex)
            rot = poly[r:] + poly[:r]
            faces.extend((rot[0], rot[i], rot[i + 1]) for i in range(1, len(rot) - 1))
        else:
            all_vertices = np.vstack(
                [all_vertices, all_vertices[poly].mean(axis=0)[None]]
            )
            c = len(all_vertices) - 1
            faces.extend(
                (c, poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))
            )

    dense: list[tuple[float, float, float]] = []
    for a, b in edges:
        dense.append(tuple(all_vertices[a]))
        dense.extend(tuple(all_vertices[j]) for j in inserted[(a, b)])

    new_mesh = trimesh.Trimesh(
        vertices=all_vertices,
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    return new_mesh, dense


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


def order_points_list(mesh: trimesh.Trimesh, points: list) -> list:
    """Order boundary points into a connected ring by walking mesh edges.

    Starting from the first point in *points*, the function follows edges to
    unvisited boundary neighbours until no further boundary neighbour can be
    reached.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh whose edge connectivity is used for traversal.
    points : list of tuple
        Boundary point coordinates to order.

    Returns
    -------
    list of tuple
        The same points reordered so that consecutive entries share a mesh edge.
    """
    if len(points) <= 1:
        return list(points)

    coord_to_idx = {tuple(coord): i for i, coord in enumerate(mesh.vertices)}

    boundary_indices = []
    idx_to_pt = {}
    for pt in points:
        idx = coord_to_idx.get(tuple(pt))
        if idx is not None:
            boundary_indices.append(idx)
            idx_to_pt[idx] = pt

    if not boundary_indices:
        return list(points)

    boundary_set = set(boundary_indices)
    adj_map = build_adjacency_map(mesh.faces.tolist())

    # Restrict adjacency to boundary-only neighbours
    boundary_adj = {
        i: [n for n in adj_map.get(i, []) if n in boundary_set]
        for i in boundary_indices
    }

    start = boundary_indices[0]
    ordered = [start]
    visited = {start}
    current = start

    while True:
        next_candidates = [n for n in boundary_adj[current] if n not in visited]
        if not next_candidates:
            break
        current = next_candidates[0]
        ordered.append(current)
        visited.add(current)

    # If connectivity reached all points, done
    if len(visited) == len(boundary_indices):
        return [idx_to_pt[i] for i in ordered]

    # Connectivity is broken —> fall back to plane-fit + counterclockwise projection
    pts_array = np.array([idx_to_pt[i] for i in boundary_indices], dtype=np.float64)
    centroid = pts_array.mean(axis=0)
    centered = pts_array - centroid

    # Fit plane via SVD: the normal is the right-singular vector with smallest singular value
    _, _, Vt = np.linalg.svd(centered)
    normal = Vt[-1]  # plane normal

    # Build an orthonormal 2-D basis on the plane
    u = Vt[0]
    v = np.cross(normal, u)

    # Project each point to 2-D and compute its angle around the centroid
    angles = np.arctan2(centered @ v, centered @ u)
    order = np.argsort(angles)  # counterclockwise by ascending angle

    return [idx_to_pt[boundary_indices[k]] for k in order]


def _adjust_start_point_by_z(boundary_pts: list) -> list:
    """Rotate a boundary ring so the point with the highest z-value is first."""
    start_idx = int(np.argmax([pt[2] for pt in boundary_pts]))
    return boundary_pts[start_idx:] + boundary_pts[:start_idx]


def _rotate_to_nearest_iv(boundary_pts: list, iv_pt) -> list:
    """Rotate a boundary ring so the point nearest to *iv_pt* is first."""
    iv_arr = np.array([iv_pt.x, iv_pt.y, iv_pt.z])
    dists = [np.linalg.norm(np.array(pt) - iv_arr) for pt in boundary_pts]
    start_idx = int(np.argmin(dists))
    return boundary_pts[start_idx:] + boundary_pts[:start_idx]


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


def _fix_ring_direction_by_distance(
    boundary_pts: list,
    iv_pts,
    point_step: int,
) -> list:
    """Subsample IV points to match the boundary ring count, then compare total
    point-wise distance for the ring as-is vs reversed (index 0 kept fixed).
    Works reliably when both rings start near the same spatial location
    (i.e. after ``_rotate_to_nearest_iv``).
    """
    iv_sub = iv_pts[0::point_step][: len(boundary_pts)]
    reversed_pts = [boundary_pts[0]] + list(reversed(boundary_pts[1:]))

    def total_dist(bpts):
        n = min(len(bpts), len(iv_sub))
        return sum(
            np.linalg.norm(
                np.array(bpts[i]) - np.array([iv_sub[i].x, iv_sub[i].y, iv_sub[i].z])
            )
            for i in range(n)
        )

    return (
        reversed_pts
        if total_dist(reversed_pts) < total_dist(boundary_pts)
        else boundary_pts
    )


def _fix_ring_direction_by_winding(
    boundary_pts: list,
    iv_pts,
) -> list:
    """Match the CCTA boundary ring's winding direction to the IV ring.

    Uses Newell's method on the IV ring to get a reference normal that by
    construction makes the IV ring appear CCW.  Projecting the CCTA ring onto
    that same normal gives a negative signed area when it winds in the opposite
    direction — in which case the ring is reversed (keeping index 0 fixed).
    """
    iv_arr = [[p.x, p.y, p.z] for p in iv_pts]
    normal = _newell_normal(iv_arr)
    # iv_sign is always positive by Newell construction; only check b_sign
    b_sign = _signed_area_projected(boundary_pts, normal)
    if b_sign < 0:
        return [boundary_pts[0]] + list(reversed(boundary_pts[1:]))
    return boundary_pts


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
