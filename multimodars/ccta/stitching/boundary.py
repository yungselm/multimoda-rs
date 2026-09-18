"""Boundary preparation for stitching: topology extraction and ring conditioning.

Cutting a labelled region out of a mesh leaves an open rim that later steps
stitch against. This module turns that rim into ordered vertex rings and
conditions each ring (flattening, smoothing, respacing, densifying, and
orienting) so it is ready to hand to :mod:`.core` for the actual stitch.

Two topology entry points, differing in whether they may modify the mesh:

* :func:`clean_open_boundary` - used while trimming.  Rejects rim vertices that
  cannot form a clean ring and reports them for deletion from the mesh, so the
  rings it returns really do trace the mesh's open edge.
* :func:`order_boundary_rings` - read-only.  Reports the rings the faces
  actually have, for inspection and debug views.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import trimesh
from scipy.interpolate import splev, splprep

from ...multimodars import build_adjacency_map
from .helpers import (
    _angle_between_planes_deg,
    _clamp_to_plane,
    _enforce_layer_gap_from_plane,
    _plane_normal_svd,
    _project_onto_plane,
    _project_to_best_fit_plane,
    _shift_plane_clear_of,
    _write_ring_to_mesh,
    _newell_normal,
    _signed_area_projected,
)

# Per-ring boundary keys: "boundary_points_1", "boundary_points_2", ...
BOUNDARY_RING_PREFIX = "boundary_points_"

# ---------------------------------------------------------------------------
# Open-boundary graph
# ---------------------------------------------------------------------------


def open_boundary_edges(faces: np.ndarray) -> np.ndarray:
    """Return the open-boundary edges of *faces* (edges used by exactly one face).

    Parameters
    ----------
    faces : (F, 3) int array
        Triangle vertex indices.

    Returns
    -------
    (E, 2) int array
        Sorted vertex-index pairs, one per edge lying on the open rim.
    """
    if len(faces) == 0:
        return np.empty((0, 2), dtype=np.int64)
    edges = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    return uniq[counts == 1]


def _faces_without(faces: np.ndarray, drop: set[int]) -> np.ndarray:
    """Return the faces of *faces* that reference none of the *drop* vertices."""
    if not drop:
        return faces
    dropped = np.fromiter(drop, dtype=np.int64, count=len(drop))
    return faces[~np.any(np.isin(faces, dropped), axis=1)]


def _boundary_graph(faces: np.ndarray) -> dict[int, set[int]]:
    """Adjacency of the open rim of *faces*, built from open boundary edges only.

    Using open edges rather than full vertex adjacency means interior chords
    between two rim vertices cannot invent junctions or fragment a ring.
    """
    graph: dict[int, set[int]] = {}
    for a, b in open_boundary_edges(faces):
        graph.setdefault(int(a), set()).add(int(b))
        graph.setdefault(int(b), set()).add(int(a))
    return graph


def _rims_touching(
    graph: Mapping[int, set[int]],
    seeds: set[int],
) -> dict[int, set[int]]:
    """Restrict *graph* to the connected rims containing at least one seed.

    Selecting whole components (rather than filtering edges by whether both
    endpoints are seeds) keeps a removal hole intact when it merges with a
    pre-existing opening: the pre-existing stretch has no removed neighbour and
    would otherwise drop out, leaving a gap in the ring.  With no seeds every
    rim is kept.
    """
    if not seeds:
        return {v: set(ns) for v, ns in graph.items()}
    keep: set[int] = set()
    unvisited = set(graph)
    while unvisited:
        stack = [unvisited.pop()]
        comp = set(stack)
        while stack:
            v = stack.pop()
            for w in graph.get(v, set()):
                if w not in comp:
                    comp.add(w)
                    unvisited.discard(w)
                    stack.append(w)
        if comp & seeds:
            keep |= comp
    return {v: set(graph[v]) & keep for v in keep}


def _walk_rings(graph: Mapping[int, set[int]]) -> list[list[int]]:
    """Trace every connected component of *graph* into an ordered vertex list."""
    remaining = set(graph)
    rings: list[list[int]] = []
    while remaining:
        start = next(iter(remaining))
        ring = [start]
        remaining.discard(start)
        prev, current = -1, start
        while True:
            nxt = next(
                (n for n in graph.get(current, set()) if n != prev and n in remaining),
                None,
            )
            if nxt is None:
                break
            ring.append(nxt)
            remaining.discard(nxt)
            prev, current = current, nxt
        rings.append(ring)
    return rings


# ---------------------------------------------------------------------------
# Ring cleanup (topology)
# ---------------------------------------------------------------------------


def _despike_ring(
    ring: list[int],
    vertices: np.ndarray,
    cos_thresh: float,
) -> list[int]:
    """Return *ring* with its "bump" spikes removed.

    A bump spike is a vertex the rim detours out to and immediately back from,
    so the two edge directions leaving it point nearly the same way (cosine
    close to ``+1``); a normal rim vertex has them pointing nearly opposite
    (cosine close to ``-1``).  Any vertex whose cosine exceeds *cos_thresh* is
    dropped, which reconnects its two neighbours directly.  Iterated because
    removing one tip can expose the next.

    *ring* is treated as a closed cycle - callers pass rings walked from a rim
    whose vertices all have degree 2, so every vertex has two neighbours.
    """
    pts = list(ring)
    changed = True
    while changed and len(pts) > 3:
        changed = False
        m = len(pts)
        for i in range(m):
            d1 = vertices[pts[i - 1]] - vertices[pts[i]]
            d2 = vertices[pts[(i + 1) % m]] - vertices[pts[i]]
            n1 = float(np.linalg.norm(d1))
            n2 = float(np.linalg.norm(d2))
            if n1 == 0.0 or n2 == 0.0:
                continue
            if float(np.dot(d1, d2)) / (n1 * n2) > cos_thresh:
                del pts[i]
                changed = True
                break
    return pts


def _join_rings(
    rings: list[list[int]],
    vertices: np.ndarray,
    target_n: int,
) -> list[list[int]]:
    """Greedily merge arcs until only *target_n* remain.

    Each pass finds the two endpoints from different arcs that are closest in
    space, flips the arcs so those endpoints meet, and concatenates them.  Used
    to reunite a ring that came back as several arcs.
    """
    comps = [list(r) for r in rings]
    while len(comps) > target_n:
        best_dist = float("inf")
        best = (0, 1, False, False)
        for a in range(len(comps)):
            for b in range(a + 1, len(comps)):
                # Flip flags are chosen so the two picked ends meet in the middle
                # of the concatenation.
                for pa, flip_a in ((comps[a][0], True), (comps[a][-1], False)):
                    for pb, flip_b in ((comps[b][0], False), (comps[b][-1], True)):
                        d = float(np.linalg.norm(vertices[pa] - vertices[pb]))
                        if d < best_dist:
                            best_dist, best = d, (a, b, flip_a, flip_b)
        a, b, flip_a, flip_b = best
        ca = comps[a][::-1] if flip_a else comps[a]
        cb = comps[b][::-1] if flip_b else comps[b]
        comps = [c for k, c in enumerate(comps) if k not in (a, b)] + [ca + cb]
    return comps


def _reduce_rings(
    rings: list[list[int]],
    vertices: np.ndarray,
    target_n: int | None,
    warn: bool = False,
) -> list[list[int]]:
    """Sort *rings* largest-first and reduce them to *target_n*.

    ``target_n=None`` reports every ring untouched.  Otherwise surplus rings are
    joined by nearest endpoints; when *warn* is set that join is announced,
    because it bridges a gap that is not a mesh edge - a sign either that
    *target_n* is too low for this region or that the pieces are separate rims.
    """
    rings = sorted((r for r in rings if r), key=len, reverse=True)
    if target_n is None or len(rings) <= target_n:
        return rings
    if warn:
        print(
            f"Warning: boundary has {len(rings)} rings {[len(r) for r in rings]} "
            f"but target_boundaries={target_n}; joining by nearest endpoints."
        )
    joined = _join_rings(rings, vertices, target_n)
    return sorted(joined, key=len, reverse=True)[:target_n]


# ---------------------------------------------------------------------------
# Topology entry points
# ---------------------------------------------------------------------------


def order_boundary_rings(
    faces: np.ndarray,
    vertices: np.ndarray,
    seeds: set[int] | None = None,
    target_n: int | None = None,
) -> list[list[int]]:
    """Order the open boundary of *faces* into rings, without touching the mesh.

    Read-only counterpart to :func:`clean_open_boundary`: it reports the rings
    the faces actually have, so a debug view shows the real state of the mesh
    rather than an idealised one.  Hence *target_n* defaults to ``None`` - every
    ring found is reported and none are silently merged.

    Parameters
    ----------
    faces : (F, 3) int array
        Faces of the mesh whose rim is wanted.
    vertices : (V, 3) float array
        Vertex coordinates, used only when joining surplus rings.
    seeds : set[int], optional
        Only the connected rims containing one of these vertices are reported.
        When omitted every open boundary is reported.
    target_n : int, optional
        When given, reduce to this many rings.  ``None`` reports all of them.

    Returns
    -------
    list[list[int]]
        Ordered rings of vertex indices, largest first.
    """
    graph = _rims_touching(_boundary_graph(faces), seeds or set())
    return _reduce_rings(_walk_rings(graph), vertices, target_n)


def clean_open_boundary(
    faces: np.ndarray,
    vertices: np.ndarray,
    seeds: set[int],
    target_n: int = 1,
    despike_cos: float = 0.0,
    max_rounds: int = 64,
) -> tuple[set[int], list[list[int]]]:
    """Cull rim vertices that cannot form a clean ring, from the *mesh*.

    Every vertex this rejects - isolated stragglers, dangling hairs, pinch
    junctions, and sharp "bump" spikes - is returned for deletion from the mesh
    itself, not merely skipped in the ring list.  Dropping a vertex removes its
    faces and so exposes a new rim, hence the loop: each round re-derives the
    boundary from the surviving faces until nothing more is rejected.  This
    keeps the returned rings a faithful trace of the mesh's actual open edge.

    Parameters
    ----------
    faces : (F, 3) int array
        Faces surviving the labelled-region removal.
    vertices : (V, 3) float array
        Vertex coordinates (used for the spike angle test and ring joining).
    seeds : set[int]
        Vertices known to lie on the rim of interest, so unrelated open
        boundaries (e.g. the aorta inlet) are left alone.
    target_n : int
        Number of rings the rim should end up as.
    despike_cos : float
        Threshold forwarded to :func:`_despike_ring`.
    max_rounds : int
        Safety bound on the re-derivation loop.

    Returns
    -------
    (set[int], list[list[int]])
        Vertices to delete from the mesh, and the resulting ordered rings.
    """
    drop: set[int] = set()
    seed_set = set(seeds)

    for _ in range(max_rounds):
        graph = _rims_touching(_boundary_graph(_faces_without(faces, drop)), seed_set)
        if not graph:
            return drop, []
        # Grow the seed set to the whole rim so vertices exposed by this round's
        # deletions are still recognised as part of it next round.
        seed_set |= set(graph)

        # A clean rim is degree-2 everywhere; anything else is a straggler (0/1)
        # or a pinch junction (3+), and gets cut out of the mesh.
        bad = {v for v, ns in graph.items() if len(ns) != 2}
        if bad:
            drop |= bad
            continue

        rings = _walk_rings(graph)
        spikes = {
            v
            for ring in rings
            for v in set(ring) - set(_despike_ring(ring, vertices, despike_cos))
        }
        if not spikes:
            return drop, _reduce_rings(rings, vertices, target_n, warn=True)
        drop |= spikes

    # Out of rounds: report the rim as it currently stands rather than a stale one.
    graph = _rims_touching(_boundary_graph(_faces_without(faces, drop)), seed_set)
    return drop, _reduce_rings(_walk_rings(graph), vertices, target_n, warn=True)


# ---------------------------------------------------------------------------
# Ring retrieval / end assignment
# ---------------------------------------------------------------------------


def _boundary_rings(results: dict, mesh: trimesh.Trimesh) -> list[list[tuple]]:
    """Return the boundary rings stored in *results*, each in walk order.

    Prefers the per-ring ``boundary_points_<n>`` lists written by
    :func:`multimodars.ccta.mesh_regions._store_boundary_rings`.  If only the
    flat ``boundary_points`` is present - a results dict built before ring
    grouping, or by hand - the rings are recovered from the mesh's open edges
    instead, so callers always get properly ordered rings rather than an
    arbitrary point order.
    """
    rings: list[list[tuple]] = []
    n = 1
    while (key := f"{BOUNDARY_RING_PREFIX}{n}") in results:
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


# ---------------------------------------------------------------------------
# Ring conditioning
# ---------------------------------------------------------------------------


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


def _largest_circular_true_run(mask: np.ndarray) -> tuple[int, int]:
    """Return ``(start, length)`` of the longest contiguous run of ``True``
    in a circular boolean array, allowing wraparound."""
    n = len(mask)
    if n == 0 or not mask.any():
        return 0, 0
    if mask.all():
        return 0, n

    doubled = np.concatenate([mask, mask])
    best_start, best_len = 0, 0
    i = 0
    while i < n:
        if not doubled[i]:
            i += 1
            continue
        j = i
        while j < i + n and doubled[j]:
            j += 1
        if j - i > best_len:
            best_start, best_len = i, j - i
        i = j
    return best_start % n, min(best_len, n)


def _split_ring_by_aorta_direction(
    ring: list[tuple[float, float, float]],
    center: np.ndarray,
    aorta_direction: np.ndarray,
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    """Split a ring into its aorta-facing half and coronary-facing half.

    Classifies each point by which side of *center* it falls on along
    *aorta_direction* (a full 3-D half-space test - no plane projection, so a
    steeply bent, non-planar ring splits correctly).  The ring is rotated so
    it starts at the aorta-facing run, then that run's own length is taken as
    "Half A"; everything else is "Half B".  Using the single largest
    aorta-facing run (rather than every point testing positive) keeps a
    handful of stray misclassified points near the transition out of Half A,
    since Half A gets a hard geometric correction while Half B only gets a
    gentle one.

    Returns
    -------
    (half_a, half_b)
        Each in the ring's own walk order; ``half_a[-1]`` is adjacent to
        ``half_b[0]``, and ``half_b[-1]`` is adjacent to ``half_a[0]``.
    """
    pts = np.asarray(ring, dtype=np.float64)
    n = len(pts)
    is_aorta = ((pts - center) @ aorta_direction) > 0.0
    best_start, best_len = _largest_circular_true_run(is_aorta)

    order = [(best_start + k) % n for k in range(n)]
    half_a = [ring[order[k]] for k in range(best_len)]
    half_b = [ring[order[k]] for k in range(best_len, n)]
    return half_a, half_b


def _ostium_aortic_side_points(iv_frame_pts) -> np.ndarray:
    """Return the IV ostial frame's own aortic-side lumen points, in order.

    Uses each point's own ``aortic`` flag - ground truth from the imaging
    labelling - rather than assuming any fixed point-index split.  Takes the
    largest contiguous run of ``aortic=True`` points (allowing wraparound),
    so a stray mislabeled point near the transition can't fragment it.
    """
    mask = np.array([bool(p.aortic) for p in iv_frame_pts])
    n = len(mask)
    start, length = _largest_circular_true_run(mask)
    order = [(start + k) % n for k in range(length)]
    return np.array(
        [[iv_frame_pts[i].x, iv_frame_pts[i].y, iv_frame_pts[i].z] for i in order],
        dtype=np.float64,
    )


def _resample_open_polyline(
    points: list[tuple[float, float, float]],
    n_out: int,
) -> list[tuple[float, float, float]]:
    """Resample an open polyline to *n_out* evenly (by arc length) spaced
    points, keeping both endpoints exactly.

    Unlike :func:`_redistribute_ring_evenly` (which treats its input as a
    closed loop), this does not wrap the last point back to the first - it is
    for an arc, not a ring.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 2 or n_out < 2:
        return [tuple(p) for p in pts]

    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])
    if total <= 0.0:
        return [tuple(pts[0])] * n_out

    out: list[tuple[float, float, float]] = []
    for target in np.linspace(0.0, total, n_out):
        k = int(np.clip(np.searchsorted(cum, target, side="right") - 1, 0, n - 2))
        span = float(cum[k + 1] - cum[k])
        frac = 0.0 if span <= 0.0 else (float(target) - float(cum[k])) / span
        out.append(tuple(pts[k] + frac * (pts[k + 1] - pts[k])))
    return out


def _duplicate_ostium_half_offset(
    aorta_side_pts: np.ndarray,
    center: np.ndarray,
    distance: float,
    target_ring_half: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    """Build Half A's target: the ostium's own aorta-side contour, scaled up.

    A uniformly scaled copy of the IV ostial frame's own aorta-side lumen
    contour (*aorta_side_pts*, from :func:`_ostium_aortic_side_points` - the
    points actually flagged ``aortic=True``, the same ones used to build the
    "Wall" extras) about the frame's own centroid (*center*) - same shape,
    same relative angles, no CCTA data involved at all - just enlarged so its
    mean radius from *center* grows by *distance* (the measured aortic wall
    thickness, i.e. ``frame.lumen.aortic_thickness`` / ``measurement_1``).

    Resampled to ``len(target_ring_half)`` points *after* scaling, not
    before, and reversed if needed so it starts/ends next to the same
    neighbours as *target_ring_half* does - matched by nearest endpoints,
    since the IV frame's own point order has no guaranteed relationship to
    the CCTA ring's mesh-walk order.
    """
    radial = aorta_side_pts - center
    mean_radius = float(np.linalg.norm(radial, axis=1).mean())
    scaled = (
        center + (1.0 + distance / mean_radius) * radial
        if mean_radius > 1e-9
        else aorta_side_pts
    )

    resampled = _resample_open_polyline(
        [tuple(p) for p in scaled], len(target_ring_half)
    )

    ref = np.asarray(target_ring_half, dtype=np.float64)
    cand = np.asarray(resampled, dtype=np.float64)
    d_forward = np.linalg.norm(ref[0] - cand[0]) + np.linalg.norm(ref[-1] - cand[-1])
    d_reversed = np.linalg.norm(ref[0] - cand[-1]) + np.linalg.norm(ref[-1] - cand[0])
    if d_reversed < d_forward:
        resampled = resampled[::-1]

    return resampled


def _fit_open_spline_ring(
    points: list[tuple[float, float, float]],
    smoothing: float = 0.0,
) -> list[tuple[float, float, float]]:
    """Fit a non-periodic cubic spline through an open arc and resample it.

    Unlike :func:`_project_to_best_fit_plane`, this does not flatten the arc
    onto a single plane, so real non-planar 3-D curvature survives.  With
    *smoothing* > 0 the fitted curve need not pass through every point, which
    is what actually lets it iron out an island or other outlier - an exact
    (``smoothing=0``) interpolating spline is forced through every point,
    however noisy, and changes nothing.  Endpoints are preserved exactly
    regardless, since callers rely on them as fixed junctions to the ring's
    other half.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 4:
        return list(points)

    try:
        tck, _ = splprep([pts[:, 0], pts[:, 1], pts[:, 2]], s=smoothing, k=3, per=False)
    except Exception:
        # Degenerate (coincident / collinear) arc: leave it untouched.
        return list(points)

    u = np.linspace(0.0, 1.0, n)
    x, y, z = splev(u, tck)
    out = list(np.column_stack([x, y, z]))
    out[0] = pts[0]
    out[-1] = pts[-1]
    return [tuple(p) for p in out]


def _taper_ring_displacement(
    mesh: trimesh.Trimesh,
    old_ring_pts: list[tuple[float, float, float]],
    new_ring_pts: list[tuple[float, float, float]],
    n_layers: int = 3,
    protected_pts: list[tuple[float, float, float]] | None = None,
    seam_damping: float = 0.25,
) -> trimesh.Trimesh:
    """Fade a rim's displacement into the surrounding mesh over *n_layers*.

    Replacing a rim wholesale (e.g. with a duplicated, offset "neo-ostium",
    see :func:`_duplicate_ostium_half_offset`) can move it far from where the
    mesh's very next layer of vertices still sits - unlike the small nudges
    :func:`_enforce_layer_gap_from_plane` was built for, this is a large
    enough jump to fold the faces bridging the two.  This assumes *mesh*
    already has the rim at *new_ring_pts* (i.e. run after
    :func:`_write_ring_to_mesh`), finds each rim vertex's displacement
    (``new - old``), and propagates a decreasing fraction of it outward
    through the face-adjacency graph: layer 1 (the rim's immediate
    neighbours) moves by most of it, layer *n_layers* by only a little, and
    anything further is untouched - so the mesh eases into the new rim
    position over several rings instead of jumping straight to it.

    *protected_pts* are excluded from ever being moved by this, even if they
    are graph-adjacent to the rim - e.g. the ring's *other* half, whose own
    two endpoints are topologically adjacent to this rim's endpoints and
    would otherwise get treated as an ordinary "layer 1" neighbour and pulled
    off their own, separately-computed position.  A vertex that is itself
    adjacent to one of those protected points - the interior vertex shared by
    the triangle spanning the seam between the two halves - gets its pull cut
    further by *seam_damping* on top of the usual layer weight: at full
    layer-1 weight it would move most of the way while its other neighbour
    (on the barely-moved protected side) stays almost still, which is enough
    to fold that one triangle over.
    """
    coord_to_idx = {tuple(v): i for i, v in enumerate(mesh.vertices)}
    rim_disp: dict[int, np.ndarray] = {}
    for old, new in zip(old_ring_pts, new_ring_pts):
        idx = coord_to_idx.get(tuple(new))
        if idx is not None:
            rim_disp[idx] = np.asarray(new, dtype=np.float64) - np.asarray(
                old, dtype=np.float64
            )
    if not rim_disp:
        return mesh

    adj_map = build_adjacency_map(mesh.faces.tolist())
    new_vertices = mesh.vertices.copy()

    protected = {
        idx
        for p in (protected_pts or [])
        if (idx := coord_to_idx.get(tuple(p))) is not None
    }
    seam_adjacent = {
        nb for p in protected for nb in adj_map.get(p, []) if nb not in protected
    }
    visited = set(rim_disp) | protected
    frontier = set(rim_disp)
    layer_disp = dict(rim_disp)

    for layer in range(1, n_layers + 1):
        weight = 1.0 - layer / (n_layers + 1)
        next_frontier: set[int] = set()
        next_layer_disp: dict[int, list[np.ndarray]] = {}
        for vi in frontier:
            for nb in adj_map.get(vi, []):
                if nb in visited:
                    continue
                next_layer_disp.setdefault(nb, []).append(layer_disp[vi])
                next_frontier.add(nb)
        averaged: dict[int, np.ndarray] = {}
        for vi, disps in next_layer_disp.items():
            avg = np.mean(disps, axis=0)
            damping = seam_damping if vi in seam_adjacent else 1.0
            new_vertices[vi] = new_vertices[vi] + weight * damping * avg
            averaged[vi] = avg  # undecayed, for the next layer to inherit

        layer_disp = averaged
        visited.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break

    return trimesh.Trimesh(vertices=new_vertices, faces=mesh.faces, process=False)


def _condition_ostium_ring_two_half(
    mesh: trimesh.Trimesh,
    ring: list[tuple[float, float, float]],
    prox_centroid: tuple[float, float, float],
    iv_frame_pts,
    aortic_thickness: float | None,
    clamp_overshoot: float,
    angle_threshold_deg: float = 45.0,
    smoothing: float | None = None,
    taper_layers: int = 3,
) -> tuple[list[tuple[float, float, float]], trimesh.Trimesh]:
    """Condition an anomalous ostial ring as two anatomically different halves.

    A steep-angle (e.g. anomalous, intramural) takeoff leaves the aorta-facing
    half of the ring almost perpendicular to the coronary-facing half - not a
    single circle or oval, and not even necessarily planar.  Treating the
    whole ring with one shape assumption (a plane, a circle, a convex hull)
    fights that real geometry.  Instead:

    * The aorta-facing half - identified from each IV ostial frame point's
      own ``aortic`` flag (ground truth from the imaging labelling), tested
      in full 3-D so it doesn't require planarity - is entirely replaced by
      :func:`_duplicate_ostium_half_offset`: a uniformly scaled-up copy of
      the ostium's own aorta-side contour, its mean radius grown by
      *aortic_thickness* (the same measurement used to build the "Wall"
      extras, i.e. ``frame.lumen.aortic_thickness`` / ``measurement_1``;
      *clamp_overshoot* is the fallback when it's unavailable).  None of the
      CCTA mesh's own (possibly badly distorted) geometry there survives -
      only the trusted, correctly-shaped IV data, same shape, just bigger.
    * The coronary-facing half keeps its own shape almost entirely -
      :func:`_fit_open_spline_ring` only irons out an island or other
      outlier, it does not pull points toward any idealised target.

    Replacing Half A wholesale can leave it far from where the mesh's very
    next layer of vertices still is; :func:`_taper_ring_displacement` fades
    that jump into the surrounding mesh over *taper_layers* rings instead of
    leaving it as an abrupt fold - and, since the taper never touches Half B,
    that half's own connection is left exactly as it is.

    The existing per-point IV-plane clamp (:func:`_clamp_to_plane`, the
    second half of :func:`_condition_ostium_ring`) still runs afterward, as a
    final safety net.  Its *whole-ring* plane-shift-and-reproject step does
    not - re-flattening the result onto one plane would undo the point of
    treating the two halves separately.
    """
    if iv_frame_pts is None or len(ring) < 6:
        return ring, mesh

    iv_arr = np.array([[p.x, p.y, p.z] for p in iv_frame_pts], dtype=np.float64)
    center = np.asarray(prox_centroid, dtype=np.float64)
    iv_normal = _plane_normal_svd(iv_arr)
    aorta_side = _ostium_aortic_side_points(iv_frame_pts)
    if len(aorta_side) < 2:
        return ring, mesh

    # In-plane direction, used only to decide which ring points are Half A vs
    # Half B - kept strictly in the ostial plane so the split doesn't depend
    # on how far off-plane a given CCTA ring point happens to be.
    aorta_direction = aorta_side.mean(axis=0) - center
    aorta_direction -= float(np.dot(aorta_direction, iv_normal)) * iv_normal
    norm = float(np.linalg.norm(aorta_direction))
    if norm < 1e-9:
        return ring, mesh
    aorta_direction /= norm

    half_a, half_b = _split_ring_by_aorta_direction(ring, center, aorta_direction)
    if len(half_a) < 2 or len(half_b) < 2:
        return ring, mesh

    distance = aortic_thickness if aortic_thickness is not None else clamp_overshoot
    duplicated_a = _duplicate_ostium_half_offset(aorta_side, center, distance, half_a)
    spline_smoothing = smoothing if smoothing is not None else 0.5 * len(half_b)
    conditioned_b = _fit_open_spline_ring(half_b, smoothing=spline_smoothing)

    original = list(half_a) + list(half_b)
    new_ring = duplicated_a + conditioned_b
    # Half A (a genuine resample) and Half B (spline-fit only, no resample)
    # don't share a spacing scale on their own; redistribute the whole
    # combined ring evenly by arc length so every point on the border - both
    # halves together - sits the same distance from its neighbours.
    new_ring = _redistribute_ring_evenly(new_ring)

    # Final safety net: clamp any point still behind (or too close in front
    # of) the IV plane, same as the per-point step in _condition_ostium_ring.
    if (
        _angle_between_planes_deg(_plane_normal_svd(np.asarray(new_ring)), iv_normal)
        >= angle_threshold_deg
    ):
        new_ring = _clamp_to_plane(
            new_ring, center, iv_normal, overshoot=clamp_overshoot
        )

    mesh, moved_indices = _write_ring_to_mesh(mesh, original, new_ring)
    # Half A's rim was replaced wholesale (see above) and can sit far from
    # where the mesh's next layers still are; fade that displacement inward
    # over a few layers instead of leaving an abrupt jump.  Use Half A's
    # *actual final* positions (post-redistribute/clamp), not the pre-
    # redistribute duplicated_a - otherwise the taper can't recognise most of
    # Half A's own rim vertices by their (now stale) coordinate, and ends up
    # treating a few of them as ordinary interior neighbours instead.
    mesh = _taper_ring_displacement(
        mesh,
        half_a,
        new_ring[: len(half_a)],
        n_layers=taper_layers,
        protected_pts=new_ring[len(half_a) :],
    )
    if moved_indices:
        mesh = _enforce_layer_gap_from_plane(mesh, moved_indices, center, iv_normal)
    return new_ring, mesh


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
    prox_start_mode: str = "nearest_iv",
    proximal_aortic_thickness: float | None = None,
) -> tuple[list, list, trimesh.Trimesh]:
    """Pick and condition the two boundary rings that will be stitched.

    Both rims get the same treatment: the ring is flattened onto its own
    best-fit plane, smoothed, respaced evenly along its perimeter, and finally
    densified to *target_n* points so the stitch is a clean strip.  Every one of
    those steps is written back into the mesh, so the returned rings are the
    mesh's real open edge.  An ostial proximal ring gets the extra plane
    handling in :func:`_condition_ostium_ring` before densification.

    A ``"highest_z"`` proximal ring skips all of that in favour of
    :func:`_condition_ostium_ring_two_half` instead: a steep-angle (e.g.
    anomalous, intramural) takeoff can leave that ring's aorta-facing half
    almost perpendicular to its coronary-facing half, so a single flatten
    -smooth-or-clamp treatment for the whole ring fights the real geometry.
    This has only been observed on the ostial proximal ring, so only that
    ring takes this path - the distal ring always takes the plain path above.
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

    if prox_start_mode == "highest_z" and proximal_is_ostium:
        prox_pts, mesh = _condition_ostium_ring_two_half(
            mesh,
            prox_ring,
            prox_centroid,
            proximal_iv_frame_pts,
            proximal_aortic_thickness,
            clamp_overshoot,
            angle_threshold_deg=ostium_angle_threshold_deg,
        )
    else:
        # Flatten + even out the rim.  Smoothing removes the in-plane
        # jaggedness that would otherwise show up as ragged stitch triangles;
        # respacing then makes the interpolated points land uniformly around
        # the ring.  The size-preserving smoother matters here: plain
        # Laplacian smoothing shrinks a coarse ring badly (~16 % at 17
        # points), which showed up as a distal seam pinched well inside the
        # vessel.
        prox_pts = _redistribute_ring_evenly(
            _smooth_ring_preserving_size(_project_to_best_fit_plane(prox_ring))
        )
        mesh, _ = _write_ring_to_mesh(mesh, prox_ring, prox_pts)
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

    dist_pts = _redistribute_ring_evenly(
        _smooth_ring_preserving_size(_project_to_best_fit_plane(dist_ring))
    )
    mesh, _ = _write_ring_to_mesh(mesh, dist_ring, dist_pts)

    # Densify last, so the inserted points interpolate between final positions
    # and inherit the ring's planarity for free.
    if target_n:
        mesh, prox_pts = _densify_boundary(mesh, prox_pts, target_n)
        mesh, dist_pts = _densify_boundary(mesh, dist_pts, target_n)

    return prox_pts, dist_pts, mesh
