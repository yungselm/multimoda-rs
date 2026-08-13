"""Open-boundary ring extraction for trimmed meshes.

Cutting a labelled region out of a mesh leaves an open rim that later steps
stitch against.  This module turns that rim into ordered vertex rings.

Two entry points, differing in whether they may modify the mesh:

* :func:`clean_open_boundary` - used while trimming.  Rejects rim vertices that
  cannot form a clean ring and reports them for deletion from the mesh, so the
  rings it returns really do trace the mesh's open edge.
* :func:`order_boundary_rings` - read-only.  Reports the rings the faces
  actually have, for inspection and debug views.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

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
# Ring cleanup
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
# Entry points
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
