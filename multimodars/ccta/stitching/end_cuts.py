"""Clean rims for the stitch, cut at slices of the discretized tree.

Removing a segment from the CCTA mesh leaves a jagged rim at each end of the
opening, along the triangles the labels happened to cover.  Away from the
ostium, :func:`cut_clean_ends` replaces such a rim by a planar cut at one of
the vessel's contours in the discretized tree (the same cut as an outlet in
:func:`~multimodars.cut_at_slice`, left open), so the rim the intravascular end
frame is joined to is round and flat like the frame itself.  Everything between
the old rim and the plane is removed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import trimesh

from .._cutting import cut_labelled_mesh
from .._results import VESSELS, branches_key
from .boundary import (
    BOUNDARY_RING_PREFIX,
    _assign_rings_to_ends,
    _boundary_rings,
    _ring_is_ostium,
)

if TYPE_CHECKING:
    from ...multimodars import PyContour, PyDiscretizedVesselTree


def cut_clean_ends(
    mesh: trimesh.Trimesh,
    results: dict,
    tree: PyDiscretizedVesselTree,
    prox_centroid: tuple[float, float, float],
    dist_centroid: tuple[float, float, float],
    proximal_is_ostium: bool | None,
    proximal_cut_slice: int | None = None,
    distal_cut_slice: int | None = None,
) -> tuple[trimesh.Trimesh, dict, dict[str, int], bool]:
    """Replace the distal rim, and the proximal one unless it is the ostium, by
    a planar cut at a contour of the vessel each rim lies on.

    The vessel (and branch) of a rim is the one most of its vertices are
    labelled with.  A rim's contour is *distal_cut_slice* /
    *proximal_cut_slice* (an index into that branch's contours), or by default
    the contour nearest the opening that has the whole rim on the opening's
    side: the first one distal of the distal rim, the last one proximal of the
    proximal rim.  The contour used is printed.

    Returns the cut mesh, *results* updated to it (the cut rims stored in
    place of the old ones), the contour each end was cut at (by
    ``"proximal"`` / ``"distal"``) and
    whether the proximal rim is the ostium (*proximal_is_ostium*, decided from
    the rim when ``None``).

    Raises
    ------
    ValueError
        If a rim's vertices carry no branch label, no contour has the whole rim
        on the opening's side, the cut fails, or it leaves part of the old rim
        behind.  The message names the argument to choose another contour
        with.
    """
    rings = _boundary_rings(results, mesh)
    if len(rings) < 2:
        # The stitch reports this itself.
        return mesh, results, {}, bool(proximal_is_ostium)
    prox_ring, dist_ring, _ = _assign_rings_to_ends(rings, prox_centroid, dist_centroid)
    if proximal_is_ostium is None:
        proximal_is_ostium = _ring_is_ostium(prox_ring, results)
        print(
            "Proximal ring is "
            + (
                "the ostium on the aorta."
                if proximal_is_ostium
                else "inside the vessel, not the ostium."
            )
        )

    ends = [("distal", dist_ring, distal_cut_slice)]
    if not proximal_is_ostium:
        ends.append(("proximal", prox_ring, proximal_cut_slice))
    used: dict[str, int] = {}
    for end, ring, slice_idx in ends:
        results, used[end] = _cut_end(results, tree, end, ring, slice_idx)
    return results["mesh"], results, used, proximal_is_ostium


def end_target(
    results: dict, tree: PyDiscretizedVesselTree, end: str, ring: list[tuple]
) -> tuple[str, int, int | None]:
    """``(vessel, branch id, default contour)`` the *end* rim *ring* is cut on.
    The default is ``None`` when no contour has the whole rim on the opening's
    side."""
    vessel, branch_id = _branch_of_ring(ring, results)
    contours = list(getattr(tree, vessel)[branch_id].contours)
    return vessel, branch_id, _nearest_clear_contour(contours, ring, end)


def _cut_end(
    results: dict,
    tree: PyDiscretizedVesselTree,
    end: str,
    ring: list[tuple],
    slice_idx: int | None,
) -> tuple[dict, int]:
    """*results* with the rim *ring* at the *end* of the opening cut away at a
    contour, the open cut rim stored in its place, and that contour."""
    arg = f"{end}_cut_slice"
    vessel, branch_id, default = end_target(results, tree, end, ring)
    name = f"{vessel.upper()} branch {branch_id}"
    contours = list(getattr(tree, vessel)[branch_id].contours)
    if slice_idx is None:
        slice_idx = default
        if slice_idx is None:
            raise ValueError(
                f"no contour of {name} has the whole {end} rim on the opening's "
                f"side; choose one with {arg}"
            )
        how = "nearest the opening with the whole rim on its side"
    elif not 0 <= slice_idx < len(contours):
        raise ValueError(
            f"{arg}={slice_idx} is out of range ({name} has {len(contours)} contours)"
        )
    else:
        how = f"from {arg}"

    # The side of the plane away from the opening stays.
    keep = end
    before = _branch_sizes(results)
    try:
        cut = cut_labelled_mesh(results, contours[slice_idx], keep=keep, cap=False)
    except ValueError as err:
        raise ValueError(
            f"cutting the {end} end at {name} contour {slice_idx} failed ({err}); "
            f"choose another contour with {arg}"
        ) from err
    updated = cut.results

    key = next(
        k
        for k in results
        if k.startswith(BOUNDARY_RING_PREFIX)
        and [tuple(p) for p in results[k]] == list(ring)
    )
    if updated[key]:
        raise ValueError(
            f"the cut at {name} contour {slice_idx} leaves {len(updated[key])} "
            f"points of the old {end} rim behind; choose a contour further from "
            f"the opening with {arg}"
        )
    updated[key] = cut.ring
    updated["boundary_points"] = [
        p
        for k in sorted(
            (k for k in updated if k.startswith(BOUNDARY_RING_PREFIX)),
            key=lambda k: int(k[len(BOUNDARY_RING_PREFIX) :]),
        )
        for p in updated[k]
    ]

    lost = {
        b: n - after
        for b, n in before.items()
        if b != (vessel, branch_id) and (after := _branch_sizes(updated).get(b, 0)) < n
    }
    for (v, b), n in lost.items():
        print(
            f"Warning: the {end} cut also removed {n} vertices of {v.upper()} "
            f"branch {b}, which leaves the stretch between the old rim and the "
            f"plane; choose a contour nearer the opening with {arg} to keep it"
        )
    print(
        f"Cut the {end} end at {name} contour {slice_idx} ({how}): open rim of "
        f"{len(cut.ring)} points"
    )
    return updated, slice_idx


def _branch_of_ring(ring: list[tuple], results: dict) -> tuple[str, int]:
    """``(vessel, branch id)`` most of *ring*'s vertices are labelled with."""
    pts = set(map(tuple, ring))
    votes = {
        (vessel, int(b)): len(pts.intersection(map(tuple, branch)))
        for vessel in VESSELS
        for b, branch in results.get(branches_key(vessel), {}).items()
    }
    best = max(votes, key=lambda k: votes[k], default=None)
    if best is None or votes[best] == 0:
        raise ValueError(
            "cutting a clean rim needs the branch labels on the rim's vertices "
            "(run label_branches_pair before the removal)"
        )
    return best


def _branch_sizes(results: dict) -> dict[tuple[str, int], int]:
    return {
        (vessel, int(b)): len(pts)
        for vessel in VESSELS
        for b, pts in results.get(branches_key(vessel), {}).items()
    }


def _contour_plane(contour: PyContour) -> tuple[np.ndarray, np.ndarray]:
    """Centroid and unit normal (pointing distally) of a tree contour, as the
    cut takes them (Newell normal about the centroid)."""
    pts = np.asarray(contour.points_as_tuples(), dtype=np.float64)
    origin = np.asarray(contour.centroid, dtype=np.float64)
    rel = pts - origin
    normal = np.cross(rel, np.roll(rel, -1, axis=0)).sum(axis=0)
    return origin, normal / np.linalg.norm(normal)


def _nearest_clear_contour(
    contours: list[PyContour], ring: list[tuple], end: str
) -> int | None:
    """Index of the contour nearest the opening whose plane has the whole rim
    on the opening's side, walking away from the opening from the contour
    nearest the rim (distally for the distal rim, proximally for the proximal
    one).  A plane is unbounded, so far along a curving vessel another one can
    have the rim on either side; starting at the rim avoids those."""
    pts = np.asarray(ring, dtype=np.float64)
    centroids = np.asarray([c.centroid for c in contours], dtype=np.float64)
    start = int(np.argmin(np.linalg.norm(centroids - pts.mean(axis=0), axis=1)))
    if end == "distal":
        order, sign = range(start, len(contours)), 1.0
    else:
        order, sign = range(start, -1, -1), -1.0
    for k in order:
        origin, normal = _contour_plane(contours[k])
        if (sign * (pts - origin) @ normal < 0.0).all():
            return k
    return None
