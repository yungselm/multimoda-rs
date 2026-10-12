"""Reconnect side branches that leave the replaced segment to the intravascular wall.

When the anomalous segment is cut out of the CCTA mesh, a side branch leaving
it loses its junction: it stays in the mesh as a loose piece (see
:func:`~multimodars.label_anomalous_region`, which leaves such branches in
``"off_path_points"``).  :func:`reconnect_side_branches` joins every such piece
to the intravascular wall that replaces the segment:

1. The branch's first contour in the discretized tree that lies fully outside
   the intravascular lumen is its starting frame.
2. The CCTA piece is cut at that contour's plane; everything between the plane
   and the parent vessel is removed.
3. The contour is projected along the branch axis onto the intravascular wall
   facing it, and the wall inside that footprint is removed.  The hole's rim
   is snapped onto the footprint's outline (on the wall), since the
   intravascular grid is coarse along the vessel (one ring per frame).
4. The rim of the cut branch and the rim of the hole are bridged by a straight
   triangle strip.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .boundary import clean_open_boundary, open_boundary_edges

if TYPE_CHECKING:
    from ...multimodars import PyDiscretizedVesselTree, PyFrame

_VESSELS = ("rca", "lca")
_MIN_PIECE_VERTICES = 20


def reconnect_side_branches(
    ccta_mesh: trimesh.Trimesh,
    iv_mesh: trimesh.Trimesh,
    iv_frames: list[PyFrame],
    results: dict,
    tree: PyDiscretizedVesselTree,
    clearance_mm: float = 0.3,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh, list[trimesh.Trimesh]]:
    """Join loose CCTA side branches to the intravascular wall.

    Parameters
    ----------
    ccta_mesh : trimesh.Trimesh
        CCTA mesh with the segment removed.  A connected piece other than the
        largest, most of whose vertices are ``"off_path_points"``, is a side
        branch cut off by the removal; other pieces (e.g. the distal vessel,
        which the stitch joins) are left alone.
    iv_mesh : trimesh.Trimesh
        Intravascular lumen surface, one ring of vertices per frame (as built
        by ``geometry_to_trimesh`` from *iv_frames*).
    iv_frames : list of PyFrame
        The frames *iv_mesh* was built from, proximal to distal.
    results : dict
        Labelled results with the per-branch labels (``"rca_branches"`` /
        ``"lca_branches"``), naming the branch of each piece.
    tree : PyDiscretizedVesselTree
        Discretized tree of the mesh before the removal; its branch contours
        give the starting frames.
    clearance_mm : float, optional
        How far outside the intravascular lumen the starting contour must lie
        (default 0.3).

    Returns
    -------
    (trimesh.Trimesh, trimesh.Trimesh, list of trimesh.Trimesh)
        The CCTA mesh with the pieces cut at their starting frames, the
        intravascular mesh with the holes (their rim vertices moved onto the
        footprints), and the bridging strips.  Vertices
        are shared by coordinate, so the parts join on ``merge_vertices``.
        A piece that cannot be reconnected is left as it is, with a warning.
    """
    lumen = _Lumen(iv_frames)
    n_per_frame = len(iv_frames[0].lumen.points)
    ccta_faces = np.asarray(ccta_mesh.faces).copy()
    iv_vertices = np.asarray(iv_mesh.vertices).copy()
    iv_faces = np.asarray(iv_mesh.faces).copy()
    patches: list[trimesh.Trimesh] = []
    off_path = set(map(tuple, results.get("off_path_points", [])))

    for piece in _loose_pieces(ccta_mesh):
        coords = [tuple(v) for v in ccta_mesh.vertices[piece]]
        if 2 * sum(c in off_path for c in coords) <= len(coords):
            continue
        found = _branch_of_piece(ccta_mesh.vertices[piece], results, tree)
        if found is None:
            warnings.warn(
                f"a loose CCTA piece of {len(piece)} vertices belongs to no labelled "
                "side branch; left as it is",
                UserWarning,
                stacklevel=2,
            )
            continue
        vessel, branch_id = found
        name = f"{vessel.upper()} branch {branch_id}"
        contours = [
            np.asarray(c.points_as_tuples(), dtype=np.float64)
            for c in getattr(tree, vessel)[branch_id].contours
        ]
        start = next(
            (
                i
                for i, c in enumerate(contours)
                if not lumen.inside(c, clearance_mm).any()
            ),
            None,
        )
        if start is None:
            warnings.warn(
                f"{name}: no contour lies outside the intravascular lumen; not reconnected",
                UserWarning,
                stacklevel=2,
            )
            continue
        ring = contours[start]
        center = ring.mean(axis=0)
        if start + 1 < len(contours):
            axis = contours[start + 1].mean(axis=0) - center
        else:
            axis = center - contours[start - 1].mean(axis=0)
        axis /= np.linalg.norm(axis)

        ccta_faces, branch_rim = _cut_piece(
            ccta_mesh.vertices, ccta_faces, piece, center, axis, ring
        )
        if branch_rim is None:
            warnings.warn(
                f"{name}: the cut branch has no clean rim; not reconnected",
                UserWarning,
                stacklevel=2,
            )
            continue
        cut = _cut_footprint(
            iv_vertices, iv_faces, n_per_frame, lumen, center, axis, ring
        )
        if isinstance(cut, str):
            warnings.warn(f"{name}: {cut}; not reconnected", UserWarning, stacklevel=2)
            continue
        iv_faces, hole_rim = cut

        branch_pts = ccta_mesh.vertices[branch_rim]
        iv_vertices[hole_rim] = _snap_to_footprint(
            iv_vertices[hole_rim], lumen, center, axis, ring
        )
        hole_pts = _match_ring(iv_vertices[hole_rim], branch_pts, axis)
        patches.append(_bridge(branch_pts, hole_pts))
        print(
            f"Reconnected {name}: contour {start}, rims of {len(branch_pts)} and "
            f"{len(hole_pts)} points"
        )

    return (
        _with_faces(ccta_mesh.vertices, ccta_faces),
        _with_faces(iv_vertices, iv_faces),
        patches,
    )


class _Lumen:
    """The intravascular lumen as frames: centroid, axis and contour per frame."""

    def __init__(self, frames: list[PyFrame]) -> None:
        self.centroids = np.array([f.centroid for f in frames], dtype=np.float64)
        tangents = np.gradient(self.centroids, axis=0)
        self.tangents = tangents / np.linalg.norm(tangents, axis=1, keepdims=True)
        self.rings = [
            np.asarray(f.lumen.points_as_tuples(), dtype=np.float64) - c
            for f, c in zip(frames, self.centroids)
        ]
        steps = np.linalg.norm(np.diff(self.centroids, axis=0), axis=1)
        self.spacing = float(steps.mean()) if len(steps) else 1.0
        self.tree = cKDTree(self.centroids)

    def nearest_frame(self, points: np.ndarray) -> np.ndarray:
        return self.tree.query(points)[1]

    def _local(self, points: np.ndarray):
        """Nearest frame, position along its axis, radial offset and wall radius
        towards each point (the radius of the contour point most in its direction)."""
        frame = self.nearest_frame(points)
        q = points - self.centroids[frame]
        t = self.tangents[frame]
        along = (q * t).sum(axis=1)
        radial = q - along[:, None] * t
        wall = np.empty(len(points))
        for i, (f, d) in enumerate(zip(frame, radial)):
            ring = self.rings[f]
            radii = np.linalg.norm(ring, axis=1)
            wall[i] = radii[np.argmax(ring @ d / np.maximum(radii, 1e-12))]
        return frame, along, radial, wall

    def outside_by(self, points: np.ndarray) -> np.ndarray:
        """Radial distance of each point outside the wall (negative inside)."""
        _, _, radial, wall = self._local(points)
        return np.linalg.norm(radial, axis=1) - wall

    def hit_along(
        self, starts: np.ndarray, direction: np.ndarray, iterations: int = 30
    ) -> np.ndarray:
        """Where the lines from *starts* (outside the lumen) along *direction*
        first meet the wall, found by bisection down to the point where each
        line passes closest to the lumen axis."""
        frame = self.nearest_frame(starts)
        deepest = ((self.centroids[frame] - starts) @ direction)[:, None]
        lo, hi = deepest, np.zeros_like(deepest)
        for _ in range(iterations):
            mid = (lo + hi) / 2
            outside = self.outside_by(starts + mid * direction) > 0
            lo = np.where(outside[:, None], lo, mid)
            hi = np.where(outside[:, None], mid, hi)
        return starts + hi * direction

    def inside(self, points: np.ndarray, margin: float) -> np.ndarray:
        """Whether each point lies inside the lumen, or within *margin* of its wall."""
        frame, along, radial, wall = self._local(points)
        r = np.linalg.norm(radial, axis=1)
        last = len(self.centroids) - 1
        beyond = ((frame == 0) & (along < -self.spacing / 2)) | (
            (frame == last) & (along > self.spacing / 2)
        )
        return (r < wall + margin) & ~beyond


def _loose_pieces(mesh: trimesh.Trimesh) -> list[np.ndarray]:
    """Vertex indices of every connected piece but the largest."""
    n = len(mesh.vertices)
    edges = mesh.faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    used = np.zeros(n, dtype=bool)
    used[mesh.faces.ravel()] = True
    sizes = np.bincount(labels[used], minlength=labels.max() + 1)
    main = int(np.argmax(sizes))
    return [
        np.flatnonzero(used & (labels == c))
        for c in np.flatnonzero(sizes >= _MIN_PIECE_VERTICES)
        if c != main
    ]


def _branch_of_piece(
    coords: np.ndarray, results: dict, tree: PyDiscretizedVesselTree
) -> tuple[str, int] | None:
    """``(vessel, branch id)`` of the side branch a loose piece starts with.

    The piece holds the cut-off branch and the branches growing out of it; its
    branch is the one whose parent is not in the piece, with the most vertices.
    """
    piece = set(map(tuple, coords))
    best: tuple[int, str, int] | None = None
    for vessel in _VESSELS:
        nodes = getattr(tree, vessel)
        votes = {
            k: len(piece.intersection(map(tuple, pts)))
            for k, pts in results.get(f"{vessel}_branches", {}).items()
        }
        present = {k for k, v in votes.items() if v}
        for k in present:
            node = nodes.get(k)
            if k == 0 or node is None:
                continue
            if node.parent is not None and node.parent[0] in present:
                continue
            if best is None or votes[k] > best[0]:
                best = (votes[k], vessel, k)
    return None if best is None else (best[1], best[2])


def _plane_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seed = (
        np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    )
    e1 = np.cross(axis, seed)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(axis, e1)


def _cut_piece(
    vertices: np.ndarray,
    faces: np.ndarray,
    piece: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
    ring: np.ndarray,
) -> tuple[np.ndarray, list[int] | None]:
    """Cut *piece* at the plane through *center* normal to *axis*.

    Returns the remaining faces and the piece's rim, ordered.
    """
    q = vertices[piece] - center
    along = q @ axis
    lateral = np.linalg.norm(q - along[:, None] * axis, axis=1)
    radius = np.linalg.norm(ring - center, axis=1).max()
    # Only the branch's own tube: a nested branch curving back stays.
    drop = piece[(along < 0.0) & (lateral < 3.0 * radius)]
    faces = faces[~np.isin(faces, drop).any(axis=1)]

    piece_set = set(piece.tolist())
    rim = {int(v) for v in open_boundary_edges(faces).ravel() if int(v) in piece_set}
    if not rim:
        return faces, None
    extra, rings = clean_open_boundary(faces, vertices, rim, target_n=1)
    if extra:
        faces = faces[~np.isin(faces, list(extra)).any(axis=1)]
    return faces, (rings[0] if rings else None)


def _in_polygon(xy: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd test of points *xy* (N, 2) against the closed polygon *poly* (M, 2)."""
    x, y = xy[:, :1], xy[:, 1:]
    x1, y1 = poly[None, :, 0], poly[None, :, 1]
    x2, y2 = np.roll(poly[:, 0], -1)[None], np.roll(poly[:, 1], -1)[None]
    crosses = (y1 > y) != (y2 > y)
    dy = np.where(y2 != y1, y2 - y1, 1e-12)
    x_at = x1 + (y - y1) * (x2 - x1) / dy
    return (crosses & (x < x_at)).sum(axis=1) % 2 == 1


def _cut_footprint(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_per_frame: int,
    lumen: _Lumen,
    center: np.ndarray,
    axis: np.ndarray,
    ring: np.ndarray,
) -> tuple[np.ndarray, list[int]] | str:
    """Remove the intravascular wall inside the footprint of *ring* projected along
    *axis*, on the side of the wall facing the branch.

    Returns the remaining faces and the hole's rim, or why there is no hole.
    """
    e1, e2 = _plane_basis(axis)
    poly = np.column_stack([(ring - center) @ e1, (ring - center) @ e2])
    q = vertices - center
    in_footprint = _in_polygon(np.column_stack([q @ e1, q @ e2]), poly)

    frame = lumen.nearest_frame(vertices)
    t = lumen.tangents[frame]
    # The branch direction across the intravascular axis, frame by frame.
    towards_branch = axis[None] - (t @ axis)[:, None] * t
    radial = vertices - lumen.centroids[frame]
    facing = (radial * towards_branch).sum(axis=1) > 0.0
    hole = np.flatnonzero(in_footprint & facing & (q @ axis < 0.0))
    if len(hole) == 0:
        return "its footprint covers no intravascular vertex"
    rows = hole // n_per_frame
    if rows.min() == 0 or rows.max() == len(lumen.centroids) - 1:
        return "its footprint reaches the end of the intravascular segment"

    kept = faces[~np.isin(faces, hole).any(axis=1)]
    removed_faces = faces[np.isin(faces, hole).any(axis=1)]
    seeds = set(np.setdiff1d(removed_faces.ravel(), hole).tolist())
    extra, rings = clean_open_boundary(kept, vertices, seeds, target_n=1)
    if extra:
        kept = kept[~np.isin(kept, list(extra)).any(axis=1)]
    if not rings:
        return "the hole has no clean rim"
    return kept, rings[0]


def _snap_to_footprint(
    points: np.ndarray,
    lumen: _Lumen,
    center: np.ndarray,
    axis: np.ndarray,
    ring: np.ndarray,
) -> np.ndarray:
    """*points* (the hole's rim) moved onto the outline of *ring* projected along
    *axis*: radially onto the outline in the plane normal to *axis*, then along
    *axis* onto the wall."""
    e1, e2 = _plane_basis(axis)
    outline = np.column_stack([(ring - center) @ e1, (ring - center) @ e2])
    angles = np.arctan2(outline[:, 1], outline[:, 0])
    radii = np.linalg.norm(outline, axis=1)
    order = np.argsort(angles)
    angles, radii = angles[order], radii[order]

    q = points - center
    xy = np.column_stack([q @ e1, q @ e2])
    a = np.arctan2(xy[:, 1], xy[:, 0])
    target = np.interp(a, angles, radii, period=2 * np.pi)
    on_outline = center + target[:, None] * (
        np.cos(a)[:, None] * e1 + np.sin(a)[:, None] * e2
    )
    return lumen.hit_along(on_outline, -axis)


def _signed_area(pts: np.ndarray, axis: np.ndarray) -> float:
    e1, e2 = _plane_basis(axis)
    x, y = pts @ e1, pts @ e2
    return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _match_ring(
    ring: np.ndarray, reference: np.ndarray, axis: np.ndarray
) -> np.ndarray:
    """*ring* turned to run the same way round *axis* as *reference*, starting at
    the point nearest ``reference[0]``."""
    if np.sign(_signed_area(ring, axis)) != np.sign(_signed_area(reference, axis)):
        ring = ring[::-1]
    start = int(np.argmin(np.linalg.norm(ring - reference[0], axis=1)))
    return np.roll(ring, -start, axis=0)


def _bridge(ring_a: np.ndarray, ring_b: np.ndarray) -> trimesh.Trimesh:
    """Straight triangle strip between two matched rings."""
    from .core import _stitch_rings

    return _stitch_rings(
        [tuple(p) for p in ring_a],
        [SimpleNamespace(x=p[0], y=p[1], z=p[2]) for p in ring_b],
    )


def _with_faces(vertices: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    out = trimesh.Trimesh(vertices=vertices.copy(), faces=faces, process=False)
    out.remove_unreferenced_vertices()
    return out
