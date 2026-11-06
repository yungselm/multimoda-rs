"""Reconnect side branches that leave the replaced segment to the intravascular wall.

When the anomalous segment is cut out of the CCTA mesh, a side branch leaving
it loses its junction: it stays in the mesh as a loose piece (see
:func:`~multimodars.label_anomalous_region`, which leaves such branches in
``"off_path_points"``).  :func:`reconnect_side_branches` joins every such piece
to the intravascular wall that replaces the segment:

1. The branch's most proximal contour in the discretized tree (contour 0,
   where it leaves the vessel the intravascular data was aligned to) gives the
   hole: it is projected towards the intravascular axis and wrapped onto the
   wall facing the branch, and the wall inside that footprint is removed.  A
   branch wider than the intravascular vessel is narrowed to at most
   *max_hole_fraction* of its circumference.  The hole's rim is snapped onto
   the footprint's outline, since the intravascular grid is coarse along the
   vessel (one ring per frame).
2. The CCTA piece is cut at the plane of the branch's first contour that has
   the whole rim left by the removal behind it (and lies outside the
   intravascular lumen), and everything between that plane and the parent
   vessel is removed.  The cut leaves a clean, round rim.
3. The rim of the cut branch and the rim of the hole are bridged by a straight
   triangle strip.

The tree must match the mesh: after scaling the mesh, move it along with
:func:`~multimodars.ccta.scaling.sync_tree_to_mesh`.
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

from .boundary import clean_open_boundary, open_boundary_edges, order_boundary_rings

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
    max_hole_fraction: float = 0.5,
    max_ring_spacing_mm: float = 0.25,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh, list[trimesh.Trimesh]]:
    """Join loose CCTA side branches to the intravascular wall.

    Parameters
    ----------
    ccta_mesh : trimesh.Trimesh
        CCTA mesh with the segment removed.  A connected piece other than the
        largest, most of whose vertices are ``"off_path_points"``, is a side
        branch cut off by the removal.  Other pieces (e.g. the distal vessel,
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
        Discretized tree of the mesh before the removal, synced to any scaling
        since.  Contour 0 of each branch gives the hole, a later contour the
        cut.
    clearance_mm : float, optional
        How far outside the intravascular lumen the cut contour must lie
        (default 0.3).
    max_hole_fraction : float, optional
        Largest share of the intravascular circumference a hole may span
        (default 0.5); a wider footprint is narrowed towards the branch side.
    max_ring_spacing_mm : float, optional
        The intravascular frames are often a millimetre or more apart, so a
        hole would cover a single ring.  Before cutting, rings are interpolated
        between the frames until they are at most this far apart (default
        0.25).  The surface stays the same and the end rings stay where they
        are, so the proximal and distal patches still join it.

    Returns
    -------
    (trimesh.Trimesh, trimesh.Trimesh, list of trimesh.Trimesh)
        The CCTA mesh with the pieces cut at their starting frames, the
        intravascular mesh with the holes (densified, their rim vertices moved
        onto the footprints; *iv_mesh* itself when nothing is reconnected), and
        the bridging strips.  Vertices
        are shared by coordinate, so the parts join on ``merge_vertices``.
        A piece that cannot be reconnected is left as it is, with a warning.
    """
    rings = _densify_rings(
        np.array([f.lumen.points_as_tuples() for f in iv_frames], dtype=np.float64),
        max_ring_spacing_mm,
    )
    lumen = _Lumen(rings)
    n_per_frame = rings.shape[1]
    ccta_faces = np.asarray(ccta_mesh.faces).copy()
    iv_vertices = rings.reshape(-1, 3).copy()
    iv_faces = _tube_faces(rings)
    patches: list[trimesh.Trimesh] = []
    off_path = set(map(tuple, results.get("off_path_points", [])))

    hole_taken = np.zeros(len(iv_vertices), dtype=bool)

    for piece in _loose_pieces(ccta_mesh):
        coords = [tuple(v) for v in ccta_mesh.vertices[piece]]
        if 2 * sum(c in off_path for c in coords) <= len(coords):
            continue
        found = _branch_of_piece(ccta_mesh.vertices[piece], results, tree)
        if found is None:
            warnings.warn(
                f"a loose CCTA piece of {len(piece)} vertices belongs to no labelled "
                "side branch, left as it is",
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
        if len(contours) < 2:
            warnings.warn(
                f"{name}: fewer than two contours, not reconnected",
                UserWarning,
                stacklevel=2,
            )
            continue
        origin = float(lumen.position(contours[0].mean(axis=0)[None])[0])
        if not 0.0 <= origin <= lumen.arclength[-1]:
            gap, where = (
                (-origin, "before")
                if origin < 0.0
                else (origin - lumen.arclength[-1], "after")
            )
            warnings.warn(
                f"{name}: it leaves its parent {gap:.1f} mm {where} the intravascular "
                "segment, where there is no intravascular "
                "wall to join it to, not reconnected",
                UserWarning,
                stacklevel=2,
            )
            continue
        removal_rim = _removal_rim(ccta_mesh.vertices, ccta_faces, piece, contours[0])
        if removal_rim is None:
            warnings.warn(
                f"{name}: the piece has no open rim, not reconnected",
                UserWarning,
                stacklevel=2,
            )
            continue
        start = _cut_contour(contours, removal_rim, lumen, clearance_mm)
        if start is None:
            warnings.warn(
                f"{name}: no contour has the removal rim behind it and lies outside "
                "the intravascular lumen, not reconnected",
                UserWarning,
                stacklevel=2,
            )
            continue
        ring = contours[start]
        center = ring.mean(axis=0)
        axis = _contour_axis(contours, start)

        footprint = _Footprint(lumen, contours[0], center, max_hole_fraction)
        if footprint.narrowed < 1.0:
            print(
                f"{name}: narrowed the hole to {footprint.narrowed:.0%} of its width "
                f"({max_hole_fraction:.0%} of the intravascular circumference)"
            )
        cut = _cut_footprint(iv_vertices, iv_faces, n_per_frame, footprint, hole_taken)
        if isinstance(cut, str):
            warnings.warn(f"{name}: {cut}, not reconnected", UserWarning, stacklevel=2)
            continue
        new_iv_faces, hole, hole_rim = cut

        new_ccta_faces, branch_rim = _cut_piece(
            ccta_mesh.vertices, ccta_faces, piece, center, axis, ring
        )
        if branch_rim is None:
            warnings.warn(
                f"{name}: the cut branch has no clean rim, not reconnected",
                UserWarning,
                stacklevel=2,
            )
            continue
        ccta_faces, iv_faces = new_ccta_faces, new_iv_faces
        hole_taken[hole] = True

        branch_pts = ccta_mesh.vertices[branch_rim]
        iv_vertices[hole_rim] = footprint.snap(
            iv_vertices[hole_rim], np.asarray(hole_rim) // n_per_frame
        )
        bridge_axis = branch_pts.mean(axis=0) - iv_vertices[hole_rim].mean(axis=0)
        bridge_axis /= np.linalg.norm(bridge_axis)
        hole_pts = _match_ring(iv_vertices[hole_rim], branch_pts, bridge_axis)
        patches.append(_bridge(branch_pts, hole_pts))
        print(
            f"Reconnected {name}: hole from contour 0, cut at contour {start}, rims of "
            f"{len(branch_pts)} and {len(hole_pts)} points"
        )

    if not patches:
        return ccta_mesh, iv_mesh, patches
    return (
        _with_faces(ccta_mesh.vertices, ccta_faces),
        _with_faces(iv_vertices, iv_faces),
        patches,
    )


def _densify_rings(rings: np.ndarray, max_spacing: float) -> np.ndarray:
    """*rings* (frames, points, 3) with rings interpolated between neighbouring
    frames until their centroids are at most *max_spacing* apart."""
    centroids = rings.mean(axis=1)
    gaps = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
    steps = np.maximum(1, np.ceil(gaps / max_spacing).astype(int))
    out = [rings[:1]]
    for i, n in enumerate(steps):
        t = (np.arange(1, n + 1) / n)[:, None, None]
        out.append((1.0 - t) * rings[i] + t * rings[i + 1])
    return np.concatenate(out)


def _tube_faces(rings: np.ndarray) -> np.ndarray:
    """Quad-strip faces between consecutive rings, wound like ``geometry_to_trimesh``
    (outward)."""
    k, n = rings.shape[:2]
    i = np.arange(k - 1)[:, None] * n
    j = np.arange(n)[None]
    a, b = i + j, i + (j + 1) % n
    c, d = b + n, a + n
    faces = np.concatenate(
        [np.stack([a, b, d], -1).reshape(-1, 3), np.stack([b, c, d], -1).reshape(-1, 3)]
    )
    v = rings.reshape(-1, 3)
    tri = v[faces[0]]
    normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    if np.dot(normal, tri.mean(axis=0) - rings[0].mean(axis=0)) < 0:
        faces = faces[:, ::-1]
    return faces


class _Lumen:
    """The intravascular lumen as rings: centroid, axis and contour per ring."""

    def __init__(self, rings: np.ndarray) -> None:
        self.centroids = rings.mean(axis=1)
        tangents = np.gradient(self.centroids, axis=0)
        self.tangents = tangents / np.linalg.norm(tangents, axis=1, keepdims=True)
        self.rings = list(rings - self.centroids[:, None])
        steps = np.linalg.norm(np.diff(self.centroids, axis=0), axis=1)
        self.spacing = float(steps.mean()) if len(steps) else 1.0
        self.arclength = np.concatenate([[0.0], np.cumsum(steps)])
        self.radii = np.array([np.linalg.norm(r, axis=1).mean() for r in self.rings])
        self.tree = cKDTree(self.centroids)

    def position(self, points: np.ndarray) -> np.ndarray:
        """Arc length along the lumen axis of each point, from the first ring."""
        frames = self.nearest_frame(points)
        along = ((points - self.centroids[frames]) * self.tangents[frames]).sum(axis=1)
        return self.arclength[frames] + along

    def basis(self, direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per frame, *direction* made normal to the frame's tangent, and the
        tangent crossed with that: the two in-plane axes of each frame."""
        d = direction[None] - (self.tangents @ direction)[:, None] * self.tangents
        # Zero only where *direction* runs along the vessel, far from the branch.
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        return d, np.cross(self.tangents, d)

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

    The piece holds the cut-off branch and the branches growing out of it.  Its
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
    cut = np.isin(faces, drop).any(axis=1)
    # Seeded from the cut faces only, so an open vessel end is not merged in.
    rim = set(np.setdiff1d(faces[cut].ravel(), drop).tolist())
    faces = faces[~cut]
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


def _removal_rim(
    vertices: np.ndarray,
    faces: np.ndarray,
    piece: np.ndarray,
    first_contour: np.ndarray,
) -> np.ndarray | None:
    """The piece's open rim left by the removal: of its open rims (the removal's,
    and any open vessel ends), the one nearest the branch's first contour."""
    piece_set = set(piece.tolist())
    seeds = {int(v) for v in open_boundary_edges(faces).ravel() if int(v) in piece_set}
    rings = order_boundary_rings(faces, vertices, seeds)
    if not rings:
        return None
    origin = first_contour.mean(axis=0)
    ring = min(rings, key=lambda r: np.linalg.norm(vertices[r].mean(axis=0) - origin))
    return vertices[ring]


def _contour_axis(contours: list[np.ndarray], i: int) -> np.ndarray:
    """Branch direction at contour *i*, away from the parent."""
    if i + 1 < len(contours):
        axis = contours[i + 1].mean(axis=0) - contours[i].mean(axis=0)
    else:
        axis = contours[i].mean(axis=0) - contours[i - 1].mean(axis=0)
    return axis / np.linalg.norm(axis)


def _cut_contour(
    contours: list[np.ndarray],
    removal_rim: np.ndarray,
    lumen: _Lumen,
    clearance_mm: float,
) -> int | None:
    """Index of the first contour whose plane has the whole removal rim on the
    parent's side, so a cut there leaves a clean ring, and which lies outside
    the intravascular lumen."""
    for i in range(1, len(contours)):
        center = contours[i].mean(axis=0)
        behind = (removal_rim - center) @ _contour_axis(contours, i) < 0.0
        if behind.all() and not lumen.inside(contours[i], clearance_mm).any():
            return i
    return None


class _Footprint:
    """A branch contour wrapped onto the intravascular wall facing the branch.

    A point on the wall is given as ``(s, a)``: arc length along the lumen
    axis, and arc length around it from the side facing the branch, measured
    on a circle of the lumen's radius at the contour.  The contour is
    projected towards the lumen axis and its offset across the lumen is laid
    around the wall as arc length, so a branch wider than the lumen still
    gives a hole on the facing side.  Its arc is narrowed towards the branch
    side to at most *max_fraction* of the circumference.
    """

    def __init__(
        self,
        lumen: _Lumen,
        contour: np.ndarray,
        towards: np.ndarray,
        max_fraction: float,
    ) -> None:
        self.lumen = lumen
        frame = int(lumen.nearest_frame(contour.mean(axis=0)[None])[0])
        self.d, self.e = lumen.basis(towards - lumen.centroids[frame])
        self.radius = float(lumen.radii[frame])
        frames = lumen.nearest_frame(contour)
        q = contour - lumen.centroids[frames]
        s = lumen.position(contour)
        a = (q * self.e[frames]).sum(axis=1) * self.radius / lumen.radii[frames]
        limit = np.pi * max_fraction * self.radius
        widest = np.abs(a).max()
        self.narrowed = min(1.0, limit / widest) if widest > 0 else 1.0
        self.outline = np.column_stack([s, a * self.narrowed])
        self.center = self.outline.mean(axis=0)
        rel = self.outline - self.center
        angles = np.arctan2(rel[:, 1], rel[:, 0])
        order = np.argsort(angles)
        self._angles = angles[order]
        self._radii = np.linalg.norm(rel, axis=1)[order]
        self._ring_angles: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def wall_coords(self, points: np.ndarray, frames: np.ndarray) -> np.ndarray:
        """``(s, a)`` of wall *points*, each on the ring of its frame."""
        lumen = self.lumen
        q = points - lumen.centroids[frames]
        s = lumen.arclength[frames] + (q * lumen.tangents[frames]).sum(axis=1)
        theta = np.arctan2(
            (q * self.e[frames]).sum(axis=1), (q * self.d[frames]).sum(axis=1)
        )
        return np.column_stack([s, theta * self.radius])

    def contains(self, sa: np.ndarray) -> np.ndarray:
        return _in_polygon(sa, self.outline)

    def _wall_radius(self, frame: int, theta: np.ndarray) -> np.ndarray:
        if frame not in self._ring_angles:
            ring = self.lumen.rings[frame]
            angles = np.arctan2(ring @ self.e[frame], ring @ self.d[frame])
            order = np.argsort(angles)
            self._ring_angles[frame] = (
                angles[order],
                np.linalg.norm(ring, axis=1)[order],
            )
        angles, radii = self._ring_angles[frame]
        return np.interp(theta, angles, radii, period=2 * np.pi)

    def _wall_point(self, frame: int, theta: np.ndarray) -> np.ndarray:
        r = self._wall_radius(frame, theta)
        return self.lumen.centroids[frame] + r[:, None] * (
            np.cos(theta)[:, None] * self.d[frame]
            + np.sin(theta)[:, None] * self.e[frame]
        )

    def to_wall(self, sa: np.ndarray) -> np.ndarray:
        """3D wall points at ``(s, a)``, interpolated between the two frames around each."""
        lumen = self.lumen
        k = np.interp(sa[:, 0], lumen.arclength, np.arange(len(lumen.arclength)))
        k0 = np.clip(np.floor(k).astype(int), 0, len(lumen.arclength) - 2)
        f = (k - k0)[:, None]
        theta = sa[:, 1] / self.radius
        out = np.empty((len(sa), 3))
        for frame in np.unique(k0):
            m = k0 == frame
            p0 = self._wall_point(frame, theta[m])
            p1 = self._wall_point(frame + 1, theta[m])
            out[m] = (1.0 - f[m]) * p0 + f[m] * p1
        return out

    def snap(self, ring: np.ndarray, frames: np.ndarray) -> np.ndarray:
        """The hole's rim, *ring* (ordered wall points), laid onto the outline.

        The points keep their order and their share of the rim's length: they
        are spread along the outline, the same way round, from where the first
        one projects (radially from the footprint's center).  Projecting every
        point radially instead can drop two of them onto one spot where the
        grid's rim zig-zags, leaving a sliver in the bridge.
        """
        sa = self.wall_coords(ring, frames)
        outline = self.outline
        if np.sign(_area_2d(sa)) != np.sign(_area_2d(outline)):
            outline = outline[::-1]
        closed = np.vstack([outline, outline[:1]])
        along = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]
        )
        steps = np.linalg.norm(np.diff(np.vstack([sa, sa[:1]]), axis=0), axis=1)
        share = np.concatenate([[0.0], np.cumsum(steps)[:-1]]) / steps.sum()
        rel = sa[0] - self.center
        angle = np.arctan2(rel[1], rel[0])
        reach = np.interp(angle, self._angles, self._radii, period=2 * np.pi)
        first = self.center + reach * np.array([np.cos(angle), np.sin(angle)])
        start = along[np.argmin(np.linalg.norm(outline - first, axis=1))]
        at = (start + share * along[-1]) % along[-1]
        target = np.column_stack(
            [np.interp(at, along, closed[:, 0]), np.interp(at, along, closed[:, 1])]
        )
        return self.to_wall(target)


def _cut_footprint(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_per_frame: int,
    footprint: _Footprint,
    taken: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[int]] | str:
    """Remove the intravascular wall inside *footprint*.

    Returns the remaining faces, the removed vertices and the hole's rim, or why
    there is no hole.  *taken* marks the vertices of earlier holes, which a new
    one must not reach.
    """
    frames = np.arange(len(vertices)) // n_per_frame
    hole = np.flatnonzero(footprint.contains(footprint.wall_coords(vertices, frames)))
    if len(hole) == 0:
        return "its footprint covers no intravascular vertex"
    rows = frames[hole]
    last = len(footprint.lumen.centroids) - 1
    if rows.min() <= 1 or rows.max() >= last - 1:
        return "its footprint reaches the end of the intravascular segment"
    if taken[hole].any():
        return "its footprint overlaps the hole of another branch"

    kept = faces[~np.isin(faces, hole).any(axis=1)]
    removed_faces = faces[np.isin(faces, hole).any(axis=1)]
    seeds = set(np.setdiff1d(removed_faces.ravel(), hole).tolist())
    extra, rings = clean_open_boundary(kept, vertices, seeds, target_n=1)
    if extra:
        if np.isin(np.asarray(list(extra)) // n_per_frame, [0, last]).any():
            return (
                "cleaning its rim would cut into the end of the intravascular segment"
            )
        kept = kept[~np.isin(kept, list(extra)).any(axis=1)]
        hole = np.union1d(hole, list(extra))
    if not rings:
        return "the hole has no clean rim"
    return kept, hole, rings[0]


def _area_2d(xy: np.ndarray) -> float:
    """Signed area (twice) of the closed polygon *xy*, positive anticlockwise."""
    x, y = xy[:, 0], xy[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _signed_area(pts: np.ndarray, axis: np.ndarray) -> float:
    e1, e2 = _plane_basis(axis)
    return _area_2d(np.column_stack([pts @ e1, pts @ e2]))


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
