"""Reconnect side branches that leave the replaced segment to the intravascular wall.

When the anomalous segment is cut out of the CCTA mesh, a side branch leaving
it loses its junction: it stays in the mesh as a loose piece (see
:func:`~multimodars.label_anomalous_region`, which leaves such branches in
``"off_path_points"``).  :func:`reconnect_side_branches` joins every such piece
to the intravascular wall that replaces the segment:

1. The CCTA piece is cut at the plane of one of the branch's contours in the
   discretized tree (contour 0 unless chosen otherwise), and everything
   between that plane and the parent vessel is removed.  The cut leaves a
   clean, planar rim.
2. That rim is moved back along the branch centerline (contour by contour to
   contour 0, then on to the intravascular axis) until it crosses the
   intravascular wall.  Where each rim point crosses gives the hole, and the
   wall inside it is removed.  A branch wider than the intravascular vessel is
   narrowed to at most *max_hole_fraction* of its circumference.  The hole's
   rim is snapped onto that outline, since the intravascular grid is coarse
   along the vessel (one ring per frame).
3. The two rims, which come from the same contour and so have about the same
   shape, are bridged by a straight triangle strip.

The tree must match the mesh: after scaling the mesh, move it along with
:func:`~multimodars.ccta.scaling.sync_tree_to_mesh`.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from ...multimodars import cut_mesh_at_contour as _cut_mesh_at_contour
from .boundary import _densify_boundary, clean_open_boundary

if TYPE_CHECKING:
    from ...multimodars import (
        PyCenterline,
        PyContour,
        PyDiscretizedVesselTree,
        PyFrame,
    )

_VESSELS = ("rca", "lca")
_MIN_PIECE_VERTICES = 20


def reconnect_side_branches(
    ccta_mesh: trimesh.Trimesh,
    iv_mesh: trimesh.Trimesh,
    iv_frames: list[PyFrame],
    results: dict,
    tree: PyDiscretizedVesselTree,
    centerlines: Mapping[str, PyCenterline],
    cut_slices: dict[tuple[str, int], int] | None = None,
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
        since.  A contour of each branch gives the cut plane.
    centerlines : mapping of str to PyCenterline
        Whole centerline of each vessel (``"rca"`` / ``"lca"``), the one
        *tree* was discretized with.  The cut rim is moved back along its
        branch's centerline, point by point to the branch's first point (on
        the parent vessel).
    cut_slices : dict of (str, int) to int, optional
        Contour index each branch is cut at, by ``(vessel, branch_id)``
        (e.g. ``{("rca", 3): 2}``); a branch not listed is cut at contour 0.
        If the cut fails (the plane crosses the rim left by the removal, or a
        junction) or the cut rim reaches into the intravascular lumen, the
        branch is left loose with a warning naming the key to set; no other
        contour is tried.
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
        The CCTA mesh with the pieces cut at their chosen contours, the
        intravascular mesh with the holes (densified, their rim vertices moved
        onto the imprints; *iv_mesh* itself when nothing is reconnected), and
        the bridging strips.  Vertices
        are shared by coordinate, so the parts join on ``merge_vertices``.
        A piece that cannot be reconnected is left as it is, with a warning.

    Notes
    -----
    Two branches leaving side by side can share a stub and so one loose
    piece; each is cut and joined on its own, and the piece is reconnected
    only if all of them can be.  A branch is refused (with a warning naming
    the other branch) if its hole would overlap or touch another's (two rims
    keep at least one wall vertex between them), or if its bridge would cut
    through another branch or its bridge.
    """
    out = reconnect_side_branches_report(
        ccta_mesh,
        iv_mesh,
        iv_frames,
        results,
        tree,
        centerlines,
        cut_slices=cut_slices,
        max_hole_fraction=max_hole_fraction,
        max_ring_spacing_mm=max_ring_spacing_mm,
    )
    return out.ccta_mesh, out.iv_mesh, out.bridges


@dataclass
class Reconnection:
    """What :func:`reconnect_side_branches_report` did."""

    #: The CCTA mesh with the joined pieces cut at their contours.
    ccta_mesh: trimesh.Trimesh
    #: The intravascular mesh with the holes.
    iv_mesh: trimesh.Trimesh
    #: The strips bridging each branch to its hole.
    bridges: list[trimesh.Trimesh]
    #: Contour each joined branch was cut at, by ``(vessel, branch_id)``.
    cut_slices: dict[tuple[str, int], int] = field(default_factory=dict)
    #: Why each branch was left loose, by ``(vessel, branch_id)``.
    not_reconnected: dict[tuple[str, int], str] = field(default_factory=dict)


def side_branch_targets(
    ccta_mesh: trimesh.Trimesh, results: dict, tree: PyDiscretizedVesselTree
) -> list[tuple[str, int]]:
    """``(vessel, branch_id)`` of every side branch the stitch would try to
    reconnect (the branches the loose ``"off_path_points"`` pieces of
    *ccta_mesh* start with), the keys of *cut_slices*."""
    return [
        key
        for piece in _off_path_pieces(ccta_mesh, results)
        for key in _root_branches(ccta_mesh.vertices[piece], results, tree)
    ]


def reconnect_side_branches_report(
    ccta_mesh: trimesh.Trimesh,
    iv_mesh: trimesh.Trimesh,
    iv_frames: list[PyFrame],
    results: dict,
    tree: PyDiscretizedVesselTree,
    centerlines: Mapping[str, PyCenterline],
    cut_slices: dict[tuple[str, int], int] | None = None,
    max_hole_fraction: float = 0.5,
    max_ring_spacing_mm: float = 0.25,
) -> Reconnection:
    """:func:`reconnect_side_branches`, also saying which contour each branch
    was cut at and why the others were left loose."""
    rings = _densify_rings(
        np.array([f.lumen.points_as_tuples() for f in iv_frames], dtype=np.float64),
        max_ring_spacing_mm,
    )
    lumen = _Lumen(rings)
    n_per_frame = rings.shape[1]
    ccta_faces = np.asarray(ccta_mesh.faces).copy()
    iv_vertices = rings.reshape(-1, 3).copy()
    iv_faces = _tube_faces(rings)
    ctx = _Context(
        lumen=lumen,
        # The wall before any hole, for the imprints.
        wall=trimesh.Trimesh(iv_vertices.copy(), iv_faces, process=False),
        tree=tree,
        centerlines=centerlines,
        cut_slices=dict(cut_slices or {}),
        max_hole_fraction=max_hole_fraction,
    )
    state = _WallState(
        iv_vertices,
        iv_faces,
        np.full(len(iv_vertices), -1),
        np.arange(len(iv_vertices)) // n_per_frame,
    )
    joined: list[_Joined] = []
    not_reconnected: dict[tuple[str, int], str] = {}

    for piece in _off_path_pieces(ccta_mesh, results):
        roots = _root_branches(ccta_mesh.vertices[piece], results, tree)
        if not roots:
            warnings.warn(
                f"a loose CCTA piece of {len(piece)} vertices belongs to no labelled "
                "side branch, left as it is",
                UserWarning,
                stacklevel=2,
            )
            continue
        # Every branch the piece starts with is reconnected, or none: cutting
        # one drops the stub the others hang off.
        trial = state.copy()
        done: list[_Joined] = []
        for vessel, branch_id in roots:
            name = f"{vessel.upper()} branch {branch_id}"
            out = _join_branch(
                ctx,
                trial,
                ccta_mesh.vertices,
                ccta_faces,
                piece,
                vessel,
                branch_id,
                [*joined, *done],
            )
            if isinstance(out, str):
                siblings = [
                    f"{v.upper()} branch {b}" for v, b in roots if b != branch_id
                ]
                also = (
                    f" (nor {', '.join(siblings)}, which leave the same loose piece)"
                    if siblings
                    else ""
                )
                warnings.warn(
                    f"{name}: {out}, not reconnected{also}",
                    UserWarning,
                    stacklevel=2,
                )
                not_reconnected[(vessel, branch_id)] = out
                for key in roots:
                    not_reconnected.setdefault(
                        key,
                        f"leaves the same loose piece as {name}, which was not reconnected",
                    )
                break
            done.append(out)
        else:
            state = trial
            ccta_faces = ccta_faces[~np.isin(ccta_faces[:, 0], piece)]
            joined += done
            for j in done:
                print(
                    f"Reconnected {j.name}: cut at contour {j.slice_idx}, rims of "
                    f"{j.n_ring} and {j.n_rim} points"
                )

    if not joined:
        return Reconnection(ccta_mesh, iv_mesh, [], {}, not_reconnected)
    ccta_out = cast(
        trimesh.Trimesh,
        trimesh.util.concatenate(
            [_with_faces(ccta_mesh.vertices, ccta_faces), *(j.piece for j in joined)]
        ),
    )
    return Reconnection(
        ccta_out,
        _with_faces(state.vertices, state.faces),
        [j.bridge for j in joined],
        {j.key: j.slice_idx for j in joined},
        not_reconnected,
    )


@dataclass
class _Context:
    """What every branch of :func:`reconnect_side_branches` is joined with."""

    lumen: _Lumen
    wall: trimesh.Trimesh
    tree: PyDiscretizedVesselTree
    centerlines: Mapping[str, PyCenterline]
    cut_slices: dict[tuple[str, int], int]
    max_hole_fraction: float


@dataclass
class _WallState:
    """The intravascular wall as holes are cut into it.  ``owner`` gives, per
    vertex, the index in ``names`` of the branch whose hole (or the band of
    vertices around its rim) it belongs to, -1 for none.  ``frame`` gives the
    (densified) frame each vertex lies on, also for vertices added to a rim."""

    vertices: np.ndarray
    faces: np.ndarray
    owner: np.ndarray
    frame: np.ndarray
    names: list[str] = field(default_factory=list)

    def copy(self) -> _WallState:
        return _WallState(
            self.vertices.copy(),
            self.faces,
            self.owner.copy(),
            self.frame.copy(),
            list(self.names),
        )


@dataclass
class _Joined:
    """A side branch joined to the wall."""

    name: str
    key: tuple[str, int]
    slice_idx: int
    #: The branch, cut at its contour and left open there.
    piece: trimesh.Trimesh
    bridge: trimesh.Trimesh
    n_ring: int
    n_rim: int


def _join_branch(
    ctx: _Context,
    state: _WallState,
    vertices: np.ndarray,
    faces: np.ndarray,
    piece: np.ndarray,
    vessel: str,
    branch_id: int,
    others: list[_Joined],
) -> _Joined | str:
    """Cut one branch of *piece*, cut its hole into *state* and bridge the two,
    or say why not (*state* is then left as it was, or partly changed)."""
    lumen = ctx.lumen
    name = f"{vessel.upper()} branch {branch_id}"
    hint = f"choose another plane with cut_slices[({vessel!r}, {branch_id})]"
    contours = list(getattr(ctx.tree, vessel)[branch_id].contours)
    slice_idx = ctx.cut_slices.get((vessel, branch_id), 0)
    if not 0 <= slice_idx < len(contours):
        return f"cut slice {slice_idx} out of range ({len(contours)} contours)"
    origin = float(lumen.position(np.array([contours[0].centroid]))[0])
    if not 0.0 <= origin <= lumen.arclength[-1]:
        gap, where = (
            (-origin, "before")
            if origin < 0.0
            else (origin - lumen.arclength[-1], "after")
        )
        return (
            f"it leaves its parent {gap:.1f} mm {where} the intravascular segment, "
            "where there is no intravascular wall to join it to"
        )

    cut_piece = _cut_branch(vertices, faces, piece, contours[slice_idx])
    if isinstance(cut_piece, str):
        return f"cutting at contour {slice_idx} failed ({cut_piece}), {hint}"
    piece_mesh, branch_pts = cut_piece
    if (lumen.outside_by(branch_pts) < 0.0).any():
        return f"contour {slice_idx} reaches into the intravascular lumen, {hint}"

    # The cut rim, stepped back along the branch centerline to its first
    # point, leaves its imprint on the wall where each point first meets it.
    path = _branch_path(
        ctx.centerlines[vessel], branch_id, np.asarray(contours[slice_idx].centroid)
    )
    imprint, hit = _imprint(branch_pts, path, ctx.wall)
    footprint = _Footprint(
        lumen, imprint, branch_pts.mean(axis=0), ctx.max_hole_fraction
    )
    if not hit.all():
        print(
            f"{name}: {np.count_nonzero(~hit)} of {len(hit)} ring points "
            "miss the intravascular wall (branch wider than the vessel)"
        )
    if footprint.narrowed < 1.0:
        print(
            f"{name}: narrowed the hole to {footprint.narrowed:.0%} of its width "
            f"({ctx.max_hole_fraction:.0%} of the intravascular circumference)"
        )
    cut = _cut_footprint(state, footprint)
    if isinstance(cut, str):
        return cut
    new_faces, hole, rim = cut
    if len(rim) < len(footprint.outline):
        # A small branch on a coarse wall: one rim vertex per ring point.
        new_faces, rim = _densify_rim(state, new_faces, rim, len(footprint.outline))
    # Each ring point's imprint becomes a rim vertex (where narrowing moved
    # the outline, the outline's point does).
    anchors = footprint.to_wall(footprint.outline)
    if footprint.narrowed == 1.0:
        anchors[hit] = imprint[hit]
    laid = footprint.anchor(state.vertices[rim], state.frame[rim], anchors)
    if isinstance(laid, str):
        return laid
    order, rim_pts, at = laid
    bridge = _anchored_bridge(branch_pts, rim_pts, at)
    for other in others:
        if (
            _meshes_cross(bridge, other.bridge)
            or _meshes_cross(bridge, other.piece)
            or _meshes_cross(piece_mesh, other.bridge)
        ):
            return (
                f"its bridge to the wall would cross {other.name} or its bridge; "
                f"{hint} nearer the parent vessel, or of {other.name}"
            )

    index = len(state.names)
    state.names.append(name)
    state.faces = new_faces
    state.vertices[rim[order]] = rim_pts
    # The hole, its rim and the vertices next to the rim are this branch's.
    band = np.unique(new_faces[np.isin(new_faces, rim).any(axis=1)])
    state.owner[hole] = index
    state.owner[band] = index
    return _Joined(
        name,
        (vessel, branch_id),
        slice_idx,
        piece_mesh,
        bridge,
        len(branch_pts),
        len(rim_pts),
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


def _off_path_pieces(mesh: trimesh.Trimesh, results: dict) -> list[np.ndarray]:
    """The loose pieces most of whose vertices are ``"off_path_points"``."""
    off_path = set(map(tuple, results.get("off_path_points", [])))
    pieces = []
    for piece in _loose_pieces(mesh):
        coords = [tuple(v) for v in mesh.vertices[piece]]
        if 2 * sum(c in off_path for c in coords) > len(coords):
            pieces.append(piece)
    return pieces


def _root_branches(
    coords: np.ndarray, results: dict, tree: PyDiscretizedVesselTree
) -> list[tuple[str, int]]:
    """``(vessel, branch id)`` of each side branch a loose piece starts with,
    most vertices first.

    The piece holds the cut-off branches and the branches growing out of them.
    A branch it starts with is one whose parent is not in the piece (the main
    vessel aside: a strip of it may stay on a piece) and that has at least
    ``_MIN_PIECE_VERTICES`` vertices in it.  Usually that is one branch; two
    that leave the removed segment side by side can share a stub and so one
    piece.
    """
    piece = set(map(tuple, coords))
    found: list[tuple[int, str, int]] = []
    for vessel in _VESSELS:
        nodes = getattr(tree, vessel)
        votes = {
            k: len(piece.intersection(map(tuple, pts)))
            for k, pts in results.get(f"{vessel}_branches", {}).items()
        }
        present = {k for k, v in votes.items() if v and k != 0}
        for k in present:
            node = nodes.get(k)
            if node is None or votes[k] < _MIN_PIECE_VERTICES:
                continue
            if node.parent is not None and node.parent[0] in present:
                continue
            found.append((votes[k], vessel, k))
    return [(vessel, k) for _, vessel, k in sorted(found, reverse=True)]


def _cut_branch(
    vertices: np.ndarray,
    faces: np.ndarray,
    piece: np.ndarray,
    contour: PyContour,
) -> tuple[trimesh.Trimesh, np.ndarray] | str:
    """Cut *piece* at the plane of *contour*, keeping the distal side.

    Returns the cut piece, left open at the cut, and its rim (ordered, on the
    plane), or why the cut failed.
    """
    local = np.full(len(vertices), -1)
    local[piece] = np.arange(len(piece))
    piece_faces = local[faces[np.isin(faces[:, 0], piece)]]
    try:
        cut = _cut_mesh_at_contour(
            [tuple(v) for v in vertices[piece].tolist()],
            piece_faces.tolist(),
            contour,
            keep="distal",
        )
    except ValueError as err:
        return str(err)
    cut_vertices = np.asarray(cut[0], dtype=np.float64)
    cut_faces = np.asarray(cut[1], dtype=np.int64)
    ring, centroid = cut[2], cut[3]
    # Drop the cap: the cut stays open for the bridge.
    open_faces = cut_faces[~(cut_faces == centroid).any(axis=1)]
    # Only what hangs off this ring: a sibling sharing the piece's stub may
    # not be separated from the branch by the cut, and is cut on its own.
    n = len(cut_vertices)
    edges = open_faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    labels = connected_components(graph, directed=False)[1]
    open_faces = open_faces[np.isin(labels[open_faces[:, 0]], labels[ring])]
    return _with_faces(cut_vertices, open_faces), cut_vertices[ring]


def _branch_path(
    centerline: PyCenterline, branch_id: int, start: np.ndarray
) -> np.ndarray:
    """*start*, then the points of the branch's centerline proximal of it, back
    to the branch's first point."""
    cl = np.asarray(centerline.get_branch(branch_id).points_as_tuples(), dtype=float)
    i = int(np.argmin(np.linalg.norm(cl - start, axis=1)))
    # Leave out the nearest point if it lies distal of *start*.
    ahead = cl[i + 1] - cl[i] if i + 1 < len(cl) else cl[i] - cl[i - 1]
    if np.dot(cl[i] - start, ahead) > 0.0:
        i -= 1
    return np.vstack([start[None], cl[i::-1]]) if i >= 0 else start[None]


def _imprint(
    ring: np.ndarray, path: np.ndarray, wall: trimesh.Trimesh
) -> tuple[np.ndarray, np.ndarray]:
    """Step *ring* rigidly along the polyline *path* (which starts at the ring)
    and find where each of its points first meets the *wall* surface.

    Returns each point's imprint and whether it met the wall at all.  A point
    that never does stays where the path ends.
    """
    offsets = path - path[0]
    out = ring + offsets[-1]
    hit = np.zeros(len(ring), dtype=bool)
    triangles = wall.vertices[wall.faces]
    for a, b in itertools.pairwise(offsets):
        todo = np.flatnonzero(~hit)
        length = float(np.linalg.norm(b - a))
        if len(todo) == 0:
            break
        if length < 1e-12:
            continue
        direction = (b - a) / length
        origins = ring[todo] + a
        # Only the triangles around this step.
        lo = np.minimum(origins, origins + b - a).min(axis=0)
        hi = np.maximum(origins, origins + b - a).max(axis=0)
        near = (triangles.max(axis=1) >= lo).all(axis=1) & (
            triangles.min(axis=1) <= hi
        ).all(axis=1)
        if not near.any():
            continue
        dist = _ray_triangle_t(
            origins, np.broadcast_to(direction, origins.shape), triangles[near]
        ).min(axis=1)
        met = dist <= length
        out[todo[met]] = origins[met] + dist[met, None] * direction
        hit[todo[met]] = True
    return out, hit


def _ray_triangle_t(
    origins: np.ndarray, directions: np.ndarray, triangles: np.ndarray
) -> np.ndarray:
    """Distance along each unit direction from each origin to each of
    *triangles* (F, 3, 3), ``inf`` where the ray misses it (Moller-Trumbore).
    Returns an array (rays, triangles)."""
    v0 = triangles[:, 0]
    e1 = triangles[:, 1] - v0
    e2 = triangles[:, 2] - v0
    d = directions[:, None, :]
    p = np.cross(d, e2[None])
    det = (e1[None] * p).sum(axis=2)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    rel = origins[:, None, :] - v0[None]
    u = (rel * p).sum(axis=2) * inv
    q = np.cross(rel, e1[None])
    v = (q * d).sum(axis=2) * inv
    t = (q * e2[None]).sum(axis=2) * inv
    eps = 1e-9
    valid = ok & (u >= -eps) & (v >= -eps) & (u + v <= 1.0 + eps) & (t >= -eps)
    return np.where(valid, t, np.inf)


def _meshes_cross(a: trimesh.Trimesh, b: trimesh.Trimesh) -> bool:
    """Whether the surfaces *a* and *b* cut through each other: an edge of one
    passes through a triangle of the other (touching at an edge's end does not
    count)."""
    lo = np.maximum(a.bounds[0], b.bounds[0])
    hi = np.minimum(a.bounds[1], b.bounds[1])
    if (lo > hi).any():
        return False
    return _edges_pierce(a, b, lo, hi) or _edges_pierce(b, a, lo, hi)


def _edges_pierce(
    a: trimesh.Trimesh, b: trimesh.Trimesh, lo: np.ndarray, hi: np.ndarray
) -> bool:
    """Whether an edge of *a* passes through a triangle of *b*, looking only
    within the box *lo*-*hi* both lie in."""

    def in_box(pts: np.ndarray) -> np.ndarray:
        return (pts.max(axis=1) >= lo).all(axis=1) & (pts.min(axis=1) <= hi).all(axis=1)

    triangles = b.triangles[in_box(b.triangles)]
    edges = a.vertices[a.edges_unique]
    edges = edges[in_box(edges)]
    if len(triangles) == 0 or len(edges) == 0:
        return False
    step = edges[:, 1] - edges[:, 0]
    length = np.linalg.norm(step, axis=1)
    keep = length > 1e-12
    edges, step, length = edges[keep], step[keep], length[keep]
    for chunk in range(0, len(edges), 256):
        sl = slice(chunk, chunk + 256)
        t = _ray_triangle_t(edges[sl, 0], step[sl] / length[sl, None], triangles)
        span = length[sl, None]
        if ((t > 1e-6 * span) & (t < (1.0 - 1e-6) * span)).any():
            return True
    return False


def _in_polygon(xy: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd test of points *xy* (N, 2) against the closed polygon *poly* (M, 2)."""
    x, y = xy[:, :1], xy[:, 1:]
    x1, y1 = poly[None, :, 0], poly[None, :, 1]
    x2, y2 = np.roll(poly[:, 0], -1)[None], np.roll(poly[:, 1], -1)[None]
    crosses = (y1 > y) != (y2 > y)
    dy = np.where(y2 != y1, y2 - y1, 1e-12)
    x_at = x1 + (y - y1) * (x2 - x1) / dy
    return (crosses & (x < x_at)).sum(axis=1) % 2 == 1


class _Footprint:
    """A hole outline on the intravascular wall, around the side facing a branch.

    A point on the wall is given as ``(s, a)``: arc length along the lumen
    axis, and arc length around it from the side facing the branch, measured
    on a circle of the lumen's radius at the branch.  The outline is built
    from *points* on (or, where a wide branch misses the wall, outside) the
    wall, each placed on the wall in the direction it lies from the axis.  Its
    arc is narrowed towards the branch side to at most *max_fraction* of the
    circumference.
    """

    def __init__(
        self,
        lumen: _Lumen,
        points: np.ndarray,
        towards: np.ndarray,
        max_fraction: float,
    ) -> None:
        self.lumen = lumen
        frame = int(lumen.nearest_frame(points.mean(axis=0)[None])[0])
        self.d, self.e = lumen.basis(towards - lumen.centroids[frame])
        self.radius = float(lumen.radii[frame])
        sa = self.wall_coords(points, lumen.nearest_frame(points))
        limit = np.pi * max_fraction * self.radius
        widest = np.abs(sa[:, 1]).max()
        self.narrowed = min(1.0, limit / widest) if widest > 0 else 1.0
        self.outline = np.column_stack([sa[:, 0], sa[:, 1] * self.narrowed])
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

    def anchor(
        self, rim: np.ndarray, frames: np.ndarray, anchors: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | str:
        """Lay the hole's rim, *rim* (ordered wall points), onto the outline.

        Outline point ``j`` gets one rim vertex of its own, placed exactly at
        ``anchors[j]``; the rim vertices between two of those are spread
        evenly along the outline between them.  The rim keeps its order: it
        is turned to run the same way as the outline and started at the
        vertex nearest outline point 0.

        Returns the order of *rim* that does that, the new rim positions in
        that order and the rim index of each outline point, or why the rim
        cannot hold the outline.
        """
        n, m = len(self.outline), len(rim)
        if m < n:
            return (
                f"the hole's rim has {m} points, fewer than the {n} of the branch "
                "ring; lower max_ring_spacing_mm"
            )
        sa = self.wall_coords(rim, frames)
        order = np.arange(m)
        if np.sign(_area_2d(sa)) != np.sign(_area_2d(self.outline)):
            order = order[::-1]
        closed = np.vstack([self.outline, self.outline[:1]])
        seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        along = np.concatenate([[0.0], np.cumsum(seg)])
        total = along[-1]
        param = _polyline_param(sa[order], closed, along)
        # Start at the vertex nearest outline point 0.
        gap = np.minimum(param, total - param)
        order = np.roll(order, -int(np.argmin(gap)))
        param = np.roll(param, -int(np.argmin(gap))) % total
        param[0] = 0.0
        at = np.zeros(n, dtype=int)
        for j in range(1, n):
            nearest = int(np.argmin(np.abs(param - along[j])))
            at[j] = min(max(nearest, at[j - 1] + 1), m - (n - j))
        ends = np.append(at[1:], m)
        target = np.empty((m, 2))
        for j in range(n):
            count = ends[j] - at[j]
            f = (np.arange(count) / count)[:, None]
            target[at[j] : ends[j]] = (1.0 - f) * closed[j] + f * closed[j + 1]
        laid = self.to_wall(target)
        laid[at] = anchors
        return order, laid, at


def _polyline_param(
    points: np.ndarray, closed: np.ndarray, along: np.ndarray
) -> np.ndarray:
    """Arc length along the polyline *closed* (cumulative lengths *along*) of
    the nearest point to each of *points* (2D)."""
    a, b = closed[:-1], closed[1:]
    d = b - a
    dd = np.maximum((d * d).sum(axis=1), 1e-24)
    rel = points[:, None, :] - a[None]
    t = np.clip((rel * d[None]).sum(axis=2) / dd[None], 0.0, 1.0)
    dist = np.linalg.norm(rel - t[..., None] * d[None], axis=2)
    k = np.argmin(dist, axis=1)
    rows = np.arange(len(points))
    return along[k] + t[rows, k] * np.sqrt(dd[k])


def _densify_rim(
    state: _WallState, faces: np.ndarray, rim: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Insert vertices along the hole's *rim* until it has *n* (see
    :func:`~.boundary._densify_boundary`), added to *state*.

    Returns the faces and the rim, in order.  An added vertex lies on the
    frame of the nearest rim vertex.
    """
    mesh = trimesh.Trimesh(state.vertices, faces, process=False)
    dense, ring = _densify_boundary(mesh, [tuple(v) for v in state.vertices[rim]], n)
    old = len(state.vertices)
    added = np.asarray(dense.vertices[old:])
    nearest = cKDTree(state.vertices[rim]).query(added)[1]
    state.vertices = np.asarray(dense.vertices)
    state.owner = np.concatenate([state.owner, np.full(len(added), -1)])
    state.frame = np.concatenate([state.frame, state.frame[rim][nearest]])
    index = {tuple(v): i for i, v in enumerate(state.vertices)}
    return np.asarray(dense.faces), np.array([index[p] for p in ring])


def _cut_footprint(
    state: _WallState, footprint: _Footprint
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | str:
    """Remove the intravascular wall inside *footprint*.

    Returns the remaining faces, the removed vertices and the hole's rim, or why
    there is no hole.  Neither the hole nor its rim may reach a vertex an
    earlier hole owns (its hole, rim or the band around the rim), so two holes
    keep at least one vertex between their rims.
    """
    vertices, faces = state.vertices, state.faces
    frames = state.frame
    hole = np.flatnonzero(footprint.contains(footprint.wall_coords(vertices, frames)))
    if len(hole) == 0:
        return "its footprint covers no intravascular vertex"
    rows = frames[hole]
    last = len(footprint.lumen.centroids) - 1
    if rows.min() <= 1 or rows.max() >= last - 1:
        return "its footprint reaches the end of the intravascular segment"

    kept = faces[~np.isin(faces, hole).any(axis=1)]
    removed_faces = faces[np.isin(faces, hole).any(axis=1)]
    seeds = set(np.setdiff1d(removed_faces.ravel(), hole).tolist())
    extra, rings = clean_open_boundary(kept, vertices, seeds, target_n=1)
    if extra:
        if np.isin(frames[np.asarray(list(extra))], [0, last]).any():
            return (
                "cleaning its rim would cut into the end of the intravascular segment"
            )
        kept = kept[~np.isin(kept, list(extra)).any(axis=1)]
        hole = np.union1d(hole, list(extra))
    if not rings:
        return "the hole has no clean rim"
    rim = np.asarray(rings[0])
    owners = state.owner[np.concatenate([hole, rim])]
    if (owners >= 0).any():
        other = state.names[int(owners[owners >= 0][0])]
        return f"its hole would overlap or touch the hole of {other}"
    return kept, hole, rim


def _area_2d(xy: np.ndarray) -> float:
    """Signed area (twice) of the closed polygon *xy*, positive anticlockwise."""
    x, y = xy[:, 0], xy[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _anchored_bridge(
    ring: np.ndarray, rim: np.ndarray, at: np.ndarray
) -> trimesh.Trimesh:
    """Triangle strip between the branch's cut *ring* and the hole's *rim*,
    with ``ring[j]`` joined to ``rim[at[j]]``.

    Between two such pairs, the first half of the rim vertices fan to
    ``ring[j]``, the second half to ``ring[j + 1]``, so nothing twists.  The
    winding matches :func:`~multimodars.ccta.stitching.core._stitch_rings`.
    """
    n, m = len(ring), len(rim)
    faces: list[tuple[int, int, int]] = []
    for j in range(n):
        b0, b1 = j, (j + 1) % n
        end = at[j + 1] if j + 1 < n else m
        run = [n + (k % m) for k in range(at[j], end + 1)]
        mid = (len(run) - 1) // 2
        faces += [(b0, run[t + 1], run[t]) for t in range(mid)]
        faces.append((b0, b1, run[mid]))
        faces += [(b1, run[t + 1], run[t]) for t in range(mid, len(run) - 1)]
    return trimesh.Trimesh(
        vertices=np.vstack([ring, rim]),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )


def _with_faces(vertices: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    out = trimesh.Trimesh(vertices=vertices.copy(), faces=faces, process=False)
    out.remove_unreferenced_vertices()
    return out
