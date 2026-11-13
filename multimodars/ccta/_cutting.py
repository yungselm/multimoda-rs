"""Cutting the labelled CCTA mesh at a plane, with its vertex labels kept in step.

Shared by :func:`~multimodars.cut_at_slice` (capped outlets) and the stitch
(open, planar rims to join the intravascular geometry to).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import trimesh

from ..multimodars import cut_mesh_at_contour as _cut_mesh_at_contour
from ._results import is_boundary_key, map_point_lists, point_lists, set_point_list

if TYPE_CHECKING:
    from ..multimodars import PyContour


@dataclass
class LabelledCut:
    """A labelled mesh cut by :func:`cut_labelled_mesh`."""

    #: The results with the cut ``"mesh"`` and every vertex list updated to it.
    results: dict
    #: The ring vertices on the plane, in order.
    ring: list[tuple[float, float, float]]
    #: The cap closing the ring (ring in order, then the centre), or ``None``
    #: when the cut was left open.
    cap: trimesh.Trimesh | None
    #: ``"proximal"`` or ``"distal"``, the side that stayed.
    kept_side: str
    #: Vertices of the input mesh that stayed.
    n_kept: int


def cut_labelled_mesh(
    results: dict,
    contour: PyContour,
    keep: str,
    search_radius_factor: float = 2.0,
    snap_fraction: float = 0.2,
    cap: bool = True,
) -> LabelledCut:
    """Cut ``results["mesh"]`` at the plane of *contour* and update the labels.

    See :func:`~multimodars.cut_at_slice` for *keep*, *search_radius_factor*
    and *snap_fraction*.  Removed vertices are dropped from every vertex list
    and snapped ones moved; the new ring (and cap) vertices join the labels
    most of the ring's kept neighbours carry.  With *cap* ``False`` the ring is
    left open.

    Raises
    ------
    ValueError
        If no closed ring of the mesh encloses the contour's centroid, the ring
        does not separate the mesh, or the cap would fold over.
    """
    mesh: trimesh.Trimesh = results["mesh"]
    vertices, faces, ring, centroid, cap_faces, old_to_new, kept_side = (
        _cut_mesh_at_contour(
            [tuple(v) for v in mesh.vertices.tolist()],
            mesh.faces.tolist(),
            contour,
            keep=keep,
            search_radius_factor=search_radius_factor,
            snap_fraction=snap_fraction,
        )
    )
    if not cap:
        faces = [f for f in faces if centroid not in f]

    new_mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    new_coords = [tuple(v) for v in new_mesh.vertices.tolist()]
    moved = {
        tuple(old): new_coords[new]
        for old, new in zip(mesh.vertices.tolist(), old_to_new, strict=True)
        if new is not None
    }

    def remap(pts: list) -> list:
        return [moved[tuple(p)] for p in pts if tuple(p) in moved]

    updated = map_point_lists(results, remap)

    # Label the new vertices like most of the kept vertices along the ring.
    kept = {new for new in old_to_new if new is not None}
    added = [i for i in range(len(new_coords)) if i not in kept]
    if not cap:
        added.remove(centroid)
    ring_set = set(ring)
    neighbours = {
        new_coords[v]
        for face in faces
        if ring_set.intersection(face)
        for v in face
        if v in kept and v not in ring_set
    }
    added_pts = [new_coords[i] for i in added]
    for where, pts in point_lists(updated):
        if is_boundary_key(where[0]):
            continue
        if 2 * len(neighbours.intersection(pts)) > len(neighbours):
            set_point_list(updated, where, pts + added_pts)

    cap_mesh = None
    if cap:
        order = [*ring, centroid]
        local = {v: i for i, v in enumerate(order)}
        cap_mesh = trimesh.Trimesh(
            vertices=new_mesh.vertices[order],
            faces=[[local[v] for v in f] for f in cap_faces],
            process=False,
        )
    else:
        new_mesh.remove_unreferenced_vertices()
    updated["mesh"] = new_mesh
    return LabelledCut(
        results=updated,
        ring=[new_coords[v] for v in ring],
        cap=cap_mesh,
        kept_side=kept_side,
        n_kept=len(kept),
    )
