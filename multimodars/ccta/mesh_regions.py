"""Add/remove mesh vertex regions by label.

Cross-cutting mesh-region utilities used throughout the pipeline: trimming a
labelled region off the CCTA mesh before stitching (:func:`stitch` in
``__init__.py``), extracting a single labelled sub-region for STL export, and
building an isolated wall mesh. None of this is stitching-specific, hence its
own module rather than living inside :mod:`multimodars.ccta.stitching`.
"""

from __future__ import annotations

import numpy as np
import trimesh

from ..multimodars import build_adjacency_map
from .stitching.boundary import BOUNDARY_RING_PREFIX, clean_open_boundary


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
    for key in [k for k in updated if k.startswith(BOUNDARY_RING_PREFIX)]:
        del updated[key]
    per_ring = [[tuple(vertices[i]) for i in ring] for ring in rings]
    for n, pts in enumerate(per_ring, start=1):
        updated[f"{BOUNDARY_RING_PREFIX}{n}"] = pts
    updated["boundary_points"] = [pt for pts in per_ring for pt in pts]


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


def extract_region_with_border_faces(
    mesh: trimesh.Trimesh,
    region_points: list,
) -> trimesh.Trimesh:
    """Return a sub-mesh containing every face that touches at least one vertex
    in *region_points*.

    Unlike :func:`keep_labeled_points_from_mesh`, which only keeps faces whose
    *all* vertices belong to the region, this function uses an
    **at-least-one-vertex** criterion.  The result therefore includes the thin
    ring of adjacent-region vertices that share a face with the target region,
    giving seamless overlapping boundaries when meshes of different labels are
    exported side-by-side.
    """
    coord_to_idx = {tuple(v): i for i, v in enumerate(mesh.vertices)}
    keep_indices = np.array(
        [coord_to_idx[tuple(p)] for p in region_points if tuple(p) in coord_to_idx],
        dtype=np.int64,
    )
    if keep_indices.size == 0:
        return trimesh.Trimesh()

    face_mask = np.isin(mesh.faces, keep_indices).any(axis=1)
    selected_faces = mesh.faces[face_mask]

    used = np.unique(selected_faces)
    remap = np.full(len(mesh.vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)

    return trimesh.Trimesh(
        vertices=mesh.vertices[used],
        faces=remap[selected_faces],
        process=False,
    )
