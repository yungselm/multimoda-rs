"""Add/remove mesh vertex regions by label.

Cross-cutting mesh-region utilities used throughout the pipeline: trimming a
labelled region off the CCTA mesh before stitching (:func:`stitch` in
``__init__.py``), extracting a single labelled sub-region for STL export, and
building an isolated wall mesh, and removing a labelled side branch
(:func:`remove_branch`). None of this is stitching-specific, hence its
own module rather than living inside :mod:`multimodars.ccta.stitching`.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ..multimodars import build_adjacency_map
from ..multimodars import cut_mesh_at_contour as _cut_mesh_at_contour
from ._results import (
    branches_key,
    map_point_lists,
    point_lists,
    set_point_list,
    vessel_branches,
    vessel_of_centerline,
)
from .postprocessing import fill_ring_on_vessel_wall
from .stitching.boundary import BOUNDARY_RING_PREFIX, clean_open_boundary

if TYPE_CHECKING:
    from ..multimodars import PyCenterline, PyDiscretizedVesselTree


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


def _is_boundary_key(key: str) -> bool:
    return key.startswith("boundary_points")


def _warn_off_path_attached(
    results: dict,
    mesh: trimesh.Trimesh,
    coord_to_idx: dict,
    remove_indices: set[int],
) -> None:
    """Warn when vertices off the pullback path border the region being removed.

    Branches leaving the removed segment are not removed with it (see
    :func:`~multimodars.label_anomalous_region`): they stay in the mesh, cut off
    from the vessel, and their rim joins the opening.
    """
    off_path = {
        coord_to_idx[tuple(p)]
        for p in results.get("off_path_points", [])
        if tuple(p) in coord_to_idx
    } - remove_indices
    if not off_path:
        return
    edges = mesh.edges_unique
    removed = np.isin(edges, list(remove_indices))
    off = np.isin(edges, list(off_path))
    touching = edges[(removed[:, 0] & off[:, 1]) | (removed[:, 1] & off[:, 0])]
    n = len(off_path.intersection(touching.ravel().tolist()))
    if n:
        warnings.warn(
            f"{n} off-path vertices border the removed region: branches leaving it "
            "stay in the mesh, cut off from the vessel. Pass the discretized tree "
            "to the stitch to reconnect them",
            UserWarning,
            stacklevel=3,
        )


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
        Dictionary containing at minimum the key ``"mesh"``.  Every other
        vertex list (``"aorta_points"``, ``"distal_points"``, each branch in
        ``"rca_branches"``, ...) is updated to the trimmed mesh.
    region_keys : str or list of str
        Key(s) in *results* whose point lists should be removed from the mesh.
        Defaults to ``"anomalous_points"`` for backwards compatibility.
    target_boundaries : int
        Number of open-boundary rings the removal is expected to create.
        Removing a single blob leaves one ring (default); removing a region
        that splits the surface (e.g. aorta + intramural wall) can leave two.
        The rim is cleaned and reduced to this many rings before it is stored.
        Rims of branches off the pullback path (``"off_path_points"``) that
        leave the removed region are not counted: they stay as they are, with
        a warning, for the stitch to reconnect.

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

    _warn_off_path_attached(results, mesh, coord_to_idx, remove_indices)

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
    # Rims of branches off the pullback path that leave the removed region are
    # not part of the opening: leaving them unseeded keeps them out of the
    # reduction to *target_boundaries* rings.  The stitch reconnects them.
    off_path = {
        coord_to_idx[tuple(p)]
        for p in results.get("off_path_points", [])
        if tuple(p) in coord_to_idx
    }
    boundary_indices = (boundary_indices - off_path) or boundary_indices

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

    return map_point_lists(
        updated,
        lambda pts: [p for p in pts if tuple(p) in new_coord_set],
        skip=lambda key: key in region_keys or _is_boundary_key(key),
    )


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
        Dictionary containing at minimum the key ``"mesh"``.  Every other
        vertex list (``"aorta_points"``, ``"distal_points"``, each branch in
        ``"rca_branches"``, ...) is updated to the trimmed mesh.
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

    return map_point_lists(
        updated,
        lambda pts: [p for p in pts if tuple(p) in new_coord_set],
        skip=_is_boundary_key,
    )


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


def _detached_fragments(mesh: trimesh.Trimesh, remove_idx: set[int]) -> set[int]:
    """Kept vertices that only hung on to the mesh through *remove_idx*.

    Vertices of the branch that its labels missed (e.g. a wide distal end)
    would be left floating once the branch is cut off.  Of the connected pieces
    touching the removed region, the largest is the parent vessel, and the rest
    are returned.
    """
    n = len(mesh.vertices)
    keep = np.ones(n, dtype=bool)
    keep[list(remove_idx)] = False
    faces = mesh.faces
    kept_faces = faces[keep[faces].all(axis=1)]
    edges = kept_faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)

    cut_faces = faces[(~keep[faces]).any(axis=1)]
    touching = set(labels[cut_faces[keep[cut_faces]]].tolist())
    if len(touching) <= 1:
        return set()
    sizes = np.bincount(labels)
    parent = max(touching, key=lambda c: sizes[c])
    loose = keep & np.isin(labels, list(touching - {parent}))
    return set(np.flatnonzero(loose).tolist())


def remove_branch(
    results: dict,
    centerline: PyCenterline,
    branch_id: int,
    fill_hole: bool = True,
) -> tuple[dict, PyCenterline]:
    """Cut a side branch off the mesh and the centerline, and close its ostium.

    The branch's vertices are its label from :func:`~multimodars.label_branches`
    (``results["rca_branches"][branch_id]`` or the LCA's, whichever vessel
    *centerline* belongs to).  Every branch growing out of *branch_id* is
    removed with it.

    Branch ids are not renumbered: the removed branches are dropped from the
    vessel's branch labels and from the returned centerline, the other branches
    keep their id.

    Parameters
    ----------
    results : dict
        Labelled results with ``"mesh"`` and the per-branch labels.
    centerline : PyCenterline
        Whole centerline of the vessel the branch belongs to.
    branch_id : int
        Side branch to remove (not 0, the main vessel).
    fill_hole : bool, optional
        Close the ostium with a patch following the parent vessel's wall
        (default).  With ``False`` the hole stays open and its rim is stored
        as ``"boundary_points_1"`` / ``"boundary_points"``.

    Returns
    -------
    (dict, PyCenterline)
        Updated *results* (mesh without the branch, every vertex list updated
        to it, patch vertices added to the labels most of the ostium rim
        carries) and the centerline without the removed branches.

    Raises
    ------
    ValueError
        If *branch_id* is 0 or not in *centerline*, or the branches were not
        labelled.
    """
    if branch_id == 0:
        raise ValueError("branch 0 is the main vessel and cannot be removed")
    vessel = vessel_of_centerline(results, centerline)
    branches = vessel_branches(results, vessel)
    new_centerline = centerline.remove_branches([branch_id])
    removed = sorted(set(centerline.branch_ids()) - set(new_centerline.branch_ids()))

    mesh: trimesh.Trimesh = results["mesh"]
    coord_to_idx = {tuple(v): i for i, v in enumerate(mesh.vertices)}
    remove_idx = {
        coord_to_idx[tuple(p)]
        for k in removed
        for p in branches.get(k, [])
        if tuple(p) in coord_to_idx
    }
    remove_idx |= _detached_fragments(mesh, remove_idx)

    staging_key = "_remove_branch_points"
    staged = dict(results)
    staged[staging_key] = [tuple(mesh.vertices[i]) for i in remove_idx]
    updated = remove_labeled_points_from_mesh(staged, staging_key, target_boundaries=1)
    del updated[staging_key]
    key = branches_key(vessel)
    updated[key] = {k: pts for k, pts in updated[key].items() if k not in removed}

    ring_coords = updated.get(f"{BOUNDARY_RING_PREFIX}1", [])
    if fill_hole and ring_coords:
        new_mesh = updated["mesh"]
        idx = {tuple(v): i for i, v in enumerate(new_mesh.vertices)}
        parent = centerline.branch_parents()[branch_id]
        assert parent is not None  # branch_id is a side branch of centerline
        axis = np.asarray(
            centerline.get_branch(parent).points_as_tuples(), dtype=np.float64
        )
        patched, added = fill_ring_on_vessel_wall(
            new_mesh, [idx[tuple(c)] for c in ring_coords], axis
        )
        updated["mesh"] = patched

        # The patch is part of the parent wall: give it the labels most of the
        # rim has (e.g. rca_points, the parent's branch, distal_points).
        ring_set = {tuple(c) for c in ring_coords}
        added_pts = [tuple(float(x) for x in p) for p in added]
        for where, pts in point_lists(updated):
            if _is_boundary_key(where[0]):
                continue
            if 2 * len(ring_set.intersection(map(tuple, pts))) > len(ring_set):
                set_point_list(updated, where, pts + added_pts)
        # The rim is closed now, and keeping it would describe a hole that is gone.
        _store_boundary_rings(updated, patched.vertices, [])
        print(f"Closed the ostium with {len(added_pts)} patch vertices")

    print(f"Removed {vessel.upper()} branch(es) {removed} ({len(remove_idx)} vertices)")
    return updated, new_centerline


def cut_at_slice(
    results: dict,
    tree: PyDiscretizedVesselTree,
    vessel: str,
    slice_idx: int,
    branch_id: int = 0,
    keep: str = "larger",
    search_radius_factor: float = 2.0,
    snap_fraction: float = 0.2,
) -> tuple[dict, trimesh.Trimesh]:
    """Cut the mesh flat at a slice of the discretized tree and cap the cut.

    Only the slice's plane is taken from *tree* (through the contour's
    centroid, normal to it), so ``results["mesh"]`` may have been scaled,
    smoothed or remeshed since :func:`~multimodars.discretize_vessel_tree`.
    Only the vessel ring around the slice centre is cut, every other part of
    the mesh crossing the plane is left alone. Ring vertices lie exactly on
    the plane, and the ring is closed with a flat fan around its centroid, so
    the end is a clean planar inlet/outlet. Branches leaving the dropped part
    are dropped with it.

    Parameters
    ----------
    results : dict
        Results with ``"mesh"`` and its vertex labels.
    tree : PyDiscretizedVesselTree
        Discretized tree giving the plane.
    vessel : {"aorta", "rca", "lca"}
        Vessel to cut: ``tree.discretized_aorta`` or the branch
        ``tree.<vessel>[branch_id]``.
    slice_idx : int
        Index into that vessel's ``contours`` list (negative counts from the
        distal end). Not the contour ``id``: end slices may have been trimmed.
    branch_id : int, optional
        Branch of *vessel* (ignored for the aorta). Default 0, the main vessel.
    keep : {"larger", "proximal", "distal"}, optional
        Side of the plane that stays: the one with the larger surface area
        (default, independent of the centerline direction), or the side
        against / along the centerline direction.
    search_radius_factor : float, optional
        Only the mesh within this multiple of the slice radius from its
        centre is cut. Raise it if the vessel has widened since discretizing.
    snap_fraction : float, optional
        Vertices closer to the plane than this fraction of a crossed edge are
        moved onto it instead of splitting the edge, avoiding sliver
        triangles. ``0`` splits every crossed edge.

    Returns
    -------
    (dict, trimesh.Trimesh)
        *results* with the cut and capped ``"mesh"`` and every vertex list
        updated to it (removed vertices dropped, snapped ones moved, the new
        ring and cap vertices added to the labels the ring's neighbours carry),
        and a copy of the cap as its own mesh for export: its vertices are the
        ring in order, then the centre, with the faces' normals pointing out
        of the kept mesh.

    Raises
    ------
    ValueError
        If no closed ring of the mesh encloses the slice centre (the plane
        cuts a junction, choose a slice further from it), the ring does not
        separate the mesh, or the ring is not star-shaped around its centroid.
    """
    if vessel == "aorta":
        contours = tree.discretized_aorta
    elif vessel in ("rca", "lca"):
        branches = getattr(tree, vessel)
        if branch_id not in branches:
            raise ValueError(
                f"{vessel.upper()} has no branch {branch_id} (branches {sorted(branches)})"
            )
        contours = branches[branch_id].contours
    else:
        raise ValueError(f"vessel must be 'aorta', 'rca' or 'lca', got {vessel!r}")
    if not -len(contours) <= slice_idx < len(contours):
        raise IndexError(f"slice_idx {slice_idx} out of range ({len(contours)} slices)")

    mesh: trimesh.Trimesh = results["mesh"]
    vertices, faces, ring, centroid, cap_faces, old_to_new = _cut_mesh_at_contour(
        [tuple(v) for v in mesh.vertices.tolist()],
        mesh.faces.tolist(),
        contours[slice_idx],
        keep=keep,
        search_radius_factor=search_radius_factor,
        snap_fraction=snap_fraction,
    )
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
    updated["mesh"] = new_mesh

    # Label the new vertices like most of the kept vertices along the ring.
    kept = {new for new in old_to_new if new is not None}
    added = [i for i in range(len(new_coords)) if i not in kept]
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
        if _is_boundary_key(where[0]):
            continue
        if 2 * len(neighbours.intersection(pts)) > len(neighbours):
            set_point_list(updated, where, pts + added_pts)

    # The cap on its own: ring in order, then the centre.
    order = [*ring, centroid]
    local = {v: i for i, v in enumerate(order)}
    cap = trimesh.Trimesh(
        vertices=new_mesh.vertices[order],
        faces=[[local[v] for v in f] for f in cap_faces],
        process=False,
    )
    print(
        f"Cut {vessel.upper() if vessel != 'aorta' else 'aorta'} at slice {slice_idx}: "
        f"kept {len(kept)} of {len(mesh.vertices)} vertices, cap with {len(ring)} ring points"
    )
    return updated, cap
