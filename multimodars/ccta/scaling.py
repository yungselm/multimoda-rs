from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from ..multimodars import (
    PyCenterline,
    PyDiscretizedVesselTree,
    PyFrame,
    adjust_radius_centerline_morphing,
    find_aortic_scaling,
    find_proximal_distal_scaling,
)
from ..multimodars import (
    find_aortic_wall_scaling as _find_aortic_wall_scaling,
)
from ._results import map_point_lists
from .scaling_helpers import _extract_wall_from_frames


def scale_region_centerline_morphing(
    mesh: trimesh.Trimesh,
    region_points: list,
    centerline: PyCenterline,
    radius_adjustment_mm: float,
    branch_points: dict[int, list] | None = None,
) -> trimesh.Trimesh:
    """Scale a mesh region radially around its centerline.

    Each vertex in *region_points* is displaced along the direction from the
    nearest centerline point outward (positive *radius_adjustment_mm*) or
    inward (negative).

    Without *branch_points* the nearest point may lie on any branch, so a
    vertex near a junction can be pushed away from the wrong branch.  With
    *branch_points* (``results["rca_branches"]`` / ``results["lca_branches"]``
    from :func:`label_branches`), each labelled vertex is morphed around its
    own branch (``centerline.get_branch(branch_id)``) instead.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The original mesh.  A copy is returned; the input is not modified.
    region_points : list of tuple
        ``(x, y, z)`` coordinates of the vertices to be scaled.  Only vertices
        present in this list are moved.
    centerline : PyCenterline
        Centerline of the vessel region used as the morphing axis.
    radius_adjustment_mm : float
        Radial change in millimetres (diameter changes by twice this).
        Positive values expand the lumen; negative values contract it.
    branch_points : dict of int to list of tuple, optional
        Branch id -> vertices, as labelled by :func:`label_branches`.  Vertices
        without a label, or labelled with a branch *centerline* does not have,
        are morphed around the whole *centerline*.

    Returns
    -------
    trimesh.Trimesh
        A new mesh with the selected region scaled.

    Warns
    -----
    If no vertices matching *region_points* are found in the mesh, a warning
    is printed and the unmodified copy is returned.

    Notes
    -----
    A negative *radius_adjustment_mm* never brings a vertex closer to its
    nearest centerline point than 10 % of its distance to it: going further
    would carry it through the centerline to the opposite side, which scaling
    back cannot undo.  The number of vertices stopped there is printed.  They
    move less than ``abs(radius_adjustment_mm)``, so scaling back by
    ``-radius_adjustment_mm`` does not return them exactly; every other vertex
    returns exactly (as long as its nearest centerline point stays the same).
    """
    scaled_mesh = mesh.copy()

    region_vertex_indices_list: list[int] = []
    region_set = set(region_points)

    for idx, vertex in enumerate(scaled_mesh.vertices):
        if tuple(vertex) in region_set:
            region_vertex_indices_list.append(idx)

    region_vertex_indices = np.array(region_vertex_indices_list)

    if len(region_vertex_indices) == 0:
        print("Warning: No vertices found for scaling region")
        return scaled_mesh

    print(f"\nScaling {len(region_vertex_indices)} vertices around {centerline}")
    print(f"Radius adjustment: {np.round(radius_adjustment_mm, 2)} mm")

    vertices = np.array(scaled_mesh.vertices, dtype=np.float64)
    before = vertices[region_vertex_indices].copy()
    for indices, axis in _morph_axes(
        vertices, region_vertex_indices, centerline, branch_points
    ):
        adjusted_points = adjust_radius_centerline_morphing(
            centerline=axis,
            points=[tuple(v) for v in vertices[indices]],
            radius_adjustment_mm=radius_adjustment_mm,
        )
        vertices[indices] = np.array(adjusted_points, dtype=np.float64)

    if radius_adjustment_mm < 0.0:
        moved = np.linalg.norm(vertices[region_vertex_indices] - before, axis=1)
        # Vertices on the centerline do not move at all.
        clamped = np.count_nonzero(
            (moved > 0.0) & (moved < -radius_adjustment_mm - 1e-9)
        )
        if clamped:
            print(
                f"Warning: {clamped} of {len(moved)} vertices would have come "
                "closer to their centerline than 10% of their distance to it (or "
                "through it); they were stopped there, and scaling back by "
                f"{-radius_adjustment_mm} mm will not return them exactly"
            )

    # Assigning the array clears the mesh cache
    scaled_mesh.vertices = vertices

    return scaled_mesh


def _morph_axes(
    vertices: np.ndarray,
    region_indices: np.ndarray,
    centerline: PyCenterline,
    branch_points: dict[int, list] | None,
) -> list[tuple[np.ndarray, PyCenterline]]:
    """Region vertex indices grouped by the centerline they are morphed around.

    Every labelled branch gets its own branch of *centerline*.  Vertices without
    a usable label keep the whole *centerline*.
    """
    if not branch_points:
        return [(region_indices, centerline)]
    branch_of: dict[tuple, int] = {}
    present = set(centerline.branch_ids())
    for branch_id, points in branch_points.items():
        if branch_id in present:
            branch_of.update((tuple(p), branch_id) for p in points)
    labels = np.array([branch_of.get(tuple(vertices[i]), -1) for i in region_indices])
    groups = [
        (region_indices[labels == b], centerline.get_branch(int(b)))
        for b in np.unique(labels)
        if b >= 0
    ]
    rest = region_indices[labels < 0]
    if len(rest):
        groups.append((rest, centerline))
    return groups


def find_distal_and_proximal_scaling(
    frames,
    centerline,
    results: dict,
    dist_range: int = 3,
    prox_range: int = 2,
) -> tuple[float, float]:
    """Compute the optimal radial scaling factors for the proximal and distal segments.

    Collects lumen wall points from the first *prox_range* and last *dist_range*
    imaging frames as reference geometry, then calls the Rust
    ``find_proximal_distal_scaling`` routine to find the scaling factors that
    best match the anomalous segment endpoints to those references.

    Parameters
    ----------
    frames : list of PyFrame
        Ordered intravascular imaging frames for the vessel.
    centerline : PyCenterline
        Centerline of the vessel region.
    results : dict
        Labelled results dictionary containing ``"anomalous_points"``.
    dist_range : int, optional
        Number of frames from the distal end used as the distal reference.
        Default is ``3``.
    prox_range : int, optional
        Number of frames from the proximal end used as the proximal reference.
        Default is ``2``.

    Returns
    -------
    prox_scaling : float
        Optimal radial scaling factor for the proximal segment.
    dist_scaling : float
        Optimal radial scaling factor for the distal segment.
    """
    frame_points_dist = [
        (p.x, p.y, p.z) for f in frames[-dist_range:] for p in f.lumen.points
    ]
    frame_points_prox = [
        (p.x, p.y, p.z) for f in frames[0:prox_range] for p in f.lumen.points
    ]
    n_anomalous_points = len(results["anomalous_points"])
    n_section: int = int(np.ceil(0.25 * n_anomalous_points))

    print("\nFinding best proximal/distal radial scaling factors...")
    prox_scaling, dist_scaling = find_proximal_distal_scaling(
        results["anomalous_points"],
        n_section,
        n_section,
        centerline,
        frame_points_prox,
        frame_points_dist,
    )
    print(f"Proximal scaling: {np.round(prox_scaling, 2)} mm")
    print(f"Distal scaling: {np.round(dist_scaling, 2)} mm")

    return prox_scaling, dist_scaling


def find_aorta_scaling(
    frames: list[PyFrame],
    cl_aorta: PyCenterline,
    results: dict,
) -> float:
    """Compute the optimal radial scaling factor for the aortic region.

    Extracts reconstructed wall points from the intravascular frames (using
    ``aortic_thickness`` and the ``"Wall"`` extras) as a reference, then calls
    the Rust ``find_aortic_scaling`` routine to determine the factor that best
    aligns the removed RCA points to those references.

    Parameters
    ----------
    frames : list of PyFrame
        Intravascular imaging frames containing ``aortic_thickness`` and
        ``extras["Wall"]`` data.
    cl_aorta : PyCenterline
        Centerline of the aortic region.
    results : dict
        Labelled results dictionary containing ``"rca_removed_points"``.
    debug_plot : bool, optional
        Reserved for future use; currently unused.  Default is ``True``.

    Returns
    -------
    float
        Optimal radial scaling factor for the aortic segment.
    """
    reference_points = _extract_wall_from_frames(frames)
    if reference_points is None:
        raise ValueError("No aortic wall points found in frames for scaling reference")

    print("\nFinding best aortic radial scaling factor...")
    scaling = find_aortic_scaling(
        results["rca_removed_points"],  # For now work with removed points
        reference_points,
        cl_aorta,
    )
    print(f"Aortic scaling: {np.round(scaling, 2)} mm")

    return scaling


def find_aortic_wall_scaling(
    frames: list[PyFrame],
    cl_aorta: PyCenterline,
    results: dict,
) -> float:
    """Compute the optimal radial scaling factor for the aortic wall region.
    This is created for anomalous coronaries, and tries to optimize the aortic wall
    to the point on the first quarter towards the aortic wall of the first round
    lumen (marking the end of the intramural course).

    End of the intramural course is defined as the first lumen with an elliptic ratio <1.3

    Parameters
    ----------
    frames : list of PyFrame
        Intravascular imaging frames.
    cl_aorta : PyCenterline
        Centerline of the aortic region.
    results : dict
        Labelled results dictionary containing ``"rca_removed_points"``.

    Returns
    -------
    float
        Optimal radial scaling factor for the aortic wall.
    """
    ref_point = None

    print("\nFinding best aortic wall radial scaling factor...")
    for frame in frames:
        elliptic_ratio = frame.lumen.get_elliptic_ratio()
        if elliptic_ratio < 1.3:
            print(f"elliptic ratio <1.3 for frame index {frame.id}")
            point_idx = len(frame.lumen) // 4
            ref_point_ir = frame.lumen.points[point_idx]
            ref_point = (ref_point_ir.x, ref_point_ir.y, ref_point_ir.z)
            break
        else:
            continue

    if ref_point is None:
        raise ValueError("No coronary reference point found")
    scaling = _find_aortic_wall_scaling(cl_aorta, ref_point, results["aorta_points"])
    print(f"Aortic wall scaling: {np.round(scaling, 2)} mm")

    return scaling


def sync_results_to_mesh(
    results: dict,
    old_mesh: trimesh.Trimesh,
    new_mesh: trimesh.Trimesh,
) -> dict:
    """Update all coordinate lists in *results* after vertices have been moved.

    Use this after :func:`scale_region_centerline_morphing` to keep the stored
    point lists consistent with the new vertex positions.  The two meshes must
    have the same vertex count and ordering (only positions change, no vertices
    are added or removed).

    Parameters
    ----------
    results : dict
        Results dict whose coordinate lists should be refreshed.
    old_mesh : trimesh.Trimesh
        The mesh whose vertex positions match the current coordinate lists.
    new_mesh : trimesh.Trimesh
        The mesh with updated vertex positions (same indices, new coordinates).

    Returns
    -------
    dict
        Updated *results* with ``"mesh"`` replaced by *new_mesh* and all
        coordinate lists remapped to the new vertex positions.
    """
    old_coord_to_idx = {tuple(v): i for i, v in enumerate(old_mesh.vertices)}

    def remap(pts: list) -> list:
        indices = [old_coord_to_idx.get(tuple(p)) for p in pts]
        return [tuple(new_mesh.vertices[i]) for i in indices if i is not None]

    updated = map_point_lists(results, remap)
    updated["mesh"] = new_mesh
    return updated


def sync_tree_to_mesh(
    tree: PyDiscretizedVesselTree,
    old_mesh: trimesh.Trimesh,
    new_mesh: trimesh.Trimesh,
    k: int = 4,
) -> PyDiscretizedVesselTree:
    """Move the contours of a discretized tree along with the mesh they were cut from.

    The tree's contours lie on the mesh surface, but are not mesh vertices, so
    unlike :func:`sync_results_to_mesh` they cannot be looked up by position.
    Each contour point instead moves by the displacement of its *k* nearest
    vertices of *old_mesh* (weighted by inverse distance), which follows any
    region scaling - proximal, distal, aortic or the remaining vessels - without
    knowing which region a contour belongs to.  Use it after every
    :func:`scale_region_centerline_morphing`, next to
    :func:`sync_results_to_mesh`, so the stitch (which reads the tree) sees the
    scaled vessels.

    Parameters
    ----------
    tree : PyDiscretizedVesselTree
        Tree discretized on *old_mesh* (or on a mesh synced to it).  Updated in
        place.
    old_mesh, new_mesh : trimesh.Trimesh
        The mesh before and after the vertices were moved, with the same vertex
        count and ordering.
    k : int, optional
        Number of nearest vertices each contour point follows (default 4).

    Returns
    -------
    PyDiscretizedVesselTree
        *tree*, with every contour (aorta and branches) and its centroid and
        the two ostium points moved, and the reference points recomputed.
    """
    old = np.asarray(old_mesh.vertices, dtype=np.float64)
    new = np.asarray(new_mesh.vertices, dtype=np.float64)
    if new.shape != old.shape:
        raise ValueError(
            "sync_tree_to_mesh needs meshes with the same vertices, only moved"
        )
    shift = new - old
    if not np.any(shift):
        return tree
    lookup = cKDTree(old)

    def follow(xyz: np.ndarray) -> np.ndarray:
        dist, idx = lookup.query(xyz, k=k)
        weight = 1.0 / np.maximum(dist, 1e-9)
        weight /= weight.sum(axis=1, keepdims=True)
        return xyz + (shift[idx] * weight[..., None]).sum(axis=1)

    def moved(contours: list) -> list:
        for contour in contours:
            points = contour.points
            xyz = follow(np.array([(p.x, p.y, p.z) for p in points], dtype=np.float64))
            for p, (x, y, z) in zip(points, xyz):
                p.x, p.y, p.z = float(x), float(y), float(z)
            contour.points = points
            contour.compute_centroid()
        return contours

    tree.discretized_aorta = moved(tree.discretized_aorta)
    for vessel in ("rca", "lca"):
        branches = getattr(tree, vessel)
        for branch in branches.values():
            branch.contours = moved(branch.contours)
        setattr(tree, vessel, branches)
    ostia = follow(np.array([tree.ao_rca, tree.ao_lca], dtype=np.float64))
    tree.ao_rca, tree.ao_lca = ((float(x), float(y), float(z)) for x, y, z in ostia)
    tree.calculate_ref_pts()
    return tree
