from __future__ import annotations

from collections.abc import Mapping
import numpy as np
import trimesh

from ..multimodars import (
    PyGeometry,
    PyFrame,
    PyCenterline,
    build_adjacency_map,
    adjust_diameter_centerline_morphing_simple,
    find_proximal_distal_scaling,
    find_aortic_scaling,
    find_aortic_wall_scaling as _find_aortic_wall_scaling,
)
from .._converters import geometry_to_trimesh


def _project_to_best_fit_plane(
    points: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    """Project a ring of boundary points onto their best-fit plane.

    Fits a plane via SVD (the plane normal is the direction of minimum variance)
    and orthogonally projects every point onto it, flattening noise perpendicular
    to the ring.
    """
    if len(points) < 3:
        return points
    pts = np.array(points, dtype=np.float64)
    centroid = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - centroid, full_matrices=False)
    normal = Vt[-1]
    distances = (pts - centroid) @ normal
    projected = pts - np.outer(distances, normal)
    return [tuple(p) for p in projected]


def _plane_normal_svd(pts: np.ndarray) -> np.ndarray:
    """Best-fit plane normal for a point cloud via SVD (minimum-variance axis)."""
    centroid = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - centroid, full_matrices=False)
    return Vt[-1]


def _angle_between_planes_deg(n1: np.ndarray, n2: np.ndarray) -> float:
    """Acute angle in degrees between two planes given their normals."""
    cos = np.clip(np.abs(np.dot(n1, n2)), 0.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def _clamp_to_plane(
    points: list[tuple[float, float, float]],
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    overshoot: float = 0.0,
) -> list[tuple[float, float, float]]:
    """Clamp wrong-side points to the IV plane, then enforce a minimum gap.

    Step 1: project any point on the wrong side of the plane onto it.
    Step 2: if ``overshoot`` > 0, every point (including freshly clamped ones
    that now sit exactly on the plane) that is within ``overshoot`` mm of the
    plane on the correct side is pushed further away until it is exactly
    ``overshoot`` mm from the plane.  This creates a clean buffer zone between
    the aortic boundary ring and the IV ostium plane, avoiding the sharp angle
    that would otherwise form.
    """
    pts = np.array(points, dtype=np.float64)
    dists = (pts - plane_origin) @ plane_normal
    correct_sign = np.sign(np.median(dists))

    # Step 1: project wrong-side points onto the plane
    wrong = (np.sign(dists) != correct_sign) & (dists != 0.0)
    pts[wrong] -= np.outer(dists[wrong], plane_normal)

    if overshoot > 0.0:
        # Step 2: recompute distances and push any point within the buffer zone
        # further away on the aortic (correct) side
        dists2 = (pts - plane_origin) @ plane_normal
        signed_dist = correct_sign * dists2  # positive = on correct side
        too_close = signed_dist < overshoot
        deficit = overshoot - signed_dist[too_close]
        pts[too_close] += np.outer(deficit * correct_sign, plane_normal)

    return [tuple(p) for p in pts]


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


def _order_boundary_components(
    boundary_indices: set[int],
    adj_map: Mapping[int, list[int] | set[int]],
) -> list[list[int]]:
    """Walk every connected component of the boundary in edge order.

    Returns one ordered list per component so callers can project each ring
    onto its own best-fit plane rather than a combined plane fitted to all rings.
    """
    if not boundary_indices:
        return []
    if len(boundary_indices) == 1:
        return [list(boundary_indices)]

    ring_adj = {
        i: [j for j in adj_map.get(i, []) if j in boundary_indices]
        for i in boundary_indices
    }

    remaining = set(boundary_indices)
    components: list[list[int]] = []

    while remaining:
        start = next(iter(remaining))
        component = [start]
        remaining.discard(start)
        prev, current = -1, start

        while True:
            nxt = next(
                (n for n in ring_adj.get(current, []) if n != prev and n in remaining),
                None,
            )
            if nxt is None:
                break
            component.append(nxt)
            remaining.discard(nxt)
            prev, current = current, nxt

        components.append(component)

    return components


def scale_region_centerline_morphing(
    mesh: trimesh.Trimesh,
    region_points: list,
    centerline,
    diameter_adjustment_mm: float,
) -> trimesh.Trimesh:
    """Scale a mesh region radially around its centerline.

    Each vertex in *region_points* is displaced along the direction from the
    nearest centerline point outward (positive *diameter_adjustment_mm*) or
    inward (negative).

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The original mesh.  A copy is returned; the input is not modified.
    region_points : list of tuple
        ``(x, y, z)`` coordinates of the vertices to be scaled.  Only vertices
        present in this list are moved.
    centerline : PyCenterline
        Centerline of the vessel region used as the morphing axis.
    diameter_adjustment_mm : float
        Diameter change in millimetres.  Positive values expand the lumen;
        negative values contract it.

    Returns
    -------
    trimesh.Trimesh
        A new mesh with the selected region scaled.

    Warns
    -----
    If no vertices matching *region_points* are found in the mesh, a warning
    is printed and the unmodified copy is returned.
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
    print(f"Diameter adjustment: {np.round(diameter_adjustment_mm, 2)} mm")

    region_vertices_list = [
        tuple(vertex) for vertex in scaled_mesh.vertices[region_vertex_indices]
    ]
    adjusted_points = adjust_diameter_centerline_morphing_simple(
        centerline=centerline,
        points=region_vertices_list,
        diameter_adjustment_mm=diameter_adjustment_mm,
    )

    scaled_mesh.vertices[region_vertex_indices] = np.array(
        adjusted_points, dtype=np.float64
    )

    # Clear mesh cache since we modified vertices directly
    scaled_mesh.vertices.flags["WRITEABLE"] = False

    return scaled_mesh


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


def _extract_wall_from_frames(frames) -> list[tuple[float, float, float]] | None:
    """Extract the straight-wall (coronary-side) points from intravascular frames.

    ``create_aortic_wall`` in ``wall.rs`` builds the ``"Wall"`` extra contour
    in two halves:

    * **Straight wall** - ``point_index`` 0 to ``n // 2`` (exclusive): the lumen
      contour offset outward by 1 mm, following the true circular/elliptic vessel
      geometry on the coronary side.
    * **Aortic wall** - ``point_index`` ``n // 2`` to ``n``: the rectangular
      aortic-thickness shape constructed from ``aortic_thickness``.

    Only the straight-wall half is returned because it preserves the actual vessel
    cross-section shape and is therefore a stable reference for radial scaling.
    Assumes an even number of points per frame (the standard 500-point geometry).

    Parameters
    ----------
    frames : list of PyFrame
        Intravascular imaging frames.  Frames without ``aortic_thickness`` are
        skipped.

    Returns
    -------
    list of tuple
        ``(x, y, z)`` tuples of straight-wall points from the last eligible frame.
        Returns ``None`` if no eligible frame is found.

    Raises
    ------
    ValueError
        If an eligible frame is missing the ``"Wall"`` extras entry or that
        entry is empty.
    """
    n_points = len(frames[0].lumen.points)
    half = n_points // 2

    reference_points = None

    for frame in frames:
        if frame.lumen.aortic_thickness is None:
            continue
        wall = frame.extras.get("Wall")
        if wall is None:
            raise ValueError(
                f"No Wall extras found for frame {getattr(frame, 'frame', '?')}"
            )
        if not wall.points:
            raise ValueError(
                f"Empty Wall extras for frame {getattr(frame, 'frame', '?')}"
            )

        # Straight wall: coronary-side offset lumen, point_index 0..half.
        # Aortic wall:   rectangular aortic-thickness shape, point_index half..n_points.
        reference_points = [
            (p.x, p.y, p.z) for p in wall.points if p.point_index < half
        ]

    return reference_points


def remove_labeled_points_from_mesh(
    results: dict,
    region_keys: list[str] | str = "anomalous_points",
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

    Returns
    -------
    dict
        Updated *results* dict with ``"mesh"`` replaced by the trimmed mesh,
        all *region_keys* cleared, and all other coordinate lists remapped to
        the new vertex set.  A new ``"boundary_points"`` entry is added
        containing the open-boundary ring vertices adjacent to the removed
        region(s).
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

    # 3. Build adjacency map and find boundary vertices before removing faces:
    #    kept vertices that had at least one removed neighbour will form the
    #    open boundary ring used for stitching.
    adj_map = build_adjacency_map(mesh.faces.tolist())
    boundary_indices = {
        i
        for i in range(n_vertices)
        if keep_mask[i] and any(j in remove_indices for j in adj_map.get(i, []))
    }
    components = _order_boundary_components(boundary_indices, adj_map)
    boundary_points: list[tuple] = [
        tuple(mesh.vertices[i]) for component in components for i in component
    ]

    # 4. Drop faces that reference any removed vertex
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
    updated["boundary_points"] = boundary_points

    print(f"Applying removal of '{region_keys}'")
    print(f"Removed {len(points_to_remove)}")
    print(f"Created {len(updated['boundary_points'])} boundary points")

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

    Returns
    -------
    dict
        Updated *results* dict with ``"mesh"`` replaced by the trimmed mesh
        and all other coordinate lists filtered to the surviving vertex set.
        A new ``"boundary_points"`` entry is added containing the open-boundary
        ring vertices adjacent to the removed region.
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
    components = _order_boundary_components(boundary_indices, adj_map)
    boundary_points: list[tuple] = [
        tuple(mesh.vertices[i]) for component in components for i in component
    ]

    # Drop faces that reference any removed vertex
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
    updated["boundary_points"] = boundary_points

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

    updated = dict(results)
    updated["mesh"] = new_mesh

    for key in (
        "aorta_points",
        "rca_points",
        "lca_points",
        "rca_removed_points",
        "lca_removed_points",
        "proximal_points",
        "distal_points",
        "anomalous_points",
        "boundary_points",
    ):
        if key not in updated or not updated[key]:
            continue
        indices = [old_coord_to_idx.get(tuple(p)) for p in updated[key]]
        updated[key] = [tuple(new_mesh.vertices[i]) for i in indices if i is not None]

    return updated


def _rotate_to_nearest_iv(boundary_pts: list, iv_pt) -> list:
    """Rotate a boundary ring so the point nearest to *iv_pt* is first."""
    iv_arr = np.array([iv_pt.x, iv_pt.y, iv_pt.z])
    dists = [np.linalg.norm(np.array(pt) - iv_arr) for pt in boundary_pts]
    start_idx = int(np.argmin(dists))
    return boundary_pts[start_idx:] + boundary_pts[:start_idx]


def _adjust_start_point_by_z(boundary_pts: list) -> list:
    """Rotate a boundary ring so the point with the highest z-value is first."""
    start_idx = int(np.argmax([pt[2] for pt in boundary_pts]))
    return boundary_pts[start_idx:] + boundary_pts[:start_idx]


def stitch_ccta_to_intravascular(
    iv_mesh: PyGeometry,
    mesh: trimesh.Trimesh,
    results: dict,
    n_points_iv_cont: int = 100,
    prox_start_mode: str = "nearest_iv",
    dist_start_mode: str = "nearest_iv",
    proximal_is_ostium: bool = True,
    clamp_overshoot: float = 0.5,
) -> dict:
    """Stitch an aligned intravascular mesh to a CCTA mesh.

    ``prox_start_mode`` / ``dist_start_mode`` control how index 0 of each
    boundary ring is chosen before stitching:

    * ``"nearest_iv"`` (default) - rotate to the point closest to IV point 0.
    * ``"highest_z"`` - rotate to the point with the largest z-coordinate.

    ``clamp_overshoot`` sets the minimum distance (mm) that every proximal
    boundary point must sit away from the IV plane after clamping.  Points
    that land too close are pushed further until they are exactly
    ``clamp_overshoot`` mm from the plane, creating a slight inward step that
    softens the stitching angle.  The two mesh rings adjacent to the boundary
    are also pushed radially outward (ring 1: 0.1 mm, ring 2: 0.2 mm) within
    the IV plane to avoid ridges at the clamping zone.  Only active when the
    boundary-ring plane and the IV plane form an angle ≥ ``ostium_angle_threshold_deg``
    (default 45°).
    """
    iv_mesh = iv_mesh.downsample(n_points_iv_cont)
    iv_mesh_points = [
        (p.x, p.y, p.z) for frame in iv_mesh.frames for p in frame.lumen.points
    ]
    proximal_centroid = iv_mesh.frames[0].centroid
    distal_centroid = iv_mesh.frames[-1].centroid
    proximal_points = iv_mesh.frames[0].lumen.points
    distal_points = iv_mesh.frames[-1].lumen.points

    prox_boundary_pts, dist_boundary_pts, mesh = _prepare_prox_dist_boundary_pts(
        mesh,
        results,
        proximal_centroid,
        distal_centroid,
        proximal_is_ostium=proximal_is_ostium,
        proximal_iv_frame_pts=iv_mesh.frames[0].lumen.points,
        clamp_overshoot=clamp_overshoot,
    )
    prox_point_step = len(proximal_points) // len(prox_boundary_pts)
    dist_point_step = len(distal_points) // len(dist_boundary_pts)

    # Adjust start point
    if prox_start_mode == "highest_z" or dist_start_mode == "highest_z":
        iv_mesh = iv_mesh.sort_frame_points()
        proximal_points = iv_mesh.frames[0].lumen.points
        distal_points = iv_mesh.frames[-1].lumen.points
    if prox_start_mode == "highest_z":
        prox_boundary_pts = _adjust_start_point_by_z(prox_boundary_pts)
    else:
        prox_boundary_pts = _rotate_to_nearest_iv(prox_boundary_pts, proximal_points[0])
    if dist_start_mode == "highest_z":
        dist_boundary_pts = _adjust_start_point_by_z(dist_boundary_pts)
    else:
        dist_boundary_pts = _rotate_to_nearest_iv(dist_boundary_pts, distal_points[0])

    # Compute the vessel axis so each patch can be flipped to face outward,
    # and also used as the consistent reference normal for direction checking.
    # frames[0] is the "proximal" end, frames[-1] the "distal" end.
    # The outward direction for the proximal patch points away from the mesh
    # interior (toward frames[0]), and vice-versa for the distal patch.
    prox_c = np.array(iv_mesh.frames[0].centroid)
    dist_c = np.array(iv_mesh.frames[-1].centroid)
    prox_outward = prox_c - dist_c  # points toward the proximal end
    dist_outward = dist_c - prox_c  # points toward the distal end

    # Check / fix winding direction of each boundary ring vs its IV ring
    # independently, using the method that matches the start-point strategy.
    if prox_start_mode == "highest_z":
        prox_boundary_pts = _fix_ring_direction_by_winding(
            prox_boundary_pts, proximal_points
        )
    else:
        prox_boundary_pts = _fix_ring_direction_by_distance(
            prox_boundary_pts, proximal_points, prox_point_step
        )

    if dist_start_mode == "highest_z":
        dist_boundary_pts = _fix_ring_direction_by_winding(
            dist_boundary_pts, distal_points
        )
    else:
        dist_boundary_pts = _fix_ring_direction_by_distance(
            dist_boundary_pts, distal_points, dist_point_step
        )

    # Step 3: stitch each boundary ring to its IV ring
    prox_patch = _stitch_boundary_ring(
        prox_boundary_pts, proximal_points, prox_point_step, prox_outward
    )
    dist_patch = _stitch_boundary_ring(
        dist_boundary_pts, distal_points, dist_point_step, dist_outward
    )
    test_mesh = geometry_to_trimesh(iv_mesh)
    test_mesh.update_faces(test_mesh.unique_faces())
    test_mesh.update_faces(test_mesh.nondegenerate_faces())
    test_mesh.fix_normals()
    mesh = trimesh.util.concatenate([mesh, prox_patch, dist_patch, test_mesh])
    trimesh.tol.merge = 0.001
    mesh.merge_vertices()
    if not mesh.is_watertight:
        mesh.fill_holes()
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()

    results["prox_boundary_points"] = prox_boundary_pts
    results["dist_boundary_points"] = dist_boundary_pts
    results["anomalous_points"] = iv_mesh_points
    results["rca_points"] = (
        iv_mesh_points + results["distal_points"] + results["proximal_points"]
    )
    results["mesh"] = mesh

    return results


def _enforce_layer_gap_from_plane(
    mesh: trimesh.Trimesh,
    seed_indices: set[int],
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    layer_step_mm: float = 0.1,
    n_rings: int = 2,
) -> trimesh.Trimesh:
    """Push neighboring mesh rings radially away from the IV ring center.

    The boundary ring was clamped toward the IV plane, which can leave second-
    and third-layer aortic vertices sitting closer to the coronary axis than
    the boundary ring itself — creating a visible ridge.  The fix is to push
    those vertices outward *within* the IV plane (i.e. along the aortic
    surface, away from the coronary center), not perpendicular to it.

    Ring k is displaced by ``k * layer_step_mm`` in the radial direction:
    the projection of the vertex onto the IV plane, measured from the IV
    ring centre (``plane_origin``), gives the outward direction.
    """
    adj_map = build_adjacency_map(mesh.faces.tolist())
    new_vertices = mesh.vertices.copy()

    frontier = set(seed_indices)
    visited = set(seed_indices)

    for ring in range(1, n_rings + 1):
        push_dist = ring * layer_step_mm
        next_frontier = set()
        for vi in frontier:
            for nb in adj_map.get(vi, []):
                if nb not in visited:
                    next_frontier.add(nb)

        for vi in next_frontier:
            p = new_vertices[vi]
            # Project the vertex onto the IV plane to get its lateral position
            p_proj = p - float(np.dot(p - plane_origin, plane_normal)) * plane_normal
            # Radial direction: from IV ring centre outward, within the IV plane
            radial = p_proj - plane_origin
            r_norm = np.linalg.norm(radial)
            if r_norm < 1e-10:
                continue
            new_vertices[vi] = p + (push_dist / r_norm) * radial

        visited.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break

    return trimesh.Trimesh(vertices=new_vertices, faces=mesh.faces, process=False)


def _prepare_prox_dist_boundary_pts(
    mesh: trimesh.Trimesh,
    results: dict,
    prox_centroid: tuple[float, float, float],
    dist_centroid: tuple[float, float, float],
    proximal_is_ostium: bool = True,
    proximal_iv_frame_pts=None,
    ostium_angle_threshold_deg: float = 45.0,
    clamp_overshoot: float = 1.0,
) -> tuple[list, list, trimesh.Trimesh]:
    proximal_boundary_pts = []
    distal_boundary_pts = []
    for pt in results["boundary_points"]:
        distance_prox = np.linalg.norm(np.array(prox_centroid) - np.array(pt))
        distance_dist = np.linalg.norm(np.array(dist_centroid) - np.array(pt))
        if distance_prox <= distance_dist:
            proximal_boundary_pts.append(pt)
        else:
            distal_boundary_pts.append(pt)

    if proximal_is_ostium:
        # Project onto best-fit plane + Laplacian smooth.
        prox_projected = _project_to_best_fit_plane(proximal_boundary_pts)
        prox_boundary_pts_ord = _smooth_ring_laplacian(prox_projected)

        # Final check for anomalous vessels: the aortic ostium is nearly
        # perpendicular to the coronary ostial plane, which can leave some
        # boundary points on the wrong side after smoothing.  Compare the two
        # plane normals and, if the angle exceeds the threshold, project any
        # outlier that crossed the IV-frame plane back onto it.
        iv_origin: np.ndarray | None = None
        iv_normal: np.ndarray | None = None
        clamping_applied = False
        if proximal_iv_frame_pts is not None and len(prox_boundary_pts_ord) >= 3:
            boundary_arr = np.array(prox_boundary_pts_ord, dtype=np.float64)
            iv_arr = np.array(
                [[p.x, p.y, p.z] for p in proximal_iv_frame_pts], dtype=np.float64
            )
            boundary_normal = _plane_normal_svd(boundary_arr)
            iv_normal = _plane_normal_svd(iv_arr)
            angle = _angle_between_planes_deg(boundary_normal, iv_normal)
            if angle >= ostium_angle_threshold_deg:
                iv_origin = np.array(prox_centroid, dtype=np.float64)
                prox_boundary_pts_ord = _clamp_to_plane(
                    prox_boundary_pts_ord,
                    iv_origin,
                    iv_normal,
                    overshoot=clamp_overshoot,
                )
                clamping_applied = True

        coord_to_idx = {tuple(v): i for i, v in enumerate(mesh.vertices)}
        new_vertices = mesh.vertices.copy()
        fixed_indices: set[int] = set()
        for old_pt, new_pt in zip(proximal_boundary_pts, prox_boundary_pts_ord):
            idx = coord_to_idx.get(tuple(old_pt))
            if idx is not None:
                new_vertices[idx] = new_pt
                fixed_indices.add(idx)
        mesh = trimesh.Trimesh(vertices=new_vertices, faces=mesh.faces, process=False)

        if clamping_applied and fixed_indices:
            assert iv_origin is not None and iv_normal is not None
            mesh = _enforce_layer_gap_from_plane(
                mesh, fixed_indices, iv_origin, iv_normal
            )
    else:
        prox_boundary_pts_ord = order_points_list(mesh, proximal_boundary_pts)

    dist_boundary_pts_ord = order_points_list(mesh, distal_boundary_pts)

    return prox_boundary_pts_ord, dist_boundary_pts_ord, mesh


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


def _signed_area_projected(pts: list, normal: np.ndarray) -> float:
    """Signed area of a polygon projected onto the plane with the given normal.

    Positive = CCW when viewed from the normal direction.
    """
    ref = (
        np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    )
    u = np.cross(normal, ref)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    arr = np.array(pts)
    us = arr @ u
    vs = arr @ v
    return float(0.5 * np.sum(us * np.roll(vs, -1) - np.roll(us, -1) * vs))


def _newell_normal(pts: list) -> np.ndarray:
    """Compute a polygon's outward normal via Newell's method.

    The resulting normal points in the direction from which the polygon
    appears CCW — so ``_signed_area_projected(pts, _newell_normal(pts))``
    is always positive for any non-degenerate polygon.
    """
    normal = np.zeros(3)
    n = len(pts)
    arr = np.array(pts)
    for i in range(n):
        c = arr[i]
        nx = arr[(i + 1) % n]
        normal[0] += (c[1] - nx[1]) * (c[2] + nx[2])
        normal[1] += (c[2] - nx[2]) * (c[0] + nx[0])
        normal[2] += (c[0] - nx[0]) * (c[1] + nx[1])
    length = np.linalg.norm(normal)
    return normal / length if length > 1e-10 else np.array([0.0, 0.0, 1.0])


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


def _stitch_boundary_ring(
    boundary_pts: list,
    iv_pts,
    step: int,
    outward_direction: np.ndarray | None = None,
) -> trimesh.Trimesh:
    """Create a patch mesh stitching an IV lumen ring to a CCTA boundary ring.

    For each pair of consecutive boundary vertices (b, b+1), *step* IV points
    are assigned (the first ``r = n_iv % n_boundary`` segments receive
    ``step + 1`` points to absorb the remainder without gaps).  Each segment
    is split at its midpoint:

    * Indices [0, mid)      → fan triangles into boundary[b].
    * Indices [mid, end-1)  → fan triangles into boundary[b+1].
    * One bridging triangle: (boundary[b], boundary[b+1], iv[mid]).

    Parameters
    ----------
    boundary_pts : list of tuple
        Ordered CCTA boundary vertices.
    iv_pts : list of Point
        Ordered IV lumen points (with .x / .y / .z attributes).
    step : int
        Base number of IV points per boundary segment
        (``len(iv_pts) // len(boundary_pts)``).

    Returns
    -------
    trimesh.Trimesh
        Patch mesh with combined vertices and stitching faces.
    """
    n_boundary = len(boundary_pts)
    n_iv = len(iv_pts)
    remainder = n_iv % n_boundary  # extra IV points distributed to first segments

    b_arr = np.array(boundary_pts, dtype=np.float64)
    iv_arr = np.array([(p.x, p.y, p.z) for p in iv_pts], dtype=np.float64)

    # Vertices: boundary indices 0..n_boundary-1, IV indices n_boundary..n_boundary+n_iv-1
    vertices = np.vstack([b_arr, iv_arr])

    faces = []
    iv_start = 0

    for b in range(n_boundary):
        b_next = (b + 1) % n_boundary
        seg_len = step + 1 if b < remainder else step
        iv_end = iv_start + seg_len
        mid = iv_start + seg_len // 2

        # First half: fan into boundary[b]
        for i in range(iv_start, mid):
            i_next = (i + 1) % n_iv
            faces.append((n_boundary + i, n_boundary + i_next, b))

        # Second half: fan into boundary[b+1]
        for i in range(mid, iv_end - 1):
            i_next = (i + 1) % n_iv
            faces.append((n_boundary + i, n_boundary + i_next, b_next))

        # Bridging triangle connecting both boundary vertices at the midpoint.
        # Winding is reversed vs the naive (b, b_next, mid) order so that the
        # shared edges with the adjacent fan triangles are traversed in opposite
        # directions — the requirement for consistent outward normals.
        faces.append((b_next, b, n_boundary + mid))

        iv_start = iv_end

    print(
        f"Stitching: {len(faces)}/{n_iv} triangles created "
        f"(n_boundary={n_boundary}, n_iv={n_iv}, step={step}, remainder={remainder})"
    )

    patch = trimesh.Trimesh(
        vertices=vertices,
        faces=np.array(faces, dtype=np.int64),
        process=False,
    )

    if outward_direction is not None:
        # After the bridging-winding fix all faces in this patch are consistently
        # oriented, but the whole patch may still face inward (this happens for
        # the proximal ring because its IV lumen winds in the opposite sense vs
        # the distal ring when viewed from a fixed external direction).
        # Compare the average face normal against the known vessel-axis direction
        # that points outward for this patch.  For an approximately flat annular
        # patch the average normal is a reliable indicator of face orientation.
        face_normals = patch.face_normals  # (N, 3), unit normals per face
        valid = ~np.isnan(face_normals).any(axis=1)
        if valid.any():
            avg_normal = face_normals[valid].mean(axis=0)
            if np.dot(avg_normal, outward_direction) < 0:
                patch.faces = patch.faces[:, ::-1]

    return patch
