from __future__ import annotations

import numpy as np
import trimesh
from pathlib import Path
import warnings
from ..multimodars import (
    PyGeometry,
    find_centerline_bounded_points_simple,
    remove_occluded_points_ray_triangle,
    clean_outlier_points,
    build_adjacency_map,
    find_points_by_cl_region,
    adjust_diameter_centerline_morphing_simple,
    find_proximal_distal_scaling,
    find_aortic_scaling,
)
from .._converters import numpy_to_centerline, geometry_to_trimesh
from ..io.read_geometrical import read_mesh
from .debug_plots import labeled_geometry_plot, plot_anomalous_region, compare_centerline_scaling


def label_geometry(
    path_ccta_geometry: Path | str,
    path_centerline_aorta: Path | str,
    path_centerline_rca: Path | str,
    path_centerline_lca: Path | str,
    anomalous_rca: bool = False,
    anomalous_lca: bool = False,
    n_points_intramural: int = 120,
    bounding_sphere_radius_mm: float = 3.0,
    tolerance_float: float = 1e-6,
    control_plot: bool = True,
) -> tuple[dict, tuple[any, any, any]]:
    """Label CCTA mesh vertices as aorta, RCA, or LCA using centerline-based region detection.

    Loads a 3-D surface mesh and three centerlines (aorta, RCA, LCA), then assigns
    each mesh vertex to one of the anatomical regions. For anomalous vessels an
    additional occlusion-removal step uses ray-triangle intersection to strip
    intramural segments, followed by adjacency-map reclassification to clean up
    isolated mis-labelled vertices. Herfore, a ray is cast from every aorta point to the centerline
    points of the anomalous section and if 3 faces are intersected by the ray the points from
    the first face must correspond to the intramural section.

    Parameters
    ----------
    path_ccta_geometry : Path or str
        Path to the CCTA surface mesh file (any format supported by
        :func:`multimodars.io.read_geometrical.read_mesh`).
    path_centerline_aorta : Path or str
        Path to a CSV file containing the aortic centerline (comma-delimited,
        columns: x, y, z, …).
    path_centerline_rca : Path or str
        Path to a CSV file containing the RCA centerline.
    path_centerline_lca : Path or str
        Path to a CSV file containing the LCA centerline.
    anomalous_rca : bool, optional
        When ``True`` applies ray-triangle occlusion removal to the RCA region
        to handle anomalous (intramural) courses.  Default is ``False``.
    anomalous_lca : bool, optional
        When ``True`` applies ray-triangle occlusion removal to the LCA region.
        Default is ``False``.
    n_points_intramural : int, optional
        Number of coronary centerline points examined during occlusion removal
        (the intramural segment length).  Default is ``120``.
    bounding_sphere_radius_mm : float, optional
        Radius in millimetres of the rolling sphere used to collect candidate
        mesh vertices around each centerline point.  Default is ``3.0``.
    tolerance_float : float, optional
        Distance tolerance used when matching mesh vertices to points during
        face lookup.  Default is ``1e-6``.
    control_plot : bool, optional
        When ``True`` opens an interactive 3-D scene showing the labelled mesh
        after processing.  Default is ``True``.

    Returns
    -------
    results : dict
        Dictionary with keys:

        * ``"mesh"`` - the original :class:`trimesh.Trimesh` object.
        * ``"aorta_points"`` - list of ``(x, y, z)`` tuples for aortic vertices.
        * ``"rca_points"`` - list of ``(x, y, z)`` tuples for RCA vertices.
        * ``"lca_points"`` - list of ``(x, y, z)`` tuples for LCA vertices.
        * ``"rca_removed_points"`` - RCA vertices removed by occlusion detection.
        * ``"lca_removed_points"`` - LCA vertices removed by occlusion detection.

    centerlines : tuple
        A 3-tuple ``(cl_rca, cl_lca, cl_aorta)`` of ``PyCenterline`` objects.

    Raises
    ------
    Exception
        Re-raises any error that occurs while reading the mesh or centerline
        files, after printing a descriptive message.
    """
    try:
        mesh = read_mesh(path_ccta_geometry)
        print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    except Exception as e:
        print(f"Error reading CCTA mesh from {path_ccta_geometry}: {e}")
        raise

    try:
        cl_aorta_raw = np.genfromtxt(path_centerline_aorta, delimiter=",")
        cl_aorta = numpy_to_centerline(cl_aorta_raw)
        print(f"Loaded aorta centerline: {len(cl_aorta.points)} points")
    except Exception as e:
        print(f"Error reading Aorta centerline from {path_centerline_aorta}: {e}")
        raise

    try:
        cl_lca_raw = np.genfromtxt(path_centerline_lca, delimiter=",")
        cl_lca = numpy_to_centerline(cl_lca_raw)
        print(f"Loaded LCA centerline: {len(cl_lca.points)} points")
    except Exception as e:
        print(f"Error reading LCA centerline from {path_centerline_lca}: {e}")
        raise

    try:
        cl_rca_raw = np.genfromtxt(path_centerline_rca, delimiter=",")
        cl_rca = numpy_to_centerline(cl_rca_raw)
        print(f"Loaded RCA centerline: {len(cl_rca.points)} points")
    except Exception as e:
        print(f"Error reading RCA centerline from {path_centerline_rca}: {e}")
        raise

    points_list = [tuple(vertex) for vertex in mesh.vertices.tolist()]

    # Rust implementation using a rolling sphere with fixed radius
    rca_points_found = find_centerline_bounded_points_simple(
        cl_rca, points_list, bounding_sphere_radius_mm
    )
    lca_points_found = find_centerline_bounded_points_simple(
        cl_lca, points_list, bounding_sphere_radius_mm
    )

    print(f"RCA points found: {len(rca_points_found)}")
    print(f"LCA points found: {len(lca_points_found)}")

    rca_removed_points = []
    lca_removed_points = []

    if anomalous_rca:
        print("Applying occlusion removal for anomalous RCA...")
        rca_faces_for_rust = _prepare_faces_for_rust(
            mesh, points=rca_points_found, tol=tolerance_float
        )
        # Rust implementation, that creates ray between aortic and coronary centerline, and
        # removes faces if 3 consecutive faces are "pierced" by the ray
        final_rca_points_found = remove_occluded_points_ray_triangle(
            centerline_coronary=cl_rca,
            centerline_aorta=cl_aorta,
            range_coronary=n_points_intramural,
            points=rca_points_found,
            faces=rca_faces_for_rust,
        )
        rca_removed_points = [
            p for p in rca_points_found if p not in final_rca_points_found
        ]
        print(f"RCA: removed {len(rca_removed_points)} occluded points")
    else:
        final_rca_points_found = rca_points_found.copy()

    if anomalous_lca:
        print("Applying occlusion removal for anomalous LCA...")
        lca_faces_for_rust = _prepare_faces_for_rust(
            mesh, points=lca_points_found, tol=tolerance_float
        )
        final_lca_points_found = remove_occluded_points_ray_triangle(
            centerline_coronary=cl_lca,
            centerline_aorta=cl_aorta,
            range_coronary=n_points_intramural,
            points=lca_points_found,
            faces=lca_faces_for_rust,
        )
        lca_removed_points = [
            p for p in lca_points_found if p not in final_lca_points_found
        ]
        print(f"LCA: removed {len(lca_removed_points)} occluded points")
    else:
        final_lca_points_found = lca_points_found.copy()

    print(f"Removing LCA and RCA island points...")
    aortic_points = _find_aortic_points(
        mesh.vertices, final_rca_points_found, final_lca_points_found
    )
    print(f"length before: {len(final_lca_points_found)}")
    final_lca_points, final_aortic_points = clean_outlier_points(
        final_lca_points_found, aortic_points, 2.0, 0.4
    )  # based on patient data, only precleaning anyways, rest done by final_reclassification
    final_rca_points, _aortic_points = clean_outlier_points(
        final_rca_points_found, final_aortic_points, 2.0, 0.4
    )
    final_aortic_points = _find_aortic_points(
        mesh.vertices, final_rca_points, final_lca_points
    )
    # add also the rca_removed points and lca_removed points to aortic points
    final_aortic_points = list(
        set(final_aortic_points) | set(rca_removed_points) | set(lca_removed_points)
    )
    print(f"length after: {len(final_lca_points)}")

    results = {
        "mesh": mesh,
        "aorta_points": final_aortic_points,
        "rca_points": final_rca_points_found,
        "lca_points": final_lca_points,
        "rca_removed_points": rca_removed_points,
        "lca_removed_points": lca_removed_points,
    }

    # final reclassification based on adjacency map
    print("Applying final reclassification based on adjacency map...")
    new_results = _final_reclassification(results)

    if control_plot:
        labeled_geometry_plot(
            mesh=new_results["mesh"],
            rca_points=new_results["rca_points"],
            lca_points=new_results["lca_points"],
            rca_removed_points=new_results["rca_removed_points"],
            lca_removed_points=new_results["lca_removed_points"],
            cl_rca=cl_rca,
            cl_lca=cl_lca,
            cl_aorta=cl_aorta,
        )

    return new_results, (cl_rca, cl_lca, cl_aorta)


def _prepare_faces_for_rust(
    mesh: trimesh.Trimesh, *, points=None, face_indices=None, tol: float = 1e-6
):
    """Convert selected mesh faces to the Rust-friendly format.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Source mesh whose faces will be converted.
    points : list of tuple, optional
        If provided and *face_indices* is ``None``, the face indices are
        derived by finding which faces reference vertices closest to these
        points (within *tol*).
    face_indices : list of int, optional
        Explicit list of face indices to convert.  When given, *points* is
        ignored.  When both are ``None``, all faces are converted.
    tol : float, optional
        Distance tolerance for vertex matching when using *points*.
        Default is ``1e-6``.

    Returns
    -------
    list of tuple
        Each element is a ``((x0,y0,z0), (x1,y1,z1), (x2,y2,z2))`` triple of
        vertex coordinate tuples suitable for passing to Rust functions.
    """
    if face_indices is None:
        if points is not None:
            face_indices = _find_faces_for_points(mesh, points, tol=tol)
        else:
            face_indices = list(range(len(mesh.faces)))

    rust_faces = []
    for fi in face_indices:
        face = mesh.faces[fi]
        v0 = tuple(map(float, mesh.vertices[face[0]]))
        v1 = tuple(map(float, mesh.vertices[face[1]]))
        v2 = tuple(map(float, mesh.vertices[face[2]]))
        rust_faces.append((v0, v1, v2))
    return rust_faces


def _find_faces_for_points(mesh: trimesh.Trimesh, points_found, tol: float = 1e-6):
    """Find face indices whose vertices are within tolerance of the given points.

    For each point in *points_found* the nearest mesh vertex is located.
    Any face that references at least one of those vertices is included in the
    result.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to search.
    points_found : array-like of shape (N, 3)
        Query points.
    tol : float, optional
        Maximum distance from a query point to a mesh vertex for the vertex to
        be considered a match.  Default is ``1e-6``.

    Returns
    -------
    list of int
        Indices into ``mesh.faces`` for all faces that contain at least one
        matched vertex.  Returns an empty list when *points_found* is empty or
        no vertices fall within *tol*.
    """
    points_array = np.asarray(points_found, dtype=np.float64)
    if points_array.size == 0:
        return []

    found_vertex_indices = set()
    verts = mesh.vertices

    for p in points_array:
        distances = np.linalg.norm(verts - p, axis=1)
        closest_idx = int(np.argmin(distances))
        if distances[closest_idx] <= tol:
            found_vertex_indices.add(closest_idx)

    if not found_vertex_indices:
        return []

    face_indices = []
    for i, face in enumerate(mesh.faces):
        if (
            (face[0] in found_vertex_indices)
            or (face[1] in found_vertex_indices)
            or (face[2] in found_vertex_indices)
        ):
            face_indices.append(i)

    return face_indices


def _find_aortic_points(all_vertices, rca_points, lca_points):
    """Return mesh vertices that belong neither to the RCA nor to the LCA region.

    Parameters
    ----------
    all_vertices : array-like of shape (N, 3)
        All vertex coordinates of the mesh.
    rca_points : list of tuple
        Vertices classified as RCA.
    lca_points : list of tuple
        Vertices classified as LCA.

    Returns
    -------
    list of tuple
        ``(x, y, z)`` tuples for vertices not present in *rca_points* or
        *lca_points*.
    """
    rca_set = set(rca_points)
    lca_set = set(lca_points)
    aortic_points = [
        tuple(vertex)
        for vertex in all_vertices
        if tuple(vertex) not in rca_set and tuple(vertex) not in lca_set
    ]
    return aortic_points


def _final_reclassification(results: dict) -> dict:
    """Refine vertex labels using a mesh adjacency map.

    Applies two adjacency-based correction rules:

    * **Logic A** - An isolated RCA or LCA vertex (no same-label neighbours) is
      re-assigned to the aorta class.
    * **Logic B** - A vertex that was removed by occlusion detection but whose
      neighbours are predominantly (> 70 %) the corresponding coronary label is
      restored to that label.

    Parameters
    ----------
    results : dict
        Dictionary produced by :func:`label_geometry` containing keys
        ``"mesh"``, ``"rca_points"``, ``"lca_points"``,
        ``"rca_removed_points"``, and ``"lca_removed_points"``.

    Returns
    -------
    dict
        Updated dictionary with the same keys as *results* plus
        ``"aorta_points"``, with corrected point lists.
    """
    mesh = results["mesh"]
    n_vertices = len(mesh.vertices)

    # 1. Create a coordinate -> index map for fast lookup
    coord_to_idx = {tuple(coord): i for i, coord in enumerate(mesh.vertices)}

    # 2. Create the initial label array (Default to 0/Aorta)
    labels = np.zeros(n_vertices, dtype=np.uint8)

    # Labels based on existing result lists
    for pt in results["rca_points"]:
        if pt in coord_to_idx:
            labels[coord_to_idx[pt]] = 1
    for pt in results["lca_points"]:
        if pt in coord_to_idx:
            labels[coord_to_idx[pt]] = 2
    for pt in results["rca_removed_points"]:
        if pt in coord_to_idx:
            labels[coord_to_idx[pt]] = 3
    for pt in results["lca_removed_points"]:
        if pt in coord_to_idx:
            labels[coord_to_idx[pt]] = 4

    # 3. Build Adjacency Map
    adj_map = build_adjacency_map(mesh.faces.tolist())

    new_labels = labels.copy()

    # 4. Apply logic
    for i in range(n_vertices):
        neighbors = list(adj_map.get(i, []))
        if not neighbors:
            continue

        neighbor_labels = labels[neighbors]
        current_label = labels[i]

        # LOGIC A: Isolated RCA/LCA -> Aorta
        if current_label == 1 and not np.any(neighbor_labels == 1):
            new_labels[i] = 0
        elif current_label == 2 and not np.any(neighbor_labels == 2):
            new_labels[i] = 0

        # LOGIC B: Removed RCA/LCA points with most neighbours RCA/LCA -> RCA/LCA
        # If I am RCA_REMOVED(3) but MOST neighbors are NOT removed (e.g., they are RCA)
        elif current_label == 3:
            # "Most" here defined as > 70%
            non_removed_neighbors = np.sum(neighbor_labels == 1)
            if non_removed_neighbors > (len(neighbors) * 0.7):
                new_labels[i] = 1

        elif current_label == 4:
            non_removed_neighbors = np.sum(neighbor_labels == 2)
            if non_removed_neighbors > (len(neighbors) * 0.7):
                new_labels[i] = 2

    # 5. Convert back to coordinate lists for results dict
    updated_results = {
        "mesh": mesh,
        "rca_points": [
            tuple(mesh.vertices[i]) for i in range(n_vertices) if new_labels[i] == 1
        ],
        "lca_points": [
            tuple(mesh.vertices[i]) for i in range(n_vertices) if new_labels[i] == 2
        ],
        "rca_removed_points": [
            tuple(mesh.vertices[i]) for i in range(n_vertices) if new_labels[i] == 3
        ],
        "lca_removed_points": [
            tuple(mesh.vertices[i]) for i in range(n_vertices) if new_labels[i] == 4
        ],
    }
    updated_results["aorta_points"] = [
        tuple(mesh.vertices[i]) for i in range(n_vertices) if new_labels[i] == 0
    ]

    return updated_results


def label_anomalous_region(
    centerline,
    frames,
    results: dict,
    results_key: str = "rca_points",
    debug_plot: bool = False,
) -> dict:
    """Partition a coronary region into proximal, anomalous, and distal sub-regions.

    Uses the intravascular imaging frames to determine where along the centerline
    the anomalous (intramural) segment begins and ends, then tags each mesh
    vertex accordingly.

    Parameters
    ----------
    centerline : PyCenterline
        Centerline of the coronary vessel of interest.
    frames : list of PyFrame
        Ordered list of intravascular imaging frames for the vessel.
    results : dict
        Labelled results dictionary (e.g. from :func:`label_geometry`).
        Must contain the key specified by *results_key*.
    results_key : str, optional
        Key in *results* whose point list is partitioned.
        Default is ``"rca_points"``.
    debug_plot : bool, optional
        When ``True`` opens an interactive visualisation of the three
        sub-regions.  Default is ``False``.

    Returns
    -------
    dict
        The input *results* dictionary extended with three new keys:

        * ``"proximal_points"`` - vertices proximal to the anomalous segment.
        * ``"distal_points"`` - vertices distal to the anomalous segment.
        * ``"anomalous_points"`` - vertices within the anomalous segment.
    """
    proximal_points, distal_points, anomalous_points = find_points_by_cl_region(
        centerline=centerline,
        frames=frames,
        points=results[results_key],
    )

    results["proximal_points"] = proximal_points
    results["distal_points"] = distal_points
    results["anomalous_points"] = anomalous_points

    if debug_plot:
        plot_anomalous_region(
            results=results,
            centerline=centerline,
        )

    return results


def remove_anomalous_points_from_mesh(results: dict) -> dict:
    """Remove anomalous vertices from the mesh and update all point lists.

    Deletes the vertices listed under ``"anomalous_points"`` (and any faces
    referencing them) from the mesh, remaps the remaining faces, and rebuilds
    every coordinate list in *results* to reflect the new vertex indices.

    Parameters
    ----------
    results : dict
        Dictionary produced by :func:`label_anomalous_region` containing at
        minimum the keys ``"mesh"`` and ``"anomalous_points"``.  Any of
        ``"aorta_points"``, ``"rca_points"``, ``"lca_points"``,
        ``"rca_removed_points"``, ``"lca_removed_points"``,
        ``"proximal_points"``, and ``"distal_points"`` are also updated if
        present.

    Returns
    -------
    dict
        Updated *results* dict with ``"mesh"`` replaced by the trimmed mesh
        and all coordinate lists remapped to the new vertex set. Additionally
        new entry with ``"boundary_points"``.
    """
    mesh: trimesh.Trimesh = results["mesh"]
    anomalous_points: list = results.get("anomalous_points", [])

    if not anomalous_points:
        return results

    # 1. Map coordinates -> vertex index
    coord_to_idx = {tuple(coord): i for i, coord in enumerate(mesh.vertices)}

    # 2. Collect vertex indices to remove
    remove_indices = set()
    for pt in anomalous_points:
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
    boundary_points = [tuple(mesh.vertices[i]) for i in boundary_indices]

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
    updated["anomalous_points"] = []
    updated["boundary_points"] = boundary_points

    for key in (
        "aorta_points",
        "rca_points",
        "lca_points",
        "rca_removed_points",
        "lca_removed_points",
        "proximal_points",
        "distal_points",
    ):
        if key in updated:
            updated[key] = [p for p in updated[key] if tuple(p) in new_coord_set]

    return updated


def stitch_ccta_to_intravascular(
    iv_mesh: PyGeometry,
    mesh: trimesh.Trimesh,
    results: dict,
    n_points_iv_cont: int = 100,
) -> dict:
    """
    Uses an aligned intravascular mesh and stitches it to a CCTA mesh using the following steps.
    """
    iv_mesh = iv_mesh.downsample(n_points_iv_cont)
    proximal_centroid = iv_mesh.frames[0].centroid
    distal_centroid = iv_mesh.frames[-1].centroid
    proximal_points = iv_mesh.frames[0].lumen.points
    distal_points = iv_mesh.frames[-1].lumen.points

    prox_boundary_pts, dist_boundary_pts = _prepare_prox_dist_boundary_pts(
        mesh,
        results,
        proximal_centroid,
        distal_centroid,
    )
    prox_point_step = len(proximal_points) // len(prox_boundary_pts)
    dist_point_step = len(distal_points) // len(dist_boundary_pts)

    # Adjust start point
    prox_boundary_pts, dist_boundary_pts = _adjust_start_point(
        prox_boundary_pts,
        dist_boundary_pts,
        proximal_points,
        distal_points,
    )

    # Check direction of boundary points by keeping only
    prox_boundary_pts, dist_boundary_pts = _check_ring_direction(
        prox_boundary_pts,
        dist_boundary_pts,
        proximal_points,
        distal_points,
        prox_point_step,
        dist_point_step,
    )

    # Compute the vessel axis so each patch can be flipped to face outward.
    # frames[0] is the "proximal" end, frames[-1] the "distal" end.
    # The outward direction for the proximal patch points away from the mesh
    # interior (toward frames[0]), and vice-versa for the distal patch.
    prox_c = np.array(iv_mesh.frames[0].centroid)
    dist_c = np.array(iv_mesh.frames[-1].centroid)
    prox_outward = prox_c - dist_c   # points toward the proximal end
    dist_outward = dist_c - prox_c   # points toward the distal end

    # Step 3: stitch each boundary ring to its IV ring
    prox_patch = _stitch_boundary_ring(prox_boundary_pts, proximal_points, prox_point_step, prox_outward)
    dist_patch = _stitch_boundary_ring(dist_boundary_pts, distal_points, dist_point_step, dist_outward)
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
    results["mesh"] = mesh
    
    return results 


def _prepare_prox_dist_boundary_pts(
        mesh: trimesh.Trimesh, 
        results: dict, 
        prox_centroid: tuple[float, float, float],
        dist_centroid: tuple[float, float, float],
) -> tuple[list, list]:
    proximal_boundary_pts = []
    distal_boundary_pts = []
    print(f"Number of boundary points: {len(results["boundary_points"])}")
    for pt in results["boundary_points"]:
        distance_prox = np.linalg.norm(np.array(prox_centroid) - np.array(pt))
        distance_dist = np.linalg.norm(np.array(dist_centroid) - np.array(pt))
        if distance_prox <= distance_dist:
            proximal_boundary_pts.append(pt)
        else:
            distal_boundary_pts.append(pt)

    prox_boundary_pts_ord = _order_points_list(mesh, proximal_boundary_pts)
    dist_boundary_pts_ord = _order_points_list(mesh, distal_boundary_pts)

    return prox_boundary_pts_ord, dist_boundary_pts_ord


def _order_points_list(mesh: trimesh.Trimesh, points: list) -> list:
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


def _adjust_start_point(
    prox_boundary_pts: list,
    dist_boundary_pts: list,
    proximal_points,
    distal_points,
) -> tuple[list, list]:
    """Rotate each boundary ring so the point nearest to iv point 0 is first."""

    def _rotate_to_nearest(boundary_pts: list, iv_pt) -> list:
        iv_arr = np.array([iv_pt.x, iv_pt.y, iv_pt.z])
        dists = [np.linalg.norm(np.array(pt) - iv_arr) for pt in boundary_pts]
        start_idx = int(np.argmin(dists))
        return boundary_pts[start_idx:] + boundary_pts[:start_idx]

    prox_boundary_pts = _rotate_to_nearest(prox_boundary_pts, proximal_points[0])
    dist_boundary_pts = _rotate_to_nearest(dist_boundary_pts, distal_points[0])
    return prox_boundary_pts, dist_boundary_pts


def _check_ring_direction(
    prox_boundary_pts: list,
    dist_boundary_pts: list,
    proximal_points,
    distal_points,
    prox_point_step: int,
    dist_point_step: int,
) -> tuple[list, list]:
    """Ensure boundary rings wind in the same direction as the IV point rings.

    Subsamples the IV points (every *step*-th, starting at 0) to match the
    number of boundary points, then computes total point-wise distance for
    the ring as-is and with the order reversed (keeping index 0 fixed).
    The order with the smaller total distance is returned.
    """

    def _total_dist(boundary_pts: list, iv_subsampled) -> float:
        n = min(len(boundary_pts), len(iv_subsampled))
        return sum(
            np.linalg.norm(
                np.array(boundary_pts[i])
                - np.array([iv_subsampled[i].x, iv_subsampled[i].y, iv_subsampled[i].z])
            )
            for i in range(n)
        )

    def _check_direction(boundary_pts: list, iv_pts, step: int) -> list:
        iv_sub = iv_pts[0::step][: len(boundary_pts)]
        reversed_pts = [boundary_pts[0]] + list(reversed(boundary_pts[1:]))
        if _total_dist(reversed_pts, iv_sub) < _total_dist(boundary_pts, iv_sub):
            return reversed_pts
        return boundary_pts

    prox_boundary_pts = _check_direction(prox_boundary_pts, proximal_points, prox_point_step)
    dist_boundary_pts = _check_direction(dist_boundary_pts, distal_points, dist_point_step)
    return prox_boundary_pts, dist_boundary_pts


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

    print(f"Stitching: {len(faces)}/{n_iv} triangles created "
          f"(n_boundary={n_boundary}, n_iv={n_iv}, step={step}, remainder={remainder})")

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

    region_vertex_indices = []
    region_set = set(region_points)

    for idx, vertex in enumerate(scaled_mesh.vertices):
        if tuple(vertex) in region_set:
            region_vertex_indices.append(idx)

    region_vertex_indices = np.array(region_vertex_indices)

    if len(region_vertex_indices) == 0:
        print("Warning: No vertices found for scaling region")
        return scaled_mesh

    print(
        f"Scaling {len(region_vertex_indices)} vertices using centerline-based morphing"
    )
    print(f"Diameter adjustment: {diameter_adjustment_mm} mm")

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
    debug_plot: bool = True,
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
    debug_plot : bool, optional
        Reserved for future use; currently unused.  Default is ``True``.

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

    print("=== Finding best scaling factors ===")
    prox_scaling, dist_scaling = find_proximal_distal_scaling(
        results["anomalous_points"],
        n_section,
        n_section,
        centerline,
        frame_points_prox,
        frame_points_dist,
    )
    print(f"Best proximal scaling: {prox_scaling}")
    print(f"Best distal scaling: {dist_scaling}")

    return prox_scaling, dist_scaling


def find_aorta_scaling(
    frames,
    centerline,
    results: dict,
    debug_plot: bool = True,
) -> float:
    """Compute the optimal radial scaling factor for the aortic wall region.

    Extracts reconstructed wall points from the intravascular frames (using
    ``aortic_thickness`` and the ``"Wall"`` extras) as a reference, then calls
    the Rust ``find_aortic_scaling`` routine to determine the factor that best
    aligns the removed RCA points to those references.

    Parameters
    ----------
    frames : list of PyFrame
        Intravascular imaging frames containing ``aortic_thickness`` and
        ``extras["Wall"]`` data.
    centerline : PyCenterline
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

    print("=== Finding best scaling factor ===")
    scaling = find_aortic_scaling(
        results["rca_removed_points"],  # For now work with removed points
        reference_points,
        centerline,
    )
    print(f"Best aortic scaling: {scaling}")

    return scaling


def _extract_wall_from_frames(frames) -> list[tuple[float, float, float]]:
    """Extract reconstructed aortic wall points from intravascular imaging frames.

    Iterates over *frames* looking for those that carry ``aortic_thickness``
    data.  From each such frame, a subset of the ``"Wall"`` extra contour points
    is sampled (indices in the range ``[step, 2*step)`` where
    ``step = n_points // 8``) to represent the aortic wall.

    Parameters
    ----------
    frames : list of PyFrame
        Intravascular imaging frames.  Frames without ``aortic_thickness`` are
        skipped.

    Returns
    -------
    list of tuple
        ``(x, y, z)`` tuples of wall sample points from the last eligible frame.
        Returns ``None`` if no eligible frame is found.

    Raises
    ------
    ValueError
        If an eligible frame is missing the ``"Wall"`` extras entry or that
        entry is empty.
    """
    # since geometries always have the same number of points per frame we can take one frame
    n_points = len(frames[0].lumen.points)
    step = n_points // 8
    lower_limit = step
    upper_limit = step * 2

    reference_points = None

    # this extracts the recreated wall from aortic thickness if it exists.
    for frame in frames:
        if frame.lumen.aortic_thickness is None:
            continue
        if "Wall" not in frame.extras or frame.extras["Wall"] is None:
            raise ValueError(
                f"No Wall extras found for frame {getattr(frame, 'frame', '?')}"
            )

        walls = [frame.extras.get("Wall")]

        if not walls:
            raise ValueError(
                f"Empty Wall extras for frame {getattr(frame, 'frame', '?')}"
            )

        all_points = [
            (p.x, p.y, p.z)
            for w in walls
            for p in w.points
            if lower_limit <= p.point_index <= upper_limit
        ]

        reference_points = all_points

    return reference_points
