from pathlib import Path
import trimesh
import numpy as np

from ..multimodars import (
    find_centerline_bounded_points_simple,
    remove_occluded_points_ray_triangle,
    clean_outlier_points,
    build_adjacency_map,
    find_points_by_cl_region,
)
from .._converters import numpy_to_centerline
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
