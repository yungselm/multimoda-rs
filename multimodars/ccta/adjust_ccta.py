import numpy as np
import trimesh
from pathlib import Path
import warnings
from typing import Optional, Tuple, List


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
) -> Tuple[dict, Tuple[any, any, any]]:
    import multimodars as mm
    from multimodars.io.read_geometrical import read_mesh

    try:
        mesh = read_mesh(path_ccta_geometry)
        print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    except Exception as e:
        print(f"Error reading CCTA mesh from {path_ccta_geometry}: {e}")
        raise

    try:
        cl_aorta_raw = np.genfromtxt(path_centerline_aorta, delimiter=",")
        cl_aorta = mm.numpy_to_centerline(cl_aorta_raw)
        print(f"Loaded aorta centerline: {len(cl_aorta.points)} points")
    except Exception as e:
        print(f"Error reading Aorta centerline from {path_centerline_aorta}: {e}")
        raise

    try:
        cl_lca_raw = np.genfromtxt(path_centerline_lca, delimiter=",")
        cl_lca = mm.numpy_to_centerline(cl_lca_raw)
        print(f"Loaded LCA centerline: {len(cl_lca.points)} points")
    except Exception as e:
        print(f"Error reading LCA centerline from {path_centerline_lca}: {e}")
        raise

    try:
        cl_rca_raw = np.genfromtxt(path_centerline_rca, delimiter=",")
        cl_rca = mm.numpy_to_centerline(cl_rca_raw)
        print(f"Loaded RCA centerline: {len(cl_rca.points)} points")
    except Exception as e:
        print(f"Error reading RCA centerline from {path_centerline_rca}: {e}")
        raise

    points_list = [tuple(vertex) for vertex in mesh.vertices.tolist()]

    # Rust implementation using a rolling sphere with fixed radius
    rca_points_found = mm.find_centerline_bounded_points_simple(
        cl_rca, points_list, bounding_sphere_radius_mm
    )
    lca_points_found = mm.find_centerline_bounded_points_simple(
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
        final_rca_points_found = mm.remove_occluded_points_ray_triangle(
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
        final_lca_points_found = mm.remove_occluded_points_ray_triangle(
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
    final_lca_points, final_aortic_points = mm.clean_outlier_points(
        final_lca_points_found, aortic_points, 2.0, 0.4
    )  # based on patient data, only precleaning anyways, rest done by final_reclassification
    final_rca_points, final_aortic_points = mm.clean_outlier_points(
        final_rca_points_found, final_aortic_points, 2.0, 0.4
    )
    aortic_points = _find_aortic_points(
        mesh.vertices, final_rca_points, final_lca_points
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
        _labeled_geometry_plot(
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
    """
    Convert selected mesh faces to the Rust-friendly format.
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
    """
    For each point in points_found find nearest vertex on `mesh` (within tol)
    and return the list of face indices that reference any of those vertices.
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
    """Find aortic points (points not in RCA or LCA)."""
    rca_set = set(rca_points)
    lca_set = set(lca_points)
    aortic_points = [
        tuple(vertex)
        for vertex in all_vertices
        if tuple(vertex) not in rca_set and tuple(vertex) not in lca_set
    ]
    return aortic_points


def _final_reclassification(results: dict) -> dict:
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
    import multimodars as mm

    adj_map = mm.build_adjacency_map(mesh.faces.tolist())

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


def _labeled_geometry_plot(
    mesh: trimesh.Trimesh,
    rca_points: list,
    lca_points: list,
    rca_removed_points: list,
    lca_removed_points: list,
    cl_rca=None,
    cl_lca=None,
    cl_aorta=None,
):
    """Create a debug plot showing different point classifications with colors:
    - Yellow: Aortic points (points not in RCA or LCA)
    - Blue: RCA coronary points
    - Green: LCA coronary points
    - Red: Removed points (from occlusion removal)
    """
    from trimesh.points import PointCloud

    all_vertices = mesh.vertices
    aortic_points = _find_aortic_points(all_vertices, rca_points, lca_points)

    print(f"\n=== DEBUG PLOT STATISTICS ===")
    print(f"Total mesh vertices: {len(all_vertices)}")
    print(f"Aortic points (yellow): {len(aortic_points)}")
    print(f"RCA coronary points (blue): {len(rca_points)}")
    print(f"LCA coronary points (green): {len(lca_points)}")
    print(f"RCA reassigned points (red): {len(rca_removed_points)}")
    print(f"LCA reassigned points (red): {len(lca_removed_points)}")

    scene_geoms = []

    # Add mesh (semi-transparent)
    mesh_visual = mesh.copy()
    mesh_visual.visual.face_colors = [200, 200, 200, 100]  # Semi-transparent gray
    scene_geoms.append(mesh_visual)

    if aortic_points:
        aortic_array = np.array(aortic_points, dtype=np.float64)
        aortic_colors = np.tile([255, 255, 0, 255], (len(aortic_points), 1))  # Yellow
        scene_geoms.append(PointCloud(aortic_array, colors=aortic_colors))

    if rca_points:
        rca_array = np.array(rca_points, dtype=np.float64)
        rca_colors = np.tile([0, 0, 255, 255], (len(rca_points), 1))  # Blue
        scene_geoms.append(PointCloud(rca_array, colors=rca_colors))

    if lca_points:
        lca_array = np.array(lca_points, dtype=np.float64)
        lca_colors = np.tile([0, 255, 0, 255], (len(lca_points), 1))  # Green
        scene_geoms.append(PointCloud(lca_array, colors=lca_colors))

    all_removed = rca_removed_points + lca_removed_points
    if all_removed:
        removed_array = np.array(all_removed, dtype=np.float64)
        removed_colors = np.tile([255, 0, 0, 255], (len(all_removed), 1))  # Red
        scene_geoms.append(PointCloud(removed_array, colors=removed_colors))

    if cl_rca is not None:
        rca_centerline_points = np.array(
            [
                (p.contour_point.x, p.contour_point.y, p.contour_point.z)
                for p in cl_rca.points
            ],
            dtype=np.float64,
        )
        scene_geoms.append(
            PointCloud(rca_centerline_points, colors=[0, 100, 200, 255])
        )  # Dark blue

    if cl_lca is not None:
        lca_centerline_points = np.array(
            [
                (p.contour_point.x, p.contour_point.y, p.contour_point.z)
                for p in cl_lca.points
            ],
            dtype=np.float64,
        )
        scene_geoms.append(
            PointCloud(lca_centerline_points, colors=[0, 150, 0, 255])
        )  # Dark green

    if cl_aorta is not None:
        aorta_centerline_points = np.array(
            [
                (p.contour_point.x, p.contour_point.y, p.contour_point.z)
                for p in cl_aorta.points
            ],
            dtype=np.float64,
        )
        scene_geoms.append(
            PointCloud(aorta_centerline_points, colors=[200, 200, 0, 255])
        )  # Dark yellow

    scene1 = trimesh.Scene(scene_geoms)

    print("\nShowing Scene 1: Mesh with colored points")
    print("Colors: Yellow=Aorta, Blue=RCA, Green=LCA, Red=Removed")
    scene1.show()


def label_anomalous_region(
    centerline,
    frames,
    results: dict,
    results_key: str = "rca_points",
    debug_plot: bool = False,
) -> dict:
    import multimodars as mm
    import numpy as np
    import trimesh

    proximal_points, distal_points, anomalous_points = mm.find_points_by_cl_region(
        centerline=centerline,
        frames=frames,
        points=results[results_key],
    )

    results["proximal_points"] = proximal_points
    results["distal_points"] = distal_points
    results["anomalous_points"] = anomalous_points

    if debug_plot:
        _plot_anomalous_region(
            results=results,
            centerline=centerline,
        )

    return results


def _plot_anomalous_region(results, centerline):
    """Plot the results."""
    import numpy as np
    import trimesh
    from trimesh.points import PointCloud

    print(f"\n=== ANOMALOUS REGION VISUALIZATION ===")

    scene_geoms = []

    if results["proximal_points"]:
        proximal_array = np.array(results["proximal_points"])
        proximal_cloud = PointCloud(proximal_array, colors=[0, 0, 255, 255])  # Blue
        scene_geoms.append(proximal_cloud)

    if results["anomalous_points"]:
        anomalous_array = np.array(results["anomalous_points"])
        anomalous_cloud = PointCloud(anomalous_array, colors=[255, 0, 0, 255])  # Red
        scene_geoms.append(anomalous_cloud)

    if results["distal_points"]:
        distal_array = np.array(results["distal_points"])
        distal_cloud = PointCloud(distal_array, colors=[0, 255, 0, 255])  # Green
        scene_geoms.append(distal_cloud)

    if centerline is not None:
        centerline_points = np.array(
            [
                (p.contour_point.x, p.contour_point.y, p.contour_point.z)
                for p in centerline.points
            ],
            dtype=np.float64,
        )
        centerline_cloud = PointCloud(
            centerline_points, colors=[255, 255, 0, 255]
        )  # Yellow
        scene_geoms.append(centerline_cloud)

    scene = trimesh.Scene(scene_geoms)
    scene.show()


def scale_region_centerline_morphing(
    mesh: trimesh.Trimesh,
    region_points: list,
    centerline,
    diameter_adjustment_mm: float,
) -> trimesh.Trimesh:
    """
    Scale a specific region of the mesh using centerline-based radial morphing.

    Args:
        mesh: The original mesh
        region_points: List of points defining the region to scale
        centerline: PyCenterline object for the region
        diameter_adjustment_mm: Amount to adjust diameter (positive to expand, negative to contract)

    Returns:
        A new mesh with the region scaled using centerline-based morphing
    """
    import multimodars as mm

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
    adjusted_points = mm.adjust_diameter_centerline_morphing_simple(
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


def compare_centerline_scaling(
    original_mesh: trimesh.Trimesh,
    scaled_mesh: trimesh.Trimesh,
    region_points: list,
    centerline=None,
):
    """Create a visualization comparing original vs centerline-scaled mesh."""
    from trimesh.points import PointCloud

    print(f"\n=== CENTERLINE SCALING COMPARISON ===")

    region_array = np.array(region_points, dtype=np.float64)
    region_colors = np.tile([255, 0, 0, 255], (len(region_points), 1))  # Red

    # Scene 1: Original mesh with region highlighted
    scene1 = trimesh.Scene(
        [original_mesh, PointCloud(region_array, colors=region_colors)]
    )
    original_mesh.visual.face_colors = [200, 200, 200, 100]  # Semi-transparent

    if centerline is not None:
        centerline_points = np.array(
            [
                (p.contour_point.x, p.contour_point.y, p.contour_point.z)
                for p in centerline.points
            ],
            dtype=np.float64,
        )
        centerline_colors = np.tile(
            [0, 255, 255, 255], (len(centerline_points), 1)
        )  # Cyan
        scene1.add_geometry(PointCloud(centerline_points, colors=centerline_colors))

    print("Showing Scene 1: Original mesh with RCA region (red) and centerline (cyan)")
    scene1.show()

    # Scene 2: Scaled mesh with region highlighted
    scene2 = trimesh.Scene(
        [scaled_mesh, PointCloud(region_array, colors=region_colors)]
    )
    scaled_mesh.visual.face_colors = [200, 200, 200, 100]  # Semi-transparent

    if centerline is not None:
        scene2.add_geometry(PointCloud(centerline_points, colors=centerline_colors))

    print("Showing Scene 2: Scaled mesh with RCA region (red) and centerline (cyan)")
    scene2.show()

    # Scene 3: Side-by-side comparison
    scaled_mesh_shifted = scaled_mesh.copy()
    shift_amount = np.array([150, 0, 0])  # Adjust based on your mesh size
    scaled_mesh_shifted.apply_translation(shift_amount)

    scene3 = trimesh.Scene([original_mesh, scaled_mesh_shifted])

    original_mesh.visual.face_colors = [0, 100, 200, 100]  # Blue-ish
    scaled_mesh_shifted.visual.face_colors = [200, 100, 0, 100]  # Orange-ish

    if centerline is not None:
        centerline_shifted = centerline_points + shift_amount
        scene3.add_geometry(PointCloud(centerline_shifted, colors=centerline_colors))

    print("Showing Scene 3: Side-by-side comparison (Blue=Original, Orange=Scaled)")
    scene3.show()


def find_distal_and_proximal_scaling(
    frames,
    centerline,
    results: dict,
    dist_range: int = 3,
    prox_range: int = 2,
    debug_plot: bool = True,
) -> Tuple[float, float]:
    import multimodars as mm

    frame_points_dist = [
        (p.x, p.y, p.z) for f in frames[-dist_range:] for p in f.lumen.points
    ]
    frame_points_prox = [
        (p.x, p.y, p.z) for f in frames[0:prox_range] for p in f.lumen.points
    ]
    n_anomalous_points = len(results["anomalous_points"])
    n_section: int = int(np.ceil(0.25 * n_anomalous_points))

    print("=== Finding best scaling factors ===")
    prox_scaling, dist_scaling = mm.find_proximal_distal_scaling(
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
    import multimodars as mm

    reference_points = _extract_wall_from_frames(frames)

    print("=== Finding best scaling factor ===")
    scaling = mm.find_aortic_scaling(
        results["rca_removed_points"],  # For now work with removed points
        reference_points,
        centerline,
    )
    print(f"Best aortic scaling: {scaling}")

    return scaling


def _extract_wall_from_frames(frames) -> List[Tuple[float, float, float]]:
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
