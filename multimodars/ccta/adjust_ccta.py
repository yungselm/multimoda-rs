import numpy as np
import trimesh
from pathlib import Path
import warnings
from typing import Optional, Tuple


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
):
    import multimodars as mm
    from multimodars.io.read_geometrical import read_mesh
    
    try:
        mesh = read_mesh(path_ccta_geometry)
        print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    except Exception as e:
        print(f"Error reading CCTA mesh from {path_ccta_geometry}: {e}")
        raise

    try:
        cl_aorta_raw = np.genfromtxt(path_centerline_aorta, delimiter=',')
        cl_aorta = mm.numpy_to_centerline(cl_aorta_raw)
        print(f"Loaded aorta centerline: {len(cl_aorta.points)} points")
    except Exception as e:
        print(f"Error reading Aorta centerline from {path_centerline_aorta}: {e}")
        raise

    try:
        cl_lca_raw = np.genfromtxt(path_centerline_lca, delimiter=',')
        cl_lca = mm.numpy_to_centerline(cl_lca_raw)
        print(f"Loaded LCA centerline: {len(cl_lca.points)} points")
    except Exception as e:
        print(f"Error reading LCA centerline from {path_centerline_lca}: {e}")
        raise

    try:
        cl_rca_raw = np.genfromtxt(path_centerline_rca, delimiter=',')
        cl_rca = mm.numpy_to_centerline(cl_rca_raw)
        print(f"Loaded RCA centerline: {len(cl_rca.points)} points")
    except Exception as e:
        print(f"Error reading RCA centerline from {path_centerline_rca}: {e}")
        raise

    points_list = [tuple(vertex) for vertex in mesh.vertices.tolist()]
    
    # Rust implementation using a rolling sphere with fixed radius
    rca_points_found = mm.find_centerline_bounded_points_simple(
        cl_rca, 
        points_list, 
        bounding_sphere_radius_mm
    )
    lca_points_found = mm.find_centerline_bounded_points_simple(
        cl_lca, 
        points_list, 
        bounding_sphere_radius_mm
    )
    
    print(f"RCA points found: {len(rca_points_found)}")
    print(f"LCA points found: {len(lca_points_found)}")
    
    rca_removed_points = []
    lca_removed_points = []
    
    if anomalous_rca:
        print("Applying occlusion removal for anomalous RCA...")
        rca_faces_for_rust = prepare_faces_for_rust(mesh, points=rca_points_found, tol=tolerance_float)
        # Rust implementation, that creates ray between aortic and coronary centerline, and
        # removes faces if 3 consecutive faces are "pierced" by the ray
        final_rca_points_found = mm.remove_occluded_points_ray_triangle(
            centerline_coronary=cl_rca,
            centerline_aorta=cl_aorta,
            range_coronary=n_points_intramural,
            points=rca_points_found,
            faces=rca_faces_for_rust
        )
        rca_removed_points = [p for p in rca_points_found if p not in final_rca_points_found]
        print(f"RCA: removed {len(rca_removed_points)} occluded points")
    else:
        final_rca_points_found = rca_points_found.copy()

    if anomalous_lca:
        print("Applying occlusion removal for anomalous LCA...")
        lca_faces_for_rust = prepare_faces_for_rust(mesh, points=lca_points_found, tol=tolerance_float)
        final_lca_points_found = mm.remove_occluded_points_ray_triangle(
            centerline_coronary=cl_lca,
            centerline_aorta=cl_aorta,
            range_coronary=n_points_intramural,
            points=lca_points_found,
            faces=lca_faces_for_rust
        )
        lca_removed_points = [p for p in lca_points_found if p not in final_lca_points_found]
        print(f"LCA: removed {len(lca_removed_points)} occluded points")
    else:
        final_lca_points_found = lca_points_found.copy()

    if control_plot:
        debug_plot(
            mesh=mesh,
            rca_points=final_rca_points_found,
            lca_points=final_lca_points_found,
            rca_removed_points=rca_removed_points,
            lca_removed_points=lca_removed_points,
            cl_rca=cl_rca,
            cl_lca=cl_lca,
            cl_aorta=cl_aorta
        )

    results = {
        'mesh': mesh,
        'aorta_points': find_aortic_points(mesh.vertices, final_rca_points_found, final_lca_points_found),
        'rca_points': final_rca_points_found,
        'lca_points': final_lca_points_found,
        'rca_removed_points': rca_removed_points,
        'lca_removed_points': lca_removed_points
    }
    
    return results


def find_aortic_points(all_vertices, rca_points, lca_points):
    """Find aortic points (points not in RCA or LCA)."""
    rca_set = set(rca_points)
    lca_set = set(lca_points)
    aortic_points = [tuple(vertex) for vertex in all_vertices 
                    if tuple(vertex) not in rca_set and tuple(vertex) not in lca_set]
    return aortic_points


def debug_plot(
    mesh: trimesh.Trimesh,
    rca_points: list,
    lca_points: list,
    rca_removed_points: list,
    lca_removed_points: list,
    cl_rca=None,
    cl_lca=None,
    cl_aorta=None
):
    """Create a debug plot showing different point classifications with colors:
    - Yellow: Aortic points (points not in RCA or LCA)
    - Blue: RCA coronary points
    - Green: LCA coronary points  
    - Red: Removed points (from occlusion removal)
    """
    from trimesh.points import PointCloud
    
    all_vertices = mesh.vertices
    aortic_points = find_aortic_points(all_vertices, rca_points, lca_points)
    
    print(f"\n=== DEBUG PLOT STATISTICS ===")
    print(f"Total mesh vertices: {len(all_vertices)}")
    print(f"Aortic points (yellow): {len(aortic_points)}")
    print(f"RCA coronary points (blue): {len(rca_points)}")
    print(f"LCA coronary points (green): {len(lca_points)}")
    print(f"RCA reassigned points (red): {len(rca_removed_points)}")
    print(f"LCA reassigned points (red): {len(lca_removed_points)}")
    
    scenes = []
    
    # Scene 1: Mesh with all points
    scene1_geoms = []
    
    # Add mesh (semi-transparent)
    mesh_visual = mesh.copy()
    mesh_visual.visual.face_colors = [200, 200, 200, 100]  # Semi-transparent gray
    scene1_geoms.append(mesh_visual)
    
    if aortic_points:
        aortic_array = np.array(aortic_points, dtype=np.float64)
        aortic_colors = np.tile([255, 255, 0, 255], (len(aortic_points), 1))  # Yellow
        scene1_geoms.append(PointCloud(aortic_array, colors=aortic_colors))
    
    if rca_points:
        rca_array = np.array(rca_points, dtype=np.float64)
        rca_colors = np.tile([0, 0, 255, 255], (len(rca_points), 1))  # Blue
        scene1_geoms.append(PointCloud(rca_array, colors=rca_colors))
    
    if lca_points:
        lca_array = np.array(lca_points, dtype=np.float64)
        lca_colors = np.tile([0, 255, 0, 255], (len(lca_points), 1))  # Green
        scene1_geoms.append(PointCloud(lca_array, colors=lca_colors))
    
    all_removed = rca_removed_points + lca_removed_points
    if all_removed:
        removed_array = np.array(all_removed, dtype=np.float64)
        removed_colors = np.tile([255, 0, 0, 255], (len(all_removed), 1))  # Red
        scene1_geoms.append(PointCloud(removed_array, colors=removed_colors))
    
    if cl_rca is not None:
        rca_centerline_points = np.array([(p.contour_point.x, p.contour_point.y, p.contour_point.z) for p in cl_rca.points], dtype=np.float64)
        scene1_geoms.append(PointCloud(rca_centerline_points, colors=[0, 100, 200, 255]))  # Dark blue
    
    if cl_lca is not None:
        lca_centerline_points = np.array([(p.contour_point.x, p.contour_point.y, p.contour_point.z) for p in cl_lca.points], dtype=np.float64)
        scene1_geoms.append(PointCloud(lca_centerline_points, colors=[0, 150, 0, 255]))  # Dark green
    
    if cl_aorta is not None:
        aorta_centerline_points = np.array([(p.contour_point.x, p.contour_point.y, p.contour_point.z) for p in cl_aorta.points], dtype=np.float64)
        scene1_geoms.append(PointCloud(aorta_centerline_points, colors=[200, 200, 0, 255]))  # Dark yellow
    
    scene1 = trimesh.Scene(scene1_geoms)
    
    print("\nShowing Scene 1: Mesh with colored points")
    print("Colors: Yellow=Aorta, Blue=RCA, Green=LCA, Red=Removed")
    scene1.show()


def prepare_faces_for_rust(mesh: trimesh.Trimesh, *, points=None, face_indices=None, tol: float = 1e-6):
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
        if (face[0] in found_vertex_indices) or (face[1] in found_vertex_indices) or (face[2] in found_vertex_indices):
            face_indices.append(i)

    return face_indices


def label_anomalous_region(
        centerline,
        frames,
        results: dict,
        results_key: str = 'rca_points',
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

    results['proximal_points'] = proximal_points
    results['distal_points'] = distal_points
    results['anomalous_points'] = anomalous_points

    if debug_plot:
        plot_anomalous_region(
            results=results,
            centerline=centerline,
        )

    return results


def plot_anomalous_region(results, centerline):
    """Plot the results."""
    import numpy as np
    import trimesh
    from trimesh.points import PointCloud
    
    print(f"\n=== ANOMALOUS REGION VISUALIZATION ===")
    
    scene_geoms = []
    
    if results['proximal_points']:
        proximal_array = np.array(results['proximal_points'])
        proximal_cloud = PointCloud(proximal_array, colors=[0, 0, 255, 255])  # Blue
        scene_geoms.append(proximal_cloud)
    
    if results['anomalous_points']:
        anomalous_array = np.array(results['anomalous_points'])
        anomalous_cloud = PointCloud(anomalous_array, colors=[255, 0, 0, 255])  # Red
        scene_geoms.append(anomalous_cloud)
    
    if results['distal_points']:
        distal_array = np.array(results['distal_points'])
        distal_cloud = PointCloud(distal_array, colors=[0, 255, 0, 255])  # Green
        scene_geoms.append(distal_cloud)

    if centerline is not None:
        centerline_points = np.array([(p.contour_point.x, p.contour_point.y, p.contour_point.z) for p in centerline.points], dtype=np.float64)
        centerline_cloud = PointCloud(centerline_points, colors=[255, 255, 0, 255])  # Yellow
        scene_geoms.append(centerline_cloud)
    
    scene = trimesh.Scene(scene_geoms)
    scene.show()


def scale_region_centerline_morphing(
    mesh: trimesh.Trimesh, 
    region_points: list, 
    centerline,
    diameter_adjustment_mm: float
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
    
    print(f"Scaling {len(region_vertex_indices)} vertices using centerline-based morphing")
    print(f"Diameter adjustment: {diameter_adjustment_mm} mm")
    
    region_vertices_list = [tuple(vertex) for vertex in scaled_mesh.vertices[region_vertex_indices]]
    adjusted_points = mm.adjust_diameter_centerline_morphing_simple(
        centerline=centerline,
        points=region_vertices_list,
        diameter_adjustment_mm=diameter_adjustment_mm
    )
    
    scaled_mesh.vertices[region_vertex_indices] = np.array(adjusted_points, dtype=np.float64)
    
    # Clear mesh cache since we modified vertices directly
    scaled_mesh.vertices.flags['WRITEABLE'] = False
    
    return scaled_mesh


def compare_centerline_scaling(
    original_mesh: trimesh.Trimesh, 
    scaled_mesh: trimesh.Trimesh, 
    region_points: list,
    centerline=None
):
    """Create a visualization comparing original vs centerline-scaled mesh."""
    from trimesh.points import PointCloud
    
    print(f"\n=== CENTERLINE SCALING COMPARISON ===")
    
    region_array = np.array(region_points, dtype=np.float64)
    region_colors = np.tile([255, 0, 0, 255], (len(region_points), 1))  # Red
    
    # Scene 1: Original mesh with region highlighted
    scene1 = trimesh.Scene([
        original_mesh,
        PointCloud(region_array, colors=region_colors)
    ])
    original_mesh.visual.face_colors = [200, 200, 200, 100]  # Semi-transparent
    
    if centerline is not None:
        centerline_points = np.array([(p.contour_point.x, p.contour_point.y, p.contour_point.z) for p in centerline.points], dtype=np.float64)
        centerline_colors = np.tile([0, 255, 255, 255], (len(centerline_points), 1))  # Cyan
        scene1.add_geometry(PointCloud(centerline_points, colors=centerline_colors))
    
    print("Showing Scene 1: Original mesh with RCA region (red) and centerline (cyan)")
    scene1.show()
    
    # Scene 2: Scaled mesh with region highlighted
    scene2 = trimesh.Scene([
        scaled_mesh,
        PointCloud(region_array, colors=region_colors)
    ])
    scaled_mesh.visual.face_colors = [200, 200, 200, 100]  # Semi-transparent
    
    if centerline is not None:
        scene2.add_geometry(PointCloud(centerline_points, colors=centerline_colors))

    print("Showing Scene 2: Scaled mesh with RCA region (red) and centerline (cyan)")
    scene2.show()
    
    # Scene 3: Side-by-side comparison
    scaled_mesh_shifted = scaled_mesh.copy()
    shift_amount = np.array([150, 0, 0])  # Adjust based on your mesh size
    scaled_mesh_shifted.apply_translation(shift_amount)
    
    scene3 = trimesh.Scene([
        original_mesh,
        scaled_mesh_shifted
    ])
    
    original_mesh.visual.face_colors = [0, 100, 200, 100]  # Blue-ish
    scaled_mesh_shifted.visual.face_colors = [200, 100, 0, 100]  # Orange-ish
    
    if centerline is not None:
        centerline_shifted = centerline_points + shift_amount
        scene3.add_geometry(PointCloud(centerline_shifted, colors=centerline_colors))
    
    print("Showing Scene 3: Side-by-side comparison (Blue=Original, Orange=Scaled)")
    scene3.show()


def find_distal_and_proximal_scaling(
    frames,
    results: dict,
    dist_range: int=3,
    prox_range: int=2,
    debug_plot: bool=False,
    percentile_lower=25,
    percentile_upper=75,
    anomalous=True,
) -> Tuple[list, list]:
    n_anomalous = len(results['anomalous_points'])
    n_frames = len(frames)
    dist_ratio = dist_range / n_frames
    prox_ratio = prox_range / n_frames
    n_distal_anomalous = int(np.ceil(n_anomalous * dist_ratio))
    n_proximal_anomalous = int(np.ceil(n_anomalous * prox_ratio))

    frame_points = [(p.x, p.y, p.z) for f in frames for p in f.lumen.points]
    frame_points_dist = [(p.x, p.y, p.z) for f in frames[-dist_range:] for p in f.lumen.points]
    frame_points_prox = [(p.x, p.y, p.z) for f in frames[0:prox_range] for p in f.lumen.points]

    # calculate overal centroid of frame_points_dist
    centroid_dist = np.mean(np.array(frame_points_dist), axis=0)
    centroid_prox = np.mean(np.array(frame_points_prox), axis=0)

    distal_anomalous = _find_points_by_centroid(results['anomalous_points'], centroid_dist, n_distal_anomalous)
    proximal_anomalous = _find_points_by_centroid(results['anomalous_points'], centroid_prox, n_proximal_anomalous)
    
    # Calculate mean distances using closest-point comparison
    if anomalous:
        mean_dist = _circular_radial_expansion_numpy(frame_points, results['anomalous_points'], percentile_lower, percentile_upper)
        mean_dist_distal = _circular_radial_expansion_numpy(frame_points_dist, distal_anomalous, percentile_lower, percentile_upper)
        mean_dist_proximal = _elliptic_radial_expansion_numpy(frame_points_prox, proximal_anomalous, percentile_lower, percentile_upper)
    else:
        mean_dist = _circular_radial_expansion_numpy(frame_points, results['anomalous_points'], percentile_lower, percentile_upper)
        mean_dist_distal = _circular_radial_expansion_numpy(frame_points_dist, distal_anomalous, percentile_lower, percentile_upper)
        mean_dist_proximal = _circular_radial_expansion_numpy(frame_points_prox, proximal_anomalous, percentile_lower, percentile_upper)

    # Print results for verification
    print(f"Total - Mean distance: {mean_dist:.4f}, "
          f"Anomalous points: {n_anomalous}, "
          f"Frame points: {len(frame_points)}")
    
    print(f"Distal - Mean distance: {mean_dist_distal:.4f}, "
          f"Anomalous points: {len(distal_anomalous)}, "
          f"Frame points: {len(frame_points_dist)}")
    
    print(f"Proximal - Mean distance: {mean_dist_proximal:.4f}, "
          f"Anomalous points: {len(proximal_anomalous)}, "
          f"Frame points: {len(frame_points_prox)}")
    
    if debug_plot:
        import trimesh
        from trimesh.points import PointCloud

        # plot anomalous points in grey and distal in blue and proximal in green
        anomalous_array = np.array(results['anomalous_points'], dtype=np.float64)
        anomalous_colors = np.tile([150, 150, 150, 255],
                                      (len(results['anomalous_points']), 1))  # Grey
        distal_array = np.array(distal_anomalous, dtype=np.float64)
        distal_colors = np.tile([0, 0, 255, 255], (
            len(distal_anomalous), 1))  # Blue
        proximal_array = np.array(proximal_anomalous, dtype=np.float64)
        proximal_colors = np.tile([0, 255, 0, 255], (
            len(proximal_anomalous), 1))  # Green
        scene = trimesh.Scene([
            PointCloud(anomalous_array, colors=anomalous_colors),
            PointCloud(distal_array, colors=distal_colors),
            PointCloud(proximal_array, colors=proximal_colors),
        ])
        print("Showing anomalous points (grey), distal (blue), proximal (green)")
        scene.show()
    
    return mean_dist_distal, mean_dist_proximal, mean_dist


def _find_points_by_centroid(
        points: list[(float, float, float)], 
        centroid: tuple[float, float, float], 
        n_points: int) -> list[(float, float, float)]:
    "find n points closest to centroid"
    points_array = np.array(points)
    centroid_array = np.array(centroid)
    distances = np.linalg.norm(points_array - centroid_array, axis=1)
    closest_indices = np.argsort(distances)[:n_points]
    closest_points = points_array[closest_indices].tolist()
    return closest_points


def _circular_radial_expansion_numpy(frame_points, anomalous_points, q_low, q_high):
    """
    Handle circular pipe sections with numpy only.
    """
    centroid = np.mean(frame_points, axis=0)
    frame_centered = frame_points - centroid
    anomalous_centered = anomalous_points - centroid
    
    # Perform PCA to find principal components
    cov_matrix = np.cov(frame_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvectors by eigenvalues (ascending)
    idx = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, idx]
    
    # The first eigenvector (smallest eigenvalue) is the normal to the plane
    normal = eigenvectors[:, 0]
    
    # Project points onto the plane (use the two largest eigenvectors)
    # Create rotation matrix to align with principal axes
    frame_2d = frame_centered @ eigenvectors[:, 1:]  # Use last two eigenvectors
    anomalous_2d = anomalous_centered @ eigenvectors[:, 1:]
    
    n = len(frame_2d)
    
    x = frame_2d[:, 0]
    y = frame_2d[:, 1]
    
    # Build linear system A * params = B
    A = np.column_stack([2*x, 2*y, np.ones(n)])
    B = x**2 + y**2
    
    # Solve for circle parameters
    try:
        params = np.linalg.lstsq(A, B, rcond=None)[0]
        a, b, c = params
        center_2d = np.array([a, b])
        radius = np.sqrt(a**2 + b**2 + c)
    except np.linalg.LinAlgError:
        # Fallback: use centroid and mean distance
        center_2d = np.mean(frame_2d, axis=0)
        distances = np.linalg.norm(frame_2d - center_2d, axis=1)
        radius = np.mean(distances)
    
    anomalous_distances = np.linalg.norm(anomalous_2d - center_2d, axis=1)
    
    # Signed radial differences (positive = inside circle, negative = outside)
    radial_differences = - (anomalous_distances - radius)
    
    return _robust_statistic_numpy(radial_differences, q_low, q_high)


def _elliptic_radial_expansion_numpy(frame_points, anomalous_points, q_low, q_high):
    """
    Handle elliptical/irregular point clouds with numpy only.
    Uses PCA-based ellipsoid fitting or convex hull approximation.
    """
    centroid = np.mean(frame_points, axis=0)
    frame_centered = frame_points - centroid
    anomalous_centered = anomalous_points - centroid
    
    # Method 1: PCA-based ellipsoid approximation
    try:
        # Compute covariance matrix
        cov_matrix = np.cov(frame_centered.T)
        
        # Get eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalue magnitude
        idx = np.argsort(eigenvalues)[::-1]  # Descending
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Normalize eigenvalues to get principal axis lengths
        # Use square roots of eigenvalues (standard deviations)
        axis_lengths = np.sqrt(eigenvalues)
        
        # Avoid division by zero
        axis_lengths = np.maximum(axis_lengths, 1e-10)
        
        # For each anomalous point, compute normalized distance
        signed_distances = []
        
        for point in anomalous_centered:
            # Project point onto principal axes
            proj = point @ eigenvectors
            
            # Compute normalized coordinates (divide by axis lengths)
            normalized = proj / axis_lengths
            
            # Compute Mahalanobis-like distance
            mahalanobis_dist = np.linalg.norm(normalized)
            
            # Compute "radius" in this normalized space for frame points
            frame_normalized = (frame_centered @ eigenvectors) / axis_lengths
            frame_radii = np.linalg.norm(frame_normalized, axis=1)
            
            # Use median frame radius as reference
            ref_radius = np.median(frame_radii)
            
            # Signed distance (positive inside, negative outside)
            signed_dist = - (mahalanobis_dist - ref_radius)
            
            scale_factor = np.mean(axis_lengths)
            signed_distances.append(signed_dist * scale_factor)
        
        signed_distances = np.array(signed_distances)
        
    except:
        # Fallback method: simple distance-based approach
        signed_distances = _simple_distance_expansion_numpy(frame_points, anomalous_points)
    
    # Compute robust statistic
    return _robust_statistic_numpy(signed_distances, q_low, q_high)


def _simple_distance_expansion_numpy(frame_points, anomalous_points):
    """
    Fallback method using distance to nearest neighbor.
    """
    n_anomalous = len(anomalous_points)
    n_frame = len(frame_points)
    
    # For memory efficiency, process in batches if needed
    batch_size = min(1000, n_anomalous)
    signed_distances = []
    
    for i in range(0, n_anomalous, batch_size):
        batch = anomalous_points[i:i+batch_size]
        
        diff = batch[:, np.newaxis, :] - frame_points[np.newaxis, :, :]
        
        distances = np.sqrt(np.sum(diff * diff, axis=2))
        min_distances = np.min(distances, axis=1)
        
        nearest_indices = np.argmin(distances, axis=1)
        nearest_points = frame_points[nearest_indices]
        
        frame_centroid = np.mean(frame_points, axis=0)
        
        for j in range(len(batch)):
            anomalous_pt = batch[j]
            nearest_pt = nearest_points[j]
            min_dist = min_distances[j]
            
            v_frame = nearest_pt - frame_centroid
            v_anomalous = anomalous_pt - frame_centroid
            
            dot_product = np.dot(v_frame, v_anomalous)
            
            if dot_product > 0:
                # Same general direction - anomalous point is likely outside
                signed_distances.append(-min_dist)
            else:
                # Opposite direction - anomalous point is likely inside
                signed_distances.append(min_dist)
    
    return np.array(signed_distances)


def _robust_statistic_numpy(values, q_low, q_high):
    """
    Compute robust statistic (IQR-constrained median) of values.
    """
    if len(values) == 0:
        return 0.0
    
    if q_low < 0.0:
        q_low = 0.0
    if q_high > 100.0:
        q_high = 100.0
    if q_low >= q_high:
        return float(np.median(values))
    
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    
    idx_low = max(0, min(n-1, int(q_low / 100.0 * n)))
    idx_high = max(0, min(n-1, int(q_high / 100.0 * n)))
    
    low_val = sorted_vals[idx_low]
    high_val = sorted_vals[idx_high]
    
    mask = (values >= low_val) & (values <= high_val)
    selected = values[mask]
    
    if len(selected) == 0:
        return float(np.median(values))
    
    return float(np.median(selected))
