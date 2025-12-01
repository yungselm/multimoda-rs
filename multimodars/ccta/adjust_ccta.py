import numpy as np
import trimesh
from pathlib import Path
import warnings


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
    
    # # Scene 2: Points only (clearer view)
    # scene2_geoms = []
    
    # if aortic_points:
    #     scene2_geoms.append(PointCloud(aortic_array, colors=aortic_colors))
    # if rca_points:
    #     scene2_geoms.append(PointCloud(rca_array, colors=rca_colors))
    # if lca_points:
    #     scene2_geoms.append(PointCloud(lca_array, colors=lca_colors))
    # if all_removed:
    #     scene2_geoms.append(PointCloud(removed_array, colors=removed_colors))
    
    # # Add centerlines
    # if cl_rca is not None:
    #     scene2_geoms.append(PointCloud(rca_centerline_points, colors=[0, 100, 200, 255]))
    # if cl_lca is not None:
    #     scene2_geoms.append(PointCloud(lca_centerline_points, colors=[0, 150, 0, 255]))
    # if cl_aorta is not None:
    #     scene2_geoms.append(PointCloud(aorta_centerline_points, colors=[200, 200, 0, 255]))
    
    # scene2 = trimesh.Scene(scene2_geoms)
    
    # print("\nShowing Scene 2: Points only (clearer view)")
    # scene2.show()


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