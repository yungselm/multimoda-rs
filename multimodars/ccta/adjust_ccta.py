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
    
    # Find points for RCA and LCA
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
    
    # Apply occlusion removal for anomalous vessels
    rca_removed_points = []
    lca_removed_points = []
    
    if anomalous_rca:
        print("Applying occlusion removal for anomalous RCA...")
        rca_faces_for_rust = prepare_faces_for_rust(mesh, points=rca_points_found, tol=tolerance_float)
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

    # Create debug plot if requested
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

    # Return results
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
    
    # Get all mesh vertices and find aortic points
    all_vertices = mesh.vertices
    aortic_points = find_aortic_points(all_vertices, rca_points, lca_points)
    
    print(f"\n=== DEBUG PLOT STATISTICS ===")
    print(f"Total mesh vertices: {len(all_vertices)}")
    print(f"Aortic points (yellow): {len(aortic_points)}")
    print(f"RCA coronary points (blue): {len(rca_points)}")
    print(f"LCA coronary points (green): {len(lca_points)}")
    print(f"RCA reassigned points (red): {len(rca_removed_points)}")
    print(f"LCA reassigned points (red): {len(lca_removed_points)}")
    
    # Create point clouds with different colors
    scenes = []
    
    # Scene 1: Mesh with all points
    scene1_geoms = []
    
    # Add mesh (semi-transparent)
    mesh_visual = mesh.copy()
    mesh_visual.visual.face_colors = [200, 200, 200, 100]  # Semi-transparent gray
    scene1_geoms.append(mesh_visual)
    
    # Add aortic points (yellow)
    if aortic_points:
        aortic_array = np.array(aortic_points, dtype=np.float64)
        aortic_colors = np.tile([255, 255, 0, 255], (len(aortic_points), 1))  # Yellow
        scene1_geoms.append(PointCloud(aortic_array, colors=aortic_colors))
    
    # Add RCA points (blue)
    if rca_points:
        rca_array = np.array(rca_points, dtype=np.float64)
        rca_colors = np.tile([0, 0, 255, 255], (len(rca_points), 1))  # Blue
        scene1_geoms.append(PointCloud(rca_array, colors=rca_colors))
    
    # Add LCA points (green)
    if lca_points:
        lca_array = np.array(lca_points, dtype=np.float64)
        lca_colors = np.tile([0, 255, 0, 255], (len(lca_points), 1))  # Green
        scene1_geoms.append(PointCloud(lca_array, colors=lca_colors))
    
    # Add removed points (red)
    all_removed = rca_removed_points + lca_removed_points
    if all_removed:
        removed_array = np.array(all_removed, dtype=np.float64)
        removed_colors = np.tile([255, 0, 0, 255], (len(all_removed), 1))  # Red
        scene1_geoms.append(PointCloud(removed_array, colors=removed_colors))
    
    # Add centerlines if provided - FIXED: Convert PyCenterlinePoint to tuples
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


# Example usage
if __name__ == "__main__":
    results = label_geometry(
        path_ccta_geometry='data/NARCO_119.stl',
        path_centerline_aorta='data/centerline_aorta.csv',
        path_centerline_rca='data/centerline_rca.csv',
        path_centerline_lca='data/centerline_lca.csv',
        anomalous_rca=True,
        anomalous_lca=False,
        control_plot=True
    )