import numpy as np
import trimesh
from pathlib import Path
import warnings

def read_mesh(path: Path | str) -> trimesh.base.Trimesh:
    """Load a mesh from disk and attempt lightweight repairs.

    - Accepts Path or str.
    - If a Scene is loaded, its geometries are concatenated.
    - Performs basic cleanups and attempts to fill small holes.
    - Returns a Trimesh even if not watertight (warns in that case).
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Geometry file not found: {path}")

    try:
        loaded = trimesh.load(path, force="mesh")
    except Exception as exc:
        raise RuntimeError(f"Failed to load mesh from {path}: {exc}") from exc

    if isinstance(loaded, trimesh.Scene):
        geoms = tuple(loaded.geometry.values())
        if not geoms:
            raise RuntimeError(f"No geometry found in scene loaded from {path}")
        mesh = trimesh.util.concatenate(geoms)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise TypeError(f"Unsupported object loaded from {path}: {type(loaded)}")

    # basic cleanups
    try:
        mesh.update_faces(mesh.unique_faces())
    except Exception:
        try:
            mesh.remove_duplicate_faces()
        except Exception:
            pass

    mesh.remove_unreferenced_vertices()

    try:
        mesh.update_faces(mesh.nondegenerate_faces())
    except Exception:
        try:
            mesh.remove_degenerate_faces()
        except Exception:
            pass

    mesh.fix_normals()

    # attempt to fix small holes
    try:
        trimesh.repair.fill_holes(mesh)
    except Exception:
        # best-effort; don't fail on repair errors
        warnings.warn(f"fill_holes failed for mesh from {path}", RuntimeWarning)

    if not mesh.is_watertight:
        warnings.warn(f"Mesh from {path} is not watertight after repairs", RuntimeWarning)

    return mesh


def find_faces_for_points(mesh: trimesh.Trimesh, points_found, tol: float = 1e-6):
    """
    For each point in points_found find nearest vertex on `mesh` (within tol)
    and return the list of face indices that reference any of those vertices.

    Returns:
        List[int] -- indices into mesh.faces
    """
    points_array = np.asarray(points_found, dtype=np.float64)
    if points_array.size == 0:
        return []

    found_vertex_indices = set()
    verts = mesh.vertices  # (V,3)

    # loop is OK for moderate point counts; if you have many points use a KDTree
    for p in points_array:
        distances = np.linalg.norm(verts - p, axis=1)
        closest_idx = int(np.argmin(distances))
        if distances[closest_idx] <= tol:
            found_vertex_indices.add(closest_idx)

    if not found_vertex_indices:
        return []

    # gather face indices that include any matched vertex
    face_indices = []
    for i, face in enumerate(mesh.faces):
        if (face[0] in found_vertex_indices) or (face[1] in found_vertex_indices) or (face[2] in found_vertex_indices):
            face_indices.append(i)

    return face_indices


def prepare_faces_for_rust(mesh: trimesh.Trimesh, *, points=None, face_indices=None, tol: float = 1e-6):
    """
    Convert selected mesh faces to the Rust-friendly format:
      [ ((v0x,v0y,v0z), (v1x,v1y,v1z), (v2x,v2y,v2z)), ... ]

    Args:
        mesh: trimesh.Trimesh
        points: optional list/array of points. If provided, faces are found by calling
                find_faces_for_points(mesh, points, tol=tol)
        face_indices: optional explicit list of face indices to convert
        tol: tolerance for matching points -> vertices

    Returns:
        List of face tuples (v0, v1, v2) where each v* is a (x,y,z) tuple of floats.
    """
    if face_indices is None:
        if points is not None:
            face_indices = find_faces_for_points(mesh, points, tol=tol)
        else:
            # default to all faces (backwards-compatible)
            face_indices = list(range(len(mesh.faces)))

    rust_faces = []
    for fi in face_indices:
        face = mesh.faces[fi]
        v0 = tuple(map(float, mesh.vertices[face[0]]))
        v1 = tuple(map(float, mesh.vertices[face[1]]))
        v2 = tuple(map(float, mesh.vertices[face[2]]))
        rust_faces.append((v0, v1, v2))
    return rust_faces

if __name__ == "__main__":
    from trimesh.points import PointCloud
    import multimodars as mm

    # Load ONLY the coronary mesh (mesh2) - ignore the aorta mesh for now
    mesh2 = read_mesh('data/NARCO_119.stl')  # Coronary mesh
    
    print(f"Coronary mesh: {len(mesh2.vertices)} vertices, {len(mesh2.faces)} faces")

    # Load centerlines
    cl_raw = np.genfromtxt("data/centerline_raw.csv", delimiter=",")
    cl_coronary = mm.numpy_to_centerline(cl_raw)
    cl_aorta_raw = np.genfromtxt("data/centerline_aorta.csv", delimiter=",")
    cl_aorta = mm.numpy_to_centerline(cl_aorta_raw)

    print(f"Coronary centerline: {len(cl_coronary.points)} points")
    print(f"Aorta centerline: {len(cl_aorta.points)} points")

    # Step 1: Find initial points on coronary mesh using sphere method
    points_list = [tuple(vertex) for vertex in mesh2.vertices.tolist()]
    print(f"Total coronary vertices: {len(points_list)}")
    
    points_found = mm.find_centerline_bounded_points_simple(cl_coronary, points_list, 3.0)
    print(f"Points found with sphere method: {len(points_found)}")
    print("First 10 points:", points_found[0:10])

    # Step 2: Prepare ALL aorta faces for Rust occlusion removal
    # We need the aorta mesh to detect which faces are occluding
    aorta_mesh = read_mesh('data/output/aligned/mesh_000_None.obj')  # Aorta mesh
    coronary_faces_for_rust = prepare_faces_for_rust(mesh2, points=points_found, tol=1e-6)
    print(f"Prepared {len(coronary_faces_for_rust)} aorta faces for occlusion detection")

    # Step 3: Remove occluded points using ray-triangle method
    # This should remove points that are actually on the aorta wall near the ostium
    range_coronary = 22
    
    print(f"\nApplying occlusion removal with range_coronary={range_coronary}")
    points_filtered = mm.remove_occluded_points_ray_triangle(
        centerline_coronary=cl_coronary,
        centerline_aorta=cl_aorta,
        range_coronary=range_coronary,
        points=points_found,
        faces=coronary_faces_for_rust
    )
    
    print(f"Points after occlusion removal: {len(points_filtered)}")
    print(f"Removed {len(points_found) - len(points_filtered)} points (aorta wall points)")

    # Step 4: Create visualization to compare results
    # Original points (red) - includes some aorta wall points
    pf_original = np.array(points_found, dtype=np.float64)
    color_red = np.array([255, 0, 0, 255], dtype=np.uint8)
    colors_red = np.tile(color_red, (pf_original.shape[0], 1))
    pc_original = PointCloud(pf_original, colors=colors_red)

    # Filtered points (green) - should be only coronary points
    pf_filtered = np.array(points_filtered, dtype=np.float64)
    color_green = np.array([0, 255, 0, 255], dtype=np.uint8)
    colors_green = np.tile(color_green, (pf_filtered.shape[0], 1))
    pc_filtered = PointCloud(pf_filtered, colors=colors_green)

    # Load centerline points for reference
    centerline_points = np.loadtxt("data/centerlines_narco119.csv", delimiter=",")
    pc_centerline = PointCloud(centerline_points)

    # Create scene for comparison
    print("\nCreating visualization scenes...")
    
    # Scene 1: Show coronary mesh with both point sets
    scene1 = trimesh.Scene([
        mesh2,           # Coronary mesh
        pc_original,     # Original points (red) - includes aorta wall
        pc_filtered,     # Filtered points (green) - only coronary
        pc_centerline    # Centerline for reference
    ])
    
    # Make coronary mesh semi-transparent for better point visibility
    for geom in scene1.geometry.values():
        if hasattr(geom, 'visual') and hasattr(geom, 'faces'):
            geom.visual.face_colors = [200, 200, 200, 128]  # Semi-transparent gray
    
    print("Showing Scene 1: Red = original (includes aorta), Green = filtered (coronary only)")
    scene1.show()

    # Scene 2: Show just the points for clearer comparison
    scene2 = trimesh.Scene([
        pc_original,   # Original (red)
        pc_filtered,   # Filtered (green)
        pc_centerline  # Centerline
    ])
    
    print("Showing Scene 2: Points only - Red = original, Green = filtered")
    scene2.show()

    # Analysis
    print("\n=== ANALYSIS ===")
    print(f"Original points: {len(points_found)}")
    print(f"After occlusion removal: {len(points_filtered)} ({len(points_filtered)/len(points_found)*100:.1f}% remaining)")
    
    # Check if we successfully removed the problematic ostium points
    if len(points_filtered) < len(points_found):
        print("✓ Successfully removed some points (likely aorta wall points near ostium)")
    else:
        print("⚠ No points were removed - may need to adjust parameters")

    # Final filtered points for your further processing
    final_points = points_filtered
    print(f"\nFinal filtered points: {len(final_points)}")