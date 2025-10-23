import numpy as np
import trimesh

def read_stl_geometry():
    test_mesh = trimesh.load('data/NARCO_119.stl')
    test_mesh.show()
    # return test_mesh


if __name__ == "__main__":
    # import pyvista as pv
    # import numpy as np
    # vtk_path = 'data/center_smoothed.vtk'

    # mesh = pv.read(vtk_path)
    # points = mesh.points

    # np.savetxt("data/centerline_narco119.csv", points, delimiter=",", fmt="%.6f")

    from trimesh.points import PointCloud
    mesh = trimesh.load('output/rest/mesh_000_rest.obj')
    print("watertight?", mesh.is_watertight)

    # basic cleanups
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_faces()
    mesh.fix_normals()

    # check broken faces
    broken = trimesh.repair.broken_faces(mesh)
    print("broken faces indices:", broken)

    # attempt to fill small holes
    trimesh.repair.fill_holes(mesh)

    # # re-check
    # print("watertight after fill_holes?", mesh.is_watertight)
    # mesh.export('my_mesh_repaired_trimesh.obj')

    mesh2 = trimesh.load('data/NARCO_119.stl')
    points = np.loadtxt("data/centerline_narco119.csv", delimiter=",")
    pc = PointCloud(points)

    scene = trimesh.Scene([mesh, mesh2])
    scene.add_geometry(pc)
    scene.show()