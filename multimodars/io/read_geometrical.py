import numpy as np
import trimesh

def read_stl_geometry():
    test_mesh = trimesh.load('data/NARCO_119.stl')
    test_mesh.show()
    # return test_mesh


if __name__ == "__main__":
    # test_mesh = trimesh.load('data/NARCO_119.stl')
    # print(test_mesh.is_watertight)

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

    # re-check
    print("watertight after fill_holes?", mesh.is_watertight)
    mesh.export('my_mesh_repaired_trimesh.obj')