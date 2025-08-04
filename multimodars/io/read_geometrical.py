import numpy as np
from stl import mesh

def read_stl_geometry():
    test_mesh = mesh.Mesh.from_file("/mnt/c/Users/ansel/Downloads/NARCO200_RCA_wsc.stl")
    print(test_mesh.points)
    return test_mesh
