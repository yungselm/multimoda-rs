import os
from pathlib import Path
import multimodars as mm
import trimesh

# load the provided example data
cwd = Path.cwd()
if (cwd / "examples" / "data").exists():
    os.chdir(cwd / "examples" / "data")
elif (cwd / "data").exists():
    os.chdir(cwd / "data")
print(f"Working directory: {os.getcwd()}")

mesh = trimesh.load("../data/aorta_test.stl")
trimesh.smoothing.filter_taubin(mesh)
print(f"Watertight? {mesh.is_watertight}")
mesh = mm.manual_hole_fill(mesh)
mesh.show()
