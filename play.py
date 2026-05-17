import os
from pathlib import Path
import numpy as np
import multimodars as mm
import matplotlib

matplotlib.use("TkAgg")  # Use TkAgg backend for interactive plotting
import matplotlib.pyplot as plt

cwd = Path.cwd()
for candidate in [cwd, cwd.parent, cwd.parent.parent]:
    if (candidate / "examples" / "data").exists():
        os.chdir(candidate / "examples" / "data")
        break
    elif (candidate / "data").exists():
        os.chdir(candidate / "data")
        break
print(f"Working directory: {os.getcwd()}")

rca_cl_raw = np.genfromtxt("./centerline_rca_short.csv", delimiter=",")
lca_cl_raw = np.genfromtxt("./centerline_lca.csv", delimiter=",")
aorta_cl_raw = np.genfromtxt("./centerline_aorta.csv", delimiter=",")
rca_cl = mm.numpy_to_centerline(rca_cl_raw)
lca_cl = mm.numpy_to_centerline(lca_cl_raw)
aorta_cl = mm.numpy_to_centerline(aorta_cl_raw)

rca_cl = rca_cl.calculate_branches(2.0)
print(rca_cl)

print(rca_cl.points[0])
print(rca_cl.points[0].contour_point)

list_x = []
list_y = []
list_z = []
point_ids = []
branch_ids = []

for point in rca_cl.points:
    list_x.append(point.contour_point.x)
    list_y.append(point.contour_point.y)
    list_z.append(point.contour_point.z)
    branch_ids.append(point.branch_id)
    point_ids.append(point.contour_point.point_index)

branch_arr = np.array(branch_ids)

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
sc = ax.scatter(list_x, list_y, list_z, c=branch_arr, cmap="tab10", s=10)
fig.colorbar(sc, ax=ax, label="Branch ID")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
print(
    f"Branches: {len(rca_cl.branch_start_indices)}, start indices: {rca_cl.branch_start_indices}"
)

# for x, y, z, idx in zip(list_x, list_y, list_z, point_ids):
#     ax.text(
#         x, y, z,
#         str(idx),
#         fontsize=6,
#         ha='center',
#         va='center',
#     )

plt.tight_layout()
plt.show()

lca_cl = lca_cl.calculate_branches(2.0)

list_x = []
list_y = []
list_z = []
point_ids = []
branch_ids = []

for point in lca_cl.points:
    list_x.append(point.contour_point.x)
    list_y.append(point.contour_point.y)
    list_z.append(point.contour_point.z)
    branch_ids.append(point.branch_id)
    point_ids.append(point.contour_point.point_index)

branch_arr = np.array(branch_ids)

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
sc = ax.scatter(list_x, list_y, list_z, c=branch_arr, cmap="tab10", s=10)
fig.colorbar(sc, ax=ax, label="Branch ID")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
print(
    f"Branches: {len(lca_cl.branch_start_indices)}, start indices: {lca_cl.branch_start_indices}"
)

plt.tight_layout()
plt.show()
