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

results, (rca_cl, lca_cl, ao_cl) = mm.label_geometry(
    path_ccta_geometry="./NARCO_119_noside.stl",
    path_centerline_aorta="./centerline_aorta.csv",
    path_centerline_rca="./centerline_rca_short.csv",
    path_centerline_lca="./centerline_lca.csv",
    bounding_sphere_radius_mm=3.0,
    n_points_intramural=100,
    anomalous_rca=True,
    anomalous_lca=False,
    control_plot=False,
)

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

plt.tight_layout()
plt.show()

lca_cl = lca_cl.calculate_branches(2.0)
list_edges = lca_cl.find_sharp_angles(0, 0.0)
print(list_edges)
lca_cl = lca_cl.split_branch(0, 334)
lca_cl = lca_cl.merge_branches(0, 4)

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

# for x, y, z, idx in zip(list_x, list_y, list_z, point_ids):
#     ax.text(
#         x,
#         y,
#         z,
#         str(idx),
#         fontsize=6,
#         ha="center",
#         va="center",
#     )

plt.tight_layout()
plt.show()
results = mm.label_branches(rca_cl, results)
contours = mm.discretize_vessel(rca_cl, results["rca_points_main"], 0, 1.0, 200)
print(f"Discretized {len(contours)} contours")

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

all_pts = []
for contour in contours:
    xs = [p.x for p in contour.points]
    ys = [p.y for p in contour.points]
    zs = [p.z for p in contour.points]
    # close the loop
    xs.append(xs[0])
    ys.append(ys[0])
    zs.append(zs[0])
    ax.plot(xs, ys, zs, color="steelblue", linewidth=0.5, alpha=0.6)
    all_pts.extend(zip(xs, ys, zs))
    if contour.centroid is not None:
        cx, cy, cz = contour.centroid
        ax.scatter(cx, cy, cz, color="yellow", s=8, zorder=5)

# Equal aspect ratio (real scale)
if all_pts:
    arr = np.array(all_pts)
    mid = arr.mean(axis=0)
    half = (arr.max(axis=0) - arr.min(axis=0)).max() / 2
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title(f"Discretized vessel — {len(contours)} contours")
plt.tight_layout()
plt.show()
