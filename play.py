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
rca_cl = rca_cl.check_centerline()
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
lca_cl = lca_cl.check_centerline()
list_edges = lca_cl.find_sharp_angles(0, 0.0)
print(list_edges)
lca_cl = lca_cl.split_branch(0, 473)
lca_cl = lca_cl.merge_branches(0, 4)
lca_cl = lca_cl.check_centerline()

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
contours = mm.discretize_vessel(rca_cl, results["rca_points_main"], 0, 1.0, 100)
# contours = mm.discretize_vessel(ao_cl, results['aorta_points'], 0, 1.0, 100)
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

results = mm.label_branches(lca_cl, results, results_key="lca_points")
contours = mm.discretize_vessel(lca_cl, results["lca_points_main"], 0, 1.0, 100)
print(contours[0].get_area())
print(contours[0])
print(contours[-1].get_area())
print(contours[-1])

n_rca = len(rca_cl.branch_start_indices) - 1
side_rca = [results[f"rca_points_side_{k+1}"] for k in range(n_rca)]
n_lca = len(lca_cl.branch_start_indices) - 1
side_lca = [results[f"lca_points_side_{k+1}"] for k in range(n_lca)]

# tree = mm.discretize_vessel_tree(
#     ao_cl,
#     rca_cl,
#     lca_cl,
#     results["aorta_points"],
#     results["rca_points_main"],
#     results["lca_points_main"],
#     side_rca,
#     side_lca,
# )
tree = mm.discretize_vessel_tree(
    ao_cl,
    rca_cl,
    lca_cl,
    results,
)
print(tree)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")

RCA_BRANCH_COLORS = ["#4fa3e0", "#7ec8e3", "#a8d8ea", "#b8dfed"]
LCA_BRANCH_COLORS = ["#e07f4f", "#e3a87e", "#eac0a8", "#edd0b8"]
all_tree_pts = []


def _plot_contours(contours, color, alpha=0.6, lw=0.5):
    for c in contours:
        pts = [(p.x, p.y, p.z) for p in c.points]
        if not pts:
            continue
        pts.append(pts[0])
        xs, ys, zs = zip(*pts)
        ax.plot(xs, ys, zs, color=color, linewidth=lw, alpha=alpha)
        all_tree_pts.extend(pts)


def _plot_centroids(contours):
    for c in contours:
        if c.centroid is not None:
            cx, cy, cz = c.centroid
            ax.scatter(cx, cy, cz, color="yellow", s=8, zorder=5, depthshade=False)


def _plot_refs(refs):
    for main_ref, cc_ref, clock_ref in refs:
        ax.scatter(
            *main_ref,
            color="red",
            marker="x",
            s=60,
            linewidths=1.5,
            zorder=6,
            depthshade=False,
        )
        ax.scatter(
            *cc_ref, color="orange", marker="^", s=40, zorder=6, depthshade=False
        )
        ax.scatter(
            *clock_ref, color="magenta", marker="v", s=40, zorder=6, depthshade=False
        )


_plot_contours(tree.discretized_aorta, "silver", alpha=0.4)
_plot_centroids(tree.discretized_aorta)
_plot_contours(tree.discretized_rca_main, "steelblue", alpha=0.7)
_plot_centroids(tree.discretized_rca_main)
for i, branch in enumerate(tree.rca_branches):
    _plot_contours(branch, RCA_BRANCH_COLORS[i % len(RCA_BRANCH_COLORS)])
    _plot_centroids(branch)
_plot_contours(tree.discretized_lca_main, "coral", alpha=0.7)
_plot_centroids(tree.discretized_lca_main)
for i, branch in enumerate(tree.lca_branches):
    _plot_contours(branch, LCA_BRANCH_COLORS[i % len(LCA_BRANCH_COLORS)])
    _plot_centroids(branch)
_plot_refs(tree.rca_references)
_plot_refs(tree.lca_references)

if all_tree_pts:
    arr = np.array(all_tree_pts)
    mid = arr.mean(axis=0)
    half = (arr.max(axis=0) - arr.min(axis=0)).max() / 2.0
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Discretized Vessel Tree")
plt.tight_layout()
plt.show()

print(tree.rca_references[0])
