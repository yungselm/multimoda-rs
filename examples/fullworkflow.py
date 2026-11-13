import os
from pathlib import Path

import trimesh
from synthetic_ivus import write_synthetic_pullback

import multimodars as mm

# Pullback to use:
# - False: the example IVUS pullback from the RCA ostium, aligned with the
#   ostium triplet (tree.rca[0].ostium). An anomalous (intramural) course.
# - True: a synthetic pullback cut from the CCTA around the most proximal RCA
#   bifurcation, its reference frame in the middle, aligned with that
#   bifurcation's triplet (the "second" triplet along the RCA). A side branch
#   then leaves the anomalous segment, for debugging that case.
SYNTHETIC_PULLBACK = True

# Open a 3D view of each reconnected side branch before stitching: the cut
# branch, the intravascular wall with its hole, the bridge, the cut ring and
# the path it was moved along to the wall.
SHOW_BRANCH_RECONNECTION = True


def show_branch_reconnection(geometry, removed, tree, centerlines, cut_slices):
    """Run the side-branch reconnection on its own and show its parts.

    *centerlines* maps a vessel (``"rca"`` / ``"lca"``) to its whole
    centerline, for drawing each side branch and the branch it leaves.
    """
    import numpy as np

    from multimodars._converters import geometry_to_trimesh
    from multimodars.ccta.stitching import reconnect_side_branches
    from multimodars.ccta.stitching.side_branches import _branch_path

    # The same inputs the stitch builds internally.
    geometry = geometry.downsample(100)
    iv_mesh = geometry_to_trimesh(geometry)
    ccta, iv_wall, bridges = reconnect_side_branches(
        removed["mesh"],
        iv_mesh,
        list(geometry.frames),
        removed,
        tree,
        centerlines,
        cut_slices=cut_slices,
    )

    def coloured(mesh, rgba):
        mesh = mesh.copy()
        mesh.visual.face_colors = rgba
        return mesh

    def polyline(points, rgba, closed=True):
        pts = np.asarray(points, dtype=np.float64)
        if closed:
            pts = np.vstack([pts, pts[:1]])
        path = trimesh.load_path(pts)
        path.colors = np.tile(rgba, (len(path.entities), 1))
        return path

    parts = [
        coloured(ccta, [200, 200, 200, 90]),
        coloured(iv_wall, [90, 160, 230, 160]),
        *(coloured(b, [230, 60, 60, 255]) for b in bridges),
    ]
    for (vessel, branch_id), k in cut_slices.items():
        node = getattr(tree, vessel)[branch_id]
        contours = node.contours
        # Purple: the side branch's centerline. Dark blue: the branch it
        # leaves (its parent, the main vessel for a first-order branch).
        cl = centerlines[vessel]
        parts.append(
            polyline(
                cl.get_branch(branch_id).points_as_tuples(),
                [150, 50, 200, 255],
                closed=False,
            )
        )
        if node.parent is not None:
            parts.append(
                polyline(
                    cl.get_branch(node.parent[0]).points_as_tuples(),
                    [20, 40, 140, 255],
                    closed=False,
                )
            )
        # Green: the cut contour. Orange: contour 0. Black: the path the cut
        # ring is stepped along (the branch centerline from the cut back to
        # its first point), and the ring at the end of it.
        parts.append(polyline(contours[k].points_as_tuples(), [40, 180, 40, 255]))
        parts.append(polyline(contours[0].points_as_tuples(), [240, 150, 30, 255]))
        path = _branch_path(cl, branch_id, np.asarray(contours[k].centroid))
        parts.append(polyline(path, [0, 0, 0, 255], closed=False))
        ring = np.asarray(contours[k].points_as_tuples()) + (path[-1] - path[0])
        parts.append(polyline(ring, [0, 0, 0, 255]))
    # Cyan: the intravascular axis (frame centroids), where the walk ends.
    parts.append(
        polyline(
            [f.centroid for f in geometry.frames], [0, 200, 200, 255], closed=False
        )
    )
    print(
        f"Branch reconnection: {len(bridges)} bridge(s). Grey = CCTA, "
        "blue = IV wall, red = bridge, green = cut contour, orange = contour 0, "
        "black = walk path and the ring at its end, purple = side-branch "
        "centerline, dark blue = parent "
        "centerline, cyan = IV axis"
    )
    trimesh.Scene(parts).show()


cwd = Path.cwd()
for candidate in [cwd, cwd.parent, cwd.parent.parent]:
    if (candidate / "examples" / "data").exists():
        os.chdir(candidate / "examples" / "data")
        break
    elif (candidate / "data").exists():
        os.chdir(candidate / "data")
        break
print(f"Working directory: {os.getcwd()}")

# The aorta has no reference to orient against, so prepare it first, then use
# it as `ref_centerline` when preparing the coronaries (this also drives the
# branch-detection decision inside `prepare_centerline`: no ref -> aorta ->
# no side branches to find; a ref -> coronary -> extract branches).
aorta_cl = mm.load_centerline("./ao_cl.vtp", name="Aorta")
aorta_cl = mm.prepare_centerline(aorta_cl, spacing_mm=1.0)

rca_cl = mm.load_centerline("./rca_cl.vtp", name="RCA")
rca_cl = mm.prepare_centerline(
    rca_cl, ref_centerline=aorta_cl, spacing_mm=1.0, rm_start_mm=5.0
)

lca_cl = mm.load_centerline("./lca_cl.vtp", name="LCA")
lca_cl = mm.prepare_centerline(
    lca_cl, ref_centerline=aorta_cl, spacing_mm=1.0, rm_start_mm=5.0
)

results = mm.label_geometry(
    path_ccta_geometry="./NARCO_119.stl",
    centerline_aorta=aorta_cl,
    centerline_rca=rca_cl,
    centerline_lca=lca_cl,
    bounding_sphere_radius_mm_rca=3.5,
    bounding_sphere_radius_mm_lca=3.5,
    range_mm_takeoff_rca=60.0,
    range_mm_takeoff_lca=40.0,
    acute_takeoff_rca=True,
    acute_takeoff_lca=False,
    control_plot=True,
)

# Branches (and their ordering) already come from prepare_centerline above.
# Every RCA/LCA vertex now belongs to exactly one branch id:
# results["rca_branches"] = {0: [...], 1: [...], ...}, and everything downstream
# (discretization, anomalous labelling, remove_branch) works on these ids.
results = mm.label_branches_pair(rca_cl, lca_cl, results, control_plot=True)

tree = mm.discretize_vessel_tree(
    aorta_cl,
    rca_cl,
    lca_cl,
    results,
    step_size=1.0,
    n_points=100,
    control_plot=True,
)

print(tree)

# The pullback lies on RCA branch 0 (the main vessel). For a side-branch
# pullback use its id here, tree.rca[branch].ostium and
# rca_cl.get_branch(branch, with_parents=True).
pullback_branch = 0

if SYNTHETIC_PULLBACK:
    main = tree.rca[pullback_branch]
    # Most proximal bifurcation along the main vessel (by junction contour).
    side = min(main.bifurcations, key=lambda k: tree.rca[k].parent[1])
    ref_points = main.bifurcations[side]  # (main, clock, counter_clock)
    ivus_dir = write_synthetic_pullback(
        main.contours,
        ref_slice=tree.rca[side].parent[1],
        ref_target=ref_points[0],
        out_dir="output/ivus_synthetic",
    )
    print(f"Synthetic pullback around RCA branch {side} written to {ivus_dir}")
else:
    ivus_dir = "ivus_rest"
    ref_points = tree.rca[pullback_branch].ostium  # (main, clock, counter_clock)

rest, (dia_logs, sys_logs) = mm.from_file_singlepair(
    input_path=str(ivus_dir),
    labels=["aligned_dia", "aligned_sys"],
    output_path="output/rest",
)

rca_cl_main = rca_cl.get_branch(pullback_branch)  # alignment needs single-branch CL
aligned, spacing_mm, total_rotation_deg = mm.align_combined(
    rca_cl_main,
    rest,
    ref_points[0],  # main reference point (aortic at the ostium)
    ref_points[1],  # superior reference point
    ref_points[2],  # inferior reference point
    results["rca_points"],  # CCTA point cloud for the refinement
    angle_range_deg=30.0,
    write=True,
    watertight=False,
    output_dir="test",
    align_wall_anomalous=True,
)

# Only the pullback branch is split into proximal / anomalous / distal. Every
# other RCA branch goes to results["off_path_points"] and is neither scaled nor
# cut below.
results = mm.label_anomalous_region(
    centerline=rca_cl,
    frames=aligned.geom_a.frames,
    results=results,
    branch_id=pullback_branch,
    debug_plot=False,
)

prox_scaling, distal_scaling = mm.find_distal_and_proximal_scaling(
    frames=aligned.geom_a.frames,
    centerline=rca_cl,
    results=results,
)

print(f"Proximal scaling:    {prox_scaling:.3f} mm")
print(f"Distal scaling:      {distal_scaling:.3f} mm")

# The aortic scaling only applies to a pullback from the ostium.
aortic_scaling = 0.0
if not SYNTHETIC_PULLBACK:
    aortic_scaling = mm.find_aorta_scaling(
        frames=aligned.geom_a.frames,
        cl_aorta=aorta_cl,
        results=results,
    )
    aortic_wall_scaling = mm.find_aortic_wall_scaling(
        frames=aligned.geom_a.frames,
        cl_aorta=aorta_cl,
        results=results,
    )
    print(f"Aortic scaling:      {aortic_scaling:.3f} mm")
    print(f"Aortic wall scaling: {aortic_wall_scaling:.3f} mm")

# 1. Scale the distal segment along the RCA centerline
scaled_distal = mm.scale_region_centerline_morphing(
    mesh=results["mesh"],
    region_points=results["distal_points"],
    centerline=rca_cl,
    radius_adjustment_mm=distal_scaling,
)
results = mm.sync_results_to_mesh(results, results["mesh"], scaled_distal)

# 2. Scale the aortic region (aorta + intramural wall) along the aortic centerline
scaled_distal_aortic = mm.scale_region_centerline_morphing(
    mesh=results["mesh"],
    region_points=results["aorta_points"] + results["rca_removed_points"],
    centerline=aorta_cl,
    radius_adjustment_mm=aortic_scaling,
)
results = mm.sync_results_to_mesh(results, results["mesh"], scaled_distal_aortic)

# 3. Scale the proximal segment along the RCA centerline
scaled_proximal = mm.scale_region_centerline_morphing(
    mesh=results["mesh"],
    region_points=results["proximal_points"],
    centerline=rca_cl,
    radius_adjustment_mm=prox_scaling,
)
results = mm.sync_results_to_mesh(results, results["mesh"], scaled_proximal)

# From the ostium the intravascular vessel replaces the proximal segment too and
# is stitched to the aorta. Mid-vessel it only replaces the anomalous segment.
updated_results = mm.remove_labeled_points_from_mesh(
    results,
    (
        ["anomalous_points"]
        if SYNTHETIC_PULLBACK
        else ["anomalous_points", "proximal_points"]
    ),
    target_boundaries=2,
)

mm.plot_boundary_edges(updated_results, key="boundary_points")

# Contour each side branch leaving the anomalous segment is cut at, by
# (vessel, branch id); contour 0 for a branch not listed. The cut ring is
# then moved back along the branch centerline until it meets the
# intravascular wall, which gives the hole it is bridged to. If a cut fails
# (e.g. contour 0's plane crosses the rim left by the removal), the branch
# stays loose with a warning naming the key to set here.
BRANCH_CUT_SLICES = {("rca", side): 2} if SYNTHETIC_PULLBACK else {}

if SHOW_BRANCH_RECONNECTION and BRANCH_CUT_SLICES:
    show_branch_reconnection(
        aligned.geom_a,
        updated_results,
        tree,
        {"rca": rca_cl, "lca": lca_cl},
        BRANCH_CUT_SLICES,
    )

stitched = mm.stitch_ccta_to_intravascular(
    aligned.geom_a,
    updated_results["mesh"],
    updated_results,
    prox_start_mode="nearest_iv" if SYNTHETIC_PULLBACK else "highest_z",
    proximal_is_ostium=not SYNTHETIC_PULLBACK,
    clamp_overshoot=0.5,
    tree=tree,  # reconnects side branches leaving the anomalous segment
    centerlines={"rca": rca_cl, "lca": lca_cl},
    branch_cut_slices=BRANCH_CUT_SLICES,
)
stitched["mesh"].export("prefixed.stl")
print("Raw stitched mesh exported → prefixed_mesh.stl")

remeshed = stitched.copy()
remeshed["mesh"] = mm.fix_and_remesh_stitched_mesh(
    stitched["mesh"],
    target_edge_length_mm=0.5,
    verbose=True,
)
print(f"Watertight? {remeshed['mesh'].is_watertight}")
remeshed["mesh"].export("fixed.stl")
trimesh.smoothing.filter_taubin(remeshed["mesh"], lamb=0.6)

results_final = mm.label_geometry(
    path_ccta_geometry="fixed.stl",
    centerline_aorta=aorta_cl,
    centerline_rca=rca_cl,
    centerline_lca=lca_cl,
    bounding_sphere_radius_mm_rca=3.5,
    bounding_sphere_radius_mm_lca=3.5,
    range_mm_takeoff_rca=60.0,
    range_mm_takeoff_lca=40.0,
    acute_takeoff_rca=True,
    acute_takeoff_lca=False,
    control_plot=True,
)

results_final["mesh"].export("final.stl")

# Cut the top of the aorta flat at a slice and cap it, e.g. as a CFD outlet.
# Only the plane comes from the tree, so the remeshed and smoothed final mesh
# can be cut. The larger side (the rest of the model) stays, and the aortic
# centerline is cut on the same side, ending on the slice centre in the cap.
results_cut, aorta_cl_cut, aorta_cap = mm.cut_at_slice(
    results_final,
    tree,
    aorta_cl,
    "aorta",
    slice_idx=3,
    keep="larger",
)
print(f"Cut mesh watertight? {results_cut['mesh'].is_watertight}")
results_cut["mesh"].export("final_aorta_cut.stl")
aorta_cap.export("aorta_outlet.stl")
mm.write_centerline_vtp(aorta_cl_cut, "aorta_cl_cut.vtp")

# Cut the circumflex artery (branch 1) in half and cap it. The cut is
# chained on the aorta cut, so the result has both outlets. Branches growing
# out of the dropped half are removed from the mesh, centerline and labels.
side_branch = tree.lca[1]
results_cut, lca_cl_cut, side_cap = mm.cut_at_slice(
    results_cut,
    tree,
    lca_cl,
    "lca",
    slice_idx=len(side_branch.contours) // 2,
    branch_id=1,
    keep="larger",
)
print(f"Side branch cut watertight? {results_cut['mesh'].is_watertight}")
results_cut["mesh"].export("final_lca_side_cut.stl")
side_cap.export("lca_side_outlet.stl")
mm.write_centerline_vtp(lca_cl_cut, "lca_cl_cut.vtp")
