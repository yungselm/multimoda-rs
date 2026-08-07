import os
from pathlib import Path
import multimodars as mm
import trimesh

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
    bounding_sphere_radius_mm_rca=3.0,
    bounding_sphere_radius_mm_lca=3.0,
    range_mm_takeoff_rca=60.0,  # mm, was a point count before
    range_mm_takeoff_lca=40.0,  # mm, was a point count before
    acute_takeoff_rca=True,
    acute_takeoff_lca=False,
    control_plot=True,
)

# Branches (and their ordering) already come from prepare_centerline above.
results = mm.label_branches_pair(rca_cl, lca_cl, results)

tree = mm.discretize_vessel_tree(
    aorta_cl,
    rca_cl,
    lca_cl,
    results,
    step_size=1.0,
    n_points=100,
    b_spline=True,  # set True + tune bspline_smoothing to smooth noisy contours
    bspline_smoothing=5.0,
    control_plot=True,
)

print(tree)

rest, (dia_logs, sys_logs) = mm.from_file_singlepair(
    input_path="ivus_rest",
    labels=["aligned_dia", "aligned_sys"],
    output_path="output/rest",
)

ref_points = tree.rca_references[0]

rca_cl_main = rca_cl.get_branch(0)  # alignment needs single-branch CL
aligned, spacing_mm, total_rotation_deg = mm.align_combined(
    rca_cl_main,
    rest,
    ref_points[0],  # aortic reference point
    ref_points[1],  # superior reference point
    ref_points[2],  # inferior reference point
    results["rca_points"],  # CCTA point cloud for Hausdorff refinement
    angle_range_deg=30.0,
    write=True,
    watertight=False,
    output_dir="test",
    align_wall_anomalous=True,
)

# Resample the aorta to the same spacing align_combined derived from the frames,
# instead of re-deriving it — keeps the two centerlines' point density consistent
# for the scaling steps below.
aorta_cl = aorta_cl.resample(spacing_mm)
rca_cl = rca_cl.resample(spacing_mm)
lca_cl = lca_cl.resample(spacing_mm)

results = mm.label_anomalous_region(
    centerline=rca_cl,
    frames=aligned.geom_a.frames,
    results=results,
    results_key="rca_points",
    debug_plot=False,
)

prox_scaling, distal_scaling = mm.find_distal_and_proximal_scaling(
    frames=aligned.geom_a.frames,
    centerline=rca_cl,
    results=results,
)

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

print(f"Proximal scaling:    {prox_scaling:.3f} mm")
print(f"Distal scaling:      {distal_scaling:.3f} mm")
print(f"Aortic scaling:      {aortic_scaling:.3f} mm")
print(f"Aortic wall scaling: {aortic_wall_scaling:.3f} mm")

# 1. Scale the distal segment along the RCA centerline
scaled_distal = mm.scale_region_centerline_morphing(
    mesh=results["mesh"],
    region_points=results["distal_points"],
    centerline=rca_cl,
    diameter_adjustment_mm=distal_scaling,
)
results = mm.sync_results_to_mesh(results, results["mesh"], scaled_distal)

# 2. Scale the aortic region (aorta + intramural wall) along the aortic centerline
scaled_distal_aortic = mm.scale_region_centerline_morphing(
    mesh=results["mesh"],
    region_points=results["aorta_points"] + results["rca_removed_points"],
    centerline=aorta_cl,
    diameter_adjustment_mm=aortic_scaling,
)
results = mm.sync_results_to_mesh(results, results["mesh"], scaled_distal_aortic)

# 3. Scale the proximal segment along the RCA centerline
scaled_proximal = mm.scale_region_centerline_morphing(
    mesh=results["mesh"],
    region_points=results["proximal_points"],
    centerline=rca_cl,
    diameter_adjustment_mm=prox_scaling,
)
results = mm.sync_results_to_mesh(results, results["mesh"], scaled_proximal)

updated_results = mm.remove_labeled_points_from_mesh(
    results,
    ["anomalous_points", "proximal_points"],
)

stitched = mm.stitch_ccta_to_intravascular(
    aligned.geom_a,
    updated_results["mesh"],
    updated_results,
    prox_start_mode="highest_z",
    clamp_overshoot=0.5,
)
stitched["mesh"].export("prefixed_mesh.stl")
print("Raw stitched mesh exported → prefixed_mesh.stl")

remeshed = stitched.copy()
remeshed["mesh"] = mm.fix_and_remesh_stitched_mesh(
    stitched["mesh"],
    target_edge_length_mm=0.5,
    verbose=True,
)
print(f"Watertight? {remeshed['mesh'].is_watertight}")

trimesh.smoothing.filter_taubin(remeshed["mesh"], lamb=0.6)

results_final = mm.label_geometry(
    path_ccta_geometry="fixed_mesh.stl",
    centerline_aorta=aorta_cl,
    centerline_rca=rca_cl,
    centerline_lca=lca_cl,
    bounding_sphere_radius_mm_rca=3.0,
    bounding_sphere_radius_mm_lca=3.0,
    range_mm_takeoff_rca=60.0,  # mm, was a point count before
    range_mm_takeoff_lca=40.0,  # mm, was a point count before
    acute_takeoff_rca=True,
    acute_takeoff_lca=False,
    control_plot=True,
)
