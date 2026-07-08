import os
from pathlib import Path
import numpy as np
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

rca_cl_raw = np.genfromtxt("./centerline_rca_short.csv", delimiter=",")
lca_cl_raw = np.genfromtxt("./centerline_lca.csv", delimiter=",")
aorta_cl_raw = np.genfromtxt("./centerline_aorta.csv", delimiter=",")
rca_cl = mm.numpy_to_centerline(rca_cl_raw)
lca_cl = mm.numpy_to_centerline(lca_cl_raw)
aorta_cl = mm.numpy_to_centerline(aorta_cl_raw)

results, (rca_cl, lca_cl, ao_cl) = mm.label_geometry(
    path_ccta_geometry="./NARCO_119.stl",
    path_centerline_aorta="./centerline_aorta.csv",
    path_centerline_rca="./centerline_rca_short.csv",
    path_centerline_lca="./centerline_lca.csv",
    bounding_sphere_radius_mm=3.0,
    n_points_intramural=100,
    anomalous_rca=True,
    anomalous_lca=False,
    control_plot=True,
)

rca_cl, lca_cl, results = mm.prepare_centerlines(
    rca_cl,
    lca_cl,
    results,
    branch_sigma=2.0,
    control_plot=True,
)

list_edges = mm.find_sharp_angles(
    lca_cl, branch_id=0, cos_threshold=0.0, control_plot=True
)
lca_cl = lca_cl.split_branch(0, list_edges[4])
lca_cl = lca_cl.merge_branches(0, 4)
lca_cl = lca_cl.check_centerline()
results = mm.label_branches(lca_cl, results, results_key="lca_points")

tree = mm.discretize_vessel_tree(
    ao_cl,
    rca_cl,
    lca_cl,
    results,
    step_size=1.0,
    n_points=100,
    b_spline=True,  # set True + tune bspline_smoothing to smooth noisy contours
    bspline_smoothing=5.0,
    control_plot=True,
)

rest, (dia_logs, sys_logs) = mm.from_file_singlepair(
    input_path="ivus_rest",
    labels=["aligned_dia", "aligned_sys"],
    output_path="output/rest",
)

ref_points = tree.rca_references[0]

rca_cl_main = rca_cl.get_branch(0)  # alignment needs single-branch CL
aligned, resampled_cl = mm.align_combined(
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
    cl_aorta=ao_cl,
    results=results,
)

aortic_wall_scaling = mm.find_aortic_wall_scaling(
    frames=aligned.geom_a.frames,
    cl_aorta=ao_cl,
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

results_final, (rca_cl_f, lca_cl_f, ao_cl_f) = mm.label_geometry(
    path_ccta_geometry="fixed_mesh.stl",
    path_centerline_aorta="../data/centerline_aorta.csv",
    path_centerline_rca="../data/centerline_rca_short.csv",
    path_centerline_lca="../data/centerline_lca.csv",
    bounding_sphere_radius_mm=3.0,
    n_points_intramural=100,
    anomalous_rca=True,
    anomalous_lca=False,
    control_plot=True,
)
