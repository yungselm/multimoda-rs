import os
from pathlib import Path
import numpy as np
import multimodars as mm
import trimesh

# load the provided example data
cwd = Path.cwd()
if (cwd / "examples" / "data").exists():
    os.chdir(cwd / "examples" / "data")
elif (cwd / "data").exists():
    os.chdir(cwd / "data")
print(f"Working directory: {os.getcwd()}")

rest, (dia_logs, sys_logs) = mm.from_file_singlepair(
    input_path="ivus_rest",
    labels=["aligned_dia", "aligned_sys"],
    output_path="output/rest",
)

rca_cl_raw = np.genfromtxt("../data/centerline_rca_short.csv", delimiter=",")
lca_cl_raw = np.genfromtxt("../data/centerline_lca.csv", delimiter=",")
aorta_cl_raw = np.genfromtxt("../data/centerline_aorta.csv", delimiter=",")
rca_cl = mm.numpy_to_centerline(rca_cl_raw)
lca_cl = mm.numpy_to_centerline(lca_cl_raw)
aorta_cl = mm.numpy_to_centerline(aorta_cl_raw)

results, (rca_cl, lca_cl, ao_cl) = mm.label_geometry(
    path_ccta_geometry="../data/NARCO_119.stl",
    path_centerline_aorta="../data/centerline_aorta.csv",
    path_centerline_rca="../data/centerline_rca_short.csv",
    path_centerline_lca="../data/centerline_lca.csv",
    bounding_sphere_radius_mm=3.0,
    n_points_intramural=100,
    anomalous_rca=True,
    anomalous_lca=False,
    control_plot=False,
)

aligned, resampled_cl = mm.align_combined(
    rca_cl,
    rest,
    (12.2605, -201.3643, 1751.0554),
    (11.7567, -202.1920, 1754.7975),
    (15.6605, -202.1920, 1749.9655),
    results["rca_points"],
    angle_range_deg=10.0,
    write=True,
    watertight=False,
    output_dir="test",
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

# no just scale any desired part of the geometry
scaled_distal = mm.scale_region_centerline_morphing(
    mesh=results["mesh"],
    region_points=results["distal_points"],
    centerline=rca_cl,
    diameter_adjustment_mm=distal_scaling,
)
results = mm.sync_results_to_mesh(results, results["mesh"], scaled_distal)

scaled_distal_aortic = mm.scale_region_centerline_morphing(
    mesh=results["mesh"],
    region_points=results["aorta_points"] + results["rca_removed_points"],
    centerline=aorta_cl,
    diameter_adjustment_mm=aortic_scaling,
)
results = mm.sync_results_to_mesh(results, results["mesh"], scaled_distal_aortic)

scaled_proximal = mm.scale_region_centerline_morphing(
    mesh=results["mesh"],
    region_points=results["proximal_points"],
    centerline=rca_cl,
    diameter_adjustment_mm=prox_scaling,
)
results = mm.sync_results_to_mesh(results, results["mesh"], scaled_proximal)

# updated_results = mm.remove_labeled_points_from_mesh(results)
updated_results = mm.remove_labeled_points_from_mesh(
    results, ["anomalous_points", "proximal_points"]
)

stitched = mm.stitch_ccta_to_intravascular(
    aligned.geom_a,
    updated_results["mesh"],
    updated_results,
    prox_start_mode="highest_z",
)
stitched["mesh"].export("prefixed_mesh.stl")
remeshed = stitched.copy()

remeshed["mesh"] = mm.fix_and_remesh_stitched_mesh(
    stitched["mesh"], target_edge_length_mm=0.5, verbose=True
)
print(f"Watertight? {remeshed['mesh'].is_watertight}")
trimesh.smoothing.filter_taubin(remeshed["mesh"], lamb=0.6)
remeshed["mesh"].export("fixed_mesh.stl")

boundary_pts = np.array(remeshed["prox_boundary_points"], dtype=np.float64)
sphere_meshes = []
for i, pt in enumerate(boundary_pts):
    t = i / max(len(boundary_pts) - 1, 1)
    color = [int(255 * t), 0, int(255 * (1 - t)), 200]
    s = trimesh.creation.icosphere(radius=0.1).apply_translation(pt)
    s.visual.face_colors = color
    sphere_meshes.append(s)

# Visualize iv_mesh frame-0 lumen points as the stitching code sees them:
# downsampled + sorted by highest-Z (same transforms applied inside stitch_ccta_to_intravascular)
iv_viz = aligned.geom_a.downsample(100).sort_frame_points()
iv_pts = iv_viz.frames[0].lumen.points
n_iv = len(iv_pts)
for pt in iv_pts:
    t = pt.point_index / max(n_iv - 1, 1)
    color = [int(255 * t), 0, int(255 * (1 - t)), 220]
    s = trimesh.creation.icosphere(radius=0.15).apply_translation([pt.x, pt.y, pt.z])
    s.visual.face_colors = color
    sphere_meshes.append(s)

spheres = trimesh.util.concatenate(sphere_meshes)
scene = trimesh.Scene([remeshed["mesh"], spheres])
scene.show()

mm.plot_results_key(stitched)
aorta_mesh = mm.keep_labeled_points_from_mesh(stitched, "aorta_points")
mm.plot_results_key(
    stitched,
    False,
    True,
)
mm.export_section_stl(stitched, "all")
mm.export_section_stl(stitched, "aorta")
mm.export_section_stl(stitched, "lca")
mm.export_section_stl(stitched, "rca")

results, (rca_cl, lca_cl, ao_cl) = mm.label_geometry(
    path_ccta_geometry="../data/fixed_mesh.stl",
    path_centerline_aorta="../data/centerline_aorta.csv",
    path_centerline_rca="../data/centerline_rca_short.csv",
    path_centerline_lca="../data/centerline_lca.csv",
    bounding_sphere_radius_mm=3.0,
    n_points_intramural=100,
    anomalous_rca=True,
    anomalous_lca=False,
    control_plot=True,
)

mm.export_section_stl(results, "all")
mm.export_section_stl(results, "aorta")
mm.export_section_stl(results, "lca")
mm.export_section_stl(results, "rca")

# test = mm.create_wall_mesh(
#     frames = None,
#     cl_aorta=aorta_cl,
#     cl_rca=rca_cl,
#     cl_lca=lca_cl,
#     results=results,
#     aortic_scaling=aortic_wall_scaling,
# )

# test['mesh'] = mm.fix_and_remesh_stitched_mesh(test['mesh'], target_edge_length_mm=0.5, verbose=True)
# test['mesh'].export("all_wall.stl")
