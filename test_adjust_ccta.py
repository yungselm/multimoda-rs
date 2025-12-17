#!/usr/bin/env python3
import sys
import os

# Add the current directory to Python path so we can import multimodars
sys.path.insert(0, os.path.dirname(__file__))

from multimodars.ccta.adjust_ccta import (
    label_geometry,
    scale_region_centerline_morphing,
    compare_centerline_scaling,
    label_anomalous_region,
    find_distal_and_proximal_scaling,
)

if __name__ == "__main__":
    import numpy as np
    import trimesh
    import multimodars as mm

    results = label_geometry(
        path_ccta_geometry="data/NARCO_119.stl",
        path_centerline_aorta="data/centerline_aorta.csv",
        path_centerline_rca="data/centerline_rca.csv",
        path_centerline_lca="data/centerline_lca.csv",
        bounding_sphere_radius_mm=3.5,
        n_points_intramural=200,
        anomalous_rca=True,
        anomalous_lca=False,
        control_plot=True,
    )

    # Now scale just the RCA region using centerline-based morphing
    mesh = results["mesh"]
    rca_points = results["rca_points"]
    rca_cl_raw = np.genfromtxt("data/centerline_rca.csv", delimiter=",")
    rca_centerline = mm.numpy_to_centerline(rca_cl_raw)

    print(f"\n=== CENTERLINE-BASED RCA MORPHING ===")
    print(f"Original mesh has {len(mesh.vertices)} vertices")
    print(f"RCA region has {len(rca_points)} vertices")

    # Expand RCA by 2mm using centerline-based morphing
    scaled_mesh = scale_region_centerline_morphing(
        mesh=mesh,
        region_points=rca_points,
        centerline=rca_centerline,
        diameter_adjustment_mm=0.4902,  # Positive to expand
    )

    # Create comparison plot
    compare_centerline_scaling(
        original_mesh=mesh,
        scaled_mesh=scaled_mesh,
        region_points=rca_points,
        centerline=rca_centerline,
    )

    # Return both the original and scaled mesh for further use
    results["scaled_mesh"] = scaled_mesh

    rest, _ = mm.from_file_singlepair("data/ivus_rest", write_obj=False)

    aligned, resampled_cl = mm.align_combined(
        rca_centerline,
        rest,
        (12.2605, -201.3643, 1751.0554),
        (11.7567, -202.1920, 1754.7975),
        (15.6605, -202.1920, 1749.9655),
        results["rca_points"],
        angle_range_deg=10.0,
        write=True,
        watertight=True,
        output_dir="test",
    )

    results = label_anomalous_region(
        centerline=rca_centerline,
        frames=aligned.geom_a.frames,
        results=results,
        results_key='rca_points',
        debug_plot=True,
    )

    distal, proximal, mean = find_distal_and_proximal_scaling(
        frames=aligned.geom_a.frames,
        results=results,
        debug_plot=True,
    )

    # paper plot
    scaled_proximal = scale_region_centerline_morphing(
        mesh=mesh,
        region_points=results['proximal_points'],
        centerline=rca_centerline,
        diameter_adjustment_mm=-0.9033,  # Positive to expand
    )

    scaled_distal = scale_region_centerline_morphing(
        mesh=mesh,
        region_points=results['distal_points'],
        centerline=rca_centerline,
        diameter_adjustment_mm=0.4902,  # Positive to expand
    )

    scaled_anomalous = scale_region_centerline_morphing(
        mesh=mesh,
        region_points=results['anomalous_points'],
        centerline=rca_centerline,
        diameter_adjustment_mm=-1.5,  # Positive to expand
    )

    anomaly_mesh = trimesh.load("test/lumen_000_None.obj")
    mesh_visual = mesh.copy()
    # semitransparent red
    mesh_visual.visual.face_colors = [128, 0, 0, 255]

    scene = trimesh.Scene([scaled_proximal, scaled_distal, scaled_anomalous, mesh, anomaly_mesh])
    scene.show()

    scene = trimesh.Scene([mesh, anomaly_mesh])
    scene.show()