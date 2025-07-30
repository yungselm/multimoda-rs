# tests/test_wrappers.py
import pytest
import numpy as np

from multimodars._wrappers import from_file, from_array, to_centerline
from multimodars import (
    from_array_full,
    from_array_doublepair,
    from_array_singlepair,
    geometry_from_array,
)
from multimodars import (
    PyGeometry, 
    PyGeometryPair, 
    PyCenterline
)

def test_full_file_arr_consistency(
    sample_rest_dia_arr,
    sample_rest_sys_arr,
    sample_stress_dia_arr,
    sample_stress_sys_arr,
):
    """from_array(…, mode=…) should match the corresponding Rust-backed function."""
    rest_f, stress_f, dia_f, sys_f, (dia_logs_f, sys_logs_f, dia_logs_stress_f, sys_logs_stress_f) = from_array(
        mode="full",
        rest_geometry_dia=sample_rest_dia_arr,
        rest_geometry_sys=sample_rest_sys_arr,
        stress_geometry_dia=sample_stress_dia_arr,
        stress_geometry_sys=sample_stress_sys_arr,
        steps_best_rotation=270,
        range_rotation_rad=90,
        interpolation_steps=0,
        rest_output_path="output/rest",
        stress_output_path="output/stress",
        diastole_output_path="output/diastole",
        systole_output_path="output/systole",
    )
    # call the Rust-backed function directly
    rest_a, stress_a, dia_a, sys_a, (dia_logs_a, sys_logs_a, dia_logs_stress_a, sys_logs_stress_a) = from_file(
        mode="full",
        rest_input_path="data/fixtures/rest_csv_files",
        stress_input_path="data/fixtures/stress_csv_files",
        steps_best_rotation=270,
        range_rotation_rad=90,
        rest_output_path="output/rest",
        stress_output_path="output/stress",
        diastole_output_path="output/diastole",
        systole_output_path="output/systole",
        interpolation_steps=0,
        image_center=(4.5, 4.5),
        radius=0.5,
        n_points=20,
    )

    # test consistency between the two
    assert len(rest_f.dia_geom.contours) == len(rest_a.dia_geom.contours)
    assert len(rest_f.sys_geom.contours) == len(rest_a.sys_geom.contours)
    assert len(rest_f.dia_geom.catheters) == len(rest_a.dia_geom.catheters)
    assert len(rest_f.sys_geom.catheters) == len(rest_a.sys_geom.catheters)
    assert len(rest_f.dia_geom.walls) == len(rest_a.dia_geom.walls)
    assert len(rest_f.sys_geom.walls) == len(rest_a.sys_geom.walls)
    assert len(stress_f.dia_geom.contours) == len(stress_a.dia_geom.contours)
    assert len(stress_f.sys_geom.contours) == len(stress_a.sys_geom.contours)
    assert len(stress_f.dia_geom.catheters) == len(stress_a.dia_geom.catheters)
    assert len(stress_f.sys_geom.catheters) == len(stress_a.sys_geom.catheters)
    assert len(stress_f.dia_geom.walls) == len(stress_a.dia_geom.walls)
    assert len(stress_f.sys_geom.walls) == len(stress_a.sys_geom.walls)
    assert len(dia_f.dia_geom.contours) == len(dia_a.dia_geom.contours)
    assert len(dia_f.sys_geom.contours) == len(dia_a.sys_geom.contours)
    assert len(dia_f.dia_geom.catheters) == len(dia_a.dia_geom.catheters)
    assert len(dia_f.sys_geom.catheters) == len(dia_a.sys_geom.catheters)
    assert len(dia_f.dia_geom.walls) == len(dia_a.dia_geom.walls)
    assert len(dia_f.sys_geom.walls) == len(dia_a.sys_geom.walls)
    assert len(sys_f.dia_geom.contours) == len(sys_a.dia_geom.contours)
    assert len(sys_f.sys_geom.contours) == len(sys_a.sys_geom.contours)
    assert len(sys_f.dia_geom.catheters) == len(sys_a.dia_geom.catheters)
    assert len(sys_f.sys_geom.catheters) == len(sys_a.sys_geom.catheters)
    assert len(sys_f.dia_geom.walls) == len(sys_a.dia_geom.walls)
    assert len(sys_f.sys_geom.walls) == len(sys_a.sys_geom.walls)

    # random contour check all points the same
    contour_f = rest_f.dia_geom.contours[-1]
    contour_a = rest_a.dia_geom.contours[-1]

    for pf, pa in zip(contour_f.points, contour_a.points):
        # exact integer checks
        assert pf.frame_index == pa.frame_index, (
            f"Frame index mismatch: {pf.frame_index} != {pa.frame_index}"
        )
        assert pf.point_index == pa.point_index, (
            f"Point index mismatch: {pf.point_index} != {pa.point_index}"
        )

        # floating‐point checks with pytest.approx
        assert pf.x == pytest.approx(pa.x, abs=0.01), (
            f"X coord mismatch: {pf.x} != {pa.x}"
        )
        assert pf.y == pytest.approx(pa.y, abs=0.01), (
            f"Y coord mismatch: {pf.y} != {pa.y}"
        )
        assert pf.z == pytest.approx(pa.z, abs=0.01), (
            f"Z coord mismatch: {pf.z} != {pa.z}"
        )

    # check logs
    assert dia_logs_f == dia_logs_a, "Diastole logs mismatch"
    assert sys_logs_f == sys_logs_a, "Systole logs mismatch"

# def test_from_file_and_from_array_pairwise(tmp_path, sample_rest_dia_arr, sample_rest_sys_arr):
#     """from_file should mirror from_array for single array inputs."""
#     # dump a single-npy file
#     file_path = tmp_path / "single.npy"
#     np.save(file_path, sample_rest_dia_arr)

#     # from_file reads one .npy and returns a PyGeometry
#     geom_from_file = from_file(str(file_path))
#     # from_array_full on a singleton list is equivalent to geometry_from_array
#     geom_from_arr = geometry_from_array(sample_rest_dia_arr)

#     assert geom_from_file == geom_from_arr


# def test_to_centerline_round_contour(sample_contour_round):
#     """to_centerline should return a PyCenterline with sorted points along the curve."""
#     # to_centerline wraps a single PyContour into a PyCenterline
#     cl: PyCenterline = to_centerline(sample_contour_round)
    
#     # must be the correct type
#     assert isinstance(cl, PyCenterline)
#     # should have exactly as many points as the contour
#     assert len(cl.points) == len(sample_contour_round.points)

#     # and each point in the centerline must lie approximately on the input circle
#     radii = [np.hypot(pt.x, pt.y) for pt in cl.points]
#     # all radii should be ≈ 4.0 (within 1e‑6)
#     assert all(np.isclose(r, 4.0, atol=1e-6) for r in radii)