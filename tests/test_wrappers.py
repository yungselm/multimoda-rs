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
from multimodars import PyGeometry, PyGeometryPair, PyCenterline


def assert_log_properties(
    logs,
    expected_rotation=15,
    rotation_tol=2.5,
    translation_step=0.01,
    translation_tol=1e-6,
):
    # reference centroid from first entry
    _, _, _, _, _, _, ref_cx, ref_cy = logs[0]

    for idx, entry in enumerate(logs):
        _, _, rel_rot, _, tx, ty, cx, cy = entry

        # 1) rotation magnitude within tolerance
        assert abs(abs(rel_rot) - expected_rotation) <= rotation_tol, \
            f"[entry {idx}] unexpected relative rotation: {rel_rot}"

        # 2) translation is an integer multiple of translation_step (within tol)
        n_tx = round(tx / translation_step)
        n_ty = round(ty / translation_step)
        assert abs(tx - n_tx * translation_step) <= translation_tol, \
            f"[entry {idx}] Tx {tx} is not a multiple of {translation_step}"
        assert abs(ty - n_ty * translation_step) <= translation_tol, \
            f"[entry {idx}] Ty {ty} is not a multiple of {translation_step}"

        # 3) centroid consistency
        assert cx == pytest.approx(ref_cx, abs=1e-6), \
            f"[entry {idx}] centroid x changed: {cx} != {ref_cx}"
        assert cy == pytest.approx(ref_cy, abs=1e-6), \
            f"[entry {idx}] centroid y changed: {cy} != {ref_cy}"

def test_full_file_arr_consistency(
    sample_rest_dia_arr,
    sample_rest_sys_arr,
    sample_stress_dia_arr,
    sample_stress_sys_arr,
    tmp_path,
):
    # create subdirs under tmp_path
    rest_out = tmp_path / "output" / "rest"
    stress_out = tmp_path / "output" / "stress"
    dia_out = tmp_path / "output" / "diastole"
    sys_out = tmp_path / "output" / "systole"
    # pytest will automatically mkdir when you write files into it

    (
        rest_a,
        stress_a,
        dia_a,
        sys_a,
        (dia_logs_a, sys_logs_a, dia_logs_stress_a, sys_logs_stress_a),
    ) = from_array(
        mode="full",
        rest_geometry_dia=sample_rest_dia_arr,
        rest_geometry_sys=sample_rest_sys_arr,
        stress_geometry_dia=sample_stress_dia_arr,
        stress_geometry_sys=sample_stress_sys_arr,
        step_rotation_deg=0.1,
        range_rotation_deg=30,
        image_center=(4.5, 4.5),
        radius=0.5,
        n_points=20,
        write_obj=True,
        rest_output_path=str(rest_out),
        stress_output_path=str(stress_out),
        diastole_output_path=str(dia_out),
        systole_output_path=str(sys_out),
        interpolation_steps=0,
        bruteforce=False,
    )

    (
        rest_f,
        stress_f,
        dia_f,
        sys_f,
        (dia_logs_f, sys_logs_f, dia_logs_stress_f, sys_logs_stress_f),
    ) = from_file(
        mode="full",
        rest_input_path="data/fixtures/idealized_geometry",
        stress_input_path="data/fixtures/idealized_geometry",
        step_rotation_deg=0.1,
        range_rotation_deg=30,
        image_center=(4.5, 4.5),
        radius=0.5,
        n_points=20,
        write_obj=True,
        rest_output_path=str(rest_out),
        stress_output_path=str(stress_out),
        diastole_output_path=str(dia_out),
        systole_output_path=str(sys_out),
        interpolation_steps=0,
        bruteforce=False,
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
        assert (
            pf.frame_index == pa.frame_index
        ), f"Frame index mismatch: {pf.frame_index} != {pa.frame_index}"
        assert (
            pf.point_index == pa.point_index
        ), f"Point index mismatch: {pf.point_index} != {pa.point_index}"

        # floating‐point checks with pytest.approx
        assert pf.x == pytest.approx(
            pa.x, abs=0.01
        ), f"X coord mismatch: {pf.x} != {pa.x}"
        assert pf.y == pytest.approx(
            pa.y, abs=0.01
        ), f"Y coord mismatch: {pf.y} != {pa.y}"
        assert pf.z == pytest.approx(
            pa.z, abs=0.01
        ), f"Z coord mismatch: {pf.z} != {pa.z}"

    # check logs
    assert dia_logs_f == dia_logs_a, "Diastole logs mismatch"
    assert sys_logs_f == sys_logs_a, "Systole logs mismatch"
    assert dia_logs_stress_f == dia_logs_stress_a, "Diastole stress logs mismatch"
    assert sys_logs_stress_f == sys_logs_stress_a, "Systole stress logs mismatch"

    for logs in (dia_logs_a, dia_logs_f, sys_logs_a, sys_logs_f):
        assert_log_properties(logs)


def test_doublepair_file_arr_consistency(
    sample_rest_dia_arr,
    sample_rest_sys_arr,
    sample_stress_dia_arr,
    sample_stress_sys_arr,
    tmp_path,
):
    # create subdirs under tmp_path
    rest_out = tmp_path / "output" / "rest"
    stress_out = tmp_path / "output" / "stress"
    # pytest will automatically mkdir when you write files into it

    (
        rest_a,
        stress_a,
        (dia_logs_a, sys_logs_a, dia_logs_stress_a, sys_logs_stress_a),
    ) = from_array(
        mode="doublepair",
        rest_geometry_dia=sample_rest_dia_arr,
        rest_geometry_sys=sample_rest_sys_arr,
        stress_geometry_dia=sample_stress_dia_arr,
        stress_geometry_sys=sample_stress_sys_arr,
        step_rotation_deg=0.1,
        range_rotation_deg=30,
        image_center=(4.5, 4.5),
        radius=0.5,
        n_points=20,
        write_obj=True,
        rest_output_path=str(rest_out),
        stress_output_path=str(stress_out),
        interpolation_steps=0,
        bruteforce=False,
    )

    (
        rest_f,
        stress_f,
        (dia_logs_f, sys_logs_f, dia_logs_stress_f, sys_logs_stress_f),
    ) = from_file(
        mode="doublepair",
        rest_input_path="data/fixtures/idealized_geometry",
        stress_input_path="data/fixtures/idealized_geometry",
        step_rotation_deg=0.1,
        range_rotation_deg=30,
        image_center=(4.5, 4.5),
        radius=0.5,
        n_points=20,
        write_obj=True,
        rest_output_path=str(rest_out),
        stress_output_path=str(stress_out),
        interpolation_steps=0,
        bruteforce=False,
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

    # random contour check all points the same
    contour_f = rest_f.dia_geom.contours[-1]
    contour_a = rest_a.dia_geom.contours[-1]

    for pf, pa in zip(contour_f.points, contour_a.points):
        # exact integer checks
        assert (
            pf.frame_index == pa.frame_index
        ), f"Frame index mismatch: {pf.frame_index} != {pa.frame_index}"
        assert (
            pf.point_index == pa.point_index
        ), f"Point index mismatch: {pf.point_index} != {pa.point_index}"

        # floating‐point checks with pytest.approx
        assert pf.x == pytest.approx(
            pa.x, abs=0.01
        ), f"X coord mismatch: {pf.x} != {pa.x}"
        assert pf.y == pytest.approx(
            pa.y, abs=0.01
        ), f"Y coord mismatch: {pf.y} != {pa.y}"
        assert pf.z == pytest.approx(
            pa.z, abs=0.01
        ), f"Z coord mismatch: {pf.z} != {pa.z}"

    # check logs
    assert dia_logs_f == dia_logs_a, "Diastole logs mismatch"
    assert sys_logs_f == sys_logs_a, "Systole logs mismatch"
    assert dia_logs_stress_f == dia_logs_stress_a, "Diastole stress logs mismatch"
    assert sys_logs_stress_f == sys_logs_stress_a, "Systole stress logs mismatch"

    for logs in (dia_logs_a, dia_logs_f, sys_logs_a, sys_logs_f):
        assert_log_properties(logs)

def test_singlepair_file_arr_consistency(
    sample_rest_dia_arr,
    sample_rest_sys_arr,
    tmp_path,
):
    # create subdirs under tmp_path
    output_path = tmp_path / "output" / "singlepair"
    # pytest will automatically mkdir when you write files into it

    (
        rest_a,
        (dia_logs_a, sys_logs_a),
    ) = from_array(
        mode="singlepair",
        geometry_dia=sample_rest_dia_arr,
        geometry_sys=sample_rest_sys_arr,
        step_rotation_deg=0.1,
        range_rotation_deg=30,
        image_center=(4.5, 4.5),
        radius=0.5,
        n_points=20,
        write_obj=True,
        output_path=str(output_path),
        interpolation_steps=0,
        bruteforce=False,
    )

    (
        rest_f,
        (dia_logs_f, sys_logs_f),
    ) = from_file(
        mode="singlepair",
        input_path="data/fixtures/idealized_geometry",
        step_rotation_deg=0.1,
        range_rotation_deg=30,
        image_center=(4.5, 4.5),
        radius=0.5,
        n_points=20,
        write_obj=True,
        output_path=str(output_path),
        interpolation_steps=0,
        bruteforce=False,
    )

    # test consistency between the two
    assert len(rest_f.dia_geom.contours) == len(rest_a.dia_geom.contours)
    assert len(rest_f.sys_geom.contours) == len(rest_a.sys_geom.contours)
    assert len(rest_f.dia_geom.catheters) == len(rest_a.dia_geom.catheters)
    assert len(rest_f.sys_geom.catheters) == len(rest_a.sys_geom.catheters)
    assert len(rest_f.dia_geom.walls) == len(rest_a.dia_geom.walls)
    assert len(rest_f.sys_geom.walls) == len(rest_a.sys_geom.walls)

    # random contour check all points the same
    contour_f = rest_f.dia_geom.contours[-1]
    contour_a = rest_a.dia_geom.contours[-1]

    for pf, pa in zip(contour_f.points, contour_a.points):
        # exact integer checks
        assert (
            pf.frame_index == pa.frame_index
        ), f"Frame index mismatch: {pf.frame_index} != {pa.frame_index}"
        assert (
            pf.point_index == pa.point_index
        ), f"Point index mismatch: {pf.point_index} != {pa.point_index}"

        # floating‐point checks with pytest.approx
        assert pf.x == pytest.approx(
            pa.x, abs=0.01
        ), f"X coord mismatch: {pf.x} != {pa.x}"
        assert pf.y == pytest.approx(
            pa.y, abs=0.01
        ), f"Y coord mismatch: {pf.y} != {pa.y}"
        assert pf.z == pytest.approx(
            pa.z, abs=0.01
        ), f"Z coord mismatch: {pf.z} != {pa.z}"

    # check logs
    assert dia_logs_f == dia_logs_a, "Diastole logs mismatch"
    assert sys_logs_f == sys_logs_a, "Systole logs mismatch"

    for logs in (dia_logs_a, dia_logs_f, sys_logs_a, sys_logs_f):
        assert_log_properties(logs)

def test_single_file_arr_consistency(
    sample_rest_dia_arr,
    tmp_path,
):
    # create subdirs under tmp_path
    output_path = tmp_path / "output" / "single"
    # pytest will automatically mkdir when you write files into it

    (
        rest_a, dia_logs_a,
    ) = from_array(
        mode="single",
        geometry=sample_rest_dia_arr,
        step_rotation_deg=0.1,
        range_rotation_deg=30,
        image_center=(4.5, 4.5),
        radius=0.5,
        n_points=20,
        label="rest",
        records=None,
        delta=0.1,
        max_rounds=5,
        diastole=True,
        sort=False,
        write_obj=True,
        output_path=str(output_path),
        bruteforce=False,
    )

    (
        rest_f, dia_logs_f,
    ) = from_file(
        mode="single",
        input_path="data/fixtures/idealized_geometry",
        step_rotation_deg=0.1,
        range_rotation_deg=30,
        diastole=True,
        image_center=(4.5, 4.5),
        radius=0.5,
        n_points=20,
        write_obj=True,
        output_path=str(output_path),
        bruteforce=False,
    )

    # test consistency between the two
    assert len(rest_f.contours) == len(rest_a.contours)
    assert len(rest_f.contours) == len(rest_a.contours)
    assert len(rest_f.catheters) == len(rest_a.catheters)
    assert len(rest_f.catheters) == len(rest_a.catheters)
    assert len(rest_f.walls) == len(rest_a.walls)
    assert len(rest_f.walls) == len(rest_a.walls)

    # random contour check all points the same
    contour_f = rest_f.contours[-1]
    contour_a = rest_a.contours[-1]

    for pf, pa in zip(contour_f.points, contour_a.points):
        # exact integer checks
        assert (
            pf.frame_index == pa.frame_index
        ), f"Frame index mismatch: {pf.frame_index} != {pa.frame_index}"
        assert (
            pf.point_index == pa.point_index
        ), f"Point index mismatch: {pf.point_index} != {pa.point_index}"

        # floating‐point checks with pytest.approx
        assert pf.x == pytest.approx(
            pa.x, abs=0.01
        ), f"X coord mismatch: {pf.x} != {pa.x}"
        assert pf.y == pytest.approx(
            pa.y, abs=0.01
        ), f"Y coord mismatch: {pf.y} != {pa.y}"
        assert pf.z == pytest.approx(
            pa.z, abs=0.01
        ), f"Z coord mismatch: {pf.z} != {pa.z}"

    # check logs
    assert dia_logs_f == dia_logs_a, "Logs mismatch"

    for logs in (dia_logs_a, dia_logs_f):
        assert_log_properties(logs)
