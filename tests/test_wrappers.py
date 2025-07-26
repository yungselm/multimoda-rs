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

@pytest.mark.parametrize("mode, rust_fn", [
    ("full", from_array_full),
    ("doublepair", from_array_doublepair),
    ("singlepair", from_array_singlepair),
])
def test_wrapper_rust_consistency_arr_modes(
    sample_rest_dia_arr,
    sample_rest_sys_arr,
    sample_stress_dia_arr,
    sample_stress_sys_arr,
    mode, rust_fn
):
    """from_array(…, mode=…) should match the corresponding Rust-backed function."""
    # call the Python wrapper
    wrap_outputs = from_array(
        mode=mode,
        rest_geometry_dia=sample_rest_dia_arr,
        rest_geometry_sys=sample_rest_sys_arr,
        stress_geometry_dia=sample_stress_dia_arr,
        stress_geometry_sys=sample_stress_sys_arr,
        steps_best_rotation=300,
        range_rotation_rad=1.57,
        interpolation_steps=28,
        rest_output_path="output/rest",
        stress_output_path="output/stress",
        diastole_output_path="output/diastole",
        systole_output_path="output/systole",
    )
    # call the Rust-backed function directly
    rust_outputs = rust_fn(
        rest_geometry_dia=sample_rest_dia_arr,
        rest_geometry_sys=sample_rest_sys_arr,
        stress_geometry_dia=sample_stress_dia_arr,
        stress_geometry_sys=sample_stress_sys_arr,
        steps_best_rotation=300,
        range_rotation_rad=1.57,
        interpolation_steps=28,
        rest_output_path="output/rest",
        stress_output_path="output/stress",
        diastole_output_path="output/diastole",
        systole_output_path="output/systole",
    )

    # both should return the same number of elements
    assert len(wrap_outputs) == len(rust_outputs)

    # and each element should be “equal” (they’re dataclasses with __eq__ defined)
    for w, r in zip(wrap_outputs, rust_outputs):
        assert w == r


def test_from_file_and_from_array_pairwise(tmp_path, sample_rest_dia_arr, sample_rest_sys_arr):
    """from_file should mirror from_array for single array inputs."""
    # dump a single-npy file
    file_path = tmp_path / "single.npy"
    np.save(file_path, sample_rest_dia_arr)

    # from_file reads one .npy and returns a PyGeometry
    geom_from_file = from_file(str(file_path))
    # from_array_full on a singleton list is equivalent to geometry_from_array
    geom_from_arr = geometry_from_array(sample_rest_dia_arr)

    assert geom_from_file == geom_from_arr


def test_to_centerline_round_contour(sample_contour_round):
    """to_centerline should return a PyCenterline with sorted points along the curve."""
    # to_centerline wraps a single PyContour into a PyCenterline
    cl: PyCenterline = to_centerline(sample_contour_round)
    
    # must be the correct type
    assert isinstance(cl, PyCenterline)
    # should have exactly as many points as the contour
    assert len(cl.points) == len(sample_contour_round.points)

    # and each point in the centerline must lie approximately on the input circle
    radii = [np.hypot(pt.x, pt.y) for pt in cl.points]
    # all radii should be ≈ 4.0 (within 1e‑6)
    assert all(np.isclose(r, 4.0, atol=1e-6) for r in radii)