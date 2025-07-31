# tests/test_converters.py
import numpy as np
import pytest

from multimodars import (
    PyContourPoint,
    PyContour,
    PyCenterline,
    PyGeometry,
    PyGeometryPair,
)
from multimodars import (
    to_array,
    numpy_to_geometry,
    numpy_to_centerline,
)


def _make_simple_contour(contour_id: int, n: int = 4, offset: float = 0.0):
    pts = [
        PyContourPoint(
            frame_index=contour_id,
            point_index=i,
            x=float(i) + offset,
            y=2.0 * float(i) + offset,
            z=3.0 * float(i) + offset,
            aortic=(i % 2 == 0),
        )
        for i in range(n)
    ]
    return PyContour(contour_id, pts)


def _make_simple_centerline(n: int = 5):
    pts = [
        PyContourPoint(
            frame_index=i,
            point_index=0,
            x=float(i),
            y=float(i) + 0.5,
            z=float(i) + 1.0,
            aortic=False,
        )
        for i in range(n)
    ]
    return PyCenterline.from_contour_points(pts)


def test_to_array_and_back_contour():
    c = _make_simple_contour(7, n=3, offset=1.0)
    arr = to_array(c)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3, 4)

    # Use the arrays directly in numpy_to_geometry
    rebuilt = numpy_to_geometry(
        contours_arr=arr,
        catheters_arr=np.zeros((0, 4)),
        walls_arr=np.zeros((0, 4)),
        reference_arr=np.array([[0.0, 0.0, 0.0, 0.0]]),
    )
    assert len(rebuilt.contours) == 1
    pts = rebuilt.contours[0].points
    assert len(pts) == 3
    for orig, new in zip(c.points, pts):
        assert pytest.approx(orig.x) == new.x
        assert pytest.approx(orig.y) == new.y
        assert pytest.approx(orig.z) == new.z


def test_to_array_centerline_and_back():
    cl = _make_simple_centerline(n=4)
    arr = to_array(cl)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (4, 4)

    new_cl = numpy_to_centerline(arr[:, 1:4])  # expects (N,3)
    assert isinstance(new_cl, PyCenterline)

    arr2 = to_array(new_cl)
    assert arr2.shape == arr.shape
    np.testing.assert_allclose(arr2[:, 1:4], arr[:, 1:4], rtol=1e-6, atol=0)


def test_to_array_and_back_geometry_roundtrip():
    c0 = _make_simple_contour(0, n=2, offset=0.0)
    c1 = _make_simple_contour(1, n=3, offset=10.0)
    cat = _make_simple_contour(0, n=2, offset=-5.0)
    wall = _make_simple_contour(0, n=4, offset=2.0)
    ref = PyContourPoint(0, 0, 100.0, 101.0, 102.0, False)
    geom = PyGeometry(
        contours=[c0, c1], catheters=[cat], walls=[wall], reference_point=ref
    )

    # Convert to dictionary of arrays
    arr_dict = to_array(geom)
    assert isinstance(arr_dict, dict)
    assert set(arr_dict.keys()) == {"contours", "catheters", "walls", "reference"}

    # Round-trip using numpy_to_geometry
    geom2 = numpy_to_geometry(
        contours_arr=arr_dict["contours"],
        catheters_arr=arr_dict["catheters"],
        walls_arr=arr_dict["walls"],
        reference_arr=arr_dict["reference"],
    )

    # Validate geometry structure
    assert len(geom2.contours) == 2
    assert len(geom2.catheters) == 1
    assert len(geom2.walls) == 1
    assert pytest.approx(geom2.reference_point.x) == ref.x
    assert pytest.approx(geom2.reference_point.y) == ref.y
    assert pytest.approx(geom2.reference_point.z) == ref.z

    # Validate contour points
    for orig, roundp in zip(geom.contours, geom2.contours):
        assert len(orig.points) == len(roundp.points)
        for o_pt, r_pt in zip(orig.points, roundp.points):
            assert pytest.approx(o_pt.x) == r_pt.x
            assert pytest.approx(o_pt.y) == r_pt.y
            assert pytest.approx(o_pt.z) == r_pt.z


def test_to_array_geometry_pair():
    g1 = PyGeometry(
        contours=[_make_simple_contour(0, n=1)],
        catheters=[],
        walls=[],
        reference_point=PyContourPoint(0, 0, 0, 0, 0, False),
    )
    g2 = PyGeometry(
        contours=[_make_simple_contour(1, n=2)],
        catheters=[],
        walls=[],
        reference_point=PyContourPoint(0, 0, 1, 1, 1, False),
    )
    pair = PyGeometryPair(dia_geom=g1, sys_geom=g2)

    # Convert to tuple of dictionaries
    dia_dict, sys_dict = to_array(pair)

    assert isinstance(dia_dict, dict) and isinstance(sys_dict, dict)
    assert dia_dict["contours"].shape == (1, 4)
    assert sys_dict["contours"].shape == (2, 4)
    assert dia_dict["reference"].shape == (1, 4)
    assert sys_dict["reference"].shape == (1, 4)
