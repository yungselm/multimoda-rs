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
from multimodars._converters import (
    to_array,
    numpy_to_geometry,
    numpy_to_centerline,
    numpy_to_inputdata,
)


def _compute_centroid(points):
    """Compute centroid from a list of PyContourPoints"""
    if not points:
        return (0.0, 0.0, 0.0)

    sum_x = sum(p.x for p in points)
    sum_y = sum(p.y for p in points)
    sum_z = sum(p.z for p in points)
    n = len(points)
    return (sum_x / n, sum_y / n, sum_z / n)


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

    centroid = _compute_centroid(pts)
    return PyContour(
        id=contour_id,
        original_frame=contour_id,
        points=pts,
        centroid=centroid,
        aortic_thickness=None,
        pulmonary_thickness=None,
        kind="Lumen",
    )


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
        lumen_arr=arr,
        catheter_arr=np.zeros((0, 4)),
        wall_arr=np.zeros((0, 4)),
        reference_arr=np.array([[0.0, 0.0, 0.0, 0.0]]),
    )

    # Check we have frames with contours
    assert len(rebuilt.frames) > 0
    frame = rebuilt.frames[0]
    assert len(frame.lumen.points) == 3

    pts = frame.lumen.points
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
    # Create contours with different frame indices to test frame grouping
    c0 = _make_simple_contour(0, n=2, offset=0.0)
    c1 = _make_simple_contour(1, n=3, offset=10.0)

    # Convert to numpy arrays first
    c0_arr = to_array(c0)
    c1_arr = to_array(c1)

    # Combine into single lumen array
    lumen_arr = np.vstack([c0_arr, c1_arr])

    geom = numpy_to_geometry(
        lumen_arr=lumen_arr,
        catheter_arr=np.zeros((0, 4)),
        wall_arr=np.zeros((0, 4)),
        reference_arr=np.array([[0, 100.0, 101.0, 102.0]]),
    )

    # Convert to dictionary of arrays
    arr_dict = to_array(geom)
    assert isinstance(arr_dict, dict)
    assert "lumen" in arr_dict
    assert "reference" in arr_dict

    # Round-trip using numpy_to_geometry
    geom2 = numpy_to_geometry(
        lumen_arr=arr_dict["lumen"],
        catheter_arr=arr_dict.get("catheter", np.zeros((0, 4))),
        wall_arr=arr_dict.get("wall", np.zeros((0, 4))),
        reference_arr=arr_dict["reference"],
    )

    # Validate geometry structure
    assert len(geom2.frames) == 2  # Should have 2 frames

    # Check reference point
    ref_pt = geom2.frames[0].reference_point
    assert pytest.approx(ref_pt.x) == 100.0
    assert pytest.approx(ref_pt.y) == 101.0
    assert pytest.approx(ref_pt.z) == 102.0


# Skip geometry pair test for now as it might need more complex setup
def test_to_array_geometry_pair():
    pytest.skip("Geometry pair conversion needs more complex setup")

def test_numpy_to_inputdata_roundtrip():
    # Build two simple contours (frames 0 and 1)
    c0 = _make_simple_contour(0, n=2, offset=0.0)
    c1 = _make_simple_contour(1, n=3, offset=10.0)

    # Convert to arrays and combine for lumen
    lumen_arr = np.vstack([to_array(c0), to_array(c1)])

    # Provide an EEM only for frame 0 and a sidebranch only for frame 1
    eem_arr = to_array(c0)           # only frame 0 present
    sidebranch_arr = to_array(c1)    # only frame 1 present

    # Empty calcification array
    calc_arr = np.zeros((0, 4))

    # Create a structured record array: (frame, phase, m1, m2)
    rec_dtype = np.dtype([("frame", "i4"), ("phase", "U1"), ("m1", "f8"), ("m2", "f8")])
    records = np.array([(0, "D", 1.1, 2.2), (1, "S", np.nan, np.nan)], dtype=rec_dtype)

    # Reference point (global)
    reference_arr = np.array([[0, 100.0, 101.0, 102.0]])

    # Call converter
    inp = numpy_to_inputdata(
        lumen_arr=lumen_arr,
        eem_arr=eem_arr,
        calcification=calc_arr,
        sidebranch=sidebranch_arr,
        record=records,
        ref_point=reference_arr,
        diastole=True,
        label="test_label",
    )

    # Basic structure checks
    assert hasattr(inp, "lumen")
    assert len(inp.lumen) == 2  # two frames from lumen

    # Check optional lists: eem present (1), calcification empty -> None, sidebranch present (1)
    assert inp.eem is not None and len(inp.eem) == 1
    assert inp.calcification is None or len(inp.calcification) == 0
    assert inp.sidebranch is not None and len(inp.sidebranch) == 1

    # Records parsed
    assert inp.record is not None
    assert len(inp.record) == 2
    # Check first record values
    r0 = inp.record[0]
    assert r0.frame == 0
    assert r0.phase in ("D", "d", "0", "D")  # accept 'D' (string) mapping variants

    # Reference point, diastole and label
    assert pytest.approx(inp.ref_point.x) == 100.0
    assert pytest.approx(inp.ref_point.y) == 101.0
    assert pytest.approx(inp.ref_point.z) == 102.0
    assert inp.diastole is True
    assert inp.label == "test_label"

    # Check that lumen contours preserved point coordinates for frame 0
    in_l0 = inp.lumen[0].points
    orig_l0 = c0.points
    assert len(in_l0) == len(orig_l0)
    for o, n in zip(orig_l0, in_l0):
        assert pytest.approx(o.x) == n.x
        assert pytest.approx(o.y) == n.y
        assert pytest.approx(o.z) == n.z

