# tests/test_annotations.py
import numpy as np
import pytest

from multimodars import (
    PyAnnotation,
    PyContour,
    PyContourPoint,
    PyFrame,
    from_array_single,
    numpy_to_geometry,
    numpy_to_inputdata,
    to_array,
)

N_FRAMES = 6
N_POINTS = 32


def _point(frame, i, x, y, z):
    return PyContourPoint(frame_index=frame, point_index=i, x=x, y=y, z=z, aortic=False)


def _lumen_arr():
    """Elliptic lumen, each frame shifted and twisted so alignment has work to do."""
    rows = []
    t = np.linspace(0, 2 * np.pi, N_POINTS, endpoint=False)
    for k in range(N_FRAMES):
        frame = 10 + k
        a = np.deg2rad(8.0 * k)
        x, y = 2.0 * np.cos(t), 1.4 * np.sin(t)
        xr = x * np.cos(a) - y * np.sin(a) + 4.5 + 0.2 * k
        yr = x * np.sin(a) + y * np.cos(a) + 4.5 - 0.1 * k
        rows += [(frame, xi, yi, float(k)) for xi, yi in zip(xr, yr)]
    return np.array(rows, dtype=float)


def _rows_on_lumen(lumen, frame, idx, group):
    sel = lumen[lumen[:, 0] == frame][idx]
    return np.column_stack(
        [sel[:, 0], np.full(len(sel), group), sel[:, 1], sel[:, 2], sel[:, 3]]
    )


def _make_input():
    lumen = _lumen_arr()
    annotations = {
        ("sidebranch", "region"): np.vstack(
            [
                _rows_on_lumen(lumen, 11, [0, 3, 6, 9], 0),
                _rows_on_lumen(lumen, 11, [16, 18], 1),
                _rows_on_lumen(lumen, 13, [2, 4, 8], 0),
            ]
        ),
        ("sidebranch", "landmark"): _rows_on_lumen(lumen, 11, [5], 0),
        ("fibrous_cap", "point"): _rows_on_lumen(lumen, 12, [7, 20], 0),
    }
    ref = np.array([10, 6.5, 4.5, 0.0])
    inp = numpy_to_inputdata(
        lumen_arr=lumen,
        ref_point=ref,
        diastole=True,
        label="ann",
        annotations=annotations,
    )
    return lumen, annotations, inp


def test_pyannotation_kind_role_and_transforms():
    pts = [_point(0, 0, 1.0, 0.0, 2.0), _point(0, 1, 0.0, 1.0, 2.0)]
    ann = PyAnnotation("sidebranch", "region", 2, pts)
    assert (ann.kind, ann.role, ann.group, len(ann)) == ("sidebranch", "region", 2, 2)
    assert not ann.is_custom

    custom = PyAnnotation("fibrous_cap", "axis_tip")
    assert custom.kind == "fibrous_cap" and custom.is_custom
    custom.kind = "Lipid"
    assert custom.kind == "lipid" and not custom.is_custom

    with pytest.raises(ValueError):
        PyAnnotation("lipid", "outline")
    with pytest.raises(ValueError):
        custom.role = "outline"
    assert set(PyAnnotation.roles()) == {
        "region",
        "landmark",
        "axis_tip",
        "diameter_end",
        "point",
    }

    rot = ann.rotate(90.0, center=(0.0, 0.0))
    assert rot.points[0].x == pytest.approx(0.0, abs=1e-12)
    assert rot.points[0].y == pytest.approx(1.0)
    moved = ann.translate(1.0, 2.0, 3.0)
    assert (moved.points[1].x, moved.points[1].y, moved.points[1].z) == (1.0, 3.0, 5.0)


def test_pyframe_annotations_round_trip_and_follow_rotation():
    lumen_pts = [
        _point(3, i, 1.0 + np.cos(t), 2.0 + np.sin(t), 0.5)
        for i, t in enumerate(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    ]
    lumen = PyContour(3, 3, lumen_pts, (1.0, 2.0, 0.5), None, None, "Lumen")
    anns = [
        PyAnnotation("sidebranch", "region", 0, [lumen_pts[1], lumen_pts[2]]),
        PyAnnotation("fibrous_cap", "diameter_end", 4, [lumen_pts[5]]),
    ]
    frame = PyFrame(3, (1.0, 2.0, 0.5), lumen, {}, None, annotations=anns)
    assert [a.kind for a in frame.annotations] == ["sidebranch", "fibrous_cap"]

    rotated = frame.rotate(33.0).translate(0.5, -1.0, 2.0)
    assert [(a.kind, a.role, a.group) for a in rotated.annotations] == [
        ("sidebranch", "region", 0),
        ("fibrous_cap", "diameter_end", 4),
    ]
    lumen_xyz = np.array([(p.x, p.y, p.z) for p in rotated.lumen.points])
    for ann in rotated.annotations:
        for p in ann.points:
            d = np.linalg.norm(lumen_xyz - (p.x, p.y, p.z), axis=1).min()
            assert d < 1e-9


def test_annotations_survive_pipeline_rigidly():
    lumen, annotations, inp = _make_input()
    assert len(inp.annotations) == 5

    # distance of each annotation point to its frame centroid before alignment
    centroids = {f: lumen[lumen[:, 0] == f][:, 1:4].mean(axis=0) for f in range(10, 16)}
    before: dict = {}
    for key, rows in annotations.items():
        for r in rows:
            d = np.linalg.norm(r[2:5] - centroids[int(r[0])])
            before.setdefault((key, int(r[0])), []).append(d)

    for smooth in (False, True):
        geom, _ = from_array_single(
            inp,
            step_rotation_deg=1.0,
            range_rotation_deg=30.0,
            n_points=0,
            smooth=smooth,
        )
        out = to_array(geom)["annotations"]
        assert set(out) == set(annotations)
        for key, rows in annotations.items():
            assert out[key].shape == rows.shape
            assert sorted(out[key][:, 1]) == sorted(rows[:, 1])

        by_original = {f.lumen.original_frame: f for f in geom.frames}
        after: dict = {}
        for original, frame in by_original.items():
            c = np.array(frame.centroid)
            lumen_xyz = np.array([(p.x, p.y, p.z) for p in frame.lumen.points])
            for ann in frame.annotations:
                for p in ann.points:
                    xyz = np.array([p.x, p.y, p.z])
                    d = np.linalg.norm(xyz - c)
                    after.setdefault(((ann.kind, ann.role), original), []).append(d)
                    if not smooth:
                        # smoothing averages lumen points across frames, rigid moves do not
                        assert np.linalg.norm(lumen_xyz - xyz, axis=1).min() < 1e-6
        assert set(after) == set(before)
        for k, expected in before.items():
            np.testing.assert_allclose(after[k], expected, atol=1e-6)
        assert len(by_original[11].annotations) == 3
        assert len(by_original[10].annotations) == 0


def test_custom_kind_round_trips_through_numpy_geometry():
    _, _, inp = _make_input()
    geom, _ = from_array_single(inp, step_rotation_deg=1.0, range_rotation_deg=30.0)
    arrays = to_array(geom)
    rebuilt = numpy_to_geometry(
        arrays["lumen"],
        reference_arr=arrays["reference"],
        annotations=arrays["annotations"],
    )
    kinds = sorted(
        {(a.kind, a.is_custom) for f in rebuilt.frames for a in f.annotations}
    )
    assert kinds == [("fibrous_cap", True), ("sidebranch", False)]
    again = to_array(rebuilt)["annotations"]

    def _sorted(a):
        return a[np.lexsort(a.T[::-1])]

    assert set(again) == set(arrays["annotations"])
    for key, rows in arrays["annotations"].items():
        np.testing.assert_allclose(_sorted(again[key]), _sorted(rows))


def test_deprecated_sidebranch_param_maps_to_region_annotations():
    lumen = _lumen_arr()
    legacy = lumen[np.isin(lumen[:, 0], [11, 13])][:, :4]
    with pytest.warns(DeprecationWarning):
        inp = numpy_to_inputdata(
            lumen_arr=lumen,
            ref_point=np.array([10, 6.5, 4.5, 0.0]),
            diastole=True,
            sidebranch=legacy,
        )
    assert inp.sidebranch is None
    assert [(f, a.kind, a.role, a.group, len(a)) for f, a in inp.annotations] == [
        (11, "sidebranch", "region", 0, N_POINTS),
        (13, "sidebranch", "region", 0, N_POINTS),
    ]

    geom, _ = from_array_single(inp, step_rotation_deg=1.0, range_rotation_deg=30.0)
    arrays = to_array(geom)
    # deprecated key keeps the old (frame_index, x, y, z) layout
    assert arrays["sidebranch"].shape == (2 * N_POINTS, 4)
    np.testing.assert_allclose(
        arrays["sidebranch"],
        arrays["annotations"][("sidebranch", "region")][:, [0, 2, 3, 4]],
    )
    assert arrays["calcification"].shape == (0, 4)


def test_invalid_annotation_inputs_raise():
    lumen = _lumen_arr()
    ref = np.array([10, 6.5, 4.5, 0.0])
    with pytest.raises(ValueError):
        numpy_to_inputdata(
            lumen, ref, True, annotations={"sidebranch": np.zeros((1, 5))}
        )
    with pytest.raises(ValueError):
        numpy_to_inputdata(
            lumen, ref, True, annotations={("lipid", "region"): np.zeros((1, 4))}
        )
    with pytest.raises(ValueError):
        numpy_to_inputdata(
            lumen, ref, True, annotations={("lipid", "outline"): np.zeros((1, 5))}
        )


def test_deprecated_pyframe_extras_and_pyinputdata_layers():
    from multimodars import PyInputData

    pts = [
        _point(2, i, 1.0 + np.cos(t), np.sin(t), 0.0)
        for i, t in enumerate(np.linspace(0, 2 * np.pi, 6, endpoint=False))
    ]
    lumen = PyContour(2, 2, pts, (1.0, 0.0, 0.0), None, None, "Lumen")
    branch = PyContour(2, 2, pts[:3], (1.0, 0.0, 0.0), None, None, "Sidebranch")
    frame = PyFrame(2, (1.0, 0.0, 0.0), lumen, {"Sidebranch": branch}, None)
    moved = frame.translate(1.0, 0.0, 0.0)
    assert moved.extras == {}
    assert [(a.kind, a.role, len(a)) for a in moved.annotations] == [
        ("sidebranch", "region", 3)
    ]
    assert moved.annotations[0].points[0].x == pytest.approx(pts[0].x + 1.0)

    with pytest.warns(DeprecationWarning):
        inp = PyInputData([lumen], None, [branch], None, None, pts[0], True, "x")
    assert inp.calcification is not None and inp.annotations == []

    # the deprecated layer reaches the geometry as region annotations
    lumen_arr, _, inp = _make_input()
    rows = lumen_arr[lumen_arr[:, 0] == 12][:4]
    calc_pts = [_point(int(r[0]), i, r[1], r[2], r[3]) for i, r in enumerate(rows)]
    inp.calcification = [
        PyContour(12, 12, calc_pts, (0.0, 0.0, 0.0), None, None, "Lumen")
    ]
    geom, _ = from_array_single(inp, step_rotation_deg=1.0, range_rotation_deg=30.0)
    arrays = to_array(geom)
    assert arrays["calcification"].shape == (4, 4)
    assert arrays["annotations"][("calcification", "region")].shape == (4, 5)
