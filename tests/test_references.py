# tests/test_references.py
import shutil
from pathlib import Path

import numpy as np
import pytest

from multimodars import (
    PyGeometryPair,
    align_manual,
    align_three_point,
    from_array_single,
    from_file_single,
    from_file_singlepair,
    numpy_to_centerline,
    numpy_to_inputdata,
    to_array,
)

FIXTURES = Path(__file__).resolve().parents[1] / "data" / "fixtures"
N_POINTS = 32


def _copy_fixture(name, tmp_path, skip=("diastolic_reference_points.csv",)):
    dst = tmp_path / name
    dst.mkdir()
    for f in (FIXTURES / name).iterdir():
        if f.name not in skip:
            shutil.copy(f, dst / f.name)
    return dst


def _references(frame):
    return [
        (a.group, (a.points[0].x, a.points[0].y, a.points[0].z))
        for a in frame.annotations
        if (a.kind, a.role) == ("reference", "point")
    ]


def _by_original(geom):
    return {f.lumen.original_frame: f for f in geom.frames}


def _single(path, **kwargs):
    geom, _ = from_file_single(
        str(path),
        write_obj=False,
        step_rotation_deg=5.0,
        range_rotation_deg=30.0,
        **kwargs,
    )
    return geom


def test_reference_file_rows_become_annotations_and_default_anchor(tmp_path):
    case = _copy_fixture("ivus_rest", tmp_path)
    (case / "diastolic_reference_points.csv").write_text(
        "frame\tx\ty\tz\tauto\n"
        "347\t3.0\t3.0\t2.27\t1\n"
        "367\t3.2\t3.3\t0.94\n"
        "385\t3.57\t3.67\t24.5\t0\n"
    )
    geom = _single(case)

    # first hand-placed row anchors, every row is a reference annotation on its frame
    assert geom.anchor_frame() == 367
    refs = geom.references()
    assert sorted((f, auto) for f, auto, _ in refs) == [
        (347, True),
        (367, False),
        (385, False),
    ]
    frames = _by_original(geom)
    for original, group in [(347, 1), (367, 0), (385, 0)]:
        assert [g for g, _ in _references(frames[original])] == [group]
    anchored = [f.lumen.original_frame for f in geom.frames if f.reference_point]
    assert anchored == [367]
    rp = frames[367].reference_point
    np.testing.assert_allclose(
        (rp.x, rp.y, rp.z), _references(frames[367])[0][1], atol=1e-9
    )

    assert _single(case, reference_frame=385).anchor_frame() == 385
    with pytest.raises(RuntimeError, match="no reference point on frame 1"):
        _single(case, reference_frame=1)


def test_single_row_reference_file_is_unchanged(tmp_path):
    geom = _single(FIXTURES / "ivus_rest")
    assert geom.anchor_frame() == 385
    refs = geom.references()
    assert [(f, auto) for f, auto, _ in refs] == [(385, False)]
    rp = _by_original(geom)[385].reference_point
    assert rp.point_index == 0
    np.testing.assert_allclose(refs[0][2], (rp.x, rp.y, rp.z), atol=1e-9)


def _lumen_arr(n_frames=6):
    rows = []
    t = np.linspace(0, 2 * np.pi, N_POINTS, endpoint=False)
    for k in range(n_frames):
        x, y = 2.0 * np.cos(t) + 4.5, 1.4 * np.sin(t) + 4.5
        rows += [(10 + k, xi, yi, float(k)) for xi, yi in zip(x, y)]
    return np.array(rows, dtype=float)


REFS = np.array(
    [
        [11, 6.5, 4.5, 1.0, 1],
        [13, 4.5, 5.9, 3.0, 0],
        [10, 6.5, 4.5, 0.0, 0],
    ]
)


def test_numpy_to_inputdata_keeps_every_reference():
    lumen = _lumen_arr()
    inp = numpy_to_inputdata(lumen, REFS, diastole=True)
    assert inp.ref_point.frame_index == 13
    assert [(p.frame_index, auto) for p, auto in inp.references] == [
        (11, True),
        (13, False),
        (10, False),
    ]
    np.testing.assert_allclose(to_array(inp)["references"], REFS)
    np.testing.assert_allclose(to_array(inp)["reference"], REFS[[1], :4])

    chosen = numpy_to_inputdata(lumen, REFS, diastole=True, reference_frame=11)
    assert chosen.ref_point.frame_index == 11
    with pytest.raises(ValueError, match="no reference point on frame 99"):
        numpy_to_inputdata(lumen, REFS, diastole=True, reference_frame=99)

    # a single 4 column row keeps working as before
    single = numpy_to_inputdata(lumen, np.array([10, 6.5, 4.5, 0.0]), diastole=True)
    assert single.ref_point.frame_index == 10
    assert len(single.references) == 1


def _geometry(**kwargs):
    inp = numpy_to_inputdata(_lumen_arr(), REFS, diastole=True)
    geom, _ = from_array_single(
        inp,
        step_rotation_deg=5.0,
        range_rotation_deg=30.0,
        n_points=0,
        write_obj=False,
        **kwargs,
    )
    return geom


def test_set_reference_frame_moves_anchor_with_transformed_frame():
    geom = _geometry().rotate(30.0).translate(1.0, 2.0, 3.0)
    assert geom.anchor_frame() == 13

    moved = geom.set_reference_frame(11)
    assert moved.anchor_frame() == 11
    assert geom.anchor_frame() == 13, "returns a new geometry"
    frames = _by_original(moved)
    assert [f.lumen.original_frame for f in moved.frames if f.reference_point] == [11]
    rp = frames[11].reference_point
    np.testing.assert_allclose(
        (rp.x, rp.y, rp.z), _references(frames[11])[0][1], atol=1e-9
    )
    # nearest lumen point, the index three-point alignment takes as the reference
    lumen = frames[11].lumen.points
    nearest = min(lumen, key=lambda p: (p.x - rp.x) ** 2 + (p.y - rp.y) ** 2)
    assert rp.point_index == nearest.point_index

    with pytest.raises(ValueError, match="frame 12 has no reference annotation"):
        geom.set_reference_frame(12)
    with pytest.raises(ValueError, match="no frame with original frame 99"):
        geom.set_reference_frame(99)

    pair = PyGeometryPair(geom, geom, "pair").set_reference_frame(10)
    assert pair.geom_a.anchor_frame() == 10
    assert pair.geom_b.anchor_frame() == 13


def _z_axis_centerline():
    z = np.linspace(30.0, 0.0, 301)
    return numpy_to_centerline(np.column_stack([np.zeros_like(z), np.zeros_like(z), z]))


def _centroid(geom, original):
    return np.array(_by_original(geom)[original].lumen.centroid)


def test_align_reference_frame_places_the_chosen_frame():
    geom = _geometry()
    target = np.array([0.0, 0.0, 15.0])

    default, _, _ = align_manual(_z_axis_centerline(), geom, 0.0, tuple(target))
    chosen, _, _ = align_manual(
        _z_axis_centerline(), geom, 0.0, tuple(target), reference_frame=10
    )
    np.testing.assert_allclose(_centroid(default, 13), target, atol=1e-6)
    np.testing.assert_allclose(_centroid(chosen, 10), target, atol=1e-6)
    assert chosen.anchor_frame() == 10

    pair = PyGeometryPair(geom, geom, "pair")
    aligned, _, _ = align_manual(
        _z_axis_centerline(), pair, 0.0, tuple(target), reference_frame=11
    )
    np.testing.assert_allclose(_centroid(aligned.geom_a, 11), target, atol=1e-6)
    np.testing.assert_allclose(_centroid(aligned.geom_b, 11), target, atol=1e-6)

    three, _, _ = align_three_point(
        _z_axis_centerline(),
        geom,
        (1.0, 0.0, 15.0),
        (0.0, 0.0, 15.0),
        (0.0, 0.0, 15.0),
        reference_frame=11,
    )
    np.testing.assert_allclose(_centroid(three, 11), target, atol=1e-6)
    assert three.anchor_frame() == 11

    with pytest.raises(RuntimeError, match="no reference annotation"):
        align_manual(_z_axis_centerline(), geom, 0.0, tuple(target), reference_frame=12)


def test_pair_matching_anchors_keep_the_overlap(tmp_path):
    case = _copy_fixture("idealized_geometry", tmp_path)
    # a hand-placed reference on frame 2 comes first, systole is anchored on frame 10
    (case / "diastolic_reference_points.csv").write_text("2,6,4.5,2,0\n10,6,4.5,10,0\n")

    def run(**kwargs):
        pair, _ = from_file_singlepair(
            str(case),
            write_obj=False,
            step_rotation_deg=5.0,
            range_rotation_deg=30.0,
            **kwargs,
        )
        return pair

    mismatched = run()
    matched = run(reference_frames=(10, None))

    assert mismatched.geom_a.anchor_frame() == 2
    assert matched.geom_a.anchor_frame() == 10
    assert len(matched.geom_a.frames) == len(matched.geom_b.frames) == 11
    assert len(mismatched.geom_a.frames) < len(matched.geom_a.frames)
    # the reference that does not anchor still travels with its frame
    assert sorted(f for f, _, _ in matched.geom_a.references()) == [2, 10]
