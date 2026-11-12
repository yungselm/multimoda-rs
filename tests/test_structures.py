"""In-wall structures: drawn contours become jittered clouds that move with their frame."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from multimodars import (
    PyContourPoint,
    PyStructure,
    from_array_single,
    from_file_single,
    numpy_to_geometry,
    numpy_to_inputdata,
    structure_to_trimesh,
    to_array,
)

N_FRAMES = 6
N_POINTS = 32
COPIES = 4  # default jittered copies per contour point


def _lumen_arr():
    """Elliptic lumen 1 mm apart, each frame shifted and twisted so alignment has work to do."""
    rows = []
    t = np.linspace(0, 2 * np.pi, N_POINTS, endpoint=False)
    for k in range(N_FRAMES):
        a = np.deg2rad(8.0 * k)
        x, y = 2.0 * np.cos(t), 1.4 * np.sin(t)
        xr = x * np.cos(a) - y * np.sin(a) + 4.5 + 0.2 * k
        yr = x * np.sin(a) + y * np.cos(a) + 4.5 - 0.1 * k
        rows += [(10 + k, xi, yi, float(k)) for xi, yi in zip(xr, yr)]
    return np.array(rows, dtype=float)


def _on_lumen(lumen, frame, idx, contour=0, closed=1):
    sel = lumen[lumen[:, 0] == frame][idx]
    n = len(sel)
    return np.column_stack([sel, np.full(n, contour), np.full(n, closed)])


def _make_input(structures=None):
    lumen = _lumen_arr()
    if structures is None:
        structures = {
            "calcium": np.vstack(
                [
                    _on_lumen(lumen, 11, [0, 3, 6, 9]),
                    _on_lumen(lumen, 11, [16, 18, 20], contour=1),
                    _on_lumen(lumen, 13, [2, 4, 8]),
                ]
            ),
            "fibrous_cap": _on_lumen(lumen, 12, [7, 20, 25])[:, :4],
        }
    inp = numpy_to_inputdata(
        lumen_arr=lumen,
        reference_points=np.array([10, 6.5, 4.5, 0.0]),
        diastole=True,
        label="st",
        structures=structures,
    )
    return lumen, inp


def _align(inp, smooth=False):
    geom, _ = from_array_single(
        inp, step_rotation_deg=1.0, range_rotation_deg=30.0, n_points=0, smooth=smooth
    )
    return geom


def test_pystructure_kind_and_transforms():
    pts = [
        PyContourPoint(frame_index=0, point_index=i, x=x, y=0.0, z=1.0, aortic=False)
        for i, x in enumerate([1.0, 2.0])
    ]
    s = PyStructure("calcium", pts)
    assert (s.kind, s.closed, len(s), s.is_custom) == ("calcification", True, 2, False)
    assert PyStructure("branch", closed=False).kind == "sidebranch"
    assert PyStructure("fibrous_cap").is_custom
    assert "lipid" in PyStructure.known_kinds()

    r = s.rotate(90.0, center=(0.0, 0.0))
    assert (r.points[0].x, r.points[0].y) == pytest.approx((0.0, 1.0))
    t = s.translate(1.0, 2.0, 3.0)
    assert (t.points[1].x, t.points[1].y, t.points[1].z) == (3.0, 2.0, 4.0)


def test_input_groups_rows_by_frame_and_contour():
    _, inp = _make_input()
    summary = [
        (s.kind, s.points[0].frame_index, s.closed, len(s)) for s in inp.structures
    ]
    assert summary == [
        ("calcification", 11, True, 4),
        ("calcification", 11, True, 3),
        ("calcification", 13, True, 3),
        ("fibrous_cap", 12, True, 3),
    ]


def test_clouds_are_jittered_and_move_with_their_frame():
    _, inp = _make_input()
    geom = _align(inp)

    by_original = {f.lumen.original_frame: f for f in geom.frames}
    sizes = {o: [len(s) for s in f.structures] for o, f in by_original.items()}
    assert sizes[11] == [4 * (COPIES + 1), 3 * (COPIES + 1)]
    assert sizes[12] == [3 * (COPIES + 1)]
    assert sizes[13] == [3 * (COPIES + 1)]
    assert sizes[10] == []

    for frame in by_original.values():
        lumen_xyz = np.array([(p.x, p.y, p.z) for p in frame.lumen.points])
        for s in frame.structures:
            n = len(s) // (COPIES + 1)
            # the drawn points stay on the lumen points they were drawn on
            for p in s.points[:n]:
                d = np.linalg.norm(lumen_xyz - (p.x, p.y, p.z), axis=1).min()
                assert d < 1e-6
            # the jitter spreads over half the 1 mm frame spacing in z
            z = np.array([p.z for p in s.points])
            assert np.all(np.abs(z - frame.centroid[2]) <= 0.5 + 1e-9)
            assert np.ptp(z) > 0.1

    # deterministic
    again = to_array(_align(inp))["structures"]
    for kind, arr in to_array(geom)["structures"].items():
        np.testing.assert_array_equal(arr, again[kind])


def test_open_in_wall_contour_is_rejected():
    lumen = _lumen_arr()
    _, inp = _make_input({"lipid": _on_lumen(lumen, 11, [0, 3, 6], closed=0)})
    with pytest.raises(RuntimeError, match="must be closed"):
        _align(inp)


def test_structures_round_trip_through_numpy_geometry():
    _, inp = _make_input()
    arrays = to_array(_align(inp))
    assert set(arrays["structures"]) == {"calcification", "fibrous_cap"}
    assert arrays["structures"]["calcification"].shape[1] == 6

    rebuilt = numpy_to_geometry(
        lumen_arr=arrays["lumen"],
        reference_arr=arrays["reference"],
        structures=arrays["structures"],
    )
    again = to_array(rebuilt)["structures"]

    # the rebuilt frames run by frame index, compare rows in one order
    def rows(a):
        return a[np.lexsort(a.T[::-1])]

    for kind, arr in arrays["structures"].items():
        np.testing.assert_allclose(rows(again[kind]), rows(arr))


def _ring(cx, z, frame, r=1.0, n=12):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack(
        [np.full(n, frame), cx + r * np.cos(t), r * np.sin(t), np.full(n, z)]
    )


def test_structure_to_trimesh_one_hull_per_cluster():
    lumen = np.vstack([_ring(0.0, float(k), k, r=3.0) for k in range(3)])
    # one cluster on frames 0 and 1, one far away on frames 1 and 2
    calcium = np.vstack(
        [
            _ring(0.0, 0.0, 0),
            _ring(0.0, 1.0, 1),
            _ring(10.0, 1.0, 1),
            _ring(10.0, 2.0, 2),
        ]
    )
    geom = numpy_to_geometry(
        lumen_arr=lumen,
        reference_arr=np.array([0, 3.0, 0.0, 0.0]),
        structures={"calcification": calcium},
    )

    meshes = structure_to_trimesh(geom, "calcium")

    assert len(meshes) == 2
    assert all(m.is_watertight and m.volume > 0 for m in meshes)
    # unlinked, every ring is flat and gives no mesh
    assert structure_to_trimesh(geom, "calcium", link_distance=0.5) == []
    assert structure_to_trimesh(geom, "lipid") == []
    with pytest.raises(ValueError, match="side branches"):
        structure_to_trimesh(geom, "sidebranch")


def test_type_contour_files_are_read(tmp_path: Path):
    for f in Path("data/fixtures/ivus_rest").iterdir():
        shutil.copy(f, tmp_path / f.name)
    # a closed calcium contour with contour and closed columns on frame 385
    t = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    rows = [(385, 3.5 + np.cos(a), 3.6 + np.sin(a), 24.5, 0, 1) for a in t]
    np.savetxt(tmp_path / "calcium_diastolic_contours.csv", rows, delimiter="\t")

    geom, _ = from_file_single(str(tmp_path), write_obj=False)

    frame = next(f for f in geom.frames if f.lumen.original_frame == 385)
    assert [(s.kind, s.closed, len(s)) for s in frame.structures] == [
        ("calcification", True, 8 * (COPIES + 1))
    ]


def test_open_branch_arc_is_revolved():
    lumen = _lumen_arr()
    on_frame = lumen[lumen[:, 0] == 12]
    center = on_frame[:, 1:3].mean(axis=0)
    # ends on lumen points 6 and 10, the middle bulges out of the lumen
    arc = on_frame[[6, 7, 8, 9, 10]].copy()
    arc[1:4, 1:3] = center + 1.4 * (arc[1:4, 1:3] - center)
    rows = np.column_stack([arc, np.zeros(5), np.zeros(5)])
    _, inp = _make_input({"branch": rows})

    geom = _align(inp)

    frame = next(f for f in geom.frames if f.lumen.original_frame == 12)
    (branch,) = frame.structures
    assert (branch.kind, branch.closed, len(branch)) == ("sidebranch", False, 5 * 12)
    # the drawn arc is kept, its ends stay on the lumen
    lumen_xyz = np.array([(p.x, p.y, p.z) for p in frame.lumen.points])
    for p in (branch.points[0], branch.points[4]):
        assert np.linalg.norm(lumen_xyz - (p.x, p.y, p.z), axis=1).min() < 1e-6
    # turning the arc lifts it out of the frame plane, at most by its own size
    z = np.array([p.z for p in branch.points]) - frame.centroid[2]
    width = np.ptp(arc[:, 1:3], axis=0).max()
    assert 0.3 < np.abs(z).max() <= width
    assert z.min() < -0.3 < 0.3 < z.max()
    with pytest.raises(ValueError, match="side branches"):
        structure_to_trimesh(geom, "branch")
