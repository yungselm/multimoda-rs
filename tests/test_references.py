"""Several reference points per geometry and the master reference."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from multimodars import (
    from_file_singlepair,
    numpy_to_geometry,
    numpy_to_inputdata,
)

FIXTURE = Path("data/fixtures/ivus_rest")
# diastolic frames 347, 367, 385 and systolic 279, 300, 319, the proximal end is 385 / 319
DIA_REFS = "347\t3.0\t3.0\t22.0\t1\n385\t3.57\t3.67\t24.5\t0\n"
SYS_REFS = "279\t3.0\t3.0\t19.0\n319\t3.53\t3.46\t20.8\n"


@pytest.fixture
def rest_dir(tmp_path: Path) -> Path:
    for f in FIXTURE.iterdir():
        shutil.copy(f, tmp_path / f.name)
    (tmp_path / "diastolic_reference_points.csv").write_text(DIA_REFS)
    (tmp_path / "systolic_reference_points.csv").write_text(SYS_REFS)
    return tmp_path


def _pair(path: Path, **kwargs):
    pair, _ = from_file_singlepair(
        str(path), labels=["dia", "sys"], write_obj=False, **kwargs
    )
    return pair


def _master_idx(geom) -> int:
    return next(
        i
        for i, f in enumerate(geom.frames)
        if f.reference_point is not None
        and f.reference_point.frame_index == geom.master_reference
    )


def test_every_reference_is_stored_and_the_master_is_proximal(rest_dir: Path):
    pair = _pair(rest_dir)

    assert sorted(pair.geom_a.reference_frames()) == [347, 385]
    assert sorted(pair.geom_b.reference_frames()) == [279, 319]
    # the most proximal reference is the default master
    assert pair.geom_a.master_reference == 385
    assert pair.geom_b.master_reference == 319
    # before resampling every reference sits on its own original frame, resampling can
    # move one onto the nearest blended frame
    raw = _pair(rest_dir, postprocessing=False)
    for geom in (raw.geom_a, raw.geom_b):
        refs = [f for f in geom.frames if f.reference_point is not None]
        assert len(refs) == 2
        assert all(
            f.reference_point.frame_index == f.lumen.original_frame for f in refs
        )


def test_master_references_parameter(rest_dir: Path):
    pair = _pair(rest_dir, master_references=[347, None])
    assert pair.geom_a.master_reference == 347
    assert pair.geom_b.master_reference == 319

    with pytest.raises(RuntimeError, match="367"):
        _pair(rest_dir, master_references=[367, None])
    with pytest.raises(RuntimeError, match="one entry per geometry"):
        _pair(rest_dir, master_references=[347])


def test_overlap_is_matched_on_the_masters(rest_dir: Path):
    pair = _pair(rest_dir, master_references=[347, None])

    (start_a, stop_a), (start_b, stop_b) = pair.overlap()
    assert stop_a - start_a == stop_b - start_b
    assert _master_idx(pair.geom_a) - start_a == _master_idx(pair.geom_b) - start_b

    facing = pair.overlapping()
    assert len(facing.geom_a.frames) == stop_a - start_a
    assert len(facing.geom_b.frames) == stop_b - start_b
    # no frame is trimmed from the pair itself
    assert len(pair.geom_a.frames) >= len(facing.geom_a.frames)


def test_set_master_references_realigns_the_pair(rest_dir: Path):
    pair = _pair(rest_dir)
    n_a, n_b = len(pair.geom_a.frames), len(pair.geom_b.frames)

    pair.set_master_references(master_a=347, master_b=279, postprocessing=False)

    assert (pair.geom_a.master_reference, pair.geom_b.master_reference) == (347, 279)
    assert (len(pair.geom_a.frames), len(pair.geom_b.frames)) == (n_a, n_b)
    ca = pair.geom_a.frames[_master_idx(pair.geom_a)].centroid
    cb = pair.geom_b.frames[_master_idx(pair.geom_b)].centroid
    assert ca == pytest.approx(cb, abs=1e-6)

    with pytest.raises(ValueError, match="367"):
        pair.set_master_references(master_a=367)


def _square_lumen(frames: list[int]) -> np.ndarray:
    corners = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]
    return np.array(
        [[f, x, y, float(f)] for f in frames for x, y in corners], dtype=float
    )


def test_numpy_to_geometry_puts_each_reference_on_its_frame():
    refs = np.array(
        [[1, 1.0, 0.0, 1.0, 0], [3, 0.0, 1.0, 3.0, 1], [9, 0.0, 0.0, 9.0, 0]]
    )

    with pytest.warns(UserWarning, match=r"\[9\]"):
        geom = numpy_to_geometry(
            _square_lumen([0, 1, 2, 3]), reference_arr=refs, master_reference=3
        )

    assert geom.reference_frames() == [1, 3]
    assert [f.reference_point is not None for f in geom.frames] == [
        False,
        True,
        False,
        True,
    ]
    assert geom.master_reference == 3

    geom.master_reference = 1
    assert geom.master_reference == 1
    with pytest.raises(ValueError, match="frame 2"):
        geom.master_reference = 2
    assert geom.master_reference == 1


def test_numpy_to_inputdata_reads_every_reference_row():
    refs = np.array([[1, 1.0, 0.0, 1.0, 0], [3, 0.0, 1.0, 3.0, 1]])
    inp = numpy_to_inputdata(
        _square_lumen([0, 1, 2, 3]), refs, diastole=True, master_reference=3
    )
    assert [p.frame_index for p in inp.reference_points] == [1, 3]
    assert inp.master_reference == 3

    with pytest.raises(ValueError, match="one reference point per frame"):
        numpy_to_inputdata(_square_lumen([0, 1]), np.array([[1, 0, 0, 0]] * 2), True)
