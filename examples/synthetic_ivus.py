"""Synthetic intravascular pullback cut from a discretized CCTA branch.

For debugging the alignment / anomalous labelling / stitching around a side
branch: the pullback is the CCTA vessel itself, so after alignment its frames
should land back on the CCTA contours they were made from.  The reference point
sits on a frame in the middle of the pullback, where a side branch leaves,
instead of at the ostium, so it pairs with that bifurcation's reference triplet
(``tree.rca[0].bifurcations[child]``) rather than the ostium triplet.

Writes ``{phase}_contours.csv`` and ``{phase}_reference_points.csv`` (tab
separated ``frame x y z``, as the readers of :func:`multimodars.from_file_single`
expect) for a diastolic and a systolic phase.  Frame ids increase towards the
ostium and ``z`` is the pullback position, as in the example IVUS data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _frames_basis(centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tangent and an in-plane basis ``(u, v)`` per slice, transported along the
    vessel so the image does not twist from frame to frame.

    ``v = u x t`` (``t`` pointing distally) is the image handedness the loader
    and alignment expect: with ``v = t x u`` the aligned pullback comes out
    mirrored (checked with an asymmetric bump on the contours, see
    ``tests/test_ccta.py::TestSyntheticPullback``).
    """
    tangents = np.gradient(centroids, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    seed = np.array([0.0, 0.0, 1.0])
    if abs(seed @ tangents[0]) > 0.9:
        seed = np.array([1.0, 0.0, 0.0])
    us, vs = [], []
    u = seed
    for t in tangents:
        u = u - t * (u @ t)
        u /= np.linalg.norm(u)
        us.append(u)
        vs.append(np.cross(u, t))
    return tangents, np.array(us), np.array(vs)


def _resample_ring(ring: np.ndarray, n: int) -> np.ndarray:
    """*n* points evenly spaced along the closed polyline *ring*."""
    loop = np.vstack([ring, ring[:1]])
    s = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(loop, axis=0), axis=1))]
    )
    targets = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.column_stack([np.interp(targets, s, loop[:, k]) for k in range(3)])


def write_synthetic_pullback(
    contours,
    ref_slice: int,
    ref_target: tuple[float, float, float],
    out_dir: str | Path,
    n_proximal: int = 20,
    n_distal: int = 20,
    image_center: tuple[float, float] = (4.5, 4.5),
    systolic_scale: float = 1.0,
    n_points: int = 500,
) -> Path:
    """Write a pullback made of ``contours[ref_slice - n_proximal : ref_slice + n_distal + 1]``.

    Parameters
    ----------
    contours : list of PyContour
        Contours of one discretized branch, proximal to distal
        (``tree.rca[0].contours``).
    ref_slice : int
        Contour that gets the reference point, e.g. the parent contour of a
        bifurcation (``tree.rca[child].parent[1]``).
    ref_target : tuple of float
        The reference point is the lumen point of that contour nearest to this,
        e.g. the bifurcation's ``main_ref`` (the side branch's start).
    out_dir : str or Path
        Directory to write into, created if needed.
    n_proximal, n_distal : int
        Contours taken proximal and distal of *ref_slice*.
    image_center : tuple of float
        Where the vessel centre lands in the image plane, in mm.
    systolic_scale : float
        Radial scale of the systolic phase relative to the diastolic one.
    n_points : int
        Points per contour, as dense as real IVUS contours. Coarse contours
        make the loader's frame-to-frame rotation unreliable.

    Returns
    -------
    Path
        *out_dir*.
    """
    lo = max(0, ref_slice - n_proximal)
    hi = min(len(contours), ref_slice + n_distal + 1)
    selected = contours[lo:hi]
    rings = [
        _resample_ring(np.asarray(c.points_as_tuples(), dtype=np.float64), n_points)
        for c in selected
    ]
    centroids = np.array([r.mean(axis=0) for r in rings])
    _, us, vs = _frames_basis(centroids)

    # Pullback position: arc length from the distal end, so z and the frame id
    # grow towards the ostium.
    steps = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
    from_proximal = np.concatenate([[0.0], np.cumsum(steps)])
    z = from_proximal[-1] - from_proximal
    n = len(selected)
    frame_id = [n - 1 - i for i in range(n)]

    ref_i = ref_slice - lo
    ref_ring = rings[ref_i]
    ref_point = ref_ring[
        np.argmin(np.linalg.norm(ref_ring - np.asarray(ref_target), axis=1))
    ]

    def to_image(i: int, p: np.ndarray, scale: float) -> tuple[float, float, float]:
        d = p - centroids[i]
        return (
            image_center[0] + scale * float(d @ us[i]),
            image_center[1] + scale * float(d @ vs[i]),
            float(z[i]),
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for phase, scale in (("diastolic", 1.0), ("systolic", systolic_scale)):
        with open(out / f"{phase}_contours.csv", "w") as f:
            for i, ring in enumerate(rings):
                for p in ring:
                    x, y, zz = to_image(i, p, scale)
                    f.write(f"{frame_id[i]}\t{x}\t{y}\t{zz}\n")
        x, y, zz = to_image(ref_i, ref_point, scale)
        with open(out / f"{phase}_reference_points.csv", "w") as f:
            f.write(f"{frame_id[ref_i]}\t{x}\t{y}\t{zz}\n")
    return out
