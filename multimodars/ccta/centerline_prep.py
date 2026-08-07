from __future__ import annotations

from pathlib import Path
import numpy as np

from ..multimodars import read_centerline_vtp, PyCenterline
from .._converters import numpy_to_centerline


def load_centerline(
    source: PyCenterline | Path | str | np.ndarray, name: str
) -> PyCenterline:
    """Load a centerline from any supported source.

    Parameters
    ----------
    source : PyCenterline, Path, str, or numpy.ndarray
        A ``.vtp`` file path, a CSV file path (comma-delimited, columns
        x, y, z, ...), an existing ``PyCenterline`` (returned as-is), or an
        ``(N, 3+)`` array of points.
    name : str
        Label used in log output (e.g. ``"Aorta"``, ``"RCA"``, ``"LCA"``).

    Returns
    -------
    PyCenterline
        The loaded centerline, unprepared - pass it to
        :func:`prepare_centerline` next.
    """
    if isinstance(source, PyCenterline):
        cl = source
        print(f"Using provided {name} centerline: {len(cl.points)} points")
    elif isinstance(source, np.ndarray):
        cl = numpy_to_centerline(source)
        print(f"Using provided {name} centerline: {len(cl.points)} points")
    elif str(source).lower().endswith(".vtp"):
        try:
            cl = read_centerline_vtp(str(source))
            print(f"Loaded {name} centerline from VTP: {len(cl.points)} points")
        except Exception as e:
            print(f"Error reading {name} centerline from {source}: {e}")
            raise
    else:
        try:
            cl_raw = np.genfromtxt(source, delimiter=",")
            cl = numpy_to_centerline(cl_raw)
            print(f"Loaded {name} centerline: {len(cl.points)} points")
        except Exception as e:
            print(f"Error reading {name} centerline from {source}: {e}")
            raise
    return cl


def prepare_centerline(
    centerline: PyCenterline,
    ref_centerline: PyCenterline | None = None,
    spacing_mm: float | None = None,
    branch_spacing_tolerance: float = 2.0,
    rm_start_mm: float = 0.0,
    smooth_sigma: float = 2.5,
) -> PyCenterline:
    """Run the standard branch/order/smooth prep pipeline on a centerline.

    *ref_centerline* doubles as the "is this a coronary?" signal: the aorta has
    no upstream reference to orient against, while a coronary (RCA/LCA) orients
    towards the aorta's branch 0. Applies, in order:

    1. ``calculate_branches(branch_spacing_tolerance)`` - only when
       *ref_centerline* is given (i.e. *centerline* is a coronary) and
       *centerline* does not already carry branch structure (e.g. it came from
       a ``.vtp`` file, which reports its branches directly). The aorta
       (*ref_centerline* is ``None``) never needs branch detection - it has no
       side branches to find. Skipping this for an already-branched centerline
       avoids re-running detection on clean data, which risks its
       artefact-discard logic truncating a genuine end of the centerline.
    2. ``remove_branch_overlap()`` - trims the run-alongside-main-branch prefix
       some centerline export formats (e.g. VTP) attach to every side branch.
       A no-op for a single-branch centerline.
    3. ``trim_start(rm_start_mm)`` - only if ``rm_start_mm > 0``.
    4. ``resample(spacing_mm)`` - only if ``spacing_mm`` is given.
    5. ``orient_to_reference(ref_centerline)`` if *ref_centerline* is given,
       otherwise ``orient_by_max_z()`` - normalise branch ordering.
    6. ``smooth(smooth_sigma)`` - only if ``smooth_sigma > 0``.

    Parameters
    ----------
    centerline : PyCenterline
        Centerline to prepare, e.g. the output of :func:`load_centerline`.
    ref_centerline : PyCenterline, optional
        Reference centerline to orient towards (e.g. the aorta, for a coronary
        centerline) - see ``PyCenterline.orient_to_reference``. ``None``
        (default) means *centerline* has no reference (e.g. it is the aorta
        itself): falls back to ``PyCenterline.orient_by_max_z`` and skips
        branch detection.
    spacing_mm : float, optional
        Target arc-length spacing in mm passed to ``resample``. ``None``
        (default) skips resampling.
    branch_spacing_tolerance : float, optional
        Passed to ``calculate_branches`` when branch extraction is needed.
        Default ``2.0``.
    rm_start_mm : float, optional
        Arc-length in mm to trim from the start of branch 0 (e.g. the aortic
        inlet region). Default ``0.0`` (no trim).
    smooth_sigma : float, optional
        Half-width of the Gaussian smoothing kernel in number of centerline
        points. Default ``2.5``. Set to ``0.0`` to skip smoothing.

    Returns
    -------
    PyCenterline
        The prepared centerline.
    """
    if ref_centerline is not None and len(centerline.branch_start_indices) <= 1:
        cl = centerline.calculate_branches(branch_spacing_tolerance)
    else:
        cl = centerline

    cl = cl.remove_branch_overlap()

    if rm_start_mm > 0:
        cl = cl.trim_start(rm_start_mm)

    if spacing_mm:
        cl = cl.resample(spacing_mm)

    if ref_centerline is not None:
        cl = cl.orient_to_reference(ref_centerline)
    else:
        cl = cl.orient_by_max_z()

    if smooth_sigma > 0:
        cl = cl.smooth(smooth_sigma)

    return cl
