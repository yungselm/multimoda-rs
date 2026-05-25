from __future__ import annotations

import numpy as np
from scipy.interpolate import splprep, splev

from ..multimodars import (
    PyContour,
    PyContourPoint,
    PyCenterline,
    PyDiscretizedVesselTree,
    discretize_vessel_tree as _discretize_vessel_tree,
)
from .labeling import label_branches as _label_branches


def _fit_bspline_contour(
    contour: PyContour,
    smoothing: float = 0.0,
    degree: int = 3,
) -> PyContour:
    """Return a copy of *contour* whose points lie on a closed B-spline fit.

    Parameters
    ----------
    contour:
        Source contour.
    smoothing:
        Smoothing condition ``s`` for ``scipy.interpolate.splprep``.
        Range: ``[0, ∞)``.

        * ``s = 0``       — exact interpolation; spline passes through every point.
        * ``s ≈ n``       — gentle smoothing (~1 mm² average residual per point).
        * ``s ≈ 5 * n``   — strong smoothing.

          where ``n`` is the number of contour points (e.g. ``n = 100`` →
          start around ``s = 100.0`` and tune from there).
    degree:
        B-spline degree (1-5).  3 (cubic) is the typical choice.
    """
    pts = contour.points
    n = len(pts)
    if n < degree + 1:
        return contour

    xs = np.array([p.x for p in pts], dtype=np.float64)
    ys = np.array([p.y for p in pts], dtype=np.float64)
    zs = np.array([p.z for p in pts], dtype=np.float64)

    try:
        tck, _ = splprep([xs, ys, zs], s=smoothing, k=degree, per=True)
    except Exception:
        return contour

    u_new = np.linspace(0.0, 1.0, n, endpoint=False)
    xs_new, ys_new, zs_new = splev(u_new, tck)

    centroid = (
        float(np.mean(xs_new)),
        float(np.mean(ys_new)),
        float(np.mean(zs_new)),
    )
    new_points = [
        PyContourPoint(
            pts[i].frame_index,
            pts[i].point_index,
            float(xs_new[i]),
            float(ys_new[i]),
            float(zs_new[i]),
            pts[i].aortic,
        )
        for i in range(n)
    ]

    return PyContour(
        contour.id,
        contour.original_frame,
        new_points,
        centroid,
        contour.aortic_thickness,
        contour.pulmonary_thickness,
        contour.kind,
    )


def _replace_contours_with_bsplines(
    tree: PyDiscretizedVesselTree,
    smoothing: float = 0.0,
    degree: int = 3,
) -> PyDiscretizedVesselTree:
    """Replace every contour in *tree* with a closed B-spline fit (in-place)."""

    def fit(c: PyContour) -> PyContour:
        return _fit_bspline_contour(c, smoothing, degree)

    tree.discretized_aorta = [fit(c) for c in tree.discretized_aorta]
    tree.discretized_rca_main = [fit(c) for c in tree.discretized_rca_main]
    tree.discretized_lca_main = [fit(c) for c in tree.discretized_lca_main]
    tree.rca_branches = [[fit(c) for c in branch] for branch in tree.rca_branches]
    tree.lca_branches = [[fit(c) for c in branch] for branch in tree.lca_branches]
    return tree


def _extract_side_branches(results_dict: dict, prefix: str) -> list[list[tuple]]:
    """Return ``[points_side_1, points_side_2, ...]`` from *results_dict*."""
    branches = []
    i = 1
    while True:
        key = f"{prefix}_side_{i}"
        if key not in results_dict:
            break
        branches.append(results_dict[key])
        i += 1
    return branches


def discretize_vessel_tree(
    ao_cl: PyCenterline,
    rca_cl: PyCenterline,
    lca_cl: PyCenterline,
    results_dict: dict,
    branch_id_rca: int = 0,
    branch_id_lca: int = 0,
    step_size: float = 1.0,
    n_points: int = 100,
    b_spline: bool = False,
    bspline_smoothing: float = 100.0,
    bspline_degree: int = 3,
    control_plot: bool = False,
) -> PyDiscretizedVesselTree:
    """Discretize a coronary vessel tree, optionally smoothing contours with B-splines.

    Expects *results_dict* to already contain labelled branch point keys
    (``aorta_points``, ``rca_points_main``, ``rca_points_side_1``, …,
    ``lca_points_main``, ``lca_points_side_1``, …).  Use
    :func:`prepare_and_discretize` if you also need branch labelling to run
    automatically.

    Parameters
    ----------
    ao_cl, rca_cl, lca_cl:
        Aortic, RCA, and LCA centerlines (branches already computed and
        labelled).
    results_dict:
        Dictionary produced by :func:`~multimodars.label_branches` containing
        keys ``aorta_points``, ``rca_points_main``, ``lca_points_main``, and
        any ``rca_points_side_N`` / ``lca_points_side_N`` entries.
    branch_id_rca, branch_id_lca:
        Main-vessel branch IDs (almost always ``0``).
    step_size:
        Arc-length distance between consecutive cross-sections in mm.
    n_points:
        Number of evenly-spaced points per output contour.
    b_spline:
        When ``True``, all contours are replaced with closed B-spline fits
        before reference points are computed.
    control_plot:
        When ``True``, open an interactive Plotly 3-D visualisation of the
        finished tree (calls
        :func:`~multimodars.ccta.debug_plots.plot_vessel_tree`).
    bspline_smoothing:
        Smoothing condition ``s`` for ``scipy.interpolate.splprep``.
        Range: ``[0, ∞)``.

        * ``s = 0``            — exact interpolation.
        * ``s ≈ n_points``     — gentle smoothing (~1 mm² average residual per point).
        * ``s ≈ 5 * n_points`` — strong smoothing.

          Tune empirically; the right value depends on how irregular the raw
          contours are.
    bspline_degree:
        B-spline polynomial degree (1-5).  Default is cubic (3).

    Returns
    -------
    PyDiscretizedVesselTree
        Fully populated vessel tree including orientation reference triplets.
    """
    points_ao = results_dict["aorta_points"]
    points_rca_main = results_dict["rca_points_main"]
    points_lca_main = results_dict["lca_points_main"]
    side_rca = _extract_side_branches(results_dict, "rca_points")
    side_lca = _extract_side_branches(results_dict, "lca_points")

    if b_spline:
        tree = _discretize_vessel_tree(
            ao_cl,
            rca_cl,
            lca_cl,
            points_ao,
            points_rca_main,
            points_lca_main,
            side_rca,
            side_lca,
            branch_id_rca=branch_id_rca,
            branch_id_lca=branch_id_lca,
            step_size=step_size,
            n_points=n_points,
            calculate_ref_pts=False,
        )
        tree = _replace_contours_with_bsplines(tree, bspline_smoothing, bspline_degree)
        tree.calculate_ref_pts()
    else:
        tree = _discretize_vessel_tree(
            ao_cl,
            rca_cl,
            lca_cl,
            points_ao,
            points_rca_main,
            points_lca_main,
            side_rca,
            side_lca,
            branch_id_rca=branch_id_rca,
            branch_id_lca=branch_id_lca,
            step_size=step_size,
            n_points=n_points,
            calculate_ref_pts=True,
        )

    if control_plot:
        from .debug_plots import plot_vessel_tree

        plot_vessel_tree(tree)

    return tree


def prepare_centerlines(
    rca_cl: PyCenterline,
    lca_cl: PyCenterline,
    results_dict: dict,
    branch_sigma: float = 2.0,
    control_plot: bool = False,
) -> tuple[PyCenterline, PyCenterline, dict]:
    """Compute branches, validate, and label both coronary centerlines.

    This is the standard preparation step before :func:`discretize_vessel_tree`.
    It runs the non-interactive part of centerline setup automatically:

    1. ``rca_cl.calculate_branches(branch_sigma)`` + ``check_centerline()``
    2. ``lca_cl.calculate_branches(branch_sigma)`` + ``check_centerline()``
    3. :func:`~multimodars.label_branches` for RCA, then LCA

    .. note::
        Manual edits — ``find_sharp_angles``, ``split_branch``,
        ``merge_branches`` — cannot be automated.  If your data needs them,
        call those methods on the returned centerlines before passing them to
        :func:`discretize_vessel_tree`.

    Parameters
    ----------
    rca_cl, lca_cl:
        Raw centerlines as returned by :func:`~multimodars.numpy_to_centerline`
        or :func:`~multimodars.label_geometry`.
    results_dict:
        Dictionary produced by :func:`~multimodars.label_geometry`.
    branch_sigma:
        Smoothing sigma (mm) passed to ``calculate_branches`` for both vessels.
    control_plot:
        When ``True``, open an interactive Plotly 3-D visualisation showing
        centerline points coloured by branch ID and the labelled surface-mesh
        points, so you can verify assignments before discretizing (calls
        :func:`~multimodars.ccta.debug_plots.plot_centerline_branches`).

    Returns
    -------
    rca_cl : PyCenterline
        RCA centerline with branches computed, validated, and labelled.
    lca_cl : PyCenterline
        LCA centerline with branches computed, validated, and labelled.
    results_dict : dict
        Updated dictionary with ``rca_points_main``, ``rca_points_side_N``,
        ``lca_points_main``, and ``lca_points_side_N`` keys added.
    """
    rca_cl = rca_cl.calculate_branches(branch_sigma)
    rca_cl = rca_cl.check_centerline()

    lca_cl = lca_cl.calculate_branches(branch_sigma)
    lca_cl = lca_cl.check_centerline()

    results_dict = _label_branches(rca_cl, results_dict)
    results_dict = _label_branches(lca_cl, results_dict, results_key="lca_points")

    if control_plot:
        from .debug_plots import plot_centerline_branches

        plot_centerline_branches(rca_cl, lca_cl, results_dict)

    return rca_cl, lca_cl, results_dict


def find_sharp_angles(
    cl: PyCenterline,
    branch_id: int,
    cos_threshold: float = 0.0,
    control_plot: bool = False,
) -> list[int]:
    """Find sharp angles in a centerline branch and optionally plot them.

    A thin wrapper around ``cl.find_sharp_angles`` that adds an optional
    debug visualisation where each flagged position is shown in a distinct
    colour so they can be counted and identified before deciding whether to
    call ``split_branch`` / ``merge_branches``.

    Parameters
    ----------
    cl:
        Centerline after ``calculate_branches`` (and optionally
        ``check_centerline``).
    branch_id:
        Branch to inspect (0 = main vessel).
    cos_threshold:
        Cosine above which an angle is considered sharp.
        Use ``0.0`` for < 90°, ``0.5`` for < 60°, ``0.866`` for < 30°.
    control_plot:
        When ``True`` opens an interactive 3-D scene with each sharp-angle
        position highlighted in a distinct colour.

    Returns
    -------
    list[int]
        0-indexed positions within the branch (suitable for ``split_branch``).
    """
    positions = cl.find_sharp_angles(branch_id, cos_threshold)
    print(
        f"Branch {branch_id}: {len(positions)} sharp angle(s) at positions {positions}"
    )
    if control_plot:
        from .debug_plots import plot_sharp_angles

        plot_sharp_angles(cl, branch_id, positions)
    return positions
