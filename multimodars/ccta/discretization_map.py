from __future__ import annotations

from ..multimodars import (
    PyCenterline,
    PyDiscretizedVesselTree,
)
from ..multimodars import (
    discretize_vessel_tree as _discretize_vessel_tree,
)
from .labeling import label_branches as _label_branches


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
    control_plot: bool = False,
) -> PyDiscretizedVesselTree:
    """Discretize a coronary vessel tree into cross-sectional contours.

    Expects *results_dict* to already contain the labelled ``mesh`` and labelled
    branch point keys (``aorta_points``, ``rca_points_main``, ``rca_points_side_1``, …,
    ``lca_points_main``, ``lca_points_side_1``, …). Each vessel is cut with planes
    perpendicular to its centerline, using only the mesh faces of its own region.  Use
    :func:`label_branches_pair` first to add the branch labels.

    ``ao_cl``, ``rca_cl`` and ``lca_cl`` must already be smoothed, resampled and oriented
    (e.g. via :func:`load_centerline` and :func:`prepare_centerline`).

    Parameters
    ----------
    ao_cl, rca_cl, lca_cl:
        Aortic, RCA, and LCA centerlines (branches already computed and
        labelled), already prepared.
    results_dict:
        Dictionary produced by :func:`~multimodars.label_branches_pair` containing
        keys ``mesh``, ``aorta_points``, ``rca_points_main``, ``lca_points_main``, and
        any ``rca_points_side_N`` / ``lca_points_side_N`` entries.
    branch_id_rca, branch_id_lca:
        Main-vessel branch IDs (almost always ``0``).
    step_size:
        Arc-length distance between consecutive cross-sections in mm.
    n_points:
        Number of evenly-spaced points per output contour.
    control_plot:
        When ``True``, open an interactive Plotly 3-D visualisation of the
        finished tree (calls
        :func:`~multimodars.ccta.debug_plots.plot_vessel_tree`).

    Returns
    -------
    PyDiscretizedVesselTree
        Fully populated vessel tree including orientation reference triplets.

    Raises
    ------
    ValueError
        If the aorta or a main vessel cannot be discretized (see :func:`discretize_vessel`).
        A failing side branch is skipped with a warning and left empty.
    """
    points_ao = results_dict["aorta_points"] + results_dict["rca_removed_points"]
    points_rca_main = results_dict["rca_points_main"]
    points_lca_main = results_dict["lca_points_main"]
    side_rca = _extract_side_branches(results_dict, "rca_points")
    side_lca = _extract_side_branches(results_dict, "lca_points")
    mesh = results_dict["mesh"]

    tree = _discretize_vessel_tree(
        ao_cl,
        rca_cl,
        lca_cl,
        [tuple(v) for v in mesh.vertices.tolist()],
        mesh.faces.tolist(),
        points_ao,
        points_rca_main,
        points_lca_main,
        side_rca,
        side_lca,
        branch_id_rca=branch_id_rca,
        branch_id_lca=branch_id_lca,
        step_size=step_size,
        n_points=n_points,
    )

    if control_plot:
        from .debug_plots import plot_vessel_tree

        plot_vessel_tree(tree)

    return tree


def label_branches_pair(
    rca_cl: PyCenterline,
    lca_cl: PyCenterline,
    results_dict: dict,
    control_plot: bool = False,
) -> dict:
    """Label both coronary centerlines' branch-point sets for :func:`discretize_vessel_tree`.

    Assumes `rca_cl`/`lca_cl` already have branches computed and correctly
    ordered (e.g. via :func:`~multimodars.load_centerline` and
    :func:`~multimodars.prepare_centerline`) — this
    only calls :func:`~multimodars.label_branches` for RCA, then LCA, to
    project the branch structure onto the surface-mesh point sets in
    `results_dict`. It does not modify `rca_cl`/`lca_cl` in any way.

    .. note::
        Manual edits - ``find_sharp_angles``, ``split_branch``,
        ``merge_branches`` - must be applied to `rca_cl`/`lca_cl` *before*
        calling this function, so `results_dict`'s branch labels reflect them.

    Parameters
    ----------
    rca_cl, lca_cl:
        Prepared centerlines (branches already computed and ordered), e.g. via
        :func:`~multimodars.load_centerline` and :func:`~multimodars.prepare_centerline`.
    results_dict:
        Dictionary produced by :func:`~multimodars.label_geometry`.
    control_plot:
        When ``True``, open an interactive Plotly 3-D visualisation showing
        centerline points coloured by branch ID and the labelled surface-mesh
        points, so you can verify assignments before discretizing (calls
        :func:`~multimodars.ccta.debug_plots.plot_centerline_branches`).

    Returns
    -------
    results_dict : dict
        Updated dictionary with ``rca_points_main``, ``rca_points_side_N``,
        ``lca_points_main``, and ``lca_points_side_N`` keys added.
    """
    results_dict = _label_branches(rca_cl, results_dict)
    results_dict = _label_branches(lca_cl, results_dict, results_key="lca_points")

    if control_plot:
        from .debug_plots import plot_centerline_branches

        plot_centerline_branches(rca_cl, lca_cl, results_dict)

    return results_dict


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
        ``orient_by_max_z``).
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
        Global point_index values (suitable for ``split_branch``).
    """
    positions = cl.find_sharp_angles(branch_id, cos_threshold)
    print(
        f"Branch {branch_id}: {len(positions)} sharp angle(s) at point_index {positions}"
    )
    if control_plot:
        from .debug_plots import plot_sharp_angles

        plot_sharp_angles(cl, branch_id, positions)
    return positions
