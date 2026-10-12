"""Helpers for the labelled *results* dictionary the CCTA steps pass along.

Vertex labels are lists of ``(x, y, z)`` tuples (``"aorta_points"``,
``"proximal_points"``, ...), except the per-branch labels from
:func:`~multimodars.label_branches`: ``"rca_branches"`` / ``"lca_branches"`` map
each centerline branch id to its vertices.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from ..multimodars import PyCenterline

VESSELS = ("rca", "lca")
BRANCHES_SUFFIX = "_branches"

PointList = list[tuple[float, float, float]]


def branches_key(vessel: str) -> str:
    """Key of the per-branch labels of *vessel* (``"rca"`` -> ``"rca_branches"``)."""
    return f"{vessel}{BRANCHES_SUFFIX}"


def vessel_of_centerline(results: dict, centerline: PyCenterline) -> str:
    """``"rca"`` or ``"lca"``: the vessel whose labelled vertices lie nearest *centerline*."""
    tree = cKDTree(np.asarray(centerline.points_as_tuples(), dtype=np.float64))
    distances = {}
    for vessel in VESSELS:
        pts = results.get(f"{vessel}_points")
        if pts:
            sample = np.asarray(pts[:: max(1, len(pts) // 2000)], dtype=np.float64)
            distances[vessel] = tree.query(sample)[0].mean()
    if not distances:
        raise ValueError("results has no 'rca_points' or 'lca_points'")
    return min(distances, key=lambda v: distances[v])


def vessel_branches(results: dict, vessel: str) -> dict[int, PointList]:
    """Per-branch labels of *vessel*; raises if :func:`label_branches` has not run."""
    key = branches_key(vessel)
    if key not in results:
        raise ValueError(f"results has no '{key}'; run label_branches_pair first")
    return results[key]


def _is_point_list(value) -> bool:
    return isinstance(value, list) and all(
        isinstance(p, (tuple, list, np.ndarray)) and len(p) == 3 for p in value[:1]
    )


def _is_branch_map(key: str, value) -> bool:
    return key.endswith(BRANCHES_SUFFIX) and isinstance(value, dict)


def map_point_lists(
    results: dict,
    fn: Callable[[list], list],
    skip: Callable[[str], bool] = lambda key: False,
) -> dict:
    """Copy of *results* with *fn* applied to every vertex list, branch lists included.

    Keys for which *skip* holds are copied unchanged.
    """
    updated = dict(results)
    for key, value in results.items():
        if skip(key):
            continue
        if _is_branch_map(key, value):
            updated[key] = {k: fn(pts) for k, pts in value.items()}
        elif _is_point_list(value):
            updated[key] = fn(value)
    return updated


def point_lists(results: dict) -> list[tuple[tuple, list]]:
    """Every vertex list in *results* with its location: ``(key,)`` or ``(key, branch id)``."""
    found: list[tuple[tuple, list]] = []
    for key, value in results.items():
        if _is_branch_map(key, value):
            found += [((key, k), pts) for k, pts in value.items()]
        elif _is_point_list(value):
            found.append(((key,), value))
    return found


def set_point_list(results: dict, where: tuple, value: list) -> None:
    """Store *value* at a location from :func:`point_lists`."""
    if len(where) == 1:
        results[where[0]] = value
    else:
        key, branch = where
        results[key] = {**results[key], branch: value}
