"""Names of the per-branch point lists that :func:`label_branches` writes into *results*."""

from __future__ import annotations

import re


def side_branch_ids(results: dict, prefix: str) -> list[int]:
    """Branch ids ``k`` of every ``{prefix}_side_{k}`` key in *results*, ascending."""
    pattern = re.compile(rf"{re.escape(prefix)}_side_(\d+)")
    return sorted(int(m.group(1)) for key in results if (m := pattern.fullmatch(key)))
