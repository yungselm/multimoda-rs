from __future__ import annotations

import warnings
from pathlib import Path

import trimesh


def read_mesh(path: Path | str) -> trimesh.base.Trimesh:
    """Load a mesh from disk and attempt lightweight repairs.

    - Accepts Path or str.
    - If a Scene is loaded, its geometries are concatenated.
    - Performs basic cleanups and attempts to fill small holes.
    - Returns a Trimesh even if not watertight (warns in that case).
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Geometry file not found: {path}")

    try:
        loaded = trimesh.load(path, force="mesh")
    except Exception as exc:
        raise RuntimeError(f"Failed to load mesh from {path}: {exc}") from exc

    if isinstance(loaded, trimesh.Scene):
        geoms = tuple(loaded.geometry.values())
        if not geoms:
            raise RuntimeError(f"No geometry found in scene loaded from {path}")
        loaded = trimesh.util.concatenate(geoms)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unsupported object loaded from {path}: {type(loaded)}")
    mesh = loaded

    # basic cleanups
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.fix_normals()

    # attempt to fix small holes
    try:
        trimesh.repair.fill_holes(mesh)
    except Exception:  # noqa: BLE001 - best-effort; don't fail on repair errors
        warnings.warn(f"fill_holes failed for mesh from {path}", RuntimeWarning)

    if not mesh.is_watertight:
        warnings.warn(
            f"Mesh from {path} is not watertight after repairs", RuntimeWarning
        )

    return mesh
