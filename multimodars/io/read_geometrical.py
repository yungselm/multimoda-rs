from __future__ import annotations

import trimesh
from pathlib import Path
import warnings


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
        mesh = trimesh.util.concatenate(geoms)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise TypeError(f"Unsupported object loaded from {path}: {type(loaded)}")

    # basic cleanups
    try:
        mesh.update_faces(mesh.unique_faces())
    except Exception:
        try:
            mesh.remove_duplicate_faces()
        except Exception:
            pass

    mesh.remove_unreferenced_vertices()

    try:
        mesh.update_faces(mesh.nondegenerate_faces())
    except Exception:
        try:
            mesh.remove_degenerate_faces()
        except Exception:
            pass

    mesh.fix_normals()

    # attempt to fix small holes
    try:
        trimesh.repair.fill_holes(mesh)
    except Exception:
        # best-effort; don't fail on repair errors
        warnings.warn(f"fill_holes failed for mesh from {path}", RuntimeWarning)

    if not mesh.is_watertight:
        warnings.warn(
            f"Mesh from {path} is not watertight after repairs", RuntimeWarning
        )

    return mesh
