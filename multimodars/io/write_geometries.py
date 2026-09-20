from __future__ import annotations

import math
from pathlib import Path

import trimesh

from ..multimodars import PyCenterline


def centerline_to_obj(cl, filename: str) -> None:
    """
    Write out a centerline as an OBJ with:
      - vertex positions (v x y z)
      - vertex normals (vn nx ny nz), if normals are set
      - a single poly-line (l 1 2 3 ... N)

    Args:
        cl:        A PyCenterline instance
        filename:  Path to write (e.g. "my_centerline.obj")
    """
    if not isinstance(cl, PyCenterline):
        raise TypeError("Expected PyCenterline instance")

    with open(filename, "w") as f:
        good_pts = []
        for i, pt in enumerate(cl.points):
            x, y, z = pt.contour_point.x, pt.contour_point.y, pt.contour_point.z
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                # skip any malformed point
                continue
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
            good_pts.append(pt)

        has_normals = any(
            math.isfinite(nx) and math.isfinite(ny) and math.isfinite(nz)
            for pt in good_pts
            for nx, ny, nz in [pt.tangent]
        )
        if has_normals:
            for pt in good_pts:
                nx, ny, nz = pt.tangent
                if math.isfinite(nx) and math.isfinite(ny) and math.isfinite(nz):
                    f.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")
                else:
                    f.write("vn 0.000000 0.000000 0.000000\n")

        idxs = " ".join(str(i + 1) for i in range(len(good_pts)))
        f.write(f"l {idxs}\n")
    print(f"Wrote {len(good_pts)} valid points to {filename!r}")


def export_section_stl(
    results: dict,
    type: str = "all",
    output_dir: Path | str | None = None,
) -> None:
    """Export the mesh (or a labeled sub-region) as an STL file.

    Parameters
    ----------
    results : dict
        Labeled results dictionary containing ``"mesh"`` and the point-label
        lists produced by :func:`multimodars.ccta.label` / :func:`multimodars.ccta.scale`.
    type : str, optional
        Which region to export.  One of:

        * ``"all"``   - the full mesh as-is.
        * ``"aorta"`` - only the aorta region.
        * ``"rca"``   - only the RCA region (includes adjacent aorta ring).
        * ``"lca"``   - only the LCA region (includes adjacent aorta ring).

        Default is ``"all"``.
    output_dir : Path, str, or None, optional
        Directory in which to write the STL file.  Defaults to the current
        working directory when ``None``.
    """
    # Imported lazily: multimodars.ccta.__init__ re-exports this function from
    # here, so importing mesh_regions at module load time would be circular.
    from ..ccta.mesh_regions import (
        extract_region_with_border_faces,
        keep_labeled_points_from_mesh,
    )

    output_dir = Path(output_dir) if output_dir is not None else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh: trimesh.Trimesh = results["mesh"]

    _REGION_KEYS = {
        "aorta": "aorta_points",
        "rca": "rca_points",
        "lca": "lca_points",
    }

    if type == "all":
        mesh.export(str(output_dir / "all.stl"))
    elif type in _REGION_KEYS:
        region_points = results.get(_REGION_KEYS[type], [])
        if type == "aorta":
            sub_mesh_dict = keep_labeled_points_from_mesh(
                results, ["aorta_points", "rca_removed_points", "lca_removed_points"]
            )
            sub_mesh = sub_mesh_dict["mesh"]
        else:
            sub_mesh = extract_region_with_border_faces(mesh, region_points)
        sub_mesh.export(str(output_dir / f"{type}.stl"))
    else:
        raise ValueError(
            f"Unknown export type {type!r}. "
            f"Choose one of: 'all', 'aorta', 'rca', 'lca'."
        )
