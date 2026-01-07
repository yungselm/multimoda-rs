from __future__ import annotations

from . import adjust_ccta

from pathlib import Path
from trimesh import Trimesh
from typing import Tuple


def label_mesh(
    aorta_centerline,
    rca_centerline,
    lca_centerline,
    aligned_geometry,
    mesh_path: str | Path,
) -> Tuple[Trimesh, dict]:
    """ Completely labels a mesh by centerlines (RCA, LCA, Aorta)
        and an aligned intravascular mm.PyGeometry object (Anomalous,
        Proximal, Distal).

        Args:
            aorta_centerline: PyCenterline = The aortic centerline,
            rca_centerline: PyCenterline = The RCA centerline,
            lca_centerline: PyCenterline = The LCA centerline,
            aligned_geometry: PyGeometry = The already centerline aligned intravascular geometry,
            mesh_path: str | Tuple = Path to the stl or obj file containing the aorta with coronaries.
        
        Returns:
            mesh: trimesh.Trimesh = The original mesh.
            labels: dict = a dictionary with entries containing the points for each label.
    """
    from adjust_ccta import label_geometry
    pass