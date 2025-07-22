from typing import Literal, Any, Dict, Tuple, Union
from multimodars import (
    from_file_full,
    from_file_doublepair,
    from_file_singlepair,
    from_file_single,
)

from multimodars import (
    geometry_from_array,
    from_array_full,
    from_array_doublepair,
    from_array_singlepair,
)

from multimodars import (
    align_three_point,
    align_manual,
)

from .multimodars import PyGeometry, PyGeometryPair

Mode = Literal["full", "doublepair", "singlepair", "single"]

def from_file(
    mode: Mode,
    **kwargs: Any,
) -> Union[Tuple[PyGeometryPair, PyGeometryPair, PyGeometryPair, PyGeometryPair],
           Tuple[PyGeometryPair, PyGeometryPair],
           PyGeometryPair,
           PyGeometry]:
    """
    A unified entrypoint for all `from_file_*` variants.

    Parameters
    ----------
    mode : {"full","doublepair","singlepair","single"}
        Selects which low-level function is called.  
        
          - "full"       → needs rest_input_path, stress_input_path,
                            rest_output_path, stress_output_path,
                            diastole_output_path, systole_output_path,
                            steps_best_rotation, range_rotation_rad,
                            interpolation_steps, image_center, radius, n_points
          - "doublepair" → needs rest_input_path, stress_input_path,
                            rest_output_path, stress_output_path,
                            steps_best_rotation, range_rotation_rad,
                            interpolation_steps, image_center, radius, n_points
          - "singlepair" → needs input_path, output_path,
                            steps_best_rotation, range_rotation_rad,
                            interpolation_steps, image_center, radius, n_points
          - "single"     → needs input_path, output_path,
                            steps_best_rotation, range_rotation_rad,
                            diastole, image_center, radius, n_points

    Other Parameters
    ----------------
    **kwargs : dict
        Keyword arguments required depend on `mode` (see above).

    Returns
    -------
    Union[
      Tuple[PyGeometryPair,PyGeometryPair,PyGeometryPair,PyGeometryPair],
      Tuple[PyGeometryPair,PyGeometryPair],
      PyGeometryPair,
      PyGeometry
    ]
        The exact return shape depends on `mode`.

    Raises
    ------
    ValueError
        If an unsupported `mode` is passed.
    """
    defaults = {
        "steps_best_rotation": 300,
        "range_rotation_rad": 1.57,
        "rest_output_path": "output/rest",
        "stress_output_path": "output/stress",
        "diastole_output_path": "output/diastole",
        "systole_output_path": "output/systole",
        "interpolation_steps": 28,
        "image_center": (4.5, 4.5),
        "radius": 0.5,
        "n_points": 20,
    }
    merged = {**defaults, **kwargs}

    if mode == "full":
        required = (
            "rest_input_path", "stress_input_path",
            "steps_best_rotation", "range_rotation_rad",
            "rest_output_path", "stress_output_path",
            "diastole_output_path", "systole_output_path",
            "interpolation_steps", "image_center",
            "radius", "n_points"
        )
        args = {k: merged[k] for k in required}
        return from_file_full(
            args["rest_input_path"],
            args["stress_input_path"],
            args["steps_best_rotation"],
            args["range_rotation_rad"],
            args["rest_output_path"],
            args["stress_output_path"],
            args["diastole_output_path"],
            args["systole_output_path"],
            args["interpolation_steps"],
            args["image_center"],
            args["radius"],
            args["n_points"],
        )

    elif mode == "doublepair":
        required = (
            "rest_input_path", "stress_input_path",
            "steps_best_rotation", "range_rotation_rad",
            "rest_output_path", "stress_output_path",
            "interpolation_steps", "image_center",
            "radius", "n_points"
        )
        args = {k: merged[k] for k in required}
        return from_file_doublepair(
            args["rest_input_path"],
            args["stress_input_path"],
            args["steps_best_rotation"],
            args["range_rotation_rad"],
            args["rest_output_path"],
            args["stress_output_path"],
            args["interpolation_steps"],
            args["image_center"],
            args["radius"],
            args["n_points"],
        )

    elif mode == "singlepair":
        required = (
            "input_path", "output_path",
            "steps_best_rotation", "range_rotation_rad",
            "interpolation_steps", "image_center",
            "radius", "n_points"
        )
        args = {k: merged[k] for k in required}
        return from_file_singlepair(
            args["input_path"],
            args["output_path"],
            args["steps_best_rotation"],
            args["range_rotation_rad"],
            args["interpolation_steps"],
            args["image_center"],
            args["radius"],
            args["n_points"],
        )

    elif mode == "single":
        required = (
            "input_path", "output_path",
            "steps_best_rotation", "range_rotation_rad",
            "diastole", "image_center", "radius", "n_points",
        )
        args = {k: merged[k] for k in required}
        return from_file_single(
            args["input_path"],
            args["output_path"],
            args["steps_best_rotation"],
            args["range_rotation_rad"],
            args["diastole"],
            args["image_center"],
            args["radius"],
            args["n_points"],
        )

    else:
        raise ValueError(f"Unsupported mode: {mode!r}")


def from_array(
    mode: Mode,
    **kwargs: Any,
) -> Union[
    Tuple[PyGeometryPair, PyGeometryPair, PyGeometryPair, PyGeometryPair],
    Tuple[PyGeometryPair, PyGeometryPair],
    PyGeometryPair,
    PyGeometry,
]:
    """
    Unified entry for all array-based pipelines.

    Supported modes
    ---------------

    ::
      - "full"       → from_array_full(rest_dia, rest_sys, stress_dia, stress_sys,
                                       steps_best_rotation, range_rotation_rad,
                                       interpolation_steps,
                                       rest_output_path, stress_output_path,
                                       diastole_output_path, systole_output_path)
      - "doublepair" → from_array_doublepair(rest_dia, rest_sys, stress_dia, stress_sys,
                                             steps_best_rotation, range_rotation_rad,
                                             interpolation_steps,
                                             rest_output_path, stress_output_path)
      - "singlepair" → from_array_singlepair(rest_dia, rest_sys,
                                             output_path,
                                             steps_best_rotation, range_rotation_rad,
                                             interpolation_steps)
      - "single"     → geometry_from_array(contours, walls, reference_point,
                                           steps, range, image_center, radius, n_points,
                                           label, records, delta, max_rounds,
                                           diastole, sort, write_obj, output_path)

    Parameters
    ----------
    mode : {"full","doublepair","singlepair","single"}
        Which array-based pipeline to run (see “Supported modes” above).
    **kwargs : dict
        Keyword-only arguments required vary by mode (see above).

    Returns
    -------
    Union[
      Tuple[PyGeometryPair,PyGeometryPair,PyGeometryPair,PyGeometryPair],
      Tuple[PyGeometryPair,PyGeometryPair],
      PyGeometryPair,
      PyGeometry
    ]
        Depends on `mode`.

    Raises
    ------
    ValueError
        If an unsupported `mode` is passed.
    """
    defaults = {
        "steps_best_rotation": 300,
        "range_rotation_rad": 1.57,
        "interpolation_steps": 28,
        "image_center": (4.5, 4.5),
        "radius": 0.5,
        "n_points": 20,
        "rest_output_path": "output/rest",
        "stress_output_path": "output/stress",
        "diastole_output_path": "output/diastole",
        "systole_output_path": "output/systole",
        "output_path": "output/single",
        "label": "None",
        "records": None,
        "delta": 0.1,
        "max_rounds": 5,
        "diastole": True,
        "sort": True,
        "write_obj": False,
    }
    merged = {**defaults, **kwargs}

    if mode == "singlepair" and "output_path" not in kwargs:
        merged["output_path"] = "output/singlepair"

    if mode == "full":
        required = (
            "rest_geometry_dia", "rest_geometry_sys",
            "stress_geometry_dia", "stress_geometry_sys",
            "steps_best_rotation", "range_rotation_rad", "interpolation_steps",
            "rest_output_path", "stress_output_path",
            "diastole_output_path", "systole_output_path",
        )
        args = {k: merged[k] for k in required}
        return from_array_full(
            args["rest_geometry_dia"],
            args["rest_geometry_sys"],
            args["stress_geometry_dia"],
            args["stress_geometry_sys"],
            args["steps_best_rotation"],
            args["range_rotation_rad"],
            args["interpolation_steps"],
            args["rest_output_path"],
            args["stress_output_path"],
            args["diastole_output_path"],
            args["systole_output_path"],
        )

    elif mode == "doublepair":
        required = (
            "rest_geometry_dia", "rest_geometry_sys",
            "stress_geometry_dia", "stress_geometry_sys",
            "steps_best_rotation", "range_rotation_rad", "interpolation_steps",
            "rest_output_path", "stress_output_path",
        )
        args = {k: merged[k] for k in required}
        return from_array_doublepair(
            args["rest_geometry_dia"],
            args["rest_geometry_sys"],
            args["stress_geometry_dia"],
            args["stress_geometry_sys"],
            args["steps_best_rotation"],
            args["range_rotation_rad"],
            args["interpolation_steps"],
            args["rest_output_path"],
            args["stress_output_path"],
        )

    elif mode == "singlepair":
        required = (
            "geometry_dia", "geometry_sys",
            "output_path",
            "steps_best_rotation", "range_rotation_rad", "interpolation_steps",
        )
        args = {k: merged[k] for k in required}
        return from_array_singlepair(
            args["geometry_dia"],
            args["geometry_sys"],
            args["output_path"],
            args["steps_best_rotation"],
            args["range_rotation_rad"],
            args["interpolation_steps"],
        )

    elif mode == "single":
        required = (
            "geometry",
            "steps_best_rotation", "range_rotation_rad", "image_center", "radius", "n_points",
            "label", "records", "delta", "max_rounds",
            "diastole", "sort", "write_obj", "output_path",
        )
        args = {k: merged[k] for k in required}
        return geometry_from_array(
            args["geometry"],
            args["steps_best_rotation"],
            args["range_rotation_rad"],
            args["image_center"],
            args["radius"],
            args["n_points"],
            args["label"],
            args["records"],
            args["delta"],
            args["max_rounds"],
            args["diastole"],
            args["sort"],
            args["write_obj"],
            args["output_path"],
        )

    else:
        raise ValueError(f"Unsupported mode: {mode!r}")

Mode_cl = Literal["three_pt", "manual"]

def to_centerline(
    mode: Mode_cl,
    **kwargs: Any,
) -> PyGeometryPair:
    """
    Unified entry for all to_centerline pipelines.

    Supported modes
    ---------------

    ::
      - "three_pt"      → align_three_point(centerline, geometry_pair, aortic_ref_pt, upper_ref_pt,
                                       lower_ref_pt, angle_step, write, interpolation_steps,
                                       output_dir, case_name)
      - "manual"        → align_manual(centerline, geometry_pair, rotation_angle, start_point,
                                             write, interpolation_steps, output_dir, case_name)

    Parameters
    ----------
    mode : {"three_pt","manual"}
        Which array-based pipeline to run (see “Supported modes” above).
    **kwargs : dict
        Keyword-only arguments required vary by mode (see above).

    Returns
    -------
    PyGeometryPair
        Depends on `mode`.

    Raises
    ------
    ValueError
        If an unsupported `mode` is passed.
    """
    defaults = {
        "angle_step": 0.01745329, # approx 1° in radians
        "write": False,
        "interpolation_steps": 28,
        "output_dir": "output/aligned",
        "case_name": "None",
    }
    merged = {**defaults, **kwargs}

    if mode == "three_pt":
        required = (
            "centerline", "geometry_pair",
            "aortic_ref_pt", "upper_ref_pt",
            "lower_ref_pt", "angle_step",
            "write", "interpolation_steps",
            "output_dir", "case_name",
        )
        args = {k: merged[k] for k in required}
        return align_three_point(
            args["centerline"],
            args["geometry_pair"],
            args["aortic_ref_pt"],
            args["upper_ref_pt"],
            args["lower_ref_pt"],
            args["angle_step"],
            args["write"],
            args["interpolation_steps"],
            args["output_dir"],
            args["case_name"],
        )

    elif mode == "manual":
        required = (
            "centerline", "geometry_pair", "rotation_angle", 
            "start_point", "write", "interpolation_steps",
            "output_dir", "case_name",
        )
        args = {k: merged[k] for k in required}
        return align_manual(
            args["centerline"],
            args["geometry_pair"],
            args["rotation_angle"],
            args["start_point"],
            args["write"],
            args["interpolation_steps"],
            args["output_dir"],
            args["case_name"],
        )

    else:
        raise ValueError(f"Unsupported mode: {mode!r}")
