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
    if mode == "full":
        required = (
            "rest_input_path", "stress_input_path",
            "rest_output_path", "stress_output_path",
            "diastole_output_path", "systole_output_path",
            "steps_best_rotation", "range_rotation_rad",
            "interpolation_steps", "image_center", "radius", "n_points",
        )
        args = {k: kwargs[k] for k in required}
        return from_file_full(
            args["rest_input_path"],
            args["steps_best_rotation"],
            args["range_rotation_rad"],
            args["rest_output_path"],
            args["interpolation_steps"],
            args["stress_input_path"],
            args["stress_output_path"],
            args["diastole_output_path"],
            args["systole_output_path"],
            args["image_center"],
            args["radius"],
            args["n_points"],
        )

    elif mode == "doublepair":
        required = (
            "rest_input_path", "stress_input_path",
            "rest_output_path", "stress_output_path",
            "steps_best_rotation", "range_rotation_rad",
            "interpolation_steps", "image_center", "radius", "n_points",
        )
        args = {k: kwargs[k] for k in required}
        return from_file_doublepair(
            args["rest_input_path"],
            args["steps_best_rotation"],
            args["range_rotation_rad"],
            args["rest_output_path"],
            args["interpolation_steps"],
            args["stress_input_path"],
            args["stress_output_path"],
            args["image_center"],
            args["radius"],
            args["n_points"],
        )

    elif mode == "singlepair":
        required = (
            "input_path", "output_path",
            "steps_best_rotation", "range_rotation_rad",
            "interpolation_steps", "image_center", "radius", "n_points",
        )
        args = {k: kwargs[k] for k in required}
        return from_file_singlepair(
            args["input_path"],
            args["steps_best_rotation"],
            args["range_rotation_rad"],
            args["output_path"],
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
        args = {k: kwargs[k] for k in required}
        return from_file_single(
            args["input_path"],
            args["steps_best_rotation"],
            args["range_rotation_rad"],
            args["output_path"],
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
    if mode == "full":
        required = (
            "rest_dia", "rest_sys",
            "stress_dia", "stress_sys",
            "steps_best_rotation","range_rotation_rad","interpolation_steps",
            "rest_output_path","stress_output_path",
            "diastole_output_path","systole_output_path",
        )
        args = {k: kwargs[k] for k in required}
        return from_array_full(
            args["rest_dia"],
            args["rest_sys"],
            args["stress_dia"],
            args["stress_sys"],
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
            "rest_dia", "rest_sys",
            "stress_dia", "stress_sys",
            "steps_best_rotation","range_rotation_rad","interpolation_steps",
            "rest_output_path","stress_output_path",
        )
        args = {k: kwargs[k] for k in required}
        return from_array_doublepair(
            args["rest_dia"],
            args["rest_sys"],
            args["stress_dia"],
            args["stress_sys"],
            args["steps_best_rotation"],
            args["range_rotation_rad"],
            args["interpolation_steps"],
            args["rest_output_path"],
            args["stress_output_path"],
        )

    elif mode == "singlepair":
        required = (
            "rest_dia", "rest_sys",
            "output_path",
            "steps_best_rotation","range_rotation_rad","interpolation_steps",
        )
        args = {k: kwargs[k] for k in required}
        return from_array_singlepair(
            args["rest_dia"],
            args["rest_sys"],
            args["output_path"],
            args["steps_best_rotation"],
            args["range_rotation_rad"],
            args["interpolation_steps"],
        )

    elif mode == "single":
        # This one takes raw contours, walls, reference_point, plus all the flags
        required = (
            "contours", "walls", "reference_point",
            "steps", "range", "image_center", "radius", "n_points",
            "label", "records", "delta", "max_rounds",
            "diastole","sort","write_obj","output_path",
        )
        args = {k: kwargs[k] for k in required}
        return geometry_from_array(
            args["contours"],
            args["walls"],
            args["reference_point"],
            args["steps"],
            args["range"],
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