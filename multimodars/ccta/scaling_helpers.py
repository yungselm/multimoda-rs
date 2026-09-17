from __future__ import annotations


def _extract_wall_from_frames(frames) -> list[tuple[float, float, float]] | None:
    """Extract the straight-wall (coronary-side) points from intravascular frames.

    ``create_aortic_wall`` in ``wall.rs`` builds the ``"Wall"`` extra contour
    in two halves:

    * **Straight wall** - ``point_index`` 0 to ``n // 2`` (exclusive): the lumen
      contour offset outward by 1 mm, following the true circular/elliptic vessel
      geometry on the coronary side.
    * **Aortic wall** - ``point_index`` ``n // 2`` to ``n``: the rectangular
      aortic-thickness shape constructed from ``aortic_thickness``.

    Only the straight-wall half is returned because it preserves the actual vessel
    cross-section shape and is therefore a stable reference for radial scaling.
    Assumes an even number of points per frame (the standard 500-point geometry).

    Parameters
    ----------
    frames : list of PyFrame
        Intravascular imaging frames.  Frames without ``aortic_thickness`` are
        skipped.

    Returns
    -------
    list of tuple
        ``(x, y, z)`` tuples of straight-wall points from the last eligible frame.
        Returns ``None`` if no eligible frame is found.

    Raises
    ------
    ValueError
        If an eligible frame is missing the ``"Wall"`` extras entry or that
        entry is empty.
    """
    n_points = len(frames[0].lumen.points)
    half = n_points // 2

    reference_points = None

    for frame in frames:
        if frame.lumen.aortic_thickness is None:
            continue
        wall = frame.extras.get("Wall")
        if wall is None:
            raise ValueError(
                f"No Wall extras found for frame {getattr(frame, 'frame', '?')}"
            )
        if not wall.points:
            raise ValueError(
                f"Empty Wall extras for frame {getattr(frame, 'frame', '?')}"
            )

        # Straight wall: coronary-side offset lumen, point_index 0..half.
        # Aortic wall:   rectangular aortic-thickness shape, point_index half..n_points.
        reference_points = [
            (p.x, p.y, p.z) for p in wall.points if p.point_index < half
        ]

    return reference_points
