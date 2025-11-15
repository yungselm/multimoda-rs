use anyhow::bail;

use crate::intravascular::io::geometry::{Contour, ContourType, Frame, Geometry};
use crate::intravascular::io::input::ContourPoint;
use std::collections::HashMap;

/// Interpolates between two aligned Geometry configurations with number of steps
/// used to visualize deformation over a cardiac cycle.
pub fn interpolate_contours(
    start: &Geometry,
    end: &Geometry,
    steps: usize,
    contour_types: &[ContourType],
) -> anyhow::Result<Vec<Geometry>> {
    use std::cmp::min;

    // Find matching number of frames
    let n_frames = min(start.frames.len(), end.frames.len());

    let mut geoms = Vec::with_capacity(steps + 2);
    geoms.push(start.clone());

    for step in 0..steps {
        let t = step as f64 / (steps - 1) as f64;

        let mut frames = Vec::with_capacity(n_frames);

        for i in 0..n_frames {
            let start_frame = &start.frames[i];
            let end_frame = &end.frames[i];

            // Interpolate lumen contour
            let lumen = interpolate_contour(&start_frame.lumen, &end_frame.lumen, t)?;

            // Interpolate extras based on requested contour types
            let mut extras = HashMap::new();
            for contour_type in contour_types {
                match contour_type {
                    ContourType::Lumen => {} // Already handled above
                    _ => {
                        if let (Some(start_contour), Some(end_contour)) = (
                            start_frame.extras.get(contour_type),
                            end_frame.extras.get(contour_type),
                        ) {
                            let interp_contour =
                                interpolate_contour(start_contour, end_contour, t)?;
                            extras.insert(*contour_type, interp_contour);
                        }
                    }
                }
            }

            // Interpolate reference point
            let reference_point = match (&start_frame.reference_point, &end_frame.reference_point) {
                (Some(start_rp), Some(end_rp)) => {
                    Some(interpolate_contour_point(start_rp, end_rp, t))
                }
                (Some(start_rp), None) => Some(*start_rp),
                (None, Some(end_rp)) => Some(*end_rp),
                (None, None) => None,
            };

            // Interpolate centroid
            let centroid = (
                start_frame.centroid.0 * (1.0 - t) + end_frame.centroid.0 * t,
                start_frame.centroid.1 * (1.0 - t) + end_frame.centroid.1 * t,
                start_frame.centroid.2 * (1.0 - t) + end_frame.centroid.2 * t,
            );

            frames.push(Frame {
                id: start_frame.id,
                centroid,
                lumen,
                extras,
                reference_point,
            });
        }

        geoms.push(Geometry {
            frames,
            label: format!("{}_inter_{}", start.label, step),
        });
    }

    geoms.push(end.clone());
    Ok(geoms)
}

/// Interpolate between two contours
fn interpolate_contour(start: &Contour, end: &Contour, t: f64) -> anyhow::Result<Contour> {
    if start.points.len() != end.points.len() {
        bail!("Contour point counts do not match between start and end");
    }

    let points = start
        .points
        .iter()
        .zip(&end.points)
        .map(|(ps, pe)| interpolate_contour_point(ps, pe, t))
        .collect();

    let centroid = match (start.centroid, end.centroid) {
        (Some(s_centroid), Some(e_centroid)) => Some((
            s_centroid.0 * (1.0 - t) + e_centroid.0 * t,
            s_centroid.1 * (1.0 - t) + e_centroid.1 * t,
            s_centroid.2 * (1.0 - t) + e_centroid.2 * t,
        )),
        (Some(s_centroid), None) => Some(s_centroid),
        (None, Some(e_centroid)) => Some(e_centroid),
        (None, None) => None,
    };

    Ok(Contour {
        id: start.id,
        original_frame: start.original_frame,
        points,
        centroid,
        aortic_thickness: interpolate_thickness(&start.aortic_thickness, &end.aortic_thickness, t),
        pulmonary_thickness: interpolate_thickness(
            &start.pulmonary_thickness,
            &end.pulmonary_thickness,
            t,
        ),
        kind: start.kind,
    })
}

/// Interpolate between two contour points
fn interpolate_contour_point(start: &ContourPoint, end: &ContourPoint, t: f64) -> ContourPoint {
    ContourPoint {
        frame_index: start.frame_index,
        point_index: start.point_index,
        x: start.x * (1.0 - t) + end.x * t,
        y: start.y * (1.0 - t) + end.y * t,
        z: start.z * (1.0 - t) + end.z * t,
        aortic: start.aortic, // Keep aortic flag from start
    }
}

/// Interpolate two optional thickness values at fraction `t` (0.0..1.0).
fn interpolate_thickness(start: &Option<f64>, end: &Option<f64>, t: f64) -> Option<f64> {
    match (start, end) {
        (Some(s), Some(e)) => Some(s * (1.0 - t) + e * t),
        _ => None,
    }
}

#[cfg(test)]
mod interpolation {
    use super::*;
    use crate::intravascular::io::geometry::{Contour, ContourType, Frame, Geometry};
    use crate::intravascular::io::input::ContourPoint;
    use approx::assert_relative_eq;
    use std::collections::HashMap;

    // Helper to create mock frame
    fn mock_frame(id: u32, z_offset: f64) -> Frame {
        let lumen = Contour {
            id,
            original_frame: id,
            points: vec![
                ContourPoint {
                    frame_index: id,
                    point_index: 0,
                    x: 1.0 + z_offset,
                    y: 2.0 + z_offset,
                    z: 3.0 + z_offset,
                    aortic: true,
                },
                ContourPoint {
                    frame_index: id,
                    point_index: 1,
                    x: 4.0 + z_offset,
                    y: 5.0 + z_offset,
                    z: 6.0 + z_offset,
                    aortic: true,
                },
            ],
            centroid: Some((2.5 + z_offset, 3.5 + z_offset, 4.5 + z_offset)),
            aortic_thickness: Some(1.0 + z_offset),
            pulmonary_thickness: Some(2.0 + z_offset),
            kind: ContourType::Lumen,
        };

        let mut extras = HashMap::new();

        // Add catheter contour
        let catheter = Contour {
            id,
            original_frame: id,
            points: vec![ContourPoint {
                frame_index: id,
                point_index: 0,
                x: 10.0 + z_offset,
                y: 20.0 + z_offset,
                z: 30.0 + z_offset,
                aortic: false,
            }],
            centroid: Some((10.0 + z_offset, 20.0 + z_offset, 30.0 + z_offset)),
            aortic_thickness: None,
            pulmonary_thickness: None,
            kind: ContourType::Catheter,
        };
        extras.insert(ContourType::Catheter, catheter);

        // Add EEM contour
        let eem = Contour {
            id,
            original_frame: id,
            points: vec![ContourPoint {
                frame_index: id,
                point_index: 0,
                x: 7.0 + z_offset,
                y: 8.0 + z_offset,
                z: 9.0 + z_offset,
                aortic: false,
            }],
            centroid: Some((7.0 + z_offset, 8.0 + z_offset, 9.0 + z_offset)),
            aortic_thickness: None,
            pulmonary_thickness: None,
            kind: ContourType::Eem,
        };
        extras.insert(ContourType::Eem, eem);

        Frame {
            id,
            centroid: (5.0 + z_offset, 6.0 + z_offset, 7.0 + z_offset),
            lumen,
            extras,
            reference_point: Some(ContourPoint {
                frame_index: id,
                point_index: 0,
                x: 0.0 + z_offset,
                y: 0.0 + z_offset,
                z: 0.0 + z_offset,
                aortic: false,
            }),
        }
    }

    // Helper to create mock geometry
    fn mock_geometry(label: &str, frame_count: usize) -> Geometry {
        let mut frames = Vec::new();
        for i in 0..frame_count {
            frames.push(mock_frame(i as u32, i as f64 * 10.0));
        }

        Geometry {
            frames,
            label: label.to_string(),
        }
    }

    #[test]
    fn test_interpolate_contours_basic() {
        let start = mock_geometry("start", 2);
        let end = mock_geometry("end", 2);
        let steps = 2;
        let contour_types = &[ContourType::Lumen, ContourType::Catheter, ContourType::Eem];

        let result = interpolate_contours(&start, &end, steps, contour_types).unwrap();

        // Should have start + steps interpolated + end
        assert_eq!(result.len(), steps + 2);

        // Verify start is unchanged
        assert_eq!(result[0].label, "start");
        assert_eq!(result[0].frames[0].lumen.points[0].x, 1.0);

        // Verify end is unchanged
        assert_eq!(result[result.len() - 1].label, "end");
        assert_eq!(result[result.len() - 1].frames[0].lumen.points[0].x, 1.0);

        // Verify interpolation at midpoint
        let mid = &result[1];
        assert_eq!(mid.label, "start_inter_0");

        // Point interpolation
        assert_relative_eq!(mid.frames[0].lumen.points[0].x, 1.0, epsilon = 1e-5);
        assert_relative_eq!(mid.frames[0].lumen.points[1].y, 5.0, epsilon = 1e-5);

        // Centroid interpolation
        assert_relative_eq!(mid.frames[0].centroid.0, 5.0, epsilon = 1e-5);

        // Catheter interpolation
        let catheter = mid.frames[0].extras.get(&ContourType::Catheter).unwrap();
        assert_relative_eq!(catheter.points[0].z, 30.0, epsilon = 1e-5);

        // EEM interpolation
        let eem = mid.frames[0].extras.get(&ContourType::Eem).unwrap();
        assert_relative_eq!(eem.points[0].x, 7.0, epsilon = 1e-5);
    }

    #[test]
    fn test_interpolate_contours_different_frame_counts() {
        let start = mock_geometry("start", 2); // 2 frames
        let end = mock_geometry("end", 3); // 3 frames

        let contour_types = &[ContourType::Lumen];
        let result = interpolate_contours(&start, &end, 1, contour_types).unwrap();

        // Should use minimum frame count (2)
        assert_eq!(result[0].frames.len(), 2); // start
        assert_eq!(result[1].frames.len(), 2); // interpolated
        assert_eq!(result[2].frames.len(), 3); // end (keeps original)
    }

    #[test]
    fn test_interpolate_contours_partial_contour_types() {
        let start = mock_geometry("start", 1);
        let end = mock_geometry("end", 1);

        // Only interpolate lumen, not catheter or EEM
        let contour_types = &[ContourType::Lumen];

        let result = interpolate_contours(&start, &end, 1, contour_types).unwrap();

        let interp_frame = &result[1].frames[0];

        // Should have lumen
        assert!(interp_frame.lumen.points.len() > 0);

        // Should NOT have catheter or EEM since we didn't request them
        assert!(interp_frame.extras.get(&ContourType::Catheter).is_none());
        assert!(interp_frame.extras.get(&ContourType::Eem).is_none());
    }

    #[test]
    fn test_interpolate_contours_with_missing_contours() {
        let mut start = mock_geometry("start", 1);
        let end = mock_geometry("end", 1);

        // Remove catheter from start but keep in end
        start.frames[0].extras.remove(&ContourType::Catheter);

        let contour_types = &[ContourType::Lumen, ContourType::Catheter];
        let result = interpolate_contours(&start, &end, 1, contour_types).unwrap();

        let interp_frame = &result[1].frames[0];

        // Should still have lumen
        assert!(interp_frame.lumen.points.len() > 0);

        // Should NOT have catheter since it was missing in start
        assert!(interp_frame.extras.get(&ContourType::Catheter).is_none());
    }

    #[test]
    fn test_interpolate_contour_point() {
        let start = ContourPoint {
            frame_index: 0,
            point_index: 0,
            x: 1.0,
            y: 2.0,
            z: 3.0,
            aortic: true,
        };

        let end = ContourPoint {
            frame_index: 0,
            point_index: 0,
            x: 11.0,
            y: 12.0,
            z: 13.0,
            aortic: false, // This should be ignored, start's aortic flag is kept
        };

        let result = interpolate_contour_point(&start, &end, 0.5);

        assert_relative_eq!(result.x, 6.0, epsilon = 1e-5);
        assert_relative_eq!(result.y, 7.0, epsilon = 1e-5);
        assert_relative_eq!(result.z, 8.0, epsilon = 1e-5);
        assert_eq!(result.aortic, true); // Keeps start's aortic flag
        assert_eq!(result.frame_index, 0);
        assert_eq!(result.point_index, 0);
    }

    #[test]
    fn test_interpolate_contour() {
        let start = Contour {
            id: 1,
            original_frame: 1,
            points: vec![ContourPoint {
                frame_index: 1,
                point_index: 0,
                x: 1.0,
                y: 2.0,
                z: 3.0,
                aortic: true,
            }],
            centroid: Some((1.0, 2.0, 3.0)),
            aortic_thickness: Some(1.0),
            pulmonary_thickness: Some(2.0),
            kind: ContourType::Lumen,
        };

        let end = Contour {
            id: 1,
            original_frame: 1,
            points: vec![ContourPoint {
                frame_index: 1,
                point_index: 0,
                x: 11.0,
                y: 12.0,
                z: 13.0,
                aortic: false,
            }],
            centroid: Some((11.0, 12.0, 13.0)),
            aortic_thickness: Some(3.0),
            pulmonary_thickness: Some(4.0),
            kind: ContourType::Lumen,
        };

        let result = interpolate_contour(&start, &end, 0.5).unwrap();

        assert_eq!(result.id, 1);
        assert_eq!(result.original_frame, 1);
        assert_eq!(result.kind, ContourType::Lumen);

        assert_relative_eq!(result.points[0].x, 6.0, epsilon = 1e-5);
        assert_relative_eq!(result.points[0].y, 7.0, epsilon = 1e-5);
        assert_relative_eq!(result.points[0].z, 8.0, epsilon = 1e-5);
        assert_eq!(result.points[0].aortic, true); // Keeps start's aortic flag

        assert_relative_eq!(result.centroid.unwrap().0, 6.0, epsilon = 1e-5);
        assert_relative_eq!(result.aortic_thickness.unwrap(), 2.0, epsilon = 1e-5);
        assert_relative_eq!(result.pulmonary_thickness.unwrap(), 3.0, epsilon = 1e-5);
    }

    #[test]
    fn test_interpolate_contour_mismatched_points() {
        let start = Contour {
            id: 1,
            original_frame: 1,
            points: vec![ContourPoint {
                frame_index: 1,
                point_index: 0,
                x: 1.0,
                y: 2.0,
                z: 3.0,
                aortic: true,
            }],
            centroid: Some((1.0, 2.0, 3.0)),
            aortic_thickness: Some(1.0),
            pulmonary_thickness: Some(2.0),
            kind: ContourType::Lumen,
        };

        let end = Contour {
            id: 1,
            original_frame: 1,
            points: vec![
                ContourPoint {
                    frame_index: 1,
                    point_index: 0,
                    x: 11.0,
                    y: 12.0,
                    z: 13.0,
                    aortic: false,
                },
                ContourPoint {
                    // Extra point
                    frame_index: 1,
                    point_index: 1,
                    x: 21.0,
                    y: 22.0,
                    z: 23.0,
                    aortic: false,
                },
            ],
            centroid: Some((11.0, 12.0, 13.0)),
            aortic_thickness: Some(3.0),
            pulmonary_thickness: Some(4.0),
            kind: ContourType::Lumen,
        };

        let result = interpolate_contour(&start, &end, 0.5);
        assert!(result.is_err(), "Should fail with mismatched point counts");
    }

    #[test]
    fn test_interpolate_thickness() {
        // Both present
        assert_eq!(
            interpolate_thickness(&Some(1.0), &Some(3.0), 0.5),
            Some(2.0)
        );

        // Start missing
        assert_eq!(interpolate_thickness(&None, &Some(3.0), 0.5), None);

        // End missing
        assert_eq!(interpolate_thickness(&Some(1.0), &None, 0.5), None);

        // Both missing
        assert_eq!(interpolate_thickness(&None, &None, 0.5), None);
    }

    #[test]
    fn test_interpolate_contours_zero_steps() {
        let start = mock_geometry("start", 1);
        let end = mock_geometry("end", 1);
        let contour_types = &[ContourType::Lumen];

        let result = interpolate_contours(&start, &end, 0, contour_types).unwrap();

        // Should still have start and end frames
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].label, "start");
        assert_eq!(result[1].label, "end");
    }

    #[test]
    fn test_interpolate_contours_missing_reference_points() {
        let mut start = mock_geometry("start", 1);
        let end = mock_geometry("end", 1);

        // Remove reference point from start
        start.frames[0].reference_point = None;

        let contour_types = &[ContourType::Lumen];
        let result = interpolate_contours(&start, &end, 1, contour_types).unwrap();

        let interp_frame = &result[1].frames[0];

        // Should have reference point from end since start was missing
        assert!(interp_frame.reference_point.is_some());
    }
}
