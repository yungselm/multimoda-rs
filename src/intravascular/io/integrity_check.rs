use anyhow::{anyhow, Result};
use std::collections::HashSet;

use super::geometry::{ContourType, Geometry};
use super::input::ContourPoint;

pub fn check_geometry_integrity(geometry: &Geometry) -> Result<()> {
    if geometry.frames.is_empty() {
        return Err(anyhow!("Geometry has no frames"));
    }

    let checks: &[(&str, fn(&Geometry) -> Result<()>)] = &[
        ("check_frame_ids_consecutive", check_frame_ids_consecutive),
        ("check_centroids_match", check_centroids_match),
        ("check_lumen_presence", check_lumen_presence),
        ("check_reference_point", check_reference_point),
        ("check_contour_point_counts", check_contour_point_counts),
        (
            "check_original_frame_consistency",
            check_original_frame_consistency,
        ),
        ("check_proximal_end_index", check_proximal_end_index),
        ("check_z_distribution", check_z_distribution),
    ];

    for (name, f) in checks {
        if let Err(e) = f(geometry) {
            println!("Integrity check '{}' failed: {}", name, e);
            return Err(e);
        }
    }

    Ok(())
}

/// Check that frame IDs are consecutive (0, 1, 2, ...)
fn check_frame_ids_consecutive(geometry: &Geometry) -> Result<()> {
    for (index, frame) in geometry.frames.iter().enumerate() {
        if frame.id != index as u32 {
            return Err(anyhow!(
                "Frame IDs are not consecutive. Expected ID {}, found ID {}",
                index,
                frame.id
            ));
        }
    }
    Ok(())
}

/// Check that frame centroid matches lumen centroid
fn check_centroids_match(geometry: &Geometry) -> Result<()> {
    for (frame_index, frame) in geometry.frames.iter().enumerate() {
        let frame_centroid = frame.centroid;

        let lumen_centroid = match frame.lumen.centroid {
            Some(centroid) => centroid,
            None => compute_centroid_from_points(&frame.lumen.points),
        };

        if !points_approximately_equal(frame_centroid, lumen_centroid) {
            return Err(anyhow!(
                "Frame centroid does not match lumen centroid in frame {} (ID {}). Frame: {:?}, Lumen: {:?}",
                frame_index,
                frame.id,
                frame_centroid,
                lumen_centroid
            ));
        }

        if let Some(stored_lumen_centroid) = frame.lumen.centroid {
            if !points_approximately_equal(stored_lumen_centroid, lumen_centroid) {
                return Err(anyhow!(
                    "Stored lumen centroid does not match computed centroid in frame {} (ID {}). Stored: {:?}, Computed: {:?}",
                    frame_index,
                    frame.id,
                    stored_lumen_centroid,
                    lumen_centroid
                ));
            }
        }
    }
    Ok(())
}

/// Check that lumen is present in every frame
fn check_lumen_presence(geometry: &Geometry) -> Result<()> {
    for (frame_index, frame) in geometry.frames.iter().enumerate() {
        if frame.lumen.points.is_empty() {
            return Err(anyhow!(
                "Lumen contour has no points in frame {} (ID {})",
                frame_index,
                frame.id
            ));
        }

        if frame.lumen.kind != ContourType::Lumen {
            return Err(anyhow!(
                "Lumen contour has incorrect type in frame {} (ID {}). Expected Lumen, found {:?}",
                frame_index,
                frame.id,
                frame.lumen.kind
            ));
        }
    }
    Ok(())
}

/// Check that at most one reference point exists in the geometry
fn check_reference_point(geometry: &Geometry) -> Result<()> {
    let reference_frames: Vec<u32> = geometry
        .frames
        .iter()
        .filter_map(|frame| frame.reference_point.as_ref().map(|_| frame.id))
        .collect();

    match reference_frames.len() {
        1 => Ok(()),
        n => Err(anyhow!("Expected exactly one reference point, found {}", n)),
    }
}

/// Check that all contours (lumen and extras) maintain consistent point counts across all frames
fn check_contour_point_counts(geometry: &Geometry) -> Result<()> {
    use std::collections::HashMap;

    let mut expected_counts: HashMap<ContourType, usize> = HashMap::new();

    for (frame_index, frame) in geometry.frames.iter().enumerate() {
        let lumen_count = frame.lumen.points.len();
        match expected_counts.get(&ContourType::Lumen) {
            Some(&expected) => {
                if lumen_count != expected {
                    return Err(anyhow!(
                        "Lumen point count mismatch in frame {} (ID {}). Expected {}, found {}",
                        frame_index,
                        frame.id,
                        expected,
                        lumen_count
                    ));
                }
            }
            None => {
                expected_counts.insert(ContourType::Lumen, lumen_count);
            }
        }

        for (_contour_type, contour) in &frame.extras {
            let kind = contour.kind;
            let count = contour.points.len();
            if let Some(&expected) = expected_counts.get(&kind) {
                if count != expected {
                    return Err(anyhow!(
                        "{:?} contour point count mismatch in frame {} (ID {}). Expected {}, found {}",
                        kind,
                        frame_index,
                        frame.id,
                        expected,
                        count
                    ));
                }
            } else {
                expected_counts.insert(kind, count);
            }
        }
    }

    Ok(())
}

/// Check that lumen, extras, and reference point have the same original_frame within each frame
fn check_original_frame_consistency(geometry: &Geometry) -> Result<()> {
    for (frame_index, frame) in geometry.frames.iter().enumerate() {
        let expected_original_frame = frame.lumen.original_frame;

        for (contour_type, contour) in &frame.extras {
            if contour.original_frame != expected_original_frame {
                return Err(anyhow!(
                    "Original frame mismatch in frame {} (ID {}). Lumen has original_frame {}, {} has original_frame {}",
                    frame_index,
                    frame.id,
                    expected_original_frame,
                    contour_type.as_str(),
                    contour.original_frame
                ));
            }
        }

        if let Some(ref_point) = &frame.reference_point {
            if ref_point.frame_index != expected_original_frame {
                return Err(anyhow!(
                    "Reference point original frame mismatch in frame {} (ID {}). Lumen has original_frame {}, reference point has frame_index {}",
                    frame_index,
                    frame.id,
                    expected_original_frame,
                    ref_point.frame_index
                ));
            }
        }
    }
    Ok(())
}

/// Check that the proximal end reported by the Geometry matches the frame with the minimum z coordinate.
/// This is more robust than expecting index 0 unconditionally.
fn check_proximal_end_index(geometry: &Geometry) -> Result<()> {
    let proximal_idx = geometry.find_proximal_end_idx();

    let mut min_z = std::f64::INFINITY;
    let mut min_idx = 0usize;
    for (i, f) in geometry.frames.iter().enumerate() {
        let z = f.centroid.2;
        if z < min_z {
            min_z = z;
            min_idx = i;
        }
    }

    if proximal_idx != min_idx {
        return Err(anyhow!(
            "Proximal end index is {}, but frame with minimum z is {} (z={}).",
            proximal_idx,
            min_idx,
            min_z
        ));
    }
    Ok(())
}

fn check_z_distribution(geometry: &Geometry) -> Result<()> {
    let n = geometry.frames.len();
    let z_pos_zero = geometry.frames[0].centroid.2;
    let z_pos_n = geometry.frames[n - 1].centroid.2;
    if z_pos_zero > z_pos_n {
        return Err(anyhow!(
            "First frame has higher z-coords {} than last frame {}",
            z_pos_zero,
            z_pos_n,
        ));
    }
    Ok(())
}

/// Helper function to compute centroid from contour points
fn compute_centroid_from_points(points: &[ContourPoint]) -> (f64, f64, f64) {
    if points.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let (sum_x, sum_y, sum_z) = points.iter().fold((0.0, 0.0, 0.0), |(sx, sy, sz), p| {
        (sx + p.x, sy + p.y, sz + p.z)
    });

    let n = points.len() as f64;
    (sum_x / n, sum_y / n, sum_z / n)
}

/// Helper function to check if two 3D points are approximately equal
fn points_approximately_equal(a: (f64, f64, f64), b: (f64, f64, f64)) -> bool {
    const EPSILON: f64 = 1e-6;
    (a.0 - b.0).abs() < EPSILON && (a.1 - b.1).abs() < EPSILON && (a.2 - b.2).abs() < EPSILON
}

/// Additional detailed checks that can be run separately
#[allow(dead_code)]
pub fn detailed_geometry_analysis(geometry: &Geometry) -> Result<()> {
    println!("=== Detailed Geometry Analysis ===");
    println!("Geometry label: {}", geometry.label);
    println!("Number of frames: {}", geometry.frames.len());

    // Frame statistics
    for (i, frame) in geometry.frames.iter().enumerate() {
        println!("Frame {} (ID {}):", i, frame.id);
        println!("  - Original frame: {}", frame.lumen.original_frame);
        println!("  - Lumen points: {}", frame.lumen.points.len());
        println!("  - Centroid: {:?}", frame.centroid);
        println!("  - Extra contours: {}", frame.extras.len());

        for (contour_type, contour) in &frame.extras {
            println!(
                "    * {}: {} points",
                contour_type.as_str(),
                contour.points.len()
            );
        }

        if frame.reference_point.is_some() {
            println!("  - Has reference point");
        }
    }

    let unique_original_frames: HashSet<u32> = geometry
        .frames
        .iter()
        .map(|f| f.lumen.original_frame)
        .collect();

    if unique_original_frames.len() != geometry.frames.len() {
        println!("WARNING: Duplicate original frames detected");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intravascular::io::geometry::{Contour, Frame};
    use std::collections::HashMap;

    fn create_test_contour_points(count: usize, frame_index: u32, z: f64) -> Vec<ContourPoint> {
        (0..count)
            .map(|i| ContourPoint {
                frame_index,
                point_index: i as u32,
                x: i as f64,
                y: i as f64 * 2.0,
                z,
                aortic: false,
            })
            .collect()
    }

    fn create_test_frame(id: u32, original_frame: u32, has_reference: bool, z: f64) -> Frame {
        let points = create_test_contour_points(4, original_frame, z);
        let centroid = compute_centroid_from_points(&points);

        Frame {
            id,
            centroid,
            lumen: Contour {
                id,
                original_frame,
                points: points.clone(),
                centroid: Some(centroid),
                aortic_thickness: None,
                pulmonary_thickness: None,
                kind: ContourType::Lumen,
            },
            extras: HashMap::new(),
            reference_point: if has_reference {
                Some(ContourPoint {
                    frame_index: original_frame,
                    point_index: 0,
                    x: centroid.0,
                    y: centroid.1,
                    z: centroid.2,
                    aortic: false,
                })
            } else {
                None
            },
        }
    }

    #[test]
    fn test_valid_geometry() {
        let mut geometry = Geometry {
            frames: vec![
                create_test_frame(0, 10, false, 0.0), // z=0.0 - should be proximal end
                create_test_frame(1, 11, true, 1.0),  // Only ONE frame has reference point
                create_test_frame(2, 12, false, 2.0),
            ],
            label: "test".to_string(),
        };
        geometry.ensure_proximal_at_position_zero();

        assert!(check_geometry_integrity(&geometry).is_ok());
    }

    #[test]
    fn test_non_consecutive_frame_ids() {
        let geometry = Geometry {
            frames: vec![
                create_test_frame(0, 10, false, 0.0),
                create_test_frame(2, 11, false, 1.0), // Missing ID 1
            ],
            label: "test".to_string(),
        };

        let result = check_geometry_integrity(&geometry);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("consecutive"));
    }

    #[test]
    fn test_missing_lumen() {
        let mut frame = create_test_frame(0, 10, false, 0.0);
        frame.lumen.points.clear(); // Empty lumen points

        let geometry = Geometry {
            frames: vec![frame],
            label: "test".to_string(),
        };

        let result = check_geometry_integrity(&geometry);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("no points"));
    }

    #[test]
    fn test_multiple_reference_points() {
        let geometry = Geometry {
            frames: vec![
                create_test_frame(0, 10, true, 0.0),
                create_test_frame(1, 11, true, 1.0), // Two frames with reference points
            ],
            label: "test".to_string(),
        };

        let result = check_geometry_integrity(&geometry);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("exactly one reference point"));
    }

    #[test]
    fn test_point_count_mismatch_across_frames() {
        let frame1 = create_test_frame(0, 10, true, 0.0);
        let mut frame2 = create_test_frame(1, 11, false, 1.0);

        // Frame2 lumen has different point count than frame1
        frame2.lumen.points = create_test_contour_points(5, 11, 1.0); // 5 points vs 4 in frame1

        let geometry = Geometry {
            frames: vec![frame1, frame2],
            label: "test".to_string(),
        };

        let result = check_geometry_integrity(&geometry);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Lumen point count mismatch"));
    }

    #[test]
    fn test_extra_contour_point_count_mismatch() {
        let mut frame1 = create_test_frame(0, 10, false, 0.0);
        let mut frame2 = create_test_frame(1, 11, true, 1.0);

        // Add catheter contour to both frames with consistent point counts
        let catheter_points1 = create_test_contour_points(6, 10, 0.0);
        let catheter_points2 = create_test_contour_points(6, 11, 1.0); // Same count (6)

        frame1.extras.insert(
            ContourType::Catheter,
            Contour {
                id: 0,
                original_frame: 10,
                points: catheter_points1,
                centroid: None,
                aortic_thickness: None,
                pulmonary_thickness: None,
                kind: ContourType::Catheter,
            },
        );

        frame2.extras.insert(
            ContourType::Catheter,
            Contour {
                id: 1,
                original_frame: 11,
                points: catheter_points2,
                centroid: None,
                aortic_thickness: None,
                pulmonary_thickness: None,
                kind: ContourType::Catheter,
            },
        );

        let mut geometry = Geometry {
            frames: vec![frame1.clone(), frame2.clone()],
            label: "test".to_string(),
        };
        geometry.ensure_proximal_at_position_zero();

        // This should pass - consistent point counts across frames
        assert!(check_geometry_integrity(&geometry).is_ok());

        // Now test with inconsistent point counts
        let mut frame3 = create_test_frame(2, 12, false, 2.0);
        frame3.extras.insert(
            ContourType::Catheter,
            Contour {
                id: 2,
                original_frame: 12,
                points: create_test_contour_points(8, 12, 2.0), // Different count (8 vs 6)
                centroid: None,
                aortic_thickness: None,
                pulmonary_thickness: None,
                kind: ContourType::Catheter,
            },
        );

        let geometry2 = Geometry {
            frames: vec![frame1, frame2, frame3],
            label: "test".to_string(),
        };

        let result = check_geometry_integrity(&geometry2);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Catheter contour point count mismatch"));
    }

    #[test]
    fn test_original_frame_mismatch() {
        let mut frame = create_test_frame(0, 10, true, 0.0);

        // Add an extra contour with wrong original_frame
        frame.extras.insert(
            ContourType::Eem,
            Contour {
                id: 0,
                original_frame: 99, // Different from lumen's 10
                points: create_test_contour_points(4, 10, 0.0),
                centroid: None,
                aortic_thickness: None,
                pulmonary_thickness: None,
                kind: ContourType::Eem,
            },
        );

        let geometry = Geometry {
            frames: vec![frame],
            label: "test".to_string(),
        };

        let result = check_geometry_integrity(&geometry);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Original frame mismatch"));
    }
}
