use anyhow::{anyhow, Result};
use std::collections::{HashMap, HashSet};

use super::geometry::{ContourType, Geometry};
use super::input::ContourPoint;

/// Performs integrity checks on a Geometry structure
pub fn check_geometry_integrity(geometry: &Geometry) -> Result<()> {
    if geometry.frames.is_empty() {
        return Err(anyhow!("Geometry has no frames"));
    }

    check_frame_ids_consecutive(geometry)?;
    check_centroids_match(geometry)?;
    check_lumen_presence(geometry)?;
    check_reference_point(geometry)?;
    check_contour_point_counts(geometry)?;
    check_original_frame_consistency(geometry)?;
    check_proximal_end_index(geometry)?;
    check_z_distribution(geometry)?;

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

        // Compute lumen centroid from points if not present
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

        // Also check that the stored lumen centroid matches computed if present
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

/// Check that exactly one reference point exists in the geometry
fn check_reference_point(geometry: &Geometry) -> Result<()> {
    let reference_frames: Vec<u32> = geometry
        .frames
        .iter()
        .filter_map(|frame| frame.reference_point.as_ref().map(|_| frame.id))
        .collect();

    match reference_frames.len() {
        0 => Err(anyhow!("No reference point found in geometry")),
        1 => Ok(()),
        n => Err(anyhow!("Expected exactly one reference point, found {}", n)),
    }
}

/// Check that all contours (lumen and extras) maintain consistent point counts across all frames
fn check_contour_point_counts(geometry: &Geometry) -> Result<()> {
    let mut point_counts: HashMap<ContourType, usize> = HashMap::new();

    // First frame sets the expected counts
    if let Some(first_frame) = geometry.frames.first() {
        point_counts.insert(ContourType::Lumen, first_frame.lumen.points.len());
        for (_, contour) in &first_frame.extras {
            point_counts.insert(contour.kind, contour.points.len());
        }
    }

    // Check all other frames
    for (frame_index, frame) in geometry.frames.iter().enumerate() {
        // Check lumen
        if let Some(&expected_count) = point_counts.get(&ContourType::Lumen) {
            if frame.lumen.points.len() != expected_count {
                return Err(anyhow!(
                    "Lumen point count mismatch in frame {} (ID {}). Expected {}, found {}",
                    frame_index,
                    frame.id,
                    expected_count,
                    frame.lumen.points.len()
                ));
            }
        }

        // Check extras
        for (_, contour) in &frame.extras {
            if let Some(&expected_count) = point_counts.get(&contour.kind) {
                if contour.points.len() != expected_count {
                    return Err(anyhow!(
                        "{:?} contour point count mismatch in frame {} (ID {}). Expected {}, found {}",
                        contour.kind,
                        frame_index,
                        frame.id,
                        expected_count,
                        contour.points.len()
                    ));
                }
            } else {
                // This contour type wasn't in the first frame, add it
                point_counts.insert(contour.kind, contour.points.len());
            }
        }
    }
    Ok(())
}

/// Check that lumen, extras, and reference point have the same original_frame within each frame
fn check_original_frame_consistency(geometry: &Geometry) -> Result<()> {
    for (frame_index, frame) in geometry.frames.iter().enumerate() {
        let expected_original_frame = frame.lumen.original_frame;

        // Check extras
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

        // Check reference point
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

/// check that proximal end has index 0
fn check_proximal_end_index(geometry: &Geometry) -> Result<()> {
    let proximal_idx = geometry.find_proximal_end_idx();
    if proximal_idx != 0 {
        return Err(anyhow!(
            "Proximal end index is {}, expected 0",
            proximal_idx
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

    // Check for unique original frames
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

    fn create_test_frame(id: u32, original_frame: u32, has_reference: bool) -> Frame {
        let points = create_test_contour_points(4, original_frame, id as f64);
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
        let geometry = Geometry {
            frames: vec![
                create_test_frame(0, 10, false),
                create_test_frame(1, 11, true), // Only one frame has reference point
                create_test_frame(2, 12, false),
            ],
            label: "test".to_string(),
        };

        assert!(check_geometry_integrity(&geometry).is_ok());
    }

    #[test]
    fn test_non_consecutive_frame_ids() {
        let geometry = Geometry {
            frames: vec![
                create_test_frame(0, 10, false),
                create_test_frame(2, 11, false), // Missing ID 1
            ],
            label: "test".to_string(),
        };

        let result = check_geometry_integrity(&geometry);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("consecutive"));
    }

    #[test]
    fn test_missing_lumen() {
        let mut frame = create_test_frame(0, 10, false);
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
                create_test_frame(0, 10, true),
                create_test_frame(1, 11, true), // Two frames with reference points
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
        let frame1 = create_test_frame(0, 10, false);
        let mut frame2 = create_test_frame(1, 11, false);

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
        let mut frame1 = create_test_frame(0, 10, false);
        let mut frame2 = create_test_frame(1, 11, false);

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

        let geometry = Geometry {
            frames: vec![frame1.clone(), frame2.clone()],
            label: "test".to_string(),
        };

        // This should pass - consistent point counts across frames
        assert!(check_geometry_integrity(&geometry).is_ok());

        // Now test with inconsistent point counts
        let mut frame3 = create_test_frame(2, 12, false);
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
        let mut frame = create_test_frame(0, 10, false);

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
