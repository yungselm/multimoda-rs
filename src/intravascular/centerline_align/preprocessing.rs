use nalgebra::Vector3;

use crate::intravascular::io::geometry::Geometry;
use crate::intravascular::io::input::{Centerline, CenterlinePoint, ContourPoint};

/// Resample `centerline` along its arc-length so that adjacent points are spaced at the
/// mean Euclidean distance between consecutive contour centroids in `ref_mesh`.
///
/// Precondition (expected caller behavior):
/// - `centerline` should be trimmed so the first point corresponds to the aortic start
///   (you already call `remove_leading_points_cl` before this).
/// - `centerline` should be in decreasing z-order if that matters (you call `ensure_descending_z`).
pub fn preprocess_centerline(
    centerline: Centerline,
    ref_mesh: &Geometry,
) -> Result<Centerline, &'static str> {
    let mut cl = centerline;
    ensure_descending_z(&mut cl);
    let cl = resample_centerline_by_contours(&cl, ref_mesh)?;
    Ok(cl)
}

fn ensure_descending_z(centerline: &mut Centerline) {
    if !centerline.points.is_empty() {
        let first_z = centerline.points[0].contour_point.z;
        let last_z = centerline.points.last().unwrap().contour_point.z;
        if first_z < last_z {
            centerline.points.reverse();
        }
    }
}

fn resample_centerline_by_contours(
    centerline: &Centerline,
    ref_mesh: &Geometry,
) -> Result<Centerline, &'static str> {
    if centerline.points.is_empty() {
        return Err("Centerline is empty");
    }
    if ref_mesh.frames.is_empty() {
        return Err("Reference mesh has no frames");
    }

    let (centroids, mean_spacing_opt) = calculate_mean_spacing(ref_mesh);

    let cum = cumulative_arc_length(centerline);
    let total_length = *cum.last().unwrap_or(&0.0);

    // Use n_segments = original segments count
    let n_segments = centerline.points.len().saturating_sub(1);

    let spacing = match decide_spacing(mean_spacing_opt, total_length, n_segments) {
        Some(s) => s,
        None => {
            eprintln!("resample_centerline_by_contours: invalid spacing computed, returning original centerline");
            return Ok(centerline.clone());
        }
    };

    eprintln!(
        "resample_centerline_by_contours: centroid_count={}, centroid_mean_spacing={:?}, centerline_length={}, spacing={:.6}",
        centroids.len(),
        mean_spacing_opt,
        total_length,
        spacing
    );

    let s_new = build_samples(total_length, spacing);

    let mut new_points = Vec::with_capacity(s_new.len());
    for (k, &target_s) in s_new.iter().enumerate() {
        new_points.push(interpolate_centerline_at_s(centerline, &cum, target_s, k));
    }

    eprintln!(
        "resample_centerline_by_contours: produced {} points",
        new_points.len()
    );

    Ok(Centerline { points: new_points })
}

fn cumulative_arc_length(centerline: &Centerline) -> Vec<f64> {
    let mut cum: Vec<f64> = Vec::with_capacity(centerline.points.len());
    if centerline.points.is_empty() {
        return cum;
    }
    cum.push(0.0f64);
    for i in 1..centerline.points.len() {
        let p0 = &centerline.points[i - 1].contour_point;
        let p1 = &centerline.points[i].contour_point;
        let dx = p1.x - p0.x;
        let dy = p1.y - p0.y;
        let dz = p1.z - p0.z;
        let d = (dx * dx + dy * dy + dz * dz).sqrt();
        cum.push(cum.last().unwrap() + d);
    }
    cum
}

fn decide_spacing(mean_opt: Option<f64>, total_length: f64, n_segments: usize) -> Option<f64> {
    if let Some(s) = mean_opt {
        if s.is_finite() && s > 1e-12 {
            return Some(s);
        }
    }
    if n_segments >= 1 {
        // n_segments is number of segments in original centerline: len-1
        let denom = n_segments as f64;
        let fallback = total_length / denom;
        if fallback.is_finite() && fallback > 1e-12 {
            return Some(fallback);
        }
    }
    None
}

fn build_samples(total_length: f64, spacing: f64) -> Vec<f64> {
    let mut s_new = Vec::new();
    let mut s = 0.0f64;
    let eps = 1e-9;
    while s <= total_length + eps {
        s_new.push(s);
        s += spacing;
    }
    if let Some(&last) = s_new.last() {
        if last > total_length + 1e-6 {
            s_new.pop();
            s_new.push(total_length);
        }
    }
    s_new
}

fn interpolate_centerline_at_s(
    centerline: &Centerline,
    cum: &[f64],
    target_s: f64,
    sample_index: usize,
) -> CenterlinePoint {
    // find segment idx
    let idx = match cum.binary_search_by(|v| v.partial_cmp(&target_s).unwrap()) {
        Ok(i) => i,          // exact match
        Err(0) => 0usize,    // before first
        Err(pos) => pos - 1, // segment index
    };

    // if at the very end, return last point
    if idx >= centerline.points.len().saturating_sub(1) {
        let last_pt = &centerline.points.last().unwrap().contour_point;
        let normal = centerline.points.last().unwrap().normal;
        return CenterlinePoint {
            contour_point: ContourPoint {
                frame_index: sample_index as u32,
                point_index: sample_index as u32,
                x: last_pt.x,
                y: last_pt.y,
                z: last_pt.z,
                aortic: false,
            },
            normal,
        };
    }

    // Interpolate between idx and idx+1
    let p0 = &centerline.points[idx].contour_point;
    let p1 = &centerline.points[idx + 1].contour_point;
    let s0 = cum[idx];
    let s1 = cum[idx + 1];
    let denom = s1 - s0;
    let t = if denom.abs() < 1e-12 {
        0.0
    } else {
        (target_s - s0) / denom
    };

    let x = p0.x + t * (p1.x - p0.x);
    let y = p0.y + t * (p1.y - p0.y);
    let z = p0.z + t * (p1.z - p0.z);

    // interpolate normal if available, else zeros
    let n0 = centerline.points[idx].normal;
    let n1 = centerline.points[idx + 1].normal;
    let mut normal = Vector3::zeros();
    if n0.norm() > 0.0 || n1.norm() > 0.0 {
        normal = n0 * (1.0 - t) + n1 * t;
        let n_norm = normal.norm();
        if n_norm > 1e-12 {
            normal /= n_norm;
        } else {
            normal = Vector3::zeros();
        }
    }

    CenterlinePoint {
        contour_point: ContourPoint {
            frame_index: sample_index as u32,
            point_index: sample_index as u32,
            x,
            y,
            z,
            aortic: false,
        },
        normal,
    }
}

fn calculate_mean_spacing(ref_mesh: &Geometry) -> (Vec<(f64, f64, f64)>, Option<f64>) {
    // 1) Compute centroid positions from ref_mesh
    let centroids: Vec<(f64, f64, f64)> = ref_mesh
        .frames
        .iter()
        .map(|f| (f.centroid.0, f.centroid.1, f.centroid.2))
        .collect();

    // 2) Compute distances between consecutive centroids (Euclidean)
    let centroid_dists: Vec<f64> = centroids
        .windows(2)
        .map(|w| {
            let dx = w[1].0 - w[0].0;
            let dy = w[1].1 - w[0].1;
            let dz = w[1].2 - w[0].2;
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .collect();

    // 3) Mean spacing from contours (fallbacks later)
    let mean_spacing_opt = if !centroid_dists.is_empty() {
        let sum: f64 = centroid_dists.iter().sum();
        let mean = sum / centroid_dists.len() as f64;
        if mean.is_finite() && mean > 1e-12 {
            Some(mean)
        } else {
            // Error: mean spacing is invalid
            eprintln!("calculate_mean_spacing: invalid mean spacing computed");
            None
        }
    } else {
        // Error: no centroid distances
        eprintln!("calculate_mean_spacing: no centroid distances found");
        None
    };
    (centroids, mean_spacing_opt)
}

#[cfg(test)]
mod cl_preprocessing_tests {
    use super::*;
    use crate::intravascular::io::input::ContourPoint;
    use approx::assert_relative_eq;

    #[test]
    fn test_ensure_descending_z() {
        let mut cl = Centerline {
            points: vec![
                CenterlinePoint {
                    contour_point: ContourPoint {
                        frame_index: 0,
                        point_index: 0,
                        x: 0.0,
                        y: 0.0,
                        z: 1.0,
                        aortic: false,
                    },
                    normal: Vector3::new(0.0, 0.0, -1.0),
                },
                CenterlinePoint {
                    contour_point: ContourPoint {
                        frame_index: 1,
                        point_index: 1,
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                        aortic: false,
                    },
                    normal: Vector3::new(0.0, 0.0, -1.0),
                },
            ],
        };
        ensure_descending_z(&mut cl);
        assert_eq!(cl.points[0].contour_point.z, 1.0);
        assert_eq!(cl.points[1].contour_point.z, 0.0);

        let mut cl = Centerline {
            points: vec![
                CenterlinePoint {
                    contour_point: ContourPoint {
                        frame_index: 0,
                        point_index: 0,
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                        aortic: false,
                    },
                    normal: Vector3::new(0.0, 0.0, -1.0),
                },
                CenterlinePoint {
                    contour_point: ContourPoint {
                        frame_index: 1,
                        point_index: 1,
                        x: 0.0,
                        y: 0.0,
                        z: 1.0,
                        aortic: false,
                    },
                    normal: Vector3::new(0.0, 0.0, -1.0),
                },
            ],
        };
        ensure_descending_z(&mut cl);
        assert_eq!(cl.points[0].contour_point.z, 1.0);
        assert_eq!(cl.points[1].contour_point.z, 0.0);
    }

    #[test]
    fn test_calculate_mean_spacing() {
        use crate::intravascular::io::geometry::{Contour, ContourType, Frame, Geometry};
        use crate::intravascular::io::input::ContourPoint;
        use std::collections::HashMap;

        // Case 1: multiple frames → valid mean spacing
        let geom = Geometry {
            frames: vec![
                Frame {
                    id: 0,
                    centroid: (0.0, 0.0, 0.0),
                    lumen: Contour {
                        id: 0,
                        original_frame: 0,
                        points: vec![ContourPoint {
                            frame_index: 0,
                            point_index: 0,
                            x: 0.0,
                            y: 0.0,
                            z: 0.0,
                            aortic: false,
                        }],
                        centroid: Some((0.0, 0.0, 0.0)),
                        aortic_thickness: None,
                        pulmonary_thickness: None,
                        kind: ContourType::Lumen,
                    },
                    extras: HashMap::new(),
                    reference_point: None,
                },
                Frame {
                    id: 1,
                    centroid: (3.0, 4.0, 0.0), // distance from (0,0,0) = 5
                    lumen: Contour {
                        id: 1,
                        original_frame: 1,
                        points: vec![],
                        centroid: Some((3.0, 4.0, 0.0)),
                        aortic_thickness: None,
                        pulmonary_thickness: None,
                        kind: ContourType::Lumen,
                    },
                    extras: HashMap::new(),
                    reference_point: None,
                },
                Frame {
                    id: 2,
                    centroid: (6.0, 8.0, 0.0), // distance from (3,4,0) = 5
                    lumen: Contour {
                        id: 2,
                        original_frame: 2,
                        points: vec![],
                        centroid: Some((6.0, 8.0, 0.0)),
                        aortic_thickness: None,
                        pulmonary_thickness: None,
                        kind: ContourType::Lumen,
                    },
                    extras: HashMap::new(),
                    reference_point: None,
                },
            ],
            label: "test".to_string(),
        };

        let (centroids, mean_opt) = calculate_mean_spacing(&geom);
        assert_eq!(centroids.len(), 3);
        assert_eq!(centroids[0], (0.0, 0.0, 0.0));
        assert_eq!(centroids[1], (3.0, 4.0, 0.0));
        assert_eq!(centroids[2], (6.0, 8.0, 0.0));

        // Mean of [5.0, 5.0] = 5.0
        assert_eq!(mean_opt, Some(5.0));

        // Case 2: single frame → no distances
        let geom2 = Geometry {
            frames: vec![Frame {
                id: 0,
                centroid: (1.0, 2.0, 3.0),
                lumen: Contour {
                    id: 0,
                    original_frame: 0,
                    points: vec![],
                    centroid: Some((1.0, 2.0, 3.0)),
                    aortic_thickness: None,
                    pulmonary_thickness: None,
                    kind: ContourType::Lumen,
                },
                extras: HashMap::new(),
                reference_point: None,
            }],
            label: "test2".to_string(),
        };
        let (centroids2, mean_opt2) = calculate_mean_spacing(&geom2);
        assert_eq!(centroids2.len(), 1);
        assert_eq!(centroids2[0], (1.0, 2.0, 3.0));
        assert!(mean_opt2.is_none());
    }

    #[test]
    fn test_cumulative_arc_length_and_decide_spacing() {
        // create a simple centerline along z = 0..3 with 4 points
        let cl = Centerline {
            points: vec![
                CenterlinePoint {
                    contour_point: ContourPoint {
                        frame_index: 0,
                        point_index: 0,
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                        aortic: false,
                    },
                    normal: Vector3::new(0.0, 0.0, 1.0),
                },
                CenterlinePoint {
                    contour_point: ContourPoint {
                        frame_index: 1,
                        point_index: 1,
                        x: 0.0,
                        y: 0.0,
                        z: 1.0,
                        aortic: false,
                    },
                    normal: Vector3::new(0.0, 0.0, 1.0),
                },
                CenterlinePoint {
                    contour_point: ContourPoint {
                        frame_index: 2,
                        point_index: 2,
                        x: 0.0,
                        y: 0.0,
                        z: 2.0,
                        aortic: false,
                    },
                    normal: Vector3::new(0.0, 0.0, 1.0),
                },
                CenterlinePoint {
                    contour_point: ContourPoint {
                        frame_index: 3,
                        point_index: 3,
                        x: 0.0,
                        y: 0.0,
                        z: 3.0,
                        aortic: false,
                    },
                    normal: Vector3::new(0.0, 0.0, 1.0),
                },
            ],
        };

        let cum = cumulative_arc_length(&cl);
        assert_eq!(cum, vec![0.0, 1.0, 2.0, 3.0]);

        let total_length = *cum.last().unwrap();
        // no centroid mean -> fallback spacing = total_length / (n_segments)
        let spacing = decide_spacing(None, total_length, cl.points.len() - 1).unwrap();
        assert_relative_eq!(spacing, 1.0);
    }

    #[test]
    fn test_build_samples_and_interpolate() {
        // same centerline
        let cl = Centerline {
            points: vec![
                CenterlinePoint {
                    contour_point: ContourPoint {
                        frame_index: 0,
                        point_index: 0,
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                        aortic: false,
                    },
                    normal: Vector3::new(0.0, 0.0, 1.0),
                },
                CenterlinePoint {
                    contour_point: ContourPoint {
                        frame_index: 1,
                        point_index: 1,
                        x: 0.0,
                        y: 0.0,
                        z: 1.0,
                        aortic: false,
                    },
                    normal: Vector3::new(0.0, 0.0, 1.0),
                },
                CenterlinePoint {
                    contour_point: ContourPoint {
                        frame_index: 2,
                        point_index: 2,
                        x: 0.0,
                        y: 0.0,
                        z: 2.0,
                        aortic: false,
                    },
                    normal: Vector3::new(0.0, 0.0, 1.0),
                },
                CenterlinePoint {
                    contour_point: ContourPoint {
                        frame_index: 3,
                        point_index: 3,
                        x: 0.0,
                        y: 0.0,
                        z: 3.0,
                        aortic: false,
                    },
                    normal: Vector3::new(0.0, 0.0, 1.0),
                },
            ],
        };

        let cum = cumulative_arc_length(&cl);
        let samples = build_samples(3.0, 0.75); // 0.0,0.75,1.5,2.25,3.0
        assert!((samples.len() >= 2) && samples[0] == 0.0 && *samples.last().unwrap() == 3.0);

        // interpolate at s = 1.5 (should be z = 1.5)
        let pt = interpolate_centerline_at_s(&cl, &cum, 1.5, 0);
        assert_relative_eq!(pt.contour_point.z, 1.5, epsilon = 1e-12);
        // normal should be normalized and in z direction
        assert_relative_eq!(pt.normal.z, 1.0, epsilon = 1e-12);
    }
}
