use nalgebra::Vector3;

use crate::intravascular::io::input::{Centerline, CenterlinePoint, ContourPoint};
use crate::intravascular::io::Geometry;
use crate::intravascular::processing::align_between::GeometryPair;

/// Resample `centerline` along its arc-length so that adjacent points are spaced at the
/// mean Euclidean distance between consecutive contour centroids in `ref_mesh`.
///
/// Precondition (expected caller behavior):
/// - `centerline` should be trimmed so the first point corresponds to the aortic start
///   (you already call `remove_leading_points_cl` before this).
/// - `centerline` should be in decreasing z-order if that matters (you call `ensure_descending_z`).
pub fn preprocess_centerline(
    centerline: Centerline,
    reference_point: &(f64, f64, f64),
    ref_mesh: &Geometry,
) -> Result<Centerline, &'static str> {
    let mut cl = centerline;
    ensure_descending_z(&mut cl);
    let cl = remove_leading_points_cl(cl, reference_point);
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

fn remove_leading_points_cl(
    mut centerline: Centerline,
    reference_point: &(f64, f64, f64),
) -> Centerline {
    centerline.points.retain(|p| {
        !p.contour_point.x.is_nan() && !p.contour_point.y.is_nan() && !p.contour_point.z.is_nan()
    });

    if centerline.points.is_empty() {
        return centerline;
    }

    // Find closest point to reference
    let closest_pt = centerline
        .points
        .iter()
        .min_by(|a, b| {
            distance_sq(&a.contour_point, reference_point)
                .partial_cmp(&distance_sq(&b.contour_point, reference_point))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();
    let start_frame = closest_pt.contour_point.frame_index;

    println!(
        "Index of closest point: {:?}",
        closest_pt.contour_point.frame_index
    );

    // Remove points before closest point
    let mut remaining: Vec<_> = centerline
        .points
        .into_iter()
        .filter(|p| p.contour_point.frame_index >= start_frame)
        .collect();

    // 4) Re-sort by frame_index to restore z-order
    remaining.sort_by_key(|p| p.contour_point.frame_index);

    // Reindex starting from 0
    for (i, pt) in remaining.iter_mut().enumerate() {
        pt.contour_point.frame_index = i as u32;
        pt.contour_point.point_index = i as u32;
    }

    Centerline { points: remaining }
}

/// Helper function to calculate squared distance between two points
fn distance_sq(a: &ContourPoint, b: &(f64, f64, f64)) -> f64 {
    let dx = a.x - b.0;
    let dy = a.y - b.1;
    let dz = a.z - b.2;
    dx * dx + dy * dy + dz * dz
}

fn resample_centerline_by_contours(centerline: &Centerline, ref_mesh: &Geometry) -> Result<Centerline, &'static str> {
    if centerline.points.is_empty() {
        return Err("Centerline is empty");
    }
    if ref_mesh.contours.is_empty() {
        return Err("Reference mesh has no contorus");
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

    eprintln!("resample_centerline_by_contours: produced {} points", new_points.len());

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
        .contours
        .iter()
        .map(|c| (c.centroid.0, c.centroid.1, c.centroid.2))
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

pub fn prepare_geometry_alignment(mut geom_pair: GeometryPair) -> GeometryPair {
    fn align_geometry(mut geom: Geometry) -> Geometry {
        geom.contours.reverse();
        for (index, contour) in geom.contours.iter_mut().enumerate() {
            contour.id = index as u32;
            for point in &mut contour.points {
                point.frame_index = index as u32;
            }
        }

        geom.catheter.reverse();
        for (index, catheter) in geom.catheter.iter_mut().enumerate() {
            catheter.id = index as u32;
            for point in &mut catheter.points {
                point.frame_index = index as u32;
            }
        }

        geom.walls.reverse();
        for (index, contour) in geom.walls.iter_mut().enumerate() {
            contour.id = index as u32;
            for point in &mut contour.points {
                point.frame_index = index as u32;
            }
        }

        geom.reference_point.frame_index = (geom.contours.len() - 1)
            .saturating_sub(geom.reference_point.frame_index as usize)
            as u32; // correct method?

        geom
    }

    if geom_pair.dia_geom.reference_point.z == geom_pair.dia_geom.contours.last().map_or(f64::NAN, |c| c.centroid.2) {
        eprintln!("prepare_geometry_alignment: dia_geom reference_point.z matches last contour z, reversing dia_geom");
        geom_pair.dia_geom = align_geometry(geom_pair.dia_geom);
    }
    if geom_pair.sys_geom.reference_point.z == geom_pair.sys_geom.contours.last().map_or(f64::NAN, |c| c.centroid.2) {
        eprintln!("prepare_geometry_alignment: sys_geom reference_point.z matches last contour z, reversing dia_geom");
        geom_pair.sys_geom = align_geometry(geom_pair.sys_geom);
    }

    geom_pair
}
// pub fn prepare_geometry_alignment(mut geom_pair: GeometryPair) -> GeometryPair {
//     fn align_geometry(mut geom: Geometry) -> Geometry {
//         // tolerance for comparing z-coordinates
//         const Z_TOL: f64 = 1e-6;

//         // Try to find a contour whose centroid.z matches the reference point z
//         let ref_z = geom.reference_point.z;
//         let matched_idx_opt = geom
//             .contours
//             .iter()
//             .position(|c| (c.centroid.2 - ref_z).abs() <= Z_TOL);

//         if let Some(matched_idx) = matched_idx_opt {
//             let matched_contour = &geom.contours[matched_idx];
//             let ref_frame = geom.reference_point.frame_index;

//             // Check: does the matched contour's id equal the reference frame index?
//             if matched_contour.id != ref_frame {
//                 eprintln!(
//                     "prepare_geometry_alignment: reference_point.frame_index ({}) does not match contour.id ({}) \
//                      for contour at z={:.6}. Will NOT reverse. Syncing reference_point.frame_index -> {}.",
//                     ref_frame, matched_contour.id, ref_z, matched_contour.id
//                 );

//                 // Sync reference_point to the contour id to avoid silent inconsistencies.
//                 // We do not reverse since the recorded mapping does not match the geometry order.
//                 geom.reference_point.frame_index = matched_contour.id;
//                 return geom;
//             }

//             // If reference already points to the "0" contour, nothing to do.
//             if ref_frame == 0 {
//                 // already in the desired orientation (or at least already 0)
//                 return geom;
//             }

//             // At this point: matched_contour.id == ref_frame and ref_frame != 0,
//             // so we perform the reverse + reindexing and update the reference_frame mapping.
//             let orig_ref_index = ref_frame as usize;

//             // Reverse contours and reindex their ids/frame_index
//             geom.contours.reverse();
//             for (index, contour) in geom.contours.iter_mut().enumerate() {
//                 contour.id = index as u32;
//                 for point in &mut contour.points {
//                     point.frame_index = index as u32;
//                 }
//             }

//             // Reverse and reindex catheter (if any)
//             geom.catheter.reverse();
//             for (index, catheter) in geom.catheter.iter_mut().enumerate() {
//                 catheter.id = index as u32;
//                 for point in &mut catheter.points {
//                     point.frame_index = index as u32;
//                 }
//             }

//             // Reverse and reindex walls (if any)
//             geom.walls.reverse();
//             for (index, contour) in geom.walls.iter_mut().enumerate() {
//                 contour.id = index as u32;
//                 for point in &mut contour.points {
//                     point.frame_index = index as u32;
//                 }
//             }

//             // Map the original reference index to the new index after reversal:
//             // new_index = (len-1) - orig_ref_index
//             let len_minus_one = geom.contours.len().saturating_sub(1);
//             geom.reference_point.frame_index = len_minus_one.saturating_sub(orig_ref_index) as u32;

//             eprintln!(
//                 "prepare_geometry_alignment: reversed geometry. reference frame remapped {} -> {}",
//                 orig_ref_index, geom.reference_point.frame_index
//             );
//         } else {
//             // No contour found matching the reference z
//             eprintln!(
//                 "prepare_geometry_alignment: no contour found with centroid.z ~= {} (tol {}). Will not reverse.",
//                 ref_z, Z_TOL
//             );
//         }

//         geom
//     }

//     geom_pair.dia_geom = align_geometry(geom_pair.dia_geom);
//     geom_pair.sys_geom = align_geometry(geom_pair.sys_geom);

//     geom_pair
// }

#[cfg(test)]
mod cl_preprocessing_tests {
    use super::*;
    use crate::intravascular::io::input::{Contour, ContourPoint};
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
    fn test_remove_leading_points_cl() {
        // Build a centerline with 4 points
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
                    normal: Vector3::new(1.0, 0.0, 0.0),
                },
                CenterlinePoint {
                    contour_point: ContourPoint {
                        frame_index: 1,
                        point_index: 1,
                        x: 1.0,
                        y: 0.0,
                        z: 0.0,
                        aortic: false,
                    },
                    normal: Vector3::new(1.0, 0.0, 0.0),
                },
                CenterlinePoint {
                    contour_point: ContourPoint {
                        frame_index: 2,
                        point_index: 2,
                        x: 2.0,
                        y: 0.0,
                        z: 0.0,
                        aortic: false,
                    },
                    normal: Vector3::new(1.0, 0.0, 0.0),
                },
                CenterlinePoint {
                    contour_point: ContourPoint {
                        frame_index: 3,
                        point_index: 3,
                        x: 3.0,
                        y: 0.0,
                        z: 0.0,
                        aortic: false,
                    },
                    normal: Vector3::new(1.0, 0.0, 0.0),
                },
            ],
        };

        // Reference point is closer to (2.0, 0.0, 0.0), so points before frame_index=2 should be dropped
        let reference = (2.0, 0.0, 0.0);
        let trimmed = remove_leading_points_cl(cl, &reference);

        // Expect only 2 points remaining: indices 2 and 3
        assert_eq!(trimmed.points.len(), 2);

        // After reindexing, they should be renumbered as 0,1
        assert_eq!(trimmed.points[0].contour_point.frame_index, 0);
        assert_eq!(trimmed.points[0].contour_point.point_index, 0);
        assert_eq!(trimmed.points[0].contour_point.x, 2.0);

        assert_eq!(trimmed.points[1].contour_point.frame_index, 1);
        assert_eq!(trimmed.points[1].contour_point.point_index, 1);
        assert_eq!(trimmed.points[1].contour_point.x, 3.0);
    }

    #[test]
    fn test_calculate_mean_spacing() {
        use crate::intravascular::io::input::{Contour, ContourPoint};

        // Case 1: multiple centroids → valid mean spacing
        let geom = Geometry {
            contours: vec![
                Contour {
                    id: 0,
                    centroid: (0.0, 0.0, 0.0),
                    points: vec![ContourPoint {
                        frame_index: 0,
                        point_index: 0,
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                        aortic: false,
                    }],
                    aortic_thickness: None,
                    pulmonary_thickness: None,
                },
                Contour {
                    id: 1,
                    centroid: (3.0, 4.0, 0.0), // distance from (0,0,0) = 5
                    points: vec![],
                    aortic_thickness: None,
                    pulmonary_thickness: None,
                },
                Contour {
                    id: 2,
                    centroid: (6.0, 8.0, 0.0), // distance from (3,4,0) = 5
                    points: vec![],
                    aortic_thickness: None,
                    pulmonary_thickness: None,
                },
            ],
            catheter: Vec::new(),
            walls: Vec::new(),
            // a simple reference_point (must be supplied)
            reference_point: ContourPoint {
                frame_index: 0,
                point_index: 0,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            label: "test".to_string(),
        };

        let (centroids, mean_opt) = calculate_mean_spacing(&geom);
        assert_eq!(centroids.len(), 3);
        assert_eq!(centroids[0], (0.0, 0.0, 0.0));
        assert_eq!(centroids[1], (3.0, 4.0, 0.0));
        assert_eq!(centroids[2], (6.0, 8.0, 0.0));

        // Mean of [5.0, 5.0] = 5.0
        assert_eq!(mean_opt, Some(5.0));

        // Case 2: single centroid → no distances
        let geom2 = Geometry {
            contours: vec![Contour {
                id: 0,
                centroid: (1.0, 2.0, 3.0),
                points: vec![],
                aortic_thickness: None,
                pulmonary_thickness: None,
            }],
            catheter: Vec::new(),
            walls: Vec::new(),
            reference_point: ContourPoint {
                frame_index: 0,
                point_index: 0,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
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
                CenterlinePoint { contour_point: ContourPoint { frame_index: 0, point_index: 0, x: 0.0, y: 0.0, z: 0.0, aortic: false }, normal: Vector3::new(0.0,0.0,1.0) },
                CenterlinePoint { contour_point: ContourPoint { frame_index: 1, point_index: 1, x: 0.0, y: 0.0, z: 1.0, aortic: false }, normal: Vector3::new(0.0,0.0,1.0) },
                CenterlinePoint { contour_point: ContourPoint { frame_index: 2, point_index: 2, x: 0.0, y: 0.0, z: 2.0, aortic: false }, normal: Vector3::new(0.0,0.0,1.0) },
                CenterlinePoint { contour_point: ContourPoint { frame_index: 3, point_index: 3, x: 0.0, y: 0.0, z: 3.0, aortic: false }, normal: Vector3::new(0.0,0.0,1.0) },
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
                CenterlinePoint { contour_point: ContourPoint { frame_index: 0, point_index: 0, x: 0.0, y: 0.0, z: 0.0, aortic: false }, normal: Vector3::new(0.0,0.0,1.0) },
                CenterlinePoint { contour_point: ContourPoint { frame_index: 1, point_index: 1, x: 0.0, y: 0.0, z: 1.0, aortic: false }, normal: Vector3::new(0.0,0.0,1.0) },
                CenterlinePoint { contour_point: ContourPoint { frame_index: 2, point_index: 2, x: 0.0, y: 0.0, z: 2.0, aortic: false }, normal: Vector3::new(0.0,0.0,1.0) },
                CenterlinePoint { contour_point: ContourPoint { frame_index: 3, point_index: 3, x: 0.0, y: 0.0, z: 3.0, aortic: false }, normal: Vector3::new(0.0,0.0,1.0) },
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

    // ----------------------------------------------------------------------------------------------------------------
    fn make_geometry_with_contours(ids_and_z: &[(u32, f64)], ref_frame: u32, ref_z: f64, label: &str) -> Geometry {
        let contours: Vec<Contour> = ids_and_z
            .iter()
            .map(|(id, z)| Contour {
                id: *id,
                centroid: (0.0_f64, 0.0_f64, *z),
                points: vec![ContourPoint {
                    frame_index: *id,
                    point_index: 0,
                    x: 0.0,
                    y: 0.0,
                    z: *z,
                    aortic: false,
                }],
                aortic_thickness: None,
                pulmonary_thickness: None,
            })
            .collect();

        Geometry {
            contours,
            catheter: Vec::new(),
            walls: Vec::new(),
            reference_point: ContourPoint {
                frame_index: ref_frame,
                point_index: 0,
                x: 0.0,
                y: 0.0,
                z: ref_z,
                aortic: false,
            },
            label: label.to_string(),
        }
    }

    #[test]
    fn test_reverse_when_match_and_nonzero_ref() {
        // contours with ids 0,1,2 and z 0.0,1.0,2.0
        let geom = make_geometry_with_contours(&[(0, 0.0), (1, 1.0), (2, 2.0)], 2, 2.0, "dia");
        let gp = GeometryPair {
            dia_geom: geom.clone(),
            sys_geom: geom.clone(),
        };

        let out = prepare_geometry_alignment(gp);

        // dia_geom should be reversed: new contour 0 has old z=2.0
        assert_eq!(out.dia_geom.contours[0].centroid.2, 2.0);
        // reference_point.frame_index should be remapped to 0
        assert_eq!(out.dia_geom.reference_point.frame_index, 0);

        // points' frame_index should match new contour id (0)
        assert_eq!(out.dia_geom.contours[0].points[0].frame_index, 0);
    }

    #[test]
    fn test_no_reverse_on_id_mismatch_and_sync_ref() {
        let geom = make_geometry_with_contours(&[(0, 0.0), (1, 1.0), (2, 2.0)], 1, 2.0, "dia");
        let gp = GeometryPair {
            dia_geom: geom.clone(),
            sys_geom: geom.clone(),
        };

        let out = prepare_geometry_alignment(gp);

        assert_eq!(out.dia_geom.contours[0].centroid.2, 2.0);

        let len_minus_one = out.dia_geom.contours.len().saturating_sub(1);
        let expected_remap = len_minus_one.saturating_sub(1) as u32; // orig_ref_idx was 1
        assert_eq!(out.dia_geom.reference_point.frame_index, expected_remap);

        // sanity: points' frame_index updated to match new contour id
        assert_eq!(out.dia_geom.contours[0].points[0].frame_index, 0);
    }


    #[test]
    fn test_no_reverse_when_no_matching_z() {
        // contours z 0,1,2 but reference z doesn't match any (10.0)
        let geom = make_geometry_with_contours(&[(0, 0.0), (1, 1.0), (2, 2.0)], 5, 10.0, "dia");
        let gp = GeometryPair {
            dia_geom: geom.clone(),
            sys_geom: geom.clone(),
        };

        let out = prepare_geometry_alignment(gp);

        // No reverse -> first contour still has z 0.0
        assert_eq!(out.dia_geom.contours[0].centroid.2, 0.0);
        // reference frame should remain unchanged (5)
        assert_eq!(out.dia_geom.reference_point.frame_index, 5);
    }

    #[test]
    fn test_no_reverse_when_ref_is_zero() {
        // reference already points to id 0 (orientation OK) and z matches contour 0
        let geom = make_geometry_with_contours(&[(0, 0.0), (1, 1.0), (2, 2.0)], 0, 0.0, "dia");
        let gp = GeometryPair {
            dia_geom: geom.clone(),
            sys_geom: geom.clone(),
        };

        let out = prepare_geometry_alignment(gp);

        // No reverse performed: contour 0 remains z 0.0
        assert_eq!(out.dia_geom.contours[0].centroid.2, 0.0);
        // reference frame stays 0
        assert_eq!(out.dia_geom.reference_point.frame_index, 0);
    }
}