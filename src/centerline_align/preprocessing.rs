use nalgebra::Vector3;

use crate::io::input::{Centerline, CenterlinePoint, ContourPoint};
use crate::io::Geometry;
use crate::processing::align_between::GeometryPair;

pub fn ensure_descending_z(centerline: &mut Centerline) {
    if !centerline.points.is_empty() {
        let first_z = centerline.points[0].contour_point.z;
        let last_z = centerline.points.last().unwrap().contour_point.z;
        if first_z < last_z {
            centerline.points.reverse();
        }
    }
}

pub fn remove_leading_points_cl(
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

/// Resample `centerline` along its arc-length so that adjacent points are spaced at the
/// mean Euclidean distance between consecutive contour centroids in `ref_mesh`.
///
/// Precondition (expected caller behavior):
/// - `centerline` should be trimmed so the first point corresponds to the aortic start
///   (you already call `remove_leading_points_cl` before this).
/// - `centerline` should be in decreasing z-order if that matters (you call `ensure_descending_z`).
pub fn resample_centerline_by_contours(centerline: &Centerline, ref_mesh: &Geometry) -> Centerline {
    // If no centerline -> nothing to do
    if centerline.points.is_empty() {
        return Centerline { points: Vec::new() };
    }

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
        Some(sum / centroid_dists.len() as f64)
    } else {
        None
    };

    // 4) Compute cumulative arc-length along the centerline
    let mut cum: Vec<f64> = Vec::with_capacity(centerline.points.len());
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
    let total_length = *cum.last().unwrap();

    // 5) Decide spacing: prefer centroid mean, fallback to average centerline segment length
    let spacing = match mean_spacing_opt {
        Some(s) if s.is_finite() && s > 1e-12 => s,
        _ => {
            if centerline.points.len() > 1 {
                total_length / ((centerline.points.len() - 1) as f64)
            } else {
                // degenerate single-point centerline -> return it
                return centerline.clone();
            }
        }
    };

    if !(spacing.is_finite() && spacing > 1e-12) {
        // nothing we can do reliably
        eprintln!("resample_centerline_by_contours: invalid spacing computed, returning original centerline");
        return centerline.clone();
    }

    // Debug log (remove or lower verbosity later)
    eprintln!(
        "resample_centerline_by_contours: centroid_count={}, centroid_mean_spacing={:?}, centerline_length={}, spacing={:.6}",
        centroids.len(),
        mean_spacing_opt,
        total_length,
        spacing
    );

    // 6) Build target arc-length samples s = 0, spacing, 2*spacing, ... <= total_length
    let mut s_new: Vec<f64> = Vec::new();
    let mut s = 0.0;
    let eps = 1e-9;
    while s <= total_length + eps {
        s_new.push(s);
        s += spacing;
    }

    // Ensure last sample isn't just slightly beyond due to FP; clamp
    if let Some(&last) = s_new.last() {
        if last > total_length + 1e-6 {
            s_new.pop();
            s_new.push(total_length);
        }
    }

    // 7) For each target s, find segment index and interpolate position and normal
    let mut new_points: Vec<CenterlinePoint> = Vec::with_capacity(s_new.len());

    for (k, &target_s) in s_new.iter().enumerate() {
        // binary search to find first cum[idx] > target_s
        let idx = match cum.binary_search_by(|v| v.partial_cmp(&target_s).unwrap()) {
            Ok(i) => i,             // exact match
            Err(0) => 0usize,       // before first (shouldn't happen since cum[0]=0 and target_s >=0)
            Err(pos) => pos - 1,    // segment index
        };

        // if at the very end, return last point
        if idx >= centerline.points.len() - 1 {
            let last_pt = &centerline.points.last().unwrap().contour_point;
            let normal = centerline.points.last().unwrap().normal;
            new_points.push(CenterlinePoint {
                contour_point: ContourPoint {
                    frame_index: k as u32,
                    point_index: k as u32,
                    x: last_pt.x,
                    y: last_pt.y,
                    z: last_pt.z,
                    aortic: false,
                },
                normal,
            });
            continue;
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
        // If normals are both non-zero, interpolate and normalize
        if n0.norm() > 0.0 || n1.norm() > 0.0 {
            normal = n0 * (1.0 - t) + n1 * t;
            let n_norm = normal.norm();
            if n_norm > 1e-12 {
                normal /= n_norm;
            } else {
                normal = Vector3::zeros();
            }
        }

        new_points.push(CenterlinePoint {
            contour_point: ContourPoint {
                frame_index: k as u32,
                point_index: k as u32,
                x,
                y,
                z,
                aortic: false,
            },
            normal,
        });
    }

    eprintln!("resample_centerline_by_contours: produced {} points", new_points.len());

    Centerline { points: new_points }
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

    geom_pair.dia_geom = align_geometry(geom_pair.dia_geom);
    geom_pair.sys_geom = align_geometry(geom_pair.sys_geom);

    geom_pair
}
