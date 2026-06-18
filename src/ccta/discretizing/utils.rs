use crate::types::native::{Centerline, CenterlinePoint, ContourPoint};
use nalgebra::Vector3;

/// Smooths centerline positions with a Gaussian kernel (per branch) and recomputes normals.
///
/// `sigma` is the half-width in number of centerline points.  A value of 1.0 is a gentle
/// neighbourhood average; 3–5 removes noise while keeping the overall vessel path; larger
/// values heavily round corners.  Branches are processed independently so no smoothing
/// bleeds across the bifurcation.
pub fn smooth_centerline(centerline: &Centerline, sigma: f64) -> Centerline {
    if centerline.points.is_empty() || sigma < 1e-12 {
        return centerline.clone();
    }

    let n = centerline.points.len();
    let max_branch = centerline
        .points
        .iter()
        .map(|p| p.branch_id)
        .max()
        .unwrap_or(0);

    let mut sx = vec![0.0f64; n];
    let mut sy = vec![0.0f64; n];
    let mut sz = vec![0.0f64; n];

    for branch_id in 0..=max_branch {
        let indices: Vec<usize> = centerline
            .points
            .iter()
            .enumerate()
            .filter(|(_, p)| p.branch_id == branch_id)
            .map(|(i, _)| i)
            .collect();

        if indices.is_empty() {
            continue;
        }

        // Truncate kernel at 3σ to avoid O(n²) cost on long vessels.
        let radius = (3.0 * sigma).ceil() as usize;

        for (li, &gi) in indices.iter().enumerate() {
            // Symmetric truncation: equal radius on both sides so that a
            // linear trend is preserved exactly (weighted mean of symmetric
            // neighbours always equals the centre value).
            let sym_r = li.min(radius).min(indices.len() - 1 - li);
            let j_start = li - sym_r;
            let j_end = li + sym_r + 1;
            let (mut wx, mut wy, mut wz, mut wt) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);

            for (k, &gi_j) in indices[j_start..j_end].iter().enumerate() {
                let j = j_start + k;
                let diff = (li as f64) - (j as f64);
                let w = (-0.5 * diff * diff / (sigma * sigma)).exp();
                let pt = &centerline.points[gi_j].contour_point;
                wx += w * pt.x;
                wy += w * pt.y;
                wz += w * pt.z;
                wt += w;
            }

            if wt > 1e-12 {
                sx[gi] = wx / wt;
                sy[gi] = wy / wt;
                sz[gi] = wz / wt;
            } else {
                let pt = &centerline.points[gi].contour_point;
                sx[gi] = pt.x;
                sy[gi] = pt.y;
                sz[gi] = pt.z;
            }
        }
    }

    // Build new points with smoothed positions (normals are placeholder for now).
    let mut new_points: Vec<CenterlinePoint> = centerline
        .points
        .iter()
        .enumerate()
        .map(|(i, p)| CenterlinePoint {
            contour_point: ContourPoint {
                x: sx[i],
                y: sy[i],
                z: sz[i],
                ..p.contour_point
            },
            tangent: p.tangent,
            branch_id: p.branch_id,
        })
        .collect();

    // Recompute tangents from smoothed positions per branch.
    for branch_id in 0..=max_branch {
        let indices: Vec<usize> = new_points
            .iter()
            .enumerate()
            .filter(|(_, p)| p.branch_id == branch_id)
            .map(|(i, _)| i)
            .collect();

        if indices.is_empty() {
            continue;
        }

        // Compute forward-difference tangents; last point is handled separately.
        let branch_tangents: Vec<Vector3<f64>> = indices
            .iter()
            .enumerate()
            .map(|(li, &gi)| {
                if li + 1 < indices.len() {
                    let ni = indices[li + 1];
                    let curr = &new_points[gi].contour_point;
                    let next = &new_points[ni].contour_point;
                    let v = Vector3::new(next.x - curr.x, next.y - curr.y, next.z - curr.z);
                    let norm = v.norm();
                    if norm > 1e-12 {
                        v / norm
                    } else {
                        new_points[gi].tangent
                    }
                } else {
                    Vector3::zeros() // placeholder; filled below from the previous entry
                }
            })
            .collect();

        let last_tangent = if indices.len() >= 2 {
            branch_tangents[indices.len() - 2]
        } else {
            new_points[indices[0]].tangent
        };

        for (li, &gi) in indices.iter().enumerate() {
            new_points[gi].tangent = if li + 1 < indices.len() {
                branch_tangents[li]
            } else {
                last_tangent
            };
        }
    }

    Centerline {
        points: new_points,
        branch_start_indices: centerline.branch_start_indices.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::native::ContourPoint;

    fn make_centerline(pts: &[(f64, f64, f64)]) -> Centerline {
        let contour_pts: Vec<ContourPoint> = pts
            .iter()
            .enumerate()
            .map(|(i, &(x, y, z))| ContourPoint {
                frame_index: i as u32,
                point_index: i as u32,
                x,
                y,
                z,
                aortic: false,
            })
            .collect();
        Centerline::from_contour_points(contour_pts)
    }

    #[test]
    fn straight_line_is_unchanged() {
        // A perfectly straight line should not move after smoothing.
        let pts: Vec<(f64, f64, f64)> = (0..20).map(|i| (i as f64, 0.0, 0.0)).collect();
        let cl = make_centerline(&pts);
        let smoothed = smooth_centerline(&cl, 3.0);

        for (orig, sm) in cl.points.iter().zip(smoothed.points.iter()) {
            let dx = (orig.contour_point.x - sm.contour_point.x).abs();
            let dy = (orig.contour_point.y - sm.contour_point.y).abs();
            let dz = (orig.contour_point.z - sm.contour_point.z).abs();
            assert!(
                dx < 1e-10 && dy < 1e-10 && dz < 1e-10,
                "straight line moved"
            );
        }
    }

    #[test]
    fn spike_is_damped() {
        // Insert a sharp lateral spike at position 5 in an otherwise straight line.
        let mut pts: Vec<(f64, f64, f64)> = (0..15).map(|i| (i as f64, 0.0, 0.0)).collect();
        pts[7] = (7.0, 5.0, 0.0); // spike
        let cl = make_centerline(&pts);
        let smoothed = smooth_centerline(&cl, 2.0);

        let spike_y = smoothed.points[7].contour_point.y;
        assert!(spike_y < 5.0, "spike should be damped, got y = {spike_y}");
        assert!(spike_y > 0.0, "spike should not be fully erased");
    }

    #[test]
    fn normals_are_unit_vectors() {
        let mut pts: Vec<(f64, f64, f64)> = (0..20).map(|i| (i as f64, 0.0, 0.0)).collect();
        pts[10] = (10.0, 3.0, 0.0);
        let cl = make_centerline(&pts);
        let smoothed = smooth_centerline(&cl, 2.0);

        for p in &smoothed.points {
            let len = p.tangent.norm();
            assert!(
                (len - 1.0).abs() < 1e-10 || len < 1e-12,
                "tangent not unit: {len}"
            );
        }
    }

    #[test]
    fn sigma_zero_returns_clone() {
        let pts: Vec<(f64, f64, f64)> = (0..10).map(|i| (i as f64, 0.0, 0.0)).collect();
        let cl = make_centerline(&pts);
        let smoothed = smooth_centerline(&cl, 0.0);
        assert_eq!(cl, smoothed);
    }
}
