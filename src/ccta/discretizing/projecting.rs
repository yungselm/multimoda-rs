use crate::intravascular::io::geometry::{Contour, ContourType};
use crate::intravascular::io::input::{Centerline, CenterlinePoint, ContourPoint};
use nalgebra::Vector3;

/// Walks branch `branch_id` at uniform arc-length steps of `step_size`, assigns each mesh point
/// to its geometrically closest anchor via Voronoi partitioning, projects it onto that anchor's
/// perpendicular plane, and returns one `Contour` per sampled position.
///
/// Voronoi assignment prevents far-away vessel sections from contaminating a slice: a point that
/// is physically on a distant part of the vessel will always be closer (in 3-D) to the anchors on
/// that distant section, so it never ends up in the wrong cross-section.
pub fn walk_centerline_slices(
    centerline: &Centerline,
    points: &[(f64, f64, f64)],
    branch_id: u32,
    step_size: f64,
) -> Vec<Contour> {
    let branch_pts: Vec<&CenterlinePoint> = centerline
        .points
        .iter()
        .filter(|p| p.branch_id == branch_id)
        .collect();
    if branch_pts.is_empty() {
        return vec![];
    }

    let cum = branch_cum_arc(&branch_pts);
    let total = *cum.last().unwrap();
    let sample_positions = build_sample_positions(total, step_size);

    let anchors: Vec<CenterlinePoint> = sample_positions
        .iter()
        .enumerate()
        .map(|(k, &s)| interpolate_branch_at_s(&branch_pts, &cum, s, k))
        .collect();

    if anchors.is_empty() {
        return vec![];
    }

    // Collect raw centerline points from every OTHER branch for competitive Voronoi.
    // A mesh point is only assigned to the target branch if no other-branch centerline
    // point is closer (in 3-D) than the nearest target-branch anchor.
    let other_pts: Vec<&CenterlinePoint> = centerline
        .points
        .iter()
        .filter(|p| p.branch_id != branch_id)
        .collect();

    // Voronoi: each mesh point goes to its geometrically closest anchor.
    let mut buckets: Vec<Vec<ContourPoint>> = vec![vec![]; anchors.len()];
    for &(px, py, pz) in points {
        let closest = anchors.iter().enumerate().min_by(|(_, a), (_, b)| {
            let da = sq_dist3(
                px,
                py,
                pz,
                a.contour_point.x,
                a.contour_point.y,
                a.contour_point.z,
            );
            let db = sq_dist3(
                px,
                py,
                pz,
                b.contour_point.x,
                b.contour_point.y,
                b.contour_point.z,
            );
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });
        if let Some((anchor_idx, anchor)) = closest {
            let own_dist = sq_dist3(
                px,
                py,
                pz,
                anchor.contour_point.x,
                anchor.contour_point.y,
                anchor.contour_point.z,
            );
            // Reject the point if any other-branch centerline point is closer.
            let outcompeted = other_pts.iter().any(|op| {
                sq_dist3(
                    px,
                    py,
                    pz,
                    op.contour_point.x,
                    op.contour_point.y,
                    op.contour_point.z,
                ) < own_dist
            });
            if !outcompeted {
                let (qx, qy, qz) = project_to_plane((px, py, pz), anchor);
                let point_index = buckets[anchor_idx].len() as u32;
                buckets[anchor_idx].push(ContourPoint {
                    frame_index: anchor_idx as u32,
                    point_index,
                    x: qx,
                    y: qy,
                    z: qz,
                    aortic: false,
                });
            }
        }
    }

    anchors
        .into_iter()
        .zip(buckets)
        .enumerate()
        .map(|(k, (anchor, pts))| Contour {
            id: k as u32,
            original_frame: anchor.contour_point.frame_index,
            centroid: Some((
                anchor.contour_point.x,
                anchor.contour_point.y,
                anchor.contour_point.z,
            )),
            points: pts,
            aortic_thickness: None,
            pulmonary_thickness: None,
            kind: ContourType::Lumen,
        })
        .collect()
}

// Projects a point onto the plane perpendicular to `anchor` at its position.
// Formula: p_proj = p − ((p − center) · n̂) · n̂
fn project_to_plane(point: (f64, f64, f64), anchor: &CenterlinePoint) -> (f64, f64, f64) {
    let center = Vector3::new(
        anchor.contour_point.x,
        anchor.contour_point.y,
        anchor.contour_point.z,
    );
    let p = Vector3::new(point.0, point.1, point.2);
    let n = anchor.normal;
    let proj = p - n * (p - center).dot(&n);
    (proj.x, proj.y, proj.z)
}

fn sq_dist3(ax: f64, ay: f64, az: f64, bx: f64, by: f64, bz: f64) -> f64 {
    (ax - bx).powi(2) + (ay - by).powi(2) + (az - bz).powi(2)
}

fn branch_cum_arc(pts: &[&CenterlinePoint]) -> Vec<f64> {
    let mut cum = vec![0.0f64];
    for i in 1..pts.len() {
        let d = pts[i - 1].contour_point.distance_to(&pts[i].contour_point);
        cum.push(cum.last().unwrap() + d);
    }
    cum
}

fn build_sample_positions(total: f64, step: f64) -> Vec<f64> {
    let mut positions = Vec::new();
    let mut s = 0.0f64;
    while s <= total + 1e-9 {
        positions.push(s);
        s += step;
    }
    if let Some(&last) = positions.last() {
        if last > total + 1e-6 {
            positions.pop();
            positions.push(total);
        }
    }
    positions
}

fn interpolate_branch_at_s(
    pts: &[&CenterlinePoint],
    cum: &[f64],
    target_s: f64,
    idx_out: usize,
) -> CenterlinePoint {
    let seg = match cum.binary_search_by(|v| v.partial_cmp(&target_s).unwrap()) {
        Ok(i) => i,
        Err(0) => 0,
        Err(pos) => pos - 1,
    };

    if seg >= pts.len().saturating_sub(1) {
        let last = pts.last().unwrap();
        return CenterlinePoint {
            contour_point: ContourPoint {
                frame_index: idx_out as u32,
                point_index: idx_out as u32,
                ..last.contour_point
            },
            normal: last.normal,
            branch_id: last.branch_id,
        };
    }

    let p0 = &pts[seg].contour_point;
    let p1 = &pts[seg + 1].contour_point;
    let s0 = cum[seg];
    let s1 = cum[seg + 1];
    let t = if (s1 - s0).abs() < 1e-12 {
        0.0
    } else {
        (target_s - s0) / (s1 - s0)
    };

    let n0 = pts[seg].normal;
    let n1 = pts[seg + 1].normal;
    let mut normal = n0 * (1.0 - t) + n1 * t;
    let n_norm = normal.norm();
    if n_norm > 1e-12 {
        normal /= n_norm;
    }

    CenterlinePoint {
        contour_point: ContourPoint {
            frame_index: idx_out as u32,
            point_index: idx_out as u32,
            x: p0.x + t * (p1.x - p0.x),
            y: p0.y + t * (p1.y - p0.y),
            z: p0.z + t * (p1.z - p0.z),
            aortic: false,
        },
        normal,
        branch_id: pts[seg].branch_id,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intravascular::io::input::{Centerline, CenterlinePoint, ContourPoint};
    use nalgebra::Vector3;

    fn cp(idx: u32, x: f64, y: f64, z: f64) -> ContourPoint {
        ContourPoint {
            frame_index: idx,
            point_index: idx,
            x,
            y,
            z,
            aortic: false,
        }
    }

    fn cl_pt(frame_idx: u32, x: f64, y: f64, z: f64, nx: f64, ny: f64, nz: f64) -> CenterlinePoint {
        CenterlinePoint {
            contour_point: cp(frame_idx, x, y, z),
            normal: Vector3::new(nx, ny, nz).normalize(),
            branch_id: 0,
        }
    }

    fn z_centerline(n: usize) -> Centerline {
        Centerline {
            points: (0..n)
                .map(|i| cl_pt(i as u32, 0.0, 0.0, i as f64, 0.0, 0.0, 1.0))
                .collect(),
            branch_start_indices: vec![0],
        }
    }

    /// Ring of points on a cylinder of `radius` around Z at height `z`, with deterministic jitter.
    fn cylinder_ring(
        z: f64,
        radius: f64,
        n: usize,
        jitter: f64,
        seed: u64,
    ) -> Vec<(f64, f64, f64)> {
        use std::f64::consts::TAU;
        (0..n)
            .map(|i| {
                let angle = TAU * i as f64 / n as f64;
                let lcg = |s: u64| -> f64 {
                    let v = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (v >> 33) as f64 / u32::MAX as f64 - 0.5
                };
                let s = seed + i as u64;
                (
                    radius * angle.cos() + jitter * lcg(s),
                    radius * angle.sin() + jitter * lcg(s ^ 0xdead),
                    z + jitter * lcg(s ^ 0xbeef),
                )
            })
            .collect()
    }

    fn plane_dist(p: (f64, f64, f64), center: (f64, f64, f64), n: Vector3<f64>) -> f64 {
        let v = Vector3::new(p.0 - center.0, p.1 - center.1, p.2 - center.2);
        v.dot(&n)
    }

    fn pt_to_tuple(p: &ContourPoint) -> (f64, f64, f64) {
        (p.x, p.y, p.z)
    }

    fn cl_center(c: &CenterlinePoint) -> (f64, f64, f64) {
        (c.contour_point.x, c.contour_point.y, c.contour_point.z)
    }

    // ---- project_to_plane ----

    #[test]
    fn test_projected_point_lies_on_plane() {
        let anchor = cl_pt(0, 0.0, 0.0, 5.0, 0.0, 0.0, 1.0);
        let projected = project_to_plane((1.5, 2.0, 6.3), &anchor);
        let dist = plane_dist(projected, cl_center(&anchor), anchor.normal);
        assert!(
            dist.abs() < 1e-10,
            "projected point not on plane: dist={dist}"
        );
    }

    #[test]
    fn test_projection_is_idempotent() {
        let anchor = cl_pt(0, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0);
        let once = project_to_plane((4.0, 5.0, 7.0), &anchor);
        let twice = project_to_plane(once, &anchor);
        assert!(
            (once.0 - twice.0).abs() < 1e-10
                && (once.1 - twice.1).abs() < 1e-10
                && (once.2 - twice.2).abs() < 1e-10,
            "projection not idempotent: {once:?} vs {twice:?}"
        );
    }

    #[test]
    fn test_straight_centerline_removes_z_jitter() {
        let anchor = cl_pt(0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        for p in cylinder_ring(0.0, 3.0, 8, 0.5, 42) {
            let proj = project_to_plane(p, &anchor);
            assert!(
                proj.2.abs() < 1e-10,
                "z={} after projection to z=0 plane",
                proj.2
            );
        }
    }

    #[test]
    fn test_tilted_normal_projection() {
        let sq2 = std::f64::consts::SQRT_2 / 2.0;
        let anchor = cl_pt(0, 0.0, 0.0, 0.0, sq2, 0.0, sq2);
        for raw in [
            (1.0_f64, 0.0, 1.0),
            (-1.0, 0.0, -1.0),
            (0.0, 2.0, 0.0),
            (1.0, -1.5, 0.5),
            (0.5, 0.5, -0.5),
        ] {
            let proj = project_to_plane(raw, &anchor);
            let dist = plane_dist(proj, cl_center(&anchor), anchor.normal);
            assert!(
                dist.abs() < 1e-10,
                "tilted projection off-plane: dist={dist}"
            );
        }
    }

    // ---- walk_centerline_slices ----

    #[test]
    fn test_walk_straight_step_equals_spacing() {
        let cl = z_centerline(5);
        let cloud: Vec<(f64, f64, f64)> = (0..5usize)
            .flat_map(|i| cylinder_ring(i as f64, 3.0, 8, 0.3, i as u64 * 17))
            .collect();
        let slices = walk_centerline_slices(&cl, &cloud, 0, 1.0);
        assert_eq!(slices.len(), 5);
        for (i, s) in slices.iter().enumerate() {
            assert_eq!(s.id, i as u32);
            assert!(
                s.points.len() >= 5,
                "frame {i}: only {} points",
                s.points.len()
            );
        }
    }

    #[test]
    fn test_walk_coarser_step_fewer_slices() {
        let cl = z_centerline(9);
        let cloud: Vec<(f64, f64, f64)> = (0..9usize)
            .flat_map(|i| cylinder_ring(i as f64, 3.0, 8, 0.3, i as u64 * 7))
            .collect();
        let slices = walk_centerline_slices(&cl, &cloud, 0, 2.0);
        assert_eq!(
            slices.len(),
            5,
            "expected 5 slices at step 2.0, got {}",
            slices.len()
        );
        for (i, s) in slices.iter().enumerate() {
            assert_eq!(s.id, i as u32);
            assert!(
                s.points.len() >= 5,
                "frame {i}: only {} points",
                s.points.len()
            );
        }
    }

    #[test]
    fn test_walk_finer_step_more_slices() {
        let cl = z_centerline(3);
        let cloud: Vec<(f64, f64, f64)> = (0..3usize)
            .flat_map(|i| cylinder_ring(i as f64, 3.0, 8, 0.1, i as u64 * 11))
            .collect();
        let slices = walk_centerline_slices(&cl, &cloud, 0, 0.5);
        assert_eq!(
            slices.len(),
            5,
            "expected 5 slices at step 0.5, got {}",
            slices.len()
        );
    }

    #[test]
    fn test_projected_points_lie_on_their_anchor_plane() {
        // After Voronoi assignment every point in slice i must lie on anchor i's plane.
        let cl = z_centerline(4);
        let cloud: Vec<(f64, f64, f64)> = (0..4usize)
            .flat_map(|i| cylinder_ring(i as f64, 3.0, 8, 0.3, i as u64 * 5))
            .collect();
        let slices = walk_centerline_slices(&cl, &cloud, 0, 1.0);
        let branch_pts: Vec<&CenterlinePoint> = cl.points.iter().collect();
        let cum = branch_cum_arc(&branch_pts);
        let total = *cum.last().unwrap();
        let anchors: Vec<CenterlinePoint> = build_sample_positions(total, 1.0)
            .into_iter()
            .enumerate()
            .map(|(k, s)| interpolate_branch_at_s(&branch_pts, &cum, s, k))
            .collect();
        for (i, (s, anchor)) in slices.iter().zip(anchors.iter()).enumerate() {
            for p in &s.points {
                let dist = plane_dist(pt_to_tuple(p), cl_center(anchor), anchor.normal);
                assert!(dist.abs() < 1e-10, "slice {i}: point off-plane by {dist}");
            }
        }
    }

    #[test]
    fn test_voronoi_no_cross_contamination() {
        // Two rings at z=0 and z=20 on a straight centerline — they must end up in separate
        // slices with no cross-contamination.
        let cl = Centerline {
            points: vec![
                cl_pt(0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                cl_pt(1, 0.0, 0.0, 20.0, 0.0, 0.0, 1.0),
            ],
            branch_start_indices: vec![0],
        };
        let mut cloud = cylinder_ring(0.0, 3.0, 8, 0.1, 1);
        cloud.extend(cylinder_ring(20.0, 3.0, 8, 0.1, 2));
        let slices = walk_centerline_slices(&cl, &cloud, 0, 20.0);
        assert_eq!(slices.len(), 2);
        // All z≈0 points must be in slice 0 and all z≈20 in slice 1.
        for p in &slices[0].points {
            assert!(
                p.z.abs() < 1.0,
                "slice 0 got a point with z={} (should be near 0)",
                p.z
            );
        }
        for p in &slices[1].points {
            assert!(
                (p.z - 20.0).abs() < 1.0,
                "slice 1 got a point with z={} (should be near 20)",
                p.z
            );
        }
    }

    #[test]
    fn test_walk_curved_centerline_points_on_planes() {
        use std::f64::consts::FRAC_PI_2;
        let n = 8usize;
        let r = 10.0_f64;
        let cl = Centerline {
            points: (0..n)
                .map(|i| {
                    let t = FRAC_PI_2 * i as f64 / (n - 1) as f64;
                    cl_pt(
                        i as u32,
                        r * t.cos(),
                        0.0,
                        r * t.sin(),
                        -t.sin(),
                        0.0,
                        t.cos(),
                    )
                })
                .collect(),
            branch_start_indices: vec![0],
        };

        let step_size = 2.0_f64;
        let cloud: Vec<(f64, f64, f64)> = cl
            .points
            .iter()
            .enumerate()
            .flat_map(|(i, p)| {
                let (cx, cy, cz) = (p.contour_point.x, p.contour_point.y, p.contour_point.z);
                cylinder_ring(0.0, 2.0, 7, 0.3, i as u64 * 31)
                    .into_iter()
                    .map(move |(x, y, z)| (x + cx, y + cy, z + cz))
            })
            .collect();

        let slices = walk_centerline_slices(&cl, &cloud, 0, step_size);
        let arc_len = FRAC_PI_2 * r;
        let expected = (arc_len / step_size).floor() as usize + 1;
        assert_eq!(
            slices.len(),
            expected,
            "expected {expected} slices, got {}",
            slices.len()
        );

        let branch_pts: Vec<&CenterlinePoint> = cl.points.iter().collect();
        let cum = branch_cum_arc(&branch_pts);
        let total = *cum.last().unwrap();
        let anchors: Vec<CenterlinePoint> = build_sample_positions(total, step_size)
            .into_iter()
            .enumerate()
            .map(|(k, pos)| interpolate_branch_at_s(&branch_pts, &cum, pos, k))
            .collect();

        for (i, (s, anchor)) in slices.iter().zip(anchors.iter()).enumerate() {
            assert_eq!(s.id, i as u32);
            for proj in &s.points {
                let dist = plane_dist(pt_to_tuple(proj), cl_center(anchor), anchor.normal);
                assert!(
                    dist.abs() < 1e-10,
                    "curved frame {i}: off-plane dist={dist}"
                );
            }
        }
    }
}
