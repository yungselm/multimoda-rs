use crate::intravascular::io::geometry::{Contour, ContourType};
use crate::intravascular::io::input::{Centerline, CenterlinePoint, ContourPoint};
use nalgebra::Vector3;

/// Walks branch `branch_id` at uniform arc-length steps of `step_size`, collecting a projected
/// slice at each resampled position. The slab half-thickness is `step_size / 2.0`, so adjacent
/// slices just touch without overlap. Returns one `Contour` per resampled position.
pub fn walk_centerline_slices(
    centerline: &Centerline,
    points: &[(f64, f64, f64)],
    branch_id: u32,
    step_size: f64,
) -> Vec<Contour> {
    let half = step_size / 2.0;
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
    sample_positions
        .iter()
        .enumerate()
        .map(|(k, &s)| {
            let anchor = interpolate_branch_at_s(&branch_pts, &cum, s, k);
            slice_nearest_points(&anchor, points, half, k as u32)
        })
        .collect()
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

// Filters points within `half_thickness` of the plane at `anchor`, projects them onto it,
// then wraps the result in a `Contour`. `id` is the slice index along the walk.
fn slice_nearest_points(
    anchor: &CenterlinePoint,
    points: &[(f64, f64, f64)],
    half_thickness: f64,
    id: u32,
) -> Contour {
    let cx = anchor.contour_point.x;
    let cy = anchor.contour_point.y;
    let cz = anchor.contour_point.z;
    let center = Vector3::new(cx, cy, cz);
    let n = anchor.normal;
    let projected: Vec<ContourPoint> = points
        .iter()
        .filter(|&&(x, y, z)| {
            let v = Vector3::new(x - center.x, y - center.y, z - center.z);
            v.dot(&n).abs() <= half_thickness
        })
        .enumerate()
        .map(|(i, &p)| {
            let (px, py, pz) = project_to_plane(p, anchor);
            ContourPoint {
                frame_index: id,
                point_index: i as u32,
                x: px,
                y: py,
                z: pz,
                aortic: false,
            }
        })
        .collect();
    Contour {
        id,
        original_frame: anchor.contour_point.frame_index,
        centroid: Some((cx, cy, cz)),
        points: projected,
        aortic_thickness: None,
        pulmonary_thickness: None,
        kind: ContourType::Lumen,
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intravascular::io::input::{Centerline, CenterlinePoint, ContourPoint};
    use nalgebra::Vector3;

    // builds a ContourPoint — used only for constructing CenterlinePoints
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

    // ---- slice_nearest_points ----

    #[test]
    fn test_slice_has_at_least_five_points() {
        let anchor = cl_pt(0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        let slice = slice_nearest_points(&anchor, &cylinder_ring(0.0, 3.0, 8, 0.3, 7), 1.0, 0);
        assert!(
            slice.points.len() >= 5,
            "need ≥ 5 points, got {}",
            slice.points.len()
        );
    }

    #[test]
    fn test_slice_contour_fields() {
        let anchor = cl_pt(3, 1.0, 2.0, 3.0, 0.0, 0.0, 1.0);
        let slice = slice_nearest_points(&anchor, &cylinder_ring(3.0, 3.0, 8, 0.2, 5), 1.0, 7);
        assert_eq!(slice.id, 7);
        assert_eq!(slice.original_frame, anchor.contour_point.frame_index);
        assert_eq!(slice.centroid, Some((1.0, 2.0, 3.0)));
        assert!(slice.aortic_thickness.is_none());
        assert!(slice.pulmonary_thickness.is_none());
    }

    #[test]
    fn test_sliced_points_lie_on_plane() {
        let anchor = cl_pt(0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        let slice = slice_nearest_points(&anchor, &cylinder_ring(0.0, 3.0, 8, 0.4, 99), 1.0, 0);
        for p in &slice.points {
            let dist = plane_dist(pt_to_tuple(p), cl_center(&anchor), anchor.normal);
            assert!(dist.abs() < 1e-10, "sliced point off-plane: dist={dist}");
        }
    }

    #[test]
    fn test_jitter_points_stay_distinct_after_projection() {
        let anchor = cl_pt(0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        let slice = slice_nearest_points(&anchor, &cylinder_ring(0.0, 5.0, 8, 0.2, 13), 1.0, 0);
        let pts = &slice.points;
        for i in 0..pts.len() {
            for j in (i + 1)..pts.len() {
                let (a, b) = (&pts[i], &pts[j]);
                let d = ((a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.z - b.z).powi(2)).sqrt();
                assert!(d > 0.1, "points {i} and {j} collapsed: d={d}");
            }
        }
    }

    #[test]
    fn test_points_outside_slab_are_excluded() {
        let anchor = cl_pt(0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        let mut cloud = cylinder_ring(0.0, 3.0, 6, 0.1, 55);
        cloud.push((1.0, 0.0, 5.0)); // outside
        cloud.push((0.0, 1.0, -5.0)); // outside
        cloud.push((2.0, 2.0, 3.0)); // outside
        let slice = slice_nearest_points(&anchor, &cloud, 0.5, 0);
        assert_eq!(
            slice.points.len(),
            6,
            "expected 6 in-slab points, got {}",
            slice.points.len()
        );
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
