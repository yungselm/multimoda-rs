use crate::types::native::{cumulative_arc_length, Centerline, CenterlinePoint, ContourPoint};
use nalgebra::Vector3;
use rstar::primitives::GeomWithData;
use rstar::RTree;

/// Samples branch `branch_id` of `centerline` every `step_size` of arc length (plus the branch
/// end, see [`build_sample_positions`]). Each anchor carries the interpolated position and unit
/// tangent; the tangent is the normal of that anchor's cutting plane.
pub fn branch_anchors(
    centerline: &Centerline,
    branch_id: u32,
    step_size: f64,
) -> Vec<CenterlinePoint> {
    let branch_pts: Vec<&CenterlinePoint> = centerline
        .points
        .iter()
        .filter(|p| p.branch_id == branch_id)
        .collect();
    if branch_pts.is_empty() {
        return vec![];
    }

    let cum = cumulative_arc_length(&branch_pts);
    let total = *cum.last().unwrap();
    build_sample_positions(total, step_size)
        .iter()
        .enumerate()
        .map(|(slice_idx, &arc_pos)| interpolate_branch_at_s(&branch_pts, &cum, arc_pos, slice_idx))
        .collect()
}

/// R-tree over anchor positions.
///
/// Used to limit each triangle to the cutting planes of nearby anchors, so a plane never picks
/// up a distant part of the vessel that it happens to intersect when extended.
pub struct AnchorIndex {
    tree: RTree<GeomWithData<[f64; 3], usize>>,
}

impl AnchorIndex {
    pub fn new(anchors: &[CenterlinePoint]) -> Self {
        let tree = RTree::bulk_load(
            anchors
                .iter()
                .enumerate()
                .map(|(i, a)| {
                    GeomWithData::new([a.contour_point.x, a.contour_point.y, a.contour_point.z], i)
                })
                .collect(),
        );
        Self { tree }
    }

    pub fn nearest(&self, p: &Vector3<f64>) -> Option<usize> {
        self.tree
            .nearest_neighbor([p.x, p.y, p.z])
            .map(|nn| nn.data)
    }
}

/// Distances along the branch (from its start) at which slices are cut: every `step` up to the
/// branch length `total`, plus `total` itself if it isn't a multiple of `step`, so the branch end
/// is always sliced. E.g. `total = 10, step = 3` → `[0, 3, 6, 9, 10]`.
///
/// The small tolerances absorb floating-point noise (e.g. `3.0 / 0.1 = 29.999…`). Returns an
/// empty vector for a non-positive `step` or invalid `total`.
fn build_sample_positions(total: f64, step: f64) -> Vec<f64> {
    if step.is_nan() || step <= 0.0 || !total.is_finite() || total < 0.0 {
        return vec![];
    }
    let n = (total / step + 1e-9).floor() as usize;
    let mut positions: Vec<f64> = (0..=n).map(|i| i as f64 * step).collect();
    if total - positions[n] > 1e-6 {
        positions.push(total);
    }
    positions
}

fn interpolate_branch_at_s(
    pts: &[&CenterlinePoint],
    cum: &[f64],
    target_s: f64,
    idx_out: usize,
) -> CenterlinePoint {
    let seg = match cum
        .binary_search_by(|v| v.partial_cmp(&target_s).unwrap_or(std::cmp::Ordering::Less))
    {
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
            tangent: last.tangent,
            branch_id: last.branch_id,
            radius: last.radius,
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

    let n0 = pts[seg].tangent;
    let n1 = pts[seg + 1].tangent;
    let mut tangent = n0 * (1.0 - t) + n1 * t;
    let n_norm = tangent.norm();
    if n_norm > 1e-12 {
        tangent /= n_norm;
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
        tangent,
        branch_id: pts[seg].branch_id,
        radius: pts[seg].radius + t * (pts[seg + 1].radius - pts[seg].radius),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cl_pt(idx: u32, x: f64, y: f64, z: f64, nx: f64, ny: f64, nz: f64) -> CenterlinePoint {
        CenterlinePoint {
            contour_point: ContourPoint {
                frame_index: idx,
                point_index: idx,
                x,
                y,
                z,
                aortic: false,
            },
            tangent: Vector3::new(nx, ny, nz).normalize(),
            branch_id: 0,
            radius: 0.0,
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

    // ---- build_sample_positions ----

    #[test]
    fn test_sample_positions_include_branch_end() {
        assert_eq!(
            build_sample_positions(10.0, 3.0),
            vec![0.0, 3.0, 6.0, 9.0, 10.0]
        );
    }

    #[test]
    fn test_sample_positions_exact_multiple_no_duplicate_end() {
        assert_eq!(
            build_sample_positions(8.0, 2.0),
            vec![0.0, 2.0, 4.0, 6.0, 8.0]
        );
        // 0.1 does not accumulate drift: 30 steps land exactly on 3.0 without an extra slice.
        assert_eq!(build_sample_positions(3.0, 0.1).len(), 31);
    }

    #[test]
    fn test_sample_positions_degenerate_inputs() {
        assert_eq!(build_sample_positions(0.0, 1.0), vec![0.0]);
        assert!(build_sample_positions(5.0, 0.0).is_empty());
        assert!(build_sample_positions(5.0, -1.0).is_empty());
        assert!(build_sample_positions(f64::NAN, 1.0).is_empty());
    }

    // ---- branch_anchors ----

    #[test]
    fn test_anchor_count_follows_step() {
        let cl = z_centerline(9);
        assert_eq!(branch_anchors(&cl, 0, 1.0).len(), 9);
        assert_eq!(branch_anchors(&cl, 0, 2.0).len(), 5);
        assert_eq!(branch_anchors(&cl, 0, 0.5).len(), 17);
    }

    #[test]
    fn test_anchors_include_tail() {
        // Length 4.5 at step 1.0: anchors at 0,1,2,3,4 and the end at 4.5.
        let cl = Centerline {
            points: vec![
                cl_pt(0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                cl_pt(1, 0.0, 0.0, 4.5, 0.0, 0.0, 1.0),
            ],
            branch_start_indices: vec![0],
        };
        let anchors = branch_anchors(&cl, 0, 1.0);
        assert_eq!(anchors.len(), 6);
        assert!((anchors.last().unwrap().contour_point.z - 4.5).abs() < 1e-12);
    }

    #[test]
    fn test_unknown_branch_has_no_anchors() {
        assert!(branch_anchors(&z_centerline(5), 3, 1.0).is_empty());
    }

    #[test]
    fn test_anchor_radius_is_interpolated() {
        let mut p0 = cl_pt(0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        let mut p1 = cl_pt(1, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0);
        p0.radius = 1.0;
        p1.radius = 3.0;
        let pts = vec![&p0, &p1];
        let cum = cumulative_arc_length(&pts);
        let mid = interpolate_branch_at_s(&pts, &cum, 1.0, 0);
        assert!((mid.radius - 2.0).abs() < 1e-12);
    }

    // ---- AnchorIndex ----

    #[test]
    fn test_anchor_index_finds_nearest() {
        let anchors = branch_anchors(&z_centerline(5), 0, 1.0);
        let index = AnchorIndex::new(&anchors);
        assert_eq!(index.nearest(&Vector3::new(3.0, 0.0, 2.2)), Some(2));
        assert_eq!(index.nearest(&Vector3::new(0.0, 5.0, -10.0)), Some(0));
        assert_eq!(AnchorIndex::new(&[]).nearest(&Vector3::zeros()), None);
    }
}
