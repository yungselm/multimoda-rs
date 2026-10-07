use crate::types::native::{cumulative_arc_length, Contour, ContourPoint};
use nalgebra::Vector3;

/// An end slice counts as complete only if its outline is at least this fraction of the
/// branch's median outline length; shorter ones cut the rounded cap where the segmented
/// vessel ends rather than the lumen.
const MIN_END_LENGTH_RATIO: f64 = 0.5;

/// Filters and resamples raw slice outlines from `slicing::slice_mesh`:
/// - Removes empty slices (the plane did not cut the mesh).
/// - Trims incomplete entry/exit slices from both ends: a slice is complete when it covers all
///   four angular quadrants and is not much shorter than the branch's typical outline (see
///   [`MIN_END_LENGTH_RATIO`]). The test only locates the first and last complete slice; every
///   interior slice is kept so gaps are never introduced mid-vessel (a stenosis stays).
/// - Resamples each remaining closed outline to exactly `n_points` points evenly spaced
///   along its length.
pub fn create_uniform_contours(contours: Vec<Contour>, n_points: usize) -> Vec<Contour> {
    let non_empty: Vec<Contour> = contours
        .into_iter()
        .filter(|c| !c.points.is_empty())
        .collect();

    // Trim incomplete entry/exit slices from both ends only.
    // Interior slices are kept as-is: a slightly imperfect interior ring is far
    // better than a hole in the vessel wall.
    let lengths: Vec<f64> = non_empty.iter().map(outline_length).collect();
    let min_length = MIN_END_LENGTH_RATIO * median(&lengths);
    let complete: Vec<bool> = non_empty
        .iter()
        .zip(&lengths)
        .map(|(c, &len)| has_full_angular_coverage(c) && len >= min_length)
        .collect();
    let start = complete.iter().position(|&ok| ok).unwrap_or(0);
    let end = complete
        .iter()
        .rposition(|&ok| ok)
        .map(|i| i + 1)
        .unwrap_or(non_empty.len());

    non_empty[start..end]
        .iter()
        .cloned()
        .filter_map(|c| resample_outline(c, n_points))
        .collect()
}

/// Length of the closed outline through the contour's points.
fn outline_length(contour: &Contour) -> f64 {
    let n = contour.points.len();
    (0..n)
        .map(|i| {
            let (a, b) = (&contour.points[i], &contour.points[(i + 1) % n]);
            ((a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.z - b.z).powi(2)).sqrt()
        })
        .sum()
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

/// Returns true when the contour's points cover all four 90-degree sectors around the centroid.
fn has_full_angular_coverage(contour: &Contour) -> bool {
    if contour.points.len() < 4 {
        return false;
    }
    let centroid = match contour.centroid {
        Some(c) => c,
        None => return false,
    };
    let (axis_u, axis_v) = match local_basis(&contour.points, centroid) {
        Some(basis) => basis,
        None => return false,
    };
    let centroid_v = Vector3::new(centroid.0, centroid.1, centroid.2);
    let mut quadrants = [false; 4];
    for p in &contour.points {
        let offset = Vector3::new(p.x - centroid_v.x, p.y - centroid_v.y, p.z - centroid_v.z);
        let proj_u = offset.dot(&axis_u);
        let proj_v = offset.dot(&axis_v);
        let quadrant = match (proj_u >= 0.0, proj_v >= 0.0) {
            (true, true) => 0,
            (false, true) => 1,
            (false, false) => 2,
            (true, false) => 3,
        };
        quadrants[quadrant] = true;
    }
    quadrants.iter().all(|&q| q)
}

/// Resamples a closed outline (points in order, first point not repeated) to `n_points`
/// positions evenly spaced along its length, interpolating linearly between outline points.
/// The outline comes from an exact mesh cut, so no smoothing is applied.
fn resample_outline(contour: Contour, n_points: usize) -> Option<Contour> {
    if n_points < 2 || contour.points.len() < 3 {
        return None;
    }
    let mut ring: Vec<Vector3<f64>> = contour
        .points
        .iter()
        .map(|p| Vector3::new(p.x, p.y, p.z))
        .collect();
    ring.push(ring[0]);

    let arc_lengths = cumulative_arc_length(&ring);
    let total_length = *arc_lengths.last().unwrap();
    if total_length < 1e-10 {
        return None;
    }
    let resampled = uniform_resample(&ring, &arc_lengths, total_length, n_points);
    Some(build_output_contour(contour, resampled))
}

/// Resample `curve` at `n_points` uniformly-spaced arc-length positions.
fn uniform_resample(
    curve: &[Vector3<f64>],
    arc_lengths: &[f64],
    total_length: f64,
    n_points: usize,
) -> Vec<Vector3<f64>> {
    let step = total_length / n_points as f64;
    (0..n_points)
        .map(|point_idx| {
            let target = point_idx as f64 * step;
            let seg = arc_lengths
                .partition_point(|&s| s < target)
                .saturating_sub(1)
                .min(curve.len() - 2);
            let seg_start = arc_lengths[seg];
            let seg_end = arc_lengths[seg + 1];
            let frac = if (seg_end - seg_start).abs() < 1e-12 {
                0.0
            } else {
                (target - seg_start) / (seg_end - seg_start)
            };
            curve[seg] * (1.0 - frac) + curve[seg + 1] * frac
        })
        .collect()
}

/// Build the output `Contour` from the source metadata and resampled point positions.
fn build_output_contour(source: Contour, points: Vec<Vector3<f64>>) -> Contour {
    let new_pts = points
        .into_iter()
        .enumerate()
        .map(|(point_idx, pos)| ContourPoint {
            frame_index: source.id,
            point_index: point_idx as u32,
            x: pos.x,
            y: pos.y,
            z: pos.z,
            aortic: false,
        })
        .collect();
    Contour {
        id: source.id,
        original_frame: source.original_frame,
        centroid: source.centroid,
        points: new_pts,
        aortic_thickness: None,
        pulmonary_thickness: None,
        kind: source.kind,
    }
}

/// Derive two orthonormal basis vectors spanning the plane of `points` through `centroid`.
fn local_basis(
    points: &[ContourPoint],
    centroid: (f64, f64, f64),
) -> Option<(Vector3<f64>, Vector3<f64>)> {
    let centroid_v = Vector3::new(centroid.0, centroid.1, centroid.2);
    let mut axis_u: Option<Vector3<f64>> = None;
    for p in points {
        let offset = Vector3::new(p.x - centroid_v.x, p.y - centroid_v.y, p.z - centroid_v.z);
        if offset.norm() > 1e-10 {
            axis_u = Some(offset.normalize());
            break;
        }
    }
    let axis_u = axis_u?;
    for p in points {
        let offset = Vector3::new(p.x - centroid_v.x, p.y - centroid_v.y, p.z - centroid_v.z);
        let cross = axis_u.cross(&offset);
        if cross.norm() > 1e-10 {
            let normal = cross.normalize();
            let axis_v = normal.cross(&axis_u).normalize();
            return Some((axis_u, axis_v));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::native::{Contour, ContourPoint, ContourType};

    fn make_point(x: f64, y: f64, z: f64) -> ContourPoint {
        ContourPoint {
            frame_index: 0,
            point_index: 0,
            x,
            y,
            z,
            aortic: false,
        }
    }

    fn make_contour(id: u32, points: Vec<ContourPoint>, centroid: (f64, f64, f64)) -> Contour {
        Contour {
            id,
            original_frame: id,
            centroid: Some(centroid),
            points,
            aortic_thickness: None,
            pulmonary_thickness: None,
            kind: ContourType::Lumen,
        }
    }

    /// Circle of n points at given radius in the XY plane centered at `center`.
    fn circle_ring(center: (f64, f64, f64), radius: f64, n: usize) -> Vec<ContourPoint> {
        use std::f64::consts::TAU;
        (0..n)
            .map(|i| {
                let a = TAU * i as f64 / n as f64;
                make_point(
                    center.0 + radius * a.cos(),
                    center.1 + radius * a.sin(),
                    center.2,
                )
            })
            .collect()
    }

    /// Upper-half circle (only y >= 0, so quadrants 0 and 1 are covered, 2 and 3 are not).
    fn half_circle_ring(radius: f64, n: usize) -> Vec<ContourPoint> {
        use std::f64::consts::PI;
        (0..n)
            .map(|i| {
                let a = PI * i as f64 / (n - 1) as f64;
                make_point(radius * a.cos(), radius * a.sin(), 0.0)
            })
            .collect()
    }

    // ---- has_full_angular_coverage ----

    #[test]
    fn test_empty_contour_has_no_coverage() {
        let c = make_contour(0, vec![], (0.0, 0.0, 0.0));
        assert!(!has_full_angular_coverage(&c));
    }

    #[test]
    fn test_fewer_than_four_points_has_no_coverage() {
        let pts = circle_ring((0.0, 0.0, 0.0), 3.0, 3);
        let c = make_contour(0, pts, (0.0, 0.0, 0.0));
        assert!(!has_full_angular_coverage(&c));
    }

    #[test]
    fn test_half_circle_missing_coverage() {
        let pts = half_circle_ring(3.0, 10);
        let c = make_contour(0, pts, (0.0, 0.0, 0.0));
        assert!(!has_full_angular_coverage(&c));
    }

    #[test]
    fn test_full_circle_has_coverage() {
        let pts = circle_ring((0.0, 0.0, 0.0), 3.0, 16);
        let c = make_contour(0, pts, (0.0, 0.0, 0.0));
        assert!(has_full_angular_coverage(&c));
    }

    #[test]
    fn test_full_circle_tilted_plane_has_coverage() {
        // Circle in the XZ plane instead of XY.
        use std::f64::consts::TAU;
        let pts: Vec<ContourPoint> = (0..16)
            .map(|i| {
                let a = TAU * i as f64 / 16.0;
                make_point(3.0 * a.cos(), 0.0, 3.0 * a.sin())
            })
            .collect();
        let c = make_contour(0, pts, (0.0, 0.0, 0.0));
        assert!(has_full_angular_coverage(&c));
    }

    // ---- create_uniform_contours ----

    #[test]
    fn test_empty_contours_removed() {
        let contours = vec![
            make_contour(0, vec![], (0.0, 0.0, 0.0)),
            make_contour(1, circle_ring((0.0, 0.0, 0.0), 3.0, 16), (0.0, 0.0, 0.0)),
        ];
        let result = create_uniform_contours(contours, 50);
        assert_eq!(result.len(), 1, "empty contour should be removed");
    }

    #[test]
    fn test_half_circle_contours_removed() {
        let contours = vec![
            make_contour(0, half_circle_ring(3.0, 12), (0.0, 0.0, 0.0)),
            make_contour(1, circle_ring((0.0, 0.0, 0.0), 3.0, 16), (0.0, 0.0, 0.0)),
        ];
        let result = create_uniform_contours(contours, 50);
        assert_eq!(result.len(), 1, "half-circle contour should be removed");
    }

    #[test]
    fn test_output_has_exact_n_points() {
        let contours = vec![make_contour(
            0,
            circle_ring((0.0, 0.0, 0.0), 3.0, 20),
            (0.0, 0.0, 0.0),
        )];
        for n in [8, 50, 200] {
            let result = create_uniform_contours(contours.clone(), n);
            assert_eq!(
                result[0].points.len(),
                n,
                "expected {n} points, got {}",
                result[0].points.len()
            );
        }
    }

    #[test]
    fn test_contour_metadata_preserved() {
        let contours = vec![make_contour(
            7,
            circle_ring((1.0, 2.0, 3.0), 3.0, 16),
            (1.0, 2.0, 3.0),
        )];
        let result = create_uniform_contours(contours, 50);
        assert_eq!(result[0].id, 7);
        assert_eq!(result[0].centroid, Some((1.0, 2.0, 3.0)));
        assert_eq!(result[0].kind, ContourType::Lumen);
    }

    #[test]
    fn test_resampled_points_close_to_input_circle() {
        // Input: perfect circle of radius 5 in XY. Output points should be ≈ 5 units from origin.
        let radius = 5.0_f64;
        let contours = vec![make_contour(
            0,
            circle_ring((0.0, 0.0, 0.0), radius, 24),
            (0.0, 0.0, 0.0),
        )];
        let result = create_uniform_contours(contours, 200);
        for p in &result[0].points {
            let r = (p.x * p.x + p.y * p.y + p.z * p.z).sqrt();
            assert!(
                (r - radius).abs() < 0.05,
                "point radius {r:.4} deviates from expected {radius}"
            );
        }
    }

    #[test]
    fn test_resampled_points_lie_on_input_plane() {
        // Circle lies in the plane z = 4.0; all output points should have z ≈ 4.0.
        let contours = vec![make_contour(
            0,
            circle_ring((0.0, 0.0, 4.0), 3.0, 20),
            (0.0, 0.0, 4.0),
        )];
        let result = create_uniform_contours(contours, 100);
        for p in &result[0].points {
            assert!((p.z - 4.0).abs() < 1e-10, "z={} expected 4.0", p.z);
        }
    }

    #[test]
    fn test_point_indices_are_sequential() {
        let contours = vec![make_contour(
            0,
            circle_ring((0.0, 0.0, 0.0), 3.0, 16),
            (0.0, 0.0, 0.0),
        )];
        let result = create_uniform_contours(contours, 50);
        for (i, p) in result[0].points.iter().enumerate() {
            assert_eq!(p.point_index, i as u32);
            assert_eq!(p.frame_index, 0);
        }
    }

    #[test]
    fn test_multiple_contours_pipeline() {
        // 3 full circles + 1 empty + 1 interior half-circle → 4 in output.
        // The empty slice is dropped; the half-circle is interior (between two full circles)
        // so it is kept to avoid creating a hole in the vessel wall.
        let contours = vec![
            make_contour(0, circle_ring((0.0, 0.0, 0.0), 3.0, 16), (0.0, 0.0, 0.0)),
            make_contour(1, vec![], (0.0, 0.0, 1.0)),
            make_contour(2, circle_ring((0.0, 0.0, 2.0), 3.0, 16), (0.0, 0.0, 2.0)),
            make_contour(3, half_circle_ring(3.0, 10), (0.0, 0.0, 0.0)),
            make_contour(4, circle_ring((0.0, 0.0, 4.0), 3.0, 16), (0.0, 0.0, 4.0)),
        ];
        let result = create_uniform_contours(contours, 100);
        assert_eq!(result.len(), 4);
        for c in &result {
            assert_eq!(c.points.len(), 100);
        }
    }

    #[test]
    fn test_collapsed_end_slices_trimmed() {
        // A tiny last ring (plane cutting the vessel's end cap) is trimmed, but a narrow
        // interior ring (stenosis) between full-size rings is kept.
        let contours = vec![
            make_contour(0, circle_ring((0.0, 0.0, 0.0), 3.0, 16), (0.0, 0.0, 0.0)),
            make_contour(1, circle_ring((0.0, 0.0, 1.0), 1.0, 16), (0.0, 0.0, 1.0)),
            make_contour(2, circle_ring((0.0, 0.0, 2.0), 3.0, 16), (0.0, 0.0, 2.0)),
            make_contour(3, circle_ring((0.0, 0.0, 3.0), 3.0, 16), (0.0, 0.0, 3.0)),
            make_contour(4, circle_ring((0.0, 0.0, 4.0), 0.2, 16), (0.0, 0.0, 4.0)),
        ];
        let result = create_uniform_contours(contours, 50);
        let ids: Vec<u32> = result.iter().map(|c| c.id).collect();
        assert_eq!(ids, vec![0, 1, 2, 3]);
    }
}
