use crate::intravascular::io::geometry::Contour;
use crate::intravascular::io::input::ContourPoint;
use nalgebra::Vector3;

/// Filters and resamples raw projected slices from `walk_centerline_slices`:
/// - Removes empty slices (CL was outside the vessel).
/// - Removes slices whose points don't cover all four angular quadrants (partial entry slices).
/// - Fits a closed Catmull-Rom spline through the remaining points and resamples each to
///   exactly `n_points` evenly-spaced points along the spline.
pub fn create_uniform_contours(contours: Vec<Contour>, n_points: usize) -> Vec<Contour> {
    contours
        .into_iter()
        .filter(|c| !c.points.is_empty())
        .filter(has_full_angular_coverage)
        .filter_map(|c| resample_spline(c, n_points))
        .collect()
}

// Returns true when the contour's points cover all four 90-degree sectors around the centroid.
fn has_full_angular_coverage(contour: &Contour) -> bool {
    if contour.points.len() < 4 {
        return false;
    }
    let centroid = match contour.centroid {
        Some(c) => c,
        None => return false,
    };
    let (u, v) = match local_basis(&contour.points, centroid) {
        Some(b) => b,
        None => return false,
    };
    let c = Vector3::new(centroid.0, centroid.1, centroid.2);
    let mut quadrants = [false; 4];
    for p in &contour.points {
        let pv = Vector3::new(p.x - c.x, p.y - c.y, p.z - c.z);
        let qu = pv.dot(&u);
        let qv = pv.dot(&v);
        let idx = match (qu >= 0.0, qv >= 0.0) {
            (true, true) => 0,
            (false, true) => 1,
            (false, false) => 2,
            (true, false) => 3,
        };
        quadrants[idx] = true;
    }
    quadrants.iter().all(|&q| q)
}

// Sort by angle in the local plane, fit a closed Catmull-Rom spline, resample to n_points.
fn resample_spline(contour: Contour, n_points: usize) -> Option<Contour> {
    if n_points < 2 || contour.points.len() < 3 {
        return None;
    }
    let centroid = contour.centroid?;
    let (u, v) = local_basis(&contour.points, centroid)?;
    let c = Vector3::new(centroid.0, centroid.1, centroid.2);

    // Sort control points by angle in the local 2D plane.
    let mut angle_pts: Vec<(f64, Vector3<f64>)> = contour
        .points
        .iter()
        .map(|p| {
            let d = Vector3::new(p.x - c.x, p.y - c.y, p.z - c.z);
            let angle = d.dot(&v).atan2(d.dot(&u));
            (angle, Vector3::new(p.x, p.y, p.z))
        })
        .collect();
    angle_pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let ctrl: Vec<Vector3<f64>> = angle_pts.into_iter().map(|(_, p)| p).collect();
    let n = ctrl.len();

    // Dense-sample the closed Catmull-Rom spline.
    const SAMPLES_PER_SEG: usize = 32;
    let mut curve: Vec<Vector3<f64>> = Vec::with_capacity(n * SAMPLES_PER_SEG + 1);
    for i in 0..n {
        let p0 = ctrl[(i + n - 1) % n];
        let p1 = ctrl[i];
        let p2 = ctrl[(i + 1) % n];
        let p3 = ctrl[(i + 2) % n];
        for j in 0..SAMPLES_PER_SEG {
            let t = j as f64 / SAMPLES_PER_SEG as f64;
            curve.push(catmull_rom(p0, p1, p2, p3, t));
        }
    }
    curve.push(curve[0]); // close

    // Cumulative arc lengths.
    let mut cum = vec![0.0f64];
    for i in 1..curve.len() {
        cum.push(cum[i - 1] + (curve[i] - curve[i - 1]).norm());
    }
    let total = *cum.last().unwrap();
    if total < 1e-10 {
        return None;
    }

    // Resample at n_points uniform arc-length positions (open, so last ≠ first).
    let step = total / n_points as f64;
    let new_pts: Vec<ContourPoint> = (0..n_points)
        .map(|k| {
            let target = k as f64 * step;
            let seg = cum
                .partition_point(|&s| s < target)
                .saturating_sub(1)
                .min(curve.len() - 2);
            let s0 = cum[seg];
            let s1 = cum[seg + 1];
            let t = if (s1 - s0).abs() < 1e-12 {
                0.0
            } else {
                (target - s0) / (s1 - s0)
            };
            let p = curve[seg] * (1.0 - t) + curve[seg + 1] * t;
            ContourPoint {
                frame_index: contour.id,
                point_index: k as u32,
                x: p.x,
                y: p.y,
                z: p.z,
                aortic: false,
            }
        })
        .collect();

    Some(Contour {
        id: contour.id,
        original_frame: contour.original_frame,
        centroid: contour.centroid,
        points: new_pts,
        aortic_thickness: None,
        pulmonary_thickness: None,
        kind: contour.kind,
    })
}

// Derive two orthonormal basis vectors spanning the plane of `points` through `centroid`.
fn local_basis(
    points: &[ContourPoint],
    centroid: (f64, f64, f64),
) -> Option<(Vector3<f64>, Vector3<f64>)> {
    let c = Vector3::new(centroid.0, centroid.1, centroid.2);
    let mut u: Option<Vector3<f64>> = None;
    for p in points {
        let d = Vector3::new(p.x - c.x, p.y - c.y, p.z - c.z);
        if d.norm() > 1e-10 {
            u = Some(d.normalize());
            break;
        }
    }
    let u = u?;
    for p in points {
        let d = Vector3::new(p.x - c.x, p.y - c.y, p.z - c.z);
        let cross = u.cross(&d);
        if cross.norm() > 1e-10 {
            let normal = cross.normalize();
            let v = normal.cross(&u).normalize();
            return Some((u, v));
        }
    }
    None
}

fn catmull_rom(
    p0: Vector3<f64>,
    p1: Vector3<f64>,
    p2: Vector3<f64>,
    p3: Vector3<f64>,
    t: f64,
) -> Vector3<f64> {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intravascular::io::geometry::{Contour, ContourType};
    use crate::intravascular::io::input::ContourPoint;

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
        // 3 full circles + 1 empty + 1 half-circle → 3 in output.
        let contours = vec![
            make_contour(0, circle_ring((0.0, 0.0, 0.0), 3.0, 16), (0.0, 0.0, 0.0)),
            make_contour(1, vec![], (0.0, 0.0, 1.0)),
            make_contour(2, circle_ring((0.0, 0.0, 2.0), 3.0, 16), (0.0, 0.0, 2.0)),
            make_contour(3, half_circle_ring(3.0, 10), (0.0, 0.0, 0.0)),
            make_contour(4, circle_ring((0.0, 0.0, 4.0), 3.0, 16), (0.0, 0.0, 4.0)),
        ];
        let result = create_uniform_contours(contours, 100);
        assert_eq!(result.len(), 3);
        for c in &result {
            assert_eq!(c.points.len(), 100);
        }
    }
}
