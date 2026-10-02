use crate::types::native::geometry::Geometry;
use crate::types::native::{mean_spacing, Centerline, CenterlinePoint};

/// Resample `centerline` along its arc-length so that adjacent points are spaced at the
/// mean Euclidean distance between consecutive contour centroids in `ref_mesh`, falling
/// back to the centerline's own mean point spacing when the mesh has fewer than two frames.
///
/// Only branch-0 points are used for resampling — side branches (branch_id > 0) are
/// stripped before processing so that `ensure_descending_z` and the arc-length
/// calculation see only the main-vessel path. Tangents are recomputed in the final
/// (descending-z) point order.
///
/// The centerline point closest to `ref_pt` is located on the dense input centerline and
/// the sampling grid is anchored on it ([`Centerline::resample_anchored`]), so the
/// reference stays exact instead of snapping to a sample up to half a spacing away.
///
/// Returns the resampled centerline, the reference point's index in it, and the spacing
/// (mm) that was used, so callers can apply the same spacing to other centerlines via
/// `Centerline::resample` instead of re-deriving it.
pub fn preprocess_centerline(
    centerline: Centerline,
    ref_mesh: &Geometry,
    ref_pt: &(f64, f64, f64),
) -> Result<(Centerline, usize, f64), &'static str> {
    // Strip side-branch points so they cannot corrupt ensure_descending_z or the
    // cumulative arc-length (a side-branch tip with high z would trigger an erroneous
    // reversal of the entire array, placing the main-vessel reference at the end).
    let pts: Vec<CenterlinePoint> = centerline
        .points
        .into_iter()
        .filter(|p| p.branch_id == 0)
        .collect();
    if pts.is_empty() {
        return Err("Centerline has no branch-0 points");
    }
    if ref_mesh.frames.is_empty() {
        return Err("Reference mesh has no frames");
    }
    let mut cl = Centerline {
        points: pts,
        branch_start_indices: vec![0],
    };
    ensure_descending_z(&mut cl);
    let ref_idx = cl.find_reference_cl_point_idx(ref_pt);

    let Some(spacing) = decide_spacing(ref_mesh, &cl) else {
        eprintln!("preprocess_centerline: invalid spacing computed, returning original centerline");
        return Ok((cl, ref_idx, 0.0));
    };
    let ref_idx = cl.resample_anchored(spacing, ref_idx);

    eprintln!("preprocess_centerline: produced {} points", cl.points.len());
    Ok((cl, ref_idx, spacing))
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

fn calculate_mean_spacing(ref_mesh: &Geometry) -> Option<f64> {
    let centroids: Vec<(f64, f64, f64)> = ref_mesh.frames.iter().map(|f| f.centroid).collect();
    mean_spacing(&centroids)
}

/// Mean contour-centroid spacing of `ref_mesh`, or the centerline's own mean point
/// spacing if the mesh gives no usable value. `None` if neither is usable.
fn decide_spacing(ref_mesh: &Geometry, centerline: &Centerline) -> Option<f64> {
    let is_valid = |s: &f64| s.is_finite() && *s > 1e-12;
    calculate_mean_spacing(ref_mesh)
        .filter(is_valid)
        .or_else(|| mean_spacing(&centerline.points).filter(is_valid))
}

#[cfg(test)]
mod cl_preprocessing_tests {
    use super::*;
    use crate::types::native::contour::{Contour, ContourType};
    use crate::types::native::frame::Frame;
    use crate::types::native::ContourPoint;
    use approx::assert_relative_eq;
    use nalgebra::Vector3;
    use std::collections::HashMap;

    fn geom_from_centroids(centroids: &[(f64, f64, f64)]) -> Geometry {
        let frames = centroids
            .iter()
            .enumerate()
            .map(|(i, &centroid)| Frame {
                id: i as u32,
                centroid,
                lumen: Contour {
                    id: i as u32,
                    original_frame: i as u32,
                    points: vec![],
                    centroid: Some(centroid),
                    aortic_thickness: None,
                    pulmonary_thickness: None,
                    kind: ContourType::Lumen,
                },
                extras: HashMap::new(),
                reference_point: None,
            })
            .collect();
        Geometry {
            frames,
            label: "test".to_string(),
        }
    }

    fn cl_from_coords(coords: &[(f64, f64, f64)]) -> Centerline {
        Centerline::from_contour_points(
            coords
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
                .collect(),
        )
    }

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
                    tangent: Vector3::new(0.0, 0.0, -1.0),
                    branch_id: 0,
                    radius: 0.0,
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
                    tangent: Vector3::new(0.0, 0.0, -1.0),
                    branch_id: 0,
                    radius: 0.0,
                },
            ],
            branch_start_indices: vec![0],
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
                    tangent: Vector3::new(0.0, 0.0, -1.0),
                    branch_id: 0,
                    radius: 0.0,
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
                    tangent: Vector3::new(0.0, 0.0, -1.0),
                    branch_id: 0,
                    radius: 0.0,
                },
            ],
            branch_start_indices: vec![0],
        };
        ensure_descending_z(&mut cl);
        assert_eq!(cl.points[0].contour_point.z, 1.0);
        assert_eq!(cl.points[1].contour_point.z, 0.0);
    }

    #[test]
    fn test_calculate_mean_spacing() {
        // distances 5.0 and 5.0
        let geom = geom_from_centroids(&[(0.0, 0.0, 0.0), (3.0, 4.0, 0.0), (6.0, 8.0, 0.0)]);
        assert_eq!(calculate_mean_spacing(&geom), Some(5.0));

        // single frame → no distances
        let geom = geom_from_centroids(&[(1.0, 2.0, 3.0)]);
        assert_eq!(calculate_mean_spacing(&geom), None);
    }

    #[test]
    fn test_decide_spacing_prefers_mesh_then_centerline() {
        let cl = cl_from_coords(&[
            (0.0, 0.0, 3.0),
            (0.0, 0.0, 2.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0),
        ]);

        let geom = geom_from_centroids(&[(0.0, 0.0, 0.0), (0.0, 0.0, 0.5)]);
        assert_relative_eq!(decide_spacing(&geom, &cl).unwrap(), 0.5);

        // single frame → no mesh spacing → centerline mean spacing
        let geom = geom_from_centroids(&[(0.0, 0.0, 0.0)]);
        assert_relative_eq!(decide_spacing(&geom, &cl).unwrap(), 1.0);

        // coincident centroids → zero mesh spacing is rejected as well
        let geom = geom_from_centroids(&[(1.0, 1.0, 1.0), (1.0, 1.0, 1.0)]);
        assert_relative_eq!(decide_spacing(&geom, &cl).unwrap(), 1.0);
    }

    #[test]
    fn test_preprocess_centerline_resamples_to_mesh_spacing() {
        // ascending z → reversed by ensure_descending_z; total length 3.0
        let cl = cl_from_coords(&[
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 2.0),
            (0.0, 0.0, 3.0),
        ]);
        let geom = geom_from_centroids(&[(0.0, 0.0, 0.0), (0.0, 0.0, 0.75), (0.0, 0.0, 1.5)]);

        let (resampled, ref_idx, spacing) =
            preprocess_centerline(cl, &geom, &(0.0, 0.0, 3.0)).unwrap();

        assert_relative_eq!(spacing, 0.75);
        assert_eq!(ref_idx, 0);
        let expected_z = [3.0, 2.25, 1.5, 0.75, 0.0];
        assert_eq!(resampled.points.len(), expected_z.len());
        for (i, (p, z)) in resampled.points.iter().zip(expected_z).enumerate() {
            assert_relative_eq!(p.contour_point.z, z, epsilon = 1e-12);
            assert_eq!(p.contour_point.frame_index, i as u32);
            assert_eq!(p.branch_id, 0);
        }
    }

    #[test]
    fn test_preprocess_centerline_tangents_follow_final_order() {
        // from_contour_points gives +z tangents for ascending input; after the
        // reversal every tangent must point along the new descending-z order.
        let cl = cl_from_coords(&[(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 2.0)]);
        let geom = geom_from_centroids(&[(0.0, 0.0, 0.0), (0.0, 0.0, 0.5)]);

        let (resampled, _, _) = preprocess_centerline(cl, &geom, &(0.0, 0.0, 2.0)).unwrap();

        for p in &resampled.points {
            assert_relative_eq!(p.tangent.z, -1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_preprocess_centerline_keeps_reference_exact() {
        // Dense 0.1 mm centerline, 1 mm frame spacing. A grid started at the top
        // (z = 10) would snap the reference to z = 4.0; anchored it stays at 4.3.
        let coords: Vec<_> = (0..=100).map(|i| (0.0, 0.0, i as f64 * 0.1)).collect();
        let cl = cl_from_coords(&coords);
        let geom = geom_from_centroids(&[(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]);

        let (resampled, ref_idx, _) = preprocess_centerline(cl, &geom, &(0.5, 0.0, 4.32)).unwrap();

        let z = |i: usize| resampled.points[i].contour_point.z;
        assert_relative_eq!(z(ref_idx), 4.3, epsilon = 1e-9);
        assert_relative_eq!(z(ref_idx - 1), 5.3, epsilon = 1e-9);
        assert_relative_eq!(z(ref_idx + 1), 3.3, epsilon = 1e-9);
        assert_relative_eq!(z(0), 10.0, epsilon = 1e-9);
    }

    #[test]
    fn test_preprocess_centerline_errors() {
        let cl = cl_from_coords(&[(0.0, 0.0, 1.0), (0.0, 0.0, 0.0)]);
        assert_eq!(
            preprocess_centerline(cl, &geom_from_centroids(&[]), &(0.0, 0.0, 0.0)).unwrap_err(),
            "Reference mesh has no frames"
        );

        let empty = Centerline {
            points: vec![],
            branch_start_indices: vec![],
        };
        assert_eq!(
            preprocess_centerline(
                empty,
                &geom_from_centroids(&[(0.0, 0.0, 0.0)]),
                &(0.0, 0.0, 0.0)
            )
            .unwrap_err(),
            "Centerline has no branch-0 points"
        );
    }
}
