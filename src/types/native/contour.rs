use super::contour_point::ContourPoint;
use super::record::Record;
use super::{Point3D, Transform};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContourType {
    Lumen,
    Eem,
    Calcification,
    Sidebranch,
    Catheter,
    Wall,
}

impl fmt::Display for ContourType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                ContourType::Lumen => "Lumen",
                ContourType::Eem => "Eem",
                ContourType::Calcification => "Calcification",
                ContourType::Sidebranch => "Sidebranch",
                ContourType::Catheter => "Catheter",
                ContourType::Wall => "Wall",
            }
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Contour {
    pub id: u32,
    pub original_frame: u32,
    pub points: Vec<ContourPoint>,
    pub centroid: Option<(f64, f64, f64)>,
    pub aortic_thickness: Option<f64>,
    pub pulmonary_thickness: Option<f64>,
    pub kind: ContourType,
}

/// Returns up to `n` evenly-strided samples from `points`, preserving order.
pub fn downsample_contour_points(points: &[ContourPoint], n: usize) -> Vec<ContourPoint> {
    if points.len() <= n {
        return points.to_vec();
    }
    let step = points.len() as f64 / n as f64;
    (0..n)
        .map(|i| {
            let index = (i as f64 * step) as usize;
            points[index]
        })
        .collect()
}

impl Transform for Contour {
    fn translate(mut self, dx: f64, dy: f64, dz: f64) -> Self {
        for p in &mut self.points {
            p.translate_mut(dx, dy, dz);
        }
        self
    }

    fn rotate(mut self, angle: f64, center: (f64, f64)) -> Self {
        if angle == 0.0 {
            return self;
        }
        for p in &mut self.points {
            p.rotate_mut(angle, center);
        }
        self
    }
}

impl Contour {
    pub fn new(
        id: u32,
        points: Vec<ContourPoint>,
        records: Option<Vec<Record>>,
        kind: ContourType,
    ) -> Self {
        let (aortic, pulmonary) = if let Some(ref records) = records {
            records
                .iter()
                .find(|r| r.frame == id)
                .map(|r| (Some(r.measurement_1), Some(r.measurement_2)))
                .unwrap_or((None, None))
        } else {
            (None, None)
        };

        Contour {
            id,
            original_frame: id,
            points,
            centroid: None,
            aortic_thickness: aortic.flatten(),
            pulmonary_thickness: pulmonary.flatten(),
            kind,
        }
    }

    pub fn build_contour(
        points: Vec<ContourPoint>,
        records: Option<Vec<Record>>,
        kind: ContourType,
    ) -> anyhow::Result<Vec<Contour>> {
        let mut groups: HashMap<u32, Vec<ContourPoint>> = HashMap::new();
        for p in points {
            groups.entry(p.frame_index).or_default().push(p);
        }

        let mut contours = Vec::new();

        // Only process measurements for Lumen contour type
        let measurements = if kind == ContourType::Lumen {
            let mut meas = HashMap::new();
            if let Some(records) = records {
                for record in records {
                    meas.insert(record.frame, (record.measurement_1, record.measurement_2));
                }
            }
            Some(meas)
        } else {
            None
        };

        let mut sorted_groups: Vec<(u32, Vec<ContourPoint>)> = groups.into_iter().collect();
        sorted_groups.sort_by_key(|(frame_idx, _)| *frame_idx);

        for (i, (frame_idx, points)) in sorted_groups.into_iter().enumerate() {
            let (aortic, pulmonary) = if let Some(ref meas) = measurements {
                meas.get(&frame_idx)
                    .map(|&(m1, m2)| (Some(m1), Some(m2)))
                    .unwrap_or((None, None))
            } else {
                (None, None)
            };

            contours.push(Contour {
                id: i as u32,
                original_frame: frame_idx,
                points,
                centroid: None,
                aortic_thickness: aortic.flatten(),
                pulmonary_thickness: pulmonary.flatten(),
                kind,
            });
        }

        Ok(contours)
    }

    pub fn build_contour_with_mapping(
        points: Vec<ContourPoint>,
        records: Option<Vec<Record>>,
        kind: ContourType,
        frame_mapping: &HashMap<u32, u32>,
    ) -> anyhow::Result<Vec<Contour>> {
        let mut groups: HashMap<u32, Vec<ContourPoint>> = HashMap::new();
        for p in points {
            groups.entry(p.frame_index).or_default().push(p);
        }

        let mut contours = Vec::new();

        let measurements = if kind == ContourType::Lumen {
            let mut meas = HashMap::new();
            if let Some(records) = records {
                for record in records {
                    meas.insert(record.frame, (record.measurement_1, record.measurement_2));
                }
            }
            Some(meas)
        } else {
            None
        };

        let mut sorted_groups: Vec<(u32, Vec<ContourPoint>)> = groups.into_iter().collect();
        sorted_groups.sort_by_key(|(frame_idx, _)| *frame_idx);

        for (original_frame_idx, points) in sorted_groups {
            let sequential_id = *frame_mapping.get(&original_frame_idx).ok_or_else(|| {
                anyhow::anyhow!("No mapping found for original frame {original_frame_idx}")
            })?;

            let (aortic, pulmonary) = if let Some(ref meas) = measurements {
                meas.get(&original_frame_idx)
                    .map(|&(m1, m2)| (Some(m1), Some(m2)))
                    .unwrap_or((None, None))
            } else {
                (None, None)
            };

            contours.push(Contour {
                id: sequential_id,                  // Use the shared sequential ID
                original_frame: original_frame_idx, // Keep original frame for reference
                points,
                centroid: None,
                aortic_thickness: aortic.flatten(),
                pulmonary_thickness: pulmonary.flatten(),
                kind,
            });
        }

        Ok(contours)
    }

    pub fn compute_centroid(&mut self) {
        if self.points.is_empty() {
            self.centroid = None;
            return;
        }

        let (sum_x, sum_y, sum_z) = self.points.iter().fold((0.0, 0.0, 0.0), |(sx, sy, sz), p| {
            (sx + p.x, sy + p.y, sz + p.z)
        });
        let n = self.points.len() as f64;
        self.centroid = Some((sum_x / n, sum_y / n, sum_z / n));
    }

    /// Finds the pair of farthest points in the current contour.
    pub fn find_farthest_points(&self) -> ((&ContourPoint, &ContourPoint), f64) {
        let mut max_dist = 0.0;
        let mut farthest_pair = (&self.points[0], &self.points[0]);

        for i in 0..self.points.len() {
            for j in i + 1..self.points.len() {
                let dist = self.points[i].distance_to(&self.points[j]);
                if dist > max_dist {
                    max_dist = dist;
                    farthest_pair = (&self.points[i], &self.points[j]);
                }
            }
        }

        (farthest_pair, max_dist)
    }

    /// Find the pair of points whose coord is the smallest diameter,
    /// by matching each point to the one whose angle (about the centroid)
    /// differs by as close to π radians as possible.
    pub fn find_closest_opposite(&self) -> ((&ContourPoint, &ContourPoint), f64) {
        let n = self.points.len();
        assert!(n > 2, "Need at least 3 points");

        let (cx, cy, _) = if let Some(c) = self.centroid {
            c
        } else {
            let (sum_x, sum_y, sum_z) =
                self.points.iter().fold((0.0, 0.0, 0.0), |(sx, sy, sz), p| {
                    (sx + p.x, sy + p.y, sz + p.z)
                });
            let n_f = n as f64;
            (sum_x / n_f, sum_y / n_f, sum_z / n_f)
        };

        let thetas: Vec<f64> = self
            .points
            .iter()
            .map(|p| {
                let mut t = (p.y - cy).atan2(p.x - cx);
                if t < 0.0 {
                    t += 2.0 * PI;
                }
                t
            })
            .collect();

        let mut min_dist = f64::MAX;
        let mut best_pair = (&self.points[0], &self.points[1]);

        // 3) Brute‐force: for each i, find j that best approximates θi+π
        for i in 0..n {
            let mut best_angle_diff = f64::MAX;
            let mut best_j = i;

            for j in 0..n {
                if j == i {
                    continue;
                }
                // compute angular separation in [0,2π)
                let mut delta = (thetas[j] - thetas[i]).abs();
                if delta > PI {
                    delta = 2.0 * PI - delta;
                }
                let diff = (delta - PI).abs();
                if diff < best_angle_diff {
                    best_angle_diff = diff;
                    best_j = j;
                }
            }

            // 4) Compute chord length between i and best_j
            let pi = &self.points[i];
            let pj = &self.points[best_j];
            let dist = pi.distance_2d_to(pj);
            if dist < min_dist {
                min_dist = dist;
                best_pair = (pi, pj);
            }
        }

        (best_pair, min_dist)
    }

    /// Finds the minimum diameter in 3D by pairing each point with the point at
    /// the opposite index (i + n/2) and returning the pair with the smallest 3D distance.
    pub fn find_closest_opposite_3d(&self) -> ((&ContourPoint, &ContourPoint), f64) {
        let n = self.points.len();
        assert!(n > 2, "Need at least 3 points");

        let half = n / 2;
        let mut min_dist = f64::MAX;
        let mut best_pair = (&self.points[0], &self.points[half]);

        for i in 0..n {
            let j = (i + half) % n;
            let pi = &self.points[i];
            let pj = &self.points[j];
            let dist = pi.distance_to(pj);
            if dist < min_dist {
                min_dist = dist;
                best_pair = (pi, pj);
            }
        }

        (best_pair, min_dist)
    }

    pub fn elliptic_ratio(&self) -> f64 {
        let major_length = self.find_farthest_points().1;
        let minor_length = self.find_closest_opposite_3d().1;
        if major_length < minor_length {
            minor_length / major_length
        } else {
            major_length / minor_length
        }
    }

    pub fn area(&self) -> f64 {
        let n = self.points.len();
        if n < 3 {
            return 0.0;
        }

        let mut cx = 0.0_f64;
        let mut cy = 0.0_f64;
        let mut cz = 0.0_f64;
        for i in 0..n {
            let p1 = &self.points[i];
            let p2 = &self.points[(i + 1) % n];
            cx += p1.y * p2.z - p1.z * p2.y;
            cy += p1.z * p2.x - p1.x * p2.z;
            cz += p1.x * p2.y - p1.y * p2.x;
        }
        0.5 * (cx * cx + cy * cy + cz * cz).sqrt()
    }

    /// Reorders `self.points` so that:
    /// 1) They're sorted counterclockwise around the centroid,
    /// 2) The point with the highest Y-value is at index 0,
    /// 3) Each `point.point_index` matches its position in the Vec.
    pub fn sort_contour_points(&mut self) {
        let n = self.points.len() as f64;
        if n == 0.0 {
            return;
        }

        // 1) Compute centroid (cx, cy)
        let (sum_x, sum_y) = self
            .points
            .iter()
            .fold((0.0, 0.0), |(sx, sy), p| (sx + p.x, sy + p.y));
        let cx = sum_x / n;
        let cy = sum_y / n;

        // 2) Sort by *descending* angle around centroid (clockwise sweep)
        self.points.sort_by(|a, b| {
            let angle_a = (a.y - cy).atan2(a.x - cx);
            let angle_b = (b.y - cy).atan2(b.x - cx);
            // flip the comparison order ?
            angle_a.partial_cmp(&angle_b).unwrap()
        });

        // 3) Find the index of the highest y-coord point and rotate it to front
        if let Some(start_idx) = self
            .points
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.y.partial_cmp(&b.y).unwrap())
            .map(|(i, _)| i)
        {
            self.points.rotate_left(start_idx);
        }

        // 4) Re-index in array order
        for (i, pt) in self.points.iter_mut().enumerate() {
            pt.point_index = i as u32;
        }
    }

    /// Rotate the Vec so the point currently at position `shift` moves to
    /// index 0, then reassign `point_index` sequentially.  X/Y/Z unchanged.
    pub fn rotate_and_reindex(&mut self, shift: usize) {
        let n = self.points.len();
        if n == 0 || shift == 0 {
            return;
        }
        self.points.rotate_left(shift % n);
        for (i, pt) in self.points.iter_mut().enumerate() {
            pt.point_index = i as u32;
        }
    }
}

#[cfg(test)]
mod contour_tests {
    use super::*;
    use crate::intravascular::utils::test_utils::dummy_geometry;
    use crate::types::native::record::Record;

    #[test]
    fn test_build_contour_groups_by_frame() -> anyhow::Result<()> {
        // create points: two points for frame 1, one point for frame 2
        let pts = vec![
            ContourPoint {
                frame_index: 1,
                point_index: 0,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 1,
                x: 1.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 2,
                point_index: 0,
                x: 2.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
        ];

        let contours = Contour::build_contour(pts, None, ContourType::Lumen)?;
        // sort by id for deterministic assertions
        let mut contours = contours;
        contours.sort_by_key(|c| c.id);

        assert_eq!(
            contours.len(),
            2,
            "should produce two contours (frames 1 and 2)"
        );
        assert_eq!(contours[0].id, 0);
        assert_eq!(contours[0].original_frame, 1);
        assert_eq!(
            contours[0].points.len(),
            2,
            "frame 1 should have two points"
        );
        assert_eq!(contours[1].id, 1);
        assert_eq!(contours[1].original_frame, 2);
        assert_eq!(contours[1].points.len(), 1, "frame 2 should have one point");

        Ok(())
    }

    #[test]
    fn test_build_contour_attaches_measurements_for_lumen() -> anyhow::Result<()> {
        // points for frame 1
        let pts = vec![ContourPoint {
            frame_index: 1,
            point_index: 0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            aortic: false,
        }];

        // create a Record that contains measurement_1 and measurement_2 for frame 1
        let records = vec![Record {
            frame: 1,
            phase: "systolic".to_string(),
            measurement_1: Some(1.23),
            measurement_2: Some(4.56),
        }];

        let mut contours = Contour::build_contour(pts, Some(records), ContourType::Lumen)?;
        assert_eq!(contours.len(), 1);
        let c = contours.pop().unwrap();

        assert_eq!(c.id, 0);
        assert_eq!(c.aortic_thickness, Some(1.23));
        assert_eq!(c.pulmonary_thickness, Some(4.56));

        Ok(())
    }

    #[test]
    fn test_build_contour_ignores_measurements_for_non_lumen() -> anyhow::Result<()> {
        // points for frame 1
        let pts = vec![ContourPoint {
            frame_index: 1,
            point_index: 0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            aortic: false,
        }];

        // create a Record that contains measurement_1 and measurement_2 for frame 1
        let records = vec![Record {
            frame: 1,
            phase: "systolic".to_string(),
            measurement_1: Some(7.0),
            measurement_2: Some(8.0),
        }];

        // Use a non-lumen kind — measurements should not be applied
        let mut contours = Contour::build_contour(pts, Some(records), ContourType::Eem)?;
        assert_eq!(contours.len(), 1);
        let c = contours.pop().unwrap();

        assert_eq!(c.id, 0);
        assert_eq!(
            c.aortic_thickness, None,
            "non-lumen contours shouldn't pick up measurements"
        );
        assert_eq!(c.pulmonary_thickness, None);

        Ok(())
    }

    #[test]
    fn test_downsample_geometry() {
        let dummy = dummy_geometry();
        let cont_points = dummy.frames[0].lumen.points.clone();
        let downsampled_contour = downsample_contour_points(&cont_points, 3);

        assert_eq!(downsampled_contour.len(), 3);
        assert_eq!(downsampled_contour[0].point_index, 0);
        assert_eq!(downsampled_contour[1].point_index, 2);

        let downsampled_contour = downsample_contour_points(&cont_points, 6);

        assert_eq!(downsampled_contour.len(), 6);
        assert_eq!(downsampled_contour[0].point_index, 0);
        assert_eq!(downsampled_contour[1].point_index, 1);

        let downsampled_contour = downsample_contour_points(&cont_points, 5);
        let n = downsampled_contour.len();
        assert_eq!(downsampled_contour[n - 1].point_index, 4);
    }

    #[test]
    fn test_downsample_edge_cases() {
        let points = vec![
            ContourPoint {
                frame_index: 1,
                point_index: 0,
                x: 1.0,
                y: 2.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 1,
                x: 3.0,
                y: 4.0,
                z: 0.0,
                aortic: false,
            },
        ];

        let result = downsample_contour_points(&points, 5);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].point_index, 0);
        assert_eq!(result[1].point_index, 1);

        let result = downsample_contour_points(&points, 2);
        assert_eq!(result.len(), 2);

        let result = downsample_contour_points(&points, 0);
        assert_eq!(result.len(), 0);

        let empty: Vec<ContourPoint> = Vec::new();
        let result = downsample_contour_points(&empty, 3);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_compute_centroid() {
        let points = vec![
            ContourPoint {
                frame_index: 1,
                point_index: 0,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 1,
                x: 2.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 2,
                x: 2.0,
                y: 2.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 3,
                x: 0.0,
                y: 2.0,
                z: 0.0,
                aortic: false,
            },
        ];

        let mut contour = Contour {
            id: 1,
            original_frame: 1,
            points,
            centroid: None,
            aortic_thickness: None,
            pulmonary_thickness: None,
            kind: ContourType::Lumen,
        };

        contour.compute_centroid();
        assert_eq!(contour.centroid, Some((1.0, 1.0, 0.0)));
    }

    #[test]
    fn test_find_farthest_points() {
        let contour = Contour {
            id: 1,
            original_frame: 1,
            points: vec![
                ContourPoint {
                    frame_index: 1,
                    point_index: 0,
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    aortic: false,
                },
                ContourPoint {
                    frame_index: 1,
                    point_index: 1,
                    x: 2.0,
                    y: 0.0,
                    z: 0.0,
                    aortic: false,
                },
                ContourPoint {
                    frame_index: 1,
                    point_index: 2,
                    x: 2.0,
                    y: 2.0,
                    z: 0.0,
                    aortic: false,
                },
                ContourPoint {
                    frame_index: 1,
                    point_index: 3,
                    x: 0.0,
                    y: 2.0,
                    z: 0.0,
                    aortic: false,
                },
            ],
            centroid: Some((1.0, 1.0, 0.0)),
            aortic_thickness: None,
            pulmonary_thickness: None,
            kind: ContourType::Lumen,
        };

        let (pair, distance) = contour.find_farthest_points();
        assert!((distance - (8.0_f64).sqrt()).abs() < 1e-6);
        assert!(
            (pair.0.x == 0.0 && pair.0.y == 0.0 && pair.1.x == 2.0 && pair.1.y == 2.0)
                || (pair.0.x == 2.0 && pair.0.y == 2.0 && pair.1.x == 0.0 && pair.1.y == 0.0)
        );
    }

    #[test]
    fn test_find_closest_opposite() {
        let points = vec![
            ContourPoint {
                frame_index: 1,
                point_index: 0,
                x: 0.0,
                y: 1.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 1,
                x: 1.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 2,
                x: 0.0,
                y: -0.5,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 3,
                x: -1.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
        ];

        let contour = Contour {
            id: 1,
            original_frame: 1,
            points,
            centroid: Some((0.0, 0.125, 0.0)), // Pre-computed centroid
            aortic_thickness: None,
            pulmonary_thickness: None,
            kind: ContourType::Lumen,
        };

        let (pair, distance) = contour.find_closest_opposite();
        assert!((distance - 1.5).abs() < 1e-6);
        let p0 = &contour.points[0];
        let p2 = &contour.points[2];
        assert!((pair.0 == p0 && pair.1 == p2) || (pair.0 == p2 && pair.1 == p0));
    }

    #[test]
    fn test_sort_contour_points() {
        let mut contour = Contour {
            id: 1,
            original_frame: 1,
            points: vec![
                ContourPoint {
                    frame_index: 1,
                    point_index: 0,
                    x: -2.0,
                    y: 0.0,
                    z: 0.0,
                    aortic: false,
                },
                ContourPoint {
                    frame_index: 1,
                    point_index: 1,
                    x: 0.0,
                    y: 2.0,
                    z: 0.0,
                    aortic: false,
                },
                ContourPoint {
                    frame_index: 1,
                    point_index: 2,
                    x: 2.0,
                    y: 0.0,
                    z: 0.0,
                    aortic: false,
                },
                ContourPoint {
                    frame_index: 1,
                    point_index: 3,
                    x: 0.0,
                    y: -2.0,
                    z: 0.0,
                    aortic: false,
                },
            ],
            centroid: Some((0.0, 0.0, 0.0)),
            aortic_thickness: None,
            pulmonary_thickness: None,
            kind: ContourType::Lumen,
        };

        contour.sort_contour_points();

        for (i, point) in contour.points.iter().enumerate() {
            match i {
                0 => {
                    assert!((point.x - 0.0).abs() < 1e-6);
                    assert!((point.y - 2.0).abs() < 1e-6);
                }
                1 => {
                    assert!((point.x - (-2.0)).abs() < 1e-6);
                    assert!((point.y - 0.0).abs() < 1e-6);
                }
                2 => {
                    assert!((point.x - 0.0).abs() < 1e-6);
                    assert!((point.y - (-2.0)).abs() < 1e-6);
                }
                3 => {
                    assert!((point.x - 2.0).abs() < 1e-6);
                    assert!((point.y - 0.0).abs() < 1e-6);
                }
                _ => panic!("Unexpected index"),
            }
        }
    }

    #[test]
    fn test_area_triangle() {
        // Right triangle with base 3 and height 4 -> area = 6
        let points = vec![
            ContourPoint {
                frame_index: 1,
                point_index: 0,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 1,
                x: 3.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 2,
                x: 0.0,
                y: 4.0,
                z: 0.0,
                aortic: false,
            },
        ];
        let c = Contour {
            id: 1,
            original_frame: 1,
            points,
            centroid: None,
            aortic_thickness: None,
            pulmonary_thickness: None,
            kind: ContourType::Lumen,
        };

        let area = c.area();
        assert!(
            (area - 6.0).abs() < 1e-6,
            "triangle area should be 6.0, got {area}"
        );
    }

    #[test]
    fn test_area_square_ccw_and_cw() {
        let pts_ccw = vec![
            ContourPoint {
                frame_index: 1,
                point_index: 0,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 1,
                x: 1.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 2,
                x: 1.0,
                y: 1.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 3,
                x: 0.0,
                y: 1.0,
                z: 0.0,
                aortic: false,
            },
        ];
        let c_ccw = Contour {
            id: 1,
            original_frame: 1,
            points: pts_ccw,
            centroid: None,
            aortic_thickness: None,
            pulmonary_thickness: None,
            kind: ContourType::Lumen,
        };
        assert!(
            (c_ccw.area() - 1.0).abs() < 1e-6,
            "ccw square area should be 1.0"
        );

        let pts_cw = c_ccw.points.iter().rev().cloned().collect::<Vec<_>>();
        let c_cw = Contour {
            points: pts_cw,
            ..c_ccw.clone()
        };
        assert!(
            (c_cw.area() - 1.0).abs() < 1e-6,
            "cw square area should be 1.0"
        );
    }

    #[test]
    fn test_area_less_than_three_points_is_zero() {
        let empty = Contour {
            id: 0,
            original_frame: 0,
            points: vec![],
            centroid: None,
            aortic_thickness: None,
            pulmonary_thickness: None,
            kind: ContourType::Lumen,
        };
        assert_eq!(empty.area(), 0.0);

        let two = Contour {
            id: 1,
            original_frame: 1,
            points: vec![
                ContourPoint {
                    frame_index: 1,
                    point_index: 0,
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    aortic: false,
                },
                ContourPoint {
                    frame_index: 1,
                    point_index: 1,
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                    aortic: false,
                },
            ],
            centroid: None,
            aortic_thickness: None,
            pulmonary_thickness: None,
            kind: ContourType::Lumen,
        };
        assert_eq!(two.area(), 0.0);
    }

    #[test]
    fn test_elliptic_ratio_and_area() {
        let contour = Contour {
            id: 1,
            original_frame: 1,
            points: vec![
                ContourPoint {
                    frame_index: 1,
                    point_index: 0,
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                    aortic: false,
                },
                ContourPoint {
                    frame_index: 1,
                    point_index: 1,
                    x: 0.0,
                    y: 2.0,
                    z: 0.0,
                    aortic: false,
                },
                ContourPoint {
                    frame_index: 1,
                    point_index: 2,
                    x: 1.0,
                    y: 4.0,
                    z: 0.0,
                    aortic: false,
                },
                ContourPoint {
                    frame_index: 1,
                    point_index: 3,
                    x: 2.0,
                    y: 2.0,
                    z: 0.0,
                    aortic: false,
                },
            ],
            centroid: Some((1.0, 2.0, 0.0)),
            aortic_thickness: None,
            pulmonary_thickness: None,
            kind: ContourType::Lumen,
        };

        let ratio = contour.elliptic_ratio();
        let area = contour.area();

        // For a 4x2 rectangle, elliptic ratio should be ~2.0
        assert!((ratio - 2.0).abs() < 1e-6);
        // Area should be 8.0
        assert!((area - 4.0).abs() < 1e-6);
    }
}
