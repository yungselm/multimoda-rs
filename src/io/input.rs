use anyhow::{anyhow, Context, Result};
use csv::ReaderBuilder;
use nalgebra::Vector3;
use serde::Deserialize;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Utility: detect whether the file uses comma or tab as delimiter.
fn detect_delimiter<P: AsRef<Path>>(path: P) -> Result<u8> {
    let file = File::open(&path).with_context(|| {
        format!(
            "failed to open file for delimiter sniffing: {:?}",
            path.as_ref()
        )
    })?;
    let mut reader = BufReader::new(file);
    let mut first_line = String::new();
    reader
        .read_line(&mut first_line)
        .with_context(|| "failed to read first line for delimiter detection")?;

    // Count occurrences
    let tabs = first_line.matches('\t').count();
    let commas = first_line.matches(',').count();

    if tabs > commas {
        Ok(b'\t')
    } else if commas > tabs {
        Ok(b',')
    } else {
        // default to comma
        Ok(b',')
    }
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
pub struct Contour {
    pub id: u32,
    pub points: Vec<ContourPoint>,
    pub centroid: (f64, f64, f64),
    pub aortic_thickness: Option<f64>,
    pub pulmonary_thickness: Option<f64>,
}

impl Contour {
    pub fn create_contours(
        points: Vec<ContourPoint>,
        result: Vec<Record>,
    ) -> anyhow::Result<Vec<Contour>> {
        let mut groups: HashMap<u32, Vec<ContourPoint>> = HashMap::new();
        for p in points {
            groups.entry(p.frame_index).or_default().push(p);
        }

        let mut contours = Vec::new();
        for (frame_index, group_points) in groups {
            let centroid = Self::compute_centroid(&group_points);
            let aortic_thickness = result
                .iter()
                .find(|r| r.frame == frame_index as u32)
                .and_then(|r| r.measurement_1);

            let pulmonary_thickness = result
                .iter()
                .find(|r| r.frame == frame_index as u32)
                .and_then(|r| r.measurement_2);

            contours.push(Contour {
                id: frame_index,
                points: group_points,
                centroid,
                aortic_thickness: aortic_thickness,
                pulmonary_thickness: pulmonary_thickness,
            });

            for contour in &mut contours {
                for (i, point) in contour.points.iter_mut().enumerate() {
                    point.point_index = i as u32;
                }
            }
        }
        Ok(contours)
    }

    pub fn create_catheter_contours(
        points: &Vec<ContourPoint>,
        image_center: (f64, f64),
        radius: f64,
        n_points: u32,
    ) -> anyhow::Result<Vec<Contour>> {
        let catheter_points =
            ContourPoint::create_catheter_points(&points, image_center, radius, n_points);

        let mut groups: HashMap<u32, Vec<ContourPoint>> = HashMap::new();
        for p in catheter_points {
            groups.entry(p.frame_index).or_default().push(p);
        }

        let mut contours = Vec::new();
        for (frame_index, group_points) in groups {
            let centroid = Self::compute_centroid(&group_points);
            let aortic_thickness = None;
            let pulmonary_thickness = None;

            contours.push(Contour {
                id: frame_index,
                points: group_points,
                centroid,
                aortic_thickness: aortic_thickness,
                pulmonary_thickness: pulmonary_thickness,
            });
        }
        Ok(contours)
    }

    pub fn compute_centroid(points: &Vec<ContourPoint>) -> (f64, f64, f64) {
        let (sum_x, sum_y, sum_z) = points.iter().fold((0.0, 0.0, 0.0), |(sx, sy, sz), p| {
            (sx + p.x, sy + p.y, sz + p.z)
        });
        let n = points.len() as f64;
        (sum_x / n, sum_y / n, sum_z / n)
    }

    /// Finds the pair of farthest points in the current contour.
    pub fn find_farthest_points(&self) -> ((&ContourPoint, &ContourPoint), f64) {
        let mut max_dist = 0.0;
        let mut farthest_pair = (&self.points[0], &self.points[0]);

        for i in 0..self.points.len() {
            for j in i + 1..self.points.len() {
                let dx = self.points[i].x - self.points[j].x;
                let dy = self.points[i].y - self.points[j].y;
                let dist = (dx * dx + dy * dy).sqrt();
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

        // 1) Compute centroid (x0,y0)
        let (cx, cy, _) = self.centroid;

        // 2) Precompute angles
        let thetas: Vec<f64> = self
            .points
            .iter()
            .map(|p| {
                let mut t = (p.y - cy).atan2(p.x - cx);
                if t < 0.0 {
                    t += 2.0 * std::f64::consts::PI;
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
                if delta > std::f64::consts::PI {
                    delta = 2.0 * std::f64::consts::PI - delta;
                }
                let diff = (delta - std::f64::consts::PI).abs();
                if diff < best_angle_diff {
                    best_angle_diff = diff;
                    best_j = j;
                }
            }

            // 4) Compute chord length between i and best_j
            let pi = &self.points[i];
            let pj = &self.points[best_j];
            let dx = pi.x - pj.x;
            let dy = pi.y - pj.y;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < min_dist {
                min_dist = dist;
                best_pair = (pi, pj);
            }
        }

        (best_pair, min_dist)
    }

    pub fn elliptic_ratio(&self) -> f64 {
        let major_length = self.find_farthest_points().1;
        let minor_length = self.find_closest_opposite().1;
        if major_length < minor_length {
            minor_length / major_length
        } else {
            major_length / minor_length
        }
    }

    pub fn area(&self) -> f64 {
        let n = self.points.len();
        let mut area = 0.0;
        for i in 0..n {
            let p1 = &self.points[i];
            let p2 = &self.points[(i + 1) % n];
            area += p1.x * p2.y - p2.x * p1.y;
        }
        0.5 * area.abs()
    }

    /// Angle in radians
    pub fn rotate_contour(&mut self, angle: f64) {
        let center = (self.centroid.0, self.centroid.1);
        // Replace entire points instead of just coordinates
        self.points = self
            .points
            .iter()
            .map(|p| p.rotate_point(angle, center))
            .collect();
    }

    /// Angle in radians
    pub fn rotate_contour_around_point(&mut self, angle: f64, center: (f64, f64)) {
        // Replace entire points instead of just coordinates
        self.points = self
            .points
            .iter()
            .map(|p| p.rotate_point(angle, center))
            .collect();
    }

    /// Reorders `self.points` so that:
    /// 1) They’re sorted counterclockwise around the centroid,
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

    /// Translates a contour by a given (dx, dy, dz) offset and recalculates the centroid.
    pub fn translate_contour(&mut self, translation: (f64, f64, f64)) {
        let (dx, dy, dz) = translation;
        for p in self.points.iter_mut() {
            p.x += dx;
            p.y += dy;
            p.z += dz;
        }
        // Recalculate the centroid
        self.centroid = Self::compute_centroid(&self.points);
    }
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq)]
pub struct ContourPoint {
    pub frame_index: u32,

    #[serde(default, skip_deserializing)]
    pub point_index: u32,

    pub x: f64,
    pub y: f64,
    pub z: f64,

    #[serde(default)]
    pub aortic: bool,
}

impl ContourPoint {
    /// Reads contour points from a CSV file.
    pub fn read_contour_data<P: AsRef<Path> + std::fmt::Debug + Clone>(
        path: P,
    ) -> anyhow::Result<Vec<ContourPoint>> {
        let delim = detect_delimiter(&path)?;
        let file = File::open(path)?;
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(false)
            .delimiter(delim)
            .from_reader(file);

        let mut points = Vec::new();
        for result in rdr.records() {
            match result {
                Ok(record) => match record.deserialize(None) {
                    Ok(point) => points.push(point),
                    Err(e) => eprintln!("Skipping invalid record: {:?}", e),
                },
                Err(e) => eprintln!("Skipping invalid row: {:?}", e),
            }
        }

        Ok(points)
    }

    pub fn read_reference_point<P: AsRef<Path>>(path: P) -> Result<ContourPoint> {
        let delim = detect_delimiter(&path)?;
        // 1) Open the file, with context on failur
        let file = File::open(&path)
            .with_context(|| format!("failed to open reference-point file {:?}", path.as_ref()))?;

        // 2) Build a TSV reader
        let mut rdr = ReaderBuilder::new()
            .has_headers(false)
            .delimiter(delim)
            .from_reader(file);

        // 3) Grab the first record (if any)…
        let first = rdr
            .deserialize()
            .next()
            // if there was literally no row at all:
            .ok_or_else(|| {
                anyhow!(
                    "reference-point file {:?} was empty — this data is required",
                    path.as_ref()
                )
            })?;

        // 4) And now propagate any parse / I/O error with its own context:
        let point: ContourPoint =
            first.with_context(|| "failed to deserialize first reference-point record")?;
        Ok(point)
    }

    pub fn create_catheter_points(
        points: &Vec<ContourPoint>,
        image_center: (f64, f64),
        radius: f64,
        n_points: u32,
    ) -> Vec<ContourPoint> {
        // Map to store unique frame indices and one associated z coordinate per frame.
        let mut frame_z: HashMap<u32, f64> = HashMap::new();
        for point in points {
            // Use the first encountered z-coordinate for each frame index.
            frame_z.entry(point.frame_index).or_insert(point.z);
        }

        let mut catheter_points = Vec::new();
        // Sort the frame indices to ensure a predictable order.
        let mut frames: Vec<u32> = frame_z.keys().cloned().collect();
        frames.sort();

        // Parameters for the catheter circle.
        let center_x = image_center.0;
        let center_y = image_center.1;
        let radius = radius;
        let num_points = n_points;

        // For each unique frame, generate 20 catheter points around a circle.
        for frame in frames {
            let z = frame_z[&frame];
            for i in 0..num_points {
                let angle = 2.0 * PI * (i as f64) / (num_points as f64);
                let x = center_x + radius * angle.cos();
                let y = center_y + radius * angle.sin();
                catheter_points.push(ContourPoint {
                    frame_index: frame,
                    point_index: i,
                    x,
                    y,
                    z,
                    aortic: false,
                });
            }
        }
        catheter_points
    }

    /// Computes the Euclidean distance between two contour points.
    pub fn distance_to(&self, other: &ContourPoint) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Rotates a single point about a given center (cx, cy) by a specified angle (in radians).
    pub fn rotate_point(&self, angle: f64, center: (f64, f64)) -> ContourPoint {
        let (cx, cy) = center;
        let x = self.x - cx;
        let y = self.y - cy;
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        ContourPoint {
            frame_index: self.frame_index,
            point_index: self.point_index,
            x: x * cos_a - y * sin_a + cx,
            y: x * sin_a + y * cos_a + cy,
            z: self.z,
            aortic: self.aortic,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Record {
    pub frame: u32,
    pub phase: String,
    #[serde(deserialize_with = "csv::invalid_option")]
    pub measurement_1: Option<f64>,
    #[serde(deserialize_with = "csv::invalid_option")]
    pub measurement_2: Option<f64>,
}

pub fn read_records<P: AsRef<Path>>(path: P) -> anyhow::Result<Vec<Record>> {
    let delim = detect_delimiter(&path)?;
    let file = File::open(path)?;
    let mut reader = ReaderBuilder::new()
        .delimiter(delim)
        .has_headers(true)
        .from_reader(file);

    let mut records = Vec::new();
    for result in reader.deserialize() {
        let record: Record = result?;
        records.push(record);
    }
    Ok(records)
}

#[derive(Debug, Clone, PartialEq)]
pub struct Centerline {
    pub points: Vec<CenterlinePoint>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CenterlinePoint {
    pub contour_point: ContourPoint,
    pub normal: Vector3<f64>,
}

impl Centerline {
    pub fn from_contour_points(contour_points: Vec<ContourPoint>) -> Self {
        let mut points: Vec<CenterlinePoint> = Vec::with_capacity(contour_points.len());

        // Calculate normals for all but the last point.
        for i in 0..contour_points.len() {
            let current = &contour_points[i];
            let normal = if i < contour_points.len() - 1 {
                let next = &contour_points[i + 1];
                Vector3::new(next.x - current.x, next.y - current.y, next.z - current.z).normalize()
            } else if !contour_points.is_empty() {
                points[i - 1].normal
            } else {
                Vector3::zeros()
            };

            points.push(CenterlinePoint {
                contour_point: current.clone(),
                normal,
            });
        }

        Centerline { points }
    }

    /// Retrieves a centerline point by matching frame index.
    pub fn get_by_frame(&self, frame_index: u32) -> Option<&CenterlinePoint> {
        self.points
            .iter()
            .find(|p| p.contour_point.frame_index == frame_index)
    }
}

#[cfg(test)]
mod input_tests {
    use super::*;

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
        let centroid = Contour::compute_centroid(&points);
        assert_eq!(centroid, (1.0, 1.0, 0.0));
    }

    #[test]
    fn test_find_farthest_points() {
        let contour = Contour {
            id: 1,
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
            centroid: (1.0, 1.0, 0.0),
            aortic_thickness: None,
            pulmonary_thickness: None,
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
        let contour = Contour {
            id: 1,
            points: vec![
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
            ],
            centroid: Contour::compute_centroid(&vec![
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
            ]),
            aortic_thickness: None,
            pulmonary_thickness: None,
        };
        let (pair, distance) = contour.find_closest_opposite();
        assert!((distance - 1.5).abs() < 1e-6);
        let p0 = &contour.points[0];
        let p2 = &contour.points[2];
        assert!((pair.0 == p0 && pair.1 == p2) || (pair.0 == p2 && pair.1 == p0));
    }

    #[test]
    fn test_rotate_contour() {
        let mut contour = Contour {
            id: 1,
            points: vec![
                ContourPoint {
                    frame_index: 1,
                    point_index: 0,
                    x: 2.0,
                    y: 1.0,
                    z: 0.0,
                    aortic: false,
                },
                ContourPoint {
                    frame_index: 1,
                    point_index: 1,
                    x: 1.0,
                    y: 2.0,
                    z: 0.0,
                    aortic: false,
                },
                ContourPoint {
                    frame_index: 1,
                    point_index: 2,
                    x: 0.0,
                    y: 1.0,
                    z: 0.0,
                    aortic: false,
                },
                ContourPoint {
                    frame_index: 1,
                    point_index: 3,
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                    aortic: false,
                },
            ],
            centroid: (1.0, 1.0, 0.0),
            aortic_thickness: None,
            pulmonary_thickness: None,
        };
        contour.rotate_contour(PI / 2.0);
        let expected_points = vec![
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
                x: 0.0,
                y: 1.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 2,
                x: 1.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 3,
                x: 2.0,
                y: 1.0,
                z: 0.0,
                aortic: false,
            },
        ];
        for (i, point) in contour.points.iter().enumerate() {
            assert!((point.x - expected_points[i].x).abs() < 1e-6);
            assert!((point.y - expected_points[i].y).abs() < 1e-6);
        }
    }

    #[test]
    fn test_rotate_contour_around_point() {
        // Create a square contour centered at (0, 0)
        let mut contour = Contour {
            id: 1,
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
                    y: 1.0,
                    z: 0.0,
                    aortic: false,
                },
                ContourPoint {
                    frame_index: 1,
                    point_index: 2,
                    x: -1.0,
                    y: 0.0,
                    z: 0.0,
                    aortic: false,
                },
                ContourPoint {
                    frame_index: 1,
                    point_index: 3,
                    x: 0.0,
                    y: -1.0,
                    z: 0.0,
                    aortic: false,
                },
            ],
            centroid: (0.0, 0.0, 0.0),
            aortic_thickness: None,
            pulmonary_thickness: None,
        };
        // Rotate 180 degrees (PI) around point (1, 1)
        let mut new_contour = contour.clone();

        new_contour.rotate_contour_around_point(PI, (1.0, 1.0));
        let expected_points = vec![
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
                x: 2.0,
                y: 1.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 2,
                x: 3.0,
                y: 2.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 3,
                x: 2.0,
                y: 3.0,
                z: 0.0,
                aortic: false,
            },
        ];

        for (i, point) in new_contour.points.iter().enumerate() {
            assert!(
                (point.x - expected_points[i].x).abs() < 1e-6,
                "x mismatch at {}",
                i
            );
            assert!(
                (point.y - expected_points[i].y).abs() < 1e-6,
                "y mismatch at {}",
                i
            );
        }
        // Rotate 90 degrees (PI/2) around point (1, 1) to ensure direction
        contour.rotate_contour_around_point(PI / 2.0, (1.0, 1.0));
        let expected_points = vec![
            ContourPoint {
                frame_index: 1,
                point_index: 0,
                x: 2.0,
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
                x: 2.0,
                y: -1.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 1,
                point_index: 3,
                x: 3.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
        ];

        for (i, point) in contour.points.iter().enumerate() {
            assert!(
                (point.x - expected_points[i].x).abs() < 1e-6,
                "x mismatch at {}",
                i
            );
            assert!(
                (point.y - expected_points[i].y).abs() < 1e-6,
                "y mismatch at {}",
                i
            );
        }
    }

    #[test]
    fn test_sort_contour_points() {
        let mut contour = Contour {
            id: 1,
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
            centroid: (0.0, 0.0, 0.0),
            aortic_thickness: None,
            pulmonary_thickness: None,
        };
        contour.sort_contour_points();
        // let expected_order = vec![0.0, 2.0, 0.0, -2.0, -2.0, 0.0, 2.0, 0.0];
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
    fn test_translate_contour() {
        let mut contour = Contour {
            id: 1,
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
            centroid: (1.0, 1.0, 0.0),
            aortic_thickness: None,
            pulmonary_thickness: None,
        };
        contour.translate_contour((1.0, 2.0, 3.0));
        assert_eq!(contour.centroid, (2.0, 3.0, 3.0));
        for point in contour.points {
            assert_eq!(point.z, 3.0);
        }
    }

    #[test]
    fn test_contour_point_distance() {
        let p1 = ContourPoint {
            frame_index: 1,
            point_index: 0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            aortic: false,
        };
        let p2 = ContourPoint {
            frame_index: 1,
            point_index: 1,
            x: 3.0,
            y: 4.0,
            z: 0.0,
            aortic: false,
        };
        assert!((p1.distance_to(&p2) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_rotate_point() {
        let p = ContourPoint {
            frame_index: 1,
            point_index: 0,
            x: 1.0,
            y: 0.0,
            z: 0.0,
            aortic: false,
        };
        let rotated = p.rotate_point(PI / 2.0, (0.0, 0.0));
        assert!((rotated.x - 0.0).abs() < 1e-6);
        assert!((rotated.y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_create_catheter_points() {
        let points = vec![ContourPoint {
            frame_index: 1,
            point_index: 0,
            x: 0.0,
            y: 0.0,
            z: 5.0,
            aortic: false,
        }];
        let catheter_points = ContourPoint::create_catheter_points(&points, (4.5, 4.5), 0.5, 20);
        assert_eq!(catheter_points.len(), 20);
        for point in catheter_points {
            assert_eq!(point.frame_index, 1);
            assert_eq!(point.z, 5.0);
            let dx = point.x - 4.5;
            let dy = point.y - 4.5;
            let dist = (dx * dx + dy * dy).sqrt();
            assert!((dist - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_centerline_normals() {
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
                frame_index: 2,
                point_index: 1,
                x: 1.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
            ContourPoint {
                frame_index: 3,
                point_index: 2,
                x: 2.0,
                y: 0.0,
                z: 0.0,
                aortic: false,
            },
        ];
        let centerline = Centerline::from_contour_points(points);
        assert_eq!(centerline.points[0].normal, Vector3::new(1.0, 0.0, 0.0));
        assert_eq!(centerline.points[1].normal, Vector3::new(1.0, 0.0, 0.0));
        assert_eq!(centerline.points[2].normal, Vector3::new(1.0, 0.0, 0.0));
    }
}
