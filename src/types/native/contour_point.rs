use anyhow::{anyhow, Context, Result};
use csv::ReaderBuilder;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Shared accessor trait for any 3-D point type.
/// Implement this for a type to make it work with generic geometry utilities
/// (e.g. `calculate_squared_distance`).
pub trait Point3D {
    fn x(&self) -> f64;
    fn y(&self) -> f64;
    fn z(&self) -> f64;
}

impl Point3D for ContourPoint {
    fn x(&self) -> f64 {
        self.x
    }
    fn y(&self) -> f64 {
        self.y
    }
    fn z(&self) -> f64 {
        self.z
    }
}

impl Point3D for (f64, f64, f64) {
    fn x(&self) -> f64 {
        self.0
    }
    fn y(&self) -> f64 {
        self.1
    }
    fn z(&self) -> f64 {
        self.2
    }
}

/// Utility: detect whether the file uses comma or tab as delimiter.
pub(crate) fn detect_delimiter<P: AsRef<Path>>(path: P) -> Result<u8> {
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
                    Err(e) => eprintln!("Skipping invalid record: {e:?}"),
                },
                Err(e) => eprintln!("Skipping invalid row: {e:?}"),
            }
        }

        Ok(points)
    }

    pub fn read_reference_point<P: AsRef<Path>>(path: P) -> Result<ContourPoint> {
        let delim = detect_delimiter(&path)?;
        // 1) Open the file, with context on failure
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

    /// Computes the Euclidean 3-D distance between two contour points.
    pub fn distance_to(&self, other: &ContourPoint) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Computes the 2-D (XY-plane) distance between two contour points.
    pub fn distance_2d_to(&self, other: &ContourPoint) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Returns a new point translated by `(dx, dy, dz)`.
    pub fn translate(&self, dx: f64, dy: f64, dz: f64) -> Self {
        ContourPoint {
            x: self.x + dx,
            y: self.y + dy,
            z: self.z + dz,
            ..*self
        }
    }

    /// Rotates a single point about a given center (cx, cy) by a specified angle (in radians).
    pub fn rotate_point(&self, angle: f64, center: (f64, f64)) -> ContourPoint {
        if angle == 0.0 {
            return *self;
        }
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

#[cfg(test)]
mod contour_point_tests {
    use super::*;
    use crate::intravascular::utils::test_utils::dummy_geometry;
    use std::f64::consts::PI;

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
}
