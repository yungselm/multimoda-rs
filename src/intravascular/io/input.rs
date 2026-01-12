use super::geometry::ContourType;

use anyhow::{anyhow, Context, Result};
use csv::ReaderBuilder;
use nalgebra::Vector3;
use serde::Deserialize;

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

const RECORD_FILE_NAME: &str = "combined_sorted_manual.csv";

/// InputData stores raw data from typical AIVUS-CAA output
/// where dataframes have the following order
///
/// .. text::
///     
///     frame_idx, x, y, z
///
/// not automatically kept in sync with geometry.rs
#[derive(Debug, Clone)]
pub struct InputData {
    pub lumen: Vec<ContourPoint>,
    pub eem: Option<Vec<ContourPoint>>,
    pub calcification: Option<Vec<ContourPoint>>,
    pub sidebranch: Option<Vec<ContourPoint>>,
    pub record: Option<Vec<Record>>,
    pub ref_point: ContourPoint,
    pub diastole: bool,
    pub label: String,
}

impl InputData {
    pub fn new(
        lumen: Vec<ContourPoint>,
        eem: Option<Vec<ContourPoint>>,
        calcification: Option<Vec<ContourPoint>>,
        sidebranch: Option<Vec<ContourPoint>>,
        record: Option<Vec<Record>>,
        ref_point: ContourPoint,
        diastole: bool,
        label: String,
    ) -> anyhow::Result<InputData> {
        let input = InputData {
            lumen,
            eem,
            calcification,
            sidebranch,
            record,
            ref_point,
            diastole,
            label,
        };
        Ok(input)
    }

    // #[allow(unused_assignments)]
    pub fn process_directory<P: AsRef<Path>>(
        path: P,
        names: HashMap<ContourType, &str>,
        diastole: bool,
    ) -> anyhow::Result<InputData> {
        let path = path.as_ref();

        let lumen: Vec<ContourPoint>;
        let ref_point: ContourPoint;

        let mut eem: Option<Vec<ContourPoint>> = None;
        let mut calcification: Option<Vec<ContourPoint>> = None;
        let mut sidebranch: Option<Vec<ContourPoint>> = None;
        let mut record: Option<Vec<Record>> = None;

        let phase = if diastole { "diastolic" } else { "systolic" };

        // Read required files - these will crash if missing
        let contours_fname = format!("{}_contours.csv", phase);
        let contours_path = path.join(&contours_fname);
        if contours_path.exists() {
            lumen = ContourPoint::read_contour_data(&contours_path)
                .with_context(|| format!("reading {}", contours_path.display()))?;
        } else {
            return Err(anyhow::anyhow!(
                "required contours file missing: {:?}",
                contours_path
            ));
        }

        let ref_fname = format!("{}_reference_points.csv", phase);
        let ref_path = path.join(&ref_fname);
        if ref_path.exists() {
            ref_point = ContourPoint::read_reference_point(&ref_path)
                .with_context(|| format!("reading {}", ref_path.display()))?;
        } else {
            return Err(anyhow::anyhow!(
                "required reference-point file missing: {:?}",
                ref_path
            ));
        }

        for (_ctype, raw_name) in names.iter() {
            let name = raw_name.trim().to_lowercase();
            match name.as_str() {
                "" | "lumen" => {
                    // Already handled above, skip
                }

                "branch" | "sidebranch" => {
                    let fname = format!("{}_{}_contours.csv", "branch", phase);
                    let p = path.join(&fname);
                    if p.exists() {
                        sidebranch = Some(
                            ContourPoint::read_contour_data(&p)
                                .with_context(|| format!("reading {}", p.display()))?,
                        );
                    } else {
                        eprintln!("sidebranch file not found, skipping: {:?}", p);
                    }
                }

                "calcium" | "calcification" => {
                    let fname = format!("{}_{}_contours.csv", "calcium", phase);
                    let p = path.join(&fname);
                    if p.exists() {
                        calcification = Some(
                            ContourPoint::read_contour_data(&p)
                                .with_context(|| format!("reading {}", p.display()))?,
                        );
                    } else {
                        eprintln!("calcification file not found, skipping: {:?}", p);
                    }
                }

                "eem" | "e_e_m" => {
                    let fname = format!("{}_{}_contours.csv", "eem", phase);
                    let p = path.join(&fname);
                    if p.exists() {
                        eem = Some(
                            ContourPoint::read_contour_data(&p)
                                .with_context(|| format!("reading {}", p.display()))?,
                        );
                    } else {
                        eprintln!("eem file not found, skipping: {:?}", p);
                    }
                }

                "records" | "record" | "phases" => {
                    let fname = "combined_sorted_manual.csv";
                    let p = path.join(fname);
                    if p.exists() {
                        record = Some(
                            read_records(&p).with_context(|| format!("reading {}", p.display()))?,
                        );
                    } else {
                        eprintln!("records file not found, skipping: {:?}", p);
                    }
                }

                other => {
                    eprintln!(
                        "process_directory: unknown mapping name '{}', skipping",
                        other
                    );
                }
            }
        }

        // Also attempt to read records.csv if present even if not requested explicitly
        if record.is_none() {
            let maybe_records = path.join(RECORD_FILE_NAME);
            if maybe_records.exists() {
                record = Some(read_records(&maybe_records).with_context(|| {
                    format!("reading optional records file {}", maybe_records.display())
                })?);
            }
        }

        let input = InputData {
            lumen,
            eem,
            calcification,
            sidebranch,
            record,
            ref_point,
            diastole,
            label: phase.to_string(),
        };

        Ok(input)
    }

    // fn quick_check_integrity(&self) {
    //     todo!()
    // }
}

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

    /// Computes the Euclidean distance between two contour points.
    pub fn distance_to(&self, other: &ContourPoint) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Rotates a single point about a given center (cx, cy) by a specified angle (in radians).
    pub fn rotate_point(&self, angle: f64, center: (f64, f64)) -> ContourPoint {
        if angle == 0.0 {
            return self.clone();
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

    /// Finds the index of the centerline point closest to the reference point
    pub fn find_reference_cl_point_idx(&self, reference_point: &(f64, f64, f64)) -> usize {
        // Helper function to calculate squared distance (avoids sqrt for performance)
        fn distance_sq(contour_point: &ContourPoint, reference: &(f64, f64, f64)) -> f64 {
            let dx = contour_point.x - reference.0;
            let dy = contour_point.y - reference.1;
            let dz = contour_point.z - reference.2;
            dx * dx + dy * dy + dz * dz
        }

        self.points
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                distance_sq(&a.contour_point, reference_point)
                    .partial_cmp(&distance_sq(&b.contour_point, reference_point))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx)
            .unwrap()
    }
}

#[cfg(test)]
mod input_tests {
    use super::*;

    #[test]
    fn test_process_directory_runs_with_example_data() -> anyhow::Result<()> {
        use crate::intravascular::io::geometry::ContourType;

        let mut names: HashMap<ContourType, &str> = HashMap::new();
        names.insert(ContourType::Lumen, "lumen");
        names.insert(ContourType::Eem, "eem");
        names.insert(ContourType::Calcification, "calcification");
        names.insert(ContourType::Sidebranch, "sidebranch");

        let data_path = Path::new("./data/fixtures/idealized_geometry");
        let input = InputData::process_directory(data_path, names, true)?;

        assert!(
            !input.lumen.is_empty(),
            "lumen contour vector should not be empty"
        );
        assert!(
            input.eem.is_some(),
            "eem contour vector should not be empty"
        );
        assert!(
            input.calcification.is_some(),
            "calcification contour vector should not be empty"
        );
        assert!(input.record.is_none(), "record vector should be empty");
        assert!(input.ref_point.x > 0.0, "ref_point should not be empty");

        Ok(())
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
        use std::f64::consts::PI;

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

    #[test]
    fn test_cl_find_ref_pt() {
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
        let ref_pt = (0.0, 0.0, 0.0);
        let ref_id = centerline.find_reference_cl_point_idx(&ref_pt);
        assert_eq!(centerline.points[0], centerline.points[ref_id]);
    }
}
