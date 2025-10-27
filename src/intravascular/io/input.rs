use super::geometry::ContourType;

use anyhow::{anyhow, Context, Result};
use csv::ReaderBuilder;
use nalgebra::Vector3;
use serde::Deserialize;

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// InputData stores raw data from typical AIVUS-CAA output
/// where dataframes have the following order
/// 
/// .. text::
///     
///     frame_idx, x, y, z
/// 
/// not automatically kept in sync with geometry.rs
pub struct InputData {
    lumen: Vec<ContourPoint>,
    eem: Option<Vec<ContourPoint>>,
    calcification: Option<Vec<ContourPoint>>,
    sidebranch: Option<Vec<ContourPoint>>,
    record: Option<Vec<Record>>,
    ref_point: ContourPoint,
    label: &str,
}

impl InputData {
    pub fn new(
        lumen: Vec<ContourPoint>,
        eem: Option<Vec<ContourPoint>>,
        calcification: Option<Vec<ContourPoint>>,
        sidebranch: Option<Vec<ContourPoint>>,
        record: Option<Vec<Record>>,
        ref_point: ContourPoint,
        label: &str,
    ) -> anyhow::Result<InputData> {
        let input = InputData {
            lumen,
            eem,
            calcification,
            sidebranch,
            record,
            ref_point,
            label,
        };
        Ok(input)
    }

    pub fn process_directory<P: AsRef<Path>>(
        path: P,
        names: HashMap<ContourType, &str>,
        diastole: bool,
    ) -> anyhow::Result<InputData> {
        let path = path.as_ref();

        let mut lumen: Option<Vec<ContourPoint>> = None;
        let mut eem: Option<Vec<ContourPoint>> = None;
        let mut calcification: Option<Vec<ContourPoint>> = None;
        let mut sidebranch: Option<Vec<ContourPoint>> = None;
        let mut record: Option<Vec<Record>> = None;
        let mut ref_point: Option<ContourPoint> = None;

        let phase = if diastole { "diastolic" } else { "systolic" };

        // iterate over provided names and attempt to build/read the corresponding files
        for (_ctype, raw_name) in names.iter() {
            let name = raw_name.trim().to_lowercase();
            match name.as_str() {
                "" | "lumen" => {
                    // main contour files: "<phase>_contours.csv" and "<phase>_reference_points.csv"
                    let contours_fname = format!("{}_contours.csv", phase);
                    let contours_path = path.join(&contours_fname);
                    if contours_path.exists() {
                        lumen = Some(
                            ContourPoint::read_contour_data(&contours_path)
                                .with_context(|| format!("reading {}", contours_path.display()))?,
                        );
                    } else {
                        return Err(anyhow::anyhow!(
                            "expected lumen contours file missing: {:?}",
                            contours_path
                        ));
                    }

                    let ref_fname = format!("{}_reference_points.csv", phase);
                    let ref_path = path.join(&ref_fname);
                    if ref_path.exists() {
                        ref_point = Some(
                            ContourPoint::read_reference_point(&ref_path)
                                .with_context(|| format!("reading {}", ref_path.display()))?,
                        );
                    } else {
                        return Err(anyhow::anyhow!(
                            "expected reference-point file missing: {:?}",
                            ref_path
                        ));
                    }
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
                        return Err(anyhow::anyhow!("expected file missing: {:?}", p));
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
                        return Err(anyhow::anyhow!("expected file missing: {:?}", p));
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
                        return Err(anyhow::anyhow!("expected file missing: {:?}", p));
                    }
                }

                "records" | "record" | "phases" => {
                    let fname = "combined_sorted_manual.csv";
                    let p = path.join(fname);
                    if p.exists() {
                        record = Some(read_records(&p).with_context(|| format!("reading {}", p.display()))?);
                    } else {
                        return Err(anyhow::anyhow!("expected records file missing: {:?}", p));
                    }
                }

                other => {
                    eprintln!("process_directory: unknown mapping name '{}', skipping", other);
                }
            }
        }

        // Also attempt to read records.csv if present even if not requested explicitly
        if record.is_none() {
            let maybe_records = path.join("records.csv");
            if maybe_records.exists() {
                record = Some(read_records(&maybe_records).with_context(|| {
                    format!("reading optional records file {}", maybe_records.display())
                })?);
            }
        }

        // Validate required fields
        let lumen = lumen.ok_or_else(|| anyhow::anyhow!("lumen contours not found in directory"))?;
        let ref_point =
            ref_point.ok_or_else(|| anyhow::anyhow!("reference point not found in directory"))?;

        let input = InputData {
            lumen,
            eem,
            calcification,
            sidebranch,
            record,
            ref_point,
            label: phase,
        };

        Ok(input)
    }

    fn quick_check_integrity(&self) {
        todo!()
    }
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
