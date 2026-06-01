use super::contour::ContourType;
use super::contour_point::ContourPoint;
use super::record::{read_records, Record};
use anyhow::Context;
use std::collections::HashMap;
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
        label: &str,
    ) -> anyhow::Result<InputData> {
        let path = path.as_ref();
        let label_string = label.to_string();

        let mut eem: Option<Vec<ContourPoint>> = None;
        let mut calcification: Option<Vec<ContourPoint>> = None;
        let mut sidebranch: Option<Vec<ContourPoint>> = None;
        let mut record: Option<Vec<Record>> = None;

        let phase = if diastole { "diastolic" } else { "systolic" };

        // Read required files - these will crash if missing
        let contours_fname = format!("{phase}_contours.csv");
        let contours_path = path.join(&contours_fname);
        let lumen = if contours_path.exists() {
            ContourPoint::read_contour_data(&contours_path)
                .with_context(|| format!("reading {}", contours_path.display()))?
        } else {
            return Err(anyhow::anyhow!(
                "required contours file missing: {contours_path:?}"
            ));
        };

        let ref_fname = format!("{phase}_reference_points.csv");
        let ref_path = path.join(&ref_fname);
        let ref_point = if ref_path.exists() {
            ContourPoint::read_reference_point(&ref_path)
                .with_context(|| format!("reading {}", ref_path.display()))?
        } else {
            return Err(anyhow::anyhow!(
                "required reference-point file missing: {ref_path:?}"
            ));
        };

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
                        eprintln!("sidebranch file not found, skipping: {p:?}");
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
                        eprintln!("calcification file not found, skipping: {p:?}");
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
                        eprintln!("eem file not found, skipping: {p:?}");
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
                        eprintln!("records file not found, skipping: {p:?}");
                    }
                }

                other => {
                    eprintln!("process_directory: unknown mapping name '{other}', skipping");
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
            label: label_string,
        };

        Ok(input)
    }
}

#[cfg(test)]
mod input_data_tests {
    use super::*;
    use crate::types::native::contour::ContourType;
    use std::collections::HashMap;
    use std::path::Path;

    #[test]
    fn test_process_directory_runs_with_example_data() -> anyhow::Result<()> {
        let mut names: HashMap<ContourType, &str> = HashMap::new();
        names.insert(ContourType::Lumen, "lumen");
        names.insert(ContourType::Eem, "eem");
        names.insert(ContourType::Calcification, "calcification");
        names.insert(ContourType::Sidebranch, "sidebranch");

        let data_path = Path::new("./data/fixtures/idealized_geometry");
        let input = InputData::process_directory(data_path, names, true, "")?;

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
}
