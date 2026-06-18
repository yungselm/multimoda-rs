use crate::types::native::contour::ContourType;
use anyhow::{anyhow, Context, Result};
use csv::ReaderBuilder;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

// Re-export types that downstream modules import from here.
pub use crate::types::native::centerline::Centerline;
pub use crate::types::native::centerline_point::CenterlinePoint;
pub use crate::types::native::contour_point::ContourPoint;
pub use crate::types::native::record::Record;

/// Raw intravascular imaging input for one cardiac phase, loaded from a measurement directory.
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
        Ok(InputData {
            lumen,
            eem,
            calcification,
            sidebranch,
            record,
            ref_point,
            diastole,
            label,
        })
    }

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

        let contours_fname = format!("{phase}_contours.csv");
        let contours_path = path.join(&contours_fname);
        let lumen = if contours_path.exists() {
            read_contour_data(&contours_path)
                .with_context(|| format!("reading {}", contours_path.display()))?
        } else {
            return Err(anyhow::anyhow!(
                "required contours file missing: {contours_path:?}"
            ));
        };

        let ref_fname = format!("{phase}_reference_points.csv");
        let ref_path = path.join(&ref_fname);
        let ref_point = if ref_path.exists() {
            read_reference_point(&ref_path)
                .with_context(|| format!("reading {}", ref_path.display()))?
        } else {
            return Err(anyhow::anyhow!(
                "required reference-point file missing: {ref_path:?}"
            ));
        };

        for (_ctype, raw_name) in names.iter() {
            let name = raw_name.trim().to_lowercase();
            match name.as_str() {
                "" | "lumen" => {}

                "branch" | "sidebranch" => {
                    let fname = format!("{}_{}_contours.csv", "branch", phase);
                    let p = path.join(&fname);
                    if p.exists() {
                        sidebranch = Some(
                            read_contour_data(&p)
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
                            read_contour_data(&p)
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
                            read_contour_data(&p)
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

        const RECORD_FILE_NAME: &str = "combined_sorted_manual.csv";
        if record.is_none() {
            let maybe_records = path.join(RECORD_FILE_NAME);
            if maybe_records.exists() {
                record = Some(read_records(&maybe_records).with_context(|| {
                    format!("reading optional records file {}", maybe_records.display())
                })?);
            }
        }

        Ok(InputData {
            lumen,
            eem,
            calcification,
            sidebranch,
            record,
            ref_point,
            diastole,
            label: label_string,
        })
    }
}

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

    let tabs = first_line.matches('\t').count();
    let commas = first_line.matches(',').count();

    if tabs > commas {
        Ok(b'\t')
    } else {
        Ok(b',')
    }
}

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
    let file = File::open(&path)
        .with_context(|| format!("failed to open reference-point file {:?}", path.as_ref()))?;

    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .delimiter(delim)
        .from_reader(file);

    let first = rdr.deserialize().next().ok_or_else(|| {
        anyhow!(
            "reference-point file {:?} was empty — this data is required",
            path.as_ref()
        )
    })?;

    let point: ContourPoint =
        first.with_context(|| "failed to deserialize first reference-point record")?;
    Ok(point)
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

// pub fn read_centerline_vtp<P: AsRef<Path>>(path: P) -> anyhow::Result<Centerline> {
//     let file = File::open(path)?;
//     let reader = BufReader::new(file);
//     let points = Vec::new();
//     let branch_start_indices = Vec::new();

//     for line in reader.deserialize() {

//     }
//     todo!()
// }

#[cfg(test)]
mod input_tests {
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
