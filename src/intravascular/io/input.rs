use crate::types::native::centerline::Centerline;
use crate::types::native::contour::ContourType;
use crate::types::native::contour_point::ContourPoint;
use crate::types::native::record::Record;
use anyhow::{anyhow, Context, Result};
use csv::ReaderBuilder;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

const RECORD_FILE_NAME: &str = "combined_sorted_manual.csv"; // legacy AIVUS
const RECORD_FILE_NAME_ALT: &str = "diastolic_systolic_records.csv"; // holOrama

/// Resolves the records file in `dir`, preferring `RECORD_FILE_NAME` and
/// falling back to `RECORD_FILE_NAME_ALT` if the preferred name isn't present.
fn resolve_record_path(dir: &Path) -> PathBuf {
    let primary = dir.join(RECORD_FILE_NAME);
    if primary.exists() {
        primary
    } else {
        dir.join(RECORD_FILE_NAME_ALT)
    }
}

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
                    sidebranch = read_optional_contour_file(path, "branch", phase, "sidebranch")?;
                }

                "calcium" | "calcification" => {
                    calcification =
                        read_optional_contour_file(path, "calcium", phase, "calcification")?;
                }

                "eem" | "e_e_m" => {
                    eem = read_optional_contour_file(path, "eem", phase, "eem")?;
                }

                "records" | "record" | "phases" => {
                    let p = resolve_record_path(path);
                    record = read_optional_records(&p)?;
                    if record.is_none() {
                        eprintln!("records file not found, skipping: {p:?}");
                    }
                }

                other => {
                    eprintln!("process_directory: unknown mapping name '{other}', skipping");
                }
            }
        }

        if record.is_none() {
            record = read_optional_records(&resolve_record_path(path))?;
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

/// Read an optional `{prefix}_{phase}_contours.csv` file, warning and returning
/// `None` (rather than erroring) if it is missing.
fn read_optional_contour_file(
    dir: &Path,
    prefix: &str,
    phase: &str,
    label: &str,
) -> anyhow::Result<Option<Vec<ContourPoint>>> {
    let p = dir.join(format!("{prefix}_{phase}_contours.csv"));
    if !p.exists() {
        eprintln!("{label} file not found, skipping: {p:?}");
        return Ok(None);
    }
    let points = read_contour_data(&p).with_context(|| format!("reading {}", p.display()))?;
    Ok(Some(points))
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

fn read_optional_records(path: &Path) -> anyhow::Result<Option<Vec<Record>>> {
    if !path.exists() {
        return Ok(None);
    }
    let records = read_records(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(Some(records))
}

pub fn read_centerline_vtp<P: AsRef<Path>>(path: P) -> anyhow::Result<Centerline> {
    use crate::types::native::centerline_point::CenterlinePoint;
    use nalgebra::Vector3;

    const BINARY_PROBE_BYTES: usize = 512;
    const MIN_TANGENT_NORM: f64 = 1e-12;

    let path = path.as_ref();

    let raw = std::fs::read(path).with_context(|| format!("cannot open {:?}", path))?;

    if raw
        .iter()
        .take(BINARY_PROBE_BYTES)
        .any(|&b| b < 0x09 || (b > 0x0d && b < 0x20))
    {
        anyhow::bail!(
            "{:?} appears to be a binary VTP file; only ASCII-format VTP is supported. \
             Re-export from your software with 'ASCII' data mode.",
            path
        );
    }

    let xml = String::from_utf8(raw).with_context(|| format!("{:?}: not valid UTF-8", path))?;

    for fmt in ["format=\"binary\"", "format=\"appended\""] {
        if xml.contains(fmt) {
            anyhow::bail!(
                "{:?}: binary-encoded DataArrays detected ({}); only ASCII format is \
                 supported. Re-export with 'ASCII' data mode.",
                path,
                fmt
            );
        }
    }

    fn extract_section<'a>(xml: &'a str, tag: &str) -> anyhow::Result<&'a str> {
        let open = format!("<{}", tag);
        let close = format!("</{}>", tag);
        let start = xml
            .find(&open)
            .ok_or_else(|| anyhow::anyhow!("VTP: <{}> section not found", tag))?;
        let rest = &xml[start..];
        let end_rel = rest
            .find(&close)
            .ok_or_else(|| anyhow::anyhow!("VTP: </{}> not found", tag))?;
        Ok(&rest[..end_rel + close.len()])
    }

    fn dataarray_text<'a>(section: &'a str, name: &str) -> anyhow::Result<&'a str> {
        let needle = format!("Name=\"{}\"", name);
        let pos = section
            .find(&needle)
            .ok_or_else(|| anyhow::anyhow!("VTP: DataArray Name=\"{}\" not found", name))?;
        let before = &section[..pos];
        let da_start = before
            .rfind("<DataArray")
            .ok_or_else(|| anyhow::anyhow!("VTP: no <DataArray before Name=\"{}\"", name))?;
        let rest = &section[da_start..];
        let tag_end = rest
            .find('>')
            .ok_or_else(|| anyhow::anyhow!("VTP: unclosed <DataArray Name=\"{}\">", name))?;
        let inner = &rest[tag_end + 1..];
        let close_pos = inner
            .find("</DataArray>")
            .ok_or_else(|| anyhow::anyhow!("VTP: no </DataArray> for Name=\"{}\"", name))?;
        // <InformationKey> nodes can appear inside the Points array in some exporters.
        let text = inner[..close_pos].trim();
        let text_end = text.find('<').unwrap_or(text.len());
        Ok(text[..text_end].trim())
    }

    fn parse_nums<T>(text: &str) -> anyhow::Result<Vec<T>>
    where
        T: std::str::FromStr,
        T::Err: std::error::Error + Send + Sync + 'static,
    {
        text.split_ascii_whitespace()
            .map(|s| {
                s.parse::<T>()
                    .with_context(|| format!("VTP: bad number '{s}'"))
            })
            .collect()
    }

    let pts_raw: Vec<f64> =
        parse_nums(dataarray_text(extract_section(&xml, "Points")?, "Points")?)?;
    if !pts_raw.len().is_multiple_of(3) {
        anyhow::bail!(
            "VTP: Points array length {} not divisible by 3",
            pts_raw.len()
        );
    }
    let coords: Vec<_> = pts_raw
        .chunks_exact(3)
        .map(|c| (c[0], c[1], c[2]))
        .collect();
    let n_pts = coords.len();

    let radii: Vec<f64> = extract_section(&xml, "PointData")
        .ok()
        .and_then(|s| dataarray_text(s, "MaximumInscribedSphereRadius").ok())
        .and_then(|t| parse_nums(t).ok())
        .filter(|r| r.len() == n_pts)
        .unwrap_or_else(|| vec![0.0; n_pts]);

    let lines_sec = extract_section(&xml, "Lines")?;
    let connectivity: Vec<usize> = parse_nums(dataarray_text(lines_sec, "connectivity")?)?;
    let offsets: Vec<usize> = parse_nums(dataarray_text(lines_sec, "offsets")?)?;

    if offsets.is_empty() {
        anyhow::bail!("VTP: Lines section is empty (no branches)");
    }
    let last_off = *offsets.last().unwrap();
    if last_off != connectivity.len() {
        anyhow::bail!(
            "VTP: last offset ({}) != connectivity length ({})",
            last_off,
            connectivity.len()
        );
    }

    let vtk_branches: Vec<&[usize]> = std::iter::once(0)
        .chain(offsets.iter().copied())
        .zip(offsets.iter().copied())
        .map(|(start, end)| &connectivity[start..end])
        .collect();

    let mut order: Vec<usize> = (0..vtk_branches.len()).collect();
    order.sort_unstable_by(|&a, &b| vtk_branches[b].len().cmp(&vtk_branches[a].len()));

    let mut cl_points: Vec<CenterlinePoint> = Vec::with_capacity(connectivity.len());
    let mut branch_start_indices: Vec<usize> = Vec::with_capacity(order.len());
    for (branch_id, &vtk_idx) in order.iter().enumerate() {
        let branch_id = branch_id as u32;
        branch_start_indices.push(cl_points.len());
        let branch = vtk_branches[vtk_idx];
        let branch_len = branch.len();

        for (local_i, &pt_idx) in branch.iter().enumerate() {
            if pt_idx >= n_pts {
                anyhow::bail!(
                    "VTP: connectivity index {} out of range ({} points)",
                    pt_idx,
                    n_pts
                );
            }
            let (x, y, z) = coords[pt_idx];
            let idx = cl_points.len() as u32;

            let tangent = if local_i + 1 < branch_len {
                let (nx, ny, nz) = coords[branch[local_i + 1]];
                let diff = Vector3::new(nx - x, ny - y, nz - z);
                let norm = diff.norm();
                if norm > MIN_TANGENT_NORM {
                    diff / norm
                } else {
                    Vector3::zeros()
                }
            } else if local_i > 0 {
                cl_points.last().unwrap().tangent
            } else {
                Vector3::zeros()
            };

            cl_points.push(CenterlinePoint {
                contour_point: ContourPoint {
                    frame_index: idx,
                    point_index: idx,
                    x,
                    y,
                    z,
                    aortic: false,
                },
                tangent,
                radius: radii[pt_idx],
                branch_id,
            });
        }
    }

    Ok(Centerline {
        points: cl_points,
        branch_start_indices,
    })
}

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

    #[test]
    fn test_read_centerline_vtp_rca() -> anyhow::Result<()> {
        let cl = read_centerline_vtp("examples/data/rca_cl.vtp")?;

        assert_eq!(cl.branch_start_indices.len(), 4, "expected 4 branches");
        assert_eq!(cl.points.len(), 2652, "expected 2652 total points");

        let b0_len = cl
            .branch_start_indices
            .get(1)
            .copied()
            .unwrap_or(cl.points.len())
            - cl.branch_start_indices[0];
        assert_eq!(b0_len, 763, "branch 0 should be the longest VTK line");

        // All branch_id values must match the branch they live in.
        for (i, &start) in cl.branch_start_indices.iter().enumerate() {
            let end = cl
                .branch_start_indices
                .get(i + 1)
                .copied()
                .unwrap_or(cl.points.len());
            for pt in &cl.points[start..end] {
                assert_eq!(pt.branch_id, i as u32);
            }
        }

        assert!(
            cl.points.iter().any(|p| p.radius > 0.0),
            "radii should be populated"
        );

        let b0_pts = &cl.points[cl.branch_start_indices[0]..cl.branch_start_indices[0] + b0_len];
        assert!(
            b0_pts.iter().any(|p| p.tangent.norm() > 0.5),
            "branch 0 tangents should be non-zero"
        );

        Ok(())
    }
}
