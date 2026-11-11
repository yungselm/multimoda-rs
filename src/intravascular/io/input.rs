use crate::types::native::annotation::{Annotation, AnnotationKind, AnnotationRole};
use crate::types::native::centerline::Centerline;
use crate::types::native::contour::ContourType;
use crate::types::native::contour_point::ContourPoint;
use crate::types::native::record::Record;
use crate::types::native::Point3D;
use anyhow::{anyhow, Context, Result};
use csv::ReaderBuilder;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Once;

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
    pub record: Option<Vec<Record>>,
    pub ref_point: ContourPoint,
    pub diastole: bool,
    pub label: String,
    /// `(original frame, annotation)` pairs, frame numbered like `lumen`.
    pub annotations: Vec<(u32, Annotation)>,
}

impl InputData {
    pub fn new(
        lumen: Vec<ContourPoint>,
        eem: Option<Vec<ContourPoint>>,
        record: Option<Vec<Record>>,
        ref_point: ContourPoint,
        diastole: bool,
        label: String,
        annotations: Vec<(u32, Annotation)>,
    ) -> anyhow::Result<InputData> {
        Ok(InputData {
            lumen,
            eem,
            record,
            ref_point,
            diastole,
            label,
            annotations,
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

        for raw_name in names.values() {
            let name = raw_name.trim().to_lowercase();
            match name.as_str() {
                "" | "lumen" => {}

                // deprecated, read below as annotations when there is no annotations file
                "branch" | "sidebranch" | "calcium" | "calcification" => {}

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

        // The deprecated side-branch and calcium files are only a fallback. A directory with an
        // annotations file already holds those regions there, reading both would double them.
        let annotations_path = path.join(format!("{phase}_annotations.csv"));
        let annotations = if annotations_path.exists() {
            read_annotations(&annotations_path)
                .with_context(|| format!("reading {}", annotations_path.display()))?
        } else {
            let mut legacy = Vec::new();
            for (prefixes, kind) in [
                (["branch", "sidebranch"], AnnotationKind::Sidebranch),
                (["calcium", "calcification"], AnnotationKind::Calcification),
            ] {
                legacy.extend(read_legacy_region_file(path, &prefixes, phase, kind)?);
            }
            legacy
        };

        Ok(InputData {
            lumen,
            eem,
            record,
            ref_point,
            diastole,
            label: label_string,
            annotations,
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

/// Region annotations of `kind` from the first existing deprecated
/// `{prefix}_{phase}_contours.csv` file, empty if none exists.
fn read_legacy_region_file(
    dir: &Path,
    prefixes: &[&str],
    phase: &str,
    kind: AnnotationKind,
) -> anyhow::Result<Vec<(u32, Annotation)>> {
    static DEPRECATION: Once = Once::new();
    let Some(p) = prefixes
        .iter()
        .map(|prefix| dir.join(format!("{prefix}_{phase}_contours.csv")))
        .find(|p| p.exists())
    else {
        return Ok(Vec::new());
    };
    DEPRECATION.call_once(|| {
        eprintln!(
            "DeprecationWarning: side-branch and calcium contour files ({p:?}) are read as \
             region annotations, write {phase}_annotations.csv instead"
        )
    });
    let points = read_contour_data(&p).with_context(|| format!("reading {}", p.display()))?;
    Ok(region_annotations_from_points(points, kind))
}

/// One `Region` annotation (group 0) per frame from flat contour points, frames in first-seen order.
pub fn region_annotations_from_points(
    points: Vec<ContourPoint>,
    kind: AnnotationKind,
) -> Vec<(u32, Annotation)> {
    let mut out: Vec<(u32, Annotation)> = Vec::new();
    for p in points {
        let idx = match out.iter().position(|(f, _)| *f == p.frame_index) {
            Some(i) => i,
            None => {
                let a = Annotation::new(kind.clone(), AnnotationRole::Region, 0, Vec::new());
                out.push((p.frame_index, a));
                out.len() - 1
            }
        };
        let pts = &mut out[idx].1.points;
        pts.push(ContourPoint {
            point_index: pts.len() as u32,
            ..p
        });
    }
    out
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

/// Reads `frame, kind, role, group, x, y, z` rows (a leading header row is skipped).
///
/// Consecutive rows with the same `(frame, kind, role, group)` form one annotation in file order.
/// Unknown kinds become `AnnotationKind::Custom`, unknown roles are an error.
pub fn read_annotations<P: AsRef<Path>>(path: P) -> anyhow::Result<Vec<(u32, Annotation)>> {
    let delim = detect_delimiter(&path)?;
    let file = File::open(&path)?;
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .delimiter(delim)
        .flexible(true)
        .from_reader(file);

    let mut out: Vec<(u32, Annotation)> = Vec::new();
    for (row, result) in rdr.records().enumerate() {
        let record = match result {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Skipping invalid annotation row: {e:?}");
                continue;
            }
        };
        let parsed = parse_annotation_row(&record);
        let (frame, kind, role, group, x, y, z) = match parsed {
            Some(v) => v,
            None if row == 0 => continue, // header
            None => {
                eprintln!("Skipping invalid annotation row {}: {record:?}", row + 1);
                continue;
            }
        };
        let role = AnnotationRole::from_name(&role).ok_or_else(|| {
            anyhow!(
                "unknown annotation role '{role}' in row {}, expected one of {:?}",
                row + 1,
                AnnotationRole::ALL.map(|r| r.name())
            )
        })?;
        let kind = AnnotationKind::from_name(&kind);

        let same = matches!(out.last(), Some((f, a))
            if *f == frame && a.kind == kind && a.role == role && a.group == group);
        if !same {
            out.push((frame, Annotation::new(kind, role, group, Vec::new())));
        }
        let annotation = &mut out.last_mut().expect("pushed above").1;
        annotation.points.push(ContourPoint {
            frame_index: frame,
            point_index: annotation.points.len() as u32,
            x,
            y,
            z,
            aortic: false,
        });
    }
    Ok(out)
}

type AnnotationRow = (u32, String, String, u32, f64, f64, f64);

fn parse_annotation_row(record: &csv::StringRecord) -> Option<AnnotationRow> {
    if record.len() < 7 {
        return None;
    }
    let field = |i: usize| record.get(i).map(str::trim);
    let frame = field(0)?.parse::<u32>().ok()?;
    let kind = field(1)?.to_string();
    let role = field(2)?.to_string();
    let group = field(3)?.parse::<u32>().ok()?;
    let x = field(4)?.parse::<f64>().ok()?;
    let y = field(5)?.parse::<f64>().ok()?;
    let z = field(6)?.parse::<f64>().ok()?;
    Some((frame, kind, role, group, x, y, z))
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

/// Name of the per-point radius array VMTK writes into centerline VTP files.
pub(crate) const VTP_RADIUS_ARRAY: &str = "MaximumInscribedSphereRadius";

/// Reads `path` as text, rejecting binary VTP files and binary- or appended-encoded
/// DataArrays: only the ASCII format is supported.
fn read_ascii_vtp(path: &Path) -> anyhow::Result<String> {
    const BINARY_PROBE_BYTES: usize = 512;

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
    Ok(xml)
}

/// The first `<tag ...>...</tag>` element in `xml`, tags included.
pub(crate) fn extract_section<'a>(xml: &'a str, tag: &str) -> anyhow::Result<&'a str> {
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

/// Text content of the DataArray called `name` inside `section`.
pub(crate) fn dataarray_text<'a>(section: &'a str, name: &str) -> anyhow::Result<&'a str> {
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

/// Whitespace-separated numbers of a DataArray's text.
pub(crate) fn parse_nums<T>(text: &str) -> anyhow::Result<Vec<T>>
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

/// Splits a VTK cell array into its cells: cell `i` is `connectivity[offsets[i - 1]..offsets[i]]`
/// (starting at 0). Fails if there are no cells or the offsets do not cover `connectivity`.
pub(crate) fn split_cells<'a>(
    connectivity: &'a [usize],
    offsets: &[usize],
) -> anyhow::Result<Vec<&'a [usize]>> {
    let Some(&last_off) = offsets.last() else {
        anyhow::bail!("VTP: Lines section is empty (no branches)");
    };
    if last_off != connectivity.len() {
        anyhow::bail!(
            "VTP: last offset ({}) != connectivity length ({})",
            last_off,
            connectivity.len()
        );
    }
    std::iter::once(0)
        .chain(offsets.iter().copied())
        .zip(offsets.iter().copied())
        .map(|(start, end)| {
            connectivity
                .get(start..end)
                .ok_or_else(|| anyhow::anyhow!("VTP: offsets not ascending ({start} > {end})"))
        })
        .collect()
}

pub fn read_centerline_vtp<P: AsRef<Path>>(path: P) -> anyhow::Result<Centerline> {
    use crate::types::native::centerline_point::CenterlinePoint;
    use nalgebra::Vector3;

    const MIN_TANGENT_NORM: f64 = 1e-12;

    let xml = read_ascii_vtp(path.as_ref())?;

    let pts_raw: Vec<f64> =
        parse_nums(dataarray_text(extract_section(&xml, "Points")?, "Points")?)?;
    if !pts_raw.len().is_multiple_of(3) {
        anyhow::bail!(
            "VTP: Points array length {} not divisible by 3",
            pts_raw.len()
        );
    }
    let coords: Vec<_> = pts_raw
        .as_chunks::<3>()
        .0
        .iter()
        .map(|&[x, y, z]| (x, y, z))
        .collect();
    let n_pts = coords.len();

    let radii: Vec<f64> = extract_section(&xml, "PointData")
        .ok()
        .and_then(|s| dataarray_text(s, VTP_RADIUS_ARRAY).ok())
        .and_then(|t| parse_nums(t).ok())
        .filter(|r| r.len() == n_pts)
        .unwrap_or_else(|| vec![0.0; n_pts]);

    let lines_sec = extract_section(&xml, "Lines")?;
    let connectivity: Vec<usize> = parse_nums(dataarray_text(lines_sec, "connectivity")?)?;
    let offsets: Vec<usize> = parse_nums(dataarray_text(lines_sec, "offsets")?)?;
    let vtk_branches = split_cells(&connectivity, &offsets)?;

    let branch_arc_length = |branch: &[usize]| -> f64 {
        branch
            .windows(2)
            .map(|w| coords[w[0]].distance_to(&coords[w[1]]))
            .sum()
    };
    let branch_lengths: Vec<f64> = vtk_branches.iter().map(|b| branch_arc_length(b)).collect();

    let mut order: Vec<usize> = (0..vtk_branches.len()).collect();
    order.sort_unstable_by(|&a, &b| {
        branch_lengths[b]
            .partial_cmp(&branch_lengths[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

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
        assert!(input.record.is_none(), "record vector should be empty");
        assert!(input.ref_point.x > 0.0, "ref_point should not be empty");

        // deprecated branch_/calcium_ files become one Region annotation per frame
        let lumen_frames: std::collections::HashSet<u32> =
            input.lumen.iter().map(|p| p.frame_index).collect();
        for kind in [AnnotationKind::Calcification, AnnotationKind::Sidebranch] {
            let regions: Vec<_> = input
                .annotations
                .iter()
                .filter(|(_, a)| a.kind == kind)
                .collect();
            assert!(!regions.is_empty(), "{kind} regions expected");
            let mut frames = std::collections::HashSet::new();
            for (f, a) in &regions {
                assert!(frames.insert(*f), "one {kind} annotation per frame");
                assert!(lumen_frames.contains(f));
                assert_eq!((a.role, a.group), (AnnotationRole::Region, 0));
                assert!(a.points.iter().all(|p| p.frame_index == *f));
                assert!(a
                    .points
                    .iter()
                    .enumerate()
                    .all(|(i, p)| p.point_index == i as u32));
            }
        }
        assert_eq!(
            input.annotations.len(),
            input
                .annotations
                .iter()
                .filter(|(_, a)| a.role == AnnotationRole::Region)
                .count()
        );

        Ok(())
    }

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("multimodars_{name}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_read_annotations_groups_consecutive_rows() -> anyhow::Result<()> {
        let dir = temp_dir("read_annotations");
        let p = dir.join("diastolic_annotations.csv");
        std::fs::write(
            &p,
            "frame\tkind\trole\tgroup\tx\ty\tz\n\
             3\tsidebranch\tregion\t0\t1.0\t2.0\t0.5\n\
             3\tsidebranch\tregion\t0\t1.5\t2.5\t0.5\n\
             3\tsidebranch\tregion\t1\t4.0\t4.0\t0.5\n\
             3\tsidebranch\tlandmark\t1\t4.0\t4.0\t0.5\n\
             3\tfibrous_cap\tregion\t0\t0.0\t0.0\t0.5\n\
             4\tsidebranch\taxis_tip\t0\t1.0\t1.0\t1.0\n\
             3\tsidebranch\tregion\t0\t9.0\t9.0\t0.5\n",
        )?;
        let annotations = read_annotations(&p)?;

        let summary: Vec<(u32, &str, AnnotationRole, u32, usize)> = annotations
            .iter()
            .map(|(f, a)| (*f, a.kind.name(), a.role, a.group, a.points.len()))
            .collect();
        use AnnotationRole::*;
        assert_eq!(
            summary,
            vec![
                (3, "sidebranch", Region, 0, 2),
                (3, "sidebranch", Region, 1, 1),
                (3, "sidebranch", Landmark, 1, 1),
                (3, "fibrous_cap", Region, 0, 1),
                (4, "sidebranch", AxisTip, 0, 1),
                (3, "sidebranch", Region, 0, 1),
            ]
        );
        assert_eq!(annotations[0].1.kind, AnnotationKind::Sidebranch);
        assert_eq!(
            annotations[3].1.kind,
            AnnotationKind::Custom("fibrous_cap".to_string())
        );

        std::fs::write(&p, "3,lipid,outline,0,1.0,2.0,0.5\n")?;
        assert!(
            read_annotations(&p).is_err(),
            "unknown role must be rejected"
        );
        std::fs::remove_dir_all(&dir).ok();
        let first = &annotations[0].1.points;
        assert_eq!((first[1].x, first[1].y, first[1].z), (1.5, 2.5, 0.5));
        assert_eq!((first[1].frame_index, first[1].point_index), (3, 1));
        Ok(())
    }

    #[test]
    fn test_process_directory_reads_annotations_file() -> anyhow::Result<()> {
        let src = Path::new("./data/fixtures/ivus_rest");
        let dir = temp_dir("process_annotations");
        for f in ["diastolic_contours.csv", "diastolic_reference_points.csv"] {
            std::fs::copy(src.join(f), dir.join(f))?;
        }
        std::fs::write(
            dir.join("diastolic_annotations.csv"),
            "385,eem,point,0,3.0,3.0,24.5\n385,eem,point,1,4.0,4.0,24.5\n",
        )?;
        let input = InputData::process_directory(&dir, HashMap::new(), true, "")?;
        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(input.annotations.len(), 2);
        assert!(input.annotations.iter().all(|(f, a)| *f == 385
            && a.kind == AnnotationKind::Custom("eem".to_string())
            && a.role == AnnotationRole::Point
            && a.points.len() == 1));
        Ok(())
    }

    #[test]
    fn test_legacy_region_files_are_ignored_next_to_an_annotations_file() -> anyhow::Result<()> {
        let src = Path::new("./data/fixtures/ivus_rest");
        let dir = temp_dir("legacy_next_to_annotations");
        for f in ["diastolic_contours.csv", "diastolic_reference_points.csv"] {
            std::fs::copy(src.join(f), dir.join(f))?;
        }
        // HolOrama writes both, the annotations file already holds the branch region
        std::fs::write(
            dir.join("diastolic_annotations.csv"),
            "385	sidebranch	region	1	3.0	3.0	24.5
385	sidebranch	region	1	4.0	4.0	24.5
",
        )?;
        std::fs::write(
            dir.join("branch_diastolic_contours.csv"),
            "385	3.0	3.0	24.5
385	4.0	4.0	24.5
",
        )?;
        std::fs::write(
            dir.join("calcium_diastolic_contours.csv"),
            "385	5.0	5.0	24.5
385	6.0	6.0	24.5
",
        )?;
        let input = InputData::process_directory(&dir, HashMap::new(), true, "")?;
        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(input.annotations.len(), 1);
        let (frame, annotation) = &input.annotations[0];
        assert_eq!(*frame, 385);
        assert_eq!(
            (&annotation.kind, annotation.role, annotation.group),
            (&AnnotationKind::Sidebranch, AnnotationRole::Region, 1)
        );
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
        assert_eq!(
            b0_len, 763,
            "branch 0 should be the longest VTK line by arc length"
        );

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

    /// Branch 0 must be picked by physical arc length, not by raw point
    /// count. Line A is geometrically long (40mm) but sparse (5 points);
    /// line B is geometrically short (1.9mm) but densely sampled (20
    /// points). A point-count-based sort would wrongly elect B as branch 0.
    #[test]
    fn test_read_centerline_vtp_picks_longest_by_arc_length_not_point_count() -> anyhow::Result<()>
    {
        let line_a_pts: Vec<(f64, f64, f64)> =
            (0..5).map(|i| (i as f64 * 10.0, 0.0, 0.0)).collect();
        let line_b_pts: Vec<(f64, f64, f64)> =
            (0..20).map(|i| (0.0, i as f64 * 0.1, 0.0)).collect();

        let all_pts: Vec<(f64, f64, f64)> = line_a_pts
            .iter()
            .chain(line_b_pts.iter())
            .copied()
            .collect();
        let n_pts = all_pts.len();

        let points_text = all_pts
            .iter()
            .map(|(x, y, z)| format!("{x} {y} {z}"))
            .collect::<Vec<_>>()
            .join(" ");
        let radii_text = vec!["1.0"; n_pts].join(" ");

        let connectivity: Vec<usize> = (0..line_a_pts.len())
            .chain(line_a_pts.len()..n_pts)
            .collect();
        let offsets = format!("{} {}", line_a_pts.len(), n_pts);

        let xml = format!(
            r#"<?xml version="1.0"?>
<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian" header_type="UInt32">
  <PolyData>
    <Piece NumberOfPoints="{n_pts}" NumberOfVerts="0" NumberOfLines="2" NumberOfStrips="0" NumberOfPolys="0">
      <PointData>
        <DataArray type="Float64" Name="MaximumInscribedSphereRadius" format="ascii">
          {radii_text}
        </DataArray>
      </PointData>
      <Points>
        <DataArray type="Float64" Name="Points" NumberOfComponents="3" format="ascii">
          {points_text}
        </DataArray>
      </Points>
      <Lines>
        <DataArray type="Int64" Name="connectivity" format="ascii">
          {connectivity}
        </DataArray>
        <DataArray type="Int64" Name="offsets" format="ascii">
          {offsets}
        </DataArray>
      </Lines>
    </Piece>
  </PolyData>
</VTKFile>
"#,
            connectivity = connectivity
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(" "),
        );

        let path = std::env::temp_dir().join("multimodars_test_arc_length_branch0.vtp");
        std::fs::write(&path, xml)?;
        let cl = read_centerline_vtp(&path);
        let _ = std::fs::remove_file(&path);
        let cl = cl?;

        assert_eq!(cl.branch_start_indices.len(), 2, "expected 2 branches");
        let b0_len = cl.branch_start_indices[1] - cl.branch_start_indices[0];
        assert_eq!(
            b0_len,
            line_a_pts.len(),
            "the geometrically longer (but sparser) line A must be branch 0"
        );

        Ok(())
    }
}
