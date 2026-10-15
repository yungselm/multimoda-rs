use anyhow::{anyhow, bail, Context};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::intravascular::io::input::VTP_RADIUS_ARRAY;
use crate::types::native::centerline::Centerline;
use crate::types::native::contour::{Contour, ContourType};
use crate::types::native::geometry::Geometry;
use rayon::prelude::*;

pub fn write_obj_mesh(
    contours: &[Contour],
    uv_coords: &[(f64, f64)],
    filename: &str,
    mtl_filename: &str,
    watertight: bool,
) -> anyhow::Result<()> {
    if let Some(parent) = Path::new(filename).parent() {
        std::fs::create_dir_all(parent)
            .context(format!("Could not create output directory: {parent:?}"))?;
    }

    let sorted_contours = contours.to_owned();

    if sorted_contours.len() < 2 {
        bail!("Need at least two contours to create a mesh.");
    }

    let points_per_contour = sorted_contours[0].points.len();
    for contour in &sorted_contours {
        if contour.points.len() != points_per_contour {
            bail!("All contours must have the same number of points.");
        }
    }

    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);
    let mut vertex_offsets = Vec::new();
    let mut current_offset = 1;

    // Write vertices
    for contour in &sorted_contours {
        vertex_offsets.push(current_offset);
        for point in &contour.points {
            writeln!(writer, "v {} {} {}", point.x, point.y, point.z)?;
            current_offset += 1;
        }
    }

    let total_vertices_before_centroids = current_offset - 1;

    if uv_coords.len() != total_vertices_before_centroids {
        return Err(anyhow!(
            "UV coordinates must match the number of vertices. Expected {}, got {}.",
            total_vertices_before_centroids,
            uv_coords.len()
        ));
    }

    // Write material reference
    writeln!(writer, "mtllib {mtl_filename}")?;
    writeln!(writer, "usemtl displacement_material")?;

    // Write UV coordinates for original vertices
    for (u, v) in uv_coords {
        writeln!(writer, "vt {u} {v}")?;
    }

    // Compute and write normals for original vertices
    for contour in &sorted_contours {
        let centroid = contour.centroid.unwrap_or((0.0, 0.0, 0.0));
        for point in &contour.points {
            let dx = point.x - centroid.0;
            let dy = point.y - centroid.1;
            let length = (dx * dx + dy * dy).sqrt();
            let (nx, ny, nz) = if length > 0.0 {
                (dx / length, dy / length, 0.0)
            } else {
                (0.0, 0.0, 0.0)
            };
            writeln!(writer, "vn {} {} {}", -nx, -ny, -nz)?;
        }
    }

    // Write faces for the main shell
    for c in 0..(sorted_contours.len() - 1) {
        let offset1 = vertex_offsets[c];
        let offset2 = vertex_offsets[c + 1];
        for j in 0..points_per_contour {
            let j_next = (j + 1) % points_per_contour;
            let v1 = offset1 + j;
            let v2 = offset1 + j_next;
            let v3 = offset2 + j;
            writeln!(writer, "f {v1}/{v1}/{v1} {v2}/{v2}/{v2} {v3}/{v3}/{v3}")?;

            let v1_t2 = offset2 + j;
            let v2_t2 = offset1 + j_next;
            let v3_t2 = offset2 + j_next;
            writeln!(
                writer,
                "f {v1_t2}/{v1_t2}/{v1_t2} {v2_t2}/{v2_t2}/{v2_t2} {v3_t2}/{v3_t2}/{v3_t2}"
            )?;
        }
    }

    if watertight {
        let proximal_centroid_index = current_offset;
        let first_centroid = sorted_contours[0].centroid.unwrap_or((0.0, 0.0, 0.0));
        writeln!(
            writer,
            "v {} {} {}",
            first_centroid.0, first_centroid.1, first_centroid.2
        )?;
        writeln!(writer, "vt 0.5 0.5")?;
        writeln!(writer, "vn 0.0 0.0 -1.0")?; // Pointing inward

        let distal_centroid_index = current_offset + 1;
        let last_contour = &sorted_contours[sorted_contours.len() - 1];
        let last_centroid = last_contour.centroid.unwrap_or((0.0, 0.0, 0.0));
        writeln!(
            writer,
            "v {} {} {}",
            last_centroid.0, last_centroid.1, last_centroid.2
        )?;
        writeln!(writer, "vt 0.5 0.5")?;
        writeln!(writer, "vn 0.0 0.0 1.0")?; // Pointing inward

        close_end(
            &mut writer,
            vertex_offsets[0],
            proximal_centroid_index,
            points_per_contour,
            false,
        )?;
        let last_contour_index = sorted_contours.len() - 1;
        close_end(
            &mut writer,
            vertex_offsets[last_contour_index],
            distal_centroid_index,
            points_per_contour,
            true,
        )?;
    }

    writer.flush()?;

    Ok(())
}

fn close_end(
    writer: &mut BufWriter<File>,
    vertex_offset: usize,
    centroid_vertex_index: usize,
    points_per_contour: usize,
    reverse_winding: bool,
) -> anyhow::Result<()> {
    for i in 0..points_per_contour {
        let next_i = (i + 1) % points_per_contour;

        let v1 = vertex_offset + i;
        let v2 = vertex_offset + next_i;
        let v3 = centroid_vertex_index;

        if reverse_winding {
            writeln!(writer, "f {v3}/{v3}/{v3} {v2}/{v2}/{v2} {v1}/{v1}/{v1}")?;
        } else {
            writeln!(writer, "f {v1}/{v1}/{v1} {v2}/{v2}/{v2} {v3}/{v3}/{v3}")?;
        }
    }
    Ok(())
}

pub fn write_obj_mesh_without_uv(
    contours: &[Contour],
    filename: &str,
    mtl_filename: &str,
    watertight: bool,
) -> anyhow::Result<()> {
    let empty_uv_coords = vec![(0.0, 0.0); contours.iter().map(|c| c.points.len()).sum()];
    write_obj_mesh(
        contours,
        &empty_uv_coords,
        filename,
        mtl_filename,
        watertight,
    )
    .map_err(|e| anyhow!("Failed to write OBJ mesh without UV: {e}"))
}

impl ContourType {
    // Get the contour data from a Geometry based on the enum variant
    pub fn get_contours(&self, geometry: &Geometry) -> Vec<Contour> {
        match self {
            ContourType::Lumen => geometry
                .frames
                .iter()
                .map(|frame| frame.lumen.clone())
                .collect(),
            ContourType::Catheter => geometry
                .frames
                .iter()
                .filter_map(|frame| frame.extras.get(&ContourType::Catheter))
                .cloned()
                .collect(),
            ContourType::Wall => geometry
                .frames
                .iter()
                .filter_map(|frame| frame.extras.get(&ContourType::Wall))
                .cloned()
                .collect(),
            ContourType::Eem => geometry
                .frames
                .iter()
                .filter_map(|frame| frame.extras.get(&ContourType::Eem))
                .cloned()
                .collect(),
            ContourType::Calcification => geometry
                .frames
                .iter()
                .filter_map(|frame| frame.extras.get(&ContourType::Calcification))
                .cloned()
                .collect(),
            ContourType::Sidebranch => geometry
                .frames
                .iter()
                .filter_map(|frame| frame.extras.get(&ContourType::Sidebranch))
                .cloned()
                .collect(),
        }
    }

    // Get the object string for filenames
    pub fn as_str(&self) -> &'static str {
        match self {
            ContourType::Lumen => "lumen",
            ContourType::Catheter => "catheter",
            ContourType::Wall => "wall",
            ContourType::Eem => "eem",
            ContourType::Calcification => "calcification",
            ContourType::Sidebranch => "sidebranch",
        }
    }
}

pub fn write_geometry_vec_to_obj(
    contour_type: ContourType,
    case_name: &str,
    output_dir: impl AsRef<Path>,
    geometries: &[Geometry],
    uv_coords: &[Vec<(f64, f64)>],
    watertight: bool,
) -> anyhow::Result<()> {
    // Create owned versions for thread-safe capture
    let output_dir = output_dir.as_ref();
    std::fs::create_dir_all(output_dir)
        .context(format!("Could not create output directory: {output_dir:?}"))?;

    let case_name = case_name.to_owned();
    let total = geometries.len();

    let results: Vec<anyhow::Result<()>> = geometries
        .par_iter()
        .zip(uv_coords.par_iter())
        .enumerate()
        .map(|(i, (geometry, mesh_uv))| {
            let obj_name = format!("{}_{:03}_{}.obj", contour_type.as_str(), i, case_name);
            let mtl_name = format!("{}_{:03}_{}.mtl", contour_type.as_str(), i, case_name);

            let obj_path = output_dir.join(&obj_name);

            let obj_path_str = obj_path
                .to_str()
                .ok_or_else(|| anyhow!("Invalid path for OBJ file"))?;

            let contours = contour_type.get_contours(geometry);
            write_obj_mesh(&contours, mesh_uv, obj_path_str, &mtl_name, watertight)
                .map_err(|e| anyhow!("Failed [{obj_name}]: {e}"))
        })
        .collect();

    let success_count = results.iter().filter(|r| r.is_ok()).count();
    let fail_count = total - success_count;

    println!(
        "{} .obj files: {}/{} written successfully{}",
        contour_type.as_str().to_uppercase(),
        success_count,
        total,
        if fail_count > 0 {
            format!(", {fail_count} failures")
        } else {
            String::new()
        }
    );

    // If any failed, return an Err with all messages joined
    if fail_count > 0 {
        let errors = results
            .into_iter()
            .filter_map(|r| r.err())
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        bail!("Some .obj writes failed:\n{errors}");
    }

    Ok(())
}

/// Writes `centerline` as an ASCII VTP file that [`read_centerline_vtp`] reads back: one VTK
/// line per branch (removed, empty branches are skipped) and the radii as the
/// `MaximumInscribedSphereRadius` point array, as VMTK writes it. Tangents are not stored; the
/// reader recomputes them from the points.
///
/// The reader numbers branches by descending arc length, so branch ids survive a round trip
/// only if they already follow that order.
///
/// [`read_centerline_vtp`]: crate::intravascular::io::input::read_centerline_vtp
pub fn write_centerline_vtp<P: AsRef<Path>>(
    centerline: &Centerline,
    path: P,
) -> anyhow::Result<()> {
    /// Values per text line inside a DataArray, as VTK writes them.
    const VALUES_PER_LINE: usize = 6;

    let path = path.as_ref();
    let branches: Vec<_> = centerline
        .branch_ids()
        .into_iter()
        .filter_map(|id| centerline.branch_range(id))
        .map(|range| &centerline.points[range])
        .collect();
    if branches.is_empty() {
        bail!("Centerline has no points to write.");
    }
    let n_pts: usize = branches.iter().map(|b| b.len()).sum();

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .context(format!("Could not create output directory: {parent:?}"))?;
    }
    let file = File::create(path).with_context(|| format!("cannot create {path:?}"))?;
    let mut w = BufWriter::new(file);

    // Writes `values` as the text of a DataArray, `VALUES_PER_LINE` to a line.
    let write_values = |w: &mut BufWriter<File>, values: Vec<String>| -> std::io::Result<()> {
        for line in values.chunks(VALUES_PER_LINE) {
            writeln!(w, "          {}", line.join(" "))?;
        }
        Ok(())
    };

    let points = branches.iter().flat_map(|b| b.iter());
    let coords = points
        .clone()
        .flat_map(|p| [p.contour_point.x, p.contour_point.y, p.contour_point.z])
        .map(|v| v.to_string())
        .collect();
    let radii = points.map(|p| p.radius.to_string()).collect();
    let connectivity = (0..n_pts).map(|i| i.to_string()).collect();
    let offsets = branches
        .iter()
        .scan(0, |end, b| {
            *end += b.len();
            Some(end.to_string())
        })
        .collect();

    writeln!(w, r#"<?xml version="1.0"?>"#)?;
    writeln!(
        w,
        r#"<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian" header_type="UInt32">"#
    )?;
    writeln!(w, "  <PolyData>")?;
    writeln!(
        w,
        r#"    <Piece NumberOfPoints="{n_pts}" NumberOfVerts="0" NumberOfLines="{}" NumberOfStrips="0" NumberOfPolys="0">"#,
        branches.len()
    )?;
    writeln!(w, "      <PointData>")?;
    writeln!(
        w,
        r#"        <DataArray type="Float64" Name="{VTP_RADIUS_ARRAY}" format="ascii">"#
    )?;
    write_values(&mut w, radii)?;
    writeln!(w, "        </DataArray>")?;
    writeln!(w, "      </PointData>")?;
    writeln!(w, "      <Points>")?;
    writeln!(
        w,
        r#"        <DataArray type="Float64" Name="Points" NumberOfComponents="3" format="ascii">"#
    )?;
    write_values(&mut w, coords)?;
    writeln!(w, "        </DataArray>")?;
    writeln!(w, "      </Points>")?;
    writeln!(w, "      <Lines>")?;
    writeln!(
        w,
        r#"        <DataArray type="Int64" Name="connectivity" format="ascii">"#
    )?;
    write_values(&mut w, connectivity)?;
    writeln!(w, "        </DataArray>")?;
    writeln!(
        w,
        r#"        <DataArray type="Int64" Name="offsets" format="ascii">"#
    )?;
    write_values(&mut w, offsets)?;
    writeln!(w, "        </DataArray>")?;
    writeln!(w, "      </Lines>")?;
    writeln!(w, "    </Piece>")?;
    writeln!(w, "  </PolyData>")?;
    writeln!(w, "</VTKFile>")?;
    w.flush()
        .with_context(|| format!("cannot write {path:?}"))?;
    Ok(())
}

#[cfg(test)]
mod output_tests {
    use super::*;
    use crate::intravascular::io::input::{
        dataarray_text, extract_section, parse_nums, read_centerline_vtp, split_cells,
    };

    #[test]
    fn test_write_centerline_vtp_round_trip() -> anyhow::Result<()> {
        let cl = read_centerline_vtp("examples/data/rca_cl.vtp")?;
        let path = std::env::temp_dir().join("multimodars_test_write_centerline.vtp");
        write_centerline_vtp(&cl, &path)?;
        let back = read_centerline_vtp(&path);
        let xml = std::fs::read_to_string(&path);
        let _ = std::fs::remove_file(&path);
        let (back, xml) = (back?, xml?);

        // Branch ids come back unchanged because the reader already ordered them by length.
        assert_eq!(back.branch_start_indices, cl.branch_start_indices);
        assert_eq!(back.points.len(), cl.points.len());
        for (a, b) in cl.points.iter().zip(&back.points) {
            assert_eq!(a, b, "point must round-trip exactly");
        }

        // One VTK line per branch, covering all points in order.
        let lines = extract_section(&xml, "Lines")?;
        let connectivity: Vec<usize> = parse_nums(dataarray_text(lines, "connectivity")?)?;
        let offsets: Vec<usize> = parse_nums(dataarray_text(lines, "offsets")?)?;
        let cells = split_cells(&connectivity, &offsets)?;
        assert_eq!(cells.len(), cl.branch_start_indices.len());
        assert_eq!(connectivity, (0..cl.points.len()).collect::<Vec<_>>());
        Ok(())
    }

    #[test]
    fn test_write_centerline_vtp_skips_removed_branches() -> anyhow::Result<()> {
        let mut cl = read_centerline_vtp("examples/data/rca_cl.vtp")?;
        let removed = cl.remove_branches(&[1]).map_err(|e| anyhow!(e))?;
        let path = std::env::temp_dir().join("multimodars_test_write_centerline_removed.vtp");
        write_centerline_vtp(&cl, &path)?;
        let back = read_centerline_vtp(&path);
        let _ = std::fs::remove_file(&path);
        let back = back?;

        assert_eq!(
            back.branch_start_indices.len(),
            4 - removed.len(),
            "removed branches must not be written"
        );
        assert_eq!(back.points.len(), cl.points.len());
        Ok(())
    }

    #[test]
    fn test_write_centerline_vtp_rejects_empty() {
        let cl = Centerline {
            points: Vec::new(),
            branch_start_indices: Vec::new(),
        };
        let path = std::env::temp_dir().join("multimodars_test_write_centerline_empty.vtp");
        assert!(write_centerline_vtp(&cl, &path).is_err());
        assert!(!path.exists());
    }
}
