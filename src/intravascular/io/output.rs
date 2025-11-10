use anyhow::{anyhow, bail, Context};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use super::geometry::{Contour, ContourType, Geometry};
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
            .context(format!("Could not create output directory: {:?}", parent))?;
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
    writeln!(writer, "mtllib {}", mtl_filename)?;
    writeln!(writer, "usemtl displacement_material")?;

    // Write UV coordinates for original vertices
    for (u, v) in uv_coords {
        writeln!(writer, "vt {} {}", u, v)?;
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
            writeln!(writer, "vn {} {} {}", nx * -1.0, ny * -1.0, nz * -1.0)?;
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
            writeln!(writer, "f {0}/{0}/{0} {1}/{1}/{1} {2}/{2}/{2}", v1, v2, v3)?;

            let v1_t2 = offset2 + j;
            let v2_t2 = offset1 + j_next;
            let v3_t2 = offset2 + j_next;
            writeln!(
                writer,
                "f {0}/{0}/{0} {1}/{1}/{1} {2}/{2}/{2}",
                v1_t2, v2_t2, v3_t2
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
            writeln!(writer, "f {0}/{0}/{0} {1}/{1}/{1} {2}/{2}/{2}", v3, v2, v1)?;
        } else {
            writeln!(writer, "f {0}/{0}/{0} {1}/{1}/{1} {2}/{2}/{2}", v1, v2, v3)?;
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
    .map_err(|e| anyhow!("Failed to write OBJ mesh without UV: {}", e))
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
    std::fs::create_dir_all(output_dir).context(format!(
        "Could not create output directory: {:?}",
        output_dir
    ))?;

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
                .map_err(|e| anyhow!("Failed [{}]: {}", obj_name, e))
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
            format!(", {} failures", fail_count)
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
        bail!("Some .obj writes failed:\n{}", errors);
    }

    Ok(())
}
