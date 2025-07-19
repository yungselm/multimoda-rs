use anyhow::{anyhow, bail, Context};
use std::fs::File;
use std::io::{BufWriter, Write};

use crate::io::input::Contour;
use crate::io::Geometry;
use rayon::prelude::*;
use std::path::Path;

pub fn write_obj_mesh(
    contours: &Vec<Contour>,
    uv_coords: &[(f64, f64)],
    filename: &str,
    mtl_filename: &str,
) -> anyhow::Result<()> {
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
    let mut normals = Vec::new();

    // Write vertices and compute normals
    for contour in &sorted_contours {
        vertex_offsets.push(current_offset);
        let centroid = &contour.centroid;
        for point in &contour.points {
            writeln!(writer, "v {} {} {}", point.x, point.y, point.z)?;
            let dx = point.x - centroid.0;
            let dy = point.y - centroid.1;
            let length = (dx * dx + dy * dy).sqrt();
            let (nx, ny, nz) = if length > 0.0 {
                (dx / length, dy / length, 0.0)
            } else {
                (0.0, 0.0, 0.0)
            };
            normals.push((nx * -1.0, ny * -1.0, nz * -1.0));
            current_offset += 1;
        }
    }

    // Validate UV coordinates
    if uv_coords.len() != current_offset - 1 {
        return Err(anyhow!(
            "UV coordinates must match the number of vertices. Expected {}, got {}.",
            current_offset - 1,
            uv_coords.len()
        )
        .into());
    }

    // Write material reference
    writeln!(writer, "mtllib {}", mtl_filename)?;
    writeln!(writer, "usemtl displacement_material")?;

    // Write UV coordinates
    for (u, v) in uv_coords {
        writeln!(writer, "vt {} {}", u, v)?;
    }

    // Write normals
    for (nx, ny, nz) in &normals {
        writeln!(writer, "vn {} {} {}", nx, ny, nz)?;
    }

    // Write faces with normals and UVs
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

    Ok(())
}

pub fn write_obj_mesh_without_uv(
    contours: &Vec<Contour>,
    filename: &str,
    mtl_filename: &str,
) -> anyhow::Result<()> {
    if let Some(parent) = std::path::Path::new(filename).parent() {
        std::fs::create_dir_all(parent)
            .context(format!("Could not create output directory: {:?}", parent))?;
    }
    let empty_uv_coords = vec![(0.0, 0.0); contours.iter().map(|c| c.points.len()).sum()];
    write_obj_mesh(contours, &empty_uv_coords, filename, mtl_filename)
        .map_err(|e| anyhow!("Failed to write OBJ mesh without UV: {}", e))
}

#[derive(Copy, Clone)]
pub enum GeometryType {
    Contour,
    Catheter,
    Wall,
}

impl GeometryType {
    // Get the contour data from a Geometry based on the enum variant
    pub fn get_contours<'a>(&self, geometry: &'a Geometry) -> &'a Vec<Contour> {
        match self {
            GeometryType::Contour => &geometry.contours,
            GeometryType::Catheter => &geometry.catheter,
            GeometryType::Wall => &geometry.walls,
        }
    }

    // Get the object string for filenames
    pub fn as_str(&self) -> &'static str {
        match self {
            GeometryType::Contour => "mesh",
            GeometryType::Catheter => "catheter",
            GeometryType::Wall => "wall",
        }
    }
}

pub fn write_geometry_vec_to_obj(
    geometry_type: GeometryType,
    case_name: &str,
    output_dir: impl AsRef<Path>,
    geometries: &[Geometry],
    uv_coords: &[Vec<(f64, f64)>],
) -> anyhow::Result<()> {
    // Create owned versions for thread-safe capture
    let output_dir = output_dir.as_ref(); // Get &Path reference
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
            let obj_name = format!("{}_{:03}_{}.obj", geometry_type.as_str(), i, case_name);
            let mtl_name = format!("{}_{:03}_{}.mtl", geometry_type.as_str(), i, case_name);

            let obj_path = output_dir.join(&obj_name);

            let obj_path_str = obj_path
                .to_str()
                .ok_or_else(|| anyhow!("Invalid path for OBJ file"))?;

            let contours = geometry_type.get_contours(geometry);
            write_obj_mesh(contours, mesh_uv, obj_path_str, &mtl_name)
                .map_err(|e| anyhow!("Failed [{}]: {}", obj_name, e))
        })
        .collect();

    let success_count = results.iter().filter(|r| r.is_ok()).count();
    let fail_count = total - success_count;

    println!(
        "{} .obj files: {}/{} written successfully{}",
        geometry_type.as_str().to_uppercase(),
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
