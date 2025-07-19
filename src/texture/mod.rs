pub mod texture;

use crate::io::Geometry;
use ::std::fs::File;
use ::std::io::Write;
use std::path::Path;
use texture::{
    compute_displacements, compute_uv_coordinates, create_black_texture,
    create_displacement_texture, create_transparent_texture,
};

/// Creates a UV-map based on displacement between geometries and saves the map
/// in png format. Additionally creates a .mtl file with material parameter
/// 1 1 1 1 for the moment.
pub fn write_mtl_geometry(
    geometries_to_process: &Vec<Geometry>,
    output_dir: &str,
    case_name: &str,
) -> (
    Vec<Vec<(f64, f64)>>,
    Vec<Vec<(f64, f64)>>,
    Vec<Vec<(f64, f64)>>,
) {
    let reference_contours = geometries_to_process[0].clone();

    // calculate max displacements, since all meshes in between are interpolated
    // meshes, it is enough to compare first and last element of Vec<Geometry> and
    // then extract the max value needed to normalize.
    let disp = compute_displacements(
        &geometries_to_process[0],
        geometries_to_process.last().unwrap(),
    );
    let max_disp = disp.iter().cloned().reduce(f64::max).unwrap();

    let mut uv_coords_contours = Vec::new();

    for (i, mesh) in geometries_to_process.into_iter().enumerate() {
        let uv_coords = compute_uv_coordinates(&mesh.contours);

        let texture_height = mesh.contours.len() as u32;
        let texture_width = if texture_height > 0 {
            mesh.contours[0].points.len() as u32
        } else {
            0
        };

        let displacements = compute_displacements(&mesh, &reference_contours);

        // Save the displacement texture.
        let tex_filename = format!("mesh_{:03}_{}.png", i, case_name);
        let texture_path = Path::new(output_dir).join(&tex_filename);
        create_displacement_texture(
            &displacements,
            texture_width,
            texture_height,
            max_disp,
            texture_path.to_str().unwrap(),
        )
        .unwrap();

        // Write the material file (MTL).
        let mtl_filename = format!("mesh_{:03}_{}.mtl", i, case_name);
        let mtl_path = Path::new(output_dir).join(&mtl_filename);
        let mut mtl_file = File::create(&mtl_path).unwrap();
        writeln!(
            mtl_file,
            "newmtl displacement_material\nKa 1 1 1\nKd 1 1 1\nmap_Kd {}",
            tex_filename
        )
        .unwrap();

        uv_coords_contours.push(uv_coords)
    }

    let mut uv_coords_catheter = Vec::new();

    // for catheter no displacement uv texture needed
    for (i, mesh) in geometries_to_process.into_iter().enumerate() {
        if mesh.catheter.is_empty() {
            let uv_coord = Vec::new();
            uv_coords_catheter.push(uv_coord);
        } else {
            let uv_coords = compute_uv_coordinates(&mesh.catheter);

            let texture_height = mesh.catheter.len() as u32;
            let texture_width = if texture_height > 0 {
                mesh.catheter[0].points.len() as u32
            } else {
                0
            };

            // Fixed (black) texture.
            let tex_filename = format!("catheter_{:03}_{}.png", i, case_name);
            let texture_path = Path::new(output_dir).join(&tex_filename);
            create_black_texture(
                texture_width,
                texture_height,
                texture_path.to_str().unwrap(),
            )
            .unwrap();

            // Write the material file (MTL).
            let mtl_filename = format!("catheter_{:03}_{}.mtl", i, case_name);
            let mtl_path = Path::new(output_dir).join(&mtl_filename);
            let mut mtl_file = File::create(&mtl_path).unwrap();
            // Set both ambient and diffuse to black.
            writeln!(
                mtl_file,
                "newmtl black_material\nKa 0 0 0\nKd 0 0 0\nmap_Kd {}",
                tex_filename
            )
            .unwrap();

            uv_coords_catheter.push(uv_coords)
        }
    }

    let uv_coords_walls = write_mtl_wall(geometries_to_process, output_dir, case_name);

    (uv_coords_contours, uv_coords_catheter, uv_coords_walls)
}

pub fn write_mtl_wall(
    walls_to_process: &Vec<Geometry>,
    output_dir: &str,
    case_name: &str,
) -> Vec<Vec<(f64, f64)>> {
    let mut uv_coords_walls = Vec::new();

    // for catheter no displacement uv texture needed
    for (i, wall) in walls_to_process.into_iter().enumerate() {
        let uv_coords = compute_uv_coordinates(&wall.contours);

        let texture_height = wall.contours.len() as u32;
        let texture_width = if texture_height > 0 {
            wall.contours[0].points.len() as u32
        } else {
            0
        };

        // Fixed (black) texture.
        let tex_filename = format!("wall_{:03}_{}.png", i, case_name);
        let texture_path = Path::new(output_dir).join(&tex_filename);
        create_transparent_texture(
            texture_width,
            texture_height,
            0.7,
            texture_path.to_str().unwrap(),
        )
        .unwrap();

        // Write the material file (MTL).
        let mtl_filename = format!("wall_{:03}_{}.mtl", i, case_name);
        let mtl_path = Path::new(output_dir).join(&mtl_filename);
        let mut mtl_file = File::create(&mtl_path).unwrap();
        // Set both ambient and diffuse to black.
        writeln!(
            mtl_file,
            "newmtl black_material\nKa 0 0 0\nKd 0 0 0\nmap_Kd {}",
            tex_filename
        )
        .unwrap();

        uv_coords_walls.push(uv_coords)
    }

    uv_coords_walls
}
