use super::texture::{
    compute_displacements, compute_uv_coordinates, create_black_texture,
    create_displacement_texture, create_transparent_texture,
};
use crate::intravascular::io::geometry::{Contour, ContourType, Geometry};
use ::std::fs::File;
use ::std::io::Write;
use std::collections::HashMap;
use std::path::Path;

/// Creates UV-maps and textures for different contour types based on their requirements:
/// - Lumen and EEM: displacement texture
/// - Catheter and Calcification: black texture  
/// - Wall and Sidebranch: transparent texture
pub fn write_mtl_geometry(
    geometries_to_process: &Vec<Geometry>,
    output_dir: &str,
    case_name: &str,
    contour_types: &[ContourType],
) -> HashMap<ContourType, Vec<Vec<(f64, f64)>>> {
    let mut uv_coords_map = HashMap::new();

    for &contour_type in contour_types {
        let uv_coords =
            write_mtl_for_contour_type(geometries_to_process, output_dir, case_name, contour_type);
        uv_coords_map.insert(contour_type, uv_coords);
    }

    uv_coords_map
}

/// Writes MTL and texture files for a specific contour type
fn write_mtl_for_contour_type(
    geometries_to_process: &Vec<Geometry>,
    output_dir: &str,
    case_name: &str,
    contour_type: ContourType,
) -> Vec<Vec<(f64, f64)>> {
    match contour_type {
        ContourType::Lumen | ContourType::Eem => {
            // For lumen and EEM: displacement texture
            write_displacement_texture(geometries_to_process, output_dir, case_name, contour_type)
        }
        ContourType::Catheter | ContourType::Calcification => {
            // For catheter and calcification: black texture
            write_black_texture(geometries_to_process, output_dir, case_name, contour_type)
        }
        ContourType::Wall | ContourType::Sidebranch => {
            // For wall and sidebranch: transparent texture
            write_transparent_texture(geometries_to_process, output_dir, case_name, contour_type)
        }
    }
}

/// Writes displacement texture for contour types that need displacement mapping
fn write_displacement_texture(
    geometries_to_process: &Vec<Geometry>,
    output_dir: &str,
    case_name: &str,
    contour_type: ContourType,
) -> Vec<Vec<(f64, f64)>> {
    let reference_geometry = &geometries_to_process[0];
    let mut uv_coords = Vec::new();

    // Calculate max displacements between first and last geometry for normalization
    let max_disp = if geometries_to_process.len() > 1 {
        let start_contours = extract_contours_by_type(&geometries_to_process[0], contour_type);
        let end_contours =
            extract_contours_by_type(geometries_to_process.last().unwrap(), contour_type);

        if !start_contours.is_empty() && !end_contours.is_empty() {
            let displacements = compute_displacements_for_contours(&start_contours, &end_contours);
            displacements.iter().cloned().fold(0.0, f64::max)
        } else {
            1.0 // Default to avoid division by zero
        }
    } else {
        1.0 // Only one geometry, no displacement
    };

    for (i, geometry) in geometries_to_process.iter().enumerate() {
        let contours = extract_contours_by_type(geometry, contour_type);

        if contours.is_empty() {
            uv_coords.push(Vec::new());
            continue;
        }

        let uv_coords_geometry = compute_uv_coordinates(&contours);
        uv_coords.push(uv_coords_geometry.clone());

        let texture_height = contours.len() as u32;
        let texture_width = if texture_height > 0 {
            contours[0].points.len() as u32
        } else {
            0
        };

        // Compute displacements relative to reference geometry using the texture.rs function
        let displacements = compute_displacements(geometry, reference_geometry);

        // Save displacement texture
        let type_name = get_contour_type_name(contour_type);
        let tex_filename = format!("{}_{:03}_{}.png", type_name, i, case_name);
        let texture_path = Path::new(output_dir).join(&tex_filename);

        if let Err(e) = create_displacement_texture(
            &displacements,
            texture_width,
            texture_height,
            max_disp,
            texture_path.to_str().unwrap(),
        ) {
            eprintln!(
                "Failed to create displacement texture for {}: {}",
                type_name, e
            );
            continue;
        }

        // Write MTL file
        let mtl_filename = format!("{}_{:03}_{}.mtl", type_name, i, case_name);
        let mtl_path = Path::new(output_dir).join(&mtl_filename);

        if let Ok(mut mtl_file) = File::create(&mtl_path) {
            writeln!(
                mtl_file,
                "newmtl displacement_material\nKa 1 1 1\nKd 1 1 1\nmap_Kd {}",
                tex_filename
            )
            .unwrap_or_else(|e| eprintln!("Failed to write MTL file: {}", e));
        }
    }

    uv_coords
}

/// Writes black texture for contour types that should be solid black
fn write_black_texture(
    geometries_to_process: &Vec<Geometry>,
    output_dir: &str,
    case_name: &str,
    contour_type: ContourType,
) -> Vec<Vec<(f64, f64)>> {
    let mut uv_coords = Vec::new();

    for (i, geometry) in geometries_to_process.iter().enumerate() {
        let contours = extract_contours_by_type(geometry, contour_type);

        if contours.is_empty() {
            uv_coords.push(Vec::new());
            continue;
        }

        let uv_coords_geometry = compute_uv_coordinates(&contours);
        uv_coords.push(uv_coords_geometry.clone());

        let texture_height = contours.len() as u32;
        let texture_width = if texture_height > 0 {
            contours[0].points.len() as u32
        } else {
            0
        };

        // Save black texture
        let type_name = get_contour_type_name(contour_type);
        let tex_filename = format!("{}_{:03}_{}.png", type_name, i, case_name);
        let texture_path = Path::new(output_dir).join(&tex_filename);

        if let Err(e) = create_black_texture(
            texture_width,
            texture_height,
            texture_path.to_str().unwrap(),
        ) {
            eprintln!("Failed to create black texture for {}: {}", type_name, e);
            continue;
        }

        // Write MTL file
        let mtl_filename = format!("{}_{:03}_{}.mtl", type_name, i, case_name);
        let mtl_path = Path::new(output_dir).join(&mtl_filename);

        if let Ok(mut mtl_file) = File::create(&mtl_path) {
            writeln!(
                mtl_file,
                "newmtl black_material\nKa 0 0 0\nKd 0 0 0\nmap_Kd {}",
                tex_filename
            )
            .unwrap_or_else(|e| eprintln!("Failed to write MTL file: {}", e));
        }
    }

    uv_coords
}

/// Writes transparent texture for contour types that should be semi-transparent
fn write_transparent_texture(
    geometries_to_process: &Vec<Geometry>,
    output_dir: &str,
    case_name: &str,
    contour_type: ContourType,
) -> Vec<Vec<(f64, f64)>> {
    let mut uv_coords = Vec::new();

    for (i, geometry) in geometries_to_process.iter().enumerate() {
        let contours = extract_contours_by_type(geometry, contour_type);

        if contours.is_empty() {
            uv_coords.push(Vec::new());
            continue;
        }

        let uv_coords_geometry = compute_uv_coordinates(&contours);
        uv_coords.push(uv_coords_geometry.clone());

        let texture_height = contours.len() as u32;
        let texture_width = if texture_height > 0 {
            contours[0].points.len() as u32
        } else {
            0
        };

        // Save transparent texture
        let type_name = get_contour_type_name(contour_type);
        let tex_filename = format!("{}_{:03}_{}.png", type_name, i, case_name);
        let texture_path = Path::new(output_dir).join(&tex_filename);

        if let Err(e) = create_transparent_texture(
            texture_width,
            texture_height,
            0.7, // alpha value
            texture_path.to_str().unwrap(),
        ) {
            eprintln!(
                "Failed to create transparent texture for {}: {}",
                type_name, e
            );
            continue;
        }

        // Write MTL file
        let mtl_filename = format!("{}_{:03}_{}.mtl", type_name, i, case_name);
        let mtl_path = Path::new(output_dir).join(&mtl_filename);

        if let Ok(mut mtl_file) = File::create(&mtl_path) {
            writeln!(
                mtl_file,
                "newmtl transparent_material\nKa 0 0 0\nKd 0 0 0\nmap_Kd {}",
                tex_filename
            )
            .unwrap_or_else(|e| eprintln!("Failed to write MTL file: {}", e));
        }
    }

    uv_coords
}

/// Extracts contours of a specific type from a geometry
fn extract_contours_by_type(geometry: &Geometry, contour_type: ContourType) -> Vec<Contour> {
    match contour_type {
        ContourType::Lumen => geometry
            .frames
            .iter()
            .map(|frame| frame.lumen.clone())
            .collect(),
        _ => geometry
            .frames
            .iter()
            .filter_map(|frame| frame.extras.get(&contour_type).cloned())
            .collect(),
    }
}

/// Computes displacements between two sets of contours (alternative implementation)
/// This is used for max displacement calculation since the texture.rs version only works on Geometry
fn compute_displacements_for_contours(reference: &[Contour], target: &[Contour]) -> Vec<f64> {
    reference
        .iter()
        .zip(target.iter())
        .flat_map(|(ref_contour, target_contour)| {
            ref_contour
                .points
                .iter()
                .zip(target_contour.points.iter())
                .map(|(ref_pt, target_pt)| {
                    let dx = ref_pt.x - target_pt.x;
                    let dy = ref_pt.y - target_pt.y;
                    let dz = ref_pt.z - target_pt.z;
                    (dx * dx + dy * dy + dz * dz).sqrt()
                })
                .collect::<Vec<f64>>()
        })
        .collect()
}

/// Gets the string name for a contour type for file naming
fn get_contour_type_name(contour_type: ContourType) -> &'static str {
    match contour_type {
        ContourType::Lumen => "lumen",
        ContourType::Eem => "eem",
        ContourType::Calcification => "calcification",
        ContourType::Sidebranch => "sidebranch",
        ContourType::Catheter => "catheter",
        ContourType::Wall => "wall",
    }
}
