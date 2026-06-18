use crate::intravascular::io::output;
use crate::intravascular::processing::align_between::GeometryPair;
use crate::types::native::contour::ContourType;
use crate::types::native::geometry::Geometry;
use std::io::Write;
use std::path::Path;
use {super::interpolation::interpolate_contours, super::write_mtl::write_mtl_geometry};

/// Processes a given case by reading diastolic and systolic contours, aligning them,
/// computing displacements and UV coordinates, and finally writing out the OBJ, MTL, and texture files.
/// Additionally it can be specified how many interpolation steps should be used to generate the final meshes
/// used for the animation in blender.
pub fn process_case(
    case_name: &str,
    geometries: GeometryPair,
    output_dir: &str,
    interpolation_steps: usize,
    watertight: bool,
    contour_types: &[ContourType], // Specify which contour types to process
) -> anyhow::Result<GeometryPair> {
    std::fs::create_dir_all(output_dir)?;

    let geom_a = geometries.geom_a;
    let geom_b = geometries.geom_b;

    // Interpolate between two geometries by creating new geometries with coordinates
    // in between the two geometries.
    let interpolated_geometries =
        interpolate_contours(&geom_a, &geom_b, interpolation_steps, contour_types)?;

    // Use the new write_mtl_geometry that returns a HashMap
    let uv_coords_map = write_mtl_geometry(
        &interpolated_geometries,
        output_dir,
        case_name,
        contour_types,
    );

    // Write specified contour types using UV coordinates from the map
    println!("\nSaving files for '{}' to '{}'", &case_name, &output_dir);
    for contour_type in contour_types {
        if let Some(uv_coords) = uv_coords_map.get(contour_type) {
            output::write_geometry_vec_to_obj(
                *contour_type,
                case_name,
                output_dir,
                &interpolated_geometries,
                uv_coords,
                watertight,
            )?;
        } else {
            eprintln!("Warning: No UV coordinates found for contour type {contour_type:?}");
        }
    }

    let label = geometries.label.clone();
    Ok(GeometryPair {
        geom_a,
        geom_b,
        label,
    })
}

/// Writes a single [`Geometry`] to OBJ files (one per contour type) without interpolation or UV textures.
pub fn write_single_geometry(
    case_name: &str,
    geometry: Geometry,
    output_dir: &str,
    watertight: bool,
    contour_types: &[ContourType],
) -> anyhow::Result<Geometry> {
    std::fs::create_dir_all(output_dir)?;

    println!("\nSaving files for '{case_name}' to '{output_dir}'");
    for &contour_type in contour_types {
        let contours = contour_type.get_contours(&geometry);
        if contours.is_empty() {
            eprintln!("Warning: No contours found for type {contour_type:?}, skipping");
            continue;
        }

        let type_name = format!("{contour_type}").to_lowercase();
        let obj_filename = format!("{case_name}_{type_name}.obj");
        let mtl_filename = format!("{case_name}_{type_name}.mtl");
        let obj_path = Path::new(output_dir).join(&obj_filename);
        let mtl_path = Path::new(output_dir).join(&mtl_filename);

        let mut mtl_file = std::fs::File::create(&mtl_path)?;
        match contour_type {
            ContourType::Lumen | ContourType::Eem => {
                writeln!(
                    mtl_file,
                    "newmtl material\nKa 1.0 1.0 1.0\nKd 1.0 1.0 1.0\nKs 0.0 0.0 0.0"
                )?;
            }
            ContourType::Catheter | ContourType::Calcification => {
                writeln!(
                    mtl_file,
                    "newmtl material\nKa 0.0 0.0 0.0\nKd 0.0 0.0 0.0\nKs 0.0 0.0 0.0"
                )?;
            }
            ContourType::Wall | ContourType::Sidebranch => {
                writeln!(
                    mtl_file,
                    "newmtl material\nKa 0.5 0.5 0.5\nKd 0.5 0.5 0.5\nKs 0.0 0.0 0.0\nd 0.7"
                )?;
            }
        }

        output::write_obj_mesh_without_uv(
            &contours,
            obj_path.to_str().unwrap(),
            mtl_path.to_str().unwrap(),
            watertight,
        )?;

        println!("Successfully wrote {} to {}", type_name, obj_path.display());
    }

    Ok(geometry)
}
