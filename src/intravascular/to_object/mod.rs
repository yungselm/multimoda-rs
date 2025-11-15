pub mod interpolation;
mod texture;
pub mod write_mtl;

use crate::intravascular::io::geometry::ContourType;
use crate::intravascular::io::output::write_geometry_vec_to_obj;
use crate::intravascular::processing::align_between::GeometryPair;
use {interpolation::interpolate_contours, write_mtl::write_mtl_geometry};

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
    for contour_type in contour_types {
        if let Some(uv_coords) = uv_coords_map.get(contour_type) {
            write_geometry_vec_to_obj(
                *contour_type,
                case_name,
                output_dir,
                &interpolated_geometries,
                uv_coords,
                watertight,
            )?;
        } else {
            eprintln!(
                "Warning: No UV coordinates found for contour type {:?}",
                contour_type
            );
        }
    }

    let label = geometries.label.clone();
    Ok(GeometryPair {
        geom_a,
        geom_b,
        label,
    })
}
