pub mod operations;
pub mod preprocessing;

use crate::io::{input::Contour, Geometry};
use crate::texture::write_mtl_geometry;
use operations::{find_optimal_rotation, get_transformations};
use preprocessing::{prepare_data_3d_alignment, read_interpolated_meshes};

use crate::io::output::{write_geometry_vec_to_obj, GeometryType};

pub fn create_centerline_aligned_meshes(
    state: &str,
    centerline_path: &str,
    input_dir: &str,
    output_dir: &str,
    interpolation_steps: usize,
    aortic_ref_pt: (f64, f64, f64),
    upper_ref_pt: (f64, f64, f64),
    lower_ref_pt: (f64, f64, f64),
) -> anyhow::Result<(Geometry, Geometry)> {
    let (centerline, ref_mesh_dia, ref_mesh_sys) =
        prepare_data_3d_alignment(state, centerline_path, input_dir, interpolation_steps)?;
    let interpolated_meshes = read_interpolated_meshes(state, input_dir, interpolation_steps);

    let mut geometries = vec![ref_mesh_dia.clone()];
    geometries.extend(interpolated_meshes);
    geometries.extend(vec![ref_mesh_sys.clone()]);

    let reference_point = &ref_mesh_dia.reference_point;

    let optimal_angle = find_optimal_rotation(
        &ref_mesh_dia.contours[0],
        reference_point,
        aortic_ref_pt,
        upper_ref_pt,
        lower_ref_pt,
        0.01745329, // Step size in radians (1 degree)
        &centerline.points[0],
    );

    // Rotate every contour in every geometry in geometries by the optimal angle
    for geometry in &mut geometries {
        for contour in &mut geometry.contours {
            contour.rotate_contour(optimal_angle);
        }
    }

    // Rotate the catheter points in every geometry in geometries by the optimal angle around the centroid of the corresponding contour
    for geometry in &mut geometries {
        for catheter in &mut geometry.catheter {
            if let Some(contour) = geometry.contours.iter().find(|c| c.id == catheter.id) {
                catheter.rotate_contour_around_point(
                    optimal_angle,
                    (contour.centroid.0, contour.centroid.1),
                );
            } else {
                eprintln!(
                    "No matching contour found for catheter with id {}",
                    catheter.id
                );
            }
        }
    }

    let transformations = get_transformations(ref_mesh_dia, &centerline);

    for geometry in &mut geometries {
        for contour in &mut geometry.contours {
            if let Some(transformation) =
                transformations.iter().find(|t| t.frame_index == contour.id)
            {
                for pt in &mut contour.points {
                    *pt = transformation.apply_to_point(pt);
                }
                // Recompute the centroid after transformation
                contour.centroid = Contour::compute_centroid(&contour.points);
            } else {
                eprintln!("No transformation found for contour {}", contour.id);
            }
        }
        // Apply transformations to catheter points if applicable
        for catheter in &mut geometry.catheter {
            for pt in &mut catheter.points {
                // Assuming catheter points use the same frame index as their corresponding contour
                // Find the transformation based on catheter's frame index (if applicable)
                // This part depends on how catheter points are associated with frames
                if let Some(transformation) = transformations
                    .iter()
                    .find(|t| t.frame_index == catheter.id)
                {
                    *pt = transformation.apply_to_point(pt);
                }
            }
            catheter.centroid = Contour::compute_centroid(&catheter.points);
        }
    }

    let (uv_coords_contours, uv_coords_catheter) =
        write_mtl_geometry(&geometries, output_dir, state);

    write_geometry_vec_to_obj(
        GeometryType::Contour,
        state,
        output_dir,
        &geometries,
        &uv_coords_contours,
    )?;

    write_geometry_vec_to_obj(
        GeometryType::Catheter,
        state,
        output_dir,
        &geometries,
        &uv_coords_catheter,
    )?;

    let n_geometries = geometries.len();
    let dia_geom = geometries
        .get(0)
        .expect("Expected at least one geometry for diastole");
    let sys_geom = geometries
        .get(n_geometries - 1)
        .expect("Expected at least one geometry for systole");

    Ok((dia_geom.clone(), sys_geom.clone()))
}
