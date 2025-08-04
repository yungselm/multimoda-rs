pub mod align_algorithms;
pub mod preprocessing;

use crate::centerline_align::align_algorithms::{get_transformations, FrameTransformation};
use crate::centerline_align::preprocessing::{
    ensure_descending_z, resample_centerline_by_contours,
};
use crate::io::{
    input::{Centerline, Contour},
    Geometry,
};
use crate::processing::align_between::GeometryPair;
use anyhow::Error;

use crate::io::output::{write_geometry_vec_to_obj, GeometryType};
use crate::processing::process_utils::interpolate_contours;
use crate::texture::write_mtl_geometry;
use align_algorithms::best_rotation_three_point;
use preprocessing::{prepare_geometry_alignment, remove_leading_points_cl};

pub fn align_three_point_rs(
    centerline: Centerline,
    geometry_pair: GeometryPair,
    aortic_ref_pt: (f64, f64, f64),
    upper_ref_pt: (f64, f64, f64),
    lower_ref_pt: (f64, f64, f64),
    angle_step: f64,
    write: bool,
    interpolation_steps: usize,
    output_dir: &str,
    case_name: &str,
) -> (GeometryPair, Centerline) {
    let mut geom = prepare_geometry_alignment(geometry_pair);
    let mut cl = centerline.clone();
    ensure_descending_z(&mut cl);

    let mut centerline = remove_leading_points_cl(cl, &aortic_ref_pt);
    ensure_descending_z(&mut centerline);

    let resampled_centerline = resample_centerline_by_contours(&centerline, &geom.dia_geom);

    let best_rot = best_rotation_three_point(
        &geom.dia_geom.contours[0],
        &geom.dia_geom.reference_point,
        aortic_ref_pt,
        upper_ref_pt,
        lower_ref_pt,
        angle_step,
        &resampled_centerline.points[0],
    );

    geom = rotate_by_best_rotation(geom, best_rot);

    geom = apply_transformations(geom, &resampled_centerline);

    if write {
        write_aligned_meshes(geom.clone(), interpolation_steps, output_dir, case_name).unwrap();
    }

    (geom, resampled_centerline)
}

pub fn align_manual_rs(
    centerline: Centerline,
    geometry_pair: GeometryPair,
    rotation_angle: f64,
    start_point: usize,
    write: bool,
    interpolation_steps: usize,
    output_dir: &str,
    case_name: &str,
) -> (GeometryPair, Centerline) {
    let mut geom = prepare_geometry_alignment(geometry_pair);
    let mut cl = centerline.clone();
    ensure_descending_z(&mut cl);

    // maybe stupidly extensive, but can reuse remove_leading_points function
    let ref_pt = centerline.points[start_point].contour_point;
    let ref_coords = (ref_pt.x, ref_pt.y, ref_pt.z);

    let mut centerline = remove_leading_points_cl(cl, &ref_coords);
    ensure_descending_z(&mut centerline);

    let resampled_centerline = resample_centerline_by_contours(&centerline, &geom.dia_geom);

    geom = rotate_by_best_rotation(geom, rotation_angle);

    geom = apply_transformations(geom, &resampled_centerline);

    if write {
        write_aligned_meshes(geom.clone(), interpolation_steps, output_dir, case_name).unwrap();
    }

    (geom, resampled_centerline)
}

// pub fn align_hausdorff() -> () {
//     todo!()
// }

fn rotate_by_best_rotation(mut geom_pair: GeometryPair, angle: f64) -> GeometryPair {
    fn rotate_geometry(mut geom: Geometry, angle: f64) -> Geometry {
        for contour in &mut geom.contours {
            contour.rotate_contour(angle);
        }

        for wall in &mut geom.walls {
            wall.rotate_contour(angle);
        }

        for catheter in &mut geom.catheter {
            if let Some(contour) = geom.contours.iter().find(|c| c.id == catheter.id) {
                catheter
                    .rotate_contour_around_point(angle, (contour.centroid.0, contour.centroid.1));
            } else {
                eprintln!(
                    "No matching contour found for catheter with id {}",
                    catheter.id
                );
            }
        }

        geom
    }

    geom_pair.dia_geom = rotate_geometry(geom_pair.dia_geom, angle);
    geom_pair.sys_geom = rotate_geometry(geom_pair.sys_geom, angle);

    geom_pair
}

fn apply_transformations(mut geom_pair: GeometryPair, centerline: &Centerline) -> GeometryPair {
    let reference = geom_pair.dia_geom.clone();
    let transformations = get_transformations(reference, centerline);

    fn transform_geometry(
        mut geometry: Geometry,
        transformations: &Vec<FrameTransformation>,
    ) -> Geometry {
        // Sys contours
        for contour in &mut geometry.contours {
            if let Some(tr) = transformations.iter().find(|t| t.frame_index == contour.id) {
                for pt in &mut contour.points {
                    *pt = tr.apply_to_point(pt);
                }
                contour.centroid = Contour::compute_centroid(&contour.points);
            } else {
                eprintln!("No transformation found for contour {}", contour.id);
            }
        }

        // Sys walls
        for wall in &mut geometry.walls {
            if let Some(tr) = transformations.iter().find(|t| t.frame_index == wall.id) {
                for pt in &mut wall.points {
                    *pt = tr.apply_to_point(pt);
                }
                wall.centroid = Contour::compute_centroid(&wall.points);
            } else {
                eprintln!("No transformation found for wall {}", wall.id);
            }
        }

        // Sys catheter
        for catheter in &mut geometry.catheter {
            if let Some(tr) = transformations
                .iter()
                .find(|t| t.frame_index == catheter.id)
            {
                for pt in &mut catheter.points {
                    *pt = tr.apply_to_point(pt);
                }
            }
            catheter.centroid = Contour::compute_centroid(&catheter.points);
        }

        geometry
    }

    geom_pair.dia_geom = transform_geometry(geom_pair.dia_geom, &transformations);
    geom_pair.sys_geom = transform_geometry(geom_pair.sys_geom, &transformations);

    geom_pair
}

fn write_aligned_meshes(
    geom_pair: GeometryPair,
    interpolation_steps: usize,
    output_dir: &str,
    case_name: &str,
) -> Result<(), Error> {
    let geometries = interpolate_contours(
        &geom_pair.dia_geom,
        &geom_pair.sys_geom,
        interpolation_steps,
    )?;

    let (uv_coords_contours, uv_coords_catheter, uv_coords_walls) =
        write_mtl_geometry(&geometries, output_dir, case_name);

    write_geometry_vec_to_obj(
        GeometryType::Contour,
        case_name,
        output_dir,
        &geometries,
        &uv_coords_contours,
    )?;

    write_geometry_vec_to_obj(
        GeometryType::Catheter,
        case_name,
        output_dir,
        &geometries,
        &uv_coords_catheter,
    )?;

    write_geometry_vec_to_obj(
        GeometryType::Wall,
        case_name,
        output_dir,
        &geometries,
        &uv_coords_walls,
    )?;

    Ok(())
}
