use anyhow::Context;

pub mod resampling;
pub mod align_within;
pub mod align_between;
pub mod process_utils;
pub mod walls;

use crate::processing::align_within::{align_frames_in_geometries, AlignLog};
use crate::processing::align_between::{get_geometry_pair, GeometryPair};
use crate::processing::process_utils::process_case;
use crate::processing::walls::create_wall_geometry;

pub fn align_within_and_between(
    case_name: &str,
    input_dir: &str,
    step_rotation_deg: f64,
    range_rotation_deg: f64,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
    write_obj: bool,
    output_dir: &str,
    interpolation_steps: usize,
    bruteforce: bool,
    sample_size: usize,
) -> anyhow::Result<(GeometryPair, Vec<AlignLog>, Vec<AlignLog>)> {
    let geom_pair = get_geometry_pair(
        case_name.to_string(), 
        input_dir, 
        image_center, 
        radius, 
        n_points)
        .context(format!("create_geometry_pair({}) failed", case_name))?;
    
    let (mut geom_pair, (logs_a, logs_b)) = align_frames_in_geometries(
        geom_pair,
        step_rotation_deg, 
        range_rotation_deg,
        true,
        bruteforce,
        sample_size,
    )
    .context(format!("align within geometrypair({}) failed", case_name))?;
    
    geom_pair = geom_pair.align_between_geometries(
        step_rotation_deg, 
        range_rotation_deg)
        .context(format!("align between geometrypair({}) failed", case_name))?;
    
    let geom_pair_walls: GeometryPair = if write_obj {
        process_case(
            case_name,
            geom_pair,
            output_dir,
            interpolation_steps,
        )
        .context(format!("creating walls geometrypair({}) failed", case_name))?
    } else {
        GeometryPair {
            dia_geom: create_wall_geometry(&geom_pair.dia_geom, false),
            sys_geom: create_wall_geometry(&geom_pair.sys_geom, false),
        }
    };

    Ok((geom_pair_walls, logs_a, logs_b))
}

pub fn align_within_and_between_array(
    case_name: &str,
    geometry_pair: GeometryPair,
    step_rotation_deg: f64,
    range_rotation_deg: f64,
    write_obj: bool,
    output_dir: &str,
    interpolation_steps: usize,
    bruteforce: bool,
    sample_size: usize,
) -> anyhow::Result<(GeometryPair, Vec<AlignLog>, Vec<AlignLog>)> {
    let mut geometries = geometry_pair.adjust_z_coordinates();
    geometries = geometries.trim_geometries_same_length();
    let (mut geom_pair, (logs_a, logs_b)) = align_frames_in_geometries(
        geometries,
        step_rotation_deg, 
        range_rotation_deg,
        true,
        bruteforce,
        sample_size,
    )
    .context(format!("align within geometrypair({}) failed", case_name))?;
    
    geom_pair = geom_pair.align_between_geometries(
        step_rotation_deg, 
        range_rotation_deg)
        .context(format!("align between geometrypair({}) failed", case_name))?;
    
    let geom_pair_walls: GeometryPair = if write_obj {
        process_case(
            case_name,
            geom_pair,
            output_dir,
            interpolation_steps,
        )
        .context(format!("creating walls geometrypair({}) failed", case_name))?
    } else {
        GeometryPair {
            dia_geom: create_wall_geometry(&geom_pair.dia_geom, false),
            sys_geom: create_wall_geometry(&geom_pair.sys_geom, false),
        }
    };

    Ok((geom_pair_walls, logs_a, logs_b))
}