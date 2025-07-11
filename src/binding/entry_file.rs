use anyhow::{anyhow, Context, Result};
use crossbeam::thread;

use crate::processing::comparison::prepare_geometries_comparison;
use crate::processing::contours::align_frames_in_geometry;
use crate::processing::geometries::GeometryPair;
use crate::processing::process_case::{create_geometry_pair, process_case};

use crate::io::output::write_obj_mesh_without_uv;
use crate::io::Geometry;

pub fn from_file_full_rs(
    rest_input_path: &str,
    steps_best_rotation: usize,
    range_rotation_rad: f64,
    rest_output_path: &str,
    interpolation_steps: usize,
    stress_input_path: &str,
    stress_output_path: &str,
    diastole_output_path: &str,
    systole_output_path: &str,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
) -> Result<(GeometryPair, GeometryPair, GeometryPair, GeometryPair)> {
    let result = thread::scope(
        |s| -> Result<(GeometryPair, GeometryPair, GeometryPair, GeometryPair)> {
            // REST thread
            let rest_handle = s.spawn(|_| {
                let geom = create_geometry_pair(
                    "rest".to_string(),
                    rest_input_path,
                    steps_best_rotation,
                    range_rotation_rad,
                    image_center,
                    radius,
                    n_points,
                    true, //hardcoded here, since only for from array can preprocess
                )
                .context("create_geometry_pair(rest) failed")?;
                process_case("rest", geom, rest_output_path, interpolation_steps)
                    .context("process_case(rest) failed")
            });

            // STRESS thread
            let stress_handle = s.spawn(|_| {
                let geom = create_geometry_pair(
                    "stress".to_string(),
                    stress_input_path,
                    steps_best_rotation,
                    range_rotation_rad,
                    image_center,
                    radius,
                    n_points,
                    true, //hardcoded here, since only for from array can preprocess
                )
                .context("create_geometry_pair(stress) failed")?;
                process_case("stress", geom, stress_output_path, interpolation_steps)
                    .context("process_case(stress) failed")
            });

            // Join REST & STRESS
            let rest_pair = rest_handle.join().unwrap()?;
            let stress_pair = stress_handle.join().unwrap()?;

            // Prepare diastolic & systolic geometry pairs
            let (dia_pair, sys_pair) =
                prepare_geometries_comparison(rest_pair.clone(), stress_pair.clone());

            // DIASTOLIC thread
            let dia_handle = s.spawn(move |_| {
                process_case(
                    "diastolic",
                    dia_pair.clone(),
                    diastole_output_path,
                    interpolation_steps,
                )
                .context("process_case(diastolic) failed")
            });

            // SYSTOLIC thread
            let sys_handle = s.spawn(move |_| {
                process_case(
                    "systolic",
                    sys_pair.clone(),
                    systole_output_path,
                    interpolation_steps,
                )
                .context("process_case(systolic) failed")
            });

            // Join DIASTOLIC & SYSTOLIC
            let dia_geom = dia_handle.join().unwrap()?;
            let sys_geom = sys_handle.join().unwrap()?;

            Ok((rest_pair, stress_pair, dia_geom, sys_geom))
        },
    )
    .map_err(|panic| anyhow!("Parallel processing threads panicked: {:?}", panic))?;

    let (rest_geom, stress_geom, dia_geom, sys_geom) = result?;

    Ok((rest_geom, stress_geom, dia_geom, sys_geom))
}

/// Only run the REST & STRESS threads and write their outputs.
/// Does *not* perform any comparison between them.
pub fn from_file_doublepair_rs(
    rest_input_path: &str,
    steps_best_rotation: usize,
    range_rotation_rad: f64,
    rest_output_path: &str,
    interpolation_steps: usize,
    stress_input_path: &str,
    stress_output_path: &str,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
) -> Result<(GeometryPair, GeometryPair)> {
    let result = thread::scope(|s| -> Result<(GeometryPair, GeometryPair)> {
        // REST thread
        let rest_handle = s.spawn(|_| -> Result<_> {
            let geom_rest = create_geometry_pair(
                "rest".to_string(),
                rest_input_path,
                steps_best_rotation,
                range_rotation_rad,
                image_center,
                radius,
                n_points,
                true, //hardcoded here, since only for from array can preprocess
            )
            .context("create_geometry_pair(rest) failed")?;

            let processed_rest =
                process_case("rest", geom_rest, rest_output_path, interpolation_steps)
                    .context("process_case(rest) failed")?;

            Ok(processed_rest)
        });

        // STRESS thread
        let stress_handle = s.spawn(|_| -> Result<_> {
            let geom_stress = create_geometry_pair(
                "stress".to_string(),
                stress_input_path,
                steps_best_rotation,
                range_rotation_rad,
                image_center,
                radius,
                n_points,
                true, //hardcoded here, since only for from array can preprocess
            )
            .context("create_geometry_pair(stress) failed")?;

            let processed_stress = process_case(
                "stress",
                geom_stress,
                stress_output_path,
                interpolation_steps,
            )
            .context("process_case(stress) failed")?;

            Ok(processed_stress)
        });
        // Join threads & propagate any processing errors
        let rest_geom_pair = rest_handle.join().unwrap()?;
        let stress_geom_pair = stress_handle.join().unwrap()?;

        Ok((rest_geom_pair, stress_geom_pair))
    })
    .map_err(|panic_payload| {
        anyhow!("Parallel processing threads panicked: {:?}", panic_payload)
    })?;

    let (rest_geom, stress_geom) = result?;
    Ok((rest_geom, stress_geom))
}

pub fn from_file_singlepair_rs(
    input_path: &str,
    steps_best_rotation: usize,
    range_rotation_rad: f64,
    output_path: &str,
    interpolation_steps: usize,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
) -> Result<GeometryPair> {
    // Build the raw pair
    let geom_pair = create_geometry_pair(
        "single".to_string(),
        input_path,
        steps_best_rotation,
        range_rotation_rad,
        image_center,
        radius,
        n_points,
        true, //hardcoded here, since only for from array can preprocess
    )
    .context("create_geometry_pair(single) failed")?;

    // Process it (e.g. align, interpolate, write meshes)
    let processed_pair = process_case("single", geom_pair, output_path, interpolation_steps)
        .context("process_case(single) failed")?;

    Ok(processed_pair)
}

pub fn from_file_single_rs(
    input_path: &str,
    steps_best_rotation: usize,
    range_rotation_rad: f64,
    output_path: &str,
    diastole: bool,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
) -> Result<Geometry> {
    let mut geom = Geometry::new(
        input_path,
        "single".to_string(),
        diastole,
        image_center,
        radius,
        n_points,
    )?;

    geom = align_frames_in_geometry(geom, steps_best_rotation, range_rotation_rad);

    let filename = format!("{}/mesh_000_single.obj", output_path);

    write_obj_mesh_without_uv(&geom.contours, &filename, "mesh_000_single.mtl")?;

    Ok(geom)
}
