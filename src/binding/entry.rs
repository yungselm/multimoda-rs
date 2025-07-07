use anyhow::{Result, Context, anyhow};
use crossbeam::thread;

use crate::processing::comparison::prepare_geometries_comparison;
use crate::processing::process_case::{create_geometry_pair, process_case};

use crate::processing::geometries::GeometryPair;

pub fn run_process_case(
    rest_input_path: &str, 
    steps_best_rotation: usize, 
    range_rotation_rad: f64,
    rest_output_path: &str,
    interpolation_steps: usize,
    stress_input_path: &str,
    stress_output_path: &str,
    diastole_comparison_path: &str,
    systole_comparison_path: &str,
) -> Result<(GeometryPair, GeometryPair, GeometryPair, GeometryPair)> {
    // Chain directly on thread::scope without a stray semicolon
    let result = thread::scope(|s| -> Result<(GeometryPair, GeometryPair, GeometryPair, GeometryPair)> {
        // REST thread
        let rest_handle = s.spawn(|_| -> Result<GeometryPair> {
            let geom_rest = create_geometry_pair(
                "rest".to_string(),
                rest_input_path,
                steps_best_rotation,
                range_rotation_rad,
            )
            .context("create_geometry_pair(rest) failed")?;
            
            let processed_rest = process_case(
                "rest",
                geom_rest,
                rest_output_path,
                interpolation_steps,
            )
            .context("process_case(rest) failed")?;
            
            Ok(processed_rest)
        });

        // STRESS thread
        let stress_handle = s.spawn(|_| -> Result<GeometryPair> {
            let geom_stress = create_geometry_pair(
                "stress".to_string(),
                stress_input_path,
                steps_best_rotation,
                range_rotation_rad,
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
        let rest_geom_pair   = rest_handle.join().unwrap()?;
        let stress_geom_pair = stress_handle.join().unwrap()?;

        // Diastole / systole comparison
        let (dia_geom_pair, sys_geom_pair) =
            prepare_geometries_comparison(rest_geom_pair.clone(), stress_geom_pair.clone());

        let dia_geom_clone = dia_geom_pair.clone();
        let sys_geom_clone = sys_geom_pair.clone();

        process_case(
            "diastolic",
            dia_geom_pair,
            diastole_comparison_path,
            interpolation_steps,
        )
        .context("process_case(diastolic) failed")?;

        process_case(
            "systolic",
            sys_geom_pair,
            systole_comparison_path,
            interpolation_steps,
        )
        .context("process_case(systolic) failed")?;

        Ok((rest_geom_pair, stress_geom_pair, dia_geom_clone, sys_geom_clone))
    })
    .map_err(|panic_payload| {
        anyhow!("Parallel processing threads panicked: {:?}", panic_payload)
    })?;

    let (rest_geom, stress_geom, dia_geom, sys_geom) = result?;

    Ok((rest_geom, stress_geom, dia_geom, sys_geom))
}


/// Only run the REST & STRESS threads and write their outputs.
/// Does *not* perform any comparison between them.
pub fn run_rest_stress_only(
    rest_input_path: &str,
    steps_best_rotation: usize,
    range_rotation_rad: f64,
    rest_output_path: &str,
    interpolation_steps: usize,
    stress_input_path: &str,
    stress_output_path: &str,
) -> Result<(GeometryPair, GeometryPair)> {
    // Chain directly on thread::scope without a stray semicolon
    let result = thread::scope(|s| -> Result<(GeometryPair, GeometryPair)> {
        // REST thread
        let rest_handle = s.spawn(|_| -> Result<_> {
            let geom_rest = create_geometry_pair(
                "rest".to_string(),
                rest_input_path,
                steps_best_rotation,
                range_rotation_rad,
            )
            .context("create_geometry_pair(rest) failed")?;
            
            let processed_rest = process_case(
                "rest",
                geom_rest,
                rest_output_path,
                interpolation_steps,
            )
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
        let rest_geom_pair   = rest_handle.join().unwrap()?;
        let stress_geom_pair = stress_handle.join().unwrap()?;
        
        Ok((rest_geom_pair, stress_geom_pair))
    })
    .map_err(|panic_payload| {
        anyhow!("Parallel processing threads panicked: {:?}", panic_payload)
    })?;

    let (rest_geom, stress_geom) = result?;
    Ok((rest_geom, stress_geom))
}