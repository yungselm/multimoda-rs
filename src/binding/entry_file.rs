use anyhow::{anyhow, Context, Result};
use crossbeam::thread;

use crate::processing::align_within_and_between;
use crate::processing::resampling::prepare_geometries_comparison;
use crate::processing::process_utils::process_case;
use crate::processing::align_between::GeometryPair;
use crate::processing::align_within::AlignLog;

use crate::io::output::write_obj_mesh_without_uv;
use crate::io::Geometry;

pub fn from_file_full_rs(
    rest_input_path: &str,
    stress_input_path: &str,
    step_rotation_deg: f64,
    range_rotation_deg: f64,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
    write_obj: bool,
    rest_output_path: &str,
    stress_output_path: &str,
    diastole_output_path: &str,
    systole_output_path: &str,
    interpolation_steps: usize,
    bruteforce: bool,
    sample_size: usize,
) -> anyhow::Result<(
    (GeometryPair, GeometryPair, GeometryPair, GeometryPair),
    (Vec<AlignLog>, Vec<AlignLog>, Vec<AlignLog>, Vec<AlignLog>),
)> {
    let result = thread::scope(
        |s| -> Result<(
            GeometryPair,
            GeometryPair,
            GeometryPair,
            GeometryPair,
            Vec<AlignLog>,
            Vec<AlignLog>,
            Vec<AlignLog>,
            Vec<AlignLog>,
        )> {
            // REST thread
            let rest_handle = s.spawn(|_| -> anyhow::Result<_> {
                let (geom, dia_logs, sys_logs) = align_within_and_between(
                    "rest",
                    rest_input_path,
                    step_rotation_deg,
                    range_rotation_deg,
                    image_center,
                    radius,
                    n_points,
                    write_obj,
                    rest_output_path,
                    interpolation_steps,
                    bruteforce,
                    sample_size,
                )
                .context("process geometry pair(rest) failed")?;
                Ok((geom, dia_logs, sys_logs))
            });

            // STRESS thread
            let stress_handle = s.spawn(|_| -> anyhow::Result<_> {
                let (geom_stress, dia_logs_stress, sys_logs_stress) = align_within_and_between(
                    "stress",
                    stress_input_path,
                    step_rotation_deg,
                    range_rotation_deg,
                    image_center,
                    radius,
                    n_points,
                    write_obj,
                    stress_output_path,
                    interpolation_steps,
                    bruteforce,
                    sample_size,
                )
                .context("process geometry pair(stress) failed")?;
                Ok((geom_stress, dia_logs_stress, sys_logs_stress))
            });

            // Join REST & STRESS
            let (rest_pair, dia_logs, sys_logs) = rest_handle.join().unwrap()?;
            let (stress_pair, dia_logs_stress, sys_logs_stress) = stress_handle.join().unwrap()?;

            // Prepare diastolic & systolic geometry pairs
            let (dia_pair, sys_pair) =
                prepare_geometries_comparison(
                    rest_pair.clone(), 
                    stress_pair.clone(),
                    step_rotation_deg,
                    range_rotation_deg);
            let dia_pair_for_thread = dia_pair.clone();
            let sys_pair_for_thread = sys_pair.clone();

            // DIASTOLIC thread
            let dia_handle = if write_obj {
                Some(s.spawn(move |_| {
                    process_case(
                        "diastolic",
                        dia_pair_for_thread,
                        diastole_output_path,
                        interpolation_steps,
                    )
                    .context("process_case(diastolic) failed")
                }))
            } else {
                None
            };

            // SYSTOLIC thread
            let sys_handle = if write_obj {
                Some(s.spawn(move |_| {
                    process_case(
                        "systolic",
                        sys_pair_for_thread,
                        systole_output_path,
                        interpolation_steps,
                    )
                    .context("process_case(systolic) failed")
                }))
            } else {
                None
            };

            // Join DIASTOLIC & SYSTOLIC
            let dia_geom = if let Some(handle) = dia_handle {
                handle.join().unwrap()?
            } else {
                dia_pair // fallback: return unprocessed pair
            };
            let sys_geom = if let Some(handle) = sys_handle {
                handle.join().unwrap()?
            } else {
                sys_pair // fallback: return unprocessed pair
            };

            Ok((
                rest_pair,
                stress_pair,
                dia_geom,
                sys_geom,
                dia_logs,
                sys_logs,
                dia_logs_stress,
                sys_logs_stress,
            ))
        },
    )
    .map_err(|panic| anyhow!("Parallel processing threads panicked: {:?}", panic))?;

    let (
        rest_geom,
        stress_geom,
        dia_geom,
        sys_geom,
        dia_logs,
        sys_logs,
        dia_logs_stress,
        sys_logs_stress,
    ) = result?;

    Ok((
        (rest_geom, stress_geom, dia_geom, sys_geom),
        (dia_logs, sys_logs, dia_logs_stress, sys_logs_stress),
    ))
}

/// Only run the REST & STRESS threads and write their outputs.
/// Does *not* perform any comparison between them.
pub fn from_file_doublepair_rs(
    rest_input_path: &str,
    stress_input_path: &str,
    step_rotation_deg: f64,
    range_rotation_deg: f64,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
    write_obj: bool,
    rest_output_path: &str,
    stress_output_path: &str,
    interpolation_steps: usize,
    bruteforce: bool,
    sample_size: usize,
) -> anyhow::Result<(
    (GeometryPair, GeometryPair),
    (Vec<AlignLog>, Vec<AlignLog>, Vec<AlignLog>, Vec<AlignLog>),
)> {
    let result = thread::scope(
        |s| -> anyhow::Result<(
            GeometryPair,
            GeometryPair,
            Vec<AlignLog>,
            Vec<AlignLog>,
            Vec<AlignLog>,
            Vec<AlignLog>,
        )> {
            // REST thread
            let rest_handle = s.spawn(|_| -> anyhow::Result<_> {
                let (geom, dia_logs, sys_logs) = align_within_and_between(
                    "rest",
                    rest_input_path,
                    step_rotation_deg,
                    range_rotation_deg,
                    image_center,
                    radius,
                    n_points,
                    write_obj,
                    rest_output_path,
                    interpolation_steps,
                    bruteforce,
                    sample_size,
                )
                .context("process geometry pair(rest) failed")?;
                Ok((geom, dia_logs, sys_logs))
            });

            // STRESS thread
            let stress_handle = s.spawn(|_| -> anyhow::Result<_> {
                let (geom_stress, dia_logs_stress, sys_logs_stress) = align_within_and_between(
                    "stress",
                    stress_input_path,
                    step_rotation_deg,
                    range_rotation_deg,
                    image_center,
                    radius,
                    n_points,
                    write_obj,
                    stress_output_path,
                    interpolation_steps,
                    bruteforce,
                    sample_size,
                )
                .context("process geometry pair(stress) failed")?;
                Ok((geom_stress, dia_logs_stress, sys_logs_stress))
            });
            // Join threads & propagate any processing errors
            let (rest_geom_pair, dia_logs, sys_logs) = rest_handle.join().unwrap()?;
            let (stress_geom_pair, dia_logs_stress, sys_logs_stress) =
                stress_handle.join().unwrap()?;

            Ok((
                rest_geom_pair,
                stress_geom_pair,
                dia_logs,
                sys_logs,
                dia_logs_stress,
                sys_logs_stress,
            ))
        },
    )
    .map_err(|panic_payload| {
        anyhow!("Parallel processing threads panicked: {:?}", panic_payload)
    })?;

    let (rest_geom, stress_geom, dia_logs, sys_logs, dia_logs_stress, sys_logs_stress) = result?;
    Ok((
        (rest_geom, stress_geom),
        (dia_logs, sys_logs, dia_logs_stress, sys_logs_stress),
    ))
}

pub fn from_file_singlepair_rs(
    input_path: &str,
    step_rotation_deg: f64,
    range_rotation_deg: f64,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
    write_obj: bool,
    output_path: &str,
    interpolation_steps: usize,
    bruteforce: bool,
    sample_size: usize,
) -> anyhow::Result<(GeometryPair, (Vec<AlignLog>, Vec<AlignLog>))> {
    // Build the raw pair
    let (geom_pair, dia_logs, sys_logs) = align_within_and_between(
        "single",
        input_path,
        step_rotation_deg,
        range_rotation_deg,
        image_center,
        radius,
        n_points,
        write_obj,
        output_path,
        interpolation_steps,
        bruteforce,
        sample_size,
    )
    .context("process geometry_pair(single) failed")?;

    Ok((geom_pair, (dia_logs, sys_logs)))
}

pub fn from_file_single_rs(
    input_path: &str,
    step_rotation_deg: f64,
    range_rotation_deg: f64,
    diastole: bool,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
    write_obj: bool,
    output_path: &str,
    bruteforce: bool,
    sample_size: usize,
) -> Result<(Geometry, Vec<AlignLog>)> {
    let geom = Geometry::new(
        input_path,
        "single".to_string(),
        diastole,
        image_center,
        radius,
        n_points,
    )?;

    let (geom, logs) = crate::processing::align_within::align_frames_in_geometry(
        geom, 
        step_rotation_deg, 
        range_rotation_deg,
    true,
        bruteforce,
        sample_size);
    let geom = if geom.walls.is_empty() {
        crate::processing::walls::create_wall_geometry(&geom, /*with_pulmonary=*/ false)
    } else {
        geom
    };
    if write_obj {
        let filename = format!("{}/mesh_000_single.obj", output_path);
    
        write_obj_mesh_without_uv(&geom.contours, &filename, "mesh_000_single.mtl")?;
    }

    Ok((geom, logs))
}
