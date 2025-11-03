use anyhow::{anyhow, Context, Result};
use crossbeam::thread;
use crate::intravascular::io::input::InputData;
use crate::intravascular::io::geometry::Geometry;
use crate::intravascular::neo_processing::preprocessing::{ProcessingOptions, prepare_n_geometries};
use crate::intravascular::neo_processing::align_within::{AlignLog, align_frames_in_geometry};
use crate::intravascular::neo_processing::align_between::{GeometryPair, align_between_geometries};
use std::path::Path;

pub fn full_processing_rs(
    label: String,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
    input_path_a: Option<&str>,
    input_path_b: Option<&str>,
    input_data_a: Option<InputData>,
    input_data_b: Option<InputData>,
    input_data_c: Option<InputData>,
    input_data_d: Option<InputData>,
    output_path: &str,
    step_deg: f64,
    range_deg: f64,
    smooth: bool,
    bruteforce: bool,
    sample_size: usize,
) -> Result<()> {
    let mut geometries = prepare_n_geometries(
        &label,
        image_center,
        radius,
        n_points,
        Some(vec![input_data_a, input_data_b, input_data_c, input_data_d]
            .into_iter()
            .flatten()
            .collect()),
        input_path_a.map(Path::new),
        input_path_b.map(Path::new),
        ProcessingOptions::Full,
    ).context("Failed to prepare geometries for full processing")?;

    if geometries.len() != 4 {
        return Err(anyhow!("Full processing requires exactly 4 geometries, got {}", geometries.len()));
    }

    let [mut geom_a, mut geom_b, mut geom_c, mut geom_d] = geometries.as_mut_slice() else {
        return Err(anyhow!("Failed to destructure geometries"));
    };

    let (mut geom_a, mut geom_b, mut geom_c, mut geom_d, logs_a, logs_b, logs_c, logs_d) = thread::scope(|s| -> Result<(
        Geometry,
        Geometry,
        Geometry,
        Geometry,
        Vec<AlignLog>,
        Vec<AlignLog>,
        Vec<AlignLog>,
        Vec<AlignLog>,
    )> {
        let geom_a_handle = s.spawn(|_| -> anyhow::Result<_> {
            let (geom, logs) = align_frames_in_geometry(
                &mut geom_a,
                step_deg,
                range_deg,
                smooth,
                bruteforce,
                sample_size,
            )
            .context("Failed to align frames within geometry A")?;
            Ok((geom, logs))
        });

        let geom_b_handle = s.spawn(|_| -> anyhow::Result<_> {
            let (geom, logs) = align_frames_in_geometry(
                &mut geom_b,
                step_deg,
                range_deg,
                smooth,
                bruteforce,
                sample_size,
            ).context("Failed to align frames within geometry B")?;
            Ok((geom, logs))    
        });

        let geom_c_handle = s.spawn(|_| -> anyhow::Result<_> {
            let (geom, logs) = align_frames_in_geometry(
                &mut geom_c,
                step_deg,
                range_deg,
                smooth,
                bruteforce,
                sample_size,
            ).context("Failed to align frames within geometry C")?;
            Ok((geom, logs))    
        });

        let geom_d_handle = s.spawn(|_| -> anyhow::Result<_> {
            let (geom, logs) = align_frames_in_geometry(
                &mut geom_d,
                step_deg,
                range_deg,
                smooth,
                bruteforce,
                sample_size,
            ).context("Failed to align frames within geometry D")?;
            Ok((geom, logs))    
        });

        let (geom_a, logs_a) = geom_a_handle.join().unwrap()?;
        let (geom_b, logs_b) = geom_b_handle.join().unwrap()?;
        let (geom_c, logs_c) = geom_c_handle.join().unwrap()?;
        let (geom_d, logs_d) = geom_d_handle.join().unwrap()?;

        Ok((geom_a, geom_b, geom_c, geom_d, logs_a, logs_b, logs_c, logs_d))
    }).map_err(|e| anyhow!("Thread execution failed: {:?}", e))??;

    // First parallel batch: AB and CD (independent pairs)
    let (geom_pair_ab, geom_pair_cd) = thread::scope(|s| -> Result<(GeometryPair, GeometryPair)> {
        let geompair_ab_handle = s.spawn(|_| -> anyhow::Result<_> {
            align_between_geometries(
                &mut geom_a,
                &mut geom_b,
                range_deg,
                step_deg,
                sample_size,
            ).context(format!("Failed to align frames between geometry {} and {}", geom_a.label, geom_b.label))
        });

        let geompair_cd_handle = s.spawn(|_| -> anyhow::Result<_> {
            align_between_geometries(
                &mut geom_c,
                &mut geom_d,
                range_deg,
                step_deg,
                sample_size,
            ).context(format!("Failed to align frames between geometry {} and {}", geom_c.label, geom_d.label))
        });

        let geom_pair_ab = geompair_ab_handle.join().unwrap()?;
        let geom_pair_cd = geompair_cd_handle.join().unwrap()?;
        
        Ok((geom_pair_ab, geom_pair_cd))
    }).map_err(|e| anyhow!("Thread execution failed: {:?}", e))??;

    // Second parallel batch: AC and BD (independent pairs)
    let (geom_pair_ac, geom_pair_bd) = thread::scope(|s| -> Result<(GeometryPair, GeometryPair)> {
        let geompair_ac_handle = s.spawn(|_| -> anyhow::Result<_> {
            align_between_geometries(
                &mut geom_a,
                &mut geom_c,
                range_deg,
                step_deg,
                sample_size,
            ).context(format!("Failed to align frames between geometry {} and {}", geom_a.label, geom_c.label))
        });

        let geompair_bd_handle = s.spawn(|_| -> anyhow::Result<_> {
            align_between_geometries(
                &mut geom_b,
                &mut geom_d,
                range_deg,
                step_deg,
                sample_size,
            ).context(format!("Failed to align frames between geometry {} and {}", geom_b.label, geom_d.label))
        });

        let geom_pair_ac = geompair_ac_handle.join().unwrap()?;
        let geom_pair_bd = geompair_bd_handle.join().unwrap()?;
        
        Ok((geom_pair_ac, geom_pair_bd))
    }).map_err(|e| anyhow!("Thread execution failed: {:?}", e))??;
    
    // TODO: postprocessing

    // TODO: writing to obj 

    Ok(())
}