use anyhow::{anyhow, Context, Result};
use crossbeam::thread;
use std::path::Path;

use crate::intravascular::io::geometry::{ContourType, Geometry};
use crate::intravascular::io::input::InputData;
use crate::intravascular::processing::preprocessing::{
    prepare_n_geometries, ProcessingOptions,
};
use crate::intravascular::processing::align_within::{align_frames_in_geometry, AlignLog};
use crate::intravascular::processing::align_between::{align_between_geometries, GeometryPair};
use crate::intravascular::processing::postprocessing::postprocess_geom_pair;
use crate::intravascular::to_object::process_case;

// tolerance of distance between frames [mm], that counts as 0
const TOLERANCE: f64 = 0.03;

fn maybe_postprocess(
    pair: &GeometryPair,
    tolerance: f64,
    anomalous: bool,
    postprocessing: bool,
) -> anyhow::Result<GeometryPair> {
    if postprocessing {
        postprocess_geom_pair(pair, tolerance, anomalous)
            .with_context(|| format!("Failed postprocessing of {}", pair.label))
    } else {
        Ok(pair.clone())
    }
}

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
    write_obj: bool,
    interpolation_steps: usize,
    contour_types: Vec<ContourType>,
    watertight: bool,
    output_path_a: &str,
    output_path_b: &str,
    output_path_c: &str,
    output_path_d: &str,
    step_deg: f64,
    range_deg: f64,
    smooth: bool,
    bruteforce: bool,
    sample_size: usize,
    postprocessing: bool,
) -> Result<(GeometryPair, GeometryPair, GeometryPair, GeometryPair, Vec<AlignLog>, Vec<AlignLog>, Vec<AlignLog>, Vec<AlignLog>)> {
    let mut geometries = prepare_n_geometries(
        &label,
        image_center,
        radius,
        n_points,
        Some(
            vec![input_data_a, input_data_b, input_data_c, input_data_d]
                .into_iter()
                .flatten()
                .collect(),
        ),
        input_path_a.map(Path::new),
        input_path_b.map(Path::new),
        ProcessingOptions::Full,
    )
    .context("Failed to prepare geometries for full processing")?;

    if geometries.len() != 4 {
        return Err(anyhow!(
            "Full processing requires exactly 4 geometries, got {}",
            geometries.len()
        ));
    }

    let mut geom_a = geometries.remove(0);
    let mut geom_b = geometries.remove(0);
    let mut geom_c = geometries.remove(0);
    let mut geom_d = geometries.remove(0);

    let (mut geom_a, mut geom_b, mut geom_c, mut geom_d, logs_a, logs_b, logs_c, logs_d, bool_a, bool_b, bool_c, bool_d) =
        thread::scope(
            |s| -> Result<(
                Geometry,
                Geometry,
                Geometry,
                Geometry,
                Vec<AlignLog>,
                Vec<AlignLog>,
                Vec<AlignLog>,
                Vec<AlignLog>,
                bool,
                bool,
                bool,
                bool,
            )> {
                let geom_a_handle = s.spawn(|_| -> anyhow::Result<_> {
                    let (geom, logs, anomalous_bool) = align_frames_in_geometry(
                        &mut geom_a,
                        step_deg,
                        range_deg,
                        smooth,
                        bruteforce,
                        sample_size,
                    )
                    .context("Failed to align frames within geometry A")?;
                    Ok((geom, logs, anomalous_bool))
                });

                let geom_b_handle = s.spawn(|_| -> anyhow::Result<_> {
                    let (geom, logs, anomalous_bool) = align_frames_in_geometry(
                        &mut geom_b,
                        step_deg,
                        range_deg,
                        smooth,
                        bruteforce,
                        sample_size,
                    )
                    .context("Failed to align frames within geometry B")?;
                    Ok((geom, logs, anomalous_bool))
                });

                let geom_c_handle = s.spawn(|_| -> anyhow::Result<_> {
                    let (geom, logs, anomalous_bool) = align_frames_in_geometry(
                        &mut geom_c,
                        step_deg,
                        range_deg,
                        smooth,
                        bruteforce,
                        sample_size,
                    )
                    .context("Failed to align frames within geometry C")?;
                    Ok((geom, logs, anomalous_bool))
                });

                let geom_d_handle = s.spawn(|_| -> anyhow::Result<_> {
                    let (geom, logs, anomalous_bool) = align_frames_in_geometry(
                        &mut geom_d,
                        step_deg,
                        range_deg,
                        smooth,
                        bruteforce,
                        sample_size,
                    )
                    .context("Failed to align frames within geometry D")?;
                    Ok((geom, logs, anomalous_bool))
                });

                let (geom_a, logs_a, bool_a) = geom_a_handle.join().unwrap()?;
                let (geom_b, logs_b, bool_b) = geom_b_handle.join().unwrap()?;
                let (geom_c, logs_c, bool_c) = geom_c_handle.join().unwrap()?;
                let (geom_d, logs_d, bool_d) = geom_d_handle.join().unwrap()?;

                Ok((
                    geom_a, geom_b, geom_c, geom_d, logs_a, logs_b, logs_c, logs_d, bool_a, bool_b, bool_c, bool_d,
                ))
            },
        )
        .map_err(|e| anyhow!("Thread execution failed: {:?}", e))??;

    // First parallel batch: AB and CD (independent pairs)
    let (geom_pair_ab, geom_pair_cd) = thread::scope(|s| -> Result<(GeometryPair, GeometryPair)> {
        let geompair_ab_handle = s.spawn(|_| -> anyhow::Result<_> {
            align_between_geometries(&mut geom_a, &mut geom_b, range_deg, step_deg, sample_size)
                .context(format!(
                    "Failed to align frames between geometry {} and {}",
                    geom_a.label, geom_b.label
                ))
        });

        let geompair_cd_handle = s.spawn(|_| -> anyhow::Result<_> {
            align_between_geometries(&mut geom_c, &mut geom_d, range_deg, step_deg, sample_size)
                .context(format!(
                    "Failed to align frames between geometry {} and {}",
                    geom_c.label, geom_d.label
                ))
        });

        let geom_pair_ab = geompair_ab_handle.join().unwrap()?;
        let geom_pair_cd = geompair_cd_handle.join().unwrap()?;

        Ok((geom_pair_ab, geom_pair_cd))
    })
    .map_err(|e| anyhow!("Thread execution failed: {:?}", e))??;

    // Second parallel batch: AC and BD (independent pairs)
    let (geom_pair_ac, geom_pair_bd) = thread::scope(|s| -> Result<(GeometryPair, GeometryPair)> {
        let geompair_ac_handle = s.spawn(|_| -> anyhow::Result<_> {
            align_between_geometries(&mut geom_a, &mut geom_c, range_deg, step_deg, sample_size)
                .context(format!(
                    "Failed to align frames between geometry {} and {}",
                    geom_a.label, geom_c.label
                ))
        });

        let geompair_bd_handle = s.spawn(|_| -> anyhow::Result<_> {
            align_between_geometries(&mut geom_b, &mut geom_d, range_deg, step_deg, sample_size)
                .context(format!(
                    "Failed to align frames between geometry {} and {}",
                    geom_b.label, geom_d.label
                ))
        });

        let geom_pair_ac = geompair_ac_handle.join().unwrap()?;
        let geom_pair_bd = geompair_bd_handle.join().unwrap()?;

        Ok((geom_pair_ac, geom_pair_bd))
    })
    .map_err(|e| anyhow!("Thread execution failed: {:?}", e))??;

    // safety, if any pair is anomalous try to build wall with aortic thickness, fallback anyways just wall 1mm offset.
    let anomalous = bool_a || bool_b || bool_c || bool_d;

    let geom_ab_postprocessed = maybe_postprocess(&geom_pair_ab, TOLERANCE, anomalous, postprocessing)?;
    let geom_cd_postprocessed= maybe_postprocess(&geom_pair_cd, TOLERANCE, anomalous, postprocessing)?;
    let geom_ac_postprocessed = maybe_postprocess(&geom_pair_ac, TOLERANCE, anomalous, postprocessing)?;
    let geom_bd_postprocessed= maybe_postprocess(&geom_pair_bd, TOLERANCE, anomalous, postprocessing)?;

    let geom_ab_final = if write_obj {
        process_case(&label, geom_ab_postprocessed, output_path_a, interpolation_steps, watertight, &contour_types)
            .context("process case failed for geom_ab")?
    } else {
        geom_ab_postprocessed
    };

    let geom_cd_final = if write_obj {
        process_case(&label, geom_cd_postprocessed, output_path_b, interpolation_steps, watertight, &contour_types)
            .context("process case failed for geom_cd")?
    } else {
        geom_cd_postprocessed
    };

    let geom_ac_final = if write_obj {
        process_case(&label, geom_ac_postprocessed, output_path_c, interpolation_steps, watertight, &contour_types)
            .context("process case failed for geom_ac")?
    } else {
        geom_ac_postprocessed
    };

    let geom_bd_final = if write_obj {
        process_case(&label, geom_bd_postprocessed, output_path_d, interpolation_steps, watertight, &contour_types)
            .context("process case failed for geom_bd")?
    } else {
        geom_bd_postprocessed
    };

    Ok((geom_ab_final, geom_cd_final, geom_ac_final, geom_bd_final, logs_a, logs_b, logs_c, logs_d))
}

pub fn double_pair_processing_rs(
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
    write_obj: bool,
    interpolation_steps: usize,
    contour_types: Vec<ContourType>,
    watertight: bool,
    output_path_a: &str,
    output_path_b: &str,
    step_deg: f64,
    range_deg: f64,
    smooth: bool,
    bruteforce: bool,
    sample_size: usize,
    postprocessing: bool,
) -> Result<(GeometryPair, GeometryPair, Vec<AlignLog>, Vec<AlignLog>, Vec<AlignLog>, Vec<AlignLog>)> {
    let mut geometries = prepare_n_geometries(
        &label,
        image_center,
        radius,
        n_points,
        Some(
            vec![input_data_a, input_data_b, input_data_c, input_data_d]
                .into_iter()
                .flatten()
                .collect(),
        ),
        input_path_a.map(Path::new),
        input_path_b.map(Path::new),
        ProcessingOptions::Full,
    )
    .context("Failed to prepare geometries for full processing")?;

    if geometries.len() != 4 {
        return Err(anyhow!(
            "Double Pair processing requires exactly 4 geometries, got {}",
            geometries.len()
        ));
    }

    let mut geom_a = geometries.remove(0);
    let mut geom_b = geometries.remove(0);
    let mut geom_c = geometries.remove(0);
    let mut geom_d = geometries.remove(0);

    let (mut geom_a, mut geom_b, mut geom_c, mut geom_d, logs_a, logs_b, logs_c, logs_d, bool_a, bool_b, bool_c, bool_d) =
        thread::scope(
            |s| -> Result<(
                Geometry,
                Geometry,
                Geometry,
                Geometry,
                Vec<AlignLog>,
                Vec<AlignLog>,
                Vec<AlignLog>,
                Vec<AlignLog>,
                bool,
                bool,
                bool,
                bool,
            )> {
                let geom_a_handle = s.spawn(|_| -> anyhow::Result<_> {
                    let (geom, logs, anomalous_bool) = align_frames_in_geometry(
                        &mut geom_a,
                        step_deg,
                        range_deg,
                        smooth,
                        bruteforce,
                        sample_size,
                    )
                    .context("Failed to align frames within geometry A")?;
                    Ok((geom, logs, anomalous_bool))
                });

                let geom_b_handle = s.spawn(|_| -> anyhow::Result<_> {
                    let (geom, logs, anomalous_bool) = align_frames_in_geometry(
                        &mut geom_b,
                        step_deg,
                        range_deg,
                        smooth,
                        bruteforce,
                        sample_size,
                    )
                    .context("Failed to align frames within geometry B")?;
                    Ok((geom, logs, anomalous_bool))
                });

                let geom_c_handle = s.spawn(|_| -> anyhow::Result<_> {
                    let (geom, logs, anomalous_bool) = align_frames_in_geometry(
                        &mut geom_c,
                        step_deg,
                        range_deg,
                        smooth,
                        bruteforce,
                        sample_size,
                    )
                    .context("Failed to align frames within geometry C")?;
                    Ok((geom, logs, anomalous_bool))
                });

                let geom_d_handle = s.spawn(|_| -> anyhow::Result<_> {
                    let (geom, logs, anomalous_bool) = align_frames_in_geometry(
                        &mut geom_d,
                        step_deg,
                        range_deg,
                        smooth,
                        bruteforce,
                        sample_size,
                    )
                    .context("Failed to align frames within geometry D")?;
                    Ok((geom, logs, anomalous_bool))
                });

                let (geom_a, logs_a, bool_a) = geom_a_handle.join().unwrap()?;
                let (geom_b, logs_b, bool_b) = geom_b_handle.join().unwrap()?;
                let (geom_c, logs_c, bool_c) = geom_c_handle.join().unwrap()?;
                let (geom_d, logs_d, bool_d) = geom_d_handle.join().unwrap()?;

                Ok((
                    geom_a, geom_b, geom_c, geom_d, logs_a, logs_b, logs_c, logs_d, bool_a, bool_b, bool_c, bool_d,
                ))
            },
        )
        .map_err(|e| anyhow!("Thread execution failed: {:?}", e))??;

    // First parallel batch: AB and CD (independent pairs)
    let (geom_pair_ab, geom_pair_cd) = thread::scope(|s| -> Result<(GeometryPair, GeometryPair)> {
        let geompair_ab_handle = s.spawn(|_| -> anyhow::Result<_> {
            align_between_geometries(&mut geom_a, &mut geom_b, range_deg, step_deg, sample_size)
                .context(format!(
                    "Failed to align frames between geometry {} and {}",
                    geom_a.label, geom_b.label
                ))
        });

        let geompair_cd_handle = s.spawn(|_| -> anyhow::Result<_> {
            align_between_geometries(&mut geom_c, &mut geom_d, range_deg, step_deg, sample_size)
                .context(format!(
                    "Failed to align frames between geometry {} and {}",
                    geom_c.label, geom_d.label
                ))
        });

        let geom_pair_ab = geompair_ab_handle.join().unwrap()?;
        let geom_pair_cd = geompair_cd_handle.join().unwrap()?;

        Ok((geom_pair_ab, geom_pair_cd))
    })
    .map_err(|e| anyhow!("Thread execution failed: {:?}", e))??;

    let anomalous = bool_a || bool_b || bool_c || bool_d;

    let geom_ab_postprocessed = maybe_postprocess(&geom_pair_ab, TOLERANCE, anomalous, postprocessing)?;
    let geom_cd_postprocessed= maybe_postprocess(&geom_pair_cd, TOLERANCE, anomalous, postprocessing)?;

    let geom_ab_final = if write_obj {
        process_case(&label, geom_ab_postprocessed, output_path_a, interpolation_steps, watertight, &contour_types)
            .context("process case failed for geom_ab")?
    } else {
        geom_ab_postprocessed
    };

    let geom_cd_final = if write_obj {
        process_case(&label, geom_cd_postprocessed, output_path_b, interpolation_steps, watertight, &contour_types)
            .context("process case failed for geom_cd")?
    } else {
        geom_cd_postprocessed
    };

    Ok((geom_ab_final, geom_cd_final, logs_a, logs_b, logs_c, logs_d))
}

pub fn pair_processing_rs(
    label: String,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
    input_path_a: Option<&str>,
    input_data_a: Option<InputData>,
    input_data_b: Option<InputData>,
    write_obj: bool,
    interpolation_steps: usize,
    contour_types: Vec<ContourType>,
    watertight: bool,
    output_path: &str,
    step_deg: f64,
    range_deg: f64,
    smooth: bool,
    bruteforce: bool,
    sample_size: usize,
    postprocessing: bool,
) -> Result<(GeometryPair, Vec<AlignLog>, Vec<AlignLog>)> {
    let mut geometries = prepare_n_geometries(
        &label,
        image_center,
        radius,
        n_points,
        Some(
            vec![input_data_a, input_data_b]
                .into_iter()
                .flatten()
                .collect(),
        ),
        input_path_a.map(Path::new),
        None,
        ProcessingOptions::Pair,
    )
    .context("Failed to prepare geometries for full processing")?;

    if geometries.len() != 2 {
        return Err(anyhow!(
            "Single Pair processing requires exactly 2 geometries, got {}",
            geometries.len()
        ));
    }

    let mut geom_a = geometries.remove(0);
    let mut geom_b = geometries.remove(0);

    let (mut geom_a, mut geom_b, logs_a, logs_b, bool_a, bool_b) = thread::scope(
        |s| -> Result<(Geometry, Geometry, Vec<AlignLog>, Vec<AlignLog>, bool, bool)> {
            let geom_a_handle = s.spawn(|_| -> anyhow::Result<_> {
                let (geom, logs, anomalous_bool) = align_frames_in_geometry(
                    &mut geom_a,
                    step_deg,
                    range_deg,
                    smooth,
                    bruteforce,
                    sample_size,
                )
                .context("Failed to align frames within geometry A")?;
                Ok((geom, logs, anomalous_bool))
            });

            let geom_b_handle = s.spawn(|_| -> anyhow::Result<_> {
                let (geom, logs, anomalous_bool) = align_frames_in_geometry(
                    &mut geom_b,
                    step_deg,
                    range_deg,
                    smooth,
                    bruteforce,
                    sample_size,
                )
                .context("Failed to align frames within geometry B")?;
                Ok((geom, logs, anomalous_bool))
            });

            let (geom_a, logs_a, bool_a) = geom_a_handle.join().unwrap()?;
            let (geom_b, logs_b, bool_b) = geom_b_handle.join().unwrap()?;

            Ok((geom_a, geom_b, logs_a, logs_b, bool_a, bool_b))
        },
    )
    .map_err(|e| anyhow!("Thread execution failed: {:?}", e))??;

    let geom_pair =
        align_between_geometries(&mut geom_a, &mut geom_b, range_deg, step_deg, sample_size)
            .context(format!(
                "Failed to align frames between geometry {} and {}",
                geom_a.label, geom_b.label
            )).context("Failed to align geom_a and geom_b")?;

    let anomalous = bool_a || bool_b;

    let geom_pair_postprocessed = maybe_postprocess(&geom_pair, TOLERANCE, anomalous, postprocessing)?;

    let geom_pair_final = if write_obj {
        process_case(&label, geom_pair_postprocessed, output_path, interpolation_steps, watertight, &contour_types)
            .context("process case failed for geom_ab")?
    } else {
        geom_pair_postprocessed
    };

    Ok((geom_pair_final, logs_a, logs_b))
}

pub fn single_processing_rs(
    label: String,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
    input_path_a: Option<&str>,
    input_data_a: Option<InputData>,
    write_obj: bool,
    contour_types: Vec<ContourType>,
    output_path: &str,
    step_deg: f64,
    range_deg: f64,
    smooth: bool,
    bruteforce: bool,
    sample_size: usize,
) -> Result<()> {
    let mut geom = prepare_n_geometries(
        &label,
        image_center,
        radius,
        n_points,
        Some(vec![input_data_a].into_iter().flatten().collect()),
        input_path_a.map(Path::new),
        None,
        ProcessingOptions::Single,
    )
    .context("Failed to prepare geometries for full processing")?;

    if geom.len() != 1 {
        return Err(anyhow!(
            "Single processing requires exactly 1 geometry, got {}",
            geom.len()
        ));
    }

    let (geom, logs, _) = align_frames_in_geometry(
        &mut geom[0],
        step_deg,
        range_deg,
        smooth,
        bruteforce,
        sample_size,
    )
    .context("Failed to align frames within geometry")?;

    // TODO: postprocessing geometry (resample)

    // TODO: writing to obj

    Ok(())
}
