use anyhow::{anyhow, Context, Result};
use crossbeam::thread;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::intravascular::io::geometry::{Contour, ContourType, Geometry};
use crate::intravascular::io::input::InputData;
use crate::intravascular::io::output::write_obj_mesh_without_uv;
use crate::intravascular::processing::align_between::{align_between_geometries, GeometryPair};
use crate::intravascular::processing::align_within::{align_frames_in_geometry, AlignLog};
use crate::intravascular::processing::postprocessing::postprocess_geom_pair;
use crate::intravascular::processing::preprocessing::{prepare_n_geometries, ProcessingOptions};
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
    _diastole: bool,
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
) -> Result<(
    GeometryPair,
    GeometryPair,
    GeometryPair,
    GeometryPair,
    Vec<AlignLog>,
    Vec<AlignLog>,
    Vec<AlignLog>,
    Vec<AlignLog>,
)> {
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
        _diastole,
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

    let (
        mut geom_a,
        mut geom_b,
        mut geom_c,
        mut geom_d,
        logs_a,
        logs_b,
        logs_c,
        logs_d,
        bool_a,
        bool_b,
        bool_c,
        bool_d,
    ) = thread::scope(
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
                geom_a, geom_b, geom_c, geom_d, logs_a, logs_b, logs_c, logs_d, bool_a, bool_b,
                bool_c, bool_d,
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

    let geom_ab_postprocessed =
        maybe_postprocess(&geom_pair_ab, TOLERANCE, anomalous, postprocessing)?;
    let geom_cd_postprocessed =
        maybe_postprocess(&geom_pair_cd, TOLERANCE, anomalous, postprocessing)?;
    let geom_ac_postprocessed =
        maybe_postprocess(&geom_pair_ac, TOLERANCE, anomalous, postprocessing)?;
    let geom_bd_postprocessed =
        maybe_postprocess(&geom_pair_bd, TOLERANCE, anomalous, postprocessing)?;

    let geom_ab_final = if write_obj {
        process_case(
            &label,
            geom_ab_postprocessed,
            output_path_a,
            interpolation_steps,
            watertight,
            &contour_types,
        )
        .context("process case failed for geom_ab")?
    } else {
        geom_ab_postprocessed
    };

    let geom_cd_final = if write_obj {
        process_case(
            &label,
            geom_cd_postprocessed,
            output_path_b,
            interpolation_steps,
            watertight,
            &contour_types,
        )
        .context("process case failed for geom_cd")?
    } else {
        geom_cd_postprocessed
    };

    let geom_ac_final = if write_obj {
        process_case(
            &label,
            geom_ac_postprocessed,
            output_path_c,
            interpolation_steps,
            watertight,
            &contour_types,
        )
        .context("process case failed for geom_ac")?
    } else {
        geom_ac_postprocessed
    };

    let geom_bd_final = if write_obj {
        process_case(
            &label,
            geom_bd_postprocessed,
            output_path_d,
            interpolation_steps,
            watertight,
            &contour_types,
        )
        .context("process case failed for geom_bd")?
    } else {
        geom_bd_postprocessed
    };

    Ok((
        geom_ab_final,
        geom_cd_final,
        geom_ac_final,
        geom_bd_final,
        logs_a,
        logs_b,
        logs_c,
        logs_d,
    ))
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
    _diastole: bool,
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
) -> Result<(
    GeometryPair,
    GeometryPair,
    Vec<AlignLog>,
    Vec<AlignLog>,
    Vec<AlignLog>,
    Vec<AlignLog>,
)> {
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
        _diastole,
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

    let (
        mut geom_a,
        mut geom_b,
        mut geom_c,
        mut geom_d,
        logs_a,
        logs_b,
        logs_c,
        logs_d,
        bool_a,
        bool_b,
        bool_c,
        bool_d,
    ) = thread::scope(
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
                geom_a, geom_b, geom_c, geom_d, logs_a, logs_b, logs_c, logs_d, bool_a, bool_b,
                bool_c, bool_d,
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

    let geom_ab_postprocessed =
        maybe_postprocess(&geom_pair_ab, TOLERANCE, anomalous, postprocessing)?;
    let geom_cd_postprocessed =
        maybe_postprocess(&geom_pair_cd, TOLERANCE, anomalous, postprocessing)?;

    let geom_ab_final = if write_obj {
        process_case(
            &label,
            geom_ab_postprocessed,
            output_path_a,
            interpolation_steps,
            watertight,
            &contour_types,
        )
        .context("process case failed for geom_ab")?
    } else {
        geom_ab_postprocessed
    };

    let geom_cd_final = if write_obj {
        process_case(
            &label,
            geom_cd_postprocessed,
            output_path_b,
            interpolation_steps,
            watertight,
            &contour_types,
        )
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
    diastole: bool,
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
        diastole,
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
            ))
            .context("Failed to align geom_a and geom_b")?;

    let anomalous = bool_a || bool_b;

    let geom_pair_postprocessed =
        maybe_postprocess(&geom_pair, TOLERANCE, anomalous, postprocessing)?;

    let geom_pair_final = if write_obj {
        process_case(
            &label,
            geom_pair_postprocessed,
            output_path,
            interpolation_steps,
            watertight,
            &contour_types,
        )
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
    diastole: bool,
    write_obj: bool,
    watertight: bool,
    contour_types: Vec<ContourType>,
    output_path: &str,
    step_deg: f64,
    range_deg: f64,
    smooth: bool,
    bruteforce: bool,
    sample_size: usize,
) -> Result<(Geometry, Vec<AlignLog>)> {
    let mut geom = prepare_n_geometries(
        &label,
        image_center,
        radius,
        n_points,
        Some(vec![input_data_a].into_iter().flatten().collect()),
        diastole,
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

    if write_obj {
        // Create output directory
        std::fs::create_dir_all(output_path)?;

        // Write each contour type to OBJ
        for contour_type in &contour_types {
            let contours = extract_contours_by_type(&geom, *contour_type);

            if contours.is_empty() {
                eprintln!(
                    "Warning: No contours found for type {:?}, skipping",
                    contour_type
                );
                continue;
            }

            let type_name = get_contour_type_name(*contour_type);
            let obj_filename = format!("{}_{}.obj", type_name, label);
            let mtl_filename = format!("{}_{}.mtl", type_name, label);
            let obj_path = Path::new(output_path).join(&obj_filename);
            let mtl_path = Path::new(output_path).join(&mtl_filename);

            // Create appropriate MTL file based on contour type
            create_mtl_for_contour_type(*contour_type, &mtl_path, &obj_filename)?;

            // Write OBJ without UV coordinates
            write_obj_mesh_without_uv(
                &contours,
                obj_path.to_str().unwrap(),
                mtl_path.to_str().unwrap(),
                watertight,
            )
            .context(format!("Failed to write OBJ for {}", type_name))?;
        }

        println!(
            "Successfully wrote OBJ files for geometry {} to {}",
            label, output_path
        );
    }

    Ok((geom, logs))
}

/// Helper function to extract contours of a specific type from a geometry
pub fn extract_contours_by_type(geometry: &Geometry, contour_type: ContourType) -> Vec<Contour> {
    match contour_type {
        ContourType::Lumen => geometry
            .frames
            .iter()
            .map(|frame| frame.lumen.clone())
            .collect(),
        _ => geometry
            .frames
            .iter()
            .filter_map(|frame| frame.extras.get(&contour_type).cloned())
            .collect(),
    }
}

/// Helper function to get contour type name for file naming
pub fn get_contour_type_name(contour_type: ContourType) -> &'static str {
    match contour_type {
        ContourType::Lumen => "lumen",
        ContourType::Eem => "eem",
        ContourType::Calcification => "calcification",
        ContourType::Sidebranch => "sidebranch",
        ContourType::Catheter => "catheter",
        ContourType::Wall => "wall",
    }
}

/// Helper function to create appropriate MTL file for each contour type
pub fn create_mtl_for_contour_type(
    contour_type: ContourType,
    mtl_path: &Path,
    _obj_filename: &str,
) -> anyhow::Result<()> {
    let mut mtl_file = File::create(mtl_path)?;

    match contour_type {
        ContourType::Lumen | ContourType::Eem => {
            // For lumen and EEM: white material (displacement-like)
            writeln!(
                mtl_file,
                "newmtl material\nKa 1.0 1.0 1.0\nKd 1.0 1.0 1.0\nKs 0.0 0.0 0.0"
            )?;
        }
        ContourType::Catheter | ContourType::Calcification => {
            // For catheter and calcification: black material
            writeln!(
                mtl_file,
                "newmtl material\nKa 0.0 0.0 0.0\nKd 0.0 0.0 0.0\nKs 0.0 0.0 0.0"
            )?;
        }
        ContourType::Wall | ContourType::Sidebranch => {
            // For wall and sidebranch: semi-transparent material
            writeln!(
                mtl_file,
                "newmtl material\nKa 0.5 0.5 0.5\nKd 0.5 0.5 0.5\nKs 0.0 0.0 0.0\nd 0.7"
            )?;
        }
    }

    Ok(())
}
