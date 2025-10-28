use anyhow::{Context, Result};
use std::path::Path;

use crate::intravascular::io::build_geometry_from_inputdata;
use crate::intravascular::io::input::InputData;
use crate::intravascular::io::geometry::Geometry;

#[derive(Debug, Clone, Copy)]
pub enum ProcessingOptions {
    Single,
    Pair,
    Full,
}

/// Prepare 1 / 2 / 4 geometries depending on `processing`.
pub fn prepare_n_geometries(
    label: &str,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
    mut input_data: Option<Vec<InputData>>,
    path_a: Option<&Path>,
    path_b: Option<&Path>,
    processing: ProcessingOptions,
) -> Result<Vec<Geometry>> {
    // Helper to pick the first available path (A then B)
    let pick_one_path = || path_a.or(path_b);

    match processing {
        ProcessingOptions::Single => {
            // Need at least one InputData OR at least one path
            if let Some(ref mut inputs) = input_data {
                if inputs.is_empty() {
                    // fallthrough to check paths
                    if pick_one_path().is_none() {
                        anyhow::bail!(
                            "Single processing requires at least one InputData or at least one path (path_a or path_b)"
                        );
                    }
                } else {
                    // Build one geometry from first InputData (consume it)
                    let input = inputs.remove(0);
                    let diastole = input.diastole; // copy diastole before moving `input`
                    let geom = build_geometry_from_inputdata(
                        Some(input),
                        None,
                        label,
                        diastole,
                        image_center,
                        radius,
                        n_points,
                    )
                    .context("Failed to build geometry from provided InputData (Single)")?;
                    return Ok(vec![geom]);
                }
            }

            // No suitable InputData: try paths
            if let Some(path) = pick_one_path() {
                let geom = build_geometry_from_inputdata(
                    None,
                    Some(path),
                    label,
                    true, // default convention
                    image_center,
                    radius,
                    n_points,
                )
                .context("Failed to build geometry from path (Single)")?;
                return Ok(vec![geom]);
            }

            anyhow::bail!(
                "Single processing requires at least one InputData or at least one path (path_a or path_b)"
            );
        }

        ProcessingOptions::Pair => {
            // Requires at least two InputData AND at least one path (literal interpretation)
            let mut outputs = Vec::with_capacity(2);

            // Validate input_data
            let mut inputs_present = false;
            if let Some(ref mut inputs) = input_data {
                if inputs.len() >= 2 {
                    inputs_present = true;
                }
            }
            let path_present = pick_one_path().is_some();

            if !inputs_present || !path_present {
                anyhow::bail!(
                    "Pair processing requires at least two InputData and at least one path (path_a or path_b)"
                );
            }

            // We have at least two InputData and a path -> build two geometries from the first two InputData,
            // passing the chosen path as the path argument.
            let path = pick_one_path().unwrap(); // safe after check above
            let mut inputs = input_data.take().unwrap(); // safe; we checked len >= 2

            for _ in 0..2 {
                let input = inputs.remove(0);
                let diastole = input.diastole;
                let geom = build_geometry_from_inputdata(
                    Some(input),
                    Some(path),
                    label,
                    diastole,
                    image_center,
                    radius,
                    n_points,
                )
                .context("Failed to build geometry for Pair from InputData + path")?;
                outputs.push(geom);
            }

            return Ok(outputs);
        }

        ProcessingOptions::Full => {
            // Two valid ways:
            //  - at least 4 InputData (use the first 4)
            //  - OR both path_a and path_b present (2 paths) -> create 4 geometries (2 per path, toggling diastole)
            if let Some(mut inputs) = input_data {
                if inputs.len() >= 4 {
                    let mut outs = Vec::with_capacity(4);
                    for _ in 0..4 {
                        let input = inputs.remove(0);
                        let diastole = input.diastole;
                        let geom = build_geometry_from_inputdata(
                            Some(input),
                            None,
                            label,
                            diastole,
                            image_center,
                            radius,
                            n_points,
                        )
                        .context("Failed to build geometry for Full from InputData")?;
                        outs.push(geom);
                    }
                    return Ok(outs);
                }
            }

            // Try path-based option: need both path_a and path_b
            if path_a.is_some() && path_b.is_some() {
                let mut outs = Vec::with_capacity(4);
                // For each path create 2 geometries (toggle diastole false/true)
                for &diastole_flag in &[false, true] {
                    let geom = build_geometry_from_inputdata(
                        None,
                        path_a,
                        label,
                        diastole_flag,
                        image_center,
                        radius,
                        n_points,
                    )
                    .context("Failed to build geometry for Full from path_a")?;
                    outs.push(geom);
                }
                for &diastole_flag in &[false, true] {
                    let geom = build_geometry_from_inputdata(
                        None,
                        path_b,
                        label,
                        diastole_flag,
                        image_center,
                        radius,
                        n_points,
                    )
                    .context("Failed to build geometry for Full from path_b")?;
                    outs.push(geom);
                }
                return Ok(outs);
            }

            anyhow::bail!(
                "Full processing requires either at least 4 InputData or both path_a and path_b (two paths)"
            );
        }
    }
}
