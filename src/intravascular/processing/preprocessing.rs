use anyhow::Context;
use std::path::Path;

use crate::intravascular::io::build_geometry_from_inputdata;
use crate::intravascular::io::geometry::Geometry;
use crate::intravascular::io::input::InputData;

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
    _diastole: bool,
    path_a: Option<&Path>,
    path_b: Option<&Path>,
    processing: ProcessingOptions,
) -> anyhow::Result<Vec<Geometry>> {
    match processing {
        ProcessingOptions::Single => {
            // Option 1: Use InputData if available
            if let Some(ref mut inputs) = input_data {
                if !inputs.is_empty() {
                    let input = inputs.remove(0);
                    let diastole = input.diastole;
                    let label_str = input.label.as_str().to_string();
                    let geom = build_geometry_from_inputdata(
                        Some(input),
                        None,
                        &label_str,
                        diastole,
                        image_center,
                        radius,
                        n_points,
                    )
                    .context("Failed to build geometry from provided InputData (Single)")?;
                    return Ok(vec![geom]);
                }
            }

            // Option 2: Use path
            let path = path_a.or(path_b).ok_or_else(|| {
                anyhow::anyhow!("Single processing requires at least one InputData or one path (path_a or path_b)")
            })?;

            let geom = build_geometry_from_inputdata(
                None,
                Some(path),
                label,
                true, // default diastole=true
                image_center,
                radius,
                n_points,
            )
            .context("Failed to build geometry from path (Single)")?;
            Ok(vec![geom])
        }

        ProcessingOptions::Pair => {
            // Option 1: Use InputData if available (need 2)
            if let Some(ref mut inputs) = input_data {
                if inputs.len() >= 2 {
                    let mut geometries = Vec::with_capacity(2);
                    for _ in 0..2 {
                        let input = inputs.remove(0);
                        let diastole = input.diastole;
                        let label_str = input.label.as_str().to_string();
                        let geom = build_geometry_from_inputdata(
                            Some(input),
                            None,
                            &label_str,
                            diastole,
                            image_center,
                            radius,
                            n_points,
                        )
                        .context("Failed to build geometry for Pair from InputData")?;
                        geometries.push(geom);
                    }
                    return Ok(geometries);
                }
            }

            // Option 2: Use path (create 2 geometries with different diastole values)
            let path = path_a.or(path_b).ok_or_else(|| {
                anyhow::anyhow!("Pair processing requires at least two InputData or one path (path_a or path_b)")
            })?;

            let mut geometries = Vec::with_capacity(2);
            for &diastole in &[true, false] {
                let geom = build_geometry_from_inputdata(
                    None,
                    Some(path),
                    label,
                    diastole,
                    image_center,
                    radius,
                    n_points,
                )
                .with_context(|| {
                    format!(
                        "Failed to build geometry for Pair with diastole={}",
                        diastole
                    )
                })?;
                geometries.push(geom);
            }
            Ok(geometries)
        }

        ProcessingOptions::Full => {
            // Option 1: Use InputData if available (need 4)
            if let Some(ref mut inputs) = input_data {
                if inputs.len() >= 4 {
                    let mut geometries = Vec::with_capacity(4);
                    for _ in 0..4 {
                        let input = inputs.remove(0);
                        let diastole = input.diastole;
                        let label_str = input.label.as_str().to_string();
                        let geom = build_geometry_from_inputdata(
                            Some(input),
                            None,
                            &label_str,
                            diastole,
                            image_center,
                            radius,
                            n_points,
                        )
                        .context("Failed to build geometry for Full from InputData")?;
                        geometries.push(geom);
                    }
                    return Ok(geometries);
                }
            }

            // Option 2: Use paths (need 2 paths)
            let (path_a, path_b) = match (path_a, path_b) {
                (Some(a), Some(b)) => (a, b),
                _ => anyhow::bail!("Full processing requires either at least 4 InputData or both path_a and path_b"),
            };

            let mut geometries = Vec::with_capacity(4);
            for &path in &[path_a, path_b] {
                for &diastole in &[true, false] {
                    let geom = build_geometry_from_inputdata(
                        None,
                        Some(path),
                        label,
                        diastole,
                        image_center,
                        radius,
                        n_points,
                    )
                    .with_context(|| {
                        format!(
                            "Failed to build geometry for Full from path with diastole={}",
                            diastole
                        )
                    })?;
                    geometries.push(geom);
                }
            }
            Ok(geometries)
        }
    }
}

#[cfg(test)]
mod preprocessing_tests {
    use crate::intravascular::io::input::{ContourPoint, InputData};
    use anyhow::Ok;

    use super::*;
    use std::path::Path;

    // Helper function to create mock InputData
    fn create_mock_input_data(label: &str, diastole: bool) -> InputData {
        InputData {
            lumen: vec![ContourPoint {
                frame_index: 0,
                point_index: 0,
                x: 10.0,
                y: 20.0,
                z: 0.0,
                aortic: false,
            }],
            eem: None,
            calcification: None,
            sidebranch: None,
            record: None,
            ref_point: ContourPoint {
                frame_index: 0,
                point_index: 0,
                x: 5.0,
                y: 5.0,
                z: 0.0,
                aortic: false,
            },
            diastole,
            label: label.to_string(),
        }
    }

    #[test]
    fn test_prepare_one_geometry_path() -> anyhow::Result<()> {
        let geometry = prepare_n_geometries(
            "stress",
            (4.5, 4.5),
            0.5,
            20,
            None,
            true,
            Some(Path::new("data/fixtures/ivus_stress")),
            None,
            ProcessingOptions::Single,
        )?;
        let geom = &geometry[0];

        assert_eq!(geometry.len(), 1);
        assert_eq!(geom.frames[0].lumen.original_frame, 314);
        assert!(geom.frames[0].reference_point.is_some());
        Ok(())
    }

    #[test]
    fn test_prepare_two_geometry_one_path() -> anyhow::Result<()> {
        let geometry = prepare_n_geometries(
            "stress",
            (4.5, 4.5),
            0.5,
            20,
            None,
            true,
            Some(Path::new("data/fixtures/ivus_stress")),
            None,
            ProcessingOptions::Pair,
        )?;
        let geom = &geometry[0];

        assert_eq!(geometry.len(), 2);
        assert_eq!(geom.frames[0].lumen.original_frame, 314);
        assert!(geom.frames[0].reference_point.is_some());
        Ok(())
    }

    #[test]
    fn test_single_with_one_input_data() -> anyhow::Result<()> {
        let input_data = vec![create_mock_input_data("test_input", true)];

        let geometry = prepare_n_geometries(
            "fallback_label",
            (4.5, 4.5),
            0.5,
            20,
            Some(input_data),
            true,
            None,
            None,
            ProcessingOptions::Single,
        )?;

        assert_eq!(geometry.len(), 1);
        // Should use the InputData label, not the fallback
        assert_eq!(geometry[0].label, "test_input");
        Ok(())
    }

    #[test]
    fn test_single_with_path() -> anyhow::Result<()> {
        let geometry = prepare_n_geometries(
            "stress",
            (4.5, 4.5),
            0.5,
            20,
            None,
            true,
            Some(Path::new("data/fixtures/ivus_stress")),
            None,
            ProcessingOptions::Single,
        )?;

        assert_eq!(geometry.len(), 1);
        Ok(())
    }

    #[test]
    fn test_pair_with_two_input_data() -> anyhow::Result<()> {
        let input_data = vec![
            create_mock_input_data("first", true),
            create_mock_input_data("second", false),
        ];

        let geometry = prepare_n_geometries(
            "fallback_label",
            (4.5, 4.5),
            0.5,
            20,
            Some(input_data),
            true,
            None,
            None,
            ProcessingOptions::Pair,
        )?;

        assert_eq!(geometry.len(), 2);
        assert_eq!(geometry[0].label, "first");
        assert_eq!(geometry[1].label, "second");
        Ok(())
    }

    #[test]
    fn test_pair_with_one_path() -> anyhow::Result<()> {
        let geometry = prepare_n_geometries(
            "stress",
            (4.5, 4.5),
            0.5,
            20,
            None,
            true,
            Some(Path::new("data/fixtures/ivus_stress")),
            None,
            ProcessingOptions::Pair,
        )?;

        assert_eq!(geometry.len(), 2);
        // Both should use the provided label since built from path
        assert_eq!(geometry[0].label, "stress");
        assert_eq!(geometry[1].label, "stress");
        Ok(())
    }

    #[test]
    fn test_full_with_four_input_data() -> anyhow::Result<()> {
        let input_data = vec![
            create_mock_input_data("first", true),
            create_mock_input_data("second", false),
            create_mock_input_data("third", true),
            create_mock_input_data("fourth", false),
        ];

        let geometry = prepare_n_geometries(
            "fallback_label",
            (4.5, 4.5),
            0.5,
            20,
            Some(input_data),
            true,
            None,
            None,
            ProcessingOptions::Full,
        )?;

        assert_eq!(geometry.len(), 4);
        assert_eq!(geometry[0].label, "first");
        assert_eq!(geometry[1].label, "second");
        assert_eq!(geometry[2].label, "third");
        assert_eq!(geometry[3].label, "fourth");
        Ok(())
    }

    #[test]
    fn test_full_with_two_paths() -> anyhow::Result<()> {
        let geometry = prepare_n_geometries(
            "test",
            (4.5, 4.5),
            0.5,
            20,
            None,
            true,
            Some(Path::new("data/fixtures/ivus_stress")),
            Some(Path::new("data/fixtures/ivus_rest")),
            ProcessingOptions::Full,
        )?;

        assert_eq!(geometry.len(), 4);
        // All should use the provided label since built from paths
        for geom in geometry {
            assert_eq!(geom.label, "test");
        }
        Ok(())
    }

    #[test]
    fn test_prefers_input_data_over_paths() -> anyhow::Result<()> {
        let input_data = vec![create_mock_input_data("preferred", true)];
        let geometry = prepare_n_geometries(
            "fallback",
            (4.5, 4.5),
            0.5,
            20,
            Some(input_data),
            true,
            Some(Path::new("data/ivus_stress")),
            Some(Path::new("data/ivus_rest")),
            ProcessingOptions::Single,
        )?;

        assert_eq!(geometry.len(), 1);
        assert_eq!(geometry[0].label, "preferred"); // Uses InputData label, not fallback
        Ok(())
    }

    #[test]
    fn test_insufficient_input_data_falls_back_to_paths() -> anyhow::Result<()> {
        // Only 1 InputData for Pair processing (needs 2) - should fall back to path
        let input_data = vec![create_mock_input_data("only_one", true)];

        let geometry = prepare_n_geometries(
            "from_path",
            (4.5, 4.5),
            0.5,
            20,
            Some(input_data),
            true,
            Some(Path::new("data/fixtures/ivus_stress")),
            None,
            ProcessingOptions::Pair,
        )?;

        assert_eq!(geometry.len(), 2);
        assert_eq!(geometry[0].label, "from_path"); // Uses path label
        assert_eq!(geometry[1].label, "from_path");
        Ok(())
    }

    #[test]
    fn test_single_fails_with_no_inputs() {
        let result = prepare_n_geometries(
            "test",
            (0.0, 0.0),
            10.0,
            36,
            None,
            true,
            None,
            None,
            ProcessingOptions::Single,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_pair_fails_with_insufficient_inputs() {
        // No InputData and no paths
        let result = prepare_n_geometries(
            "test",
            (0.0, 0.0),
            10.0,
            36,
            None,
            true,
            None,
            None,
            ProcessingOptions::Pair,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_full_fails_with_insufficient_inputs() {
        // Only 3 InputData (needs 4) and only 1 path (needs 2)
        let input_data = vec![
            create_mock_input_data("first", true),
            create_mock_input_data("second", false),
            create_mock_input_data("third", true),
        ];

        let result = prepare_n_geometries(
            "test",
            (0.0, 0.0),
            10.0,
            36,
            Some(input_data),
            true,
            Some(Path::new("data/ivus_stress")),
            None,
            ProcessingOptions::Full,
        );

        assert!(result.is_err());
    }
}
