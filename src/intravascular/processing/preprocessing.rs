use anyhow::Context;
use std::path::Path;

use crate::intravascular::io;
use crate::intravascular::io::input::InputData;
use crate::types::native::geometry::Geometry;

#[derive(Debug, Clone, Copy)]
pub enum ProcessingOptions {
    Single,
    Pair,
    Full,
}

fn path_basename(path: &Path) -> String {
    path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string()
}

/// Prepare 1 / 2 / 4 geometries depending on `processing`.
///
/// `labels` is only used for path-based processing (InputData carries its own
/// labels).  Pass 1 label for `Single`, 2 for `Pair`, 4 for `Full`.  If the
/// count does not match, the last component of each path is used instead.
///
/// `reference_frames[i]` chooses the anchor of geometry `i` (file frame numbering, `None`
/// keeps the default anchor of a path or the `ref_point` of an InputData, missing entries
/// count as `None`).
///
/// Geometries are numbered in the order they are returned: `Single` 0, `Pair` diastole 0 and
/// systole 1, `Full` diastole and systole of `path_a` (0, 1) then of `path_b` (2, 3). With
/// `InputData` the i-th input is geometry i.
pub fn prepare_n_geometries(
    labels: &[String],
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
    mut input_data: Option<Vec<InputData>>,
    diastole: bool,
    path_a: Option<&Path>,
    path_b: Option<&Path>,
    processing: ProcessingOptions,
    reference_frames: &[Option<u32>],
) -> anyhow::Result<Vec<Geometry>> {
    let n = match processing {
        ProcessingOptions::Single => 1,
        ProcessingOptions::Pair => 2,
        ProcessingOptions::Full => 4,
    };
    let reference_frame = |i: usize| reference_frames.get(i).copied().flatten();

    // Option 1: use InputData if there are enough
    if let Some(ref mut inputs) = input_data {
        if inputs.len() >= n {
            return inputs
                .drain(..n)
                .enumerate()
                .map(|(i, mut input)| {
                    if let Some(frame) = reference_frame(i) {
                        input
                            .set_reference_frame(frame)
                            .with_context(|| format!("choosing the anchor of {}", input.label))?;
                    }
                    let diastole = input.diastole;
                    let label = input.label.clone();
                    io::build_geometry_from_inputdata(
                        Some(input),
                        None,
                        &label,
                        diastole,
                        image_center,
                        radius,
                        n_points,
                    )
                    .with_context(|| {
                        format!("Failed to build geometry from provided InputData ({processing:?})")
                    })
                })
                .collect();
        }
    }

    // Option 2: use the paths
    let jobs: Vec<(&Path, bool)> = match processing {
        ProcessingOptions::Single => {
            let path = path_a.or(path_b).ok_or_else(|| {
                anyhow::anyhow!("Single processing requires at least one InputData or one path (path_a or path_b)")
            })?;
            vec![(path, diastole)]
        }
        ProcessingOptions::Pair => {
            let path = path_a.or(path_b).ok_or_else(|| {
                anyhow::anyhow!("Pair processing requires at least two InputData or one path (path_a or path_b)")
            })?;
            vec![(path, true), (path, false)]
        }
        ProcessingOptions::Full => match (path_a, path_b) {
            (Some(a), Some(b)) => vec![(a, true), (a, false), (b, true), (b, false)],
            _ => anyhow::bail!(
                "Full processing requires either at least 4 InputData or both path_a and path_b"
            ),
        },
    };

    let use_labels = labels.len() == n;
    jobs.into_iter()
        .enumerate()
        .map(|(i, (path, diastole))| {
            let label = if use_labels {
                labels[i].clone()
            } else {
                path_basename(path)
            };
            io::build_geometry_from_path(
                path,
                reference_frame(i),
                &label,
                diastole,
                image_center,
                radius,
                n_points,
            )
            .with_context(|| {
                format!(
                    "Failed to build geometry ({processing:?}) from path with diastole={diastole}"
                )
            })
        })
        .collect()
}

#[cfg(test)]
mod preprocessing_tests {
    use crate::intravascular::io::input::InputData;
    use crate::types::native::ContourPoint;
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
            record: None,
            ref_point: ContourPoint {
                frame_index: 0,
                point_index: 0,
                x: 5.0,
                y: 5.0,
                z: 0.0,
                aortic: false,
            },
            references: Vec::new(),
            diastole,
            label: label.to_string(),
            annotations: Vec::new(),
        }
    }

    #[test]
    fn test_reference_frames_choose_each_anchor() -> anyhow::Result<()> {
        let src = Path::new("data/fixtures/ivus_stress");
        let dir = std::env::temp_dir().join(format!(
            "multimodars_prepare_references_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir)?;
        for f in ["diastolic_contours.csv", "systolic_contours.csv"] {
            std::fs::copy(src.join(f), dir.join(f))?;
        }
        std::fs::write(
            dir.join("diastolic_reference_points.csv"),
            "288\t4.0\t3.0\t1.0\t1\n314\t4.4\t3.4\t20.6\t0\n",
        )?;
        std::fs::write(
            dir.join("systolic_reference_points.csv"),
            "255\t4.0\t3.0\t1.0\t1\n281\t4.5\t3.5\t18.0\t0\n",
        )?;
        let prepare = |frames: &[Option<u32>]| {
            prepare_n_geometries(
                &[],
                (4.5, 4.5),
                0.5,
                20,
                None,
                true,
                Some(&dir),
                None,
                ProcessingOptions::Pair,
                frames,
            )
        };
        let anchors = |geoms: &[Geometry]| -> Vec<u32> {
            geoms
                .iter()
                .map(|g| {
                    g.frames[g.find_ref_frame_idx().unwrap()]
                        .lumen
                        .original_frame
                })
                .collect()
        };
        let default = prepare(&[])?;
        let chosen = prepare(&[Some(288), None])?;
        let both = prepare(&[Some(288), Some(255)])?;
        let unknown = prepare(&[None, Some(1)]);
        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(anchors(&default), vec![314, 281]);
        assert_eq!(anchors(&chosen), vec![288, 281]);
        assert_eq!(anchors(&both), vec![288, 255]);
        let err = format!("{:#}", unknown.unwrap_err());
        assert!(err.contains("no reference point on frame 1"), "{err}");

        // with InputData the i-th input is geometry i
        let inputs = vec![
            create_mock_input_data("a", true),
            create_mock_input_data("b", false),
        ];
        let err = prepare_n_geometries(
            &[],
            (4.5, 4.5),
            0.5,
            20,
            Some(inputs),
            true,
            None,
            None,
            ProcessingOptions::Pair,
            &[None, Some(7)],
        )
        .unwrap_err();
        assert!(format!("{err:#}").contains("anchor of b"), "{err:#}");
        Ok(())
    }

    #[test]
    fn test_prepare_one_geometry_path() -> anyhow::Result<()> {
        let geometry = prepare_n_geometries(
            &["stress".to_string()],
            (4.5, 4.5),
            0.5,
            20,
            None,
            true,
            Some(Path::new("data/fixtures/ivus_stress")),
            None,
            ProcessingOptions::Single,
            &[],
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
            &["stress".to_string(), "stress".to_string()],
            (4.5, 4.5),
            0.5,
            20,
            None,
            true,
            Some(Path::new("data/fixtures/ivus_stress")),
            None,
            ProcessingOptions::Pair,
            &[],
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
            &[],
            (4.5, 4.5),
            0.5,
            20,
            Some(input_data),
            true,
            None,
            None,
            ProcessingOptions::Single,
            &[],
        )?;

        assert_eq!(geometry.len(), 1);
        // Should use the InputData label, not the fallback
        assert_eq!(geometry[0].label, "test_input");
        Ok(())
    }

    #[test]
    fn test_single_with_path() -> anyhow::Result<()> {
        let geometry = prepare_n_geometries(
            &["stress".to_string()],
            (4.5, 4.5),
            0.5,
            20,
            None,
            true,
            Some(Path::new("data/fixtures/ivus_stress")),
            None,
            ProcessingOptions::Single,
            &[],
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
            &[],
            (4.5, 4.5),
            0.5,
            20,
            Some(input_data),
            true,
            None,
            None,
            ProcessingOptions::Pair,
            &[],
        )?;

        assert_eq!(geometry.len(), 2);
        assert_eq!(geometry[0].label, "first");
        assert_eq!(geometry[1].label, "second");
        Ok(())
    }

    #[test]
    fn test_pair_with_one_path() -> anyhow::Result<()> {
        let geometry = prepare_n_geometries(
            &["stress_0".to_string(), "stress_1".to_string()],
            (4.5, 4.5),
            0.5,
            20,
            None,
            true,
            Some(Path::new("data/fixtures/ivus_stress")),
            None,
            ProcessingOptions::Pair,
            &[],
        )?;

        assert_eq!(geometry.len(), 2);
        // Both should use the provided label since built from path
        assert_eq!(geometry[0].label, "stress_0");
        assert_eq!(geometry[1].label, "stress_1");
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
            &[],
            (4.5, 4.5),
            0.5,
            20,
            Some(input_data),
            true,
            None,
            None,
            ProcessingOptions::Full,
            &[],
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
        let labels = [
            "test_0".to_string(),
            "test_1".to_string(),
            "test_2".to_string(),
            "test_3".to_string(),
        ];
        let geometry = prepare_n_geometries(
            &labels,
            (4.5, 4.5),
            0.5,
            20,
            None,
            true,
            Some(Path::new("data/fixtures/ivus_stress")),
            Some(Path::new("data/fixtures/ivus_rest")),
            ProcessingOptions::Full,
            &[],
        )?;

        assert_eq!(geometry.len(), 4);
        for (i, geom) in geometry.iter().enumerate() {
            assert_eq!(geom.label, format!("test_{i}"));
        }
        Ok(())
    }

    #[test]
    fn test_prefers_input_data_over_paths() -> anyhow::Result<()> {
        let input_data = vec![create_mock_input_data("preferred", true)];
        let geometry = prepare_n_geometries(
            &[],
            (4.5, 4.5),
            0.5,
            20,
            Some(input_data),
            true,
            Some(Path::new("data/ivus_stress")),
            Some(Path::new("data/ivus_rest")),
            ProcessingOptions::Single,
            &[],
        )?;

        assert_eq!(geometry.len(), 1);
        assert_eq!(geometry[0].label, "preferred"); // Uses InputData label, not labels vec
        Ok(())
    }

    #[test]
    fn test_insufficient_input_data_falls_back_to_paths() -> anyhow::Result<()> {
        // Only 1 InputData for Pair processing (needs 2) - should fall back to path
        let input_data = vec![create_mock_input_data("only_one", true)];

        let geometry = prepare_n_geometries(
            &["from_path_0".to_string(), "from_path_1".to_string()],
            (4.5, 4.5),
            0.5,
            20,
            Some(input_data),
            true,
            Some(Path::new("data/fixtures/ivus_stress")),
            None,
            ProcessingOptions::Pair,
            &[],
        )?;

        assert_eq!(geometry.len(), 2);
        assert_eq!(geometry[0].label, "from_path_0");
        assert_eq!(geometry[1].label, "from_path_1");
        Ok(())
    }

    #[test]
    fn test_single_fails_with_no_inputs() {
        let result = prepare_n_geometries(
            &[],
            (0.0, 0.0),
            10.0,
            36,
            None,
            true,
            None,
            None,
            ProcessingOptions::Single,
            &[],
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_pair_fails_with_insufficient_inputs() {
        // No InputData and no paths
        let result = prepare_n_geometries(
            &[],
            (0.0, 0.0),
            10.0,
            36,
            None,
            true,
            None,
            None,
            ProcessingOptions::Pair,
            &[],
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
            &["test".to_string()],
            (0.0, 0.0),
            10.0,
            36,
            Some(input_data),
            true,
            Some(Path::new("data/ivus_stress")),
            None,
            ProcessingOptions::Full,
            &[],
        );

        assert!(result.is_err());
    }
}
