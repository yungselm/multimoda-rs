use super::input::{default_names, InputData};
use crate::types::native::contour::{Contour, ContourType};
use crate::types::native::contour_point::ContourPoint;
use crate::types::native::frame::Frame;
use crate::types::native::geometry::Geometry;
use crate::types::native::structure::{
    jitter_cloud, revolve_arc, CloudParams, Structure, StructureKind,
};
use crate::types::native::Spacing;
use std::collections::{HashMap, HashSet};
use std::path::Path;

pub fn build_geometry_from_inputdata(
    input_data: Option<InputData>,
    path: Option<&Path>,
    label: &str,
    diastole: bool,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
) -> anyhow::Result<Geometry> {
    let input_data = if let Some(input_data) = input_data {
        input_data
    } else if let Some(path) = path {
        InputData::process_directory(path, default_names(), diastole, label)?
    } else {
        return Err(anyhow::anyhow!(
            "Either input_data or path must be provided"
        ));
    };
    let print_input_data = input_data.clone();

    // First, collect all unique original frame IDs from ALL contour types
    let mut all_original_frames: HashSet<u32> = HashSet::new();

    for point in &input_data.lumen {
        all_original_frames.insert(point.frame_index);
    }

    if let Some(eem_points) = &input_data.eem {
        for point in eem_points {
            all_original_frames.insert(point.frame_index);
        }
    }

    let mut sorted_original_frames: Vec<u32> = all_original_frames.into_iter().collect();
    sorted_original_frames.sort();

    let frame_mapping: HashMap<u32, u32> = sorted_original_frames
        .iter()
        .enumerate()
        .map(|(i, &original_id)| (original_id, i as u32))
        .collect();

    // Now build contours using the shared mapping
    let lumen_contours = Contour::build_contour_with_mapping(
        input_data.lumen,
        input_data.record.clone(),
        ContourType::Lumen,
        &frame_mapping,
    )?;

    let eem_contours = if let Some(eem_points) = input_data.eem {
        Contour::build_contour_with_mapping(eem_points, None, ContourType::Eem, &frame_mapping)?
    } else {
        Vec::new()
    };

    let mut frame_map: HashMap<u32, Frame> = HashMap::new();

    for mut contour in lumen_contours {
        contour.compute_centroid();
        let frame_id = contour.id;
        let mut frame = Frame {
            id: frame_id,
            centroid: contour.centroid.unwrap_or((0.0, 0.0, 0.0)),
            lumen: contour,
            extras: HashMap::new(),
            reference_point: None,
            structures: Vec::new(),
        };
        frame.reference_point = input_data
            .reference_points
            .iter()
            .find(|p| p.frame_index == frame.lumen.original_frame)
            .copied();

        frame_map.insert(frame_id, frame);
    }

    let dropped = input_data
        .reference_points
        .iter()
        .filter(|p| !frame_mapping.contains_key(&p.frame_index))
        .count();
    if dropped > 0 {
        eprintln!("Warning: dropped {dropped} reference point(s) whose frame has no lumen contour");
    }

    for mut contour in eem_contours {
        contour.compute_centroid();
        if let Some(frame) = frame_map.get_mut(&contour.id) {
            frame.extras.insert(ContourType::Eem, contour);
        }
    }

    if n_points > 0 {
        let all_points: Vec<ContourPoint> = frame_map
            .values()
            .flat_map(|frame| frame.lumen.points.iter().cloned())
            .collect();

        let catheter_points =
            Frame::create_catheter_points(&all_points, image_center, radius, n_points);

        let catheter_contours = Contour::build_contour_with_mapping(
            catheter_points,
            None,
            ContourType::Catheter,
            &frame_mapping,
        )?;

        for mut contour in catheter_contours {
            contour.compute_centroid();
            if let Some(frame) = frame_map.get_mut(&contour.id) {
                frame.extras.insert(ContourType::Catheter, contour);
            }
        }
    }

    let mut frames: Vec<Frame> = frame_map.into_values().collect();
    frames.sort_by_key(|f| f.id);

    let mut geometry = Geometry {
        frames,
        label: label.to_string(),
        master_reference: None,
    };

    if let Some(records) = &input_data.record {
        geometry.reorder_frames(records, diastole);
    }

    for frame in &mut geometry.frames {
        frame.sort_frame_points();
    }

    geometry.ensure_proximal_at_position_zero();

    for frame in geometry.frames.iter_mut() {
        let id = frame.id;
        frame.set_value(Some(id), None, None, None);
    }

    // frames now run proximal to distal, the first reference is the most proximal
    match input_data.master_reference {
        Some(master) => geometry.set_master_reference(master)?,
        None => geometry.master_reference = geometry.reference_frames().first().copied(),
    }

    // frame order and z are final, the clouds are generated around them
    add_structure_clouds(
        &mut geometry,
        input_data.structures,
        &CloudParams::default(),
    )?;

    super::integrity_check::check_geometry_integrity(&geometry)?;

    let from_path = path.is_some();
    print_success_message(print_input_data, from_path);

    Ok(geometry)
}

/// Puts the point cloud of each structure on the frame with the same original frame.
///
/// A closed contour becomes a jittered cloud spread over half the frame spacing in z on both
/// sides. An open side-branch arc is turned around its own axis (`revolve_arc`), an open
/// in-wall contour is an error.
/// Structures without a lumen frame are dropped with one warning.
fn add_structure_clouds(
    geometry: &mut Geometry,
    structures: Vec<Structure>,
    params: &CloudParams,
) -> anyhow::Result<()> {
    let half_spacing = geometry.median_spacing().unwrap_or(0.0) / 2.0;
    let mut per_frame: HashMap<u32, u64> = HashMap::new();
    let mut dropped = 0usize;
    for structure in structures {
        let Some(original) = structure.points.first().map(|p| p.frame_index) else {
            continue;
        };
        let Some(frame) = geometry
            .frames
            .iter_mut()
            .find(|f| f.lumen.original_frame == original)
        else {
            dropped += 1;
            continue;
        };
        if !structure.closed && structure.kind != StructureKind::Sidebranch {
            anyhow::bail!(
                "open {} contour on frame {original}, in-wall contours must be closed",
                structure.kind
            );
        }
        // on its frame, numbered like the lumen
        let frame_index = frame
            .lumen
            .points
            .first()
            .map_or(frame.id, |p| p.frame_index);
        let contour: Vec<ContourPoint> = structure
            .points
            .iter()
            .map(|p| ContourPoint {
                frame_index,
                z: frame.centroid.2,
                ..*p
            })
            .collect();
        let ordinal = per_frame.entry(original).or_default();
        let stream = ((original as u64) << 16) | *ordinal;
        *ordinal += 1;
        let points = if structure.closed {
            jitter_cloud(&contour, half_spacing, params, stream)
        } else {
            revolve_arc(&contour, frame.centroid, params.revolution_steps)
        };
        frame
            .structures
            .push(Structure::new(structure.kind, structure.closed, points));
    }
    if dropped > 0 {
        eprintln!("Warning: dropped {dropped} structure(s) whose frame has no lumen contour");
    }
    Ok(())
}

fn print_success_message(input_data: InputData, from_path: bool) {
    use ContourType::*;

    println!(
        "\n✅ Successfully built geometry from {}",
        if from_path { "path" } else { "input data" }
    );

    let check = |present: bool| if present { "✅" } else { "❌" };

    println!("-----------------------------------------");
    println!("{} {}", check(!input_data.lumen.is_empty()), Lumen);
    println!("{} {}", check(input_data.eem.is_some()), Eem);
    println!("{} {}", check(true), Catheter);
    println!(
        "{} Structures ({})",
        check(!input_data.structures.is_empty()),
        input_data.structures.len()
    );
    println!("-----------------------------------------");
    println!("Label: {}", input_data.label);
    println!(
        "Diastole phase: {}",
        if input_data.diastole { "Yes" } else { "No" }
    );
    println!();
}

#[cfg(test)]
mod io_tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::path::Path;

    const NUM_POINTS_CATHETER: u32 = 20;

    #[test]
    fn test_full_directory_all_frames_explicit_types() {
        use std::collections::HashMap;
        use std::path::Path;

        let result = build_geometry_from_inputdata(
            None,
            Some(Path::new("data/fixtures/ivus_full")),
            "full",
            true,
            (4.5, 4.5),
            0.5,
            NUM_POINTS_CATHETER,
        );

        assert!(result.is_ok());
        let geometry = result.unwrap();
        let n_frames = geometry.frames.len();
        assert!(n_frames > 0, "expected at least one frame");

        let wanted: Vec<ContourType> =
            vec![ContourType::Lumen, ContourType::Eem, ContourType::Catheter];

        let mut ids_map: HashMap<ContourType, Vec<u32>> = HashMap::new();
        let mut orig_map: HashMap<ContourType, Vec<u32>> = HashMap::new();

        for (idx, frame) in geometry.frames.iter().enumerate() {
            ids_map
                .entry(ContourType::Lumen)
                .or_default()
                .push(frame.lumen.id);
            orig_map
                .entry(ContourType::Lumen)
                .or_default()
                .push(frame.lumen.original_frame);

            for wanted_type in &wanted {
                if *wanted_type == ContourType::Lumen {
                    continue;
                }
                match frame.extras.get(wanted_type) {
                    Some(contour) => {
                        ids_map.entry(*wanted_type).or_default().push(contour.id);
                        orig_map
                            .entry(*wanted_type)
                            .or_default()
                            .push(contour.original_frame);
                    }
                    None => panic!("Frame {idx} missing expected contour type {wanted_type:?}"),
                }
            }
        }

        for (ctype, vec) in &ids_map {
            assert_eq!(
                vec.len(),
                n_frames,
                "ids vector for {:?} has length {}, expected {}",
                ctype,
                vec.len(),
                n_frames
            );
        }
        for (ctype, vec) in &orig_map {
            assert_eq!(
                vec.len(),
                n_frames,
                "original_frame vector for {:?} has length {}, expected {}",
                ctype,
                vec.len(),
                n_frames
            );
        }

        let types: Vec<ContourType> = ids_map.keys().cloned().collect();
        for i in 0..n_frames {
            let first_type = &types[0];
            let first_id = ids_map[first_type][i];
            let first_of = orig_map[first_type][i];

            for t in &types[1..] {
                let id = ids_map.get(t).unwrap()[i];
                let of = orig_map.get(t).unwrap()[i];
                assert_eq!(
                    id, first_id,
                    "mismatched id at frame {i}: {first_type:?} has id {first_id}, but {t:?} has {id}"
                );
                assert_eq!(
                    of, first_of,
                    "mismatched original_frame at frame {i}: {first_type:?} has {first_of}, but {t:?} has {of}"
                );
            }
        }
    }

    #[test]
    fn test_rest_directory_area_elliptic() {
        let geometry = build_geometry_from_inputdata(
            None,
            Some(Path::new("data/fixtures/ivus_rest")),
            "full",
            true,
            (4.5, 4.5),
            0.5,
            NUM_POINTS_CATHETER,
        )
        .unwrap();

        let (_, long) = geometry.frames[0].lumen.find_farthest_points();
        let (_, short) = geometry.frames[0].lumen.find_closest_opposite();

        assert_eq!(geometry.frames[0].lumen.original_frame, 385);
        assert_relative_eq!(geometry.frames[0].lumen.area(), 5.42, epsilon = 0.1);
        assert_relative_eq!(long, 5.2, epsilon = 0.1);
        assert_relative_eq!(short, 1.15, epsilon = 0.1);
        assert_relative_eq!(
            geometry.frames[0].lumen.elliptic_ratio(),
            4.52,
            epsilon = 0.1
        );
        assert_eq!(geometry.frames[0].lumen.aortic_thickness, Some(0.96));
        assert_eq!(geometry.frames[0].lumen.pulmonary_thickness, Some(1.68));
        assert_eq!(
            geometry.frames[0].reference_point.unwrap().frame_index,
            geometry.frames[0].lumen.original_frame
        );
    }

    #[test]
    fn test_build_attaches_structure_clouds_by_original_frame() -> anyhow::Result<()> {
        let mut input = InputData::process_directory(
            "data/fixtures/ivus_rest",
            HashMap::from([(ContourType::Lumen, "lumen")]),
            true,
            "rest",
        )?;
        let point = |frame: u32, x: f64| ContourPoint {
            frame_index: frame,
            point_index: 0,
            x,
            y: 2.0,
            z: -5.0,
            aortic: false,
        };
        let closed = |kind: StructureKind, frame: u32, n: usize| {
            Structure::new(kind, true, (0..n).map(|i| point(frame, i as f64)).collect())
        };
        input.structures = vec![
            closed(StructureKind::Calcification, 385, 3),
            closed(StructureKind::Calcification, 385, 2),
            closed(StructureKind::Lipid, 347, 1),
            closed(StructureKind::Lipid, 99_999, 1),
            Structure::new(
                StructureKind::Sidebranch,
                false,
                vec![point(347, 0.0), point(347, 1.0)],
            ),
        ];
        let build = |input: InputData| {
            build_geometry_from_inputdata(Some(input), None, "rest", true, (4.5, 4.5), 0.5, 20)
        };

        let geometry = build(input.clone())?;

        let copies = CloudParams::default().copies + 1;
        let spacing = geometry.median_spacing().unwrap();
        let total: usize = geometry.frames.iter().map(|f| f.structures.len()).sum();
        assert_eq!(total, 4, "a structure without lumen frame must be dropped");
        for frame in &geometry.frames {
            let sizes: Vec<usize> = frame.structures.iter().map(|s| s.points.len()).collect();
            let expected = match frame.lumen.original_frame {
                385 => vec![3 * copies, 2 * copies],
                347 => vec![copies, 2 * CloudParams::default().revolution_steps],
                _ => vec![],
            };
            assert_eq!(sizes, expected, "frame {}", frame.lumen.original_frame);
            for s in &frame.structures {
                for p in &s.points {
                    // jitter stays within half the spacing, a revolved arc spreads by its width
                    if s.closed {
                        assert!((p.z - frame.centroid.2).abs() <= spacing / 2.0 + 1e-9);
                    }
                    assert_eq!(p.frame_index, frame.lumen.points[0].frame_index);
                }
            }
        }

        // the same input gives the same clouds
        let again = build(input.clone())?;
        for (a, b) in geometry.frames.iter().zip(&again.frames) {
            assert_eq!(a.structures, b.structures);
        }

        // an open in-wall contour is an error
        input.structures = vec![Structure::new(
            StructureKind::Lipid,
            false,
            vec![point(347, 0.0), point(347, 1.0)],
        )];
        assert!(build(input).is_err());
        Ok(())
    }

    #[test]
    fn test_catheter_contour_properties() {
        let geometry = build_geometry_from_inputdata(
            None,
            Some(Path::new("data/fixtures/ivus_rest")),
            "test",
            true,
            (4.5, 4.5),
            0.5,
            NUM_POINTS_CATHETER,
        )
        .expect("Failed to load geometry");

        for frame in &geometry.frames {
            if let Some(catheter_contour) = frame.extras.get(&ContourType::Catheter) {
                assert_eq!(
                    catheter_contour.points.len(),
                    NUM_POINTS_CATHETER as usize,
                    "Incorrect number of catheter points"
                );

                assert_relative_eq!(
                    catheter_contour.centroid.unwrap().2,
                    frame.lumen.centroid.unwrap().2,
                    epsilon = 1e-6
                );
            }
        }
    }

    #[test]
    fn test_build_geometry_with_input_data() {
        let test_points = vec![ContourPoint {
            frame_index: 0,
            point_index: 0,
            x: 1.0,
            y: 2.0,
            z: 3.0,
            aortic: false,
        }];

        let input_data = InputData {
            lumen: test_points.clone(),
            eem: Some(test_points.clone()),
            record: None,
            reference_points: vec![ContourPoint {
                frame_index: 0,
                point_index: 0,
                x: 1.0,
                y: 2.0,
                z: 3.0,
                aortic: false,
            }],
            master_reference: None,
            label: "test".to_string(),
            diastole: true,
            structures: Vec::new(),
        };

        let geometry = build_geometry_from_inputdata(
            Some(input_data),
            None,
            "test_label",
            true,
            (0.0, 0.0),
            1.0,
            10,
        )
        .expect("Failed to build geometry from InputData");

        assert!(!geometry.frames.is_empty());
        assert_eq!(geometry.label, "test_label");
    }

    /// ivus_rest copied to a temp dir with references on 347 and 385, and on 999 (no lumen).
    fn rest_dir_with_references(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("multimodars_build_{name}"));
        std::fs::create_dir_all(&dir).unwrap();
        for f in ["diastolic_contours.csv", "combined_sorted_manual.csv"] {
            std::fs::copy(Path::new("data/fixtures/ivus_rest").join(f), dir.join(f)).unwrap();
        }
        std::fs::write(
            dir.join("diastolic_reference_points.csv"),
            "347\t3.0\t3.0\t22.0\n385\t3.5\t3.6\t24.5\n999\t1.0\t1.0\t1.0\n",
        )
        .unwrap();
        dir
    }

    #[test]
    fn test_build_stores_every_reference_and_the_master() -> anyhow::Result<()> {
        let dir = rest_dir_with_references("refs");
        let geometry =
            build_geometry_from_inputdata(None, Some(&dir), "refs", true, (4.5, 4.5), 0.5, 20)?;

        // 999 has no lumen and is dropped, the others sit on their own frame
        let mut refs = geometry.reference_frames();
        refs.sort();
        assert_eq!(refs, vec![347, 385]);
        for frame in &geometry.frames {
            if let Some(rp) = frame.reference_point {
                assert_eq!(rp.frame_index, frame.lumen.original_frame);
                assert_eq!(rp.z, frame.centroid.2);
            }
        }
        // default master is the most proximal reference, the first in frame order
        let first = geometry.reference_frames()[0];
        assert_eq!(geometry.master_reference, Some(first));
        assert_eq!(geometry.master_frame_idx()?, {
            geometry
                .frames
                .iter()
                .position(|f| f.reference_point.is_some())
                .unwrap()
        });

        // an explicit master is kept, a frame without a reference is an error
        let names = default_names();
        let mut input = InputData::process_directory(&dir, names.clone(), true, "refs")?;
        input.master_reference = Some(347);
        let geometry =
            build_geometry_from_inputdata(Some(input), None, "refs", true, (4.5, 4.5), 0.5, 20)?;
        assert_eq!(geometry.master_reference, Some(347));
        let idx = geometry.master_frame_idx()?;
        assert_eq!(geometry.frames[idx].lumen.original_frame, 347);

        let mut input = InputData::process_directory(&dir, names, true, "refs")?;
        input.master_reference = Some(367);
        assert!(build_geometry_from_inputdata(
            Some(input),
            None,
            "refs",
            true,
            (4.5, 4.5),
            0.5,
            20
        )
        .is_err());
        Ok(())
    }

    #[test]
    fn test_build_geometry_with_path() {
        let geometry = build_geometry_from_inputdata(
            None,
            Some(Path::new("data/fixtures/ivus_rest")),
            "path_test",
            true,
            (4.5, 4.5),
            0.5,
            20,
        );

        assert!(geometry.is_ok());
        let geometry = geometry.unwrap();
        assert!(!geometry.frames.is_empty());
        assert_eq!(geometry.label, "path_test");
    }

    #[test]
    fn test_error_on_no_input() {
        let result = build_geometry_from_inputdata(None, None, "test", true, (0.0, 0.0), 1.0, 10);

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Either input_data or path must be provided"));
    }
}
