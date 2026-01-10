pub mod geometry;
pub mod input;
mod integrity_check;
pub mod output;

use geometry::{Contour, ContourType, Frame, Geometry};
use input::{ContourPoint, InputData};
use integrity_check::check_geometry_integrity;
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
        // Default mapping for typical AIVUS-CAA structure
        let mut names = HashMap::new();
        names.insert(ContourType::Lumen, "lumen");
        names.insert(ContourType::Eem, "eem");
        names.insert(ContourType::Calcification, "calcium");
        names.insert(ContourType::Sidebranch, "branch");
        names.insert(ContourType::Catheter, "catheter");

        InputData::process_directory(path, names, diastole)?
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

    if let Some(calc_points) = &input_data.calcification {
        for point in calc_points {
            all_original_frames.insert(point.frame_index);
        }
    }

    if let Some(side_points) = &input_data.sidebranch {
        for point in side_points {
            all_original_frames.insert(point.frame_index);
        }
    }

    all_original_frames.insert(input_data.ref_point.frame_index);

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

    let calcification_contours = if let Some(calc_points) = input_data.calcification {
        Contour::build_contour_with_mapping(
            calc_points,
            None,
            ContourType::Calcification,
            &frame_mapping,
        )?
    } else {
        Vec::new()
    };

    let sidebranch_contours = if let Some(side_points) = input_data.sidebranch {
        Contour::build_contour_with_mapping(
            side_points,
            None,
            ContourType::Sidebranch,
            &frame_mapping,
        )?
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
        };

        if let Some(&mapped_frame_id) = frame_mapping.get(&input_data.ref_point.frame_index) {
            if mapped_frame_id == frame_id {
                frame.reference_point = Some(input_data.ref_point);
            }
        }

        frame_map.insert(frame_id, frame);
    }

    for mut contour in eem_contours {
        contour.compute_centroid();
        if let Some(frame) = frame_map.get_mut(&contour.id) {
            frame.extras.insert(ContourType::Eem, contour);
        }
    }

    for mut contour in calcification_contours {
        contour.compute_centroid();
        if let Some(frame) = frame_map.get_mut(&contour.id) {
            frame.extras.insert(ContourType::Calcification, contour);
        }
    }

    for mut contour in sidebranch_contours {
        contour.compute_centroid();
        if let Some(frame) = frame_map.get_mut(&contour.id) {
            frame.extras.insert(ContourType::Sidebranch, contour);
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

    check_geometry_integrity(&geometry)?;

    let from_path = path.is_some();
    print_success_message(print_input_data, from_path);

    Ok(geometry)
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
    println!(
        "{} {}",
        check(input_data.calcification.is_some()),
        Calcification
    );
    println!("{} {}", check(input_data.sidebranch.is_some()), Sidebranch);
    println!("{} {}", check(true), Catheter);
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

        // choose the types you expect to exist in every frame
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
                        ids_map
                            .entry(wanted_type.clone())
                            .or_default()
                            .push(contour.id);
                        orig_map
                            .entry(wanted_type.clone())
                            .or_default()
                            .push(contour.original_frame);
                    }
                    None => panic!(
                        "Frame {} missing expected contour type {:?}",
                        idx, wanted_type
                    ),
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
                    "mismatched id at frame {}: {:?} has id {}, but {:?} has {}",
                    i, first_type, first_id, t, id
                );
                assert_eq!(
                    of, first_of,
                    "mismatched original_frame at frame {}: {:?} has {}, but {:?} has {}",
                    i, first_type, first_of, t, of
                );
            }
        }

        println!(
            "Checked {} frames and {} explicit contour types; all ids/original_frames match per-frame.",
            n_frames,
            types.len()
        );
    }

    #[test]
    fn test_rest_directory_area_elliptic() {
        use std::path::Path;

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
        use input::InputData;

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
            calcification: None,
            sidebranch: None,
            record: None,
            ref_point: ContourPoint {
                frame_index: 0,
                point_index: 0,
                x: 1.0,
                y: 2.0,
                z: 3.0,
                aortic: false,
            },
            label: "test".to_string(),
            diastole: true,
        };

        let geometry = build_geometry_from_inputdata(
            Some(input_data),
            None, // No path provided, use InputData
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
