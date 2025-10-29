pub mod input;
pub mod output;
pub mod geometry;
pub mod wall;
mod integrity_check;

use input::{InputData, ContourPoint};
use geometry::{Contour, Frame, Geometry, ContourType};
use integrity_check::check_geometry_integrity;
use std::path::Path;
use std::collections::HashMap;

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
        return Err(anyhow::anyhow!("Either input_data or path must be provided"));
    };

    let lumen_contours = Contour::build_contour(
        input_data.lumen,
        input_data.record.clone(),
        ContourType::Lumen,
    )?;

    let eem_contours = if let Some(eem_points) = input_data.eem {
        Contour::build_contour(eem_points, None, ContourType::Eem)?
    } else {
        Vec::new()
    };

    let calcification_contours = if let Some(calc_points) = input_data.calcification {
        Contour::build_contour(calc_points, None, ContourType::Calcification)?
    } else {
        Vec::new()
    };

    let sidebranch_contours = if let Some(side_points) = input_data.sidebranch {
        Contour::build_contour(side_points, None, ContourType::Sidebranch)?
    } else {
        Vec::new()
    };

    let mut frame_map: HashMap<u32, Frame> = HashMap::new();

    // Process lumen contours (mandatory)
    for contour in lumen_contours {
        contour.compute_centroid();
        let frame_id = contour.id;
        let mut frame = Frame {
            id: frame_id,
            centroid: contour.centroid.unwrap_or((0.0, 0.0, 0.0)),
            lumen: contour,
            extras: HashMap::new(),
            reference_point: None,
        };
        
        if input_data.ref_point.frame_index == frame_id {
            frame.reference_point = Some(input_data.ref_point);
        }
        
        frame_map.insert(frame_id, frame);
    }

    // Add other contour types to frames
    for contour in eem_contours {
        if let Some(frame) = frame_map.get_mut(&contour.id) {
            frame.extras.insert(ContourType::Eem, contour.compute_centroid());
        }
    }

    for contour in calcification_contours {
        if let Some(frame) = frame_map.get_mut(&contour.id) {
            frame.extras.insert(ContourType::Calcification, contour.compute_centroid());
        }
    }

    for contour in sidebranch_contours {
        if let Some(frame) = frame_map.get_mut(&contour.id) {
            frame.extras.insert(ContourType::Sidebranch, contour.compute_centroid());
        }
    }

    // Create catheter contours if requested
    if n_points > 0 {
        let all_points: Vec<ContourPoint> = frame_map.values()
            .flat_map(|frame| frame.lumen.points.iter().cloned())
            .collect();
            
        let catheter_points = Frame::create_catheter_points(&all_points, image_center, radius, n_points);
        let catheter_contours = Contour::build_contour(catheter_points, None, ContourType::Catheter)?;
        
        for contour in catheter_contours {
            if let Some(frame) = frame_map.get_mut(&contour.id) {
                frame.extras.insert(ContourType::Catheter, contour.compute_centroid());
            }
        }
    }

    // Convert frame map to sorted vector
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

    check_geometry_integrity(&geometry)?;

    Ok(geometry)
}

fn print_success_message() {
    todo!()
}

// Helper function for backward compatibility
pub fn load_geometry(
    input_dir: &str,
    label: String,
    diastole: bool,
    image_center: (f64, f64),
    radius: f64,
    n_points: u32,
) -> anyhow::Result<Geometry> {
    let path = Path::new(input_dir);
    build_geometry_from_inputdata(
        None,
        Some(path),
        &label,
        diastole,
        image_center,
        radius,
        n_points,
    )
}

#[cfg(test)]
mod input_tests {
    use super::*;
    use approx::assert_relative_eq;
    use serde_json::Value;
    use std::fs::File;

    const NUM_POINTS_CATHETER: usize = 20;

    fn load_test_manifest(mode: &str) -> Value {
        let manifest_path = format!("data/fixtures/{}_csv_files/test_manifest.json", mode);
        let file = File::open(manifest_path).expect("Failed to open manifest");
        serde_json::from_reader(file).expect("Failed to parse manifest")
    }

    #[test]
    fn test_rest_diastolic_config_match() {
        let geometry = load_geometry(
            "data/fixtures/rest_csv_files",
            "test".to_string(),
            true,
            (4.5, 4.5),
            0.5,
            20,
        ).expect("Failed to load geometry");

        let manifest = load_test_manifest("rest");
        let dia_config = &manifest["dia"];

        assert_eq!(
            geometry.frames.len(),
            dia_config["num_contours"].as_u64().unwrap() as usize,
            "Frame count mismatch"
        );

        // Test elliptic ratios and areas for each frame
        for (i, frame) in geometry.frames.iter().enumerate() {
            // Verify elliptic ratio for lumen
            let expected_ratio = dia_config["elliptic_ratios"][i].as_f64().unwrap();
            assert_relative_eq!(
                frame.lumen.elliptic_ratio(),
                expected_ratio,
                epsilon = 0.1
            );

            // Verify area for lumen
            let expected_area = dia_config["areas"][i].as_f64().unwrap();
            assert_relative_eq!(frame.lumen.area(), expected_area, epsilon = 3.0);

            // Verify aortic thickness
            let expected_thickness = match dia_config["aortic_thickness"][i].as_f64() {
                Some(v) => Some(v),
                None => None,
            };
            assert_eq!(
                frame.lumen.aortic_thickness, expected_thickness,
                "Aortic thickness mismatch at index {}",
                i
            );
        }
    }

    #[test]
    fn test_catheter_contour_properties() {
        let geometry = load_geometry(
            "data/fixtures/rest_csv_files",
            "test".to_string(),
            true,
            (4.5, 4.5),
            0.5,
            20,
        ).expect("Failed to load geometry");

        // Verify catheter contours exist in frames
        for frame in &geometry.frames {
            if let Some(catheter_contour) = frame.extras.get(&ContourType::Catheter) {
                assert_eq!(
                    catheter_contour.points.len(),
                    NUM_POINTS_CATHETER,
                    "Incorrect number of catheter points"
                );
                
                // Verify z-coordinate consistency
                assert_relative_eq!(
                    catheter_contour.centroid.unwrap().2, 
                    frame.lumen.centroid.unwrap().2, 
                    epsilon = 1e-6
                );
            }
        }
    }
}