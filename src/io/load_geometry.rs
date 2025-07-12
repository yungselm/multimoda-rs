use std::error::Error;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;

use crate::io::input::{Contour, ContourPoint};
use crate::io::Geometry;

/// Since process_case() ensures that all points are sorted counterclockwise
/// and aortic points are always on the right side, the original geometry can be rebuild
/// using simple operations.
pub fn rebuild_geometry(contour_path: &str, catheter_path: &str, wall_path: &str) -> Geometry {
    let contours = read_obj_mesh(&contour_path).unwrap();
    let catheter = read_obj_mesh(&catheter_path).unwrap();
    let _walls =  read_obj_mesh(&wall_path).unwrap();

    let n = contours.len() as u32;

    let mut geometry = Geometry {
        contours: Vec::new(),
        catheter: Vec::new(),
        walls: Vec::new(),
        reference_point: ContourPoint {
            frame_index: n,
            point_index: n,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            aortic: false,
        },
        label: "None".to_string(),
    };

    let mut new_contours = Vec::<Contour>::new();

    // Process contours
    for (frame_idx, input_points) in contours.into_iter() {
        let centroid = Contour::compute_centroid(&input_points);

        let mut contour_new = Contour {
            id: frame_idx,
            points: Vec::new(),
            centroid,
            aortic_thickness: None,
            pulmonary_thickness: None,
        };

        let mut points = input_points;

        // Find the point with the highest y-value
        if let Some((max_idx, _)) = points
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.y.partial_cmp(&b.y).unwrap())
        {
            points.rotate_left(max_idx);
        }

        let l = points.len();

        // Assign indices and set frame_index and aortic flag
        for (i, point) in points.iter_mut().enumerate() {
            point.point_index = i as u32;
            point.frame_index = frame_idx;
            point.aortic = i >= (l / 2); // >= works since ordere counterclockwise, aortic to right
        }

        contour_new.points = points;
        new_contours.push(contour_new);
    }

    let mut new_catheter = Vec::<Contour>::new();

    // Similar correction for catheter processing
    for (frame_idx, input_points) in catheter.into_iter() {
        let centroid = Contour::compute_centroid(&input_points);

        let mut catheter_new = Contour {
            id: frame_idx,
            points: Vec::new(),
            centroid,
            aortic_thickness: None,
            pulmonary_thickness: None,
        };

        let mut points = input_points;

        if let Some((max_idx, _)) = points
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.y.partial_cmp(&b.y).unwrap())
        {
            points.rotate_left(max_idx);
        }

        for (i, point) in points.iter_mut().enumerate() {
            point.point_index = i as u32;
            point.frame_index = frame_idx;
            point.aortic = false;
        }

        catheter_new.points = points;
        new_catheter.push(catheter_new);
    }

    geometry.contours = new_contours;
    geometry.catheter = new_catheter;

    geometry
}

fn read_obj_mesh(filename: &str) -> Result<Vec<(u32, Vec<ContourPoint>)>, Box<dyn Error>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);

    let mut vertices: Vec<ContourPoint> = Vec::new();
    // This will be set based on the first face encountered.
    let mut points_per_contour: Option<usize> = None;

    for line_result in reader.lines() {
        let line = line_result?;
        let trimmed = line.trim();
        if trimmed.starts_with("v ") {
            // This is a vertex line. Split on whitespace.
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 4 {
                let x = parts[1].parse::<f64>()?;
                let y = parts[2].parse::<f64>()?;
                let z = parts[3].parse::<f64>()?;
                vertices.push(ContourPoint {
                    point_index: 0,
                    frame_index: 0,
                    x,
                    y,
                    z,
                    aortic: false,
                });
            }
        } else if trimmed.starts_with("f ") && points_per_contour.is_none() {
            // Use the first face line to deduce the number of vertices per contour.
            // A face line written by write_obj_mesh has the form:
            // f {v1}/{v1}/{v1} {v2}/{v2}/{v2} {v3}/{v3}/{v3}
            // where v1 comes from the first contour and v3 from the next.
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 4 {
                // Extract the first vertex index from parts[1] and parts[3].
                let v1_str = parts[1].split('/').next().unwrap();
                let v3_str = parts[3].split('/').next().unwrap();
                let v1: usize = v1_str.parse()?;
                let v3: usize = v3_str.parse()?;
                if v3 > v1 {
                    points_per_contour = Some(v3 - v1);
                }
            }
        }
        // Other lines (vt, vn, mtllib, usemtl, etc.) are ignored.
    }

    // If no face line was found, assume all vertices form one single contour.
    let points_per_contour = points_per_contour.unwrap_or(vertices.len());

    // Ensure the total number of vertices divides evenly into contours.
    if vertices.len() % points_per_contour != 0 {
        return Err(format!(
            "Vertex count {} is not divisible by points per contour {}.",
            vertices.len(),
            points_per_contour
        )
        .into());
    }

    let num_contours = vertices.len() / points_per_contour;
    let mut contours: Vec<(u32, Vec<ContourPoint>)> = Vec::with_capacity(num_contours);

    for i in 0..num_contours {
        let start = i * points_per_contour;
        let end = start + points_per_contour;
        let mut contour = vertices[start..end].to_vec();
        // Set the contour's frame_index to the contour index.
        for point in &mut contour {
            point.frame_index = i as u32;
        }
        contours.push((i as u32, contour));
    }

    Ok(contours)
}
