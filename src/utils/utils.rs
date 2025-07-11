use csv::Writer;
use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::io::input::Contour;
use crate::io::Geometry;

#[allow(dead_code)]
pub fn write_geometry_to_csv<P: AsRef<Path>>(
    path: P,
    geometry: &Geometry,
) -> Result<(), Box<dyn Error>> {
    let mut wtr = Writer::from_path(path)?;

    // Write header
    wtr.write_record(&[
        "label",
        "source",
        "contour_id",
        "point_index",
        "frame_index",
        "x",
        "y",
        "z",
        "aortic",
        "aortic_thickness",
        "pulmonary_thickness",
    ])?;

    // Helper to write from a source (contours or catheter)
    let write_contours = |source: &str,
                          contours: &Vec<Contour>,
                          wtr: &mut Writer<std::fs::File>|
     -> Result<(), Box<dyn Error>> {
        for contour in contours {
            for point in contour.points.iter() {
                let record = vec![
                    geometry.label.clone(),
                    source.to_string(),
                    contour.id.to_string(),
                    point.point_index.to_string(),
                    point.frame_index.to_string(),
                    point.x.to_string(),
                    point.y.to_string(),
                    point.z.to_string(),
                    point.aortic.to_string(),
                    contour
                        .aortic_thickness
                        .map_or("None".to_string(), |v| v.to_string()),
                    contour
                        .pulmonary_thickness
                        .map_or("None".to_string(), |v| v.to_string()),
                ];

                wtr.write_record(&record)?;
            }
        }
        Ok(())
    };

    write_contours("contour", &geometry.contours, &mut wtr)?;
    write_contours("catheter", &geometry.catheter, &mut wtr)?;

    wtr.flush()?;
    Ok(())
}

#[allow(dead_code)]
pub fn write_contour_to_csv<P: AsRef<Path>>(
    path: P,
    contour: &Contour,
) -> Result<(), Box<dyn Error>> {
    let mut wtr = Writer::from_path(path)?;

    // Write header
    wtr.write_record(&[
        "source",
        "contour_id",
        "point_index",
        "frame_index",
        "x",
        "y",
        "z",
        "aortic",
        "aortic_thickness",
        "pulmonary_thickness",
    ])?;

    // Write all points of the single contour
    for point in &contour.points {
        let record = vec![
            "contour".to_string(),
            contour.id.to_string(),
            point.point_index.to_string(),
            point.frame_index.to_string(),
            point.x.to_string(),
            point.y.to_string(),
            point.z.to_string(),
            point.aortic.to_string(),
            contour
                .aortic_thickness
                .map_or("None".to_string(), |v| v.to_string()),
            contour
                .pulmonary_thickness
                .map_or("None".to_string(), |v| v.to_string()),
        ];
        wtr.write_record(&record)?;
    }

    wtr.flush()?;
    Ok(())
}

#[allow(dead_code)]
pub fn write_debug_obj_mesh(contours: &Vec<Contour>, filename: &str) -> Result<(), Box<dyn Error>> {
    let sorted_contours = contours.to_owned();

    // Validation remains the same
    if sorted_contours.len() < 2 {
        return Err("Need at least two contours to create a mesh.".into());
    }

    let points_per_contour = sorted_contours[0].points.len();
    for contour in &sorted_contours {
        if contour.points.len() != points_per_contour {
            return Err("All contours must have the same number of points.".into());
        }
    }

    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);
    let mut vertex_offsets = Vec::new();
    let mut current_offset = 1;

    // Write vertices only
    for contour in &sorted_contours {
        vertex_offsets.push(current_offset);
        for point in &contour.points {
            writeln!(writer, "v {} {} {}", point.x, point.y, point.z)?;
            current_offset += 1;
        }
    }

    // Write faces without UVs or normals
    for c in 0..(sorted_contours.len() - 1) {
        let offset1 = vertex_offsets[c];
        let offset2 = vertex_offsets[c + 1];
        for j in 0..points_per_contour {
            let j_next = (j + 1) % points_per_contour;

            // First triangle
            let v1 = offset1 + j;
            let v2 = offset1 + j_next;
            let v3 = offset2 + j;
            writeln!(writer, "f {} {} {}", v1, v2, v3)?;

            // Second triangle
            let v4 = offset2 + j;
            let v5 = offset1 + j_next;
            let v6 = offset2 + j_next;
            writeln!(writer, "f {} {} {}", v4, v5, v6)?;
        }
    }

    println!("Debug OBJ mesh written to {}", filename);
    Ok(())
}
