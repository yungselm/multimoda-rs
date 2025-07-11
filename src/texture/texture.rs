use crate::io::{input::Contour, Geometry};
use image::{ImageBuffer, Rgb, Rgba};
use std::error::Error;

pub fn compute_uv_coordinates(contours: &Vec<Contour>) -> Vec<(f64, f64)> {
    if contours.is_empty() || contours[0].points.is_empty() {
        return Vec::new(); // or handle as needed
    }

    let points_per_contour = contours[0].points.len();
    let num_contours = contours.len();
    let mut uvs = Vec::with_capacity(num_contours * points_per_contour);

    for (contour_index, group) in contours.iter().enumerate() {
        if group.points.is_empty() {
            continue;
        }
        let v = (contour_index as f64 + 0.5) / num_contours as f64;
        for (point_index, _point) in group.points.iter().enumerate() {
            let u = (point_index as f64 + 0.5) / points_per_contour as f64;
            uvs.push((u, v));
        }
    }

    uvs
}

/// This function takes in a baseline Geometry and a second Geometry, and
/// then calculates the displacement for every point for the contours and
/// the catheter seperately. Therefore returning a Vec<f64, f64> where the
/// first entry is the displacements for the contours and the second for the
/// catheter.
pub fn compute_displacements(mesh: &Geometry, diastole: &Geometry) -> Vec<f64> {
    mesh.contours
        .iter()
        .zip(diastole.contours.iter())
        .flat_map(|(contour, diastole_contour)| {
            contour
                .points
                .iter()
                .zip(diastole_contour.points.iter())
                .map(|(point, diastole_point)| {
                    let dx = point.x - diastole_point.x;
                    let dy = point.y - diastole_point.y;
                    let dz = point.z - diastole_point.z;
                    (dx * dx + dy * dy + dz * dz).sqrt()
                })
        })
        .collect()
}

pub fn create_displacement_texture(
    displacements: &[f64],
    width: u32,
    height: u32,
    max_displacement: f64,
    filename: &str,
) -> Result<(), Box<dyn Error>> {
    let mut img = ImageBuffer::new(width, height);
    for (i, &disp) in displacements.iter().enumerate() {
        let x = (i % width as usize) as u32;
        // Flip Y-axis by subtracting from height - 1
        let y = (height - 1) - (i / width as usize) as u32;
        let normalized = (disp / max_displacement).clamp(0.0, 1.0);
        let color = Rgb([
            (normalized * 255.0) as u8,
            0,
            ((1.0 - normalized) * 255.0) as u8,
        ]);
        img.put_pixel(x, y, color);
    }
    img.save(filename)?;
    Ok(())
}

pub fn create_black_texture(width: u32, height: u32, filename: &str) -> Result<(), Box<dyn Error>> {
    let black = Rgb([0u8, 0u8, 0u8]); // Ensure pixel values are u8
    let img = ImageBuffer::from_pixel(width, height, black);
    img.save(filename)?; // Save as PNG
    Ok(())
}

pub fn create_transparent_texture(
    width: u32,
    height: u32,
    percent_transparent: f64,
    filename: &str,
) -> Result<(), Box<dyn Error>> {
    let transparency_value = (255.0 - (percent_transparent * 255.0)) as u8;
    let semi_transparent = Rgba([0u8, 0u8, 0u8, transparency_value]);
    let img = ImageBuffer::from_pixel(width, height, semi_transparent);
    img.save(filename)?;
    Ok(())
}
