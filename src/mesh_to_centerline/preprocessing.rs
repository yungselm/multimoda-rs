use nalgebra::Vector3;

use crate::io::input::{read_centerline_txt, ContourPoint};
use crate::io::input::{Centerline, CenterlinePoint};
use crate::io::load_geometry::rebuild_geometry;
use crate::io::Geometry;

pub fn smooth_resample_centerline(centerline: Centerline, ref_mesh: &Geometry) -> Centerline {
    // Extract z-coordinates from reference mesh contours
    let mut z_refs: Vec<f64> = ref_mesh.contours.iter().map(|c| c.centroid.2).collect();
    z_refs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    z_refs.dedup();

    // Sort centerline points by z
    let mut sorted_centerline = centerline.points.clone();
    sorted_centerline.sort_by(|a, b| a.contour_point.z.partial_cmp(&b.contour_point.z).unwrap());

    // Handle empty centerline
    if sorted_centerline.is_empty() {
        return Centerline { points: Vec::new() };
    }

    let min_z = sorted_centerline[0].contour_point.z;
    let max_z = sorted_centerline.last().unwrap().contour_point.z;

    // Filter z_refs to be within [min_z, max_z]
    let z_refs: Vec<f64> = z_refs
        .into_iter()
        .filter(|&z| z >= min_z && z <= max_z)
        .collect();

    let mut contour_points = Vec::with_capacity(z_refs.len());

    if sorted_centerline.len() == 1 {
        let p0 = &sorted_centerline[0];
        for (index, _) in z_refs.iter().enumerate() {
            contour_points.push(ContourPoint {
                frame_index: index as u32,
                point_index: index as u32,
                x: p0.contour_point.x,
                y: p0.contour_point.y,
                z: p0.contour_point.z,
                aortic: false,
            });
        }
    } else {
        for (index, &z) in z_refs.iter().enumerate() {
            // Binary search to find the segment
            let i = match sorted_centerline
                .binary_search_by(|point| point.contour_point.z.partial_cmp(&z).unwrap())
            {
                Ok(i) => i,
                Err(i) => {
                    if i == 0 {
                        0
                    } else if i == sorted_centerline.len() {
                        sorted_centerline.len() - 2
                    } else {
                        i - 1
                    }
                }
            };

            let p0 = &sorted_centerline[i];
            let p1 = &sorted_centerline[i + 1];

            if p0.contour_point.z == p1.contour_point.z {
                contour_points.push(ContourPoint {
                    frame_index: index as u32,
                    point_index: index as u32,
                    x: p0.contour_point.x,
                    y: p0.contour_point.y,
                    z: p0.contour_point.z,
                    aortic: false,
                });
            } else {
                let t = (z - p0.contour_point.z) / (p1.contour_point.z - p0.contour_point.z);
                let x = p0.contour_point.x + t * (p1.contour_point.x - p0.contour_point.x);
                let y = p0.contour_point.y + t * (p1.contour_point.y - p0.contour_point.y);

                contour_points.push(ContourPoint {
                    frame_index: index as u32,
                    point_index: index as u32,
                    x,
                    y,
                    z,
                    aortic: false,
                });
            }
        }
    }

    // Build new centerline from contour points
    if contour_points.is_empty() {
        Centerline { points: Vec::new() }
    } else if contour_points.len() == 1 {
        Centerline {
            points: vec![CenterlinePoint {
                contour_point: contour_points[0].clone(),
                normal: Vector3::zeros(),
            }],
        }
    } else {
        Centerline::from_contour_points(contour_points)
    }
}

pub fn prepare_data_3d_alignment(
    state: &str,
    centerline_path: &str,
    input_dir: &str,
    interpolation_steps: usize,
) -> anyhow::Result<(Centerline, Geometry, Geometry)> {
    let raw_centerline = read_centerline_txt(&centerline_path)?;
    let centerline = Centerline::from_contour_points(raw_centerline);

    // Process the reference mesh: mesh_000_rest.obj or mesh_000_stress.obj
    // as program currently set up will always be mesh_000_ so no magic number
    let ref_mesh_path = format!("{}/mesh_000_{}.obj", input_dir, state);
    let catheter_path = format!("{}/catheter_000_{}.obj", input_dir, state);

    todo!("Implement rereading of wall contours.");
    let mut reference_mesh = rebuild_geometry(&ref_mesh_path, &catheter_path, &ref_mesh_path); // implement later!!

    reference_mesh.contours.reverse(); // reverse contours since for centerline alignment it is beneficial to have 0 for the ostium
    for (index, contour) in reference_mesh.contours.iter_mut().enumerate() {
        contour.id = index as u32;
        for point in &mut contour.points {
            point.frame_index = index as u32; // Match the parent contour's new ID
        }
    }
    reference_mesh.catheter.reverse();

    for (index, catheter) in reference_mesh.catheter.iter_mut().enumerate() {
        catheter.id = index as u32;
        for point in &mut catheter.points {
            point.frame_index = index as u32;
        }
    }

    let ((pt1, pt2), _) = reference_mesh.contours[0].find_closest_opposite();

    let reference_point = if pt1.aortic { pt1.clone() } else { pt2.clone() };

    reference_mesh.label = format!("diastole_{}", state);
    reference_mesh.reference_point = reference_point;

    println!(
        "Reference point after reloading: {:?}",
        &reference_mesh.reference_point
    );

    // ----- Process the reference mesh: mesh_029_rest.obj or mesh_029_stress.obj -----
    let ref_mesh_path_sys = format!(
        "{}/mesh_{:03}_{}.obj",
        input_dir,
        (interpolation_steps + 1),
        state
    );
    let catheter_path_sys = format!(
        "{}/catheter_{:03}_{}.obj",
        input_dir,
        (interpolation_steps + 1),
        state
    );

    let mut reference_mesh_sys = rebuild_geometry(&ref_mesh_path_sys, &catheter_path_sys, &ref_mesh_path_sys); // Fix later !!!
    reference_mesh_sys.contours.reverse();
    for (index, contour) in reference_mesh_sys.contours.iter_mut().enumerate() {
        contour.id = index as u32;
        for point in &mut contour.points {
            point.frame_index = index as u32;
        }
    }
    reference_mesh_sys.catheter.reverse();
    for (index, catheter) in reference_mesh_sys.catheter.iter_mut().enumerate() {
        catheter.id = index as u32;
        for point in &mut catheter.points {
            point.frame_index = index as u32;
        }
    }

    let ((pt1, pt2), _) = reference_mesh_sys.contours[0].find_closest_opposite();

    let reference_point = if pt1.aortic { pt1.clone() } else { pt2.clone() };

    reference_mesh_sys.label = format!("systole_{}", state);
    reference_mesh_sys.reference_point = reference_point;

    Ok((centerline, reference_mesh, reference_mesh_sys))
}

pub fn read_interpolated_meshes(
    state: &str,
    input_dir: &str,
    interpolation_steps: usize,
) -> Vec<Geometry> {
    let mut geometries = Vec::new();

    for i in 1..(interpolation_steps + 1) {
        let mesh_path = format!("{}/mesh_{:03}_{}.obj", input_dir, i, state);
        let catheter_path = format!("{}/catheter_{:03}_{}.obj", input_dir, i, state);

        let mut mesh = rebuild_geometry(&mesh_path, &catheter_path, &mesh_path); // Fix later!!!!
        mesh.contours.reverse();
        for (index, contour) in mesh.contours.iter_mut().enumerate() {
            contour.id = index as u32;
            for point in &mut contour.points {
                point.frame_index = index as u32;
            }
        }
        mesh.catheter.reverse();

        for (index, catheter) in mesh.catheter.iter_mut().enumerate() {
            catheter.id = index as u32;
            for point in &mut catheter.points {
                point.frame_index = index as u32;
            }
        }
        geometries.push(mesh)
    }
    geometries
}
