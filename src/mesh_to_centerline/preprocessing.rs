use crate::io::input::read_centerline_txt;
use crate::io::Geometry;
use crate::io::load_geometry::rebuild_geometry;
use crate::io::input::Centerline;

pub fn prepare_data_3d_alignment(
    state: &str,
    centerline_path: &str,
    input_dir: &str,
    interpolation_steps: usize, 
) -> anyhow::Result<(Centerline, Geometry, Geometry)> {
    // ----- Build the common centerline -----
    let raw_centerline = read_centerline_txt(&centerline_path)?;
    let centerline = Centerline::from_contour_points(raw_centerline);

    // ----- Process the reference mesh: mesh_000_rest.obj or mesh_000_stress.obj -----
    let ref_mesh_path = format!("{}/mesh_000_{}.obj", input_dir, state);
    let catheter_path = format!("{}/catheter_000_{}.obj", input_dir, state);

    let mut reference_mesh = rebuild_geometry(&ref_mesh_path, &catheter_path);
    
    reference_mesh.contours.reverse(); // reverse contours since for centerline alignment it is beneficial to have 0 for the ostium
    // After reversing the contours, update their IDs to match their new indices
    for (index, contour) in reference_mesh.contours.iter_mut().enumerate() {
        contour.id = index as u32;
        // Update frame_index for all points in this contour
        for point in &mut contour.points { // Use &mut to modify points in-place
            point.frame_index = index as u32; // Match the parent contour's new ID
        }
    }
    reference_mesh.catheter.reverse();
    // After reversing the contours, update their IDs to match their new indices
    for (index, catheter) in reference_mesh.catheter.iter_mut().enumerate() {
        catheter.id = index as u32;
        // Update frame_index for all points in this contour
        for point in &mut catheter.points { // Use &mut to modify points in-place
            point.frame_index = index as u32; // Match the parent contour's new ID
        }
    }

    let ((pt1, pt2), _) = reference_mesh.contours[0].find_closest_opposite();
    
    let reference_point = if pt1.aortic {
        pt1.clone()
    } else {
        pt2.clone()
    };
    
    reference_mesh.label = format!("diastole_{}", state);
    reference_mesh.reference_point = reference_point;

    println!("Reference point after reloading: {:?}", &reference_mesh.reference_point);

    // ----- Process the reference mesh: mesh_029_rest.obj or mesh_029_stress.obj -----
    let ref_mesh_path_sys = format!("{}/mesh_{:03}_{}.obj", input_dir, (interpolation_steps + 1), state);
    let catheter_path_sys = format!("{}/catheter_{:03}_{}.obj", input_dir, (interpolation_steps + 1), state);

    let mut reference_mesh_sys = rebuild_geometry(&ref_mesh_path_sys, &catheter_path_sys);
    reference_mesh_sys.contours.reverse(); // reverse contours since for centerline alignment it is beneficial to have 0 for the ostium
    // After reversing the contours for reference_mesh_sys
    for (index, contour) in reference_mesh_sys.contours.iter_mut().enumerate() {
        contour.id = index as u32;
        for point in &mut contour.points { // Use &mut to modify points in-place
            point.frame_index = index as u32; // Match the parent contour's new ID
        }
    }
    reference_mesh_sys.catheter.reverse();
    // After reversing the contours, update their IDs to match their new indices
    for (index, catheter) in reference_mesh_sys.catheter.iter_mut().enumerate() {
        catheter.id = index as u32;
        // Update frame_index for all points in this contour
        for point in &mut catheter.points { // Use &mut to modify points in-place
            point.frame_index = index as u32; // Match the parent contour's new ID
        }
    }
    
    let ((pt1, pt2), _) = reference_mesh_sys.contours[0].find_closest_opposite();
    
    let reference_point = if pt1.aortic {
        pt1.clone()
    } else {
        pt2.clone()
    };

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

        let mut mesh = rebuild_geometry(&mesh_path, &catheter_path);
        mesh.contours.reverse(); // reverse contours since for centerline alignment it is beneficial to have 0 for the ostium
        // Update contour IDs after reversal
        for (index, contour) in mesh.contours.iter_mut().enumerate() {
            contour.id = index as u32;
            for point in &mut contour.points { // Use &mut to modify points in-place
                point.frame_index = index as u32; // Match the parent contour's new ID
            }
        }
        mesh.catheter.reverse();
        // After reversing the contours, update their IDs to match their new indices
        for (index, catheter) in mesh.catheter.iter_mut().enumerate() {
            catheter.id = index as u32;
            // Update frame_index for all points in this contour
            for point in &mut catheter.points { // Use &mut to modify points in-place
                point.frame_index = index as u32; // Match the parent contour's new ID
            }
        }
        geometries.push(mesh)
    }
    geometries
}