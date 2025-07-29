use super::geometries::GeometryPair;
use crate::io::input::{Contour, ContourPoint};
use crate::io::Geometry;
use std::error::Error;

pub fn prepare_geometries_comparison(
    geometries_rest: GeometryPair,
    geometries_stress: GeometryPair,
) -> (GeometryPair, GeometryPair) {
    let mut dia_rest = geometries_rest.dia_geom;
    let mut sys_rest = geometries_rest.sys_geom;
    let dia_stress = geometries_stress.dia_geom;
    let sys_stress = geometries_stress.sys_geom;

    translate_z_to_match(&mut dia_rest, &dia_stress);
    translate_z_to_match(&mut sys_rest, &sys_stress);

    let dia_rest = resample_to_reference_z(&dia_rest, &dia_stress).unwrap();
    let sys_rest = resample_to_reference_z(&sys_rest, &sys_stress).unwrap();

    // Translate in xy-plane, z unchanged
    let dia_rest = align_geometries(&dia_stress, dia_rest);
    let sys_rest = align_geometries(&sys_stress, sys_rest);

    let dia_pair = GeometryPair {
        dia_geom: dia_rest,
        sys_geom: dia_stress,
    };
    let sys_pair = GeometryPair {
        dia_geom: sys_rest,
        sys_geom: sys_stress,
    };

    let dia_pair = dia_pair
        .trim_geometries_same_length()
        .adjust_z_coordinates();
    let sys_pair = sys_pair
        .trim_geometries_same_length()
        .adjust_z_coordinates();

    (dia_pair, sys_pair)
}

/// Resamples an original Geometry so that it has contours at exactly the same Z positions
/// as the reference Geometry, by linearly interpolating between the two nearest frames.
fn resample_to_reference_z(
    original: &Geometry,
    reference: &Geometry,
) -> Result<Geometry, Box<dyn Error>> {
    let orig_contours = &original.contours;
    let m = orig_contours.len();
    if m < 2 {
        return Err("Original geometry must have at least two contours for interpolation".into());
    }

    let orig_z: Vec<f64> = orig_contours.iter().map(|c| c.centroid.2).collect();

    let mut new_contours = Vec::with_capacity(reference.contours.len());
    for (idx, ref_c) in reference.contours.iter().enumerate() {
        let z_ref = ref_c.centroid.2;
        // Find interval [i, i+1] in orig_z containing z_ref
        let i = match orig_z.iter().position(|&z| z >= z_ref) {
            Some(0) => 0,
            Some(pos) => pos - 1,
            None => m - 2,
        };
        let z0 = orig_z[i];
        let z1 = orig_z[i + 1];
        let t = if (z1 - z0).abs() < std::f64::EPSILON {
            0.0
        } else {
            (z_ref - z0) / (z1 - z0)
        };
        // Interpolate contour points
        let pts0 = &orig_contours[i].points;
        let pts1 = &orig_contours[i + 1].points;
        let mut interp_points: Vec<ContourPoint> = Vec::with_capacity(pts0.len());
        for (p0, p1) in pts0.iter().zip(pts1.iter()) {
            interp_points.push(ContourPoint {
                frame_index: idx as u32,
                point_index: p0.point_index,
                x: p0.x * (1.0 - t) + p1.x * t,
                y: p0.y * (1.0 - t) + p1.y * t,
                z: z_ref,
                aortic: p0.aortic,
            });
        }

        let centroid = Contour::compute_centroid(&interp_points);
        let new_cont = Contour {
            id: idx as u32,
            points: interp_points,
            centroid,
            aortic_thickness: None,
            pulmonary_thickness: None,
        };
        new_contours.push(new_cont);
    }

    // Similarly interpolate catheter if present
    let new_catheter = if !original.catheter.is_empty() {
        let cat_curve = &original.catheter;
        let orig_cat_z: Vec<f64> = cat_curve.iter().map(|c| c.centroid.2).collect();
        let mut new_cat = Vec::with_capacity(reference.catheter.len());
        for (idx, ref_c) in reference.catheter.iter().enumerate() {
            let z_ref = ref_c.centroid.2;
            let i = match orig_cat_z.iter().position(|&z| z >= z_ref) {
                Some(0) => 0,
                Some(pos) => pos - 1,
                None => orig_cat_z.len() - 2,
            };
            let z0 = orig_cat_z[i];
            let z1 = orig_cat_z[i + 1];
            let t = if (z1 - z0).abs() < std::f64::EPSILON {
                0.0
            } else {
                (z_ref - z0) / (z1 - z0)
            };
            let pts0 = &cat_curve[i].points;
            let pts1 = &cat_curve[i + 1].points;
            let mut interp_pts = Vec::with_capacity(pts0.len());
            for (p0, p1) in pts0.iter().zip(pts1.iter()) {
                interp_pts.push(ContourPoint {
                    frame_index: idx as u32,
                    point_index: p0.point_index,
                    x: p0.x * (1.0 - t) + p1.x * t,
                    y: p0.y * (1.0 - t) + p1.y * t,
                    z: z_ref,
                    aortic: p0.aortic,
                });
            }
            let centroid = Contour::compute_centroid(&interp_pts);
            new_cat.push(Contour {
                id: idx as u32,
                points: interp_pts,
                centroid,
                aortic_thickness: None,
                pulmonary_thickness: None,
            });
        }
        new_cat
    } else {
        Vec::new()
    };

    Ok(Geometry {
        contours: new_contours,
        catheter: new_catheter,
        walls: vec![],
        reference_point: original.reference_point.clone(),
        label: original.label.clone(),
    })
}

/// Translate an original geometry in Z so its last contour matches reference last contour
fn translate_z_to_match(orig: &mut Geometry, ref_geom: &Geometry) {
    let last_orig = orig.contours.last().unwrap().centroid.2;
    let last_ref = ref_geom.contours.last().unwrap().centroid.2;
    let dz = last_ref - last_orig;
    for contour in orig.contours.iter_mut() {
        contour.translate_contour((0.0, 0.0, dz));
    }
    for cath in orig.catheter.iter_mut() {
        cath.translate_contour((0.0, 0.0, dz));
    }
}

/// Align XY centroids (keeps Z unchanged)
fn align_geometries(ref_geom: &Geometry, mut geom: Geometry) -> Geometry {
    let ref_centroid = ref_geom.contours.last().unwrap().centroid;
    for contour in geom.contours.iter_mut().chain(geom.catheter.iter_mut()) {
        let c = contour.centroid;
        let tx = ref_centroid.0 - c.0;
        let ty = ref_centroid.1 - c.1;
        contour.translate_contour((tx, ty, 0.0));
    }
    geom
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use approx::assert_relative_eq;
//     use std::f64::consts::PI;
//     use crate::utils::test_utils::{generate_ellipse_points, new_dummy_contour}

//     #[test]
//     fn test_translate_z_to_match() {

//     }

// }
