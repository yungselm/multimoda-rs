use crate::intravascular::io::input::ContourPoint;

pub fn downsample_contour_points(points: &[ContourPoint], n: usize) -> Vec<ContourPoint> {
    if points.len() <= n {
        return points.to_vec();
    }
    let step = points.len() as f64 / n as f64;
    (0..n)
        .map(|i| {
            let index = (i as f64 * step) as usize;
            points[index].clone()
        })
        .collect()
}