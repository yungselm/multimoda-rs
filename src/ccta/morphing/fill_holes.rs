use nalgebra::Vector3;
use std::collections::{BTreeMap, HashSet};

/// Closes every open boundary loop with a fan of triangles around its centroid (one new vertex
/// per loop). A closed mesh comes back unchanged.
pub fn fill_holes_fan(
    vertices: &[Vector3<f64>],
    faces: &[[usize; 3]],
) -> (Vec<Vector3<f64>>, Vec<[usize; 3]>) {
    let mut vertices = vertices.to_vec();
    let mut faces = faces.to_vec();
    for ring in boundary_loops(&faces) {
        let points: Vec<Vector3<f64>> = ring.iter().map(|&i| vertices[i]).collect();
        let centroid = vertices.len();
        vertices.push(polyline_centroid(&points));
        faces.extend(fan_faces(&ring, centroid));
    }
    (vertices, faces)
}

/// Open boundary loops of a triangle mesh, each ordered so that [`fan_faces`] closes it with
/// the winding of the faces around it. Walks that do not come back to their start (non-manifold
/// boundaries) and loops shorter than 3 are skipped.
pub(in crate::ccta) fn boundary_loops(faces: &[[usize; 3]]) -> Vec<Vec<usize>> {
    let directed: HashSet<(usize, usize)> = faces
        .iter()
        .flat_map(|&[a, b, c]| [(a, b), (b, c), (c, a)])
        .collect();
    // A cap traverses each boundary edge a → b the other way round, b → a.
    let mut out: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for &(a, b) in &directed {
        if !directed.contains(&(b, a)) {
            out.entry(b).or_default().push(a);
        }
    }
    for targets in out.values_mut() {
        targets.sort_unstable();
    }

    let starts: Vec<usize> = out.keys().copied().collect();
    let mut loops = Vec::new();
    for start in starts {
        while let Some(first) = out.get_mut(&start).and_then(Vec::pop) {
            let mut ring = vec![start];
            let mut current = first;
            let closed = loop {
                if current == start {
                    break true;
                }
                ring.push(current);
                match out.get_mut(&current).and_then(Vec::pop) {
                    Some(next) => current = next,
                    None => break false,
                }
            };
            if closed && ring.len() >= 3 {
                loops.push(ring);
            }
        }
    }
    loops
}

/// Centroid of a closed polyline, each edge weighted by its length so unevenly spaced points do
/// not pull it to one side.
fn polyline_centroid(points: &[Vector3<f64>]) -> Vector3<f64> {
    let n = points.len();
    let mut sum = Vector3::zeros();
    let mut total = 0.0;
    for i in 0..n {
        let (a, b) = (points[i], points[(i + 1) % n]);
        let len = (b - a).norm();
        sum += (a + b) * (0.5 * len);
        total += len;
    }
    if total > 1e-12 {
        sum / total
    } else {
        points.iter().sum::<Vector3<f64>>() / n.max(1) as f64
    }
}

/// Triangles `[ring[i], ring[i + 1], centroid]` closing `ring` around vertex `centroid`.
pub fn fan_faces(ring: &[usize], centroid: usize) -> Vec<[usize; 3]> {
    let n = ring.len();
    (0..n)
        .map(|i| [ring[i], ring[(i + 1) % n], centroid])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Open box: the unit cube without its top (z = 1) face.
    fn open_box() -> (Vec<Vector3<f64>>, Vec<[usize; 3]>) {
        let vertices: Vec<Vector3<f64>> = (0..8)
            .map(|i| Vector3::new((i & 1) as f64, ((i >> 1) & 1) as f64, (i >> 2) as f64))
            .collect();
        // Outward-facing (counter-clockwise seen from outside).
        let faces = vec![
            [0, 2, 1],
            [1, 2, 3], // bottom
            [0, 1, 5],
            [0, 5, 4], // y = 0
            [2, 6, 7],
            [2, 7, 3], // y = 1
            [0, 4, 6],
            [0, 6, 2], // x = 0
            [1, 3, 7],
            [1, 7, 5], // x = 1
        ];
        (vertices, faces)
    }

    fn is_closed(faces: &[[usize; 3]]) -> bool {
        boundary_loops(faces).is_empty()
            && faces
                .iter()
                .flat_map(|&[a, b, c]| [(a, b), (b, c), (c, a)])
                .count()
                == faces
                    .iter()
                    .flat_map(|&[a, b, c]| [(a, b), (b, c), (c, a)])
                    .collect::<HashSet<_>>()
                    .len()
    }

    #[test]
    fn test_open_box_has_one_loop() {
        let (_, faces) = open_box();
        let loops = boundary_loops(&faces);
        assert_eq!(loops.len(), 1);
        let mut ring = loops[0].clone();
        ring.sort_unstable();
        assert_eq!(ring, vec![4, 5, 6, 7]);
    }

    #[test]
    fn test_fill_closes_box_with_consistent_winding() {
        let (vertices, faces) = open_box();
        let (v, f) = fill_holes_fan(&vertices, &faces);
        assert_eq!(v.len(), 9);
        assert!((v[8] - Vector3::new(0.5, 0.5, 1.0)).norm() < 1e-12);
        assert_eq!(f.len(), faces.len() + 4);
        // Every directed edge used once and its reverse present: closed and consistently wound.
        assert!(is_closed(&f));
        // The cap faces point up (outward).
        for face in &f[faces.len()..] {
            let n = (v[face[1]] - v[face[0]]).cross(&(v[face[2]] - v[face[0]]));
            assert!(n.z > 0.0);
        }
    }

    #[test]
    fn test_closed_mesh_is_unchanged() {
        let (vertices, mut faces) = open_box();
        faces.extend([[4, 5, 7], [4, 7, 6]]);
        let (v, f) = fill_holes_fan(&vertices, &faces);
        assert_eq!((v.len(), f.len()), (vertices.len(), faces.len()));
    }

    #[test]
    fn test_polyline_centroid_ignores_point_density() {
        // Square with many points on one side: the plain mean would shift towards it.
        let mut pts: Vec<Vector3<f64>> = (0..10)
            .map(|i| Vector3::new(i as f64 / 10.0, 0.0, 0.0))
            .collect();
        pts.extend([
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        ]);
        let c = polyline_centroid(&pts);
        assert!((c - Vector3::new(0.5, 0.5, 0.0)).norm() < 1e-12);
    }
}
