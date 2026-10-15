use std::collections::{HashMap, HashSet};

type Coords3 = (f64, f64, f64);

/// Exact-match key for a mesh-vertex coordinate. Vertex-derived point lists passed
/// around this pipeline (e.g. `rca_points`, `lca_points`) are always bit-identical
/// copies of the originating `vertices` entries (no arithmetic in between), so an
/// exact bit-pattern key is the correct tool here — unlike a radius/nearest-neighbor
/// query, this is a plain exact-membership test, so no spatial index is needed.
pub(crate) fn bits_key(p: &(f64, f64, f64)) -> (u64, u64, u64) {
    (p.0.to_bits(), p.1.to_bits(), p.2.to_bits())
}

/// Every vertex mapped to the vertices it shares a face edge with.
pub fn build_adjacency_map(faces: &[[usize; 3]]) -> HashMap<usize, HashSet<usize>> {
    let mut adjacency: HashMap<usize, HashSet<usize>> = HashMap::new();
    for &[v0, v1, v2] in faces {
        for (a, b) in [(v0, v1), (v1, v2), (v2, v0)] {
            adjacency.entry(a).or_default().insert(b);
            adjacency.entry(b).or_default().insert(a);
        }
    }
    adjacency
}

/// Connected components of `subset`, restricted to mesh adjacency: neighbour
/// traversal only follows vertices that are themselves in `subset` (islands
/// within the subset stay separate components).
pub(crate) fn connected_components(
    adjacency: &HashMap<usize, HashSet<usize>>,
    subset: &HashSet<usize>,
) -> Vec<HashSet<usize>> {
    let mut remaining: HashSet<usize> = subset.clone();
    let mut components = Vec::new();

    while let Some(&start) = remaining.iter().next() {
        let mut stack = vec![start];
        let mut component = HashSet::new();
        while let Some(i) = stack.pop() {
            if !component.insert(i) {
                continue;
            }
            if let Some(neighbors) = adjacency.get(&i) {
                for &n in neighbors {
                    if remaining.contains(&n) && !component.contains(&n) {
                        stack.push(n);
                    }
                }
            }
        }
        remaining.retain(|i| !component.contains(i));
        components.push(component);
    }

    components
}

/// The largest mesh-connected component of `points` (vertex coordinates, matched exactly).
/// `points` with fewer than 2 entries, or matching no vertex, come back unchanged.
pub fn keep_largest_connected_component(
    vertices: &[Coords3],
    faces: &[[usize; 3]],
    points: Vec<Coords3>,
) -> Vec<Coords3> {
    if points.len() < 2 {
        return points;
    }
    let coord_to_idx: HashMap<(u64, u64, u64), usize> = vertices
        .iter()
        .enumerate()
        .map(|(i, v)| (bits_key(v), i))
        .collect();
    let point_indices: HashSet<usize> = points
        .iter()
        .filter_map(|p| coord_to_idx.get(&bits_key(p)).copied())
        .collect();
    if point_indices.is_empty() {
        return points;
    }
    connected_components(&build_adjacency_map(faces), &point_indices)
        .into_iter()
        .max_by_key(|c| c.len())
        .map(|largest| largest.into_iter().map(|i| vertices[i]).collect())
        .unwrap_or(points)
}
