use std::collections::{HashMap, VecDeque};

/// Flips faces so that every pair of adjacent faces traverses their shared edge in opposite
/// directions, once per connected component, starting from its first face (a port of
/// `trimesh.repair.fix_winding`). Boundary and non-manifold edges do not connect faces.
pub fn fix_mesh_winding(faces: Vec<[usize; 3]>) -> Vec<[usize; 3]> {
    let n_faces = faces.len();
    if n_faces == 0 {
        return faces;
    }

    // (face_idx, edge_start, edge_end) - a face's own directed traversal of an edge.
    type FaceDirectedEdge = (usize, usize, usize);
    // (neighbor_face, this_face's_directed_edge, neighbor's_directed_edge)
    type FaceAdjacencyEntry = (usize, (usize, usize), (usize, usize));

    // For each undirected edge, collect the owning faces along with each
    // face's own directed (original-order) traversal of that edge.
    let mut edge_owners: HashMap<(usize, usize), Vec<FaceDirectedEdge>> = HashMap::new();
    for (fi, face) in faces.iter().enumerate() {
        let directed_edges = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])];
        for &(u, v) in &directed_edges {
            let key = if u < v { (u, v) } else { (v, u) };
            edge_owners.entry(key).or_default().push((fi, u, v));
        }
    }

    // Face adjacency: only edges owned by exactly 2 faces count (matches
    // trimesh's `face_adjacency`, which drops boundary/non-manifold edges).
    let mut adjacency: HashMap<usize, Vec<FaceAdjacencyEntry>> = HashMap::new();
    for owners in edge_owners.values() {
        if owners.len() != 2 {
            continue;
        }
        let (fa, ua, va) = owners[0];
        let (fb, ub, vb) = owners[1];
        adjacency
            .entry(fa)
            .or_default()
            .push((fb, (ua, va), (ub, vb)));
        adjacency
            .entry(fb)
            .or_default()
            .push((fa, (ub, vb), (ua, va)));
    }

    let mut flipped = vec![false; n_faces];
    let mut visited = vec![false; n_faces];

    for start in 0..n_faces {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let mut queue: VecDeque<usize> = VecDeque::new();
        queue.push_back(start);

        while let Some(current) = queue.pop_front() {
            let Some(neighbors) = adjacency.get(&current) else {
                continue;
            };
            for &(neighbor, current_edge, neighbor_edge) in neighbors {
                if visited[neighbor] {
                    continue;
                }
                // Current face's directed edge, accounting for its own flip state.
                let current_dir = if flipped[current] {
                    (current_edge.1, current_edge.0)
                } else {
                    current_edge
                };
                // Same starting vertex => both faces traverse the shared edge
                // in the same direction => inconsistent => flip the neighbor.
                if current_dir.0 == neighbor_edge.0 {
                    flipped[neighbor] = true;
                }
                visited[neighbor] = true;
                queue.push_back(neighbor);
            }
        }
    }

    faces
        .into_iter()
        .enumerate()
        .map(|(i, face)| {
            if flipped[i] {
                [face[2], face[1], face[0]]
            } else {
                face
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_already_consistent_quad_is_unchanged() {
        // Quad 0,1,2,3 split into two triangles with standard CCW winding:
        // shared edge {0,2} is (2,0) in face 0 and (0,2) in face 1 - opposite
        // directions, already consistent.
        let faces = vec![[0, 1, 2], [0, 2, 3]];
        let result = fix_mesh_winding(faces.clone());
        assert_eq!(result, faces);
    }

    #[test]
    fn test_inconsistent_pair_gets_second_face_flipped() {
        // Face 1 traverses the shared edge {0,2} as (2,0), same direction as
        // face 0's (2,0) -> inconsistent -> face 1 must be flipped.
        let faces = vec![[0, 1, 2], [2, 0, 3]];
        let result = fix_mesh_winding(faces);
        assert_eq!(result[0], [0, 1, 2]); // root face untouched
        assert_eq!(result[1], [3, 0, 2]); // reversed
    }

    #[test]
    fn test_isolated_faces_with_no_shared_edges_unchanged() {
        let faces = vec![[0, 1, 2], [5, 6, 7]];
        let result = fix_mesh_winding(faces.clone());
        assert_eq!(result, faces);
    }

    #[test]
    fn test_empty_input_returns_empty() {
        let result = fix_mesh_winding(vec![]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_bfs_propagates_across_a_triangle_fan() {
        // Fan of 3 triangles around a shared centre vertex 0, all originally
        // wound the same way as seen from +z (0,1,2), (0,2,3), (0,3,4) - i.e.
        // each shares an edge with the next in a CONSISTENT fan, so nothing
        // should need flipping; this checks BFS visits every face in one
        // component without spuriously flipping consistent faces.
        let faces = vec![[0, 1, 2], [0, 2, 3], [0, 3, 4]];
        let result = fix_mesh_winding(faces.clone());
        assert_eq!(result, faces);
    }
}
