use anyhow::{anyhow, Context, Result};
use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions;
use std::path::Path;
use stl_io::IndexedMesh;

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Label {
    LCA,
    RCA,
    Intravascular,
    Aorta,
    Unknown,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct LabeledMesh {
    pub mesh: IndexedMesh,
    pub labels: Vec<Option<Label>>,
    pub label_index: HashMap<Label, HashSet<usize>>,
}

#[allow(dead_code)]
impl LabeledMesh {
    /// Create a new LabeledMesh with all vertices unlabeled
    pub fn from_mesh(mesh: IndexedMesh) -> Self {
        let labels = vec![None; mesh.vertices.len()];
        Self {
            mesh,
            labels,
            label_index: HashMap::new(),
        }
    }

    /// Assign a label to a vertex
    pub fn set_label(&mut self, vertex_idx: usize, label: Label) {
        if vertex_idx < self.labels.len() {
            self.labels[vertex_idx] = Some(label);
        } else {
            eprintln!("Warning: vertex index {} out of bounds", vertex_idx);
        }
    }

    /// Retrieve the label (if any) for a vertex
    pub fn get_label(&self, vertex_idx: usize) -> Option<&Label> {
        self.labels.get(vertex_idx).and_then(|o| o.as_ref())
    }

    /// Count how many vertices have a given label
    pub fn count_label(&self, label: &Label) -> usize {
        self.labels
            .iter()
            .filter(|l| l.as_ref() == Some(label))
            .count()
    }

    /// Retrieve all vertices for a given label
    pub fn vertices_with_label(&self, label: &Label) -> Vec<usize> {
        self.labels
            .iter()
            .enumerate()
            .filter_map(|(i, l)| {
                if l.as_ref() == Some(label) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }
}

#[allow(dead_code)]
pub fn read_stl_ccta<P: AsRef<Path>>(path: P) -> Result<LabeledMesh> {
    let path_ref = path.as_ref();

    if !path_ref.exists() {
        return Err(anyhow!("File does not exist: {:?}", path_ref.display()));
    }

    if !path_ref.is_file() {
        return Err(anyhow!("Path is not a file: {:?}", path_ref.display()));
    }

    let mut file = OpenOptions::new()
        .read(true)
        .open(path_ref)
        .with_context(|| format!("Failed to open file: {:?}", path_ref.display()))?;

    let stl = stl_io::read_stl(&mut file)
        .with_context(|| format!("Failed to parse STL file: {:?}", path_ref.display()))?;

    if stl.validate().is_err() {
        return Err(anyhow!(
            "STL file is not a valid mesh: {:?}",
            path_ref.display()
        ));
    }

    Ok(LabeledMesh::from_mesh(stl))
}

#[allow(dead_code)]
fn read_obj_ccta() -> () {
    todo!()
}

// #[cfg(test)]
// mod input_ccta_tests {
//     use super::*;
//     use std::fs;
//     use std::path::PathBuf;
//     use std::time::{SystemTime, UNIX_EPOCH};

//     const STL_TEST_CONTENTS: &str = r#"solid tetra
//         facet normal 0 0 1
//             outer loop
//             vertex 0 0 0
//             vertex 1 0 0
//             vertex 0 1 0
//             endloop
//         endfacet
//         facet normal 0 1 0
//             outer loop
//             vertex 0 0 0
//             vertex 0 0 1
//             vertex 1 0 0
//             endloop
//         endfacet
//         facet normal 1 0 0
//             outer loop
//             vertex 0 0 0
//             vertex 0 1 0
//             vertex 0 0 1
//             endloop
//         endfacet
//         facet normal -1 -1 -1
//             outer loop
//             vertex 1 0 0
//             vertex 0 0 1
//             vertex 0 1 0
//             endloop
//         endfacet
//         endsolid tetra
//         "#;

//     fn write_temp_stl(contents: &str) -> PathBuf {
//         let mut path = std::env::temp_dir();
//         let nanos = SystemTime::now()
//             .duration_since(UNIX_EPOCH)
//             .expect("time went backwards")
//             .as_nanos();
//         let filename = format!("test_stl_{}_{}.stl", std::process::id(), nanos);
//         path.push(filename);
//         fs::write(&path, contents).expect("failed to write temp stl");
//         path
//     }

//     #[test]
//     fn test_read_stl_success() {
//         // a minimal ASCII STL content with a single triangle
//     let stl_contents = STL_TEST_CONTENTS;

//         let path = write_temp_stl(stl_contents);

//         let result = read_stl_ccta(&path);
//         let _ = fs::remove_file(&path);

//         let stl = result.expect("expected to successfully read the STL");

//         assert!(stl.mesh.faces.len() > 0);
//         assert!(stl.mesh.vertices.len() > 0);
//     }

//     #[test]
//     fn test_read_stl_missing_file() {
//         let path = Path::new("this-file-does-not-exist.stl");
//         let res = read_stl_ccta(path);
//         assert!(res.is_err(), "expected error for missing file");
//     }

//     #[test]
//     fn test_read_stl_invalid_file() {
//         let path = write_temp_stl("this is not a valid stl file");
//         let res = read_stl_ccta(&path);
//         let _ = fs::remove_file(&path);
//         assert!(res.is_err(), "expected parse error for invalid file contents");
//     }

//     #[test]
//     fn test_labeling_vertices() {
//         let stl_contents = STL_TEST_CONTENTS;
//         let path = write_temp_stl(stl_contents);
//         let mut labeled_mesh = read_stl_ccta(&path).expect("failed to read stl");
//         let _ = fs::remove_file(&path);
//         labeled_mesh.set_label(0, Label::LCA);
//         labeled_mesh.set_label(1, Label::RCA);

//         assert_eq!(labeled_mesh.get_label(0), Some(&Label::LCA));
//         assert_eq!(labeled_mesh.get_label(1), Some(&Label::RCA));
//         assert_eq!(labeled_mesh.count_label(&Label::LCA), 1);
//         assert_eq!(labeled_mesh.count_label(&Label::RCA), 1);
//     }

//     #[test]
//     fn test_vertices_with_label() {
//         let stl_contents = STL_TEST_CONTENTS;
//         let path = write_temp_stl(stl_contents);
//         let mut labeled_mesh = read_stl_ccta(&path).expect("failed to read stl");
//         let _ = fs::remove_file(&path);
//         labeled_mesh.set_label(0, Label::LCA);
//         labeled_mesh.set_label(1, Label::RCA);
//         labeled_mesh.set_label(2, Label::RCA);
//         let rca_vertices = labeled_mesh.vertices_with_label(&Label::RCA);

//         assert_eq!(rca_vertices, vec![1, 2]);
//     }
// }
