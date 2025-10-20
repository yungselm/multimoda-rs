use anyhow::{anyhow, Context, Result};
use std::fs::OpenOptions;
use std::path::Path;
use stl_io::IndexedMesh;

fn read_stl_ccta<P: AsRef<Path>>(path: P) -> Result<IndexedMesh> {
    let path_ref = path.as_ref();

    if !path_ref.exists() {
        return Err(anyhow!("File does not exist: {:?}", path_ref));
    }

    if !path_ref.is_file() {
        return Err(anyhow!("Path is not a file: {:?}", path_ref));
    }

    let mut file = OpenOptions::new()
        .read(true)
        .open(path_ref)
        .with_context(|| format!("Failed to open file: {:?}", path_ref.display()))?;

    let stl = stl_io::read_stl(&mut file)
        .with_context(|| format!("Failed to parse STL file: {:?}", path_ref.display()))?;
    
    // check if is valid mesh
    if stl.validate().is_err() {
        return Err(anyhow!("STL file is not a valid mesh: {:?}", path_ref.display()));
    }

    Ok(stl)
}

fn read_obj_ccta() -> () {
    todo!()
}

#[cfg(test)]
mod input_ccta_tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn write_temp_stl(contents: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time went backwards")
            .as_nanos();
        let filename = format!("test_stl_{}_{}.stl", std::process::id(), nanos);
        path.push(filename);
        fs::write(&path, contents).expect("failed to write temp stl");
        path
    }

    #[test]
    fn test_read_stl_success() {
        // a minimal ASCII STL content with a single triangle
    let stl_contents = r#"solid tetra
        facet normal 0 0 1
            outer loop
            vertex 0 0 0
            vertex 1 0 0
            vertex 0 1 0
            endloop
        endfacet
        facet normal 0 1 0
            outer loop
            vertex 0 0 0
            vertex 0 0 1
            vertex 1 0 0
            endloop
        endfacet
        facet normal 1 0 0
            outer loop
            vertex 0 0 0
            vertex 0 1 0
            vertex 0 0 1
            endloop
        endfacet
        facet normal -1 -1 -1
            outer loop
            vertex 1 0 0
            vertex 0 0 1
            vertex 0 1 0
            endloop
        endfacet
        endsolid tetra
        "#;

        let path = write_temp_stl(stl_contents);

        let result = read_stl_ccta(&path);
        let _ = fs::remove_file(&path);

        let stl = result.expect("expected to successfully read the STL");

        assert!(stl.faces.len() > 0);
        assert!(stl.vertices.len() > 0);
    }

    #[test]
    fn test_read_stl_missing_file() {
        let path = Path::new("this-file-does-not-exist.stl");
        let res = read_stl_ccta(path);
        assert!(res.is_err(), "expected error for missing file");
    }

    #[test]
    fn test_read_stl_invalid_file() {
        let path = write_temp_stl("this is not a valid stl file");
        let res = read_stl_ccta(&path);
        let _ = fs::remove_file(&path);
        assert!(res.is_err(), "expected parse error for invalid file contents");
    }
}