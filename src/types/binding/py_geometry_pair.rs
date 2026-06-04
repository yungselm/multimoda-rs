use super::py_geometry::PyGeometry;
use crate::types::native::GeometryPair;
use pyo3::prelude::*;

type GeomSummary = (f64, f64, f64);
type PairSummary = ((GeomSummary, GeomSummary), Vec<[f64; 6]>);

/// Python representation of a diastolic/systolic geometry pair.
///
/// Attributes
/// ----------
/// geom_a : PyGeometry
///     First geometry (typically diastolic).
/// geom_b : PyGeometry
///     Second geometry (typically systolic).
/// label : str
///     Human-readable label for this geometry pair.
///
/// Examples
/// --------
/// >>> pair = PyGeometryPair(
/// ...     geom_a=diastole,
/// ...     geom_b=systole,
/// ...     label="Pat00_rest"
/// ... )
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct PyGeometryPair {
    #[pyo3(get, set)]
    pub geom_a: PyGeometry,
    #[pyo3(get, set)]
    pub geom_b: PyGeometry,
    #[pyo3(get, set)]
    pub label: String,
}

#[pymethods]
impl PyGeometryPair {
    #[new]
    fn new(geom_a: PyGeometry, geom_b: PyGeometry, label: String) -> Self {
        Self {
            geom_a,
            geom_b,
            label,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "GeometryPair {} (diastolic: {} frames, systolic: {} frames)",
            self.label,
            self.geom_a.frames.len(),
            self.geom_b.frames.len()
        )
    }

    /// Get summaries for both geometries and a per-frame deformation table.
    ///
    /// Calls :meth:`PyGeometry.get_summary` on each contained geometry and
    /// additionally computes per-frame area and elliptic ratio for both
    /// phases.
    ///
    /// Returns
    /// -------
    /// summaries : tuple
    ///     ``((dia_mla, dia_max_stenosis, dia_len_mm), (sys_mla, sys_max_stenosis, sys_len_mm))``.
    /// table : list of list of float
    ///     Matrix of shape ``(N, 6)`` with columns
    ///     ``[id, area_dia, ellip_dia, area_sys, ellip_sys, z]``.
    pub fn get_summary(&self) -> PyResult<PairSummary> {
        let dia = self.geom_a.get_summary()?;
        let sys = self.geom_b.get_summary()?;
        let map = self.create_deformation_table();
        Ok(((dia, sys), map))
    }

    fn create_deformation_table(&self) -> Vec<[f64; 6]> {
        let dia_lumen = self.geom_a.get_lumen_contours();
        let sys_lumen = self.geom_b.get_lumen_contours();

        let areas_dia: Vec<f64> = dia_lumen.iter().map(|c| c.get_area().unwrap()).collect();
        let areas_sys: Vec<f64> = sys_lumen.iter().map(|c| c.get_area().unwrap()).collect();

        let ellip_dia: Vec<f64> = dia_lumen
            .iter()
            .map(|c| c.get_elliptic_ratio().unwrap())
            .collect();
        let ellip_sys: Vec<f64> = sys_lumen
            .iter()
            .map(|c| c.get_elliptic_ratio().unwrap())
            .collect();

        let ids: Vec<u32> = dia_lumen.iter().map(|c| c.id).collect();
        let z_coords: Vec<f64> = dia_lumen.iter().map(|c| c.centroid.2).collect();

        // Ensure all vectors have same length
        let n = ids.len();
        if areas_dia.len() != n
            || ellip_dia.len() != n
            || areas_sys.len() != n
            || ellip_sys.len() != n
            || z_coords.len() != n
        {
            eprintln!("ERROR: mismatched lengths between contour vectors");
        }

        // Build numeric matrix: each row is [id, area_dia, ellip_dia, area_sys, ellip_sys, z]
        let mut mat: Vec<[f64; 6]> = Vec::with_capacity(n);
        for i in 0..n {
            mat.push([
                ids[i] as f64,
                areas_dia[i],
                ellip_dia[i],
                areas_sys[i],
                ellip_sys[i],
                z_coords[i],
            ]);
        }

        // Prepare printable rows (format floats to 6 decimal places)
        let headers = ["id", "area_dia", "ellip_dia", "area_sys", "ellip_sys", "z"];
        let rows: Vec<[String; 6]> = (0..n)
            .map(|i| {
                [
                    ids[i].to_string(),          // id as integer
                    format!("{:.2}", mat[i][1]), // area_dia
                    format!("{:.2}", mat[i][2]), // ellip_dia
                    format!("{:.2}", mat[i][3]), // area_sys
                    format!("{:.2}", mat[i][4]), // ellip_sys
                    format!("{:.2}", mat[i][5]), // z
                ]
            })
            .collect();

        // Compute max width for each of the 6 columns
        let mut widths = [0usize; 6];
        for (i, &h) in headers.iter().enumerate() {
            widths[i] = h.len();
        }
        for row in &rows {
            for (i, cell) in row.iter().enumerate() {
                widths[i] = widths[i].max(cell.len());
            }
        }

        // Print a left-aligned data row (same style as your dump_table)
        fn print_row(cells: &[String], widths: &[usize]) {
            print!("|");
            for (i, cell) in cells.iter().enumerate() {
                let pad = widths[i] - cell.len();
                print!(" {}{} |", cell, " ".repeat(pad));
            }
            println!();
        }

        // Print a centered header row
        fn print_header(cells: &[String], widths: &[usize]) {
            print!("|");
            for (i, cell) in cells.iter().enumerate() {
                let total_pad = widths[i] - cell.len();
                let left = total_pad / 2;
                let right = total_pad - left;
                print!(" {}{}{} |", " ".repeat(left), cell, " ".repeat(right));
            }
            println!();
        }

        // Top border
        print!("+");
        for w in &widths {
            print!("{}+", "-".repeat(w + 2));
        }
        println!();

        // Header row
        let header_cells: Vec<String> = headers.iter().map(|&s| s.to_string()).collect();
        print_header(&header_cells, &widths);

        // Separator
        print!("+");
        for w in &widths {
            print!("{}+", "-".repeat(w + 2));
        }
        println!();

        // Data rows
        for row in &rows {
            print_row(row, &widths);
        }

        // Bottom border
        print!("+");
        for w in &widths {
            print!("{}+", "-".repeat(w + 2));
        }
        println!();

        mat
    }
}

impl PyGeometryPair {
    pub fn to_rust_geometry_pair(&self) -> GeometryPair {
        GeometryPair {
            geom_a: self
                .geom_a
                .to_rust_geometry()
                .expect("could not convert geom_a"),
            geom_b: self
                .geom_b
                .to_rust_geometry()
                .expect("could not convert geom_b"),
            label: self.label.clone(),
        }
    }
}

impl From<GeometryPair> for PyGeometryPair {
    fn from(pair: GeometryPair) -> Self {
        PyGeometryPair {
            geom_a: pair.geom_a.into(),
            geom_b: pair.geom_b.into(),
            label: pair.label.clone(),
        }
    }
}
