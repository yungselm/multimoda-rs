use super::py_geometry::PyGeometry;
use crate::intravascular::binding;
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
/// Notes
/// -----
/// The pair keeps every frame of both geometries. Frames are matched by their
/// offset from each geometry's master reference, see :meth:`overlap`. Analyses
/// (:meth:`get_summary`, OBJ export) only use the overlapping frames.
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
        let facing = self.overlapping()?;
        let dia = facing.geom_a.get_summary()?;
        let sys = facing.geom_b.get_summary()?;
        let map = facing.create_deformation_table();
        Ok(((dia, sys), map))
    }

    /// Frames of ``geom_a`` and ``geom_b`` that face each other.
    ///
    /// Frame ``i`` of ``geom_a`` faces frame ``i - master_a + master_b`` of
    /// ``geom_b``, where ``master_a`` and ``master_b`` are the positions of the
    /// master reference frames.
    ///
    /// Returns
    /// -------
    /// tuple of tuple of int
    ///     ``((start_a, stop_a), (start_b, stop_b))``, half-open index ranges of
    ///     equal length.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a geometry has no master reference.
    pub fn overlap(&self) -> PyResult<((usize, usize), (usize, usize))> {
        let (range_a, range_b) = self.try_to_rust()?.overlap().map_err(value_error)?;
        Ok(((range_a.start, range_a.end), (range_b.start, range_b.end)))
    }

    /// A copy holding only the overlapping frames, renumbered from 0.
    ///
    /// Returns
    /// -------
    /// PyGeometryPair
    ///     Frame ``i`` of ``geom_a`` faces frame ``i`` of ``geom_b``.
    pub fn overlapping(&self) -> PyResult<PyGeometryPair> {
        let facing = self.try_to_rust()?.overlapping().map_err(value_error)?;
        Ok(facing.into())
    }

    /// Sets new master references and re-runs the alignment between the geometries.
    ///
    /// ``geom_b`` is translated and rotated so its master frame sits on the master
    /// frame of ``geom_a``. With ``postprocessing`` both are resampled again as in
    /// the processing pipelines. No frame is removed. Call it before aligning the
    /// pair to a centerline.
    ///
    /// Parameters
    /// ----------
    /// master_a : int or None, optional
    ///     Original frame of the new master reference of ``geom_a``. ``None``
    ///     keeps the current one. Default is ``None``.
    /// master_b : int or None, optional
    ///     Same for ``geom_b``. Default is ``None``.
    /// range_rotation_deg : float, optional
    ///     Rotation search range (±) in degrees. Default is ``90.0``.
    /// step_rotation_deg : float, optional
    ///     Rotation step in degrees. Default is ``0.5``.
    /// sample_size : int, optional
    ///     Number of points used for the rotation search. Default is ``500``.
    /// postprocessing : bool, optional
    ///     Whether to resample the pair afterwards. Default is ``True``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a given frame carries no reference point.
    #[pyo3(signature = (
        master_a=None,
        master_b=None,
        range_rotation_deg=90.0,
        step_rotation_deg=0.5,
        sample_size=500,
        postprocessing=true,
    ))]
    pub fn set_master_references(
        &mut self,
        master_a: Option<u32>,
        master_b: Option<u32>,
        range_rotation_deg: f64,
        step_rotation_deg: f64,
        sample_size: usize,
        postprocessing: bool,
    ) -> PyResult<()> {
        let pair = self.try_to_rust()?;
        let realigned = binding::entry::realign_pair_rs(
            &pair,
            master_a,
            master_b,
            range_rotation_deg,
            step_rotation_deg,
            sample_size,
            postprocessing,
        )
        .map_err(value_error)?;
        *self = realigned.into();
        Ok(())
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

fn value_error(e: anyhow::Error) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(format!("{e:#}"))
}

impl PyGeometryPair {
    fn try_to_rust(&self) -> PyResult<GeometryPair> {
        Ok(GeometryPair {
            geom_a: self.geom_a.to_rust_geometry()?,
            geom_b: self.geom_b.to_rust_geometry()?,
            label: self.label.clone(),
        })
    }

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
