use super::py_contour::PyContour;
use super::py_contour_point::PyContourPoint;
use super::record::PyRecord;
use crate::types::native::{ContourPoint, InputData, Record};
use anyhow::{anyhow, Result};
use pyo3::prelude::*;
use std::convert::TryFrom;

/// Python representation of the intravascular imaging input data for one cardiac phase.
///
/// Attributes
/// ----------
/// lumen : list of PyContour
///     Vessel lumen contours.
/// eem : list of PyContour or None
///     Vessel EEM (external elastic membrane) contours.
/// calcification : list of PyContour or None
///     Vessel calcification contours.
/// sidebranch : list of PyContour or None
///     Vessel sidebranch contours.
/// record : list of PyRecord or None
///     Metadata records about the input data.
/// ref_point : PyContourPoint
///     Reference point used for alignment.
/// diastole : bool
///     ``True`` when the data corresponds to the diastolic phase.
/// label : str
///     Human-readable label for this input dataset.
///
/// Examples
/// --------
/// >>> input_data = PyInputData(
/// ...     lumen=[lumen_contour1, lumen_contour2, ...],
/// ...     eem=[eem_contour1, eem_contour2, ...],
/// ...     calcification=[],
/// ...     sidebranch=[],
/// ...     record=record,
/// ...     diastole=True,
/// ...     lablel="Pat00_diastole_rest"
/// ... )
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct PyInputData {
    #[pyo3(get, set)]
    pub lumen: Vec<PyContour>,
    #[pyo3(get, set)]
    pub eem: Option<Vec<PyContour>>,
    #[pyo3(get, set)]
    pub calcification: Option<Vec<PyContour>>,
    #[pyo3(get, set)]
    pub sidebranch: Option<Vec<PyContour>>,
    #[pyo3(get, set)]
    pub record: Option<Vec<PyRecord>>,
    #[pyo3(get, set)]
    pub ref_point: PyContourPoint,
    #[pyo3(get, set)]
    pub diastole: bool,
    #[pyo3(get, set)]
    pub label: String,
}

#[pymethods]
impl PyInputData {
    #[new]
    fn new(
        lumen: Vec<PyContour>,
        eem: Option<Vec<PyContour>>,
        calcification: Option<Vec<PyContour>>,
        sidebranch: Option<Vec<PyContour>>,
        record: Option<Vec<PyRecord>>,
        ref_point: PyContourPoint,
        diastole: bool,
        label: String,
    ) -> Self {
        Self {
            lumen,
            eem,
            calcification,
            sidebranch,
            record,
            ref_point,
            diastole,
            label,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "InputData(lumen={}, eem={}, calcification={}, sidebranch={}, record={}, ref_point={}, diastole={}, label='{}')",
            self.lumen.len(),
            self.eem.as_ref().map_or(0, |v| v.len()),
            self.calcification.as_ref().map_or(0, |v| v.len()),
            self.sidebranch.as_ref().map_or(0, |v| v.len()),
            self.record.as_ref().map_or(0, |v| v.len()),
            self.ref_point.__repr__(),
            self.diastole,
            self.label,
        )
    }
}

impl TryFrom<&PyInputData> for InputData {
    type Error = anyhow::Error;

    fn try_from(py_in: &PyInputData) -> Result<Self> {
        // Flatten Vec<PyContour> -> Vec<ContourPoint>
        let flatten = |contours_opt: &Option<Vec<PyContour>>| -> Result<Option<Vec<ContourPoint>>> {
            if let Some(contours) = contours_opt {
                let mut acc: Vec<ContourPoint> = Vec::new();
                for c in contours {
                    // convert PyContour -> Contour (may return PyErr), use to_rust_contour()
                    let rust_contour = c.to_rust_contour().map_err(|e| {
                        anyhow!(
                            "failed to convert PyContour(id={}) to Contour: {:?}",
                            c.id,
                            e
                        )
                    })?;
                    acc.extend(rust_contour.points);
                }
                Ok(Some(acc))
            } else {
                Ok(None)
            }
        };

        // Lumen (required)
        let mut lumen_points: Vec<ContourPoint> = Vec::new();
        for c in &py_in.lumen {
            let rust_contour = c.to_rust_contour().map_err(|e| {
                anyhow!(
                    "failed to convert lumen PyContour(id={}) to Contour: {:?}",
                    c.id,
                    e
                )
            })?;
            lumen_points.extend(rust_contour.points);
        }

        let eem_points = flatten(&py_in.eem)?;
        let calc_points = flatten(&py_in.calcification)?;
        let sidebranch_points = flatten(&py_in.sidebranch)?;

        // Records: Option<Vec<PyRecord>> -> Option<Vec<Record>>
        let records_rust: Option<Vec<Record>> = match &py_in.record {
            Some(py_records) => {
                let mut out = Vec::with_capacity(py_records.len());
                for r in py_records {
                    out.push(r.to_rust_record());
                }
                Some(out)
            }
            None => None,
        };

        // ref_point: PyContourPoint -> ContourPoint
        let ref_point_rust: ContourPoint = ContourPoint::from(&py_in.ref_point);

        // Build InputData via its constructor to preserve validations
        InputData::new(
            lumen_points,
            eem_points,
            calc_points,
            sidebranch_points,
            records_rust,
            ref_point_rust,
            py_in.diastole,
            py_in.label.clone(),
        )
        .map_err(|e| anyhow!("InputData::new failed: {e:?}"))
    }
}

impl TryFrom<PyInputData> for InputData {
    type Error = anyhow::Error;

    fn try_from(py_in: PyInputData) -> Result<Self> {
        InputData::try_from(&py_in)
    }
}

impl From<&InputData> for PyInputData {
    fn from(input: &InputData) -> Self {
        // helper: build a single PyContour from a flattened Vec<ContourPoint>
        fn make_pycontour_from_points(
            points: &[ContourPoint],
            id: u32,
            original_frame: u32,
        ) -> PyContour {
            // compute centroid (average) if points non-empty
            let centroid = if points.is_empty() {
                (0.0, 0.0, 0.0)
            } else {
                let (sx, sy, sz) = points
                    .iter()
                    .fold((0.0f64, 0.0f64, 0.0f64), |(sx, sy, sz), p| {
                        (sx + p.x, sy + p.y, sz + p.z)
                    });
                let n = points.len() as f64;
                (sx / n, sy / n, sz / n)
            };

            let py_points: Vec<PyContourPoint> = points.iter().map(PyContourPoint::from).collect();

            PyContour {
                id,
                original_frame,
                points: py_points,
                centroid,
                aortic_thickness: None,
                pulmonary_thickness: None,
                kind: "Lumen".to_string(),
            }
        }

        // Build lumen as a single PyContour (id = 0)
        let original_frame = input.ref_point.frame_index;
        let lumen_py = make_pycontour_from_points(&input.lumen, 0, original_frame);
        let lumen_vec = vec![lumen_py];

        // Optional groups: wrap each existing flattened vec into a single PyContour (if present)
        let wrap_opt =
            |opt_pts: &Option<Vec<ContourPoint>>, id_start: u32| -> Option<Vec<PyContour>> {
                opt_pts
                    .as_ref()
                    .map(|pts| vec![make_pycontour_from_points(pts, id_start, original_frame)])
            };

        let eem_py = wrap_opt(&input.eem, 0);
        let calc_py = wrap_opt(&input.calcification, 0);
        let sb_py = wrap_opt(&input.sidebranch, 0);

        // Records
        let record_py: Option<Vec<PyRecord>> = input
            .record
            .as_ref()
            .map(|records| records.iter().map(PyRecord::from).collect());

        let ref_point_py = PyContourPoint::from(&input.ref_point);

        PyInputData {
            lumen: lumen_vec,
            eem: eem_py,
            calcification: calc_py,
            sidebranch: sb_py,
            record: record_py,
            ref_point: ref_point_py,
            diastole: input.diastole,
            label: input.label.clone(),
        }
    }
}

impl From<InputData> for PyInputData {
    fn from(input: InputData) -> Self {
        PyInputData::from(&input)
    }
}
