use crate::types::native::Record;
use pyo3::prelude::*;

/// Python representation of a per-frame measurement record.
///
/// Attributes
/// ----------
/// frame : int
///     Frame number within the acquisition sequence.
/// phase : str
///     Cardiac phase identifier: ``"D"`` for diastole or ``"S"`` for
///     systole.
/// measurement_1 : float or None
///     Primary measurement value.  In coronary artery anomalies this is
///     the wall thickness between the aorta and the coronary artery.
/// measurement_2 : float or None
///     Secondary measurement value.  In coronary artery anomalies this is
///     the wall thickness between the pulmonary artery and the coronary
///     artery.
///
/// Examples
/// --------
/// >>> record = PyRecord(
/// ...     frame=5,
/// ...     phase="D",
/// ...     measurement_1=1.4,
/// ...     measurement_2=2.1
/// ... )
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct PyRecord {
    #[pyo3(get, set)]
    pub frame: u32,
    #[pyo3(get, set)]
    pub phase: String,
    #[pyo3(get, set)]
    pub measurement_1: Option<f64>,
    #[pyo3(get, set)]
    pub measurement_2: Option<f64>,
}

#[pymethods]
impl PyRecord {
    #[new]
    fn new(
        frame: u32,
        phase: String,
        measurement_1: Option<f64>,
        measurement_2: Option<f64>,
    ) -> Self {
        PyRecord {
            frame,
            phase,
            measurement_1,
            measurement_2,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Record(frame={}, phase={}, m1={:?}, m2={:?})",
            self.frame, self.phase, self.measurement_1, self.measurement_2
        )
    }
}

impl PyRecord {
    pub fn to_rust_record(&self) -> Record {
        Record {
            frame: self.frame,
            phase: self.phase.clone(),
            measurement_1: self.measurement_1,
            measurement_2: self.measurement_2,
        }
    }
}

impl From<&Record> for PyRecord {
    fn from(r: &Record) -> Self {
        PyRecord {
            frame: r.frame,
            phase: r.phase.clone(),
            measurement_1: r.measurement_1,
            measurement_2: r.measurement_2,
        }
    }
}
