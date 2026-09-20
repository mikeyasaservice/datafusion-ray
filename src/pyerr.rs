// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! The small slice of `datafusion-python` this crate used to depend on.
//!
//! Depending on the crate meant inheriting `pyo3/extension-module`
//! unconditionally, which keeps the test binary from linking libpython and so
//! puts every `#[pyclass]`/`#[pymethods]` body out of reach of `cargo test`.
//! It also pinned us to an unreleased git revision. The surface was five items,
//! reproduced here.

use std::future::Future;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use arrow::error::ArrowError;
use datafusion::error::DataFusionError;
use datafusion::logical_expr::LogicalPlan;
use datafusion::physical_plan::{ExecutionPlan, displayable};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use tokio::runtime::Runtime;
use tokio::time::sleep;

pub type PyDataFusionResult<T> = std::result::Result<T, PyDataFusionError>;

/// Errors that cross the Python boundary.
#[derive(Debug)]
pub enum PyDataFusionError {
    ExecutionError(Box<DataFusionError>),
    ArrowError(ArrowError),
    Common(String),
    PythonError(PyErr),
}

impl std::fmt::Display for PyDataFusionError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            PyDataFusionError::ExecutionError(e) => write!(f, "DataFusion error: {e}"),
            PyDataFusionError::ArrowError(e) => write!(f, "Arrow error: {e:?}"),
            PyDataFusionError::PythonError(e) => write!(f, "Python error {e:?}"),
            PyDataFusionError::Common(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for PyDataFusionError {}

impl From<DataFusionError> for PyDataFusionError {
    fn from(e: DataFusionError) -> Self {
        PyDataFusionError::ExecutionError(Box::new(e))
    }
}

impl From<ArrowError> for PyDataFusionError {
    fn from(e: ArrowError) -> Self {
        PyDataFusionError::ArrowError(e)
    }
}

impl From<PyErr> for PyDataFusionError {
    fn from(e: PyErr) -> Self {
        PyDataFusionError::PythonError(e)
    }
}

impl From<PyDataFusionError> for PyErr {
    fn from(e: PyDataFusionError) -> Self {
        match e {
            PyDataFusionError::PythonError(py_err) => py_err,
            _ => PyException::new_err(e.to_string()),
        }
    }
}

/// The tokio runtime that blocking Python entry points drive futures on.
pub fn get_tokio_runtime() -> &'static Runtime {
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| Runtime::new().unwrap())
}

/// Run `fut` to completion with the GIL released, staying responsive to
/// `KeyboardInterrupt`.
///
/// `py.check_signals` only raises a signal that a previous Python API call
/// already recorded, so the poll runs a no-op statement first to make the
/// interpreter process anything pending.
pub fn wait_for_future<F>(py: Python, fut: F) -> PyResult<F::Output>
where
    F: Future + Send,
    F::Output: Send,
{
    const INTERVAL_CHECK_SIGNALS: Duration = Duration::from_millis(1_000);
    let runtime = get_tokio_runtime();

    py.run(cr"pass", None, None)?;
    py.check_signals()?;

    py.detach(|| {
        runtime.block_on(async {
            tokio::pin!(fut);
            loop {
                tokio::select! {
                    res = &mut fut => break Ok(res),
                    _ = sleep(INTERVAL_CHECK_SIGNALS) => {
                        Python::attach(|py| {
                            py.run(cr"pass", None, None)?;
                            py.check_signals()
                        })?;
                    }
                }
            }
        })
    })
}

/// A physical plan handed back to Python for inspection.
#[pyclass(name = "ExecutionPlan", module = "datafusion_ray", subclass, skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct PyExecutionPlan {
    pub plan: Arc<dyn ExecutionPlan>,
}

impl PyExecutionPlan {
    pub fn new(plan: Arc<dyn ExecutionPlan>) -> Self {
        Self { plan }
    }
}

#[pymethods]
impl PyExecutionPlan {
    pub fn children(&self) -> Vec<PyExecutionPlan> {
        self.plan
            .children()
            .iter()
            .map(|p| PyExecutionPlan::new(Arc::clone(p)))
            .collect()
    }

    pub fn display(&self) -> String {
        format!("{}", displayable(self.plan.as_ref()).one_line())
    }

    pub fn display_indent(&self) -> String {
        format!("{}", displayable(self.plan.as_ref()).indent(false))
    }

    #[getter]
    pub fn partition_count(&self) -> usize {
        use datafusion::physical_plan::ExecutionPlanProperties;
        self.plan.output_partitioning().partition_count()
    }

    fn __repr__(&self) -> String {
        self.display_indent()
    }
}

/// A logical plan handed back to Python for inspection.
#[pyclass(name = "LogicalPlan", module = "datafusion_ray", subclass, skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct PyLogicalPlan {
    pub plan: Arc<LogicalPlan>,
}

impl PyLogicalPlan {
    pub fn new(plan: LogicalPlan) -> Self {
        Self {
            plan: Arc::new(plan),
        }
    }

    pub fn plan(&self) -> Arc<LogicalPlan> {
        Arc::clone(&self.plan)
    }
}

#[pymethods]
impl PyLogicalPlan {
    pub fn display(&self) -> String {
        format!("{}", self.plan.display())
    }

    pub fn display_indent(&self) -> String {
        format!("{}", self.plan.display_indent())
    }

    pub fn display_indent_schema(&self) -> String {
        format!("{}", self.plan.display_indent_schema())
    }

    fn __repr__(&self) -> String {
        self.display_indent()
    }
}
