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

use crate::pyerr::PyDataFusionResult;
use crate::pyerr::wait_for_future;
use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::listing::{ListingOptions, ListingTableUrl};
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::{CsvReadOptions, ParquetReadOptions, SessionConfig, SessionContext};
use log::debug;
use pyo3::prelude::*;
use std::sync::Arc;

use crate::dataframe::DFRayDataFrame;
use crate::physical::RayStageOptimizerRule;
use crate::util::{ResultExt, apply_planning_settings, maybe_register_object_store};

/// Internal Session Context object for the python class DFRayContext
#[pyclass]
pub struct DFRayContext {
    /// our datafusion context
    ctx: SessionContext,
}

#[pymethods]
impl DFRayContext {
    #[new]
    pub fn new() -> PyResult<Self> {
        let rule = RayStageOptimizerRule::new();

        let mut config = SessionConfig::default().with_information_schema(true);
        apply_planning_settings(&mut config);

        let state = SessionStateBuilder::new()
            .with_default_features()
            .with_physical_optimizer_rule(Arc::new(rule))
            .with_config(config)
            .build();

        let ctx = SessionContext::new_with_state(state);

        Ok(Self { ctx })
    }

    pub fn register_parquet(
        &self,
        py: Python,
        name: String,
        path: String,
    ) -> PyDataFusionResult<()> {
        let options = ParquetReadOptions::default();

        let url = ListingTableUrl::parse(&path).to_py_err()?;

        maybe_register_object_store(&self.ctx, url.as_ref()).to_py_err()?;
        debug!("register_parquet: registering table {} at {}", name, path);

        wait_for_future(py, self.ctx.register_parquet(&name, &path, options.clone()))??;
        Ok(())
    }

    pub fn register_csv(&self, py: Python, name: String, path: String) -> PyDataFusionResult<()> {
        let options = CsvReadOptions::default();

        let url = ListingTableUrl::parse(&path).to_py_err()?;

        maybe_register_object_store(&self.ctx, url.as_ref()).to_py_err()?;
        debug!("register_csv: registering table {} at {}", name, path);

        wait_for_future(py, self.ctx.register_csv(&name, &path, options.clone()))??;
        Ok(())
    }

    #[pyo3(signature = (name, path, file_extension=".parquet"))]
    pub fn register_listing_table(
        &mut self,
        py: Python,
        name: &str,
        path: &str,
        file_extension: &str,
    ) -> PyDataFusionResult<()> {
        let options =
            ListingOptions::new(Arc::new(ParquetFormat::new())).with_file_extension(file_extension);

        let path = format!("{path}/");
        let url = ListingTableUrl::parse(&path).to_py_err()?;

        maybe_register_object_store(&self.ctx, url.as_ref()).to_py_err()?;

        debug!(
            "register_listing_table: registering table {} at {}",
            name, path
        );
        wait_for_future(
            py,
            self.ctx
                .register_listing_table(name, path, options, None, None),
        )??;
        Ok(())
    }

    pub fn sql(&self, py: Python, query: String) -> PyDataFusionResult<DFRayDataFrame> {
        let df = wait_for_future(py, self.ctx.sql(&query))??;

        Ok(DFRayDataFrame::new(df))
    }

    pub fn set(&self, option: String, value: String) -> PyDataFusionResult<()> {
        let state = self.ctx.state_ref();
        let mut guard = state.write();
        let config = guard.config_mut();
        let options = config.options_mut();
        options.set(&option, &value)?;

        Ok(())
    }

    pub fn get_target_partitions(&self) -> usize {
        let state = self.ctx.state_ref();
        let guard = state.read();
        let config = guard.config();
        let options = config.options();
        options.execution.target_partitions
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::pyerr::get_tokio_runtime;
    use std::io::Write;
    use tempfile::TempDir;

    /// A directory holding a single parquet file with one `a` column.
    ///
    /// Written by a plain `SessionContext`: the context under test carries
    /// `RayStageOptimizerRule`, and its marker node is `unimplemented!()` to
    /// execute, so it cannot build its own fixtures.
    fn parquet_fixture() -> TempDir {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("d.parquet");
        get_tokio_runtime().block_on(async {
            let ctx = SessionContext::new();
            ctx.sql(&format!(
                "copy (select v as a from generate_series(1, 10) t(v)) to '{}' stored as parquet",
                path.display()
            ))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        });
        dir
    }

    #[test]
    fn registered_tables_are_queryable() {
        let dir = parquet_fixture();
        let parquet = dir.path().join("d.parquet");

        let csv = dir.path().join("d.csv");
        let mut f = std::fs::File::create(&csv).unwrap();
        writeln!(f, "a\n1\n2\n3").unwrap();
        drop(f);

        let mut ctx = DFRayContext::new().unwrap();
        Python::attach(|py| {
            ctx.register_parquet(py, "p".into(), parquet.display().to_string())
                .unwrap();
            ctx.register_csv(py, "c".into(), csv.display().to_string())
                .unwrap();
            ctx.register_listing_table(py, "l", dir.path().to_str().unwrap(), ".parquet")
                .unwrap();

            // planning resolves the table, so a successful `sql` is the
            // proof that the registration took
            for table in ["p", "c", "l"] {
                assert!(
                    ctx.sql(py, format!("select count(*) from {table}")).is_ok(),
                    "{table} was not registered"
                );
            }
        });
    }

    #[test]
    fn registering_a_missing_file_is_an_error() {
        let ctx = DFRayContext::new().unwrap();
        Python::attach(|py| {
            match ctx.register_parquet(py, "p".into(), "/nonexistent/nope.parquet".into()) {
                Ok(()) => panic!("expected a failure for a path that does not exist"),
                Err(e) => assert!(!e.to_string().is_empty()),
            }
            match ctx.register_csv(py, "c".into(), "/nonexistent/nope.csv".into()) {
                Ok(()) => panic!("expected a failure for a path that does not exist"),
                Err(e) => assert!(!e.to_string().is_empty()),
            }
        });
    }

    #[test]
    fn a_bad_query_is_reported_not_panicked() {
        let ctx = DFRayContext::new().unwrap();
        Python::attach(|py| match ctx.sql(py, "select * from nowhere".into()) {
            Ok(_) => panic!("expected a planning failure for an unregistered table"),
            Err(e) => assert!(e.to_string().contains("nowhere"), "got: {e}"),
        });
    }

    #[test]
    fn settings_round_trip_and_reject_unknown_keys() {
        let ctx = DFRayContext::new().unwrap();
        let before = ctx.get_target_partitions();
        assert!(before > 0);

        ctx.set("datafusion.execution.target_partitions".into(), "7".into())
            .unwrap();
        assert_eq!(ctx.get_target_partitions(), 7);

        match ctx.set("datafusion.not.a.real.option".into(), "1".into()) {
            Ok(()) => panic!("expected an unknown config key to be rejected"),
            Err(e) => assert!(e.to_string().contains("not.a.real.option"), "got: {e}"),
        }
    }

    /// The planning settings are what keep TPC-H q4 and q11 working; a context
    /// built here must carry them.
    #[test]
    fn a_new_context_carries_the_planning_settings() {
        let ctx = DFRayContext::new().unwrap();
        let state = ctx.ctx.state_ref();
        let guard = state.read();
        let opts = guard.config().options();
        assert_eq!(opts.optimizer.hash_join_single_partition_threshold, 0);
        assert_eq!(opts.optimizer.hash_join_single_partition_threshold_rows, 0);
        assert!(!opts.optimizer.enable_physical_uncorrelated_scalar_subquery);
        assert!(opts.catalog.information_schema);
    }
}
