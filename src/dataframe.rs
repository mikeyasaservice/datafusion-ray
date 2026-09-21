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

use arrow::array::RecordBatch;
use arrow::pyarrow::ToPyArrow;
use datafusion::common::internal_datafusion_err;
use datafusion::common::tree_node::Transformed;
use datafusion::common::tree_node::TreeNode;
use datafusion::error::DataFusionError;
use datafusion::execution::SendableRecordBatchStream;
// see coalesce_batches() below for why this deprecated operator is still used
use crate::pyerr::PyExecutionPlan;
use crate::pyerr::PyLogicalPlan;
use crate::pyerr::wait_for_future;
use crate::pyerr::{PyDataFusionError, PyDataFusionResult};
#[allow(deprecated)]
use datafusion::physical_plan::coalesce_batches::CoalesceBatchesExec;
use datafusion::physical_plan::displayable;
use datafusion::physical_plan::execution_plan::replace_children_if_necessary;
use datafusion::physical_plan::joins::NestedLoopJoinExec;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::{ExecutionPlan, ExecutionPlanProperties};
use datafusion::prelude::DataFrame;
use futures::stream::StreamExt;
use itertools::Itertools;
use log::trace;
use pyo3::exceptions::PyStopAsyncIteration;
use pyo3::exceptions::PyStopIteration;
use pyo3::prelude::*;
use std::borrow::Cow;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::isolator::PartitionIsolatorExec;
use crate::max_rows::MaxRowsExec;
use crate::pre_fetch::PrefetchExec;
use crate::stage::DFRayStageExec;
use crate::stage_reader::DFRayStageReaderExec;
use crate::util::collect_from_stage;
use crate::util::display_plan_with_partition_counts;
use crate::util::physical_plan_to_bytes;

/// Internal rust class beyind the DFRayDataFrame python object
///
/// It is a container for a plan for a query, as we would expect.
///
/// This class plays two important roles.  First, it defines the stages of the plan
/// by walking the plan provided to us in the constructor inside our dataframe.
/// That plan contains RayStageExec nodes, where are merely markers, that incidate to us where
/// to split the plan into descrete stages that can be hosted by a StageService.
///
/// The second role of this object is to be able to fetch record batches from the final_
/// stage in the plan and return them to python.
#[pyclass]
pub struct DFRayDataFrame {
    /// holds the logical plan of the query we will execute
    df: DataFrame,
    /// the physical plan we will use to consume the final stage.
    /// created when stages is run
    final_plan: Option<Arc<dyn ExecutionPlan>>,
}

impl DFRayDataFrame {
    pub fn new(df: DataFrame) -> Self {
        Self {
            df,
            final_plan: None,
        }
    }
}

#[pymethods]
impl DFRayDataFrame {
    #[pyo3(signature = (batch_size, prefetch_buffer_size, partitions_per_worker=None))]
    fn stages(
        &mut self,
        py: Python,
        batch_size: usize,
        prefetch_buffer_size: usize,
        partitions_per_worker: Option<usize>,
    ) -> PyDataFusionResult<Vec<PyDFRayStage>> {
        let physical_plan = wait_for_future(py, self.df.clone().create_physical_plan())??;
        let (stages, reader_plan) = build_stages(
            physical_plan,
            batch_size,
            prefetch_buffer_size,
            partitions_per_worker,
        )?;
        self.final_plan = Some(reader_plan);
        Ok(stages)
    }
    fn execution_plan(&self, py: Python) -> PyDataFusionResult<PyExecutionPlan> {
        let plan = wait_for_future(py, self.df.clone().create_physical_plan())??;
        Ok(PyExecutionPlan::new(plan))
    }

    fn display_execution_plan(&self, py: Python) -> PyDataFusionResult<String> {
        let plan = wait_for_future(py, self.df.clone().create_physical_plan())??;
        Ok(display_plan_with_partition_counts(&plan).to_string())
    }

    fn logical_plan(&self) -> PyResult<PyLogicalPlan> {
        Ok(PyLogicalPlan::new(self.df.logical_plan().clone()))
    }

    fn schema<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.df.schema().as_arrow().to_pyarrow(py)
    }

    fn optimized_logical_plan(&self) -> PyDataFusionResult<PyLogicalPlan> {
        Ok(PyLogicalPlan::new(self.df.clone().into_optimized_plan()?))
    }

    fn read_final_stage(
        &mut self,
        py: Python,
        stage_id: usize,
        stage_addr: &str,
    ) -> PyDataFusionResult<PyRecordBatchStream> {
        let stream = wait_for_future(
            py,
            collect_from_stage(
                stage_id,
                0,
                stage_addr,
                self.final_plan.take().unwrap().clone(),
            ),
        )??;
        Ok(PyRecordBatchStream::new(stream))
    }
}

/// DataFusion deprecated `CoalesceBatchesExec` in 52.0 in favour of arrow-rs's
/// `BatchCoalescer`, but there is no replacement *operator*: `BatchCoalescer`
/// is a stream-level utility used inside operators such as `FilterExec`, while
/// we assemble stages as plan trees and need a node. DataFusion still builds
/// this node itself and its proto codec still round-trips it, so we keep using
/// it and contain the deprecation to this one call.
#[allow(deprecated)]
fn coalesce_batches(input: Arc<dyn ExecutionPlan>, batch_size: usize) -> Arc<dyn ExecutionPlan> {
    Arc::new(CoalesceBatchesExec::new(input, batch_size)) as Arc<dyn ExecutionPlan>
}

/// Split a physical plan into the stages each processor will host.
///
/// Walks the plan bottom-up replacing every `DFRayStageExec` marker with a
/// `DFRayStageReaderExec`, recording the stage it displaced. Returns the stages
/// and the reader plan the driver uses to consume the last one.
///
/// Extracted from `DFRayDataFrame::stages` so it can be tested without a Python
/// interpreter: the only part that needed one was awaiting the physical plan.
#[allow(clippy::type_complexity)]
fn build_stages(
    physical_plan: Arc<dyn ExecutionPlan>,
    batch_size: usize,
    prefetch_buffer_size: usize,
    partitions_per_worker: Option<usize>,
) -> Result<(Vec<PyDFRayStage>, Arc<dyn ExecutionPlan>), DataFusionError> {
    let mut stages = vec![];

    let mut partition_groups = vec![];
    let mut full_partitions = false;
    // We walk up the tree from the leaves to find the stages, record ray stages, and replace
    // each ray stage with a corresponding ray reader stage.
    let up = |plan: Arc<dyn ExecutionPlan>| {
        trace!(
            "Examining plan up: {}",
            displayable(plan.as_ref()).one_line()
        );

        if let Some(stage_exec) = plan.downcast_ref::<DFRayStageExec>() {
            trace!("ray stage exec");
            let input = plan.children();
            assert!(input.len() == 1, "RayStageExec must have exactly one child");
            let input = input[0];

            let replacement = Arc::new(DFRayStageReaderExec::try_new(
                plan.output_partitioning().clone(),
                input.schema(),
                stage_exec.stage_id,
            )?) as Arc<dyn ExecutionPlan>;

            let stage = PyDFRayStage::new(
                stage_exec.stage_id,
                input.clone(),
                partition_groups.clone(),
                full_partitions,
            );
            partition_groups = vec![];
            full_partitions = false;

            stages.push(stage);
            Ok(Transformed::yes(replacement))
        } else if plan.downcast_ref::<RepartitionExec>().is_some() {
            trace!("repartition exec");
            let (calculated_partition_groups, replacement) = build_replacement(
                plan,
                prefetch_buffer_size,
                partitions_per_worker,
                true,
                batch_size,
                batch_size,
            )?;
            partition_groups = calculated_partition_groups;

            Ok(Transformed::yes(replacement))
        } else if plan.downcast_ref::<SortExec>().is_some() {
            trace!("sort exec");
            let (calculated_partition_groups, replacement) = build_replacement(
                plan,
                prefetch_buffer_size,
                partitions_per_worker,
                false,
                batch_size,
                batch_size,
            )?;
            partition_groups = calculated_partition_groups;
            full_partitions = true;

            Ok(Transformed::yes(replacement))
        } else if plan.downcast_ref::<NestedLoopJoinExec>().is_some() {
            trace!("nested loop join exec");
            // NestedLoopJoinExec must be on a stage by itself as it materializes the entire left
            // side of the join and is not suitable to be executed in a partitioned manner.
            let mut replacement = plan.clone();
            let partition_count = plan.output_partitioning().partition_count();
            trace!("nested join output partitioning {}", partition_count);

            replacement = Arc::new(MaxRowsExec::new(
                coalesce_batches(replacement, batch_size),
                batch_size,
            )) as Arc<dyn ExecutionPlan>;

            if prefetch_buffer_size > 0 {
                replacement = Arc::new(PrefetchExec::new(replacement, prefetch_buffer_size))
                    as Arc<dyn ExecutionPlan>;
            }
            partition_groups = vec![(0..partition_count).collect()];
            full_partitions = true;
            Ok(Transformed::yes(replacement))
        } else {
            trace!("not special case");
            Ok(Transformed::no(plan))
        }
    };

    physical_plan.transform_up(up)?;

    // add coalesce and max rows to last stage
    let mut last_stage = stages
        .pop()
        .ok_or(internal_datafusion_err!("No stages found"))?;

    if last_stage.num_output_partitions() > 1 {
        return Err(internal_datafusion_err!("Last stage expected to have one partition").into());
    }

    last_stage = PyDFRayStage::new(
        last_stage.stage_id,
        Arc::new(MaxRowsExec::new(
            coalesce_batches(last_stage.plan, batch_size),
            batch_size,
        )) as Arc<dyn ExecutionPlan>,
        vec![vec![0]],
        true,
    );

    // done fixing last stage

    let reader_plan = Arc::new(DFRayStageReaderExec::try_new_from_input(
        last_stage.plan.clone(),
        last_stage.stage_id,
    )?) as Arc<dyn ExecutionPlan>;

    stages.push(last_stage);

    Ok((stages, reader_plan))
}

#[allow(clippy::type_complexity)]
fn build_replacement(
    plan: Arc<dyn ExecutionPlan>,
    prefetch_buffer_size: usize,
    partitions_per_worker: Option<usize>,
    isolate: bool,
    max_rows: usize,
    inner_batch_size: usize,
) -> Result<(Vec<Vec<usize>>, Arc<dyn ExecutionPlan>), DataFusionError> {
    let mut replacement = plan.clone();
    let children = plan.children();
    assert!(children.len() == 1, "Unexpected plan structure");

    let child = children[0];
    let partition_count = child.output_partitioning().partition_count();
    trace!(
        "build_replacement for {}, partition_count: {}",
        displayable(plan.as_ref()).one_line(),
        partition_count
    );

    let partition_groups = match partitions_per_worker {
        Some(p) => (0..partition_count)
            .chunks(p)
            .into_iter()
            .map(|chunk| chunk.collect())
            .collect(),
        None => vec![(0..partition_count).collect()],
    };

    if isolate && partition_groups.len() > 1 {
        let new_child = Arc::new(PartitionIsolatorExec::new(
            child.clone(),
            partitions_per_worker.unwrap(), // we know it is a Some, here.
        ));
        replacement = replace_children_if_necessary(replacement.clone(), vec![new_child])?;
    }
    // insert a coalescing batches here too so that we aren't sending
    // too small (or too big) of batches over the network
    replacement = Arc::new(MaxRowsExec::new(
        coalesce_batches(replacement, inner_batch_size),
        max_rows,
    )) as Arc<dyn ExecutionPlan>;

    if prefetch_buffer_size > 0 {
        replacement = Arc::new(PrefetchExec::new(replacement, prefetch_buffer_size))
            as Arc<dyn ExecutionPlan>;
    }

    Ok((partition_groups, replacement))
}

/// A Python class to hold a PHysical plan of a single stage
#[pyclass]
pub struct PyDFRayStage {
    /// our stage id
    stage_id: usize,
    /// the physical plan of our stage
    plan: Arc<dyn ExecutionPlan>,
    /// the partition groups for this stage.
    partition_groups: Vec<Vec<usize>>,
    /// Are we hosting the complete partitions?  If not
    /// then RayStageReaderExecs will be inserted to consume its desired partition
    /// from all stages with this same id, and merge the results.  Using a
    /// CombinedRecordBatchStream
    full_partitions: bool,
}
impl PyDFRayStage {
    fn new(
        stage_id: usize,
        plan: Arc<dyn ExecutionPlan>,
        partition_groups: Vec<Vec<usize>>,
        full_partitions: bool,
    ) -> Self {
        Self {
            stage_id,
            plan,
            partition_groups,
            full_partitions,
        }
    }
}

#[pymethods]
impl PyDFRayStage {
    #[getter]
    fn stage_id(&self) -> usize {
        self.stage_id
    }

    #[getter]
    fn partition_groups(&self) -> Vec<Vec<usize>> {
        self.partition_groups.clone()
    }

    #[getter]
    fn full_partitions(&self) -> bool {
        self.full_partitions
    }

    /// returns the number of output partitions of this stage
    #[getter]
    fn num_output_partitions(&self) -> usize {
        self.plan.output_partitioning().partition_count()
    }

    /// returns the stage ids of that we need to read from in order to execute
    #[getter]
    pub fn child_stage_ids(&self) -> PyDataFusionResult<Vec<usize>> {
        let mut result = vec![];
        self.plan
            .clone()
            .transform_down(|node: Arc<dyn ExecutionPlan>| {
                if let Some(reader) = node.downcast_ref::<DFRayStageReaderExec>() {
                    result.push(reader.stage_id);
                }
                Ok(Transformed::no(node))
            })?;
        Ok(result)
    }

    pub fn execution_plan(&self) -> PyExecutionPlan {
        PyExecutionPlan::new(self.plan.clone())
    }

    fn display_execution_plan(&self) -> PyResult<String> {
        Ok(display_plan_with_partition_counts(&self.plan).to_string())
    }

    pub fn plan_bytes(&self) -> PyDataFusionResult<Cow<'_, [u8]>> {
        let plan_bytes = physical_plan_to_bytes(self.plan.clone())?;
        Ok(Cow::Owned(plan_bytes))
    }
}

// PyRecordBatch and PyRecordBatchStream are borrowed, and slightly modified from datafusion-python
// they are not publicly exposed in that repo

#[pyclass]
pub struct PyRecordBatch {
    pub batch: RecordBatch,
}

#[pymethods]
impl PyRecordBatch {
    fn to_pyarrow<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.batch.to_pyarrow(py)
    }
}

impl From<RecordBatch> for PyRecordBatch {
    fn from(batch: RecordBatch) -> Self {
        Self { batch }
    }
}

#[pyclass]
pub struct PyRecordBatchStream {
    stream: Arc<Mutex<SendableRecordBatchStream>>,
}

impl PyRecordBatchStream {
    pub fn new(stream: SendableRecordBatchStream) -> Self {
        Self {
            stream: Arc::new(Mutex::new(stream)),
        }
    }
}

#[pymethods]
impl PyRecordBatchStream {
    fn next<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let stream = self.stream.clone();
        wait_for_future(py, next_stream(stream, true))?.and_then(|b| b.to_pyarrow(py))
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.next(py)
    }

    fn __anext__<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let stream = self.stream.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, next_stream(stream, false))
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
}

async fn next_stream(
    stream: Arc<Mutex<SendableRecordBatchStream>>,
    sync: bool,
) -> PyResult<PyRecordBatch> {
    let mut stream = stream.lock().await;
    match stream.next().await {
        Some(Ok(batch)) => Ok(batch.into()),
        Some(Err(e)) => Err(PyDataFusionError::from(e))?,
        None => {
            // Depending on whether the iteration is sync or not, we raise either a
            // StopIteration or a StopAsyncIteration
            if sync {
                Err(PyStopIteration::new_err("stream exhausted"))
            } else {
                Err(PyStopAsyncIteration::new_err("stream exhausted"))
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use datafusion::physical_plan::ExecutionPlanProperties;
    use datafusion::prelude::{SessionConfig, SessionContext};

    /// A context whose plans split into stages the way the driver's does.
    fn ctx(target_partitions: usize) -> SessionContext {
        let mut config = SessionConfig::new().with_target_partitions(target_partitions);
        crate::util::apply_planning_settings(&mut config);
        let state = datafusion::execution::SessionStateBuilder::new()
            .with_default_features()
            .with_physical_optimizer_rule(Arc::new(crate::physical::RayStageOptimizerRule::new()))
            .with_config(config)
            .build();
        SessionContext::new_with_state(state)
    }

    async fn plan_for(ctx: &SessionContext, sql: &str) -> Arc<dyn ExecutionPlan> {
        ctx.sql(sql)
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap()
    }

    /// Registers three parquet files so scans split predictably.
    ///
    /// The files are written with a plain context: `ctx` carries
    /// `RayStageOptimizerRule`, which plants `DFRayStageExec` markers whose
    /// `execute` is `unimplemented!` by design, so it cannot run a COPY.
    async fn with_table(ctx: &SessionContext, dir: &std::path::Path) {
        let writer = SessionContext::new();
        for i in 0..3 {
            let path = dir.join(format!("p{i}.parquet"));
            writer
                .sql(&format!(
                    "copy (select {i} as k, v as a from generate_series(1, 50) t(v)) to '{}' \
                 stored as parquet",
                    path.display()
                ))
                .await
                .unwrap()
                .collect()
                .await
                .unwrap();
        }
        ctx.register_parquet(
            "t",
            dir.to_str().unwrap(),
            datafusion::prelude::ParquetReadOptions::default(),
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn last_stage_is_coalesced_to_one_partition() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ctx(3);
        with_table(&ctx, dir.path()).await;

        let plan = plan_for(&ctx, "select k, count(*) from t group by k order by k").await;
        let (stages, reader) = build_stages(plan, 8192, 0, Some(2)).unwrap();

        let last = stages.last().unwrap();
        assert_eq!(
            last.num_output_partitions(),
            1,
            "driver reads one partition"
        );
        assert!(last.full_partitions, "last stage hosts complete partitions");
        assert_eq!(last.partition_groups, vec![vec![0]]);
        // the reader the driver consumes is wired to the last stage
        assert_eq!(reader.output_partitioning().partition_count(), 1);
    }

    #[tokio::test]
    async fn stages_are_numbered_and_linked_child_to_parent() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ctx(3);
        with_table(&ctx, dir.path()).await;

        let plan = plan_for(&ctx, "select k, count(*) from t group by k order by k").await;
        let (stages, _) = build_stages(plan, 8192, 0, Some(2)).unwrap();

        assert!(
            stages.len() >= 2,
            "a repartition and a sort should both split"
        );
        let ids: Vec<usize> = stages.iter().map(|s| s.stage_id).collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(ids, sorted, "stage ids are unique and ascending");

        // every stage but the first consumes the one below it
        for w in stages.windows(2) {
            let children = w[1].child_stage_ids().unwrap();
            assert!(
                children.contains(&w[0].stage_id),
                "stage {} should read stage {}, got {children:?}",
                w[1].stage_id,
                w[0].stage_id
            );
        }
    }

    /// The isolator is what lets one stage span several processors, and it is
    /// only correct to insert it when the partitions actually split into more
    /// than one group.
    #[tokio::test]
    async fn isolator_is_inserted_only_when_partitions_span_processors() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ctx(3);
        with_table(&ctx, dir.path()).await;
        // the last stage must coalesce to a single partition, so the query needs an order by
        let sql = "select k, count(*) from t group by k order by k";

        let split = build_stages(plan_for(&ctx, sql).await, 8192, 0, Some(2))
            .unwrap()
            .0;
        assert!(
            split.iter().any(|s| s
                .display_execution_plan()
                .unwrap()
                .contains("PartitionIsolatorExec")),
            "two groups over three partitions needs an isolator"
        );
        assert_eq!(split[0].partition_groups, vec![vec![0, 1], vec![2]]);

        let whole = build_stages(plan_for(&ctx, sql).await, 8192, 0, Some(3))
            .unwrap()
            .0;
        assert!(
            !whole.iter().any(|s| s
                .display_execution_plan()
                .unwrap()
                .contains("PartitionIsolatorExec")),
            "a single group per stage must not be isolated"
        );
        assert_eq!(whole[0].partition_groups, vec![vec![0, 1, 2]]);

        let none = build_stages(plan_for(&ctx, sql).await, 8192, 0, None)
            .unwrap()
            .0;
        assert_eq!(none[0].partition_groups, vec![vec![0, 1, 2]]);
    }

    #[tokio::test]
    async fn prefetch_is_inserted_only_when_a_buffer_is_requested() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ctx(3);
        with_table(&ctx, dir.path()).await;
        // the last stage must coalesce to a single partition, so the query needs an order by
        let sql = "select k, count(*) from t group by k order by k";

        let off = build_stages(plan_for(&ctx, sql).await, 8192, 0, Some(2))
            .unwrap()
            .0;
        assert!(
            !off.iter()
                .any(|s| s.display_execution_plan().unwrap().contains("PrefetchExec"))
        );

        let on = build_stages(plan_for(&ctx, sql).await, 8192, 4, Some(2))
            .unwrap()
            .0;
        assert!(
            on.iter()
                .any(|s| s.display_execution_plan().unwrap().contains("PrefetchExec"))
        );
    }

    #[tokio::test]
    async fn every_stage_round_trips_through_the_codec() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ctx(3);
        with_table(&ctx, dir.path()).await;

        let plan = plan_for(&ctx, "select k, count(*) from t group by k order by k").await;
        let (stages, _) = build_stages(plan, 8192, 2, Some(2)).unwrap();

        // this is what actually crosses the wire to each processor
        for s in &stages {
            let bytes = s.plan_bytes().unwrap();
            assert!(!bytes.is_empty(), "stage {} serialized empty", s.stage_id);
        }
    }

    #[tokio::test]
    async fn a_plan_with_no_stage_markers_is_rejected() {
        // no RayStageOptimizerRule, so no DFRayStageExec markers are inserted
        let plain = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(1));
        let plan = plan_for(&plain, "select 1 as a").await;
        match build_stages(plan, 8192, 0, None) {
            Err(e) => assert!(e.to_string().contains("No stages found"), "got: {e}"),
            Ok((stages, _)) => panic!("expected an error, got {} stages", stages.len()),
        }
    }

    // ------------------------------------------------------------------
    // The Python-facing surface.
    //
    // `wait_for_future` drives futures on the crate's global tokio runtime,
    // and `block_on` cannot be called from inside another runtime, so these
    // are plain `#[test]`s that build their fixtures with `block_on` rather
    // than `#[tokio::test]`s.
    // ------------------------------------------------------------------

    use crate::pyerr::get_tokio_runtime;
    use arrow::array::Int32Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use futures::stream;

    fn ray_df(sql: &str) -> (tempfile::TempDir, DFRayDataFrame) {
        let dir = tempfile::tempdir().unwrap();
        let c = ctx(3);
        let df = get_tokio_runtime().block_on(async {
            with_table(&c, dir.path()).await;
            c.sql(sql).await.unwrap()
        });
        (dir, DFRayDataFrame::new(df))
    }

    #[test]
    fn the_plan_accessors_render_the_query_python_asks_about() {
        let (_dir, df) = ray_df("select k, count(*) from t group by k order by k");
        Python::attach(|py| {
            let physical = df.execution_plan(py).unwrap();
            assert!(
                physical.display_indent().contains("RayStageExec"),
                "got: {}",
                physical.display_indent()
            );

            let shown = df.display_execution_plan(py).unwrap();
            assert!(shown.contains("output_partitions"), "got: {shown}");

            let logical = df.logical_plan().unwrap();
            assert!(logical.display_indent().contains("Aggregate"));
            assert!(logical.display_indent_schema().contains("k"));
            assert!(!logical.display().is_empty());
            assert!(!logical.plan().schema().fields().is_empty());

            let optimized = df.optimized_logical_plan().unwrap();
            assert!(optimized.display_indent().contains("Aggregate"));

            let schema = df.schema(py).unwrap();
            assert!(schema.to_string().contains('k'), "got: {schema}");
        });
    }

    /// `stages()` is the pymethod wrapper over `build_stages`: it must both
    /// hand the stages back and keep the reader plan `read_final_stage` needs.
    #[test]
    fn stages_hands_back_the_stages_and_keeps_the_reader_plan() {
        let (_dir, mut df) = ray_df("select k, count(*) from t group by k order by k");
        Python::attach(|py| {
            assert!(df.final_plan.is_none());
            let stages = df.stages(py, 8192, 0, Some(2)).unwrap();
            assert!(stages.len() >= 2);
            assert!(df.final_plan.is_some(), "stages() records the reader plan");

            // the accessors python reads off each stage
            let last = stages.last().unwrap();
            assert!(last.child_stage_ids().unwrap().len() <= stages.len());
            assert!(
                last.execution_plan()
                    .display_indent()
                    .contains("RayStageReaderExec")
            );
            assert!(
                last.display_execution_plan()
                    .unwrap()
                    .contains("output_partitions")
            );
            assert!(!last.plan_bytes().unwrap().is_empty());

            // the first stage reads files, not another stage
            assert!(stages[0].child_stage_ids().unwrap().is_empty());
        });
    }

    fn int_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1, 2, 3]))]).unwrap()
    }

    fn py_stream(
        batches: Vec<Result<RecordBatch, DataFusionError>>,
    ) -> PyRecordBatchStream {
        let schema = int_batch().schema();
        PyRecordBatchStream::new(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::iter(batches),
        )))
    }

    #[test]
    fn a_batch_crosses_into_pyarrow() {
        let batch: PyRecordBatch = int_batch().into();
        Python::attach(|py| {
            let obj = batch.to_pyarrow(py).unwrap();
            assert_eq!(
                obj.getattr("num_rows").unwrap().extract::<usize>().unwrap(),
                3
            );
        });
    }

    #[test]
    fn a_stream_iterates_then_raises_stop_iteration() {
        let mut s = py_stream(vec![Ok(int_batch())]);
        Python::attach(|py| {
            let first = s.next(py).unwrap();
            assert_eq!(
                first.getattr("num_rows").unwrap().extract::<usize>().unwrap(),
                3
            );
            let err = s.__next__(py).unwrap_err();
            assert!(
                err.is_instance_of::<PyStopIteration>(py),
                "exhaustion must end a for loop, got: {err}"
            );
        });
    }

    #[test]
    fn a_stream_error_surfaces_as_a_python_exception() {
        let mut s = py_stream(vec![Err(DataFusionError::Internal("boom".into()))]);
        Python::attach(|py| {
            let err = s.next(py).unwrap_err();
            assert!(err.to_string().contains("boom"), "got: {err}");
        });
    }

    #[test]
    fn a_stream_is_its_own_iterator() {
        Python::attach(|py| {
            let obj = Py::new(py, py_stream(vec![])).unwrap();
            let bound = obj.bind(py);
            for dunder in ["__iter__", "__aiter__"] {
                let same = bound.call_method0(dunder).unwrap();
                assert!(same.is(bound), "{dunder} must return self");
            }

            // no asyncio loop runs inside a rust test, so the async arm can
            // only report that -- but it is the same line python awaits.
            let err = bound.call_method0("__anext__").unwrap_err();
            assert!(err.to_string().contains("loop"), "got: {err}");
        });
    }
}
