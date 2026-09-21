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

use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::error::Result;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::joins::NestedLoopJoinExec;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use log::debug;
use std::sync::Arc;

use crate::stage::DFRayStageExec;
use crate::util::display_plan_with_partition_counts;

/// This optimizer rule walks up the physical plan tree
/// and inserts RayStageExec nodes where appropriate to denote where we will split
/// the plan into stages.
///
/// The RayStageExec nodes are merely markers to inform where to break the plan up.
///
/// Later, the plan will be examined again to actually split it up.
/// These RayStageExecs serve as markers where we know to break it up on a network
/// boundary and we can insert readers and writers as appropriate.
#[derive(Debug)]
pub struct RayStageOptimizerRule {}

impl Default for RayStageOptimizerRule {
    fn default() -> Self {
        Self::new()
    }
}

impl RayStageOptimizerRule {
    pub fn new() -> Self {
        Self {}
    }
}

impl PhysicalOptimizerRule for RayStageOptimizerRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &datafusion::config::ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        debug!(
            "optimizing physical plan:\n{}",
            display_plan_with_partition_counts(&plan)
        );

        let mut stage_counter = 0;

        let up = |plan: Arc<dyn ExecutionPlan>| {
            if plan.downcast_ref::<RepartitionExec>().is_some()
                || plan.downcast_ref::<SortExec>().is_some()
                || plan.downcast_ref::<NestedLoopJoinExec>().is_some()
            {
                let stage = Arc::new(DFRayStageExec::new(plan, stage_counter));
                stage_counter += 1;
                Ok(Transformed::yes(stage as Arc<dyn ExecutionPlan>))
            } else {
                Ok(Transformed::no(plan))
            }
        };

        let plan = plan.transform_up(up)?.data;
        let final_plan =
            Arc::new(DFRayStageExec::new(plan, stage_counter)) as Arc<dyn ExecutionPlan>;

        debug!(
            "optimized physical plan:\n{}",
            display_plan_with_partition_counts(&final_plan)
        );
        Ok(final_plan)
    }

    fn name(&self) -> &str {
        "RayStageOptimizerRule"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use arrow::array::{Int32Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::datasource::MemTable;
    use datafusion::physical_plan::displayable;
    use datafusion::prelude::{SessionConfig, SessionContext};

    async fn optimized(sql: &str, target_partitions: usize) -> String {
        let mut config = SessionConfig::new().with_target_partitions(target_partitions);
        crate::util::apply_planning_settings(&mut config);
        let state = datafusion::execution::SessionStateBuilder::new()
            .with_default_features()
            .with_physical_optimizer_rule(Arc::new(RayStageOptimizerRule::new()))
            .with_config(config)
            .build();
        let ctx = SessionContext::new_with_state(state);
        // registered directly rather than via CTAS: this context carries the
        // rule under test, whose markers have an unimplemented execute, so it
        // cannot run a query to build its own fixture
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("k", DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from((1..=30).collect::<Vec<i32>>())),
                Arc::new(Int32Array::from(
                    (1..=30).map(|v| v % 3).collect::<Vec<i32>>(),
                )),
            ],
        )
        .unwrap();
        // several partitions so the planner has something to repartition
        let table = MemTable::try_new(schema, vec![vec![batch.clone()], vec![batch]]).unwrap();
        ctx.register_table("t", Arc::new(table)).unwrap();
        let plan = ctx
            .sql(sql)
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        displayable(plan.as_ref()).indent(true).to_string()
    }

    #[test]
    fn rule_identifies_itself() {
        let rule = RayStageOptimizerRule::new();
        assert_eq!(rule.name(), "RayStageOptimizerRule");
        assert!(rule.schema_check());
        // Default and new agree
        assert_eq!(RayStageOptimizerRule::default().name(), rule.name());
    }

    /// Every plan gets a marker at the root, because the driver always reads the
    /// final stage over Flight.
    #[tokio::test]
    async fn a_trivial_plan_gets_exactly_one_stage() {
        let shown = optimized("select 1 as a", 1).await;
        assert_eq!(shown.matches("RayStageExec").count(), 1, "{shown}");
        assert!(shown.starts_with("RayStageExec"), "{shown}");
    }

    /// Repartition and sort are network boundaries: data has to move between
    /// processors there, so each one becomes its own stage.
    #[tokio::test]
    async fn repartition_and_sort_each_open_a_stage() {
        let shown = optimized("select k, count(*) from t group by k order by k", 3).await;
        let markers = shown.matches("RayStageExec").count();
        assert!(
            markers >= 3,
            "expected a stage per boundary plus the root:\n{shown}"
        );

        // a marker sits directly above each boundary operator
        for boundary in ["RepartitionExec", "SortExec"] {
            assert!(shown.contains(boundary), "no {boundary} in:\n{shown}");
        }
    }

    /// A scan-and-filter plan crosses no boundary, so it must not be split.
    #[tokio::test]
    async fn a_plan_with_no_boundary_is_not_split() {
        let shown = optimized("select a from t where a > 5", 1).await;
        assert_eq!(shown.matches("RayStageExec").count(), 1, "{shown}");
    }

    /// Stage ids are what the address map is keyed on, so they must be distinct.
    #[tokio::test]
    async fn stage_ids_are_unique() {
        let shown = optimized("select k, count(*) from t group by k order by k", 3).await;
        let mut ids: Vec<&str> = shown
            .match_indices("RayStageExec[")
            .map(|(i, _)| {
                let rest = &shown[i + "RayStageExec[".len()..];
                &rest[..rest.find(']').unwrap()]
            })
            .collect();
        let total = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), total, "duplicate stage ids in:\n{shown}");
    }
}
