use std::{fmt::Formatter, sync::Arc};

use datafusion::{
    common::internal_datafusion_err,
    common::tree_node::TreeNodeRecursion,
    error::Result,
    execution::SendableRecordBatchStream,
    physical_expr::PhysicalExpr,
    physical_plan::{
        DisplayAs, DisplayFormatType, EmptyRecordBatchStream, ExecutionPlan, Partitioning,
        PlanProperties,
    },
};
use log::error;

pub struct PartitionGroup(pub Vec<usize>);

/// This is a simple execution plan that isolates a partition from the input plan
/// It will advertise that it has a single partition and when
/// asked to execute, it will execute a particular partition from the child
/// input plan.
///
/// This allows us to execute Repartition Exec's on different processes
/// by showing each one only a single child partition
#[derive(Debug)]
pub struct PartitionIsolatorExec {
    pub input: Arc<dyn ExecutionPlan>,
    properties: Arc<PlanProperties>,
    pub partition_count: usize,
}

impl PartitionIsolatorExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, partition_count: usize) -> Self {
        // We advertise that we only have partition_count partitions
        let properties = Arc::new(
            PlanProperties::clone(input.properties())
                .with_partitioning(Partitioning::UnknownPartitioning(partition_count)),
        );

        Self {
            input,
            properties,
            partition_count,
        }
    }
}

impl DisplayAs for PartitionIsolatorExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        write!(
            f,
            "PartitionIsolatorExec [providing upto {} partitions]",
            self.partition_count
        )
    }
}

impl ExecutionPlan for PartitionIsolatorExec {
    fn name(&self) -> &str {
        "PartitionIsolatorExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&std::sync::Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        // this node owns no physical expressions
        Ok(TreeNodeRecursion::Continue)
    }

    #[allow(deprecated)]
    fn with_new_children(
        self: std::sync::Arc<Self>,
        children: Vec<std::sync::Arc<dyn ExecutionPlan>>,
    ) -> Result<std::sync::Arc<dyn ExecutionPlan>> {
        // TODO: generalize this
        assert_eq!(children.len(), 1);
        Ok(Arc::new(Self::new(
            children[0].clone(),
            self.partition_count,
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: std::sync::Arc<datafusion::execution::TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let config = context.session_config();
        let partition_group = &config
            .get_extension::<PartitionGroup>()
            .ok_or(internal_datafusion_err!(
                "PartitionGroup not set in session config"
            ))?
            .0;

        if partition > self.partition_count {
            error!(
                "PartitionIsolatorExec asked to execute partition {} but only has {} partitions",
                partition, self.partition_count
            );
            return Err(internal_datafusion_err!(
                "Invalid partition {} for PartitionIsolatorExec",
                partition
            ));
        }

        match partition_group.get(partition) {
            Some(actual_partition_number) => self.input.execute(*actual_partition_number, context),
            None => Ok(Box::pin(EmptyRecordBatchStream::new(self.input.schema()))
                as SendableRecordBatchStream),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use arrow::array::{Int32Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::datasource::memory::MemorySourceConfig;
    use datafusion::physical_plan::ExecutionPlanProperties;
    use datafusion::physical_plan::displayable;
    use datafusion::prelude::{SessionConfig, SessionContext};

    /// A source with `n` partitions, each holding its own partition index.
    fn source(n: usize) -> Arc<dyn ExecutionPlan> {
        let schema = Arc::new(Schema::new(vec![Field::new("p", DataType::Int32, false)]));
        let parts: Vec<Vec<RecordBatch>> = (0..n)
            .map(|i| {
                vec![
                    RecordBatch::try_new(
                        schema.clone(),
                        vec![Arc::new(Int32Array::from(vec![i as i32]))],
                    )
                    .unwrap(),
                ]
            })
            .collect();
        MemorySourceConfig::try_new_exec(&parts, schema, None).unwrap() as Arc<dyn ExecutionPlan>
    }

    fn ctx_with_group(group: Vec<usize>) -> Arc<datafusion::execution::TaskContext> {
        let config = SessionConfig::new().with_extension(Arc::new(PartitionGroup(group)));
        SessionContext::new_with_config(config).task_ctx()
    }

    async fn read(plan: &Arc<dyn ExecutionPlan>, partition: usize, group: Vec<usize>) -> Vec<i32> {
        let mut s = plan.execute(partition, ctx_with_group(group)).unwrap();
        let mut out = vec![];
        while let Some(b) = futures::StreamExt::next(&mut s).await {
            let b = b.unwrap();
            let col = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            out.extend(col.iter().flatten());
        }
        out
    }

    #[test]
    fn advertises_only_the_partitions_a_processor_will_serve() {
        let iso = PartitionIsolatorExec::new(source(4), 2);
        assert_eq!(iso.name(), "PartitionIsolatorExec");
        let iso: Arc<dyn ExecutionPlan> = Arc::new(iso);
        assert_eq!(iso.output_partitioning().partition_count(), 2);
        assert_eq!(iso.children().len(), 1);
        assert!(format!("{}", displayable(iso.as_ref()).one_line()).contains("providing upto 2"));
    }

    /// The core contract: output partition `i` reads input partition
    /// `partition_group[i]`, which is how one stage spreads over processors.
    #[tokio::test]
    async fn maps_output_partitions_onto_the_partition_group() {
        let iso = Arc::new(PartitionIsolatorExec::new(source(4), 2)) as Arc<dyn ExecutionPlan>;
        assert_eq!(read(&iso, 0, vec![2, 3]).await, vec![2]);
        assert_eq!(read(&iso, 1, vec![2, 3]).await, vec![3]);
        assert_eq!(read(&iso, 0, vec![0, 1]).await, vec![0]);
    }

    /// A group shorter than the advertised count leaves slots unfilled; those
    /// must come back empty rather than reading someone else's partition.
    #[tokio::test]
    async fn slots_beyond_the_group_are_empty() {
        let iso = Arc::new(PartitionIsolatorExec::new(source(4), 2)) as Arc<dyn ExecutionPlan>;
        assert_eq!(read(&iso, 0, vec![3]).await, vec![3]);
        assert!(read(&iso, 1, vec![3]).await.is_empty());
    }

    #[tokio::test]
    async fn execute_needs_the_partition_group_extension() {
        let iso = Arc::new(PartitionIsolatorExec::new(source(2), 1)) as Arc<dyn ExecutionPlan>;
        let bare = SessionContext::new().task_ctx();
        match iso.execute(0, bare) {
            Err(e) => assert!(e.to_string().contains("PartitionGroup"), "got: {e}"),
            Ok(_) => panic!("expected the missing-extension error"),
        }
    }

    #[test]
    fn replacing_children_keeps_the_partition_count() {
        let iso = Arc::new(PartitionIsolatorExec::new(source(4), 2));
        #[allow(deprecated)]
        let replaced = iso.with_new_children(vec![source(6)]).unwrap();
        assert_eq!(replaced.output_partitioning().partition_count(), 2);
    }
}
