use std::{fmt::Formatter, sync::Arc};

use datafusion::{
    common::tree_node::TreeNodeRecursion,
    error::Result,
    execution::SendableRecordBatchStream,
    physical_expr::PhysicalExpr,
    physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties},
};

use crate::util::max_rows_stream;

/// An Execution plan that will not yield batches with greater than max_rows.
///
/// If its input produces a batch with greater than max_rows it will zero-copy
/// split the batch and continue to do this until the remaining batch has
/// <= max_rows rows.    It will yield each of these batches as separate Items
#[derive(Debug)]
pub struct MaxRowsExec {
    pub input: Arc<dyn ExecutionPlan>,
    pub max_rows: usize,
}

impl MaxRowsExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, max_rows: usize) -> Self {
        Self { input, max_rows }
    }
}

impl DisplayAs for MaxRowsExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "MaxRowsExec[max_rows={}]", self.max_rows)
    }
}

impl ExecutionPlan for MaxRowsExec {
    fn name(&self) -> &str {
        "MaxRowsExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.input.properties()
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
        Ok(Arc::new(Self::new(children[0].clone(), self.max_rows)))
    }

    fn execute(
        &self,
        partition: usize,
        context: std::sync::Arc<datafusion::execution::TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        self.input
            .execute(partition, context)
            .map(|stream| max_rows_stream(stream, self.max_rows))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use arrow::array::{Int32Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::datasource::memory::MemorySourceConfig;
    use datafusion::physical_plan::{ExecutionPlanProperties, displayable};
    use datafusion::prelude::SessionContext;

    fn source(rows: usize) -> Arc<dyn ExecutionPlan> {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(
                (0..rows as i32).collect::<Vec<i32>>(),
            ))],
        )
        .unwrap();
        MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap()
            as Arc<dyn ExecutionPlan>
    }

    #[test]
    fn mirrors_its_input_and_names_the_limit() {
        let plan = Arc::new(MaxRowsExec::new(source(4), 8192)) as Arc<dyn ExecutionPlan>;
        assert_eq!(plan.name(), "MaxRowsExec");
        assert_eq!(plan.children().len(), 1);
        assert_eq!(plan.output_partitioning().partition_count(), 1);
        assert!(format!("{}", displayable(plan.as_ref()).one_line()).contains("max_rows=8192"));
    }

    /// Batches are capped so no single message over Flight gets too large.
    #[tokio::test]
    async fn splits_oversized_batches_and_preserves_rows() {
        let plan = Arc::new(MaxRowsExec::new(source(10), 3)) as Arc<dyn ExecutionPlan>;
        let mut s = plan.execute(0, SessionContext::new().task_ctx()).unwrap();

        let mut sizes = vec![];
        let mut total = 0;
        while let Some(b) = futures::StreamExt::next(&mut s).await {
            let b = b.unwrap();
            sizes.push(b.num_rows());
            total += b.num_rows();
        }
        assert_eq!(total, 10, "no rows lost");
        assert_eq!(sizes, vec![3, 3, 3, 1]);
    }

    #[tokio::test]
    async fn a_batch_within_the_limit_passes_through_whole() {
        let plan = Arc::new(MaxRowsExec::new(source(5), 100)) as Arc<dyn ExecutionPlan>;
        let mut s = plan.execute(0, SessionContext::new().task_ctx()).unwrap();
        let first = futures::StreamExt::next(&mut s).await.unwrap().unwrap();
        assert_eq!(first.num_rows(), 5);
        assert!(futures::StreamExt::next(&mut s).await.is_none());
    }

    #[test]
    fn replacing_children_keeps_the_limit() {
        let plan = Arc::new(MaxRowsExec::new(source(4), 7));
        #[allow(deprecated)]
        let replaced = plan.with_new_children(vec![source(9)]).unwrap();
        assert!(format!("{}", displayable(replaced.as_ref()).one_line()).contains("max_rows=7"));
    }
}
