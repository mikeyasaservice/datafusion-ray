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
