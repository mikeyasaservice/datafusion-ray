use std::{fmt::Formatter, sync::Arc};

use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::error::Result;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use datafusion::{arrow::datatypes::SchemaRef, execution::SendableRecordBatchStream};
use futures::stream::StreamExt;
use tokio::sync::mpsc::channel;

/// An execution plan that will try to consume and buffer RecordBatches from its input.
/// It will hold those buffers in a bounded channel and serve them from the channel requested
/// through execute().   
///
/// The buffering begins when execute() is called.
#[derive(Debug)]
pub struct PrefetchExec {
    /// Input plan
    pub(crate) input: Arc<dyn ExecutionPlan>,
    /// maximum amount of buffered RecordBatches
    pub(crate) buf_size: usize,
    /// our plan Properties, the same as our input
    properties: Arc<PlanProperties>,
}

impl PrefetchExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, buf_size: usize) -> Self {
        // check for only one input
        if input.children().len() != 1 {
            panic!("PrefetchExec must have exactly one input");
        }
        let properties = input.children()[0].properties().clone();
        Self {
            input,
            buf_size,
            properties,
        }
    }
}
impl DisplayAs for PrefetchExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "PrefetchExec [num={}]", self.buf_size)
    }
}

impl ExecutionPlan for PrefetchExec {
    fn schema(&self) -> SchemaRef {
        self.input.schema()
    }
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn name(&self) -> &str {
        "PrefetchExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
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
    ) -> datafusion::error::Result<std::sync::Arc<dyn ExecutionPlan>> {
        // TODO: handle more general case
        assert_eq!(children.len(), 1);
        let child = children[0].clone();
        Ok(Arc::new(PrefetchExec::new(child, self.buf_size)))
    }

    fn execute(
        &self,
        partition: usize,
        context: std::sync::Arc<datafusion::execution::TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let (tx, mut rx) = channel(self.buf_size);

        let mut input_stream = self.input.execute(partition, context)?;

        let consume_fut = async move {
            while let Some(batch) = input_stream.next().await {
                // TODO: how to neatly errors within this macro?
                tx.send(batch).await.unwrap();
            }
        };

        tokio::spawn(consume_fut);

        let out_stream = async_stream::stream! {
            while let Some(batch) = rx.recv().await {
                yield batch;
            }
        };

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema().clone(),
            out_stream,
        )))
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

    fn source(batches: usize) -> Arc<dyn ExecutionPlan> {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let data: Vec<RecordBatch> = (0..batches as i32)
            .map(|i| {
                RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![i]))])
                    .unwrap()
            })
            .collect();
        MemorySourceConfig::try_new_exec(&[data], schema, None).unwrap() as Arc<dyn ExecutionPlan>
    }

    /// Wrapping a source directly would leave nothing to buffer, so the node
    /// insists on a single-child input.
    #[test]
    #[should_panic(expected = "exactly one input")]
    fn rejects_an_input_that_is_not_single_child() {
        let _ = PrefetchExec::new(source(1), 2);
    }

    fn wrapped(batches: usize, buf: usize) -> Arc<dyn ExecutionPlan> {
        let inner = Arc::new(crate::max_rows::MaxRowsExec::new(source(batches), 8192));
        Arc::new(PrefetchExec::new(inner, buf)) as Arc<dyn ExecutionPlan>
    }

    #[test]
    fn reports_the_buffer_size() {
        let plan = wrapped(3, 4);
        assert_eq!(plan.name(), "PrefetchExec");
        assert_eq!(plan.children().len(), 1);
        assert_eq!(plan.output_partitioning().partition_count(), 1);
        assert!(format!("{}", displayable(plan.as_ref()).one_line()).contains("num=4"));
    }

    /// Buffering must not reorder or drop anything.
    #[tokio::test]
    async fn forwards_every_batch_in_order() {
        let plan = wrapped(5, 2);
        let mut s = plan.execute(0, SessionContext::new().task_ctx()).unwrap();
        let mut seen = vec![];
        while let Some(b) = futures::StreamExt::next(&mut s).await {
            let b = b.unwrap();
            let col = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            seen.extend(col.iter().flatten());
        }
        assert_eq!(seen, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn replacing_children_keeps_the_buffer_size() {
        let plan = Arc::new(PrefetchExec::new(
            Arc::new(crate::max_rows::MaxRowsExec::new(source(2), 8192)),
            3,
        ));
        let new_child = Arc::new(crate::max_rows::MaxRowsExec::new(source(4), 8192));
        #[allow(deprecated)]
        let replaced = plan.with_new_children(vec![new_child]).unwrap();
        assert!(format!("{}", displayable(replaced.as_ref()).one_line()).contains("num=3"));
    }
}
