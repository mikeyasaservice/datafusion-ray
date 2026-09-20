use std::sync::Arc;

use crate::{
    isolator::PartitionIsolatorExec,
    max_rows::MaxRowsExec,
    pre_fetch::PrefetchExec,
    protobuf::{
        DfRayStageReaderExecNode, MaxRowsExecNode, PartitionIsolatorExecNode, PrefetchExecNode,
    },
};

use arrow::datatypes::Schema;
use datafusion::{
    common::{internal_datafusion_err, internal_err},
    error::Result,
    execution::TaskContext,
    physical_plan::ExecutionPlan,
};
use datafusion_proto::physical_plan::{
    PhysicalExtensionCodec, PhysicalPlanDecodeContext, PhysicalProtoConverterExtension,
    from_proto::parse_protobuf_partitioning, to_proto::serialize_partitioning,
};
use datafusion_proto::protobuf;

use prost::Message;

use crate::stage_reader::DFRayStageReaderExec;

#[derive(Debug)]
/// Physical Extension Codec for for DataFusion for Ray plans
pub struct RayCodec {}

impl PhysicalExtensionCodec for RayCodec {
    fn try_decode(
        &self,
        buf: &[u8],
        inputs: &[Arc<dyn ExecutionPlan>],
        ctx: &TaskContext,
        proto_converter: &dyn PhysicalProtoConverterExtension,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // TODO: clean this up
        if let Ok(node) = PartitionIsolatorExecNode::decode(buf) {
            if inputs.len() != 1 {
                Err(internal_datafusion_err!(
                    "PartitionIsolatorExec requires one input"
                ))
            } else {
                Ok(Arc::new(PartitionIsolatorExec::new(
                    inputs[0].clone(),
                    node.partition_count as usize,
                )))
            }
        } else if let Ok(node) = DfRayStageReaderExecNode::decode(buf) {
            let schema: Schema = node
                .schema
                .as_ref()
                .ok_or(internal_datafusion_err!("missing schema in proto"))?
                .try_into()?;

            let decode_ctx = PhysicalPlanDecodeContext::new(ctx, self);
            let part = parse_protobuf_partitioning(
                node.partitioning.as_ref(),
                &decode_ctx,
                &schema,
                proto_converter,
            )?
            .ok_or(internal_datafusion_err!("missing partitioning in proto"))?;

            Ok(Arc::new(DFRayStageReaderExec::try_new(
                part,
                Arc::new(schema),
                node.stage_id as usize,
            )?))
        } else if let Ok(node) = MaxRowsExecNode::decode(buf) {
            if inputs.len() != 1 {
                Err(internal_datafusion_err!(
                    "MaxRowsExec requires one input, got {}",
                    inputs.len()
                ))
            } else {
                Ok(Arc::new(MaxRowsExec::new(
                    inputs[0].clone(),
                    node.max_rows as usize,
                )))
            }
        } else if let Ok(node) = PrefetchExecNode::decode(buf) {
            if inputs.len() != 1 {
                Err(internal_datafusion_err!(
                    "MaxRowsExec requires one input, got {}",
                    inputs.len()
                ))
            } else {
                Ok(Arc::new(PrefetchExec::new(
                    inputs[0].clone(),
                    node.buf_size as usize,
                )))
            }
        } else {
            internal_err!("Should not reach this point")
        }
    }

    fn try_encode(
        &self,
        node: Arc<dyn ExecutionPlan>,
        buf: &mut Vec<u8>,
        proto_converter: &dyn PhysicalProtoConverterExtension,
    ) -> Result<()> {
        if let Some(reader) = node.downcast_ref::<DFRayStageReaderExec>() {
            let schema: protobuf::Schema = reader.schema().try_into()?;
            let partitioning: protobuf::Partitioning = serialize_partitioning(
                reader.properties().output_partitioning(),
                self,
                proto_converter,
            )?;

            let pb = DfRayStageReaderExecNode {
                schema: Some(schema),
                partitioning: Some(partitioning),
                stage_id: reader.stage_id as u64,
            };

            pb.encode(buf)
                .map_err(|e| internal_datafusion_err!("can't encode ray stage reader pb: {e}"))?;
            Ok(())
        } else if let Some(pi) = node.downcast_ref::<PartitionIsolatorExec>() {
            let pb = PartitionIsolatorExecNode {
                dummy: 0.0,
                partition_count: pi.partition_count as u64,
            };

            pb.encode(buf)
                .map_err(|e| internal_datafusion_err!("can't encode partition isolator pb: {e}"))?;

            Ok(())
        } else if let Some(max) = node.downcast_ref::<MaxRowsExec>() {
            let pb = MaxRowsExecNode {
                max_rows: max.max_rows as u64,
            };
            pb.encode(buf)
                .map_err(|e| internal_datafusion_err!("can't encode max rows pb: {e}"))?;

            Ok(())
        } else if let Some(pre) = node.downcast_ref::<PrefetchExec>() {
            let pb = PrefetchExecNode {
                dummy: 0,
                buf_size: pre.buf_size as u64,
            };
            pb.encode(buf)
                .map_err(|e| internal_datafusion_err!("can't encode prefetch pb: {e}"))?;

            Ok(())
        } else {
            internal_err!("Not supported")
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::stage_reader::DFRayStageReaderExec;
    use arrow::datatypes::DataType;
    use datafusion::{
        physical_plan::{Partitioning, display::DisplayableExecutionPlan, displayable},
        prelude::SessionContext,
    };
    use datafusion_proto::physical_plan::{AsExecutionPlan, DefaultPhysicalProtoConverter};

    use std::sync::Arc;

    #[test]
    fn stage_reader_round_trip() {
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("a", DataType::Int32, false),
            arrow::datatypes::Field::new("b", DataType::Int32, false),
        ]));
        let ctx = SessionContext::new();
        let part = Partitioning::UnknownPartitioning(2);
        let exec = Arc::new(DFRayStageReaderExec::try_new(part, schema, 1).unwrap());
        let codec = RayCodec {};
        let converter = DefaultPhysicalProtoConverter {};
        let mut buf = vec![];
        codec
            .try_encode(exec.clone(), &mut buf, &converter)
            .unwrap();
        let decoded = codec
            .try_decode(&buf, &[], &ctx.task_ctx(), &converter)
            .unwrap();
        assert_eq!(exec.schema(), decoded.schema());
    }
    #[test]
    fn max_rows_and_reader_round_trip() {
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("a", DataType::Int32, false),
            arrow::datatypes::Field::new("b", DataType::Int32, false),
        ]));
        let ctx = SessionContext::new();
        let part = Partitioning::UnknownPartitioning(2);
        let exec = Arc::new(MaxRowsExec::new(
            Arc::new(DFRayStageReaderExec::try_new(part, schema, 1).unwrap()),
            10,
        ));
        let codec = RayCodec {};

        // serialize execution plan to proto
        let proto: protobuf::PhysicalPlanNode =
            protobuf::PhysicalPlanNode::try_from_physical_plan(exec.clone(), &codec)
                .expect("to proto");

        // deserialize proto back to execution plan
        let result_exec_plan: Arc<dyn ExecutionPlan> = proto
            .try_into_physical_plan(&ctx.task_ctx(), &codec)
            .expect("from proto");

        let input = displayable(exec.as_ref()).indent(true).to_string();
        let round_trip = {
            let plan: &dyn ExecutionPlan = result_exec_plan.as_ref();
            DisplayableExecutionPlan::new(plan)
        }
        .indent(true)
        .to_string();
        assert_eq!(input, round_trip);
    }

    #[test]
    fn partition_isolator_round_trip() {
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("a", DataType::Int32, false),
        ]));
        let ctx = SessionContext::new();
        let part = Partitioning::UnknownPartitioning(3);
        let reader = Arc::new(DFRayStageReaderExec::try_new(part, schema, 1).unwrap());
        let exec = Arc::new(PartitionIsolatorExec::new(reader, 1)) as Arc<dyn ExecutionPlan>;
        let codec = RayCodec {};

        let proto: protobuf::PhysicalPlanNode =
            protobuf::PhysicalPlanNode::try_from_physical_plan(exec.clone(), &codec)
                .expect("to proto");
        let round_tripped: Arc<dyn ExecutionPlan> = proto
            .try_into_physical_plan(&ctx.task_ctx(), &codec)
            .expect("from proto");

        assert_eq!(
            displayable(exec.as_ref()).indent(true).to_string(),
            displayable(round_tripped.as_ref()).indent(true).to_string()
        );
    }

    /// The physical plan is serialized on the driver and rebuilt inside each
    /// processor, which then executes a single partition of it. A single large
    /// file is split into byte ranges across partitions, and if those ranges do
    /// not survive the round trip every processor reads the whole file and
    /// results are silently multiplied by the processor count.
    #[tokio::test]
    async fn file_scan_partitioning_survives_round_trip() {
        use datafusion::physical_plan::ExecutionPlanProperties;
        use datafusion::prelude::SessionConfig;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.parquet");

        let mut config = SessionConfig::new().with_target_partitions(3);
        let opts = config.options_mut();
        // split even a small file, and write many row groups so it can be split
        opts.set("datafusion.optimizer.repartition_file_min_size", "16")
            .unwrap();
        opts.set("datafusion.execution.parquet.max_row_group_size", "100")
            .unwrap();
        let ctx = SessionContext::new_with_config(config);

        ctx.sql(&format!(
            "copy (select v as a from generate_series(1, 20000) t(v)) to '{}' \
             stored as parquet",
            path.display()
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

        ctx.register_parquet(
            "t",
            path.to_str().unwrap(),
            datafusion::prelude::ParquetReadOptions::default(),
        )
        .await
        .unwrap();

        let plan = ctx
            .sql("select a from t")
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        let parts = plan.output_partitioning().partition_count();
        assert!(
            parts > 1,
            "test needs a split scan, got {parts} partition(s)"
        );

        let bytes = crate::util::physical_plan_to_bytes(plan.clone()).unwrap();
        // a processor rebuilds the plan in a fresh session, as
        // DFRayProcessorService::update_plan does, and executes it under a
        // second session built by configure_ctx
        let decode_ctx = SessionContext::new();
        let round_tripped = crate::util::bytes_to_physical_plan(&decode_ctx, &bytes).unwrap();
        let exec_ctx = SessionContext::new();
        crate::util::register_object_store_for_paths_in_plan(&exec_ctx, round_tripped.clone())
            .unwrap();

        let count = async |p: &Arc<dyn ExecutionPlan>, part: usize, c: &SessionContext| -> usize {
            let mut s = p.execute(part, c.task_ctx()).unwrap();
            let mut n = 0;
            while let Some(b) = futures::StreamExt::next(&mut s).await {
                n += b.unwrap().num_rows();
            }
            n
        };

        let mut before = vec![];
        let mut after = vec![];
        for part in 0..parts {
            before.push(count(&plan, part, &ctx).await);
            after.push(count(&round_tripped, part, &exec_ctx).await);
        }

        assert_eq!(before.iter().sum::<usize>(), 20000, "sanity: local plan");
        assert_eq!(
            after, before,
            "per-partition row counts changed across the proto round trip"
        );
    }
}
