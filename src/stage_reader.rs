use std::{fmt::Formatter, sync::Arc};

use arrow_flight::{FlightClient, Ticket};
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{internal_datafusion_err, internal_err};
use datafusion::error::Result;
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
};
use datafusion::{arrow::datatypes::SchemaRef, execution::SendableRecordBatchStream};
use futures::StreamExt;
use futures::stream::TryStreamExt;
use log::trace;
use prost::Message;

use crate::processor_service::ServiceClients;
use crate::protobuf::FlightTicketData;
use crate::util::CombinedRecordBatchStream;

/// An [`ExecutionPlan`] that will produce a stream of batches fetched from another stage
/// which is hosted by a [`crate::stage_service::StageService`] separated from a network boundary
///
/// Note that discovery of the service is handled by populating an instance of [`crate::stage_service::ServiceClients`]
/// and storing it as an extension in the [`datafusion::execution::TaskContext`] configuration.
#[derive(Debug)]
pub struct DFRayStageReaderExec {
    properties: Arc<PlanProperties>,
    schema: SchemaRef,
    pub stage_id: usize,
}

impl DFRayStageReaderExec {
    pub fn try_new_from_input(input: Arc<dyn ExecutionPlan>, stage_id: usize) -> Result<Self> {
        let properties = input.properties().clone();

        Self::try_new(properties.partitioning.clone(), input.schema(), stage_id)
    }

    pub fn try_new(partitioning: Partitioning, schema: SchemaRef, stage_id: usize) -> Result<Self> {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            Partitioning::UnknownPartitioning(partitioning.partition_count()),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));

        Ok(Self {
            properties,
            schema,
            stage_id,
        })
    }
}
impl DisplayAs for DFRayStageReaderExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        write!(
            f,
            "RayStageReaderExec[{}] (output_partitioning={:?})",
            self.stage_id,
            self.properties().partitioning
        )
    }
}

impl ExecutionPlan for DFRayStageReaderExec {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn name(&self) -> &str {
        "RayStageReaderExec"
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
        _children: Vec<std::sync::Arc<dyn ExecutionPlan>>,
    ) -> datafusion::error::Result<std::sync::Arc<dyn ExecutionPlan>> {
        // TODO: handle more general case
        unimplemented!()
    }

    fn execute(
        &self,
        partition: usize,
        context: std::sync::Arc<datafusion::execution::TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let name = format!("RayStageReaderExec[{}-{}]:", self.stage_id, partition);
        trace!("{name} execute");
        let client_map = &context
            .session_config()
            .get_extension::<ServiceClients>()
            .ok_or(internal_datafusion_err!(
                "{name} Flight Client not in context"
            ))?
            .clone()
            .0;

        trace!("{name} client_map keys {:?}", client_map.keys());

        let clients = client_map
            .get(&(self.stage_id, partition))
            .ok_or(internal_datafusion_err!(
                "{} No flight clients found for {}:{}, have {:?}",
                name,
                self.stage_id,
                partition,
                client_map.keys()
            ))?
            .lock()
            .iter()
            .map(|c| {
                let inner_clone = c.inner().clone();
                FlightClient::new_from_inner(inner_clone)
            })
            .collect::<Vec<_>>();

        let ftd = FlightTicketData {
            dummy: false,
            partition: partition as u64,
        };

        let ticket = Ticket {
            ticket: ftd.encode_to_vec().into(),
        };

        let schema = self.schema.clone();

        let stream = async_stream::stream! {
            let mut error = false;

            let mut streams = vec![];
            for mut client in clients {
                let name = name.clone();
                trace!("{name} Getting flight stream" );
                match client.do_get(ticket.clone()).await {
                    Ok(flight_stream) => {
                        trace!("{name} Got flight stream. headers:{:?}", flight_stream.headers());
                        let rbr_stream = RecordBatchStreamAdapter::new(schema.clone(),
                            flight_stream
                                .map_err(move |e| internal_datafusion_err!("{} Error consuming flight stream: {}", name, e)));

                        streams.push(Box::pin(rbr_stream) as SendableRecordBatchStream);
                    },
                    Err(e) => {
                        error = true;
                        yield internal_err!("{} Error getting flight stream: {}", name, e);
                    }
                }
            }
            if !error {
                let mut combined = CombinedRecordBatchStream::new(schema.clone(),streams);

                while let Some(maybe_batch) = combined.next().await {
                    yield maybe_batch;
                }
            }

        };

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema.clone(),
            stream,
        )))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::processor_service::ServiceClients;
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::physical_plan::{displayable, empty::EmptyExec};
    use datafusion::prelude::{SessionConfig, SessionContext};
    use parking_lot::Mutex;
    use std::collections::HashMap;

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]))
    }

    fn reader(stage_id: usize) -> DFRayStageReaderExec {
        DFRayStageReaderExec::try_new(Partitioning::UnknownPartitioning(2), schema(), stage_id)
            .unwrap()
    }

    /// A reader stands in for a stage hosted elsewhere, so it reports that
    /// stage's partition count but never its partitioning *scheme*: the rows
    /// arrive over Flight already distributed.
    #[test]
    fn a_reader_keeps_the_partition_count_of_what_it_replaces() {
        let input = Arc::new(EmptyExec::new(schema())) as Arc<dyn ExecutionPlan>;
        let r = DFRayStageReaderExec::try_new_from_input(input, 4).unwrap();
        assert_eq!(r.stage_id, 4);
        assert_eq!(r.properties().output_partitioning().partition_count(), 1);
        assert!(matches!(
            r.properties().output_partitioning(),
            Partitioning::UnknownPartitioning(_)
        ));

        let r = reader(7);
        assert_eq!(r.name(), "RayStageReaderExec");
        assert!(r.children().is_empty());
        assert_eq!(r.schema(), schema());
        let shown = displayable(&r).one_line().to_string();
        assert!(shown.contains("RayStageReaderExec[7]"), "got: {shown}");
    }

    #[test]
    fn a_reader_owns_no_expressions() {
        let r = reader(0);
        let mut seen = 0;
        let out = r
            .apply_expressions(&mut |_e| {
                seen += 1;
                Ok(TreeNodeRecursion::Continue)
            })
            .unwrap();
        assert_eq!(out, TreeNodeRecursion::Continue);
        assert_eq!(seen, 0);
    }

    #[test]
    #[should_panic]
    #[allow(deprecated)]
    fn a_reader_cannot_take_children() {
        let _ = Arc::new(reader(0)).with_new_children(vec![]);
    }

    fn ctx_with(clients: Option<ServiceClients>) -> SessionContext {
        let mut config = SessionConfig::new();
        if let Some(c) = clients {
            config = config.with_extension(Arc::new(c));
        }
        SessionContext::new_with_config(config)
    }

    /// Without the clients extension the reader has no idea who to ask. That
    /// has to be an error at `execute`, not a stream that hangs.
    #[test]
    fn executing_without_the_clients_extension_is_an_error() {
        match reader(3).execute(0, ctx_with(None).task_ctx()) {
            Err(e) => assert!(e.to_string().contains("Flight Client not in context")),
            Ok(_) => panic!("expected an error with no ServiceClients extension"),
        }
    }

    #[test]
    fn executing_a_stage_nobody_serves_is_an_error() {
        // a client map that knows about some other stage entirely
        let mut map = HashMap::new();
        map.insert((99, 0), Mutex::new(vec![]));

        match reader(3).execute(0, ctx_with(Some(ServiceClients(map))).task_ctx()) {
            Err(e) => {
                assert!(e.to_string().contains("No flight clients found for"));
                assert!(e.to_string().contains("3:0"), "got: {e}");
            }
            Ok(_) => panic!("stage 3 has no clients; expected an error"),
        }
    }

    /// With clients registered but none of them reachable, the reader must
    /// still produce a stream and report the failure as an item.
    #[tokio::test]
    async fn an_empty_client_list_yields_an_empty_stream() {
        let mut map = HashMap::new();
        map.insert((3, 1), Mutex::new(vec![]));

        let stream = reader(3)
            .execute(1, ctx_with(Some(ServiceClients(map))).task_ctx())
            .unwrap();
        let batches: Vec<_> = stream.collect().await;
        assert!(batches.is_empty(), "no peers means no rows");
    }
}
