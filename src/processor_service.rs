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

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::error::Error;
use std::sync::Arc;

use crate::pyerr::PyDataFusionResult;
use crate::pyerr::wait_for_future;
use arrow::array::RecordBatch;
use arrow_flight::FlightClient;
use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::error::FlightError;
use datafusion::common::internal_datafusion_err;
use datafusion::execution::SessionStateBuilder;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::{SessionConfig, SessionContext};
use futures::{Stream, TryStreamExt};
use local_ip_address::local_ip;
use log::{debug, error, info, trace};
use tokio::net::TcpListener;

use tonic::transport::Server;
use tonic::{Request, Response, Status, async_trait};

use datafusion::error::Result as DFResult;

use arrow_flight::{Ticket, flight_service_server::FlightServiceServer};

use pyo3::prelude::*;

use parking_lot::{Mutex, RwLock};

use tokio::sync::mpsc::{Receiver, Sender, channel};

use crate::flight::{FlightHandler, FlightServ};
use crate::isolator::PartitionGroup;
use crate::util::{
    ResultExt, apply_execution_settings, bytes_to_physical_plan,
    display_plan_with_partition_counts, extract_ticket, input_stage_ids, make_client,
    register_object_store_for_paths_in_plan,
};

/// a map of stage_id, partition to a list FlightClients that can serve
/// this (stage_id, and partition).   It is assumed that to consume a partition, the consumer
/// will consume the partition from all clients and merge the results.
pub(crate) struct ServiceClients(pub HashMap<(usize, usize), Mutex<Vec<FlightClient>>>);

/// DFRayProcessorHandler is a [`FlightHandler`] that serves streams of partitions from a hosted Physical Plan
/// It only responds to the DoGet Arrow Flight method.
struct DFRayProcessorHandler {
    /// our name, useful for logging
    name: String,
    /// Inner state of the handler
    inner: RwLock<Option<DFRayProcessorHandlerInner>>,
}

struct DFRayProcessorHandlerInner {
    /// the physical plan that comprises our stage
    pub(crate) plan: Arc<dyn ExecutionPlan>,
    /// the session context we will use to execute the plan
    pub(crate) ctx: SessionContext,
}

impl DFRayProcessorHandler {
    pub fn new(name: String) -> Self {
        let inner = RwLock::new(None);

        Self { name, inner }
    }
    async fn update_plan(
        &self,
        stage_id: usize,
        stage_addrs: HashMap<usize, HashMap<usize, Vec<String>>>,
        plan: Arc<dyn ExecutionPlan>,
        partition_group: Vec<usize>,
    ) -> DFResult<()> {
        let inner =
            DFRayProcessorHandlerInner::new(stage_id, stage_addrs, plan, partition_group).await?;
        self.inner.write().replace(inner);
        Ok(())
    }
}

impl DFRayProcessorHandlerInner {
    pub async fn new(
        stage_id: usize,
        stage_addrs: HashMap<usize, HashMap<usize, Vec<String>>>,
        plan: Arc<dyn ExecutionPlan>,
        partition_group: Vec<usize>,
    ) -> DFResult<Self> {
        let ctx = Self::configure_ctx(stage_id, stage_addrs, plan.clone(), partition_group).await?;

        Ok(Self { plan, ctx })
    }

    async fn configure_ctx(
        stage_id: usize,
        stage_addrs: HashMap<usize, HashMap<usize, Vec<String>>>,
        plan: Arc<dyn ExecutionPlan>,
        partition_group: Vec<usize>,
    ) -> DFResult<SessionContext> {
        let stage_ids_i_need = input_stage_ids(&plan)?;

        // map of stage_id, partition -> Vec<FlightClient>
        let mut client_map = HashMap::new();

        // a map of address -> FlightClient which we use while building the client map above
        // so that we don't create duplicate clients for the same address.
        let mut clients = HashMap::new();

        fn clone_flight_client(c: &FlightClient) -> FlightClient {
            let inner_clone = c.inner().clone();
            FlightClient::new_from_inner(inner_clone)
        }

        for stage_id in stage_ids_i_need {
            let partition_addrs = stage_addrs.get(&stage_id).ok_or(internal_datafusion_err!(
                "Cannot find stage addr {stage_id} in {:?}",
                stage_addrs
            ))?;

            for (partition, addrs) in partition_addrs {
                let mut flight_clients = vec![];
                for addr in addrs {
                    let client = match clients.entry(addr) {
                        Entry::Occupied(o) => clone_flight_client(o.get()),
                        Entry::Vacant(v) => {
                            let client = make_client(addr).await?;
                            let clone = clone_flight_client(&client);
                            v.insert(client);
                            clone
                        }
                    };
                    flight_clients.push(client);
                }
                client_map.insert((stage_id, *partition), Mutex::new(flight_clients));
            }
        }

        let mut config = SessionConfig::new().with_extension(Arc::new(ServiceClients(client_map)));

        // this only matters if the plan includes an PartitionIsolatorExec, which looks for this
        // for this extension and will be ignored otherwise
        config = config.with_extension(Arc::new(PartitionGroup(partition_group.clone())));

        apply_execution_settings(&mut config);

        let state = SessionStateBuilder::new()
            .with_default_features()
            .with_config(config)
            .build();
        let ctx = SessionContext::new_with_state(state);

        register_object_store_for_paths_in_plan(&ctx, plan.clone())?;

        trace!("ctx configured for stage {}", stage_id);

        Ok(ctx)
    }
}

fn make_stream(
    inner: &DFRayProcessorHandlerInner,
    partition: usize,
) -> Result<impl Stream<Item = Result<RecordBatch, FlightError>> + Send + 'static, Status> {
    let task_ctx = inner.ctx.task_ctx();

    let stream = inner
        .plan
        .execute(partition, task_ctx)
        .inspect_err(|e| error!("Could not get partition stream from plan {e}"))
        .map_err(|e| Status::internal(format!("Could not get partition stream from plan {e}")))?
        .map_err(|e| FlightError::from_external_error(Box::new(e)));

    Ok(stream)
}

#[async_trait]
impl FlightHandler for DFRayProcessorHandler {
    async fn get_stream(
        &self,
        request: Request<Ticket>,
    ) -> std::result::Result<Response<crate::flight::DoGetStream>, Status> {
        let remote_addr = request
            .remote_addr()
            .map(|a| a.to_string())
            .unwrap_or("unknown".to_string());

        let ticket = request.into_inner();

        let partition = extract_ticket(ticket).map_err(|e| {
            Status::internal(format!(
                "{}, Unexpected error extracting ticket {e}",
                self.name
            ))
        })?;

        trace!(
            "{}, request for partition {} from {}",
            self.name, partition, remote_addr
        );

        let name = self.name.clone();
        let stream = self
            .inner
            .read()
            .as_ref()
            .map(|inner| make_stream(inner, partition))
            .ok_or_else(|| Status::internal(format!("{} No inner found", &name)))??;

        let out_stream = FlightDataEncoderBuilder::new()
            .build(stream)
            .map_err(move |e| {
                Status::internal(format!("{} Unexpected error building stream {e}", name))
            });

        Ok(Response::new(Box::pin(out_stream)))
    }
}

/// DFRayProcessorService is a Arrow Flight service that serves streams of
/// partitions from a hosted Physical Plan
///
/// It only responds to the DoGet Arrow Flight method
#[pyclass]
pub struct DFRayProcessorService {
    name: String,
    listener: Option<TcpListener>,
    handler: Arc<DFRayProcessorHandler>,
    addr: Option<String>,
    all_done_tx: Arc<Mutex<Sender<()>>>,
    all_done_rx: Option<Receiver<()>>,
}

#[pymethods]
impl DFRayProcessorService {
    #[new]
    pub fn new(name: String) -> PyResult<Self> {
        let name = format!("[{}]", name);
        let listener = None;
        let addr = None;

        let (all_done_tx, all_done_rx) = channel(1);
        let all_done_tx = Arc::new(Mutex::new(all_done_tx));

        let handler = Arc::new(DFRayProcessorHandler::new(name.clone()));

        Ok(Self {
            name,
            listener,
            handler,
            addr,
            all_done_tx,
            all_done_rx: Some(all_done_rx),
        })
    }

    /// bind the listener to a socket.  This method must complete
    /// before any other methods are called.   This is separate
    /// from new() because Ray does not let you wait (AFAICT) on Actor inits to complete
    /// and we will want to wait on this with ray.get()
    pub fn start_up(&mut self, py: Python) -> PyResult<()> {
        let my_local_ip = local_ip().to_py_err()?;
        let my_host_str = format!("{my_local_ip}:0");

        self.listener = Some(wait_for_future(py, TcpListener::bind(&my_host_str))?.to_py_err()?);

        self.addr = Some(format!(
            "{}",
            self.listener.as_ref().unwrap().local_addr().unwrap()
        ));

        Ok(())
    }

    /// get the address of the listing socket for this service
    pub fn addr(&self) -> PyResult<String> {
        self.addr.clone().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                "{},Couldn't get addr",
                self.name
            ))
        })
    }

    /// signal to the service that we can shutdown
    ///
    /// returns a python coroutine that should be awaited
    pub fn all_done<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyAny>> {
        let sender = self.all_done_tx.lock().clone();

        let fut = async move {
            sender.send(()).await.to_py_err()?;
            Ok(())
        };
        pyo3_async_runtimes::tokio::future_into_py(py, fut)
    }

    /// replace the plan that this service was providing, we will do this when we want
    /// to reuse the DFRayProcessorService for a subsequent query
    ///
    /// returns a python coroutine that should be awaited
    pub fn update_plan<'a>(
        &self,
        py: Python<'a>,
        stage_id: usize,
        stage_addrs: HashMap<usize, HashMap<usize, Vec<String>>>,
        partition_group: Vec<usize>,
        plan_bytes: &[u8],
    ) -> PyDataFusionResult<Bound<'a, PyAny>> {
        let plan = bytes_to_physical_plan(&SessionContext::new(), plan_bytes)?;

        debug!(
            "{} Received New Plan: Stage:{} my addr: {}, partition_group {:?}, stage_addrs:\n{:?}\nplan:\n{}",
            self.name,
            stage_id,
            self.addr()?,
            partition_group,
            stage_addrs,
            display_plan_with_partition_counts(&plan)
        );

        let handler = self.handler.clone();
        let name = self.name.clone();
        let fut = async move {
            handler
                .update_plan(stage_id, stage_addrs, plan, partition_group.clone())
                .await
                .to_py_err()?;
            info!(
                "{} [stage: {} pg:{:?}] updated plan",
                name, stage_id, partition_group
            );
            Ok(())
        };

        Ok(pyo3_async_runtimes::tokio::future_into_py(py, fut)?)
    }

    /// start the service
    /// returns a python coroutine that should be awaited
    pub fn serve<'a>(&mut self, py: Python<'a>) -> PyResult<Bound<'a, PyAny>> {
        let mut all_done_rx = self.all_done_rx.take().unwrap();

        let signal = async move {
            all_done_rx
                .recv()
                .await
                .expect("problem receiving shutdown signal");
        };

        let service = FlightServ {
            handler: self.handler.clone(),
        };

        let svc = FlightServiceServer::new(service);

        let listener = self.listener.take().unwrap();
        let name = self.name.clone();

        let serv = async move {
            Server::builder()
                .add_service(svc)
                .serve_with_incoming_shutdown(
                    tokio_stream::wrappers::TcpListenerStream::new(listener),
                    signal,
                )
                .await
                .inspect_err(|e| error!("{}, ERROR serving {e}", name))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{e}")))?;
            Ok::<(), Box<dyn Error + Send + Sync>>(())
        };

        let fut = async move {
            serv.await.to_py_err()?;
            Ok(())
        };

        pyo3_async_runtimes::tokio::future_into_py(py, fut)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::flight::FlightServ;
    use crate::stage_reader::DFRayStageReaderExec;
    use arrow::array::{Int32Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::datasource::memory::MemorySourceConfig;
    use datafusion::physical_plan::Partitioning;
    use futures::StreamExt;
    use prost::Message as _;

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]))
    }

    /// A leaf plan: no `DFRayStageReaderExec`, so configuring a context for it
    /// needs no Flight clients and no peers.
    fn leaf_plan(partitions: usize) -> Arc<dyn ExecutionPlan> {
        let parts: Vec<Vec<RecordBatch>> = (0..partitions)
            .map(|i| {
                vec![
                    RecordBatch::try_new(
                        schema(),
                        vec![Arc::new(Int32Array::from(vec![i as i32]))],
                    )
                    .unwrap(),
                ]
            })
            .collect();
        MemorySourceConfig::try_new_exec(&parts, schema(), None).unwrap() as Arc<dyn ExecutionPlan>
    }

    async fn handler_for(plan: Arc<dyn ExecutionPlan>) -> DFRayProcessorHandler {
        let h = DFRayProcessorHandler::new("[test]".to_string());
        h.update_plan(0, HashMap::new(), plan, vec![0])
            .await
            .unwrap();
        h
    }

    fn ticket_for(partition: u64) -> Ticket {
        Ticket {
            ticket: crate::protobuf::FlightTicketData {
                dummy: false,
                partition,
            }
            .encode_to_vec()
            .into(),
        }
    }

    #[tokio::test]
    async fn a_leaf_stage_needs_no_peers() {
        let ctx =
            DFRayProcessorHandlerInner::configure_ctx(0, HashMap::new(), leaf_plan(1), vec![0])
                .await
                .unwrap();

        // the extensions the plan nodes look for are installed
        assert!(
            ctx.state()
                .config()
                .get_extension::<ServiceClients>()
                .is_some()
        );
        assert!(
            ctx.state()
                .config()
                .get_extension::<PartitionGroup>()
                .is_some()
        );
        // and the settings that keep partition isolation correct are applied
        assert!(
            !ctx.state()
                .config()
                .options()
                .execution
                .enable_file_stream_work_stealing
        );
    }

    /// A stage that reads another one cannot be configured without an address
    /// for it; that must be a clear error rather than a later hang.
    #[tokio::test]
    async fn a_missing_peer_address_is_reported() {
        let reader = Arc::new(
            DFRayStageReaderExec::try_new(Partitioning::UnknownPartitioning(1), schema(), 9)
                .unwrap(),
        ) as Arc<dyn ExecutionPlan>;

        match DFRayProcessorHandlerInner::configure_ctx(0, HashMap::new(), reader, vec![0]).await {
            Err(e) => assert!(e.to_string().contains("Cannot find stage addr"), "got: {e}"),
            Ok(_) => panic!("stage 9 has no address; expected an error"),
        }
    }

    #[tokio::test]
    async fn serving_a_partition_streams_its_rows() {
        let handler = handler_for(leaf_plan(2)).await;
        let resp = handler
            .get_stream(Request::new(ticket_for(1)))
            .await
            .unwrap();

        // the response is Flight-encoded, so just assert it carries data
        let frames: Vec<_> = resp.into_inner().collect().await;
        assert!(!frames.is_empty(), "expected at least a schema frame");
        assert!(frames.iter().all(|f| f.is_ok()));
    }

    /// `make_stream` has to turn a plan-level failure into a Flight status.
    /// The isolator is used rather than the memory source because the latter
    /// panics on an out-of-range partition instead of returning an error.
    #[tokio::test]
    async fn a_partition_that_does_not_exist_is_an_error() {
        let isolated = Arc::new(crate::isolator::PartitionIsolatorExec::new(leaf_plan(1), 1))
            as Arc<dyn ExecutionPlan>;
        let handler = handler_for(isolated).await;
        let status = match handler.get_stream(Request::new(ticket_for(99))).await {
            Err(s) => s,
            Ok(_) => panic!("partition 99 is out of range; expected an error"),
        };
        assert_eq!(status.code(), tonic::Code::Internal);
        assert!(status.message().contains("partition stream"), "{status:?}");
    }

    #[tokio::test]
    async fn a_malformed_ticket_is_rejected() {
        let handler = handler_for(leaf_plan(1)).await;
        let bad = Request::new(Ticket {
            ticket: vec![0xff, 0xff, 0xff].into(),
        });
        let status = match handler.get_stream(bad).await {
            Err(s) => s,
            Ok(_) => panic!("garbage ticket; expected an error"),
        };
        assert_eq!(status.code(), tonic::Code::Internal);
    }

    /// Processors are pooled and reused across queries, so a handler is asked
    /// for a stream before it has ever been given a plan.
    #[tokio::test]
    async fn a_handler_without_a_plan_refuses() {
        let handler = DFRayProcessorHandler::new("[fresh]".to_string());
        let status = match handler.get_stream(Request::new(ticket_for(0))).await {
            Err(s) => s,
            Ok(_) => panic!("handler has no plan; expected an error"),
        };
        assert!(status.message().contains("No inner found"), "{status:?}");
    }

    #[tokio::test]
    async fn updating_the_plan_replaces_what_is_served() {
        let handler = handler_for(leaf_plan(1)).await;
        assert!(
            handler
                .get_stream(Request::new(ticket_for(0)))
                .await
                .is_ok()
        );

        // a second query reuses the actor with a wider plan
        handler
            .update_plan(1, HashMap::new(), leaf_plan(3), vec![0, 1, 2])
            .await
            .unwrap();
        assert!(
            handler
                .get_stream(Request::new(ticket_for(2)))
                .await
                .is_ok()
        );
    }

    #[test]
    fn the_service_has_no_address_until_it_starts_up() {
        let svc = DFRayProcessorService::new("[svc]".to_string()).unwrap();
        match svc.addr() {
            Err(e) => assert!(e.to_string().contains("Couldn't get addr"), "{e:?}"),
            Ok(a) => panic!("not bound yet, but got {a}"),
        }
    }

    /// The whole path a peer actually takes: a tonic server over a real socket,
    /// reached by a real `FlightClient`.
    #[tokio::test]
    async fn a_stage_is_reachable_over_flight() {
        let handler = Arc::new(handler_for(leaf_plan(2)).await);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        let server = tokio::spawn(async move {
            Server::builder()
                .add_service(FlightServiceServer::new(FlightServ { handler }))
                .serve_with_incoming_shutdown(
                    tokio_stream::wrappers::TcpListenerStream::new(listener),
                    async {
                        let _ = shutdown_rx.await;
                    },
                )
                .await
        });

        let mut client = crate::util::make_client(&addr.to_string()).await.unwrap();
        let stream = client.do_get(ticket_for(0)).await.unwrap();
        let batches: Vec<_> = stream.collect().await;
        assert!(!batches.is_empty(), "no batches came back over Flight");
        assert!(batches.iter().all(|b| b.is_ok()));

        let _ = shutdown_tx.send(());
        server.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn connecting_to_a_bad_address_fails_cleanly() {
        assert!(crate::util::make_client("not a host:!!").await.is_err());
    }
}
