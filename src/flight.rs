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

use std::sync::Arc;

use futures::stream::BoxStream;
use tonic::{Request, Response, Status, Streaming};

use arrow_flight::{
    Action, ActionType, Criteria, Empty, FlightData, FlightDescriptor, FlightInfo,
    HandshakeRequest, HandshakeResponse, PollInfo, PutResult, SchemaResult, Ticket,
    flight_service_server::FlightService,
};

pub type DoGetStream = BoxStream<'static, Result<FlightData, Status>>;

#[tonic::async_trait]
pub trait FlightHandler: Send + Sync {
    async fn get_stream(&self, request: Request<Ticket>) -> Result<Response<DoGetStream>, Status>;
}

pub struct FlightServ {
    pub handler: Arc<dyn FlightHandler>,
}

#[tonic::async_trait]
impl FlightService for FlightServ {
    type HandshakeStream = BoxStream<'static, Result<HandshakeResponse, Status>>;
    type ListFlightsStream = BoxStream<'static, Result<FlightInfo, Status>>;
    type DoGetStream = BoxStream<'static, Result<FlightData, Status>>;
    type DoPutStream = BoxStream<'static, Result<PutResult, Status>>;
    type DoActionStream = BoxStream<'static, Result<arrow_flight::Result, Status>>;
    type ListActionsStream = BoxStream<'static, Result<ActionType, Status>>;
    type DoExchangeStream = BoxStream<'static, Result<FlightData, Status>>;

    async fn do_get(
        &self,
        request: Request<Ticket>,
    ) -> Result<Response<Self::DoGetStream>, Status> {
        self.handler.get_stream(request).await
    }

    async fn do_put(
        &self,
        _request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoPutStream>, Status> {
        Err(Status::unimplemented("Unimplemented: do put"))
    }

    async fn handshake(
        &self,
        _request: Request<Streaming<HandshakeRequest>>,
    ) -> Result<Response<Self::HandshakeStream>, Status> {
        Err(Status::unimplemented("Unimplemented: handshake"))
    }

    async fn list_flights(
        &self,
        _request: Request<Criteria>,
    ) -> Result<Response<Self::ListFlightsStream>, Status> {
        Err(Status::unimplemented("Unimplemented: list_flights"))
    }

    async fn get_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        Err(Status::unimplemented("Unimplemented: get_flight_info"))
    }

    async fn poll_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<PollInfo>, Status> {
        Err(Status::unimplemented("Unimplemented: poll_flight_info"))
    }

    async fn get_schema(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<SchemaResult>, Status> {
        Err(Status::unimplemented("Unimplemented: get_schema"))
    }

    async fn do_action(
        &self,
        _request: Request<Action>,
    ) -> Result<Response<Self::DoActionStream>, Status> {
        Err(Status::unimplemented("Unimplemented: do action"))
    }

    async fn list_actions(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<Self::ListActionsStream>, Status> {
        Err(Status::unimplemented("Unimplemented: list_actions"))
    }

    async fn do_exchange(
        &self,
        _request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoExchangeStream>, Status> {
        Err(Status::unimplemented("Unimplemented: do_exchange"))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use arrow_flight::Criteria;
    use futures::StreamExt;

    struct Echo;

    #[tonic::async_trait]
    impl FlightHandler for Echo {
        async fn get_stream(
            &self,
            request: Request<Ticket>,
        ) -> Result<Response<DoGetStream>, Status> {
            let bytes = request.into_inner().ticket;
            let out = futures::stream::once(async move {
                Ok(FlightData {
                    data_header: bytes,
                    ..Default::default()
                })
            });
            Ok(Response::new(Box::pin(out) as DoGetStream))
        }
    }

    fn serv() -> FlightServ {
        FlightServ {
            handler: Arc::new(Echo),
        }
    }

    #[tokio::test]
    async fn do_get_delegates_to_the_handler() {
        let resp = serv()
            .do_get(Request::new(Ticket {
                ticket: vec![1, 2, 3].into(),
            }))
            .await
            .unwrap();
        let got = resp.into_inner().next().await.unwrap().unwrap();
        assert_eq!(got.data_header.as_ref(), &[1, 2, 3]);
    }

    /// Only `do_get` is part of the stage protocol. The rest of the Flight
    /// surface must refuse clearly rather than half-answer.
    ///
    /// `handshake`, `do_put` and `do_exchange` are omitted: they take
    /// `tonic::Streaming`, which cannot be built outside a real connection.
    #[tokio::test]
    async fn every_other_flight_method_is_refused() {
        let s = serv();

        macro_rules! refused {
            ($call:expr, $name:literal) => {
                match $call.await {
                    Err(status) => assert_eq!(
                        status.code(),
                        tonic::Code::Unimplemented,
                        concat!($name, " should be unimplemented")
                    ),
                    Ok(_) => panic!(concat!($name, " unexpectedly succeeded")),
                }
            };
        }

        refused!(
            s.list_flights(Request::new(Criteria::default())),
            "list_flights"
        );
        refused!(
            s.get_flight_info(Request::new(FlightDescriptor::default())),
            "get_flight_info"
        );
        refused!(
            s.poll_flight_info(Request::new(FlightDescriptor::default())),
            "poll_flight_info"
        );
        refused!(
            s.get_schema(Request::new(FlightDescriptor::default())),
            "get_schema"
        );
        refused!(s.do_action(Request::new(Action::default())), "do_action");
        refused!(
            s.list_actions(Request::new(Empty::default())),
            "list_actions"
        );
    }
}
