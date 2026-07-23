//! In-process OTLP collector stub for integration tests. Implements the
//! OTLP gRPC `MetricsService` and `TraceService` and records received
//! requests in a shared `Arc<Mutex<Vec<...>>>` for assertion.

#![cfg(feature = "opentelemetry")]

use std::sync::{Arc, Mutex};

use opentelemetry_proto::tonic::collector::metrics::v1::{
    ExportMetricsServiceRequest, ExportMetricsServiceResponse,
};
use opentelemetry_proto::tonic::collector::metrics::v1::metrics_service_server::{
    MetricsService, MetricsServiceServer,
};
use opentelemetry_proto::tonic::collector::trace::v1::{
    ExportTraceServiceRequest, ExportTraceServiceResponse,
};
use opentelemetry_proto::tonic::collector::trace::v1::trace_service_server::{
    TraceService, TraceServiceServer,
};
use tonic::{Request, Response, Status};

/// Shared record of all export requests received by the stub collector.
#[derive(Default, Clone)]
pub struct RecordedExport {
    /// Every `ExportMetricsServiceRequest` received.
    pub metrics: Arc<Mutex<Vec<ExportMetricsServiceRequest>>>,
    /// Every `ExportTraceServiceRequest` received.
    pub traces: Arc<Mutex<Vec<ExportTraceServiceRequest>>>,
}

impl RecordedExport {
    /// Number of metrics export requests received.
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned (should not happen in
    /// practice — the stub collector only holds the lock briefly).
    #[must_use]
    pub fn metrics_count(&self) -> usize {
        self.metrics.lock().unwrap().len()
    }

    /// Number of trace export requests received.
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned (should not happen in
    /// practice — the stub collector only holds the lock briefly).
    #[must_use]
    pub fn traces_count(&self) -> usize {
        self.traces.lock().unwrap().len()
    }
}

/// Stub OTLP `MetricsService` — records every request for assertion.
pub struct StubMetricsService {
    pub recorded: RecordedExport,
}

#[tonic::async_trait]
impl MetricsService for StubMetricsService {
    async fn export(
        &self,
        request: Request<ExportMetricsServiceRequest>,
    ) -> Result<Response<ExportMetricsServiceResponse>, Status> {
        self.recorded.metrics.lock().unwrap().push(request.into_inner());
        Ok(Response::new(ExportMetricsServiceResponse {
            partial_success: None,
        }))
    }
}

/// Stub OTLP `TraceService` — records every request for assertion.
pub struct StubTraceService {
    pub recorded: RecordedExport,
}

#[tonic::async_trait]
impl TraceService for StubTraceService {
    async fn export(
        &self,
        request: Request<ExportTraceServiceRequest>,
    ) -> Result<Response<ExportTraceServiceResponse>, Status> {
        self.recorded.traces.lock().unwrap().push(request.into_inner());
        Ok(Response::new(ExportTraceServiceResponse {
            partial_success: None,
        }))
    }
}

/// Start a stub OTLP collector on `127.0.0.1:0` (OS-assigned port).
///
/// Returns the [`RecordedExport`] handle (for assertions) and the listening
/// URL (e.g. `"http://127.0.0.1:41234"`). The gRPC server runs in a spawned
/// tokio task; it is dropped when the test ends.
///
/// # Panics
///
/// Panics if the TCP listener cannot be bound or the server fails to start.
#[must_use]
pub async fn spawn_stub_collector() -> (RecordedExport, String) {
    use tonic::transport::Server;

    let recorded = RecordedExport::default();
    let metrics = StubMetricsService {
        recorded: recorded.clone(),
    };
    let traces = StubTraceService {
        recorded: recorded.clone(),
    };

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind stub collector");
    let addr = listener.local_addr().expect("local_addr");
    let url = format!("http://{addr}");

    tokio::spawn(async move {
        Server::builder()
            .add_service(MetricsServiceServer::new(metrics))
            .add_service(TraceServiceServer::new(traces))
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(
                listener,
            ))
            .await
            .expect("stub collector server");
    });

    // Give the server a moment to start accepting connections.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    (recorded, url)
}
