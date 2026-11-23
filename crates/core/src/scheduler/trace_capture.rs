//! Test-only capture of the tracing events emitted on the current thread.
//!
//! Several production guards exist only to decide whether a `tracing::trace!`
//! or `tracing::debug!` line fires. `cargo mutants` cannot kill them through
//! return values (there are none), so the tests install a minimal
//! [`Subscriber`] with [`tracing::subscriber::with_default`] and assert on the
//! emitted events instead.
//!
//! Every level is accepted: the capture is about WHETHER a line fires, not
//! about the configured filter.

use std::sync::{Arc, Mutex, PoisonError};
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Metadata, Subscriber};

/// Messages emitted while a capture was installed.
#[derive(Clone, Debug, Default)]
pub struct TraceCapture {
    messages: Arc<Mutex<Vec<String>>>,
}

impl TraceCapture {
    /// Every captured `message` field, in emission order.
    #[must_use]
    pub fn messages(&self) -> Vec<String> {
        // invariant: the capturing closure has returned by the time a caller
        // reads the messages, and a poisoned lock still holds a usable vector.
        self.messages
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Whether any captured message contains `needle`.
    #[must_use]
    pub fn contains(&self, needle: &str) -> bool {
        self.messages().iter().any(|m| m.contains(needle))
    }
}

struct MessageVisitor<'a>(&'a mut Option<String>);

impl Visit for MessageVisitor<'_> {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            *self.0 = Some(format!("{value:?}"));
        }
    }
}

impl Subscriber for TraceCapture {
    fn enabled(&self, _: &Metadata<'_>) -> bool {
        true
    }

    fn new_span(&self, _: &Attributes<'_>) -> Id {
        Id::from_u64(1)
    }

    fn record(&self, _: &Id, _: &Record<'_>) {}

    fn record_follows_from(&self, _: &Id, _: &Id) {}

    fn event(&self, event: &Event<'_>) {
        let mut message = None;
        event.record(&mut MessageVisitor(&mut message));
        if let Some(message) = message {
            self.messages
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(message);
        }
    }

    fn enter(&self, _: &Id) {}

    fn exit(&self, _: &Id) {}
}

/// Run `f` with a capture installed on this thread and return what it emitted.
pub fn capture<F: FnOnce()>(f: F) -> TraceCapture {
    let capture = TraceCapture::default();
    tracing::subscriber::with_default(capture.clone(), f);
    capture
}
