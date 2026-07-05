//! Small, focused utilities for the server crate.
//!
//! Anything reusable across multiple call sites lives here. Modules:
//!
//! - [`time`] — panic-free wall-clock accessors (unix epoch seconds/millis).

pub mod time;
