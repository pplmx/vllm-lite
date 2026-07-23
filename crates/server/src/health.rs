#![allow(clippy::module_name_repetitions)]
//! Health check endpoints

/// Health status returned by checks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    Ok,
    NotReady,
    Unhealthy,
}

impl HealthStatus {
    /// Returns `true` when the status is `Ok`.
    #[must_use]
    pub const fn is_ok(&self) -> bool {
        matches!(self, Self::Ok)
    }

    /// Return the kebab-case string representation (`"ok"`, `"not_ready"`, `"unhealthy"`).
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::NotReady => "not_ready",
            Self::Unhealthy => "unhealthy",
        }
    }

    /// Map a [`HealthStatus`] to its HTTP status code: `200` for `Ok`, `503`
    /// for both `NotReady` and `Unhealthy` (the orchestrator should treat 503
    /// as "drain this pod").
    #[must_use]
    pub const fn http_status(&self) -> u16 {
        match self {
            Self::Ok => 200,
            Self::NotReady | Self::Unhealthy => 503,
        }
    }
}

#[derive(Debug)]
/// Health checker for liveness and readiness probes
pub struct HealthChecker {
    alive: bool,
    ready: bool,
}

impl HealthChecker {
    /// Create a health checker with the given liveness and readiness flags.
    #[must_use]
    pub const fn new(alive: bool, ready: bool) -> Self {
        Self { alive, ready }
    }

    /// Liveness probe - is the process running?
    #[must_use]
    pub const fn check_liveness(&self) -> HealthStatus {
        if self.alive {
            HealthStatus::Ok
        } else {
            HealthStatus::Unhealthy
        }
    }

    /// Readiness probe - is the service ready to accept requests?
    #[must_use]
    pub const fn check_readiness(&self) -> HealthStatus {
        if !self.alive {
            return HealthStatus::Unhealthy;
        }
        if self.ready {
            HealthStatus::Ok
        } else {
            HealthStatus::NotReady
        }
    }

    /// Flip the readiness flag to `false`. Used by the graceful-shutdown
    /// path (`SIGTERM`, `/shutdown`) so the orchestrator's next readiness
    /// probe flips to `NotReady` and stops routing new traffic to this
    /// pod before the HTTP listener actually closes.
    ///
    /// Production-readiness §7: "SIGTERM/admin request → readiness=false →
    /// stop accepting new inference → cancel or drain queued requests →
    /// wait in-flight with deadline → flush metrics/logs → shutdown
    /// engine and join thread → exit". This method implements the first
    /// step; the caller (the shutdown coordinator) is responsible for
    /// the subsequent steps.
    ///
    /// Idempotent: calling on an already-not-ready checker is a no-op so
    /// the SIGTERM handler and the `/shutdown` handler can both invoke
    /// it without coordination.
    pub const fn mark_not_ready(&mut self) {
        self.ready = false;
    }
}

impl Default for HealthChecker {
    fn default() -> Self {
        Self::new(true, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_status_ok() {
        let checker = HealthChecker::new(true, true);
        assert_eq!(checker.check_liveness(), HealthStatus::Ok);
        assert_eq!(checker.check_readiness(), HealthStatus::Ok);
    }

    #[test]
    fn test_health_status_not_ready() {
        let checker = HealthChecker::new(true, false);
        assert_eq!(checker.check_liveness(), HealthStatus::Ok);
        assert_eq!(checker.check_readiness(), HealthStatus::NotReady);
    }

    #[test]
    fn test_health_status_unhealthy() {
        let checker = HealthChecker::new(false, false);
        assert_eq!(checker.check_liveness(), HealthStatus::Unhealthy);
        assert_eq!(checker.check_readiness(), HealthStatus::Unhealthy);
    }

    #[test]
    fn test_health_status_as_str() {
        assert_eq!(HealthStatus::Ok.as_str(), "ok");
        assert_eq!(HealthStatus::NotReady.as_str(), "not_ready");
        assert_eq!(HealthStatus::Unhealthy.as_str(), "unhealthy");
    }

    #[test]
    fn mark_not_ready_flips_a_ready_checker() {
        let mut checker = HealthChecker::new(true, true);
        assert_eq!(checker.check_readiness(), HealthStatus::Ok);
        checker.mark_not_ready();
        assert_eq!(checker.check_readiness(), HealthStatus::NotReady);
        // Liveness is unaffected — only readiness drops.
        assert_eq!(checker.check_liveness(), HealthStatus::Ok);
    }

    #[test]
    fn mark_not_ready_is_idempotent() {
        // Calling mark_not_ready on an already-not-ready checker is a
        // no-op; the SIGTERM handler and the /shutdown handler can both
        // invoke it without coordination.
        let mut checker = HealthChecker::new(true, false);
        checker.mark_not_ready();
        assert_eq!(checker.check_readiness(), HealthStatus::NotReady);
        checker.mark_not_ready();
        assert_eq!(checker.check_readiness(), HealthStatus::NotReady);
    }
}
