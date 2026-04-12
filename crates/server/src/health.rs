// crates/server/src/health.rs
//! Health check endpoints

/// Health status returned by checks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    Ok,
    NotReady,
    Unhealthy,
}

impl HealthStatus {
    pub fn is_ok(&self) -> bool {
        matches!(self, HealthStatus::Ok)
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            HealthStatus::Ok => "ok",
            HealthStatus::NotReady => "not_ready",
            HealthStatus::Unhealthy => "unhealthy",
        }
    }

    pub fn http_status(&self) -> u16 {
        match self {
            HealthStatus::Ok => 200,
            HealthStatus::NotReady => 503,
            HealthStatus::Unhealthy => 503,
        }
    }
}

/// Health checker for liveness and readiness probes
pub struct HealthChecker {
    alive: bool,
    ready: bool,
}

impl HealthChecker {
    pub fn new(alive: bool, ready: bool) -> Self {
        Self { alive, ready }
    }

    /// Liveness probe - is the process running?
    pub fn check_liveness(&self) -> HealthStatus {
        if self.alive {
            HealthStatus::Ok
        } else {
            HealthStatus::Unhealthy
        }
    }

    /// Readiness probe - is the service ready to accept requests?
    pub fn check_readiness(&self) -> HealthStatus {
        if !self.alive {
            return HealthStatus::Unhealthy;
        }
        if self.ready {
            HealthStatus::Ok
        } else {
            HealthStatus::NotReady
        }
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
}
