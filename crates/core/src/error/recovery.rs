// crates/core/src/error/recovery.rs
//! Error recovery management

use crate::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
use std::time::Duration;

/// Severity level for errors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    Warning,        // Log and continue
    Retryable,      // Attempt retry with backoff
    Degradable,     // Switch to fallback mode
    CircuitBreaker, // Trip circuit breaker
    Fatal,          // Log and terminate
}

impl ErrorSeverity {
    pub fn from_error(error: &str) -> Self {
        if error.contains("timeout") {
            ErrorSeverity::Retryable
        } else if error.contains("graph") {
            ErrorSeverity::Degradable
        } else if error.contains("crash") {
            ErrorSeverity::CircuitBreaker
        } else if error.contains("oom") || error.contains("memory") {
            ErrorSeverity::Fatal
        } else {
            ErrorSeverity::Warning
        }
    }
}

/// Recovery action to take
#[derive(Debug, Clone)]
pub enum RecoveryAction {
    Retry { max_attempts: usize },
    Degrade { component: String },
    OpenCircuit { component: String },
    Propagate,
    Terminate,
}

/// Manager for error recovery
pub struct RecoveryManager {
    circuit_breakers: dashmap::DashMap<String, CircuitBreaker>,
    config: RecoveryConfig,
}

#[derive(Debug, Clone)]
pub struct RecoveryConfig {
    pub retry_attempts: usize,
    pub retry_base_delay: Duration,
    pub default_circuit_breaker: CircuitBreakerConfig,
}

impl Default for RecoveryConfig {
    fn default() -> Self {
        Self {
            retry_attempts: 3,
            retry_base_delay: Duration::from_millis(100),
            default_circuit_breaker: CircuitBreakerConfig::default(),
        }
    }
}

impl RecoveryManager {
    pub fn new(config: RecoveryConfig) -> Self {
        Self {
            circuit_breakers: dashmap::DashMap::new(),
            config,
        }
    }

    pub fn get_or_create_circuit_breaker(&self, name: &str) -> CircuitBreaker {
        self.circuit_breakers
            .entry(name.to_string())
            .or_insert_with(|| CircuitBreaker::new(self.config.default_circuit_breaker.clone()))
            .clone()
    }

    pub fn determine_action(&self, severity: ErrorSeverity, component: &str) -> RecoveryAction {
        match severity {
            ErrorSeverity::Warning => RecoveryAction::Propagate,
            ErrorSeverity::Retryable => RecoveryAction::Retry {
                max_attempts: self.config.retry_attempts,
            },
            ErrorSeverity::Degradable => RecoveryAction::Degrade {
                component: component.to_string(),
            },
            ErrorSeverity::CircuitBreaker => RecoveryAction::OpenCircuit {
                component: component.to_string(),
            },
            ErrorSeverity::Fatal => RecoveryAction::Terminate,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit_breaker::CircuitState;

    #[tokio::test]
    async fn test_recovery_manager_creates_circuit_breaker() {
        let manager = RecoveryManager::new(RecoveryConfig::default());
        let cb = manager.get_or_create_circuit_breaker("test");
        let state = cb.state().await;
        assert!(matches!(state, CircuitState::Closed));
    }

    #[test]
    fn test_determine_action_retryable() {
        let manager = RecoveryManager::new(RecoveryConfig::default());
        let action = manager.determine_action(ErrorSeverity::Retryable, "model");
        assert!(matches!(action, RecoveryAction::Retry { .. }));
    }

    #[test]
    fn test_determine_action_degrade() {
        let manager = RecoveryManager::new(RecoveryConfig::default());
        let action = manager.determine_action(ErrorSeverity::Degradable, "cuda_graph");
        assert!(matches!(action, RecoveryAction::Degrade { .. }));
    }

    #[test]
    fn test_error_severity_from_error() {
        assert_eq!(
            ErrorSeverity::from_error("timeout"),
            ErrorSeverity::Retryable
        );
        assert_eq!(
            ErrorSeverity::from_error("graph failure"),
            ErrorSeverity::Degradable
        );
        assert_eq!(
            ErrorSeverity::from_error("crash"),
            ErrorSeverity::CircuitBreaker
        );
        assert_eq!(ErrorSeverity::from_error("oom"), ErrorSeverity::Fatal);
    }
}
