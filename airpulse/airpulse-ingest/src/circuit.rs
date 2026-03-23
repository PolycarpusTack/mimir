use airpulse_types::CircuitState;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use uuid::Uuid;

/// Per-source circuit breaker state.
struct SourceCircuit {
    state: CircuitState,
    consecutive_failures: u32,
    last_state_change: Instant,
}

/// Circuit breaker managing per-source failure tracking.
pub struct CircuitBreaker {
    failure_threshold: u32,
    open_duration: Duration,
    circuits: Mutex<HashMap<Uuid, SourceCircuit>>,
}

impl CircuitBreaker {
    pub fn new(failure_threshold: u32, open_duration_secs: u64) -> Self {
        Self {
            failure_threshold,
            open_duration: Duration::from_secs(open_duration_secs),
            circuits: Mutex::new(HashMap::new()),
        }
    }

    /// Get the current circuit state for a source.
    pub fn state(&self, source_id: Uuid) -> CircuitState {
        let circuits = self.circuits.lock().unwrap();
        circuits
            .get(&source_id)
            .map(|c| c.state)
            .unwrap_or(CircuitState::Closed)
    }

    /// Record a successful poll.
    pub fn record_success(&self, source_id: Uuid) {
        let mut circuits = self.circuits.lock().unwrap();
        let circuit = circuits.entry(source_id).or_insert(SourceCircuit {
            state: CircuitState::Closed,
            consecutive_failures: 0,
            last_state_change: Instant::now(),
        });

        circuit.state = CircuitState::Closed;
        circuit.consecutive_failures = 0;
        circuit.last_state_change = Instant::now();
    }

    /// Record a failed poll.
    pub fn record_failure(&self, source_id: Uuid) {
        let mut circuits = self.circuits.lock().unwrap();
        let circuit = circuits.entry(source_id).or_insert(SourceCircuit {
            state: CircuitState::Closed,
            consecutive_failures: 0,
            last_state_change: Instant::now(),
        });

        circuit.consecutive_failures += 1;

        if circuit.consecutive_failures >= self.failure_threshold {
            circuit.state = CircuitState::Open;
            circuit.last_state_change = Instant::now();
        }
    }

    /// Check if a poll should be attempted for this source.
    pub fn should_attempt(&self, source_id: Uuid) -> bool {
        let mut circuits = self.circuits.lock().unwrap();
        let circuit = match circuits.get_mut(&source_id) {
            Some(c) => c,
            None => return true, // No state = Closed
        };

        match circuit.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                if circuit.last_state_change.elapsed() >= self.open_duration {
                    circuit.state = CircuitState::HalfOpen;
                    circuit.last_state_change = Instant::now();
                    true // Allow one probe
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => {
                // Only one probe allowed in HalfOpen; block additional attempts
                false
            }
        }
    }

    /// Get the consecutive failure count for a source.
    pub fn consecutive_failures(&self, source_id: Uuid) -> u32 {
        let circuits = self.circuits.lock().unwrap();
        circuits
            .get(&source_id)
            .map(|c| c.consecutive_failures)
            .unwrap_or(0)
    }

    /// Get counts of sources in each state.
    pub fn state_counts(&self) -> (u32, u32, u32) {
        let circuits = self.circuits.lock().unwrap();
        let mut closed = 0u32;
        let mut open = 0u32;
        let mut half_open = 0u32;

        for circuit in circuits.values() {
            match circuit.state {
                CircuitState::Closed => closed += 1,
                CircuitState::Open => open += 1,
                CircuitState::HalfOpen => half_open += 1,
            }
        }

        (closed, open, half_open)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tc_cb_001_opens_after_threshold() {
        let cb = CircuitBreaker::new(5, 300);
        let source = Uuid::new_v4();

        // 4 failures: still closed
        for _ in 0..4 {
            cb.record_failure(source);
        }
        assert_eq!(cb.state(source), CircuitState::Closed);
        assert!(cb.should_attempt(source));

        // 5th failure: opens
        cb.record_failure(source);
        assert_eq!(cb.state(source), CircuitState::Open);
        assert!(!cb.should_attempt(source));
    }

    #[test]
    fn tc_cb_003_half_open_success_closes() {
        let cb = CircuitBreaker::new(5, 0); // 0s open duration for instant transition
        let source = Uuid::new_v4();

        // Open the circuit
        for _ in 0..5 {
            cb.record_failure(source);
        }
        assert_eq!(cb.state(source), CircuitState::Open);

        // Should transition to HalfOpen immediately (0s duration)
        assert!(cb.should_attempt(source));
        assert_eq!(cb.state(source), CircuitState::HalfOpen);

        // Success closes it
        cb.record_success(source);
        assert_eq!(cb.state(source), CircuitState::Closed);
        assert_eq!(cb.consecutive_failures(source), 0);
    }

    #[test]
    fn tc_cb_004_half_open_failure_reopens() {
        let cb = CircuitBreaker::new(5, 0);
        let source = Uuid::new_v4();

        for _ in 0..5 {
            cb.record_failure(source);
        }

        // Transition to HalfOpen
        cb.should_attempt(source);

        // Failure reopens
        cb.record_failure(source);
        assert_eq!(cb.state(source), CircuitState::Open);
    }

    #[test]
    fn tc_cb_005_open_not_elapsed() {
        let cb = CircuitBreaker::new(5, 300); // 300s
        let source = Uuid::new_v4();

        for _ in 0..5 {
            cb.record_failure(source);
        }

        // Duration hasn't elapsed
        assert!(!cb.should_attempt(source));
    }

    #[test]
    fn test_success_resets_failures() {
        let cb = CircuitBreaker::new(5, 300);
        let source = Uuid::new_v4();

        cb.record_failure(source);
        cb.record_failure(source);
        assert_eq!(cb.consecutive_failures(source), 2);

        cb.record_success(source);
        assert_eq!(cb.consecutive_failures(source), 0);
        assert_eq!(cb.state(source), CircuitState::Closed);
    }

    #[test]
    fn test_unknown_source_defaults_closed() {
        let cb = CircuitBreaker::new(5, 300);
        let source = Uuid::new_v4();
        assert_eq!(cb.state(source), CircuitState::Closed);
        assert!(cb.should_attempt(source));
    }

    #[test]
    fn test_state_counts() {
        let cb = CircuitBreaker::new(2, 300);
        let s1 = Uuid::new_v4();
        let s2 = Uuid::new_v4();
        let s3 = Uuid::new_v4();

        cb.record_success(s1); // Closed
        cb.record_failure(s2);
        cb.record_failure(s2); // Open
        cb.record_success(s3); // Closed

        let (closed, open, _half_open) = cb.state_counts();
        assert_eq!(closed, 2);
        assert_eq!(open, 1);
    }
}
