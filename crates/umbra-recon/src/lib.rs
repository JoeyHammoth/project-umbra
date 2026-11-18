use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use umbra_core::LinearOperator;

/// One iteration's worth of feedback from a solver.
#[derive(Debug, Clone, PartialEq)]
pub struct StepReport {
    /// Zero-based iteration or subset sweep index.
    pub iteration: u32,
    /// Current value of the objective, if the solver has one.
    pub objective: Option<f64>,
    /// Whether the solver considers itself converged.
    pub converged: bool,
}

/// Shared control surface between the orchestrator and a running solver.
///
/// Cancellation is checked inside inner loops roughly every 5 ms of work, so a
/// superseded interactive request stops promptly without unwinding anything.
#[derive(Debug, Clone)]
pub struct Control {
    /// Hard iteration budget.
    pub max_iters: u32,
    /// How often to emit an estimate snapshot, in iterations.
    pub snapshot_every: u32,
    stop: Arc<AtomicBool>,
}

impl Control {
    /// A control block with the given budget that has not been cancelled.
    pub fn new(max_iters: u32, snapshot_every: u32) -> Self {
        Self {
            max_iters,
            snapshot_every,
            stop: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Ask the solver to stop at its next checkpoint.
    pub fn request_stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    /// Whether a stop has been requested.
    pub fn should_stop(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }
}

/// A reconstruction algorithm, addressed by a stable identifier.
///
/// The ids mirror the algorithm portfolio in the design spec so URLs, cache
/// keys and UI labels stay in step with the documentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverKind {
    /// A1: ordered-subset transmission EM (L0 only).
    TransmissionEm,
    /// A2: nonlinear Poisson MAP with smoothed TV.
    PoissonMap,
    /// A3: opacity-domain inversion followed by PDHG.
    OpacityPdhg,
    /// A4: reference-anchored one-sided sparse deviation via FISTA.
    SparseDeviation,
    /// A5: plug-and-play with a learned denoiser (stretch).
    PlugAndPlay,
}

impl SolverKind {
    /// Stable lowercase id used in URLs and cache keys.
    pub fn id(&self) -> &'static str {
        match self {
            SolverKind::TransmissionEm => "a1-transmission-em",
            SolverKind::PoissonMap => "a2-poisson-map",
            SolverKind::OpacityPdhg => "a3-opacity-pdhg",
            SolverKind::SparseDeviation => "a4-sparse-deviation",
            SolverKind::PlugAndPlay => "a5-plug-and-play",
        }
    }
}

/// An iterative reconstruction algorithm.
///
/// Solvers are driven one step at a time so the UI can show progressive results
/// and the orchestrator can cancel between steps. They hold no global state, so
/// many can run at once on separate workers.
pub trait Solver {
    /// Algorithm-specific configuration.
    type Config;

    /// Build a solver for the given operator and observed counts.
    fn init(op: &dyn LinearOperator, counts: &[f32], cfg: &Self::Config) -> Self;

    /// Advance one iteration (or one subset sweep).
    fn step(&mut self, ctl: &Control) -> StepReport;

    /// The current density estimate, one entry per voxel.
    fn estimate(&self) -> &[f32];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_starts_running_and_can_be_stopped() {
        let ctl = Control::new(100, 5);
        assert!(!ctl.should_stop());
        ctl.request_stop();
        assert!(ctl.should_stop());
    }

    #[test]
    fn solver_ids_are_stable() {
        assert_eq!(SolverKind::PoissonMap.id(), "a2-poisson-map");
        assert_eq!(SolverKind::SparseDeviation.id(), "a4-sparse-deviation");
    }
}
