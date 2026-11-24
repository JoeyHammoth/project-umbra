//! Reconstruction solvers (A1–A5).
//!
//! | Module | Algorithm |
//! |---|---|
//! | [`a1`] | ordered-subset transmission EM, L0 only |
//! | [`a2`] | nonlinear Poisson MAP with Huber TV, projected L-BFGS |
//! | [`a3`] | opacity-domain inversion followed by Chambolle–Pock PDHG |
//! | [`a4`] | reference-anchored one-sided sparse deviation, FISTA |
//!
//! Solvers are driven one step at a time so a caller can show progressive
//! results and cancel between steps; see [`Solver`].

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use umbra_forward::expected::ForwardParams;
use umbra_forward::scene::SceneSpec;
use umbra_forward::{expected, Expected, System};
use umbra_physics::{Fidelity, FluxTable};

pub mod a1;
pub mod a2;
pub mod a3;
pub mod a4;
pub mod uncertainty;

pub use a1::{A1Config, TransmissionEm};
pub use a2::{A2Config, PoissonMap};
pub use a3::{A3Config, OpacityPdhg};
pub use a4::{prox, A4Config, DeviationSign, SparseDeviation};
pub use uncertainty::{tv_curvature, ResultCard};

/// One observation: the discretised experiment, the physics that produced the
/// counts, and the counts themselves.
///
/// Every solver needs all of it. A solver given only the operator would have to
/// guess the flux table, the exposure and the fidelity — three ways to return a
/// plausible answer for data it never described, none of which a convergence
/// test would catch.
#[derive(Debug, Clone, Copy)]
pub struct Problem<'a> {
    /// The path-length operator and its ray metadata.
    pub system: &'a System,
    /// Observed counts, one entry per ray cell.
    pub counts: &'a [f32],
    /// The flux table the counts were generated with.
    pub flux: &'a FluxTable,
    /// Which physics the counts obey.
    pub fidelity: Fidelity,
    /// Exposure, in seconds.
    pub exposure_s: f32,
    /// CSDA and L0 parameters.
    pub params: ForwardParams,
}

impl<'a> Problem<'a> {
    /// Bundles an observation.
    ///
    /// # Panics
    ///
    /// Panics if `counts` does not have one entry per ray cell: a solver cannot
    /// do anything sensible with a dataset that does not match its operator.
    pub fn new(
        system: &'a System,
        counts: &'a [f32],
        flux: &'a FluxTable,
        fidelity: Fidelity,
        exposure_s: f32,
        params: ForwardParams,
    ) -> Self {
        assert_eq!(
            counts.len(),
            system.ray_count(),
            "counts must have one entry per ray cell: {} vs {}",
            counts.len(),
            system.ray_count()
        );
        Self {
            system,
            counts,
            flux,
            fidelity,
            exposure_s,
            params,
        }
    }

    /// The problem a scene describes, from counts already sampled for it.
    pub fn from_scene(
        system: &'a System,
        counts: &'a [f32],
        scene: &SceneSpec,
        flux: &'a FluxTable,
    ) -> Self {
        Self::new(
            system,
            counts,
            flux,
            scene.physics.level,
            scene.exposure_s,
            scene.forward_params(),
        )
    }

    /// `λ` and `∂λ/∂X` at a density field, under this problem's physics.
    pub fn evaluate(&self, rho: &[f32]) -> Expected {
        expected(
            self.system,
            rho,
            self.flux,
            self.fidelity,
            self.exposure_s,
            &self.params,
        )
    }

    /// Poisson NLL of a model vector, up to the constant `Σ ln y_i!`.
    pub fn nll_of(&self, lambda: &[f32]) -> f64 {
        poisson_nll(lambda, self.counts)
    }

    /// Poisson NLL of a density field under this problem's physics.
    pub fn nll(&self, rho: &[f32]) -> f64 {
        self.nll_of(&self.evaluate(rho).lambda)
    }

    /// Number of ray cells.
    pub fn ray_count(&self) -> usize {
        self.system.ray_count()
    }

    /// Number of voxels.
    pub fn voxel_count(&self) -> usize {
        self.system.voxel_count()
    }
}

/// Poisson negative log-likelihood of `λ` given `y`, up to `Σ ln y!`.
///
/// A ray with no acceptance carries no information: zero counts against it
/// contribute nothing, and counts against it are impossible data, which the
/// infinite value states rather than hides.
pub(crate) fn poisson_nll(lambda: &[f32], counts: &[f32]) -> f64 {
    lambda
        .iter()
        .zip(counts)
        .map(|(model, observed)| {
            let model = f64::from(*model);
            let observed = f64::from(*observed);
            if model > 0.0 {
                model - observed * model.ln()
            } else if observed > 0.0 {
                f64::INFINITY
            } else {
                0.0
            }
        })
        .sum()
}

/// What a solver starts from, beyond the data and its configuration.
///
/// `Default` means the solver's own starting point, which for every method here
/// is the scene's bulk density. A caller that already has an estimate passes it
/// here instead — a warm start from a cheaper solver, or the refinement pass
/// after a slider moves — and the solver begins from it rather than discarding
/// it.
///
/// This is a parameter rather than a `Config` field because an image is data:
/// its length must match the grid, the UI has to be able to re-run a solver from
/// the answer it is showing, and a config is a value it caches. The reference
/// image A4 anchors on is the same kind of input, so it lives here too.
#[derive(Debug, Clone, Copy, Default)]
pub struct Start<'a> {
    /// Initial density, one entry per voxel, in g cm⁻³. Clamped into the box.
    pub rho: Option<&'a [f32]>,
    /// Known structure `ρ_ref`, one entry per voxel, in g cm⁻³.
    ///
    /// Required by A4, which solves for `δ = ρ − ρ_ref` and bounds `δ` by the
    /// reference itself; every other solver ignores it. It is not optional in
    /// A4's construction because a reference-anchored method given no reference
    /// is solving a different problem under the same name.
    pub reference: Option<&'a [f32]>,
}

impl<'a> Start<'a> {
    /// Begin from an existing estimate.
    pub fn from_rho(rho: &'a [f32]) -> Self {
        Self {
            rho: Some(rho),
            ..Self::default()
        }
    }

    /// Anchor on a known structure, starting the deviation at zero.
    pub fn from_reference(reference: &'a [f32]) -> Self {
        Self {
            reference: Some(reference),
            ..Self::default()
        }
    }

    /// Anchor on a known structure and begin from an existing estimate.
    pub fn with_reference(self, reference: &'a [f32]) -> Self {
        Self {
            reference: Some(reference),
            ..self
        }
    }
}

/// One iteration's worth of feedback from a solver.
#[derive(Debug, Clone, PartialEq)]
pub struct StepReport {
    /// Zero-based iteration or subset sweep index.
    pub iteration: u32,
    /// Current value of the objective, if the solver has one.
    pub objective: Option<f64>,
    /// Primal–dual gap, for a solver that has a dual certificate.
    ///
    /// A1 and A2 prove nothing about optimality: they stop on a relative
    /// decrease or on a projected-gradient ratio, which is a statement about the
    /// iterates, not about the answer. A3 can bound its own distance from the
    /// optimum, so it reports that bound rather than asking to be believed.
    pub gap: Option<f64>,
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

    /// Build a solver for one observation.
    ///
    /// The [`Problem`] carries more than the operator — the flux table, the
    /// fidelity and the exposure — because every solver needs them: A1 turns
    /// counts into transmission with `k_scale` and `Λ`, and A2/A3 evaluate the
    /// full flux law. A solver that could only see the matrix would have to
    /// assume those, and would then fit a model the data was not generated
    /// under while every convergence test still passed.
    ///
    /// `start` carries an initial estimate when the caller has one, and is
    /// empty when it does not. A solver that cannot honour a supplied start
    /// must say so — silently ignoring it turns a warm start into a cold one
    /// that reports itself as warm.
    fn init(problem: &Problem<'_>, cfg: &Self::Config, start: &Start<'_>) -> Self;

    /// Advance one iteration (or one subset sweep).
    fn step(&mut self, ctl: &Control) -> StepReport;

    /// The current density estimate, one entry per voxel.
    fn estimate(&self) -> &[f32];
}

#[cfg(test)]
mod tests {
    use super::*;
    use umbra_core::{Csr, Grid2, GridAny};
    use umbra_forward::RayCell;
    use umbra_physics::generate_default;

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

    #[test]
    fn a_problem_is_the_data_and_the_physics_that_made_it() {
        let system = System {
            l: Csr::new(vec![0, 1], vec![0], vec![100.0f32]),
            rays: vec![RayCell {
                det: 0,
                zen: 0.0,
                az: 0.0,
                k_scale: 1.0e4,
            }],
            grid: GridAny::Two(Grid2::new([1, 1], [0.0, 0.0], 1.0)),
            poses: Vec::new(),
        };
        let flux = generate_default();
        let counts = [1.0f32];
        let problem = Problem::new(
            &system,
            &counts,
            &flux,
            Fidelity::L0,
            100.0,
            ForwardParams::default(),
        );
        assert_eq!(problem.ray_count(), 1);
        assert_eq!(problem.voxel_count(), 1);
        assert!(problem.nll(&[2.65]).is_finite());
    }
}
