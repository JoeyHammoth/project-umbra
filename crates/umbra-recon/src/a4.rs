//! A4: reference-anchored one-sided sparse deviation, solved by FISTA (spec §5.5).
//!
//! ```text
//! ρ = ρ_ref + δ,    −ρ_ref ≤ δ ≤ 0        (voids only remove material)
//! min_δ  NLL(ρ_ref + δ) + μ‖δ‖₁ + τ TV_ε(δ)
//! ```
//!
//! # Why the reference changes the problem
//!
//! A2 and A3 both answer "what density field fits these counts?" with a
//! smoothness prior, so where the data are weakly informative they return the
//! smoothest field that fits — which is exactly the situation a buried void
//! creates. A4 instead assumes the structure is known and asks only for the
//! *deviation*. Two things follow.
//!
//! First, the prior stops being a preference for smoothness and becomes a
//! statement about what can physically happen: a cavity removes material, so
//! `δ ≤ 0`, and the bound `δ ≥ −ρ_ref` says the deviation cannot remove more
//! than the voxel contains. The `δ ≥ 0` mode exists for the opposite case — a
//! granite block or fill added to a rubble interior — and it is a different
//! physical statement, not a sign convention.
//!
//! Second, `‖δ‖₁` is now cheap and meaningful. Over `δ ≤ 0`,
//! `‖δ‖₁ = −1ᵀδ` identically, so the penalty is linear on the feasible set and
//! its proximal step collapses to the closed form in [`prox`]:
//!
//! ```text
//! prox_{tμ + ι_[−ρ_ref, 0]}(z) = min(0, max(−ρ_ref, z + tμ))
//! ```
//!
//! # Monotone FISTA
//!
//! Plain FISTA is not a descent method: the inertial step can overshoot and the
//! objective rises before it falls. Rather than assert monotonicity and hope,
//! this implementation checks the full objective after each step and, when it
//! rose, keeps the previous iterate and drops the momentum (the MFISTA idea of
//! Beck & Teboulle). The objective is therefore non-increasing by construction,
//! and the momentum is only given up on the steps that would have broken it.
//!
//! The step size comes from the same backtracking rule: `L` is increased until
//! `f(x⁺) ≤ f(y) + ⟨∇f(y), x⁺−y⟩ + (L/2)‖x⁺−y‖²`, which is the condition FISTA
//! needs and the reason no scene-specific tuning is asked for.
//!
//! # What the TV variant is for
//!
//! `τ > 0` adds Huber TV *on the deviation*, which keeps blocky cavities from
//! being eroded into a smooth blob while leaving the reference's own edges
//! alone. It is smoothed (Huber, as in A2) on purpose: the non-smooth part of
//! the problem has to stay the one-sided `ℓ₁` term, because that is the term
//! with the closed-form prox. A hard `‖∇δ‖₁` here would move the non-smoothness
//! into the gradient and cost the exact proximal step for no gain.

use umbra_core::{Csr, GridAny};
use umbra_forward::expected::ForwardParams;
use umbra_forward::scene::SceneSpec;
use umbra_forward::{expected, Expected, System};
use umbra_physics::{Fidelity, FluxTable};

use crate::{poisson_nll, Control, Problem, Solver, Start, StepReport};

/// Which way the deviation is allowed to go.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviationSign {
    /// Material can only be removed: `−ρ_ref ≤ δ ≤ 0`.
    Void,
    /// Material can only be added: `0 ≤ δ ≤ delta_max`.
    Inclusion,
}

impl DeviationSign {
    /// Whether `δ` is bounded above by zero.
    pub fn is_void(self) -> bool {
        matches!(self, DeviationSign::Void)
    }
}

/// The proximal step of `tμ‖·‖₁ + ι_[lo, hi]`, in closed form.
///
/// `lo` and `hi` are `(−ρ_ref, 0)` for a void and `(0, delta_max)` for an
/// inclusion. The two cases differ only by the sign of the linear push, which
/// is the whole content of "one-sided": on `δ ≤ 0` the `ℓ₁` penalty is
/// `−μ1ᵀδ`, a linear term, so its prox is a shift followed by a clamp rather
/// than a soft threshold.
pub fn prox(
    sign: DeviationSign,
    reference: f32,
    z: f64,
    step: f64,
    mu: f64,
    delta_max: f32,
) -> f64 {
    let reference = f64::from(reference);
    match sign {
        DeviationSign::Void => (z + step * mu).clamp(-reference, 0.0),
        DeviationSign::Inclusion => (z - step * mu).clamp(0.0, f64::from(delta_max)),
    }
}

/// Tuning for [`SparseDeviation`].
#[derive(Debug, Clone, PartialEq)]
pub struct A4Config {
    /// `ℓ₁` weight on the deviation, `μ`, in NLL per g cm⁻³.
    pub mu: f32,
    /// Huber TV weight on the deviation, `τ`. Zero disables it.
    pub tv: f32,
    /// Huber scale `ε`, in g cm⁻³.
    pub huber_eps: f32,
    /// Which way the deviation may go.
    pub sign: DeviationSign,
    /// Upper bound of an inclusion, in g cm⁻³. Ignored for a void, whose bound
    /// is the reference itself.
    pub delta_max: f32,
    /// Convergence when two consecutive steps lower the objective by less than
    /// this fraction of it.
    ///
    /// Tight on purpose: the objective is of order 10⁹, so `10⁻⁶` is an absolute
    /// bar of thousands of NLL units and stops the solver while the void is still
    /// half empty. `10⁻⁹` is a fraction of a unit here.
    ///
    /// The test is on the objective rather than on the iterate: once a voxel sits
    /// on its bound, `δ` can jitter in the last few bits while the objective is
    /// converged, so an iterate-change rule would never fire.
    pub tolerance: f32,
    /// Backtracking attempts before a step is abandoned.
    pub max_backtracks: u32,
    /// Override for the Lipschitz constant of the smooth part, in NLL per
    /// (g cm⁻³)². Zero estimates it from the data, which is the intended setting:
    /// a value that is orders of magnitude too small makes `μ/L` dominate the
    /// proximal step and pins the deviation to its bound.
    pub lipschitz_init: f32,
}

impl Default for A4Config {
    fn default() -> Self {
        Self {
            // The price of removing a gram per voxel, in NLL. Measured on the
            // shared scene: 10 converges in 35 steps and leaves 0.10 g cm⁻³ in
            // the corridor, 10² reaches 0.03 but needs ~600, and 10⁵ stops the
            // field from moving at all. The tempting 10⁶ comes from the size of
            // the NLL rather than from the size of its derivative, and it pins
            // every voxel to its bound.
            mu: 1.0e1,
            tv: 0.0,
            huber_eps: 0.05,
            sign: DeviationSign::Void,
            delta_max: 4.0,
            tolerance: 1.0e-9,
            max_backtracks: 60,
            lipschitz_init: 0.0,
        }
    }
}

impl A4Config {
    /// The configuration a scene implies.
    ///
    /// The deviation's upper bound in inclusion mode is scaled from the scene's
    /// bulk density, and the physics comes from the [`Problem`] rather than from
    /// a second copy here.
    pub fn from_scene(scene: &SceneSpec) -> Self {
        Self {
            delta_max: (scene.physics.rho_bulk * 1.5).max(1.0),
            ..Self::default()
        }
    }

    /// The same configuration with the TV-on-deviation term switched on.
    pub fn with_tv(mut self, tv: f32) -> Self {
        self.tv = tv;
        self
    }
}

/// Reference-anchored one-sided sparse deviation, solved by monotone FISTA.
#[derive(Debug, Clone)]
pub struct SparseDeviation {
    system: System,
    counts: Vec<f32>,
    flux: FluxTable,
    fidelity: Fidelity,
    exposure_s: f32,
    params: ForwardParams,
    nx: usize,
    ny: usize,
    reference: Vec<f32>,
    /// `x_k`.
    delta: Vec<f32>,
    /// `x_{k-1}`, for the momentum term and the monotone fallback.
    previous: Vec<f32>,
    /// `y_k`, the inertial point the gradient is taken at.
    inertial: Vec<f32>,
    /// `t_k`.
    momentum: f64,
    lipschitz: f64,
    rho: Vec<f32>,
    nll: f64,
    l1: f64,
    tv: f64,
    objective: f64,
    mu: f64,
    tau: f64,
    huber_eps: f64,
    sign: DeviationSign,
    delta_max: f32,
    tolerance: f32,
    max_backtracks: u32,
    iteration: u32,
    converged: bool,
    /// Consecutive steps that moved the deviation by less than `tolerance`.
    stalls: u32,
}

impl SparseDeviation {
    /// Builds the solver from an explicit reference structure.
    ///
    /// # Panics
    ///
    /// Panics if the reference does not have one entry per voxel.
    pub fn new(problem: &Problem<'_>, cfg: A4Config, reference: &[f32]) -> Self {
        Self::with_start(problem, cfg, &Start::from_reference(reference))
    }

    /// Builds the solver, taking the reference from `start.reference`.
    ///
    /// # Panics
    ///
    /// Panics if no reference is supplied, if it does not have one entry per
    /// voxel, if `μ` or `τ` is negative, or if the grid is not two-dimensional.
    pub fn with_start(problem: &Problem<'_>, cfg: A4Config, start: &Start<'_>) -> Self {
        let reference = start.reference.unwrap_or_else(|| {
            panic!(
                "A4 is reference-anchored: pass the known structure through \
                 `Start::from_reference`, or use `SparseDeviation::new`"
            )
        });
        assert!(
            cfg.mu.is_finite() && cfg.mu >= 0.0,
            "the sparsity weight must be non-negative"
        );
        assert!(
            cfg.tv.is_finite() && cfg.tv >= 0.0,
            "the TV weight must be non-negative"
        );
        assert!(
            cfg.huber_eps.is_finite() && cfg.huber_eps > 0.0,
            "the Huber scale must be positive"
        );
        let (nx, ny) = match &problem.system.grid {
            GridAny::Two(grid) => (grid.n[0] as usize, grid.n[1] as usize),
            _ => panic!("A4's TV term is two-dimensional; a 3-D grid needs a 3-D difference"),
        };
        let voxels = problem.voxel_count();
        assert_eq!(
            reference.len(),
            voxels,
            "the reference must have one entry per voxel"
        );
        assert_eq!(nx * ny, voxels, "the grid and the operator must agree");

        let reference: Vec<f32> = reference.iter().map(|density| density.max(0.0)).collect();
        // A supplied start is a density field, so it enters as a deviation and
        // is clamped into the one-sided box rather than trusted.
        let delta: Vec<f32> = match start.rho {
            Some(rho) => {
                assert_eq!(
                    rho.len(),
                    voxels,
                    "a starting estimate must have one entry per voxel"
                );
                rho.iter()
                    .zip(&reference)
                    .map(|(density, reference)| {
                        let deviation = f64::from(*density) - f64::from(*reference);
                        clamp(cfg.sign, deviation, f64::from(*reference), cfg.delta_max) as f32
                    })
                    .collect()
            }
            None => vec![0.0; voxels],
        };

        let mut solver = Self {
            system: problem.system.clone(),
            counts: problem.counts.to_vec(),
            flux: problem.flux.clone(),
            fidelity: problem.fidelity,
            exposure_s: problem.exposure_s,
            params: problem.params,
            nx,
            ny,
            reference: reference.clone(),
            delta: delta.clone(),
            previous: delta.clone(),
            inertial: delta,
            momentum: 1.0,
            lipschitz: 1.0,
            rho: Vec::new(),
            nll: 0.0,
            l1: 0.0,
            tv: 0.0,
            objective: f64::INFINITY,
            mu: f64::from(cfg.mu),
            tau: f64::from(cfg.tv),
            huber_eps: f64::from(cfg.huber_eps),
            sign: cfg.sign,
            delta_max: cfg.delta_max,
            tolerance: cfg.tolerance,
            max_backtracks: cfg.max_backtracks,
            iteration: 0,
            converged: false,
            stalls: 0,
        };
        solver.assess();
        solver.lipschitz = if cfg.lipschitz_init > 0.0 {
            f64::from(cfg.lipschitz_init)
        } else {
            solver.estimate_lipschitz()
        };
        solver
    }

    /// A Lipschitz constant for the smooth part, from the curvature at the
    /// reference.
    ///
    /// The data term's Hessian is `Lᵀ diag[(1/λ)(∂λ/∂X)²] L`, so its largest
    /// eigenvalue is at most `max_i (∂λ/∂X)²/λ · ‖L‖²`, and the Huber TV term
    /// contributes at most `8τ/ε` in two dimensions. Backtracking still runs
    /// every step, because a local estimate is not a global bound — but it is a
    /// starting point of the right order, which is the difference between a step
    /// that moves and one that the sparsity weight swallows whole.
    fn estimate_lipschitz(&self) -> f64 {
        let Expected {
            lambda, dlambda_dx, ..
        } = self.expected(&self.rho);
        let mut curvature = 0.0f64;
        for (model, slope) in lambda.iter().zip(&dlambda_dx) {
            let model = f64::from(*model);
            if model > 0.0 {
                let slope = f64::from(*slope);
                curvature = curvature.max(slope * slope / model);
            }
        }
        let operator = operator_norm_squared(&self.system.l, self.rho.len(), 40);
        let tv = if self.tau > 0.0 {
            8.0 * self.tau / self.huber_eps
        } else {
            0.0
        };
        (curvature * operator + tv).max(1.0e-6)
    }

    /// The deviation `δ`, one entry per voxel, in g cm⁻³.
    pub fn deviation(&self) -> &[f32] {
        &self.delta
    }

    /// `NLL(ρ_ref + δ)`, the data term.
    pub fn nll(&self) -> f64 {
        self.nll
    }

    /// `μ‖δ‖₁`, the sparsity term.
    pub fn l1(&self) -> f64 {
        self.l1
    }

    /// `τ·TV_ε(δ)`, the blockiness term.
    pub fn tv(&self) -> f64 {
        self.tv
    }

    /// The whole objective.
    pub fn objective(&self) -> f64 {
        self.objective
    }

    /// The current Lipschitz estimate for the smooth part.
    pub fn lipschitz(&self) -> f64 {
        self.lipschitz
    }

    /// Completed iterations.
    pub fn iterations(&self) -> u32 {
        self.iteration
    }

    /// Where the deviation may go.
    pub fn sign(&self) -> DeviationSign {
        self.sign
    }

    /// Recomputes `ρ`, the objective and its parts from `δ`.
    fn assess(&mut self) {
        self.rho = self
            .reference
            .iter()
            .zip(&self.delta)
            .map(|(reference, deviation)| reference + deviation)
            .collect();
        let Expected { lambda, .. } = self.expected(&self.rho);
        self.nll = poisson_nll(&lambda, &self.counts);
        self.l1 = self.mu * self.delta.iter().map(|d| f64::from(d.abs())).sum::<f64>();
        self.tv = self.tau * self.tv_of(&self.delta);
        self.objective = self.nll + self.l1 + self.tv;
    }

    fn expected(&self, rho: &[f32]) -> Expected {
        expected(
            &self.system,
            rho,
            &self.flux,
            self.fidelity,
            self.exposure_s,
            &self.params,
        )
    }

    /// `NLL(ρ_ref + δ)`, for a trial deviation.
    fn smooth_at(&self, delta: &[f32]) -> f64 {
        let rho: Vec<f32> = self
            .reference
            .iter()
            .zip(delta)
            .map(|(reference, deviation)| reference + deviation)
            .collect();
        let Expected { lambda, .. } = self.expected(&rho);
        poisson_nll(&lambda, &self.counts)
    }

    /// `∇f(δ)` for the smooth part `f = NLL + τ TV_ε`.
    fn gradient_at(&self, delta: &[f32]) -> Vec<f32> {
        let rho: Vec<f32> = self
            .reference
            .iter()
            .zip(delta)
            .map(|(reference, deviation)| reference + deviation)
            .collect();
        let Expected {
            lambda, dlambda_dx, ..
        } = self.expected(&rho);

        let mut weight = vec![0.0f32; self.counts.len()];
        for (ray, slot) in weight.iter_mut().enumerate() {
            let model = f64::from(lambda[ray]);
            if model <= 0.0 || model.is_nan() {
                // Counts against a ray with no acceptance: the objective is
                // infinite and no gradient exists. The line search rejects such
                // trials rather than pretending they have a slope.
                continue;
            }
            let observed = f64::from(self.counts[ray]);
            *slot = ((1.0 - observed / model) * f64::from(dlambda_dx[ray])) as f32;
        }
        let mut gradient = vec![0.0f32; rho.len()];
        self.system.l.matvec_t(&weight, &mut gradient);
        if self.tau > 0.0 {
            self.add_tv_gradient(delta, &mut gradient);
        }
        gradient
    }

    fn tv_of(&self, delta: &[f32]) -> f64 {
        let eps = self.huber_eps;
        let mut total = 0.0;
        for j in 0..self.ny {
            for i in 0..self.nx {
                let here = i + self.nx * j;
                let value = f64::from(delta[here]);
                if i + 1 < self.nx {
                    total += huber(f64::from(delta[here + 1]) - value, eps);
                }
                if j + 1 < self.ny {
                    total += huber(f64::from(delta[here + self.nx]) - value, eps);
                }
            }
        }
        total
    }

    fn add_tv_gradient(&self, delta: &[f32], gradient: &mut [f32]) {
        let eps = self.huber_eps;
        for j in 0..self.ny {
            for i in 0..self.nx {
                let here = i + self.nx * j;
                let value = f64::from(delta[here]);
                if i + 1 < self.nx {
                    let there = here + 1;
                    let difference = f64::from(delta[there]) - value;
                    let flux = self.tau * huber_weight(difference, eps) * difference;
                    gradient[here] -= flux as f32;
                    gradient[there] += flux as f32;
                }
                if j + 1 < self.ny {
                    let there = here + self.nx;
                    let difference = f64::from(delta[there]) - value;
                    let flux = self.tau * huber_weight(difference, eps) * difference;
                    gradient[here] -= flux as f32;
                    gradient[there] += flux as f32;
                }
            }
        }
    }

    /// `F(δ)`, recomputed for a trial.
    fn objective_at(&self, delta: &[f32]) -> f64 {
        let nll = self.smooth_at(delta);
        let l1 = self.mu * delta.iter().map(|d| f64::from(d.abs())).sum::<f64>();
        let tv = self.tau * self.tv_of(delta);
        nll + l1 + tv
    }
}

fn clamp(sign: DeviationSign, value: f64, reference: f64, delta_max: f32) -> f64 {
    match sign {
        DeviationSign::Void => value.clamp(-reference, 0.0),
        DeviationSign::Inclusion => value.clamp(0.0, f64::from(delta_max)),
    }
}

/// `‖L‖₂²` by power iteration. `L` is sparse and the estimate only sets a
/// starting point for backtracking, so a few dozen iterations is plenty.
fn operator_norm_squared(l: &Csr<f32>, voxels: usize, iterations: u32) -> f64 {
    if voxels == 0 {
        return 0.0;
    }
    // A smooth deterministic start, so two runs on the same scene agree.
    let mut vector: Vec<f32> = (0..voxels)
        .map(|voxel| 1.0 + (0.37 * voxel as f32).sin())
        .collect();
    let mut rayleigh = 0.0f64;
    for _ in 0..iterations {
        let mut forward = vec![0.0f32; l.nrows()];
        l.matvec(&vector, &mut forward);
        let mut back = vec![0.0f32; voxels];
        l.matvec_t(&forward, &mut back);
        rayleigh = vector
            .iter()
            .zip(&back)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum();
        let size = back
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>()
            .sqrt();
        if size <= 0.0 || !size.is_finite() {
            return 0.0;
        }
        for (value, next) in vector.iter_mut().zip(&back) {
            *value = (f64::from(*next) / size) as f32;
        }
    }
    rayleigh.max(0.0)
}

fn huber(difference: f64, eps: f64) -> f64 {
    let magnitude = difference.abs();
    if magnitude <= eps {
        magnitude * magnitude / (2.0 * eps)
    } else {
        magnitude - 0.5 * eps
    }
}

fn huber_weight(difference: f64, eps: f64) -> f64 {
    let magnitude = difference.abs();
    if magnitude <= eps {
        1.0 / eps
    } else {
        1.0 / magnitude
    }
}

impl Solver for SparseDeviation {
    type Config = A4Config;

    fn init(problem: &Problem<'_>, cfg: &Self::Config, start: &Start<'_>) -> Self {
        Self::with_start(problem, cfg.clone(), start)
    }

    fn step(&mut self, ctl: &Control) -> StepReport {
        if self.converged || ctl.should_stop() || self.iteration >= ctl.max_iters {
            return StepReport {
                iteration: self.iteration,
                objective: Some(self.objective),
                gap: None,
                converged: self.converged,
            };
        }

        let voxels = self.delta.len();
        let inertial = self.inertial.clone();
        let gradient = self.gradient_at(&inertial);
        let smooth = self.smooth_at(&inertial);

        // Backtracking on the smooth part: the step is legitimate only where the
        // quadratic model dominates, and `L` learns that from the data instead of
        // from a scene-specific constant.
        let mut lipschitz = self.lipschitz;
        let mut candidate = vec![0.0f32; voxels];
        let mut attempts = 0;
        loop {
            let mut quadratic = 0.0;
            let mut linear = 0.0;
            for voxel in 0..voxels {
                let step = 1.0 / lipschitz;
                let z = f64::from(inertial[voxel]) - step * f64::from(gradient[voxel]);
                candidate[voxel] = prox(
                    self.sign,
                    self.reference[voxel],
                    z,
                    step,
                    self.mu,
                    self.delta_max,
                ) as f32;
                let difference = f64::from(candidate[voxel]) - f64::from(inertial[voxel]);
                linear += f64::from(gradient[voxel]) * difference;
                quadratic += difference * difference;
            }
            let candidate_smooth = self.smooth_at(&candidate);
            if candidate_smooth <= smooth + linear + 0.5 * lipschitz * quadratic {
                break;
            }
            lipschitz *= 2.0;
            attempts += 1;
            if attempts > self.max_backtracks {
                // No step size satisfies the model. That means the smooth part is
                // not differentiable here — counts against zero acceptance — and
                // standing still is the honest response.
                self.converged = true;
                return StepReport {
                    iteration: self.iteration,
                    objective: Some(self.objective),
                    gap: None,
                    converged: true,
                };
            }
        }
        self.lipschitz = lipschitz;

        // The monotone safeguard: accept the step only if it lowers the whole
        // objective, and drop the momentum when it does not. This is what makes
        // "the objective never rises" a property of the algorithm here rather
        // than a hope about the inertia.
        let candidate_objective = self.objective_at(&candidate);
        let moved = candidate_objective <= self.objective;
        let accepted = if moved { candidate } else { self.delta.clone() };

        self.previous = std::mem::replace(&mut self.delta, accepted);
        // `t_{k+1} = (1 + √(1 + 4t_k²))/2` and `β_k = (t_k − 1)/t_{k+1}`, the
        // FISTA momentum. A rejected step restarts it at one, which is what lets
        // the safeguard recover instead of dragging bad inertia forward.
        let beta = if moved {
            let t = self.momentum;
            let next = 0.5 * (1.0 + (1.0 + 4.0 * t * t).sqrt());
            self.momentum = next;
            (t - 1.0) / next
        } else {
            self.momentum = 1.0;
            0.0
        };
        self.inertial = self
            .delta
            .iter()
            .zip(&self.previous)
            .map(|(now, before)| (now + beta as f32 * (now - before)) as f32)
            .collect();

        let before = self.objective;
        self.assess();
        self.iteration += 1;

        let decrease = before - self.objective;
        if decrease <= f64::from(self.tolerance) * self.objective.abs().max(1.0) {
            self.stalls += 1;
        } else {
            self.stalls = 0;
        }
        // One small step can be a momentum artefact; two in a row is a solver
        // that is no longer making progress.
        self.converged = self.stalls >= 2;

        StepReport {
            iteration: self.iteration,
            objective: Some(self.objective),
            gap: None,
            converged: self.converged,
        }
    }

    fn estimate(&self) -> &[f32] {
        &self.rho
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use umbra_core::{Grid2, GridAny};
    use umbra_forward::{ForwardParams, RayCell, System};
    use umbra_physics::{generate_default, FluxTable};

    fn trivial_system(rows: usize) -> System {
        let mut indptr = vec![0u32];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for row in 0..rows {
            indices.push(row as u32);
            data.push(100.0f32);
            indptr.push(indices.len() as u32);
        }
        System {
            l: Csr::new(indptr, indices, data),
            rays: (0..rows)
                .map(|_| RayCell {
                    det: 0,
                    zen: 0.0,
                    az: 0.0,
                    k_scale: 1.0e4,
                })
                .collect(),
            grid: GridAny::Two(Grid2::new([rows as u32, 1], [0.0, 0.0], 1.0)),
            poses: Vec::new(),
        }
    }

    #[test]
    #[should_panic(expected = "reference-anchored")]
    fn a_reference_is_required_rather_than_defaulted() {
        let system = trivial_system(2);
        let flux: FluxTable = generate_default();
        let counts = [10.0f32, 10.0];
        let problem = Problem::new(
            &system,
            &counts,
            &flux,
            Fidelity::L0,
            1.0e4,
            ForwardParams::default(),
        );
        let _ = SparseDeviation::with_start(&problem, A4Config::default(), &Start::default());
    }

    #[test]
    fn a_start_is_read_as_a_deviation_from_the_reference() {
        let system = trivial_system(1);
        let flux = generate_default();
        let counts = [100.0f32];
        let problem = Problem::new(
            &system,
            &counts,
            &flux,
            Fidelity::L0,
            1.0e4,
            ForwardParams::default(),
        );
        let reference = [2.0f32];
        let start = Start::from_rho(&[1.5]).with_reference(&reference);
        let solver = SparseDeviation::with_start(&problem, A4Config::default(), &start);
        assert_eq!(solver.deviation(), &[-0.5]);
        assert_eq!(solver.estimate(), &[1.5]);
    }

    #[test]
    fn the_prox_never_leaves_the_one_sided_box() {
        let reference = 2.65f32;
        for sign in [DeviationSign::Void, DeviationSign::Inclusion] {
            for z in [-10.0f64, -1.0, 0.0, 1.0, 10.0] {
                let value = prox(sign, reference, z, 0.5, 100.0, 4.0);
                match sign {
                    DeviationSign::Void => {
                        assert!((-f64::from(reference)..=0.0).contains(&value), "{value}")
                    }
                    DeviationSign::Inclusion => assert!((0.0..=4.0).contains(&value), "{value}"),
                }
            }
        }
    }
}
