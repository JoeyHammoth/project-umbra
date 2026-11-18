//! A2: nonlinear Poisson MAP with Huber-smoothed TV (spec §5.5).
//!
//! ```text
//! J(ρ) = Σ_i [λ_i(ρ) − y_i ln λ_i(ρ)] + β TV_ε(ρ),     ρ ∈ [0, ρ_max]
//! ∇J   = Lᵀ[(1 − y/λ) ∂λ/∂X] + β ∇TV_ε
//! ```
//!
//! Unlike A1 this is not an L0 method: `λ` and `∂λ/∂X` come from the same
//! forward evaluation, so the full flux law is used whenever the data was
//! generated with it. A1's L0 assumption is a property of A1, not of the crate.
//!
//! # Why Huber
//!
//! Plain TV has a kink at zero gradient, so every voxel boundary contributes a
//! non-differentiable term and a quasi-Newton method has nothing to do there.
//! `TV_ε` replaces `|d|` with `d²/(2ε)` for `|d| ≤ ε` and `d − ε/2` above it:
//! differentiable at the origin, still linear — still edge-preserving — for
//! jumps larger than `ε`. `ε` is in g cm⁻³, so it is a statement about the
//! smallest density jump that should count as an edge rather than as noise.
//!
//! An isotropic `√(dx² + dy²)` is the textbook TV, and this implementation uses
//! the separable form instead: one Huber term per axis edge. The two agree in
//! the limit of small `ε`, differ in how they round a corner, and the separable
//! gradient is exactly the discrete divergence, which is what makes the
//! finite-difference check in the tests a real check rather than a comparison of
//! two copies of the same expression.
//!
//! # Projected L-BFGS
//!
//! Direction from the two-loop recursion; the iterate is projected onto the box
//! after every trial step, and the Armijo test runs on the projected path. On a
//! limited-angle grid most voxels sit *at* zero, so the box is active over much
//! of the domain and the projection is not a safety net bolted on at the end —
//! it is most of the geometry.
//!
//! # Convergence
//!
//! The KKT residual of the box problem is the projected gradient:
//!
//! ```text
//! p_j = 0        at ρ_j = 0 with g_j > 0, or at ρ_j = ρ_max with g_j < 0
//!     = g_j      otherwise
//! ```
//!
//! Its absolute size is useless as a stopping rule here: the data gradient is of
//! order `10⁶` counts per g cm⁻³ on the shared scene, so an absolute `10⁻³`
//! would need the iterate to be exact to eight digits in f32. Convergence is
//! declared when the residual has fallen to `tolerance` times its initial
//! magnitude — a relative gradient test, the same one L-BFGS implementations
//! use — and both the absolute residual and the ratio are exposed so a caller
//! can judge for itself.

use umbra_core::GridAny;
use umbra_forward::expected::ForwardParams;
use umbra_forward::scene::SceneSpec;
use umbra_forward::{expected, Expected, System};
use umbra_physics::{Fidelity, FluxTable};

use crate::{poisson_nll, Control, Problem, Solver, Start, StepReport};

/// Tuning for [`PoissonMap`].
#[derive(Debug, Clone, PartialEq)]
pub struct A2Config {
    /// TV weight `β`. Zero is the plain Poisson ML problem with a box.
    pub beta: f32,
    /// Huber scale `ε`, in g cm⁻³: jumps below this are smoothed quadratically.
    pub huber_eps: f32,
    /// Starting density, in g cm⁻³, for every voxel.
    pub rho_init: f32,
    /// Upper bound of the box, in g cm⁻³.
    pub rho_max: f32,
    /// Convergence when the KKT residual is this fraction of its initial size.
    pub tolerance: f32,
    /// How many times a step may be halved before the direction is rebuilt.
    pub max_backtracks: u32,
    /// Number of stored `(s, y)` pairs in the L-BFGS recursion.
    pub memory: usize,
}

impl Default for A2Config {
    fn default() -> Self {
        Self {
            // The data term is of order 10⁹ on the shared scene and a single
            // edge of TV of order 1, so β has to be large to be felt at all.
            // 10³ is ~0.1% of the data term: enough to choose between the many
            // reconstructions that fit 32 rays equally well, small enough not to
            // buy smoothness with misfit. `it_improves_the_fit_and_keeps_the
            // edges` pins both halves of that.
            beta: 1.0e3,
            huber_eps: 0.05,
            rho_init: 2.65,
            rho_max: 4.0,
            tolerance: 1.0e-3,
            max_backtracks: 20,
            memory: 8,
        }
    }
}

impl A2Config {
    /// The configuration a scene implies: its bulk density and a box that holds
    /// a dense inclusion. Physics comes from the [`Problem`].
    pub fn from_scene(scene: &SceneSpec) -> Self {
        Self {
            rho_init: scene.physics.rho_bulk,
            rho_max: (scene.physics.rho_bulk * 1.5).max(1.0),
            ..Self::default()
        }
    }
}

/// Nonlinear Poisson MAP with Huber TV, solved by projected L-BFGS.
#[derive(Debug, Clone)]
pub struct PoissonMap {
    system: System,
    counts: Vec<f32>,
    flux: FluxTable,
    fidelity: Fidelity,
    exposure_s: f32,
    params: ForwardParams,
    /// Grid shape, row-major with the first axis fastest.
    nx: usize,
    ny: usize,
    rho: Vec<f32>,
    lambda: Vec<f32>,
    nll: f64,
    tv: f64,
    objective: f64,
    gradient: Vec<f32>,
    /// Absolute projected-gradient norm at the current iterate.
    kkt: f32,
    /// The same norm at the starting point, which is what convergence is
    /// measured against.
    kkt_scale: f64,
    /// Stored `(s, y)` curvature pairs, oldest first.
    history: Vec<(Vec<f32>, Vec<f32>)>,
    beta: f64,
    huber_eps: f64,
    rho_max: f32,
    tolerance: f32,
    max_backtracks: u32,
    memory: usize,
    iteration: u32,
    converged: bool,
}

impl PoissonMap {
    /// Builds the solver, starting from the scene's bulk density.
    pub fn new(problem: &Problem<'_>, cfg: A2Config) -> Self {
        Self::with_start(problem, cfg, &Start::default())
    }

    /// Builds the solver, beginning from `start.rho` when one is supplied.
    ///
    /// # Panics
    ///
    /// Panics if `start.rho` does not have one entry per voxel, if the Huber
    /// scale is not positive or the TV weight negative, or if the grid is not
    /// two-dimensional — the TV term needs a difference operator and this one is
    /// 2-D.
    pub fn with_start(problem: &Problem<'_>, cfg: A2Config, start: &Start<'_>) -> Self {
        assert!(
            cfg.huber_eps.is_finite() && cfg.huber_eps > 0.0,
            "the Huber scale must be positive"
        );
        assert!(
            cfg.beta.is_finite() && cfg.beta >= 0.0,
            "the TV weight must be non-negative"
        );
        assert!(cfg.memory >= 1, "L-BFGS needs at least one stored pair");
        let (nx, ny) = match &problem.system.grid {
            GridAny::Two(grid) => (grid.n[0] as usize, grid.n[1] as usize),
            _ => panic!("A2's TV term is two-dimensional; a 3-D grid needs a 3-D difference"),
        };
        let voxels = problem.voxel_count();
        let rho: Vec<f32> = match start.rho {
            Some(rho) => {
                assert_eq!(
                    rho.len(),
                    voxels,
                    "a starting estimate must have one entry per voxel"
                );
                rho.iter()
                    .map(|density| density.clamp(0.0, cfg.rho_max))
                    .collect()
            }
            None => vec![cfg.rho_init.clamp(0.0, cfg.rho_max); voxels],
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
            rho: rho.clone(),
            lambda: Vec::new(),
            nll: 0.0,
            tv: 0.0,
            objective: f64::INFINITY,
            gradient: vec![0.0; voxels],
            kkt: f32::INFINITY,
            kkt_scale: 1.0,
            history: Vec::new(),
            beta: f64::from(cfg.beta),
            huber_eps: f64::from(cfg.huber_eps),
            rho_max: cfg.rho_max,
            tolerance: cfg.tolerance,
            max_backtracks: cfg.max_backtracks,
            memory: cfg.memory,
            iteration: 0,
            converged: false,
        };
        solver.assign(rho);
        solver.kkt_scale = f64::from(solver.kkt).max(f64::MIN_POSITIVE);
        solver
    }

    /// `J = NLL + β·TV`, the quantity being minimised.
    pub fn objective(&self) -> f64 {
        self.objective
    }

    /// The data term alone, so callers can compare fits across solvers whose
    /// penalties differ.
    pub fn nll(&self) -> f64 {
        self.nll
    }

    /// The (Huber) total variation of the current estimate.
    pub fn tv(&self) -> f64 {
        self.tv
    }

    /// Absolute projected-gradient norm at the current iterate.
    pub fn kkt_residual(&self) -> f32 {
        self.kkt
    }

    /// The same residual relative to its value at the starting point.
    pub fn kkt_ratio(&self) -> f64 {
        f64::from(self.kkt) / self.kkt_scale
    }

    /// Completed steps.
    pub fn iterations(&self) -> u32 {
        self.iteration
    }

    /// `λ` at the current estimate.
    pub fn lambda(&self) -> &[f32] {
        &self.lambda
    }

    /// `∇J` at the current estimate.
    pub fn gradient(&self) -> &[f32] {
        &self.gradient
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

    /// Recomputes everything that depends on the iterate.
    fn assign(&mut self, rho: Vec<f32>) {
        let Expected {
            lambda, dlambda_dx, ..
        } = self.expected(&rho);
        let nll = poisson_nll(&lambda, &self.counts);
        let tv = self.tv_of(&rho);
        let mut gradient = self.data_gradient(&lambda, &dlambda_dx);
        self.add_tv_gradient(&rho, &mut gradient);
        self.rho = rho;
        self.nll = nll;
        self.tv = tv;
        self.objective = nll + self.beta * tv;
        self.lambda = lambda;
        self.gradient = gradient;
        self.kkt = self.kkt_residual_now();
    }

    /// `Lᵀ[(1 − y/λ) ∂λ/∂X]`: the gradient of the data term.
    fn data_gradient(&self, lambda: &[f32], dlambda_dx: &[f32]) -> Vec<f32> {
        let mut weight = vec![0.0f32; self.counts.len()];
        for (ray, slot) in weight.iter_mut().enumerate() {
            let model = f64::from(lambda[ray]);
            if model <= 0.0 || model.is_nan() {
                // Counts against zero acceptance: the objective is infinite and
                // no gradient exists. The line search rejects such trials.
                continue;
            }
            let observed = f64::from(self.counts[ray]);
            *slot = ((1.0 - observed / model) * f64::from(dlambda_dx[ray])) as f32;
        }
        let mut gradient = vec![0.0f32; self.system.voxel_count()];
        self.system.l.matvec_t(&weight, &mut gradient);
        gradient
    }

    fn tv_of(&self, rho: &[f32]) -> f64 {
        let eps = self.huber_eps;
        let mut total = 0.0;
        for j in 0..self.ny {
            for i in 0..self.nx {
                let here = i + self.nx * j;
                let value = f64::from(rho[here]);
                if i + 1 < self.nx {
                    total += huber(f64::from(rho[here + 1]) - value, eps);
                }
                if j + 1 < self.ny {
                    total += huber(f64::from(rho[here + self.nx]) - value, eps);
                }
            }
        }
        total
    }

    /// Adds `β ∇TV_ε` to `gradient`, in place.
    fn add_tv_gradient(&self, rho: &[f32], gradient: &mut [f32]) {
        let eps = self.huber_eps;
        for j in 0..self.ny {
            for i in 0..self.nx {
                let here = i + self.nx * j;
                let value = f64::from(rho[here]);
                if i + 1 < self.nx {
                    let there = here + 1;
                    let difference = f64::from(rho[there]) - value;
                    let flux = self.beta * huber_weight(difference, eps) * difference;
                    gradient[here] -= flux as f32;
                    gradient[there] += flux as f32;
                }
                if j + 1 < self.ny {
                    let there = here + self.nx;
                    let difference = f64::from(rho[there]) - value;
                    let flux = self.beta * huber_weight(difference, eps) * difference;
                    gradient[here] -= flux as f32;
                    gradient[there] += flux as f32;
                }
            }
        }
    }

    fn kkt_residual_now(&self) -> f32 {
        let mut worst = 0.0f32;
        for (voxel, grad) in self.gradient.iter().enumerate() {
            let at_lower = self.rho[voxel] <= 0.0;
            let at_upper = self.rho[voxel] >= self.rho_max;
            let projected = if (at_lower && *grad > 0.0) || (at_upper && *grad < 0.0) {
                0.0
            } else {
                *grad
            };
            worst = worst.max(projected.abs());
        }
        worst
    }

    /// The L-BFGS direction, or a density-scaled steepest descent before the
    /// first curvature pair exists.
    fn direction(&self) -> Vec<f32> {
        let n = self.gradient.len();
        if self.history.is_empty() {
            let largest = self
                .gradient
                .iter()
                .fold(0.0f32, |worst, grad| worst.max(grad.abs()));
            if largest <= 0.0 || largest.is_nan() {
                return vec![0.0; n];
            }
            return self
                .gradient
                .iter()
                .map(|grad| -grad * self.rho_max / largest)
                .collect();
        }

        let mut q: Vec<f64> = self.gradient.iter().map(|grad| f64::from(*grad)).collect();
        let pairs = self.history.len();
        let mut alpha = vec![0.0f64; pairs];
        for k in (0..pairs).rev() {
            let (s, y) = &self.history[k];
            let sy = dot(s, y);
            if sy <= 0.0 || sy.is_nan() {
                continue;
            }
            alpha[k] = dot_mixed(s, &q) / sy;
            for (value, slope) in q.iter_mut().zip(y) {
                *value -= alpha[k] * f64::from(*slope);
            }
        }
        let (s_last, y_last) = &self.history[pairs - 1];
        let sy = dot(s_last, y_last);
        let yy = dot(y_last, y_last);
        let gamma = if sy > 0.0 && yy > 0.0 { sy / yy } else { 1.0 };
        for value in q.iter_mut() {
            *value *= gamma;
        }
        for k in 0..pairs {
            let (s, y) = &self.history[k];
            let sy = dot(s, y);
            if sy <= 0.0 || sy.is_nan() {
                continue;
            }
            let beta = dot_mixed(y, &q) / sy;
            for (value, step) in q.iter_mut().zip(s) {
                *value += (alpha[k] - beta) * f64::from(*step);
            }
        }
        q.iter().map(|value| -*value as f32).collect()
    }
}

/// Huber on one edge: quadratic below `ε`, linear above.
fn huber(difference: f64, eps: f64) -> f64 {
    let magnitude = difference.abs();
    if magnitude <= eps {
        magnitude * magnitude / (2.0 * eps)
    } else {
        magnitude - 0.5 * eps
    }
}

/// `h'(s)/s`: turns an edge difference into its gradient contribution.
fn huber_weight(difference: f64, eps: f64) -> f64 {
    let magnitude = difference.abs();
    if magnitude <= eps {
        1.0 / eps
    } else {
        1.0 / magnitude
    }
}

fn dot(left: &[f32], right: &[f32]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum()
}

/// `⟨left, right⟩` where the right-hand side carries the recursion's `f64`.
fn dot_mixed(left: &[f32], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| f64::from(*a) * *b)
        .sum()
}

impl Solver for PoissonMap {
    type Config = A2Config;

    fn init(problem: &Problem<'_>, cfg: &Self::Config, start: &Start<'_>) -> Self {
        Self::with_start(problem, cfg.clone(), start)
    }

    fn step(&mut self, ctl: &Control) -> StepReport {
        if self.converged || ctl.should_stop() || self.iteration >= ctl.max_iters {
            return StepReport {
                iteration: self.iteration,
                objective: Some(self.objective),
                converged: self.converged,
            };
        }

        let mut direction = self.direction();
        let mut slope = dot(&self.gradient, &direction);
        if slope >= 0.0 || slope.is_nan() {
            // The quasi-Newton model is not predicting descent — possible after
            // a projection changes the active set. Steepest descent always does
            // unless the projected gradient is already zero.
            self.history.clear();
            direction = self.gradient.iter().map(|grad| -grad).collect();
            slope = dot(&self.gradient, &direction);
        }
        if slope >= 0.0 || slope.is_nan() {
            self.converged = true;
            return StepReport {
                iteration: self.iteration,
                objective: Some(self.objective),
                converged: true,
            };
        }

        let mut step_size = 1.0f32;
        let mut moved = false;
        for _ in 0..=self.max_backtracks {
            let trial: Vec<f32> = self
                .rho
                .iter()
                .zip(&direction)
                .map(|(rho, step)| (rho + step_size * step).clamp(0.0, self.rho_max))
                .collect();
            let Expected {
                lambda, dlambda_dx, ..
            } = self.expected(&trial);
            let nll = poisson_nll(&lambda, &self.counts);
            let tv = self.tv_of(&trial);
            let objective = nll + self.beta * tv;
            if objective <= self.objective + 1.0e-4 * f64::from(step_size) * slope {
                let mut gradient = self.data_gradient(&lambda, &dlambda_dx);
                self.add_tv_gradient(&trial, &mut gradient);
                let s: Vec<f32> = trial
                    .iter()
                    .zip(&self.rho)
                    .map(|(new, old)| new - old)
                    .collect();
                let y: Vec<f32> = gradient
                    .iter()
                    .zip(&self.gradient)
                    .map(|(new, old)| new - old)
                    .collect();
                let curvature = dot(&s, &y);
                if curvature > 1.0e-6 * norm(&s) * norm(&y) {
                    if self.history.len() == self.memory {
                        self.history.remove(0);
                    }
                    self.history.push((s, y));
                }
                self.rho = trial;
                self.lambda = lambda;
                self.nll = nll;
                self.tv = tv;
                self.objective = objective;
                self.gradient = gradient;
                self.kkt = self.kkt_residual_now();
                self.iteration += 1;
                self.converged = self.kkt_ratio() <= f64::from(self.tolerance);
                moved = true;
                break;
            }
            step_size *= 0.5;
        }

        if !moved {
            // No relaxation descends. Either the curvature model needs
            // rebuilding (clear it and retry with steepest descent) or this is
            // the stationary point.
            if self.history.is_empty() {
                self.converged = true;
            } else {
                self.history.clear();
            }
        }

        StepReport {
            iteration: self.iteration,
            objective: Some(self.objective),
            converged: self.converged,
        }
    }

    fn estimate(&self) -> &[f32] {
        &self.rho
    }
}

fn norm(vector: &[f32]) -> f64 {
    vector
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt()
}
