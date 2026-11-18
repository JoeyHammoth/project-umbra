//! A3: opacity-domain inversion followed by Chambolle–Pock PDHG (spec §5.5).
//!
//! ```text
//! 1.  X̂_i = g_i⁻¹(y_i)              per-ray inversion, safeguarded Newton
//!     w_i = (∂λ_i/∂X_i)² / λ_i       delta-method weight, at X̂_i
//! 2.  min_ρ  ½‖W^{1/2}(Lρ − X̂)‖² + β‖∇ρ‖_{2,1} + ι_[0, ρ_max](ρ)
//!     solved by PDHG on K = [L; ∇]
//! ```
//!
//! # Why this is a different method and not a different solver for A2
//!
//! A2 keeps the nonlinear likelihood and puts TV inside it, so the data term is
//! exactly the Poisson one and the penalty has to be smoothed to keep the
//! problem differentiable. A3 instead *linearises once*, per ray, and then solves
//! a convex problem exactly: the TV is the true non-smooth `‖·‖_{2,1}` with a
//! hard box, and the dual certificate says how far the answer is from optimal.
//! The price is that the likelihood is replaced by its delta-method quadratic,
//! which is accurate where counts are large and least so where they are zero.
//!
//! # The inversion
//!
//! `g_i(X) = λ_i(X)` is the ray's own forward law at fixed geometry, and it is
//! strictly decreasing in `X`, so inverting a count is a one-dimensional root
//! find. It is done by Newton inside a bisection bracket: Newton alone can walk
//! out of the domain where the count is zero (the root is at `X = ∞` in the
//! limit), and bisection alone converges linearly for no reason — the root
//! problem is smooth.
//!
//! `λ` comes from [`umbra_forward::expected::ray_lambda`], the same function the
//! forward model uses. Writing the flux law out again here would be the classic
//! way for a solver to fit a model the data was not generated under.
//!
//! Zero counts are handled with the `+½` continuity correction: a ray that saw
//! nothing is consistent with a *range* of opacities, and its maximum-likelihood
//! value is infinite. The correction gives it the value where the model predicts
//! half a count — finite, and near the top of what the data supports.
//!
//! # The step sizes
//!
//! PDHG needs `στ‖K‖² < 1`, and `‖K‖` is the spectral norm of the stacked
//! `[L; ∇]`. It is estimated by power iteration rather than bounded by a norm
//! inequality: `‖L‖` alone can be `O(√rays)` larger on a limited-angle system,
//! and a step chosen from a loose bound is a step that crawls. With
//! `σ = τ = 0.99/‖K‖` the product is `0.98`, inside the guarantee with margin for
//! the estimate's own error.
//!
//! # The certificate
//!
//! The primal–dual gap is reported every step:
//!
//! ```text
//! P(ρ) = ½‖W^{1/2}(Lρ − X̂)‖² + β‖∇ρ‖_{2,1} + ι_box(ρ)
//! D(p,q) = −[½Σ p_i²/w_i + Σ p_i X̂_i] − σ_box(−(Lᵀp + ∇ᵀq))
//! gap = P(ρ) − D(p,q) ≥ 0
//! ```
//!
//! which is what makes "converged" mean something checkable rather than
//! "the iterate stopped moving".

use umbra_core::{Csr, GridAny};
use umbra_forward::expected::{ray_lambda, ForwardParams};
use umbra_forward::scene::SceneSpec;
use umbra_forward::RayCell;
use umbra_physics::{Fidelity, FluxTable};

use crate::{Control, Problem, Solver, Start, StepReport};

/// Tuning for [`OpacityPdhg`].
#[derive(Debug, Clone, PartialEq)]
pub struct A3Config {
    /// TV weight `β`, in the units of the weighted quadratic.
    pub beta: f32,
    /// Starting density, in g cm⁻³, for every voxel.
    pub rho_init: f32,
    /// Upper bound of the box, in g cm⁻³.
    pub rho_max: f32,
    /// Primal step. Zero selects `0.99 / ‖K‖`.
    pub tau: f32,
    /// Dual step. Zero selects `0.99 / ‖K‖`.
    pub sigma: f32,
    /// Over-relaxation. `1.0` is the full extrapolation of the paper.
    pub theta: f32,
    /// Convergence when the primal–dual gap is this fraction of the objective.
    pub tolerance: f32,
    /// Power iterations used to estimate `‖K‖`.
    pub power_iters: u32,
}

impl Default for A3Config {
    fn default() -> Self {
        Self {
            // The weighted quadratic is of order 10⁹ on the shared scene and a
            // single edge of TV of order 1, the same ratio A2 needs.
            beta: 1.0e3,
            rho_init: 2.65,
            rho_max: 4.0,
            tau: 0.0,
            sigma: 0.0,
            theta: 1.0,
            tolerance: 1.0e-3,
            power_iters: 40,
        }
    }
}

impl A3Config {
    /// The configuration a scene implies: its bulk density and a box that holds
    /// a dense inclusion.
    pub fn from_scene(scene: &SceneSpec) -> Self {
        Self {
            rho_init: scene.physics.rho_bulk,
            rho_max: (scene.physics.rho_bulk * 1.5).max(1.0),
            ..Self::default()
        }
    }
}

/// A ray's forward law, kept beside the geometry so the inversion can call it
/// per ray without rebuilding anything.
#[derive(Debug, Clone)]
struct RayLaw {
    rays: Vec<RayCell>,
    flux: FluxTable,
    fidelity: Fidelity,
    exposure_s: f32,
    params: ForwardParams,
}

impl RayLaw {
    /// `(λ, ∂λ/∂X)` for ray `ray` at opacity `x`.
    fn at(&self, ray: usize, x: f32) -> (f32, f32) {
        ray_lambda(
            &self.rays[ray],
            x,
            &self.flux,
            self.fidelity,
            self.exposure_s,
            &self.params,
        )
    }

    /// Opacity whose expected counts are `target`, by Newton inside a bisection
    /// bracket.
    ///
    /// # Panics
    ///
    /// Panics if the counts are not strictly positive: the correction for a zero
    /// count belongs to the caller, because it is a statement about the data
    /// rather than about the arithmetic.
    fn invert(&self, ray: usize, target: f32) -> f64 {
        assert!(
            target > 0.0,
            "the inversion needs a positive target: apply the continuity correction first"
        );
        let target = f64::from(target);
        let mut lo = 0.0f64;
        let mut hi = 100.0f64;
        // Grow the bracket until the model predicts fewer counts than observed.
        // `λ` falls monotonically, so this terminates unless the flux law has a
        // floor, which the ceiling of the integral guarantees.
        while f64::from(self.at(ray, hi as f32).0) > target {
            hi *= 2.0;
            if hi > 1.0e8 {
                return hi;
            }
        }
        let mut x = 0.5 * (lo + hi);
        for _ in 0..60 {
            let (lambda, slope) = self.at(ray, x as f32);
            let lambda = f64::from(lambda);
            let slope = f64::from(slope);
            if lambda > target {
                lo = x;
            } else {
                hi = x;
            }
            let newton = if slope < 0.0 && slope.is_finite() {
                x - (lambda - target) / slope
            } else {
                f64::NAN
            };
            // The safeguard: a Newton step outside the bracket is replaced by the
            // midpoint, which always shrinks it. This is what keeps the iteration
            // sane on a ray with zero counts, where the root is far out.
            let next = if newton.is_finite() && newton > lo && newton < hi {
                newton
            } else {
                0.5 * (lo + hi)
            };
            let moved = (next - x).abs();
            x = next;
            if moved <= 1.0e-6 * x.abs().max(1.0) {
                break;
            }
        }
        x
    }

    /// The opacity-domain observation and its delta-method weight.
    ///
    /// A zero count is corrected to `½`, and rays the model cannot see at all —
    /// zero acceptance — are given no weight, because data against a ray that
    /// collects nothing says nothing about the density.
    fn observe(&self, counts: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut opacity = Vec::with_capacity(counts.len());
        let mut weight = Vec::with_capacity(counts.len());
        for (ray, observed) in counts.iter().enumerate() {
            let (transparent, _) = self.at(ray, 0.0);
            if !transparent.is_finite() || transparent <= 0.0 {
                opacity.push(0.0);
                weight.push(0.0);
                continue;
            }
            let observed = f64::from(*observed).max(0.0);
            // A ray that saw nothing is consistent with a range of opacities and
            // its maximum-likelihood value is infinite, so it is given the value
            // where the model predicts half a count. Applied only to zeros: the
            // correction is worth half a count, which is noise against 10⁴ and a
            // bias against 1.
            let target = if observed > 0.0 { observed } else { 0.5 };
            let x = self.invert(ray, target as f32);
            let (lambda, slope) = self.at(ray, x as f32);
            let lambda = f64::from(lambda);
            let ray_weight = if lambda > 0.0 {
                f64::from(slope) * f64::from(slope) / lambda
            } else {
                0.0
            };
            opacity.push(x as f32);
            weight.push(ray_weight as f32);
        }
        (opacity, weight)
    }
}

/// Opacity-domain inversion followed by Chambolle–Pock PDHG on `K = [L; ∇]`.
#[derive(Debug, Clone)]
pub struct OpacityPdhg {
    l: Csr<f32>,
    nx: usize,
    ny: usize,
    /// Inverted per-ray opacity, in g cm⁻².
    x_hat: Vec<f32>,
    /// Delta-method weights, in counts per (g cm⁻²)².
    weight: Vec<f32>,
    rho: Vec<f32>,
    rho_prev: Vec<f32>,
    rho_bar: Vec<f32>,
    dual: Vec<f32>,
    dual_x: Vec<f32>,
    dual_y: Vec<f32>,
    beta: f64,
    rho_max: f32,
    tau: f64,
    sigma: f64,
    theta: f64,
    tolerance: f32,
    norm_k: f32,
    iteration: u32,
    converged: bool,
    gap: f64,
    gap_scale: f64,
    primal: f64,
}

impl OpacityPdhg {
    /// Builds the solver, starting from the scene's bulk density.
    pub fn new(problem: &Problem<'_>, cfg: A3Config) -> Self {
        Self::with_start(problem, cfg, &Start::default())
    }

    /// Builds the solver, beginning from `start.rho` when one is supplied.
    ///
    /// # Panics
    ///
    /// Panics if `start.rho` does not have one entry per voxel, if the TV weight
    /// is negative or the over-relaxation outside `(0, 1]`, or if the grid is not
    /// two-dimensional: `∇` here is a 2-D forward difference.
    pub fn with_start(problem: &Problem<'_>, cfg: A3Config, start: &Start<'_>) -> Self {
        assert!(
            cfg.beta.is_finite() && cfg.beta >= 0.0,
            "the TV weight must be non-negative"
        );
        assert!(
            cfg.theta > 0.0 && cfg.theta <= 1.0,
            "over-relaxation must lie in (0, 1]"
        );
        let (nx, ny) = match &problem.system.grid {
            GridAny::Two(grid) => (grid.n[0] as usize, grid.n[1] as usize),
            _ => panic!("A3's gradient is two-dimensional; a 3-D grid needs a 3-D one"),
        };
        let voxels = problem.voxel_count();
        assert_eq!(nx * ny, voxels, "the grid and the operator must agree");
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

        let law = RayLaw {
            rays: problem.system.rays.clone(),
            flux: problem.flux.clone(),
            fidelity: problem.fidelity,
            exposure_s: problem.exposure_s,
            params: problem.params,
        };
        let (x_hat, weight) = law.observe(problem.counts);
        let norm_k = power_norm(&problem.system.l, nx, ny, cfg.power_iters.max(1));

        // The steps come from the operator's own scale, so a caller does not have
        // to know the geometry to pick them.
        let auto = if norm_k > 0.0 {
            f64::from(0.99 / norm_k)
        } else {
            1.0
        };
        let tau = if cfg.tau > 0.0 {
            f64::from(cfg.tau)
        } else {
            auto
        };
        let sigma = if cfg.sigma > 0.0 {
            f64::from(cfg.sigma)
        } else {
            auto
        };

        let mut solver = Self {
            l: problem.system.l.clone(),
            nx,
            ny,
            x_hat,
            weight,
            rho: rho.clone(),
            rho_prev: rho.clone(),
            rho_bar: rho.clone(),
            dual: vec![0.0; problem.ray_count()],
            dual_x: vec![0.0; voxels],
            dual_y: vec![0.0; voxels],
            beta: f64::from(cfg.beta),
            rho_max: cfg.rho_max,
            tau,
            sigma,
            theta: f64::from(cfg.theta),
            tolerance: cfg.tolerance,
            norm_k,
            iteration: 0,
            converged: false,
            gap: f64::INFINITY,
            gap_scale: 1.0,
            primal: f64::INFINITY,
        };
        solver.primal = solver.primal_objective(&rho);
        solver.gap = solver.duality_gap(&rho);
        solver.gap_scale = solver.gap.abs().max(1.0);
        solver
    }

    /// `½‖W^{1/2}(Lρ − X̂)‖²`, the data term.
    pub fn data_term(&self, rho: &[f32]) -> f64 {
        let mut opacity = vec![0.0f32; self.l.nrows()];
        self.l.matvec(rho, &mut opacity);
        opacity
            .iter()
            .zip(&self.x_hat)
            .zip(&self.weight)
            .map(|((model, observed), weight)| {
                0.5 * f64::from(*weight) * {
                    let residual = f64::from(*model) - f64::from(*observed);
                    residual * residual
                }
            })
            .sum()
    }

    /// `‖∇ρ‖_{2,1}`, the isotropic total variation.
    pub fn tv(&self, rho: &[f32]) -> f64 {
        let (mut gx, mut gy) = (vec![0.0f32; rho.len()], vec![0.0f32; rho.len()]);
        gradient(rho, self.nx, self.ny, &mut gx, &mut gy);
        gx.iter()
            .zip(&gy)
            .map(|(a, b)| {
                let (a, b) = (f64::from(*a), f64::from(*b));
                (a * a + b * b).sqrt()
            })
            .sum()
    }

    /// `P(ρ)`, the primal objective.
    pub fn primal_objective(&self, rho: &[f32]) -> f64 {
        self.data_term(rho) + self.beta * self.tv(rho)
    }

    /// `P(ρ) − D(p, q)`, which is non-negative for any dual feasible `(p, q)`.
    pub fn duality_gap(&self, rho: &[f32]) -> f64 {
        let primal = self.primal_objective(rho);

        // F*(p) = ½Σ p_i²/w_i + Σ p_i X̂_i
        let mut conjugate_data = 0.0;
        for ((dual, weight), observed) in self.dual.iter().zip(&self.weight).zip(&self.x_hat) {
            let weight = f64::from(*weight);
            if weight <= 0.0 {
                continue;
            }
            let dual = f64::from(*dual);
            conjugate_data += 0.5 * dual * dual / weight + dual * f64::from(*observed);
        }

        // −Kᵀ(p, q), needed for the support function of the box.
        let mut stationarity = vec![0.0f32; rho.len()];
        self.l.matvec_t(&self.dual, &mut stationarity);
        gradient_t(
            &self.dual_x,
            &self.dual_y,
            self.nx,
            self.ny,
            &mut stationarity,
        );
        let support: f64 = stationarity
            .iter()
            .map(|value| f64::from(self.rho_max) * f64::from(*value).max(0.0))
            .sum();

        // The TV conjugate is zero for the projected dual, which is all this
        // solver produces, so only feasibility would be asserted here.
        let dual = -(conjugate_data + support);
        primal - dual
    }

    /// Spectral norm of `[L; ∇]` as estimated at construction.
    pub fn norm_k(&self) -> f32 {
        self.norm_k
    }

    /// The inverted opacity field, one entry per ray, in g cm⁻².
    pub fn opacity_estimate(&self) -> &[f32] {
        &self.x_hat
    }

    /// Delta-method weights, one entry per ray.
    pub fn weights(&self) -> &[f32] {
        &self.weight
    }

    /// Current primal–dual gap.
    pub fn gap(&self) -> f64 {
        self.gap
    }

    /// The gap relative to its value at the starting point.
    pub fn gap_ratio(&self) -> f64 {
        self.gap.abs() / self.gap_scale
    }

    /// Primal objective at the current estimate.
    pub fn primal(&self) -> f64 {
        self.primal
    }

    /// Completed iterations.
    pub fn iterations(&self) -> u32 {
        self.iteration
    }

    /// The step sizes in use.
    pub fn steps(&self) -> (f64, f64) {
        (self.tau, self.sigma)
    }
}

/// Forward difference on the two axes, one entry per voxel for each.
fn gradient(rho: &[f32], nx: usize, ny: usize, gx: &mut [f32], gy: &mut [f32]) {
    for j in 0..ny {
        for i in 0..nx {
            let here = i + nx * j;
            gx[here] = if i + 1 < nx {
                rho[here + 1] - rho[here]
            } else {
                0.0
            };
            gy[here] = if j + 1 < ny {
                rho[here + nx] - rho[here]
            } else {
                0.0
            };
        }
    }
}

/// The exact transpose of [`gradient`], added into `out`.
///
/// Written as an accumulation from each edge rather than by differentiating the
/// loop above: the adjoint of a forward difference is a backward difference with
/// a sign, and getting that sign wrong leaves a solver that still converges,
/// still reports a gap, and reconstructs a mirrored object. Accumulating the
/// edge's contribution into both endpoints makes the transpose true by
/// construction.
fn gradient_t(gx: &[f32], gy: &[f32], nx: usize, ny: usize, out: &mut [f32]) {
    for j in 0..ny {
        for i in 0..nx {
            let here = i + nx * j;
            if i + 1 < nx {
                let edge = gx[here];
                out[here] -= edge;
                out[here + 1] += edge;
            }
            if j + 1 < ny {
                let edge = gy[here];
                out[here] -= edge;
                out[here + nx] += edge;
            }
        }
    }
}

/// `‖[L; ∇]‖₂` by power iteration on `KᵀK`.
fn power_norm(l: &Csr<f32>, nx: usize, ny: usize, iters: u32) -> f32 {
    let voxels = nx * ny;
    if voxels == 0 {
        return 0.0;
    }
    // A smooth deterministic start, so the estimate is reproducible and not
    // accidentally orthogonal to the leading singular vector.
    let mut vector: Vec<f32> = (0..voxels)
        .map(|voxel| {
            let i = (voxel % nx) as f32;
            let j = (voxel / nx) as f32;
            1.0 + (0.1 * i).sin() * (0.1 * j).cos()
        })
        .collect();

    let mut rayleigh = 0.0f64;
    for _ in 0..iters {
        let mut forward = vec![0.0f32; l.nrows()];
        l.matvec(&vector, &mut forward);
        let (mut gx, mut gy) = (vec![0.0f32; voxels], vec![0.0f32; voxels]);
        gradient(&vector, nx, ny, &mut gx, &mut gy);

        let mut back = vec![0.0f32; voxels];
        l.matvec_t(&forward, &mut back);
        gradient_t(&gx, &gy, nx, ny, &mut back);

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
    rayleigh.max(0.0).sqrt() as f32
}

impl Solver for OpacityPdhg {
    type Config = A3Config;

    fn init(problem: &Problem<'_>, cfg: &Self::Config, start: &Start<'_>) -> Self {
        Self::with_start(problem, cfg.clone(), start)
    }

    fn step(&mut self, ctl: &Control) -> StepReport {
        if self.converged || ctl.should_stop() || self.iteration >= ctl.max_iters {
            return StepReport {
                iteration: self.iteration,
                objective: Some(self.primal),
                gap: Some(self.gap),
                converged: self.converged,
            };
        }

        // Primal: descend on Kᵀ(p, q) and project onto the box.
        let mut stationarity = vec![0.0f32; self.rho.len()];
        self.l.matvec_t(&self.dual, &mut stationarity);
        gradient_t(
            &self.dual_x,
            &self.dual_y,
            self.nx,
            self.ny,
            &mut stationarity,
        );
        self.rho_prev = std::mem::take(&mut self.rho);
        self.rho = self
            .rho_prev
            .iter()
            .zip(&stationarity)
            .map(|(rho, gradient)| {
                let value = f64::from(*rho) - self.tau * f64::from(*gradient);
                value.clamp(0.0, f64::from(self.rho_max)) as f32
            })
            .collect();

        // Extrapolation. `θ = 1` is the full step of the paper.
        self.rho_bar = self
            .rho
            .iter()
            .zip(&self.rho_prev)
            .map(|(now, before)| {
                (f64::from(*now) + self.theta * (f64::from(*now) - f64::from(*before))) as f32
            })
            .collect();

        // Dual, data block: prox of the conjugate quadratic, elementwise.
        let mut predicted = vec![0.0f32; self.l.nrows()];
        self.l.matvec(&self.rho_bar, &mut predicted);
        for ((dual, model), (observed, weight)) in self
            .dual
            .iter_mut()
            .zip(&predicted)
            .zip(self.x_hat.iter().zip(&self.weight))
        {
            let weight = f64::from(*weight);
            if weight <= 0.0 {
                // No information: the dual of a ray with zero weight is pinned
                // at zero, since its conjugate is infinite everywhere else.
                *dual = 0.0;
                continue;
            }
            let value = f64::from(*dual) + self.sigma * f64::from(*model)
                - self.sigma * f64::from(*observed);
            *dual = (value * weight / (weight + self.sigma)) as f32;
        }

        // Dual, TV block: ascend, then project each voxel's pair onto the ball
        // of radius β, which is the conjugate's prox for `‖·‖_{2,1}`.
        let mut gx = vec![0.0f32; self.rho.len()];
        let mut gy = vec![0.0f32; self.rho.len()];
        gradient(&self.rho_bar, self.nx, self.ny, &mut gx, &mut gy);
        for voxel in 0..self.rho.len() {
            let x = f64::from(self.dual_x[voxel]) + self.sigma * f64::from(gx[voxel]);
            let y = f64::from(self.dual_y[voxel]) + self.sigma * f64::from(gy[voxel]);
            let size = (x * x + y * y).sqrt();
            let factor = if size > self.beta && size > 0.0 {
                self.beta / size
            } else {
                1.0
            };
            self.dual_x[voxel] = (x * factor) as f32;
            self.dual_y[voxel] = (y * factor) as f32;
        }

        self.primal = self.primal_objective(&self.rho);
        self.gap = self.duality_gap(&self.rho);
        if self.iteration == 0 {
            self.gap_scale = self.gap.abs().max(f64::MIN_POSITIVE);
        }
        self.iteration += 1;
        self.converged = self.gap_ratio() <= f64::from(self.tolerance);

        StepReport {
            iteration: self.iteration,
            objective: Some(self.primal),
            gap: Some(self.gap),
            converged: self.converged,
        }
    }

    fn estimate(&self) -> &[f32] {
        &self.rho
    }
}
