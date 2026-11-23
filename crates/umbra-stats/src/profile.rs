//! A6 nuisance parameters (spec §5.6): what a significance is worth when the
//! things we do not know are allowed to move.
//!
//! Bulk density, flux normalisation, per-detector efficiency and small alignment
//! offsets are uncertain, and a significance that ignores them is not a
//! significance. Each carries a Gaussian prior, and the test is the **profile
//! likelihood ratio**: the nuisances are refitted under `H0` and under `H1`, and
//! the statistic is twice the difference of the penalised log-likelihoods.
//!
//! # The model
//!
//! The scan's `L0` law is `λ_i = k_i e^{−X_i/Λ}`. Profiling perturbs it:
//!
//! ```text
//! X_i(θ) = κ_ρ (X_i⁰ − α T_i) + δ_x Gx_i + δ_y Gy_i
//! λ_i(θ) = κ_Φ κ_{ε,d(i)} k_i e^{−X_i(θ)/Λ}
//! ```
//!
//! `T = L(ρ ⊙ t)` is the opacity the template would remove, `X⁰ = Lρ`, and
//! `Gx`, `Gy` are the path integrals of the density gradient — the first-order
//! response of the opacity to a translation of the model relative to the data.
//! That keeps alignment inside the same linear response as everything else
//! instead of re-voxelising the scene for every trial offset.
//!
//! # Why Newton, and why analytic
//!
//! Six parameters is small enough that a damped Newton step solves a dense
//! system directly and large enough that finite-differencing the objective
//! would cost more than the whole fit. Every derivative is written out, and a
//! test compares each one against a central difference of the objective it
//! claims to differentiate — a numerical-gradient profiler would pass every
//! other test in this module while quietly being the slow thing it replaced.

use umbra_core::GridAny;
use umbra_forward::System;

use crate::scan::{chi2_mixture_p, ScanProblem, TemplateBank};

/// Gaussian priors on the nuisance parameters.
///
/// A prior is a statement about how well a quantity is known *before* the data,
/// and it is what keeps an unconstrained nuisance at its mean instead of letting
/// it wander off to absorb the signal. `alignment_cm` is in centimetres because
/// the offsets it bounds are centimetres, not voxels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NuisancePrior {
    /// Bulk-density scale `κ_ρ`, dimensionless, mean 1.
    pub density: f64,
    /// Flux normalisation `κ_Φ`, dimensionless, mean 1.
    pub flux: f64,
    /// Per-detector efficiency `κ_ε`, dimensionless, mean 1.
    pub efficiency: f64,
    /// Alignment offsets `δ_x`, `δ_y`, in cm, mean 0.
    pub alignment_cm: f64,
}

impl Default for NuisancePrior {
    fn default() -> Self {
        // Percent-level on the bulk scale and the flux, a few percent on a
        // detector's efficiency, and half a centimetre on alignment: the widths
        // the spec's "Systematics" panel starts from.
        Self {
            density: 0.05,
            flux: 0.03,
            efficiency: 0.02,
            alignment_cm: 0.5,
        }
    }
}

/// How hard the fit works.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProfilerConfig {
    /// Priors on the nuisances.
    pub prior: NuisancePrior,
    /// Cap on Newton steps. The default is well inside the spec's "short
    /// iteration"; the cap exists so a pathological scene degrades to "not
    /// converged" instead of spinning.
    pub max_steps: usize,
    /// Convergence threshold on the largest gradient component.
    pub tolerance: f64,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            prior: NuisancePrior::default(),
            max_steps: 40,
            tolerance: 1.0e-9,
        }
    }
}

/// One fitted point: the amplitude, the nuisances, and how the fit went.
#[derive(Debug, Clone, PartialEq)]
pub struct NuisanceFit {
    /// Best-fit amplitude.
    pub alpha: f64,
    /// Bulk-density scale, mean 1.
    pub density_scale: f64,
    /// Flux normalisation, mean 1.
    pub flux_scale: f64,
    /// Per-detector efficiency, mean 1, one entry per detector present.
    pub efficiency: Vec<f64>,
    /// Alignment offsets in cm, mean 0.
    pub alignment_cm: [f64; 2],
    /// Poisson log-likelihood at the fitted point.
    pub log_likelihood: f64,
    /// Log prior at the fitted point.
    pub log_prior: f64,
    /// Newton steps taken.
    pub steps: usize,
    /// Whether the gradient fell below the tolerance.
    pub converged: bool,
    /// Largest gradient component at the end, for callers that want to know how
    /// close "converged" was.
    pub gradient_norm: f64,
}

impl NuisanceFit {
    /// The penalised objective `−(ln L + ln π)` the fit minimises.
    pub fn objective(&self) -> f64 {
        -(self.log_likelihood + self.log_prior)
    }
}

/// The profiled test of one candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateProfile {
    /// `H0`: no void, nuisances fitted.
    pub h0: NuisanceFit,
    /// `H1`: void amplitude free, nuisances fitted.
    pub h1: NuisanceFit,
    /// Profile likelihood ratio, `2(ln L_H1 − ln L_H0)` on the penalised
    /// log-likelihoods.
    pub ts: f64,
    /// `√TS`, the significance with systematics profiled.
    pub z: f64,
    /// `√TS` with the nuisances held at their prior means — what the scan
    /// reported before systematics existed.
    pub z_without_systematics: f64,
    /// How much significance the systematics cost.
    pub drop: f32,
}

/// Per-candidate significance before and after profiling, in scan order.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SystematicsReport {
    /// `Z` with nuisances frozen at their prior means.
    pub without: Vec<f32>,
    /// `Z` with nuisances profiled.
    pub with: Vec<f32>,
    /// Largest `without − with`.
    pub max_drop: f32,
}

/// Which parameters the Newton step is allowed to move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// `H1`: amplitude and every nuisance free.
    Hypotheses,
    /// `H0`: amplitude pinned at zero, nuisances free.
    Null,
    /// Amplitude free, nuisances frozen at their prior means.
    AlphaOnly,
}

/// Parameter layout: `[α, κ_ρ, κ_Φ, κ_ε(d)…, δ_x, δ_y]`.
#[derive(Debug, Clone, Copy)]
struct Layout {
    detectors: usize,
}

impl Layout {
    /// `α`, `κ_ρ`, `κ_Φ`, one `κ_ε` per detector, `δ_x`, `δ_y`. Six with a
    /// single detector, which is the spec's bound.
    fn dims(&self) -> usize {
        5 + self.detectors
    }
    fn density(&self) -> usize {
        1
    }
    fn flux(&self) -> usize {
        2
    }
    fn efficiency(&self, detector: usize) -> usize {
        3 + detector
    }
    fn dx(&self) -> usize {
        3 + self.detectors
    }
    fn dy(&self) -> usize {
        4 + self.detectors
    }
    fn mean(&self, index: usize) -> f64 {
        if index == 0 || index == self.dx() || index == self.dy() {
            0.0
        } else {
            1.0
        }
    }
    fn sigma(&self, index: usize, prior: &NuisancePrior) -> Option<f64> {
        if index == 0 {
            return None; // the amplitude is the parameter of interest
        }
        Some(if index == self.density() {
            prior.density
        } else if index == self.flux() {
            prior.flux
        } else if index == self.dx() || index == self.dy() {
            prior.alignment_cm
        } else {
            prior.efficiency
        })
    }
    fn free(&self, mode: Mode) -> Vec<usize> {
        match mode {
            Mode::Hypotheses => (0..self.dims()).collect(),
            Mode::Null => (1..self.dims()).collect(),
            Mode::AlphaOnly => vec![0],
        }
    }
}

/// The per-candidate linear model the fit is run against.
struct LocalModel<'a> {
    problem: &'a ScanProblem<'a>,
    layout: Layout,
    /// `k_i` in `λ = κ_Φ κ_ε k e^{−X/Λ}`.
    k: Vec<f64>,
    /// Opacity of the reference structure, `X⁰`.
    x0: Vec<f64>,
    /// Opacity the template removes, `T`.
    template: Vec<f64>,
    /// Path integral of `∂ρ/∂x` and `∂ρ/∂y`.
    gradient: [Vec<f64>; 2],
    /// Counts as `f64`.
    counts: Vec<f64>,
    attenuation: f64,
    bounds: [f64; 2],
    config: ProfilerConfig,
}

impl LocalModel<'_> {
    fn evaluate(&self, theta: &[f64]) -> Option<(f64, Vec<f64>, Vec<f64>)> {
        let layout = self.layout;
        let dims = layout.dims();
        let alpha = theta[0];
        let density = theta[layout.density()];
        let flux = theta[layout.flux()];
        let dx = theta[layout.dx()];
        let dy = theta[layout.dy()];
        let attenuation = self.attenuation;

        let mut log_likelihood = 0.0;
        let mut grad = vec![0.0f64; dims];
        let mut hess = vec![0.0f64; dims * dims];

        for i in 0..self.counts.len() {
            let detector = self.problem.system.rays[i].det as usize;
            let efficiency = theta[layout.efficiency(detector)];
            if flux <= 0.0 || efficiency <= 0.0 || !flux.is_finite() || !efficiency.is_finite() {
                return None;
            }
            let x = density * (self.x0[i] - alpha * self.template[i])
                + dx * self.gradient[0][i]
                + dy * self.gradient[1][i];
            let lambda = flux * efficiency * self.k[i] * (-x / attenuation).exp();
            if !lambda.is_finite() || lambda <= 0.0 {
                return None;
            }
            let y = self.counts[i];
            log_likelihood += y * lambda.ln() - lambda;

            // ∂λ/∂θ for each parameter, then dF = −(y/λ − 1)∂λ/∂θ.
            let mut d_lambda = vec![0.0f64; dims];
            d_lambda[0] = lambda * density * self.template[i] / attenuation;
            d_lambda[layout.density()] =
                -lambda * (self.x0[i] - alpha * self.template[i]) / attenuation;
            d_lambda[layout.flux()] = lambda / flux;
            d_lambda[layout.efficiency(detector)] = lambda / efficiency;
            d_lambda[layout.dx()] = -lambda * self.gradient[0][i] / attenuation;
            d_lambda[layout.dy()] = -lambda * self.gradient[1][i] / attenuation;

            let residual = y / lambda - 1.0;
            for p in 0..dims {
                if d_lambda[p] != 0.0 {
                    grad[p] -= residual * d_lambda[p];
                }
            }
            // Gauss–Newton curvature: Σ (y/λ²) ∂λ_p ∂λ_q. The omitted second
            // derivative of λ is what makes this Fisher scoring rather than full
            // Newton, and it is why the step needs damping.
            let weight = y / (lambda * lambda);
            for p in 0..dims {
                if d_lambda[p] == 0.0 {
                    continue;
                }
                for q in p..dims {
                    if d_lambda[q] == 0.0 {
                        continue;
                    }
                    let value = weight * d_lambda[p] * d_lambda[q];
                    hess[p * dims + q] += value;
                    if p != q {
                        hess[q * dims + p] += value;
                    }
                }
            }
        }

        // Penalty: −ln π = Σ (θ − μ)²/(2σ²). Its gradient is (θ − μ)/σ² and its
        // curvature 1/σ², both diagonal.
        let mut log_prior = 0.0;
        for index in 0..dims {
            let Some(sigma) = layout.sigma(index, &self.config.prior) else {
                continue;
            };
            let scaled = (theta[index] - layout.mean(index)) / sigma;
            log_prior -= 0.5 * scaled * scaled;
            grad[index] += scaled / sigma;
            hess[index * dims + index] += 1.0 / (sigma * sigma);
        }

        Some((-(log_likelihood + log_prior), grad, hess))
    }

    fn fit(&self, mode: Mode, start: &[f64]) -> Option<NuisanceFit> {
        let layout = self.layout;
        let dims = layout.dims();
        let free = layout.free(mode);
        let (mut theta, mut best) = {
            let (objective, grad, hess) = self.evaluate(start)?;
            (start.to_vec(), (objective, grad, hess))
        };
        let mut damping = 1.0e-6;
        let mut steps = 0;
        let mut converged = false;
        let mut last_improvement = f64::INFINITY;

        while steps < self.config.max_steps {
            let norm = free
                .iter()
                .map(|&index| best.1[index].abs())
                .fold(0.0f64, f64::max);
            if norm < self.config.tolerance {
                converged = true;
                break;
            }
            steps += 1;
            let mut accepted = false;
            for _ in 0..40 {
                let step = solve_damped(&best.2, &best.1, &free, dims, damping)?;
                let mut trial = theta.clone();
                for (slot, index) in free.iter().enumerate() {
                    trial[*index] += step[slot];
                }
                // The amplitude lives in a bounded range, and the two scales are
                // positive by construction: project the trial point rather than
                // letting a wild step produce a negative rate.
                trial[0] = trial[0].clamp(self.bounds[0], self.bounds[1]);
                trial[layout.density()] = trial[layout.density()].max(1.0e-3);
                trial[layout.flux()] = trial[layout.flux()].max(1.0e-3);
                for detector in 0..layout.detectors {
                    let index = layout.efficiency(detector);
                    trial[index] = trial[index].max(1.0e-3);
                }
                trial[layout.dx()] = trial[layout.dx()].clamp(-50.0, 50.0);
                trial[layout.dy()] = trial[layout.dy()].clamp(-50.0, 50.0);

                if let Some(candidate) = self.evaluate(&trial) {
                    if candidate.0 < best.0 {
                        last_improvement = best.0 - candidate.0;
                        theta = trial;
                        best = candidate;
                        damping = (damping * 0.2).max(1.0e-12);
                        accepted = true;
                        break;
                    }
                }
                damping *= 10.0;
            }
            if !accepted {
                break;
            }
        }

        let gradient_norm = free
            .iter()
            .map(|&index| best.1[index].abs())
            .fold(0.0f64, f64::max);
        // A fit has converged when it can no longer improve the objective *to
        // within the precision of the objective itself*. Waiting for a tiny
        // gradient instead never fires: `F` is of order `Σ y ≈ 10⁸` here, so an
        // improvement below `1e-9·F` is below the resolution of the `f64` it is
        // stored in, and the strict-decrease test stops accepting steps long
        // before the gradient is small in absolute terms.
        let converged = converged
            || gradient_norm < self.config.tolerance
            || last_improvement < self.config.tolerance * (1.0 + best.0.abs());
        let mut efficiency = Vec::with_capacity(layout.detectors);
        for detector in 0..layout.detectors {
            efficiency.push(theta[layout.efficiency(detector)]);
        }
        Some(NuisanceFit {
            alpha: theta[0],
            density_scale: theta[layout.density()],
            flux_scale: theta[layout.flux()],
            efficiency,
            alignment_cm: [theta[layout.dx()], theta[layout.dy()]],
            log_likelihood: -best.0 - log_prior_of(&theta, &layout, &self.config),
            log_prior: log_prior_of(&theta, &layout, &self.config),
            steps,
            converged,
            gradient_norm,
        })
    }
}

fn log_prior_of(theta: &[f64], layout: &Layout, config: &ProfilerConfig) -> f64 {
    let mut log_prior = 0.0;
    for (index, &value) in theta.iter().enumerate().take(layout.dims()) {
        let Some(sigma) = layout.sigma(index, &config.prior) else {
            continue;
        };
        let scaled = (value - layout.mean(index)) / sigma;
        log_prior -= 0.5 * scaled * scaled;
    }
    log_prior
}

/// Solves `(H + μ·diag(H)) d = −g` restricted to `free`, by Gaussian
/// elimination with partial pivoting.
///
/// The damping is Levenberg's, scaled by the diagonal so a parameter whose
/// curvature differs in magnitude from the others is damped in its own units.
fn solve_damped(
    hess: &[f64],
    grad: &[f64],
    free: &[usize],
    dims: usize,
    damping: f64,
) -> Option<Vec<f64>> {
    let n = free.len();
    let mut a = vec![0.0f64; n * n];
    let mut b = vec![0.0f64; n];
    for (row, &p) in free.iter().enumerate() {
        for (column, &q) in free.iter().enumerate() {
            a[row * n + column] = hess[p * dims + q];
        }
        let diagonal = hess[p * dims + p].abs().max(1.0e-12);
        a[row * n + row] += damping * diagonal;
        b[row] = -grad[p];
    }

    for column in 0..n {
        let mut pivot = column;
        for row in column + 1..n {
            if a[row * n + column].abs() > a[pivot * n + column].abs() {
                pivot = row;
            }
        }
        if a[pivot * n + column].abs() < 1.0e-300 {
            return None;
        }
        if pivot != column {
            for k in 0..n {
                a.swap(column * n + k, pivot * n + k);
            }
            b.swap(column, pivot);
        }
        let d = a[column * n + column];
        for k in column..n {
            a[column * n + k] /= d;
        }
        b[column] /= d;
        for row in 0..n {
            if row == column {
                continue;
            }
            let factor = a[row * n + column];
            if factor == 0.0 {
                continue;
            }
            for k in column..n {
                a[row * n + k] -= factor * a[column * n + k];
            }
            b[row] -= factor * b[column];
        }
    }
    Some(b)
}

/// Profiles one candidate's amplitude and nuisances under `H0` and `H1`.
///
/// Returns `None` when the candidate has no support — no template opacity, or a
/// rate that is not positive and finite — because a profile likelihood ratio
/// there is `0/0`, not a significance.
///
/// # Panics
///
/// Panics if the attenuation length is not positive, if the problem's arrays
/// disagree with the system's dimensions, or if the system has a detector index
/// beyond the detectors it declares.
pub fn profile_candidate(
    problem: &ScanProblem<'_>,
    bank: &TemplateBank,
    scale: usize,
    voxel: usize,
    config: &ProfilerConfig,
) -> Option<CandidateProfile> {
    let attenuation = f64::from(problem.attenuation_gcm2);
    assert!(
        attenuation > 0.0 && attenuation.is_finite(),
        "the L0 attenuation length must be positive"
    );
    assert_eq!(
        problem.counts.len(),
        problem.system.ray_count(),
        "one count per ray cell"
    );
    assert_eq!(
        problem.density.len(),
        problem.system.voxel_count(),
        "density must cover every voxel"
    );
    if problem.system.rays.is_empty() {
        return None;
    }

    let detectors = problem
        .system
        .rays
        .iter()
        .map(|ray| ray.det as usize + 1)
        .max()
        .unwrap_or(1);
    let layout = Layout { detectors };
    let model = local_model(problem, bank, scale, voxel, layout, attenuation, config)?;

    let dims = layout.dims();
    let mut start = vec![0.0f64; dims];
    for (index, slot) in start.iter_mut().enumerate() {
        *slot = layout.mean(index);
    }

    let h0 = model.fit(Mode::Null, &start)?;
    let mut warm = start.clone();
    warm[0] = (h0.alpha + 0.05).clamp(model.bounds[0], model.bounds[1]);
    let h1 = model.fit(Mode::Hypotheses, &warm)?;
    let alpha_only = model.fit(Mode::AlphaOnly, &start)?;

    // `objective()` is `−(ln L + ln π)`, so the better fit has the smaller
    // objective and the ratio is `2 (F_H0 − F_H1)`.
    let ts = (2.0 * (h0.objective() - h1.objective())).max(0.0);
    let ts_without = (2.0
        * ({
            // `AlphaOnly` starts at the prior means with α = 0, which is exactly
            // `H0` with the nuisances frozen.
            let mut null = start.clone();
            null[0] = 0.0;
            model.evaluate(&null)?.0
        } - alpha_only.objective()))
    .max(0.0);

    let z = ts.sqrt();
    let z_without = ts_without.sqrt();
    Some(CandidateProfile {
        h0,
        h1,
        ts,
        z,
        z_without_systematics: z_without,
        drop: (z_without - z) as f32,
    })
}

fn local_model<'a>(
    problem: &'a ScanProblem<'a>,
    bank: &TemplateBank,
    scale: usize,
    voxel: usize,
    layout: Layout,
    attenuation: f64,
    config: &ProfilerConfig,
) -> Option<LocalModel<'a>> {
    let system: &System = problem.system;
    let rays = system.ray_count();
    let x0 = system.opacity(problem.density);
    let k: Vec<f64> = problem
        .lambda0
        .iter()
        .zip(&x0)
        .map(|(&lambda, &x)| f64::from(lambda) * (f64::from(x) / attenuation).exp())
        .collect();

    // The template's opacity, and the density gradients alignment responds to.
    let (dims, dims_grid) = grid_dims(system);
    let nx = dims_grid[0];
    let ny = dims_grid[1];
    let mut weighted = vec![0.0f32; problem.density.len()];
    let mut any = false;
    let centre = [voxel % nx, (voxel / nx) % ny, voxel / (nx * ny)];
    for &(dx, dy, dz, weight) in bank.offsets(scale) {
        let x = centre[0] as i64 + i64::from(dx);
        let y = centre[1] as i64 + i64::from(dy);
        let z = centre[2] as i64 + i64::from(dz);
        if x < 0 || y < 0 || z < 0 {
            continue;
        }
        let (x, y, z) = (x as usize, y as usize, z as usize);
        if x >= dims[0] || y >= dims[1] || z >= dims[2] {
            continue;
        }
        let index = (z * ny + y) * nx + x;
        let removed = problem.density[index] * weight;
        if removed != 0.0 {
            weighted[index] = removed;
            any = true;
        }
    }
    if !any {
        return None;
    }
    let mut template = vec![0.0f32; rays];
    system.l.matvec(&weighted, &mut template);
    if template.iter().all(|&value| value == 0.0) {
        return None;
    }

    let (gradient_x, gradient_y) = density_gradient(system, problem.density);
    let mut gradient = [vec![0.0f32; rays], vec![0.0f32; rays]];
    system.l.matvec(&gradient_x, &mut gradient[0]);
    system.l.matvec(&gradient_y, &mut gradient[1]);

    Some(LocalModel {
        problem,
        layout,
        k,
        x0: x0.iter().map(|&value| f64::from(value)).collect(),
        template: template.iter().map(|&value| f64::from(value)).collect(),
        gradient: [
            gradient[0].iter().map(|&value| f64::from(value)).collect(),
            gradient[1].iter().map(|&value| f64::from(value)).collect(),
        ],
        counts: problem
            .counts
            .iter()
            .map(|&value| f64::from(value))
            .collect(),
        attenuation,
        // The spec's amplitude range, which is also what `refine_lrt` fits in.
        bounds: [0.0, 1.0],
        config: *config,
    })
}

fn grid_dims(system: &System) -> ([usize; 3], [usize; 2]) {
    match &system.grid {
        GridAny::Two(grid) => (
            [grid.n[0] as usize, grid.n[1] as usize, 1],
            [grid.n[0] as usize, grid.n[1] as usize],
        ),
        GridAny::Three(grid) => (
            [grid.n[0] as usize, grid.n[1] as usize, grid.n[2] as usize],
            [grid.n[0] as usize, grid.n[1] as usize],
        ),
    }
}

/// Central differences of the density field, with one-sided stencils at the
/// edges, in `g cm⁻³` per cm.
fn density_gradient(system: &System, density: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let (dims, _) = grid_dims(system);
    let (nx, ny) = (dims[0], dims[1]);
    let spacing = match &system.grid {
        GridAny::Two(grid) => {
            let h = if grid.n[0] > 1 {
                (grid.centre(1, 0)[0] - grid.centre(0, 0)[0]).abs()
            } else {
                1.0
            };
            f64::from(h)
        }
        GridAny::Three(grid) => {
            let h = if grid.n[0] > 1 {
                (grid.centre(1, 0, 0)[0] - grid.centre(0, 0, 0)[0]).abs()
            } else {
                1.0
            };
            f64::from(h)
        }
    };
    let mut gx = vec![0.0f32; density.len()];
    let mut gy = vec![0.0f32; density.len()];
    if spacing <= 0.0 || !spacing.is_finite() {
        return (gx, gy);
    }
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                let index = (z * ny + y) * nx + x;
                let (left, right) = (x.saturating_sub(1), (x + 1).min(nx - 1));
                if right > left {
                    let span = (right - left) as f64 * spacing;
                    let a = density[(z * ny + y) * nx + left];
                    let b = density[(z * ny + y) * nx + right];
                    gx[index] = ((f64::from(b) - f64::from(a)) / span) as f32;
                }
                let (down, up) = (y.saturating_sub(1), (y + 1).min(ny - 1));
                if up > down {
                    let span = (up - down) as f64 * spacing;
                    let a = density[(z * ny + down) * nx + x];
                    let b = density[(z * ny + up) * nx + x];
                    gy[index] = ((f64::from(b) - f64::from(a)) / span) as f32;
                }
            }
        }
    }
    (gx, gy)
}

/// The mixture `p`-value for a profiled candidate.
///
/// Re-exported from the scan so a caller profiling one candidate by hand gets
/// the same tail every other path uses.
pub fn profile_p(ts: f64) -> f32 {
    chi2_mixture_p(ts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use umbra_core::{Csr, Grid2, GridAny};
    use umbra_forward::RayCell;

    const N: usize = 12;
    /// Short enough that `X/Λ` is of order one: at `1e3 g cm⁻²` a 2% density
    /// change moves the rate by 0.06%, which no fit can separate from the flux.
    const ATTENUATION: f64 = 30.0;

    /// A 12 × 12 slice with one ray per row and a buried low-density block.
    fn scene() -> (System, Vec<f32>, Vec<f32>, Vec<f32>) {
        let grid = Grid2::new([N as u32, N as u32], [0.0, 0.0], 0.01);
        let mut indptr = vec![0u32];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for row in 0..N {
            for column in 0..N {
                indices.push((row * N + column) as u32);
                data.push(1.0f32);
            }
            indptr.push(indices.len() as u32);
        }
        let mut density = vec![2.65f32; N * N];
        for y in 5..8 {
            for x in 5..8 {
                density[y * N + x] = 1.0;
            }
        }
        let system = System {
            l: Csr::new(indptr, indices, data),
            // Two detectors, so a per-detector efficiency is identifiable
            // separately from the shared flux: with one detector the two are a
            // product and the data cannot tell them apart.
            rays: (0..N)
                .map(|row| RayCell {
                    det: if row < N / 2 { 0 } else { 1 },
                    zen: 0.0,
                    az: 0.0,
                    k_scale: 1.0,
                })
                .collect(),
            grid: GridAny::Two(grid),
        };
        let opacity = system.opacity(&density);
        // Six-figure counts, so the data dominate the priors and a recovered
        // nuisance is the likelihood's answer rather than the prior's.
        let lambda0: Vec<f32> = opacity
            .iter()
            .map(|&x| 1.0e6 * (-f64::from(x) / ATTENUATION).exp() as f32)
            .collect();
        let dlambda_dx: Vec<f32> = lambda0.iter().map(|&l| -l / ATTENUATION as f32).collect();
        (system, density, lambda0, dlambda_dx)
    }

    fn problem<'a>(
        system: &'a System,
        counts: &'a [f32],
        lambda0: &'a [f32],
        dlambda_dx: &'a [f32],
        density: &'a [f32],
    ) -> ScanProblem<'a> {
        ScanProblem {
            system,
            counts,
            lambda0,
            dlambda_dx,
            density,
            attenuation_gcm2: ATTENUATION as f32,
        }
    }

    /// The rates the model predicts for a parameter vector.
    fn rates(model: &LocalModel<'_>, theta: &[f64]) -> Vec<f32> {
        let layout = model.layout;
        (0..model.counts.len())
            .map(|i| {
                let detector = model.problem.system.rays[i].det as usize;
                let x = theta[layout.density()] * (model.x0[i] - theta[0] * model.template[i])
                    + theta[layout.dx()] * model.gradient[0][i]
                    + theta[layout.dy()] * model.gradient[1][i];
                (theta[layout.flux()]
                    * theta[layout.efficiency(detector)]
                    * model.k[i]
                    * (-x / model.attenuation).exp()) as f32
            })
            .collect()
    }

    fn centre() -> usize {
        6 * N + 6
    }

    #[test]
    fn every_analytic_derivative_matches_a_central_difference() {
        let (system, density, lambda0, dlambda_dx) = scene();
        let counts = lambda0.clone();
        let p = problem(&system, &counts, &lambda0, &dlambda_dx, &density);
        let bank = TemplateBank::new(&system.grid, &[1.0]);
        let config = ProfilerConfig::default();
        let model = local_model(
            &p,
            &bank,
            0,
            centre(),
            Layout { detectors: 2 },
            ATTENUATION,
            &config,
        )
        .expect("the template has support");
        let theta = [0.3, 1.01, 1.02, 0.99, 1.01, 0.2, -0.1];
        let (_, grad, _) = model.evaluate(&theta).expect("valid point");
        assert_eq!(grad.len(), theta.len());

        for index in 0..theta.len() {
            // Large enough that the difference of two objectives of order `1e8`
            // is not dominated by the rounding of the objectives themselves.
            let h = 1.0e-3 * theta[index].abs().max(1.0);
            let mut up = theta;
            up[index] += h;
            let mut down = theta;
            down[index] -= h;
            let (above, _, _) = model.evaluate(&up).expect("valid point");
            let (below, _, _) = model.evaluate(&down).expect("valid point");
            let numeric = (above - below) / (2.0 * h);
            // A wrong derivative is off by percent, not by 1e-5: the slack here
            // covers the truncation error of the central difference without
            // letting a sign or a missing factor through.
            let tolerance = 1.0e-4 * numeric.abs().max(1.0);
            assert!(
                (numeric - grad[index]).abs() < tolerance,
                "parameter {index}: analytic {} vs numeric {numeric}",
                grad[index]
            );
        }
    }

    #[test]
    fn injected_nuisances_are_recovered_to_one_percent() {
        let (system, density, lambda0, dlambda_dx) = scene();
        let bank = TemplateBank::new(&system.grid, &[1.0]);
        let config = ProfilerConfig::default();
        let layout = Layout { detectors: 2 };

        let base = problem(&system, &lambda0, &lambda0, &dlambda_dx, &density);
        let model = local_model(&base, &bank, 0, centre(), layout, ATTENUATION, &config)
            .expect("the template has support");
        // Bulk density high by 2%, flux high by 1%, the two efficiencies off in
        // opposite directions by 2%.
        let truth = [0.0, 1.02, 1.01, 0.98, 1.02, 0.0, 0.0];
        let counts = rates(&model, &truth);

        let p = problem(&system, &counts, &lambda0, &dlambda_dx, &density);
        let profile = profile_candidate(&p, &bank, 0, centre(), &config).expect("a profile");
        let fitted = &profile.h1;

        assert!(
            (fitted.density_scale - 1.02).abs() < 0.0102,
            "density scale {}",
            fitted.density_scale
        );
        assert!(
            (fitted.flux_scale - 1.01).abs() < 0.0101,
            "flux scale {}",
            fitted.flux_scale
        );
        assert!(
            (fitted.efficiency[0] - 0.98).abs() < 0.0098,
            "efficiency[0] {}",
            fitted.efficiency[0]
        );
        assert!(
            (fitted.efficiency[1] - 1.02).abs() < 0.0102,
            "efficiency[1] {}",
            fitted.efficiency[1]
        );
        // The amplitude is not orthogonal to the bulk scale: raising `κ_ρ`
        // everywhere looks a little like a void under the template, so a percent
        // or so of amplitude leaks into it. That is the fit trading two
        // parameters against each other, not a failure to recover the nuisance
        // — which is why the assertion is on the nuisances above.
        assert!(
            fitted.alpha.abs() < 0.05,
            "no void was injected, so the amplitude must stay near zero, got {}",
            fitted.alpha
        );
        assert!(fitted.converged, "the fit must report convergence");
    }

    #[test]
    fn a_nuisance_no_ray_constrains_stays_at_its_prior_mean() {
        let (system, density, lambda0, dlambda_dx) = scene();
        let counts = lambda0.clone();
        let p = problem(&system, &counts, &lambda0, &dlambda_dx, &density);
        let bank = TemplateBank::new(&system.grid, &[1.0]);
        let config = ProfilerConfig::default();
        // Two detectors declared, but every ray belongs to detector 0: the second
        // efficiency appears in no rate, so its likelihood is flat and only the
        // prior can speak.
        let model = local_model(
            &p,
            &bank,
            0,
            centre(),
            Layout { detectors: 3 },
            ATTENUATION,
            &config,
        )
        .expect("the template has support");
        let mut start = vec![0.0f64; model.layout.dims()];
        for (index, slot) in start.iter_mut().enumerate() {
            *slot = model.layout.mean(index);
        }
        let fit = model.fit(Mode::Hypotheses, &start).expect("a fit");
        assert!(
            (fit.efficiency[2] - 1.0).abs() < 1.0e-9,
            "an unconstrained efficiency drifted to {}",
            fit.efficiency[2]
        );
        assert!(fit.converged);
    }

    #[test]
    fn profiling_systematics_reduces_the_significance() {
        let (system, density, lambda0, dlambda_dx) = scene();
        let bank = TemplateBank::new(&system.grid, &[1.0]);
        let config = ProfilerConfig::default();
        let layout = Layout { detectors: 2 };

        let base = problem(&system, &lambda0, &lambda0, &dlambda_dx, &density);
        let model = local_model(&base, &bank, 0, centre(), layout, ATTENUATION, &config)
            .expect("the template has support");
        // A real void plus a bulk-density deficit that looks like one: the two
        // are partly degenerate, so profiling has to give significance back.
        let truth = [0.5, 0.95, 1.0, 1.0, 1.0, 0.0, 0.0];
        let counts = rates(&model, &truth);

        let p = problem(&system, &counts, &lambda0, &dlambda_dx, &density);
        let profile = profile_candidate(&p, &bank, 0, centre(), &config).expect("a profile");
        assert!(
            profile.ts > 0.0,
            "a void is present, so the statistic must be positive"
        );
        assert!(
            profile.z < profile.z_without_systematics,
            "profiling must reduce the significance: {} with, {} without",
            profile.z,
            profile.z_without_systematics
        );
        let expected = (profile.z_without_systematics - profile.z) as f32;
        assert!(
            (profile.drop - expected).abs() < 1.0e-5,
            "the reported drop {} must equal the difference {expected}",
            profile.drop
        );
        assert!(profile.drop > 0.0);
    }

    #[test]
    fn fits_converge_in_fewer_than_twenty_newton_steps() {
        let (system, density, lambda0, dlambda_dx) = scene();
        let bank = TemplateBank::new(&system.grid, &[1.0]);
        let config = ProfilerConfig::default();
        let layout = Layout { detectors: 2 };

        let base = problem(&system, &lambda0, &lambda0, &dlambda_dx, &density);
        let model = local_model(&base, &bank, 0, centre(), layout, ATTENUATION, &config)
            .expect("the template has support");
        let truth = [0.5, 0.97, 1.02, 0.99, 1.01, 0.0, 0.0];
        let counts = rates(&model, &truth);

        let p = problem(&system, &counts, &lambda0, &dlambda_dx, &density);
        let profile = profile_candidate(&p, &bank, 0, centre(), &config).expect("a profile");
        assert!(
            profile.h0.steps <= 20,
            "H0 took {} Newton steps",
            profile.h0.steps
        );
        assert!(
            profile.h1.steps <= 20,
            "H1 took {} Newton steps",
            profile.h1.steps
        );
        assert!(profile.h0.converged && profile.h1.converged);
    }
}
