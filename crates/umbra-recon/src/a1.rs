//! A1: ordered-subset transmission EM, the classical baseline (spec §5.5).
//!
//! **L0 only.** At L0 the expected counts are
//!
//! ```text
//! λ_i = c_i · exp(−X_i / Λ),   c_i = T · ε_d A_d cos θ_i ΔΩ_i · Φ₀,   X_i = (Lρ)_i
//! ```
//!
//! and the data are Poisson, `y_i ~ Poisson(λ_i)`. At L1 and L2 the flux
//! integral is not of this form, so running this solver on L1 data is a model
//! error rather than a tuning choice: use A2 or A3 there. The golden default
//! scene is L1, which is why the accuracy tests here use the same scene at L0
//! and the L1 data is only a robustness check.
//!
//! # The update
//!
//! The Poisson log-likelihood of the L0 model is
//!
//! ```text
//! ℓ(ρ) = Σ_i [ y_i ln λ_i(ρ) − λ_i(ρ) ],   ∂ℓ/∂ρ_j = Σ_i l_ij (y_i − λ_i),
//! ```
//!
//! so the ML equations are `Lᵀ λ = Lᵀ y`.
//!
//! # The step
//!
//! The emission MLEM update — multiply by the back-projected `y/λ` — does not
//! transfer to transmission. Attenuation enters with a negative sign, so where
//! the model predicts *more* counts than were measured the density must go
//! *up*, and the emission form moves it down. The fixed point is still
//! `λ = y`, which is what makes the mistake survive a convergence test and
//! fail only as a wrong answer.
//!
//! What transfers is the SPS/OSTR surrogate step of Erdoğan & Fessler (1999):
//! the paraboloidal surrogate of the ray term has curvature `λ_i` (the second
//! derivative of `c e^{−x}`, since `c e^{−x} = λ`), giving the separable update
//!
//! ```text
//! g_j = Σ_{i∈S} (l_ij/Λ)(λ_i − y_i),   κ_j = Σ_{i∈S} l_ij λ_i (Σ_k l_ik)/Λ²,   ρ_j ← ρ_j + g_j/κ_j .
//! ```
//!
//! Both factors of `Λ` belong there: `∂λ/∂ρ = −(l/Λ)λ`, so the gradient carries
//! `1/Λ` and the Hessian `1/Λ²`. Dropping them leaves a step that is `Λ²/l` too
//! small — a solver that looks like it is converging, just glacially, and whose
//! numbers are wrong in a way no assertion about directions would catch.
//!
//! `g` is the gradient of the **log-likelihood**, not of the NLL: using
//! `y − λ` here points uphill, which the guard below turns into zero accepted
//! steps. The `Σ_k l_ik` factor is Jensen's inequality turning the
//! non-separable `(Σ_j l_ij δ_j)²` into a per-voxel quadratic. The
//! diagonal of the Fisher information alone — `Σ l_ij² λ_i` — is the tempting
//! choice and it is too small: on a limited-angle problem the off-diagonal
//! coupling is where the curvature lives, and a Jacobi step scaled by the
//! diagonal overshoots by orders of magnitude, so the guard halves it to
//! nothing and the fit creeps. That failure looks like slow convergence, not
//! like a bug, which is why the row sum is worth the extra multiply.
//!
//! clamped to `[0, ρ_max]`. `κ_j` is the diagonal of the Fisher information
//! `Lᵀ diag(λ) L`, so the step is already in density units and needs no global
//! step size. `S` is the full ray set for a full iteration, or one of `subsets`
//! interleaved groups for OSTR.
//!
//! The curvature is evaluated at the current iterate rather than bounded over
//! the whole range: the global bound `κ_i ≤ c_i` is `e^{x_i}` times larger, and
//! for a few metres of rock that overflows f32 outright. That is the one place
//! this differs from the published SPS algorithm (which uses the optimal
//! curvature of a bounded interval), and it is why the guard below is not an
//! optional extra: it is what restores monotonicity the surrogate no longer
//! guarantees.
//!
//! # Why the step is guarded
//!
//! Ordered subsets break monotonicity by construction: a subset step optimises
//! one part of the likelihood and can raise the full objective. Rather than
//! assert monotonicity and hope, each step is relaxed by `α ∈ {1, ½, …}` and
//! accepted only if it lowers the **full** NLL; the guard is one extra forward
//! evaluation on a 32-ray scene, so the guarantee is cheaper than the failure it
//! prevents. `α = 1` is tried first, so the relaxation costs nothing when the
//! surrogate is behaving.
//!
//! A step for which no `α` lowers the objective means the multiplicative update
//! has reached its fixed point — `Lᵀy = Lᵀλ`, the stationarity condition — and
//! the solver reports convergence instead of spinning.
//!
//! # What it does not do
//!
//! No regularisation, so it inherits the null space of a limited-angle
//! geometry: it fits the data it has and is free to do anything in the
//! directions the detectors cannot see. That is the honest behaviour of the
//! classical method, and A2–A4 exist to show the difference.

use umbra_core::Csr;
use umbra_forward::expected::ForwardParams;
use umbra_forward::scene::SceneSpec;
use umbra_forward::System;

use crate::{Control, Solver, StepReport};

/// Tuning for [`TransmissionEm`].
#[derive(Debug, Clone, PartialEq)]
pub struct A1Config {
    /// Number of ordered subsets. `1` is plain EM over all rays.
    pub subsets: u32,
    /// Exposure, in seconds. Multiplies the acceptance to give `c_i`.
    pub exposure_s: f32,
    /// L0 reference flux `Φ₀`, in cm⁻² s⁻¹ sr⁻¹.
    pub flux0: f32,
    /// L0 attenuation length `Λ`, in g cm⁻².
    pub attenuation_gcm2: f32,
    /// Starting density, in g cm⁻³, for every voxel.
    ///
    /// A multiplicative update cannot move a voxel out of zero, so the start is
    /// the bulk density: a voxel that no ray crosses keeps the prior instead of
    /// becoming a spurious vacuum.
    pub rho_init: f32,
    /// Upper clamp on the estimate, in g cm⁻³.
    pub rho_max: f32,
    /// Convergence when one step lowers the objective by less than this
    /// fraction of its magnitude.
    pub tolerance: f32,
    /// How many times a step may be halved before it is declared stalled.
    pub max_backtracks: u32,
}

impl Default for A1Config {
    fn default() -> Self {
        Self {
            subsets: 1,
            exposure_s: 1.0,
            flux0: 7.0e-3,
            attenuation_gcm2: 1000.0,
            rho_init: 2.65,
            rho_max: 4.0,
            tolerance: 1.0e-7,
            max_backtracks: 12,
        }
    }
}

impl A1Config {
    /// The configuration a scene implies: its exposure and bulk density, with
    /// the L0 parameters of the standard rock calibration.
    pub fn from_scene(scene: &SceneSpec) -> Self {
        let params = ForwardParams::default();
        Self {
            exposure_s: scene.exposure_s,
            flux0: params.l0_flux0,
            attenuation_gcm2: params.l0_attenuation_gcm2,
            rho_init: scene.physics.rho_bulk,
            rho_max: (scene.physics.rho_bulk * 1.5).max(1.0),
            ..Self::default()
        }
    }
}

/// Ordered-subset transmission EM.
#[derive(Debug, Clone)]
pub struct TransmissionEm {
    l: Csr<f32>,
    /// Observed counts, one per ray cell.
    counts: Vec<f32>,
    /// `c_i`: expected counts for a fully transparent scene, in counts.
    k: Vec<f32>,
    attenuation_gcm2: f32,
    rho: Vec<f32>,
    lambda: Vec<f32>,
    subsets: Vec<Vec<u32>>,
    cursor: usize,
    objective: f64,
    iteration: u32,
    converged: bool,
    rho_max: f32,
    tolerance: f32,
    max_backtracks: u32,
    /// Sum of the stored path lengths in each row, precomputed for the
    /// surrogate curvature.
    row_sum: Vec<f32>,
    /// Consecutive steps that found no descent, reset on progress.
    stalled: u32,
}

impl TransmissionEm {
    /// Builds the solver for a system and the counts it should explain.
    ///
    /// # Panics
    ///
    /// Panics if `counts` does not have one entry per ray cell, or if the
    /// attenuation length is not positive — a zero or negative `Λ` would make
    /// the model meaningless rather than merely badly tuned.
    pub fn new(system: &System, counts: &[f32], cfg: A1Config) -> Self {
        assert_eq!(
            counts.len(),
            system.ray_count(),
            "counts must have one entry per ray cell"
        );
        assert!(
            cfg.attenuation_gcm2.is_finite() && cfg.attenuation_gcm2 > 0.0,
            "the L0 attenuation length must be positive"
        );
        let k: Vec<f32> = system
            .rays
            .iter()
            .map(|ray| cfg.exposure_s.max(0.0) * ray.k_scale * cfg.flux0)
            .collect();
        let subsets = partition(system.ray_count(), cfg.subsets);
        let rho = vec![cfg.rho_init.max(0.0); system.voxel_count()];
        let row_sum: Vec<f32> = (0..system.ray_count())
            .map(|ray| system.l.row(ray).map(|(_, length)| length).sum())
            .collect();

        let mut solver = Self {
            l: system.l.clone(),
            counts: counts.to_vec(),
            k,
            attenuation_gcm2: cfg.attenuation_gcm2,
            rho,
            lambda: Vec::new(),
            subsets,
            cursor: 0,
            objective: f64::INFINITY,
            iteration: 0,
            converged: false,
            rho_max: cfg.rho_max,
            tolerance: cfg.tolerance,
            max_backtracks: cfg.max_backtracks,
            row_sum,
            stalled: 0,
        };
        let (lambda, objective) = solver.evaluate(&solver.rho.clone());
        solver.lambda = lambda;
        solver.objective = objective;
        solver
    }

    /// Poisson negative log-likelihood of the current estimate, up to the
    /// constant `Σ ln y_i!`.
    pub fn objective(&self) -> f64 {
        self.objective
    }

    /// Completed steps, counting each subset update as one.
    pub fn iterations(&self) -> u32 {
        self.iteration
    }

    /// Number of ordered subsets in use.
    pub fn subset_count(&self) -> usize {
        self.subsets.len()
    }

    /// `λ` at the current estimate, one entry per ray cell.
    pub fn lambda(&self) -> &[f32] {
        &self.lambda
    }

    /// `λ(ρ)` and the NLL, computed together so the guard costs one pass.
    fn evaluate(&self, rho: &[f32]) -> (Vec<f32>, f64) {
        let mut x = vec![0.0f32; self.counts.len()];
        self.l.matvec(rho, &mut x);
        let mut lambda = Vec::with_capacity(self.counts.len());
        let mut objective = 0.0f64;
        for (ray, (x_i, k_i)) in x.iter().zip(&self.k).enumerate() {
            let expected =
                f64::from(*k_i) * (-f64::from(*x_i) / f64::from(self.attenuation_gcm2)).exp();
            let observed = f64::from(self.counts[ray]);
            if !expected.is_finite() || expected <= 0.0 {
                // A ray with no acceptance carries no information. Observed
                // counts against zero acceptance are impossible data, and the
                // infinite objective is the honest description of that.
                if observed > 0.0 {
                    objective = f64::INFINITY;
                }
                lambda.push(0.0);
                continue;
            }
            objective += expected - observed * expected.ln();
            lambda.push(expected as f32);
        }
        (lambda, objective)
    }

    /// One multiplicative update restricted to `subset`, with the monotone
    /// guard applied.
    fn update(&mut self, subset: usize) -> (bool, f64) {
        let n = self.rho.len();
        let attenuation = f64::from(self.attenuation_gcm2);
        let mut gradient = vec![0.0f64; n];
        for &ray in &self.subsets[subset] {
            let ray = ray as usize;
            let expected = f64::from(self.lambda[ray]);
            let observed = f64::from(self.counts[ray]);
            if !observed.is_finite() || !expected.is_finite() || expected <= 0.0 {
                continue;
            }
            // Ascending in the log-likelihood: a ray that saw *fewer* counts
            // than predicted needs more density along its path.
            let residual = expected - observed;
            for (voxel, length) in self.l.row(ray) {
                gradient[voxel] += f64::from(length) * residual / attenuation;
            }
        }

        // The curvature is built from *every* ray, not from the subset. It is a
        // property of the problem, not of which rays this step looked at: using
        // the subset's curvature makes each step |S| times shorter, so a sweep
        // of subsets would cost the same and move less than one full step.
        let mut curvature = vec![0.0f64; n];
        for ray in 0..self.l.nrows() {
            let expected = f64::from(self.lambda[ray]);
            if !expected.is_finite() || expected <= 0.0 {
                continue;
            }
            let row_sum = f64::from(self.row_sum[ray]);
            for (voxel, length) in self.l.row(ray) {
                let a = f64::from(length);
                curvature[voxel] += a * expected * row_sum / (attenuation * attenuation);
            }
        }

        let mut candidate = self.rho.clone();
        for voxel in 0..n {
            if curvature[voxel] > 0.0 {
                let step = gradient[voxel] / curvature[voxel];
                candidate[voxel] =
                    (f64::from(self.rho[voxel]) + step).clamp(0.0, f64::from(self.rho_max)) as f32;
            }
        }

        // Any decrease counts. A relative threshold here is not a safety margin:
        // on a 10⁹ NLL it is an absolute bar of 10⁻³, which rejects real progress
        // on small steps and reads afterwards as "converged".
        let mut alpha = 1.0f32;
        for _ in 0..=self.max_backtracks {
            let trial: Vec<f32> = self
                .rho
                .iter()
                .zip(&candidate)
                .map(|(old, new)| old + alpha * (new - old))
                .collect();
            let (lambda, objective) = self.evaluate(&trial);
            if objective < self.objective {
                self.rho = trial;
                self.lambda = lambda;
                let decrease = self.objective - objective;
                self.objective = objective;
                self.iteration += 1;
                self.stalled = 0;
                let relative = decrease / self.objective.abs().max(1.0);
                self.converged = relative <= f64::from(self.tolerance);
                return (true, decrease);
            }
            alpha *= 0.5;
        }
        // No relaxation descends along this subset's direction. One subset is
        // not the problem: the next subset has a different direction, so only a
        // full sweep of failures means the solver is at a stationary point.
        self.stalled += 1;
        self.converged = self.stalled as usize >= self.subsets.len();
        (false, 0.0)
    }
}

/// Splits `n_rays` into `subsets` interleaved groups.
///
/// Interleaving rather than contiguous blocks matters here: the rays of one
/// detector are ordered by angle, so a contiguous block would hand every subset
/// a narrow angular wedge and the ordered-subset acceleration would have to
/// fight a nearly degenerate sub-problem.
fn partition(n_rays: usize, subsets: u32) -> Vec<Vec<u32>> {
    let count = subsets.max(1) as usize;
    let mut groups = vec![Vec::new(); count];
    for ray in 0..n_rays {
        groups[ray % count].push(ray as u32);
    }
    groups
}

impl Solver for TransmissionEm {
    type Config = A1Config;

    fn init(system: &System, counts: &[f32], cfg: &Self::Config) -> Self {
        Self::new(system, counts, cfg.clone())
    }

    fn step(&mut self, ctl: &Control) -> StepReport {
        // Cancellation is honoured between subsets, never inside one: a subset
        // update is the smallest unit that leaves a consistent estimate.
        if self.converged || ctl.should_stop() || self.iteration >= ctl.max_iters {
            return StepReport {
                iteration: self.iteration,
                objective: Some(self.objective),
                converged: self.converged,
            };
        }
        let subset = self.cursor;
        self.cursor = (self.cursor + 1) % self.subsets.len();
        let _ = self.update(subset);
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

#[cfg(test)]
mod tests {
    use super::*;
    use umbra_core::{Grid2, GridAny};
    use umbra_forward::RayCell;

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
        }
    }

    #[test]
    fn subsets_partition_every_ray_exactly_once() {
        for (rays, subsets) in [(0, 4), (1, 1), (7, 3), (32, 8), (5, 9)] {
            let groups = partition(rays, subsets);
            let total: usize = groups.iter().map(Vec::len).sum();
            assert_eq!(total, rays, "{rays} rays into {subsets} subsets");
            let mut seen: Vec<u32> = groups.iter().flatten().copied().collect();
            seen.sort_unstable();
            seen.dedup();
            assert_eq!(seen.len(), rays, "a ray landed in two subsets");
        }
    }

    #[test]
    fn interleaved_subsets_span_the_angle_range() {
        // With one detector ordered by angle, a subset must hold several
        // distinct angles or the acceleration is fighting a rank-deficient
        // sub-problem.
        let groups = partition(32, 8);
        assert!(groups.iter().all(|group| group.len() == 4));
        for group in &groups {
            let spread = group.last().unwrap() - group.first().unwrap();
            assert!(spread >= 8, "subset {group:?} is too narrow");
        }
    }

    #[test]
    fn a_scene_supplies_exposure_and_bulk_density() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../web/src/scenes/default.json");
        let scene = SceneSpec::from_json_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let cfg = A1Config::from_scene(&scene);
        assert_eq!(cfg.exposure_s, scene.exposure_s);
        assert_eq!(cfg.rho_init, scene.physics.rho_bulk);
        assert!(cfg.rho_max > scene.physics.rho_bulk);
    }

    #[test]
    #[should_panic(expected = "one entry per ray cell")]
    fn the_wrong_number_of_counts_is_rejected() {
        let system = trivial_system(3);
        let _ = TransmissionEm::new(&system, &[1.0, 2.0], A1Config::default());
    }

    #[test]
    fn a_zero_acceptance_ray_is_ignored_rather_than_dividing_by_zero() {
        let mut system = trivial_system(2);
        system.rays[1].k_scale = 0.0;
        let solver = TransmissionEm::new(&system, &[100.0, 0.0], A1Config::default());
        assert_eq!(solver.lambda()[1], 0.0);
        assert!(solver.objective().is_finite());
    }
}
