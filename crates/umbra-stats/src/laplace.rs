//! A11: Laplace/Fisher uncertainty at a reconstruction.
//!
//! The Laplace approximation turns the curvature of the objective at the MAP
//! estimate into error bars. For the Poisson likelihood the curvature of the
//! data term is the Fisher information
//!
//! ```text
//! F = Jᵀ diag(1/λ) J,        J_ij = (∂λ_i/∂X_i) · L_ij,
//! ```
//!
//! so `F` never has to be formed: `F x = Lᵀ[(∂λ/∂X)²/λ ∘ (L x)]`, two sparse
//! products with the operator every solver already holds. The prior contributes
//! its own Hessian — for TV that is a weighted graph Laplacian — and the two
//! together give the posterior precision `A = F + C`.
//!
//! # Why the diagonal is not `1/√A_jj`
//!
//! `A_jj` is the precision of voxel `j` *given every other voxel*, which is the
//! wrong quantity: neighbouring voxels compete for the same rays, so the
//! marginal variance is larger than the conditional one. This module computes
//! `[A⁻¹]_jj` by solving `A x = e_j` for each voxel and taking `x_j`, which is
//! exact to the CG tolerance and never materialises `A⁻¹` (1024² entries are
//! already 4 MB in f32, and the grid grows).
//!
//! # Voxels the geometry cannot see
//!
//! A voxel no ray responds to has `F_jj = 0`. Its variance is set entirely by
//! the prior, and reporting that number would claim the data measured something
//! it did not. Those voxels are reported as [`Uncertainty::Unconstrained`] — a
//! statement about the geometry, not a very large σ.

use umbra_forward::expected::{expected as evaluate, ForwardParams};
use umbra_forward::{Expected, System};
use umbra_physics::{Fidelity, FluxTable};

/// Everything the approximation needs to know about the observation.
///
/// The counts do not appear: at the estimate the expected Fisher information
/// depends on `λ(ρ̂)` and not on `y`. That is the difference between the Fisher
/// and the observed information, and it is why the result is smooth in the data.
#[derive(Debug, Clone, Copy)]
pub struct PoissonProblem<'a> {
    /// The path-length operator and its ray metadata.
    pub system: &'a System,
    /// The flux table the counts were generated with.
    pub flux: &'a FluxTable,
    /// Which physics the counts obey.
    pub fidelity: Fidelity,
    /// Exposure, in seconds.
    pub exposure_s: f32,
    /// CSDA and L0 parameters.
    pub params: ForwardParams,
}

/// Prior curvature, as a weighted graph Laplacian over voxels.
///
/// A TV prior has Hessian `β/ε` on every edge whose density difference is inside
/// the quadratic region and zero outside, which is exactly this: one weight per
/// neighbour pair. The Laplacian is positive semi-definite and its one null
/// direction is the uniform field, so it regularises the directions the data
/// cannot see without pretending to know their value.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GraphCurvature {
    n: usize,
    edges: Vec<(u32, u32, f64)>,
    diagonal: Vec<f64>,
}

impl GraphCurvature {
    /// An empty curvature on `n` voxels: the data term alone.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            edges: Vec::new(),
            diagonal: vec![0.0; n],
        }
    }

    /// Number of voxels.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether any edge carries curvature.
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    /// Adds `weight · (x_a − x_b)²` to the prior curvature.
    ///
    /// Zero and non-finite weights are ignored: an edge whose density jump is
    /// outside the Huber region contributes no curvature, and a non-finite one
    /// would poison every column of the solve.
    ///
    /// # Panics
    ///
    /// Panics if either voxel is out of range.
    pub fn add_edge(&mut self, a: usize, b: usize, weight: f64) {
        assert!(a < self.n && b < self.n, "edge endpoints must be voxels");
        assert_ne!(a, b, "a self-edge carries no curvature");
        if !weight.is_finite() || weight <= 0.0 {
            return;
        }
        self.edges.push((a as u32, b as u32, weight));
        self.diagonal[a] += weight;
        self.diagonal[b] += weight;
    }

    /// `y += C x`.
    fn apply_into(&self, x: &[f64], y: &mut [f64]) {
        for &(a, b, weight) in &self.edges {
            let (a, b) = (a as usize, b as usize);
            let flux = weight * (x[a] - x[b]);
            y[a] += flux;
            y[b] -= flux;
        }
    }

    /// The diagonal of `C`, which the preconditioner needs.
    fn diagonal(&self) -> &[f64] {
        &self.diagonal
    }
}

/// What the data says about one voxel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Uncertainty {
    /// A standard deviation, in g cm⁻³.
    Sigma(f32),
    /// No ray responds to this voxel: the geometry measured nothing here.
    Unconstrained,
}

impl Uncertainty {
    /// The standard deviation, if the voxel is constrained at all.
    pub fn sigma(&self) -> Option<f32> {
        match self {
            Uncertainty::Sigma(sigma) => Some(*sigma),
            Uncertainty::Unconstrained => None,
        }
    }

    /// Whether the voxel is constrained by the geometry.
    pub fn is_constrained(&self) -> bool {
        matches!(self, Uncertainty::Sigma(_))
    }
}

/// Tuning for the covariance solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LaplaceConfig {
    /// Relative residual at which a column's CG solve stops.
    pub tolerance: f64,
    /// Iteration cap per column. Reaching it marks the column unresolved
    /// rather than returning a number that has not converged.
    pub max_cg_iters: u32,
    /// Prior standard deviation on the mean density, in g cm⁻³. Zero disables
    /// it, which leaves the global level unconstrained.
    ///
    /// TV is blind to a constant offset: adding a constant to the whole image
    /// changes no edge difference, so the TV curvature has the uniform field as
    /// an exact null direction and a data-only Fisher matrix carries almost no
    /// curvature along it either. Left there, that mode is what a conjugate
    /// gradient solve diverges on — not slowly converges, diverges, with a
    /// residual that grows past `10¹⁰` — because the effective condition number
    /// leaves the range of f64.
    ///
    /// A level prior is the honest fix rather than a numerical crutch: the bulk
    /// density of a known structure is known, and saying so to ±0.5 g cm⁻³
    /// constrains only the one direction the edges and the rays leave free. It
    /// is a statement about the scene, so it is a configuration value and not a
    /// hidden epsilon.
    pub level_sigma: f32,
}

impl Default for LaplaceConfig {
    fn default() -> Self {
        Self {
            // f32 densities carry ~7 digits, so a residual tolerance below this
            // buys nothing but iterations.
            tolerance: 1.0e-6,
            max_cg_iters: 400,
            // The bulk density of a rock block is not a free parameter in
            // practice: 0.5 g cm⁻³ is a generous 20% on a 2.65 g cm⁻³ target.
            level_sigma: 0.5,
        }
    }
}

/// Per-voxel standard deviations, plus what the solve had to say about itself.
#[derive(Debug, Clone, PartialEq)]
pub struct SigmaMap {
    /// One entry per voxel, in the operator's voxel order.
    pub sigma: Vec<Uncertainty>,
    /// `F_jj` per voxel: the marginal geometric sensitivity, before any prior.
    ///
    /// Zero means no ray responds. It is exposed because it is the quantity the
    /// `Unconstrained` verdict is made from, and a caller may want to rank
    /// voxels by how well the geometry sees them.
    pub curvature: Vec<f32>,
    /// Columns whose CG solve did not reach the tolerance.
    pub unresolved: usize,
    /// Worst relative residual achieved across the columns.
    pub worst_residual: f64,
    /// Total CG iterations, as a cost figure for the card.
    pub cg_iterations: u64,
}

impl SigmaMap {
    /// One entry per voxel.
    pub fn len(&self) -> usize {
        self.sigma.len()
    }

    /// Whether the map is empty.
    pub fn is_empty(&self) -> bool {
        self.sigma.is_empty()
    }

    /// The uncertainty of one voxel.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of range.
    pub fn at(&self, voxel: usize) -> Uncertainty {
        self.sigma[voxel]
    }

    /// Number of voxels the geometry cannot see.
    pub fn unconstrained(&self) -> usize {
        self.sigma
            .iter()
            .filter(|entry| !entry.is_constrained())
            .count()
    }

    /// Largest σ over the constrained voxels, or `None` if all are unconstrained.
    pub fn worst_sigma(&self) -> Option<f32> {
        self.sigma
            .iter()
            .filter_map(Uncertainty::sigma)
            .fold(None, |worst: Option<f32>, sigma| {
                Some(worst.map_or(sigma, |current| current.max(sigma)))
            })
    }

    /// Whether every column reached the CG tolerance.
    ///
    /// A map that did not is not a set of error bars: it is a set of iterates
    /// from a solve that has not converged, and a card should say so instead of
    /// drawing bars from it.
    pub fn is_converged(&self) -> bool {
        self.unresolved == 0
    }
}

/// The posterior precision, applied without forming it.
struct Precision<'sys, 'pri> {
    system: &'sys System,
    /// `(∂λ/∂X)² / λ`, one entry per ray.
    weight: Vec<f64>,
    prior: Option<&'pri GraphCurvature>,
    /// `1/σ_level²`: the curvature of a Gaussian prior on the mean density.
    level_weight: f64,
    /// Scratch for `L x`, so a column's iterations allocate nothing.
    ray: Vec<f64>,
    diagonal: Vec<f64>,
}

impl<'sys, 'pri> Precision<'sys, 'pri> {
    fn new(
        problem: &PoissonProblem<'sys>,
        rho: &[f32],
        prior: Option<&'pri GraphCurvature>,
        level_weight: f64,
    ) -> Self {
        assert_eq!(
            rho.len(),
            problem.system.voxel_count(),
            "the estimate must have one entry per voxel"
        );
        let Expected {
            lambda, dlambda_dx, ..
        } = evaluate(
            problem.system,
            rho,
            problem.flux,
            problem.fidelity,
            problem.exposure_s,
            &problem.params,
        );
        let mut weight = vec![0.0f64; lambda.len()];
        for (slot, (&counts, &derivative)) in weight.iter_mut().zip(lambda.iter().zip(&dlambda_dx))
        {
            let counts = f64::from(counts);
            if counts > 0.0 {
                let derivative = f64::from(derivative);
                *slot = derivative * derivative / counts;
            }
        }
        let mut precision = Self {
            system: problem.system,
            weight,
            prior,
            level_weight,
            ray: vec![0.0; problem.system.ray_count()],
            diagonal: vec![0.0; problem.system.voxel_count()],
        };
        precision.recompute_diagonal();
        precision
    }

    fn nrays(&self) -> usize {
        self.system.ray_count()
    }

    fn nvoxels(&self) -> usize {
        self.system.voxel_count()
    }

    /// `y = A x`, with `A = Lᵀ diag(w) L + C`.
    fn apply(&mut self, x: &[f64], y: &mut [f64]) {
        y.fill(0.0);
        for (row, ray) in self.ray.iter_mut().enumerate() {
            let mut acc = 0.0f64;
            for (column, value) in self.system.l.row(row) {
                acc += f64::from(value) * x[column];
            }
            *ray = acc * self.weight[row];
        }
        for (row, ray) in self.ray.iter().enumerate() {
            for (column, value) in self.system.l.row(row) {
                y[column] += f64::from(value) * ray;
            }
        }
        if let Some(prior) = self.prior {
            prior.apply_into(x, y);
        }
        if self.level_weight > 0.0 {
            // Rank one: the curvature of `w · (1ᵀx)` in every coordinate.
            let level = self.level_weight * x.iter().sum::<f64>();
            for value in y.iter_mut() {
                *value += level;
            }
        }
    }

    /// `diag(F)` and `diag(A)`; the former decides constrained-ness.
    fn recompute_diagonal(&mut self) {
        let mut data = vec![0.0f64; self.nvoxels()];
        for row in 0..self.nrays() {
            let weight = self.weight[row];
            if weight == 0.0 {
                continue;
            }
            for (column, value) in self.system.l.row(row) {
                let value = f64::from(value);
                data[column] += weight * value * value;
            }
        }
        self.diagonal = data;
    }

    /// The data Fisher diagonal, `F_jj`.
    fn data_diagonal(&self) -> &[f64] {
        &self.diagonal
    }

    /// `diag(A) = diag(F) + diag(C) + 1/σ_level²`.
    fn diagonal(&self) -> Vec<f64> {
        let mut diagonal = self.diagonal.clone();
        if let Some(prior) = self.prior {
            for (value, extra) in diagonal.iter_mut().zip(prior.diagonal()) {
                *value += extra;
            }
        }
        for value in diagonal.iter_mut() {
            *value += self.level_weight;
        }
        diagonal
    }
}

/// `[A⁻¹]_jj` for one column, by conjugate gradients with a Jacobi
/// preconditioner.
///
/// Returns the entry, the iterations used and the relative residual reached.
/// Returns `None` when the column's diagonal is zero, which means the precision
/// has no curvature at all in that coordinate.
/// Reusable vectors, so a thousand columns do not each allocate five of them.
struct Scratch {
    x: Vec<f64>,
    residual: Vec<f64>,
    direction: Vec<f64>,
    applied: Vec<f64>,
}

impl Scratch {
    fn new(n: usize) -> Self {
        Self {
            x: vec![0.0; n],
            residual: vec![0.0; n],
            direction: vec![0.0; n],
            applied: vec![0.0; n],
        }
    }
}

/// `[A⁻¹]_jj` for one column, by conjugate gradients with a Jacobi
/// preconditioner.
///
/// `inverse` is `diag(A)⁻¹`, with a zero entry for a voxel whose diagonal is
/// zero. Dividing by that zero would put a NaN in the search direction that then
/// poisons the inner product for *every* column, which is why the guard lives in
/// the caller's precomputed vector rather than at each use.
fn column_variance(
    precision: &mut Precision<'_, '_>,
    column: usize,
    inverse: &[f64],
    scratch: &mut Scratch,
    cfg: &LaplaceConfig,
) -> Option<(f64, u32, f64)> {
    if inverse[column] <= 0.0 {
        return None;
    }
    let n = inverse.len();
    let Scratch {
        x,
        residual,
        direction,
        applied,
    } = scratch;
    x.fill(0.0);
    residual.fill(0.0);
    direction.fill(0.0);
    applied.fill(0.0);
    residual[column] = 1.0;
    for index in 0..n {
        direction[index] = residual[index] * inverse[index];
    }
    let mut rz = direction[column];
    let mut iterations = 0u32;
    for step in 0..cfg.max_cg_iters {
        precision.apply(direction, applied);
        let curvature = dot(direction, applied);
        if curvature <= 0.0 || !curvature.is_finite() {
            break;
        }
        let alpha = rz / curvature;
        for index in 0..n {
            x[index] += alpha * direction[index];
            residual[index] -= alpha * applied[index];
        }
        iterations = step + 1;
        let norm = dot(residual, residual).sqrt();
        if norm <= cfg.tolerance {
            break;
        }
        for index in 0..n {
            applied[index] = residual[index] * inverse[index];
        }
        let rz_next = dot(residual, applied);
        let beta = if rz != 0.0 { rz_next / rz } else { 0.0 };
        for index in 0..n {
            direction[index] = applied[index] + beta * direction[index];
        }
        rz = rz_next;
    }
    let residual_norm = dot(residual, residual).sqrt();
    let variance = x[column];
    if !variance.is_finite() {
        return None;
    }
    Some((variance, iterations, residual_norm))
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

/// Per-voxel standard deviations at `rho`, from the Fisher information and an
/// optional prior curvature.
///
/// Cost is one conjugate-gradient solve per voxel, with no dense matrix formed
/// at any point. A voxel no ray responds to is [`Uncertainty::Unconstrained`]
/// regardless of what the prior would have said about it.
///
/// # Panics
///
/// Panics if `rho` does not have one entry per voxel, or if `prior` was built
/// for a different grid.
pub fn poisson_laplace(
    problem: &PoissonProblem<'_>,
    rho: &[f32],
    prior: Option<&GraphCurvature>,
    cfg: &LaplaceConfig,
) -> SigmaMap {
    if let Some(prior) = prior {
        assert_eq!(
            prior.len(),
            problem.system.voxel_count(),
            "prior curvature must be built on the same grid as the estimate"
        );
    }
    let level_weight = if cfg.level_sigma > 0.0 {
        let sigma = f64::from(cfg.level_sigma);
        1.0 / (sigma * sigma)
    } else {
        0.0
    };
    let mut precision = Precision::new(problem, rho, prior, level_weight);
    let n = precision.nvoxels();
    let data = precision.data_diagonal().to_vec();
    let inverse: Vec<f64> = precision
        .diagonal()
        .iter()
        .map(|value| if *value > 0.0 { 1.0 / value } else { 0.0 })
        .collect();
    let mut scratch = Scratch::new(n);
    let mut sigma = Vec::with_capacity(n);
    let mut unresolved = 0usize;
    let mut worst_residual = 0.0f64;
    let mut cg_iterations = 0u64;

    for (voxel, curvature) in data.iter().enumerate() {
        if *curvature <= 0.0 {
            sigma.push(Uncertainty::Unconstrained);
            continue;
        }
        match column_variance(&mut precision, voxel, &inverse, &mut scratch, cfg) {
            Some((variance, iterations, residual)) if variance > 0.0 => {
                sigma.push(Uncertainty::Sigma(variance.sqrt() as f32));
                if residual > cfg.tolerance {
                    unresolved += 1;
                }
                worst_residual = worst_residual.max(residual);
                cg_iterations += u64::from(iterations);
            }
            _ => {
                // The precision had no usable curvature in this coordinate: the
                // prior did not regularise it either.
                sigma.push(Uncertainty::Unconstrained);
                unresolved += 1;
            }
        }
    }

    SigmaMap {
        sigma,
        curvature: data.iter().map(|value| *value as f32).collect(),
        unresolved,
        worst_residual,
        cg_iterations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use umbra_core::{Csr, Grid2, GridAny};
    use umbra_forward::RayCell;
    use umbra_physics::generate_default;

    /// A small deterministic generator: the coverage test needs many independent
    /// Poisson draws and nothing else.
    struct Lcg(u64);

    impl Lcg {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }

        /// Uniform on `(0, 1)`.
        fn uniform(&mut self) -> f64 {
            ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }
    }

    /// Knuth's product method. Fine for the few hundred counts the coverage test
    /// uses; it underflows above ~700, which is why nothing here asks it to.
    fn poisson(rng: &mut Lcg, mean: f64) -> f64 {
        let limit = (-mean).exp();
        let mut product = 1.0;
        let mut count = 0.0;
        loop {
            count += 1.0;
            product *= rng.uniform();
            if product <= limit {
                return count - 1.0;
            }
        }
    }

    /// `rays` rows, each naming exactly one voxel and a path length in cm: every
    /// voxel measured independently, which makes the coverage check analytic.
    fn independent_rays(paths_cm: &[f32]) -> System {
        let mut indptr = vec![0u32];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        let mut rays = Vec::new();
        for (voxel, path) in paths_cm.iter().enumerate() {
            indices.push(voxel as u32);
            data.push(*path);
            indptr.push(indices.len() as u32);
            rays.push(RayCell {
                det: 0,
                zen: 0.0,
                az: 0.0,
                k_scale: 1.0e4,
            });
        }
        System {
            l: Csr::new(indptr, indices, data),
            rays,
            grid: GridAny::Two(Grid2::new([paths_cm.len() as u32, 1], [0.0, 0.0], 0.1)),
            poses: Vec::new(),
        }
    }

    fn problem<'a>(system: &'a System, flux: &'a FluxTable) -> PoissonProblem<'a> {
        PoissonProblem {
            system,
            flux,
            fidelity: Fidelity::L0,
            exposure_s: 1.0,
            params: ForwardParams::default(),
        }
    }

    #[test]
    fn a_voxel_no_ray_touches_is_unconstrained_not_zero_variance() {
        let flux = generate_default();
        // Three voxels, two rays: the third is invisible to the geometry.
        let system = System {
            l: Csr::new(vec![0, 1, 2], vec![0, 1], vec![200.0f32, 200.0]),
            rays: (0..2)
                .map(|_| RayCell {
                    det: 0,
                    zen: 0.0,
                    az: 0.0,
                    k_scale: 1.0e4,
                })
                .collect(),
            grid: GridAny::Two(Grid2::new([3, 1], [0.0, 0.0], 0.1)),
            poses: Vec::new(),
        };
        let map = poisson_laplace(
            &problem(&system, &flux),
            &[2.65, 2.65, 2.65],
            None,
            &LaplaceConfig::default(),
        );
        assert_eq!(map.at(0).sigma().map(|s| s > 0.0), Some(true));
        assert_eq!(map.at(1).sigma().map(|s| s > 0.0), Some(true));
        assert_eq!(map.at(2), Uncertainty::Unconstrained);
        assert_eq!(map.unconstrained(), 1);
        assert_eq!(map.curvature[2], 0.0);
        assert!(map.is_converged(), "the constrained columns did converge");
    }

    #[test]
    fn the_diagonal_matches_a_brute_force_inverse() {
        // Four voxels, three rays that share them, plus a prior graph: the
        // normal matrix is genuinely coupled, so `1/sqrt(A_jj)` would be wrong
        // and only an actual inversion agrees.
        let flux = generate_default();
        let system = System {
            l: Csr::new(
                vec![0, 2, 4, 7],
                vec![0, 1, 0, 2, 1, 2, 3],
                vec![100.0f32, 40.0, 30.0, 90.0, 70.0, 20.0, 60.0],
            ),
            rays: (0..3)
                .map(|_| RayCell {
                    det: 0,
                    zen: 0.0,
                    az: 0.0,
                    k_scale: 1.0e4,
                })
                .collect(),
            grid: GridAny::Two(Grid2::new([4, 1], [0.0, 0.0], 0.1)),
            poses: Vec::new(),
        };
        let rho = [2.65f32, 2.4, 2.9, 2.0];
        let mut prior = GraphCurvature::new(4);
        prior.add_edge(0, 1, 3.0);
        prior.add_edge(2, 3, 1.5);

        // The dense reference below is the data and graph curvature alone, so
        // the level prior is off; the tolerance is tight because this test is
        // about the CG solve reproducing that exact matrix, and a loose one
        // would only measure the tolerance.
        let cfg = LaplaceConfig {
            tolerance: 1.0e-13,
            level_sigma: 0.0,
            ..LaplaceConfig::default()
        };
        let map = poisson_laplace(&problem(&system, &flux), &rho, Some(&prior), &cfg);
        let precision = Precision::new(&problem(&system, &flux), &rho, Some(&prior), 0.0);

        // Dense reference: A = Σ_k w_k l_k l_kᵀ + C, inverted by Gauss–Jordan.
        let n = 4;
        let mut dense = vec![vec![0.0f64; n]; n];
        for (row, weight) in precision.weight.iter().enumerate() {
            let entries: Vec<(usize, f64)> = system
                .l
                .row(row)
                .map(|(column, value)| (column, f64::from(value)))
                .collect();
            for &(a, left) in &entries {
                for &(b, right) in &entries {
                    dense[a][b] += weight * left * right;
                }
            }
        }
        for (a, b, weight) in [(0usize, 1usize, 3.0f64), (2, 3, 1.5)] {
            dense[a][a] += weight;
            dense[b][b] += weight;
            dense[a][b] -= weight;
            dense[b][a] -= weight;
        }
        let inverse = invert(dense);
        for (voxel, row) in inverse.iter().enumerate() {
            let expected = row[voxel].sqrt();
            let found = f64::from(map.at(voxel).sigma().expect("constrained"));
            let relative = ((found - expected) / expected).abs();
            assert!(
                relative < 1.0e-4,
                "voxel {voxel}: sigma {found} vs dense inverse {expected} ({relative})"
            );
        }
    }

    /// Gauss–Jordan with partial pivoting. Test-only: the module itself never
    /// forms an inverse, which is the whole point of the CG solve.
    fn invert(mut a: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let n = a.len();
        let mut inverse = vec![vec![0.0f64; n]; n];
        for (i, row) in inverse.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        for column in 0..n {
            let pivot = (column..n)
                .max_by(|&l, &r| a[l][column].abs().partial_cmp(&a[r][column].abs()).unwrap())
                .expect("non-empty");
            a.swap(column, pivot);
            inverse.swap(column, pivot);
            let scale = a[column][column];
            for slot in 0..n {
                a[column][slot] /= scale;
                inverse[column][slot] /= scale;
            }
            for row in 0..n {
                if row == column {
                    continue;
                }
                let factor = a[row][column];
                for slot in 0..n {
                    a[row][slot] -= factor * a[column][slot];
                    inverse[row][slot] -= factor * inverse[column][slot];
                }
            }
        }
        inverse
    }

    #[test]
    fn independent_poisson_cells_cover_the_truth_at_sixty_eight_percent() {
        // Each ray crosses exactly one voxel, so the MAP is the MLE and the
        // Laplace interval is the Wald interval. If the σ map is right, 68% of
        // the intervals contain the truth and the predicted σ matches the
        // spread of the estimates. A σ that is off by a factor shows up as a
        // wrong coverage immediately.
        let flux = generate_default();
        let path = 211.2f32;
        let system = independent_rays(&[path, path, path, path, path, path, path, path]);
        let truth = [2.65f32; 8];
        let problem = problem(&system, &flux);
        let truth_lambda = evaluate(
            &system,
            &truth,
            &flux,
            Fidelity::L0,
            1.0,
            &ForwardParams::default(),
        );
        for lambda in &truth_lambda.lambda {
            assert!(
                (20.0..200.0).contains(lambda),
                "the coverage check needs a count rate where Knuth is exact, got {lambda}"
            );
        }

        let mut rng = Lcg(0x5EED_1234_ABCD_0001);
        let mut inside = 0usize;
        let mut total = 0usize;
        let mut predicted = Vec::new();
        let mut errors = Vec::new();
        for _ in 0..400 {
            let counts: Vec<f32> = truth_lambda
                .lambda
                .iter()
                .map(|lambda| poisson(&mut rng, f64::from(*lambda)) as f32)
                .collect();
            // Invert λ(ρ) = y under L0: λ = T k Φ₀ exp(−Lρ/Λ), with row `voxel`
            // crossing only voxel `voxel`.
            let mut estimate = Vec::with_capacity(8);
            for (voxel, observed) in counts.iter().enumerate() {
                let observed = f64::from(*observed).max(0.5);
                let amplitude = f64::from(system.rays[voxel].k_scale) * 7.0e-3;
                let opacity = 1000.0 * (amplitude / observed).ln();
                estimate.push((opacity / f64::from(path)) as f32);
            }
            // Each cell is measured on its own, so the level is already
            // constrained by the data; adding a level prior would couple the
            // eight cells and the Wald interval would no longer be the
            // reference this test is checking against.
            let cfg = LaplaceConfig {
                level_sigma: 0.0,
                ..LaplaceConfig::default()
            };
            let map = poisson_laplace(&problem, &estimate, None, &cfg);
            for (voxel, estimate) in estimate.iter().enumerate() {
                let sigma = map.at(voxel).sigma().expect("constrained");
                let error = f64::from((estimate - truth[voxel]).abs());
                total += 1;
                if error <= f64::from(sigma) {
                    inside += 1;
                }
                predicted.push(sigma);
                errors.push(error);
            }
        }
        let coverage = inside as f64 / total as f64;
        assert!(
            (0.64..0.72).contains(&coverage),
            "68% intervals covered the truth {:.3} of the time ({inside}/{total})",
            coverage
        );
        let mean_sigma = predicted.iter().map(|s| f64::from(*s)).sum::<f64>() / total as f64;
        let empirical = (errors.iter().map(|e| e * e).sum::<f64>() / total as f64).sqrt();
        let ratio = empirical / mean_sigma;
        assert!(
            (0.93..1.07).contains(&ratio),
            "empirical spread {empirical} vs predicted {mean_sigma} (ratio {ratio})"
        );
    }

    #[test]
    fn a_prior_that_leaves_a_direction_flat_is_reported_unresolved() {
        // One ray, two voxels, no prior: the precision has rank one and e_1 is
        // not in its range. The solve cannot converge and must say so rather
        // than hand back a number.
        let flux = generate_default();
        let system = System {
            l: Csr::new(vec![0, 2], vec![0, 1], vec![100.0f32, 100.0]),
            rays: vec![RayCell {
                det: 0,
                zen: 0.0,
                az: 0.0,
                k_scale: 1.0e4,
            }],
            grid: GridAny::Two(Grid2::new([2, 1], [0.0, 0.0], 0.1)),
            poses: Vec::new(),
        };
        let map = poisson_laplace(
            &problem(&system, &flux),
            &[2.65, 2.65],
            None,
            &LaplaceConfig::default(),
        );
        assert!(
            !map.is_converged(),
            "a rank-deficient precision must not report converged bars"
        );
    }
}
