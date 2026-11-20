//! A6: the multi-scale profile-likelihood scan (spec §5.6).
//!
//! # The test
//!
//! `H0` is the reference structure; `H1(c, s, α)` adds a void template of centre
//! `c`, scale `s` and amplitude `α ∈ [0, 1]` — the fraction of the density
//! removed inside the template. With `λ_i(α) ≈ λ_i^0 + α g_i` near `H0`, the
//! score statistic is
//!
//! ```text
//! U_c = Σ_i g_i (y_i − λ_i^0) / λ_i^0,   I_c = Σ_i g_i² / λ_i^0,
//! Z_c = U_c / √I_c,                      α̂_c ≈ U_c / I_c,
//! ```
//!
//! and `g_i = Δ_i D_i`, where `D_i = −∂λ_i/∂X_i` is the (positive) sensitivity
//! of the expected counts to opacity and `Δ_i = (L (ρ ⊙ t_c))_i` is the opacity
//! the template would remove along ray `i`.
//!
//! # Why the numerator is one back projection
//!
//! Substituting `Δ_i = Σ_j l_ij ρ_j t_c(j)` into `U_c`:
//!
//! ```text
//! U_c = Σ_j ρ_j t_c(j) (Lᵀ w)_j,   w_i = D_i (y_i − λ_i^0) / λ_i^0,
//! ```
//!
//! so the whole numerator over every candidate is **one** sparse matvec `Lᵀ w`
//! followed by a correlation of that field with the template. That is the CT
//! back-projection adjoint, reused as an anomaly detector; it is why a scan of
//! `10⁶` candidates is affordable at all.
//!
//! `I_c` has no such factorisation — it carries `Δ_i²` — so it is the expensive
//! part, and it is why the footprint matters: a template only changes `Δ_i` for
//! rays that cross it. Accumulating those rays instead of summing all of them is
//! the difference between `10⁸` and `10¹²` operations, and
//! [`statistics`] is exact either way — the restriction is to the support, not
//! an approximation of it. Rendering the scan genuinely faster still (an angular
//! spatial index per detector, `~50` cells) is a further optimisation this
//! kernel does not need yet.
//!
//! # The null distribution
//!
//! `α ≥ 0` sits on the boundary of its range under `H0`, so the refined
//! likelihood-ratio statistic follows `½δ₀ + ½χ²₁`, not `χ²₁`. The one-sided
//! `p`-value is `½ P(χ²₁ > TS)`, which is what [`refine_lrt`] reports. Using the
//! two-sided `χ²₁` value instead is the classic mistake here: it doubles the
//! significance of every candidate.

use umbra_core::{Csr, GridAny};
use umbra_forward::System;

/// A scanned candidate: where the template sits and how significant it is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScanCandidate {
    /// Flat voxel index of the template centre.
    pub voxel: usize,
    /// Template scale index (multi-scale bank).
    pub scale: u8,
    /// The refined likelihood-ratio statistic, when [`refine_lrt`] ran.
    pub ts: Option<f32>,
    /// Local significance `Z = U / √I`.
    pub z: f32,
    /// Best-fit amplitude in `[0, 1]`.
    pub alpha_hat: f32,
    /// Local `p`-value: normal tail from the score, or the one-sided
    /// `½δ₀ + ½χ²₁` mixture once refined.
    pub local_p: f32,
}

/// Configuration of a multi-scale detection scan.
#[derive(Debug, Clone, PartialEq)]
pub struct ScanConfig {
    /// Template radii, in voxels. One template per entry.
    pub scales_voxels: Vec<f32>,
    /// Amplitude bounds `[min, max]` enforced during profile fits.
    pub alpha_bounds: [f32; 2],
    /// How many of the highest-scoring candidates to keep.
    pub top_k: usize,
    /// Whether nuisance parameters are profiled.
    ///
    /// Not implemented: profiling bulk density, flux, per-detector efficiency
    /// and alignment is #16. A scan that silently ignored them would report a
    /// significance it has not measured, so asking for it panics.
    pub profile_nuisances: bool,
}

impl Default for ScanConfig {
    fn default() -> Self {
        Self {
            scales_voxels: vec![1.0, 2.0, 4.0],
            alpha_bounds: [0.0, 1.0],
            top_k: 10,
            profile_nuisances: false,
        }
    }
}

/// The fixed inputs of a scan: the experiment, the data, and the `H0` model at
/// every ray cell.
#[derive(Debug, Clone, Copy)]
pub struct ScanProblem<'a> {
    /// Path-length operator and ray metadata.
    pub system: &'a System,
    /// Observed counts, one per ray cell.
    pub counts: &'a [f32],
    /// `λ_i^0`, the expected counts under the reference structure.
    pub lambda0: &'a [f32],
    /// `∂λ_i/∂X_i`, expected counts per `g cm⁻²`. Negative for transmission.
    pub dlambda_dx: &'a [f32],
    /// Reference density, one entry per voxel, in `g cm⁻³`. The template removes
    /// a fraction of *this*, so a void inside a denser region removes more
    /// opacity than the same fraction in bulk rock.
    pub density: &'a [f32],
    /// L0 attenuation length `Λ`, in `g cm⁻²`, used by the refinement.
    pub attenuation_gcm2: f32,
}

/// The result of a scan: the kept candidates, ordered by significance.
#[derive(Debug, Clone, PartialEq)]
pub struct ScanResult {
    /// Up to `top_k` candidates, highest `z` first.
    pub candidates: Vec<ScanCandidate>,
    /// The single most significant candidate, if the scan found any.
    pub peak: Option<ScanCandidate>,
}

/// A template footprint: offsets from the centre and their weights.
#[derive(Debug, Clone, PartialEq)]
struct Scale {
    radius: f32,
    offsets: Vec<(i32, i32, i32, f32)>,
}

/// The multi-scale template bank described by a [`ScanConfig`].
#[derive(Debug, Clone)]
pub struct TemplateBank {
    scales: Vec<Scale>,
}

impl TemplateBank {
    /// Builds disc templates of the given radii, in voxels, on `grid`.
    ///
    /// Weights are normalised to sum to one, so `α̂` is the fraction of density
    /// removed averaged over the footprint rather than a per-voxel amplitude that
    /// would grow with the template size.
    ///
    /// # Panics
    ///
    /// Panics if `radii` is empty or holds a non-positive radius.
    pub fn new(grid: &GridAny, radii: &[f32]) -> Self {
        assert!(!radii.is_empty(), "a scan needs at least one scale");
        let scales = radii
            .iter()
            .map(|&radius| {
                assert!(
                    radius.is_finite() && radius > 0.0,
                    "template radius must be positive, got {radius}"
                );
                Scale {
                    radius,
                    offsets: disc_offsets(grid, radius),
                }
            })
            .collect();
        Self { scales }
    }

    /// Number of scales in the bank.
    pub fn scale_count(&self) -> usize {
        self.scales.len()
    }

    /// Radius of scale `s`, in voxels.
    pub fn radius_voxels(&self, s: usize) -> f32 {
        self.scales[s].radius
    }

    /// Voxels in the footprint of scale `s`.
    pub fn footprint_cells(&self, s: usize) -> usize {
        self.scales[s].offsets.len()
    }
}

/// `U_c` and `I_c` for one candidate, from the exact support of the template.
///
/// Returns `None` when the template removes no opacity along any ray — an
/// unrecognised region, or a template in a vacuum — because `Z` would be `0/0`.
/// That is the NaN-free path, not an error.
pub fn statistics(
    problem: &ScanProblem<'_>,
    bank: &TemplateBank,
    scale: usize,
    voxel: usize,
) -> Option<(f64, f64)> {
    let transposed = transpose(&problem.system.l, problem.system.voxel_count());
    let back = back_projection(problem);
    let weights = ray_weights(problem);
    let mut footprint = Footprint::new(problem.system.ray_count());
    let offsets = &bank.scales[scale].offsets;
    footprint.candidate(&transposed, &back, &weights, problem, offsets, voxel)
}

/// Runs the score scan over every voxel and scale, keeping the best `top_k`.
///
/// # Panics
///
/// Panics if the problem's arrays disagree with the system's dimensions, or if
/// [`ScanConfig::profile_nuisances`] is set (see [`ScanConfig`]).
pub fn scan(problem: &ScanProblem<'_>, cfg: &ScanConfig) -> ScanResult {
    assert!(
        !cfg.profile_nuisances,
        "nuisance profiling is #16: a scan that ignored the nuisances would report \
         a significance it has not measured"
    );
    check_problem(problem);
    let bank = TemplateBank::new(&problem.system.grid, &cfg.scales_voxels);
    let transposed = transpose(&problem.system.l, problem.system.voxel_count());
    let back = back_projection(problem);
    let weights = ray_weights(problem);
    let mut footprint = Footprint::new(problem.system.ray_count());

    let mut best: Vec<ScanCandidate> = Vec::with_capacity(cfg.top_k.max(1));
    let mut peak: Option<ScanCandidate> = None;
    for (s, scale) in bank.scales.iter().enumerate() {
        for voxel in 0..problem.system.voxel_count() {
            let Some((u, i)) =
                footprint.candidate(&transposed, &back, &weights, problem, &scale.offsets, voxel)
            else {
                continue;
            };
            let z = u / i.sqrt();
            if !z.is_finite() {
                continue;
            }
            let candidate = ScanCandidate {
                voxel,
                scale: s as u8,
                ts: None,
                z: z as f32,
                alpha_hat: (u / i).clamp(
                    f64::from(cfg.alpha_bounds[0]),
                    f64::from(cfg.alpha_bounds[1]),
                ) as f32,
                // The tail is transcendental, and only the kept candidates and
                // the peak ever have their `p` read: computing it for every
                // candidate was one `exp` per voxel for nothing.
                local_p: 0.5,
            };
            if peak.is_none_or(|p| candidate.z > p.z) {
                peak = Some(candidate);
            }
            insert(&mut best, candidate, cfg.top_k.max(1));
        }
    }
    for candidate in &mut best {
        candidate.local_p = normal_p(f64::from(candidate.z));
    }
    if let Some(peak) = &mut peak {
        peak.local_p = normal_p(f64::from(peak.z));
    }
    ScanResult {
        candidates: best,
        peak,
    }
}

/// Refines the given candidates with the exact likelihood ratio and replaces
/// their `p`-value with the one-sided mixture (spec §5.6).
///
/// The profile maximises the Poisson log-likelihood over `α` inside
/// `alpha_bounds`; because `α` is one-dimensional and the derivative is monotone
/// in it, bisection finds the boundary or the stationary point in a fixed number
/// of steps without a line search. Under `H0`, `α ≥ 0` is on the boundary of the
/// parameter range, so `TS` follows `½δ₀ + ½χ²₁` and
/// `p = ½ P(χ²₁ > TS)`.
///
/// The refinement assumes the L0 transmission law `λ = k e^{−X/Λ}`, which is the
/// model the template's linear term is derived from; at L1 the score scan is
/// still a valid local test, but the profile would need the real flux integral
/// and this function would be fitting the wrong law.
///
/// # Panics
///
/// Panics if the problem's arrays disagree with the system's dimensions.
pub fn refine_lrt(
    problem: &ScanProblem<'_>,
    cfg: &ScanConfig,
    candidates: &[ScanCandidate],
) -> Vec<ScanCandidate> {
    check_problem(problem);
    let attenuation = f64::from(problem.attenuation_gcm2);
    assert!(
        attenuation > 0.0 && attenuation.is_finite(),
        "the L0 attenuation length must be positive"
    );
    let bank = TemplateBank::new(&problem.system.grid, &cfg.scales_voxels);
    let transposed = transpose(&problem.system.l, problem.system.voxel_count());
    let mut footprint = Footprint::new(problem.system.ray_count());
    let x0 = problem.system.opacity(problem.density);
    let k: Vec<f64> = problem
        .lambda0
        .iter()
        .zip(&x0)
        .map(|(&l, &x)| f64::from(l) * (f64::from(x) / attenuation).exp())
        .collect();

    candidates
        .iter()
        .map(|candidate| {
            let scale = &bank.scales[candidate.scale as usize];
            let delta = footprint.delta(&transposed, problem, &scale.offsets, candidate.voxel);
            let (lo, hi) = (
                f64::from(cfg.alpha_bounds[0]),
                f64::from(cfg.alpha_bounds[1]).max(f64::from(cfg.alpha_bounds[0])),
            );
            let alpha = profile_alpha(&delta, problem.counts, &k, &x0, attenuation, lo, hi);
            let ts = (2.0
                * (log_likelihood(&delta, problem.counts, &k, &x0, attenuation, alpha)
                    - log_likelihood(&delta, problem.counts, &k, &x0, attenuation, 0.0)))
            .max(0.0);
            ScanCandidate {
                ts: Some(ts as f32),
                alpha_hat: alpha as f32,
                local_p: chi2_mixture_p(ts),
                ..*candidate
            }
        })
        .collect()
}

/// One-sided `p`-value for a refined statistic under `½δ₀ + ½χ²₁`.
///
/// `p = ½ P(χ²₁ > TS) = ½ erfc(√(TS/2))`. At `TS = 0` it is `0.5`, which is the
/// half-mass at zero that the mixture carries.
pub fn chi2_mixture_p(ts: f64) -> f32 {
    let ts = ts.max(0.0);
    (0.5 * erfc((ts / 2.0).sqrt())).clamp(0.0, 0.5) as f32
}

/// One-sided normal tail `½ erfc(z/√2)`, clamped to `[0, 0.5]`.
fn normal_p(z: f64) -> f32 {
    if !z.is_finite() || z <= 0.0 {
        return 0.5;
    }
    (0.5 * erfc(z / std::f64::consts::SQRT_2)).clamp(0.0, 0.5) as f32
}

/// Complementary error function, Abramowitz & Stegun 7.1.26 (|ε| < 1.5e-7).
///
/// Enough for a `p`-value quoted to four significant figures, and it keeps the
/// crate free of a special-functions dependency for one call site.
fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * z);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let erf = 1.0 - poly * (-z * z).exp();
    if x >= 0.0 {
        1.0 - erf
    } else {
        1.0 + erf
    }
}

fn check_problem(problem: &ScanProblem<'_>) {
    let rays = problem.system.ray_count();
    assert_eq!(problem.counts.len(), rays, "one count per ray cell");
    assert_eq!(problem.lambda0.len(), rays, "lambda0 must cover every ray");
    assert_eq!(
        problem.dlambda_dx.len(),
        rays,
        "dlambda_dx must cover every ray"
    );
    assert_eq!(
        problem.density.len(),
        problem.system.voxel_count(),
        "density must cover every voxel"
    );
}

/// `D_i² / λ_i^0` per ray, zero where the ray carries no information.
fn ray_weights(problem: &ScanProblem<'_>) -> Vec<f32> {
    (0..problem.system.ray_count())
        .map(|ray| {
            let l0 = f64::from(problem.lambda0[ray]);
            if l0 <= 0.0 || !l0.is_finite() {
                return 0.0;
            }
            let d = -f64::from(problem.dlambda_dx[ray]);
            ((d * d) / l0) as f32
        })
        .collect()
}

/// `Lᵀ w` scaled by the density: the field the numerator correlates with.
fn back_projection(problem: &ScanProblem<'_>) -> Vec<f32> {
    let mut w = vec![0.0f32; problem.system.ray_count()];
    for (i, wi) in w.iter_mut().enumerate() {
        let l0 = f64::from(problem.lambda0[i]);
        let y = f64::from(problem.counts[i]);
        if l0 <= 0.0 || !l0.is_finite() || !y.is_finite() {
            // A ray with no expected counts carries no score information. Any
            // observation against it is impossible data, and the scan should not
            // turn that into a significant candidate.
            continue;
        }
        let d = -f64::from(problem.dlambda_dx[i]);
        *wi = (d * (y - l0) / l0) as f32;
    }
    let mut back = vec![0.0f32; problem.system.voxel_count()];
    problem.system.l.matvec_t(&w, &mut back);
    for (b, rho) in back.iter_mut().zip(problem.density) {
        *b *= *rho;
    }
    back
}

/// Scratch for the per-candidate footprint sum.
///
/// The generation stamp is what makes the reset free: a candidate touches a few
/// dozen rays, not the whole system, so clearing a `nrays` array per candidate
/// would dominate the scan.
struct Footprint {
    stamp: Vec<u32>,
    delta: Vec<f32>,
    touched: Vec<u32>,
    generation: u32,
}

impl Footprint {
    fn new(rays: usize) -> Self {
        Self {
            stamp: vec![0; rays],
            delta: vec![0.0; rays],
            touched: Vec::new(),
            generation: 0,
        }
    }

    /// `Δ_i` for the rays the template crosses, in `g cm⁻²`.
    fn delta(
        &mut self,
        transposed: &Csr<f32>,
        problem: &ScanProblem<'_>,
        offsets: &[(i32, i32, i32, f32)],
        voxel: usize,
    ) -> Vec<(usize, f32)> {
        self.accumulate(transposed, problem, offsets, voxel, None);
        self.touched
            .iter()
            .map(|&ray| (ray as usize, self.delta[ray as usize]))
            .collect()
    }

    /// `(U_c, I_c)` for one candidate, or `None` when it has no support.
    fn candidate(
        &mut self,
        transposed: &Csr<f32>,
        back: &[f32],
        weights: &[f32],
        problem: &ScanProblem<'_>,
        offsets: &[(i32, i32, i32, f32)],
        voxel: usize,
    ) -> Option<(f64, f64)> {
        // The weight is part of the score: dropping it inflates `U` by the
        // footprint size, which is a factor of 13 at radius 2 and looks exactly
        // like a strong detection. `U` and `Δ` are gathered in one pass — each
        // footprint cell is visited once, not once per quantity.
        let u = self.accumulate(transposed, problem, offsets, voxel, Some(back));
        // `weights` is `D²/λ0` per ray, so the inner loop is one multiply per
        // touched ray rather than a division and two conversions.
        let mut i = 0.0f32;
        for &ray in &self.touched {
            let delta = self.delta[ray as usize];
            i += delta * delta * weights[ray as usize];
        }
        let i = f64::from(i);
        if i <= 0.0 || !i.is_finite() || !u.is_finite() {
            return None;
        }
        Some((u, i))
    }

    /// Walks the footprint once: accumulates `Δ_i` on every ray it crosses, and
    /// returns `U_c` when `back` is supplied.
    ///
    /// The centre is decomposed into axes once per candidate and the bounds
    /// check is additive, which is what keeps the inner loop off the integer
    /// divisions `offset_index` would need for every offset.
    fn accumulate(
        &mut self,
        transposed: &Csr<f32>,
        problem: &ScanProblem<'_>,
        offsets: &[(i32, i32, i32, f32)],
        voxel: usize,
        back: Option<&[f32]>,
    ) -> f64 {
        self.begin();
        let mut u = 0.0f64;
        let density = problem.density;
        match &problem.system.grid {
            GridAny::Two(g) => {
                let (nx, ny) = (g.n[0] as i32, g.n[1] as i32);
                let (ci, cj) = (voxel as i32 % nx, voxel as i32 / nx);
                for &(dx, dy, _, weight) in offsets {
                    let (i, j) = (ci + dx, cj + dy);
                    if i < 0 || j < 0 || i >= nx || j >= ny {
                        continue;
                    }
                    let index = j as usize * nx as usize + i as usize;
                    if let Some(back) = back {
                        u += f64::from(weight) * f64::from(back[index]);
                    }
                    let removed = density[index] * weight;
                    if removed != 0.0 {
                        self.push(transposed, index, removed);
                    }
                }
            }
            GridAny::Three(g) => {
                let (nx, ny, nz) = (g.n[0] as i32, g.n[1] as i32, g.n[2] as i32);
                let plane = (nx * ny) as usize;
                let ci = voxel as i32 % nx;
                let cj = (voxel as i32 / nx) % ny;
                let ck = voxel as i32 / (nx * ny);
                for &(dx, dy, dz, weight) in offsets {
                    let (i, j, k) = (ci + dx, cj + dy, ck + dz);
                    if i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz {
                        continue;
                    }
                    let index = k as usize * plane + j as usize * nx as usize + i as usize;
                    if let Some(back) = back {
                        u += f64::from(weight) * f64::from(back[index]);
                    }
                    let removed = density[index] * weight;
                    if removed != 0.0 {
                        self.push(transposed, index, removed);
                    }
                }
            }
        }
        u
    }

    #[inline]
    fn push(&mut self, transposed: &Csr<f32>, index: usize, removed: f32) {
        // Direct slice access rather than `Csr::row`: the iterator's per-entry
        // bound checks land in the innermost loop of the whole scan.
        let start = transposed.indptr[index] as usize;
        let end = transposed.indptr[index + 1] as usize;
        for position in start..end {
            let ray = transposed.indices[position] as usize;
            let length = transposed.data[position];
            if self.stamp[ray] != self.generation {
                self.stamp[ray] = self.generation;
                self.delta[ray] = 0.0;
                self.touched.push(ray as u32);
            }
            self.delta[ray] += length * removed;
        }
    }

    fn begin(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.stamp.iter_mut().for_each(|s| *s = 0);
            self.generation = 1;
        }
        self.touched.clear();
    }
}

/// Poisson log-likelihood over the rays a template touches, at amplitude `alpha`.
///
/// The positive log-likelihood, not the NLL: the profile maximises it, and `TS`
/// is twice its increase over `H0`. Returning the NLL here flips the sign of
/// every refined statistic while leaving the amplitude (which is found by
/// bisection on the derivative) correct, so the two disagree silently.
fn log_likelihood(
    delta: &[(usize, f32)],
    counts: &[f32],
    k: &[f64],
    x0: &[f32],
    attenuation: f64,
    alpha: f64,
) -> f64 {
    let mut total = 0.0f64;
    for &(ray, d) in delta {
        let x = f64::from(x0[ray]) - alpha * f64::from(d);
        let lambda = k[ray] * (-(x / attenuation)).exp();
        let y = f64::from(counts[ray]);
        if lambda > 0.0 && lambda.is_finite() {
            total += y * lambda.ln() - lambda;
        }
    }
    total
}

/// `α̂` maximising the profile likelihood, by bisection on the derivative.
fn profile_alpha(
    delta: &[(usize, f32)],
    counts: &[f32],
    k: &[f64],
    x0: &[f32],
    attenuation: f64,
    lo: f64,
    hi: f64,
) -> f64 {
    let derivative = |alpha: f64| {
        let mut d = 0.0f64;
        for &(ray, step) in delta {
            let x = f64::from(x0[ray]) - alpha * f64::from(step);
            let lambda = k[ray] * (-(x / attenuation)).exp();
            if lambda > 0.0 && lambda.is_finite() {
                d += (f64::from(counts[ray]) - lambda) * f64::from(step) / attenuation;
            }
        }
        d
    };
    // The likelihood falls when counts are below the prediction, so a negative
    // derivative at the lower bound means the unconstrained maximum is below it.
    if derivative(lo) <= 0.0 {
        return lo;
    }
    if derivative(hi) >= 0.0 {
        return hi;
    }
    let (mut a, mut b) = (lo, hi);
    for _ in 0..80 {
        let mid = 0.5 * (a + b);
        if derivative(mid) > 0.0 {
            a = mid;
        } else {
            b = mid;
        }
    }
    0.5 * (a + b)
}

fn insert(best: &mut Vec<ScanCandidate>, candidate: ScanCandidate, top_k: usize) {
    let position = best
        .iter()
        .position(|existing| candidate.z > existing.z)
        .unwrap_or(best.len());
    if position < top_k {
        best.insert(position, candidate);
        best.truncate(top_k);
    }
}

/// Offsets of a disc of `radius` voxels, weights normalised to one.
fn disc_offsets(grid: &GridAny, radius: f32) -> Vec<(i32, i32, i32, f32)> {
    let r = radius.ceil() as i32;
    let r2 = radius * radius;
    let mut offsets = Vec::new();
    match grid {
        GridAny::Two(_) => {
            for dy in -r..=r {
                for dx in -r..=r {
                    if (dx * dx + dy * dy) as f32 <= r2 {
                        offsets.push((dx, dy, 0, 1.0));
                    }
                }
            }
        }
        GridAny::Three(_) => {
            for dz in -r..=r {
                for dy in -r..=r {
                    for dx in -r..=r {
                        if (dx * dx + dy * dy + dz * dz) as f32 <= r2 {
                            offsets.push((dx, dy, dz, 1.0));
                        }
                    }
                }
            }
        }
    }
    let n = offsets.len() as f32;
    for offset in &mut offsets {
        offset.3 = 1.0 / n;
    }
    offsets
}

/// `Aᵀ` as a CSR matrix, so a voxel's incident rays are one contiguous row.
fn transpose(l: &Csr<f32>, ncols: usize) -> Csr<f32> {
    let mut indptr = vec![0u32; ncols + 1];
    for &column in &l.indices {
        indptr[column as usize + 1] += 1;
    }
    for i in 0..ncols {
        indptr[i + 1] += indptr[i];
    }
    let mut cursor = indptr.clone();
    let mut indices = vec![0u32; l.nnz()];
    let mut data = vec![0.0f32; l.nnz()];
    for row in 0..l.nrows() {
        for (column, value) in l.row(row) {
            let position = cursor[column] as usize;
            indices[position] = row as u32;
            data[position] = value;
            cursor[column] += 1;
        }
    }
    Csr::new(indptr, indices, data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use umbra_core::{Grid2, GridAny};
    use umbra_forward::RayCell;

    const LAMBDA: f32 = 1000.0;

    /// A 10×10 grid with eight parallel rays, one per row of the grid's middle.
    ///
    /// Hand-built rather than DDA-traced so every expectation below is checked
    /// against a model that can be written down: a ray crosses all ten cells of
    /// its row with length `10·h`.
    fn model() -> (System, Vec<f32>, Vec<f32>, Vec<f32>) {
        model_with_rows(&[1, 2, 3, 4, 5, 6, 7, 8])
    }

    /// The same scene with rays only in `rows`, so a test can leave a region
    /// genuinely unilluminated.
    fn model_with_rows(rows: &[usize]) -> (System, Vec<f32>, Vec<f32>, Vec<f32>) {
        let grid = Grid2::new([10, 10], [0.0, 0.0], 0.1);
        let mut indptr = vec![0u32];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        let mut density = vec![2.65f32; 100];
        // A buried low-density block: rows 4–5, columns 4–5.
        for j in 4..6 {
            for i in 4..6 {
                density[j * 10 + i] = 0.5;
            }
        }
        for &row in rows {
            for i in 0..10usize {
                indices.push((row * 10 + i) as u32);
                data.push(1.0f32);
            }
            indptr.push(indices.len() as u32);
        }
        let system = System {
            l: Csr::new(indptr, indices, data),
            rays: (0..rows.len())
                .map(|_| RayCell {
                    det: 0,
                    zen: 0.0,
                    az: 0.0,
                    k_scale: 1.0,
                })
                .collect(),
            grid: GridAny::Two(grid),
        };
        let opacity = system.opacity(&density);
        let lambda0: Vec<f32> = opacity
            .iter()
            .map(|&x| LAMBDA * (-x / 1000.0).exp())
            .collect();
        let dlambda_dx: Vec<f32> = lambda0.iter().map(|&l| -l / 1000.0).collect();
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
            attenuation_gcm2: 1000.0,
        }
    }

    /// Deterministic pseudo-random numbers, so a failure reproduces exactly.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            self.0 >> 33
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() as usize) % n
        }
    }

    /// `(U, I)` from the literal definition: every ray, every template voxel.
    fn brute_forces(
        problem: &ScanProblem<'_>,
        bank: &TemplateBank,
        scale: usize,
        voxel: usize,
    ) -> (f64, f64) {
        let (system, counts, lambda0, dlambda_dx, density) = (
            problem.system,
            problem.counts,
            problem.lambda0,
            problem.dlambda_dx,
            problem.density,
        );
        let mut u = 0.0f64;
        let mut i = 0.0f64;
        for ray in 0..system.ray_count() {
            let l0 = f64::from(lambda0[ray]);
            if l0 <= 0.0 || !l0.is_finite() {
                continue;
            }
            // The row's chord through the template, from the operator itself.
            let row_delta: f64 = system
                .l
                .row(ray)
                .map(|(column, length)| {
                    if column_in_template(&system.grid, voxel, column, bank, scale) {
                        f64::from(length)
                            * f64::from(density[column])
                            * f64::from(template_weight(&system.grid, voxel, column, bank, scale))
                    } else {
                        0.0
                    }
                })
                .sum();
            let g = row_delta * -f64::from(dlambda_dx[ray]);
            u += g * (f64::from(counts[ray]) - l0) / l0;
            i += g * g / l0;
        }
        (u, i)
    }

    fn column_in_template(
        grid: &GridAny,
        centre: usize,
        column: usize,
        bank: &TemplateBank,
        scale: usize,
    ) -> bool {
        template_weight(grid, centre, column, bank, scale) > 0.0
    }

    fn template_weight(
        grid: &GridAny,
        centre: usize,
        column: usize,
        bank: &TemplateBank,
        scale: usize,
    ) -> f32 {
        let GridAny::Two(g) = grid else {
            return 0.0;
        };
        let (ci, cj) = (
            (centre % g.n[0] as usize) as i32,
            (centre / g.n[0] as usize) as i32,
        );
        let (i, j) = (
            (column % g.n[0] as usize) as i32,
            (column / g.n[0] as usize) as i32,
        );
        for &(dx, dy, _, weight) in &bank.scales[scale].offsets {
            if ci + dx == i && cj + dy == j {
                return weight;
            }
        }
        0.0
    }

    #[test]
    fn the_score_matches_a_brute_force_recomputation() {
        let (system, density, lambda0, dlambda_dx) = model();
        let counts: Vec<f32> = lambda0.iter().map(|l| l * 1.02).collect();
        let p = problem(&system, &counts, &lambda0, &dlambda_dx, &density);
        let bank = TemplateBank::new(&system.grid, &[1.0, 2.0]);
        let mut rng = Lcg(7);
        for _ in 0..50 {
            let voxel = rng.below(system.voxel_count());
            let scale = rng.below(bank.scale_count());
            let Some((u, i)) = statistics(&p, &bank, scale, voxel) else {
                continue;
            };
            let (bu, bi) = brute_forces(&p, &bank, scale, voxel);
            if bi == 0.0 {
                assert!(i == 0.0, "brute force says no information, scan says {i}");
                continue;
            }
            assert!(
                (u - bu).abs() <= 1e-6 * bu.abs().max(1.0),
                "U at voxel {voxel} scale {scale}: scan {u}, brute {bu}"
            );
            assert!(
                (i - bi).abs() <= 1e-6 * bi.abs(),
                "I at voxel {voxel} scale {scale}: scan {i}, brute {bi}"
            );
        }
    }

    #[test]
    fn the_footprint_restriction_agrees_with_the_full_sum() {
        // The restriction is to the template's support, not an approximation of
        // it: every ray with a non-zero chord is accumulated, so the two sums
        // must agree to rounding.
        let (system, density, lambda0, dlambda_dx) = model();
        let counts: Vec<f32> = lambda0.iter().map(|l| l * 1.02).collect();
        let p = problem(&system, &counts, &lambda0, &dlambda_dx, &density);
        let bank = TemplateBank::new(&system.grid, &[1.0, 2.0, 3.0]);
        let mut worst = 0.0f64;
        for voxel in 0..system.voxel_count() {
            for scale in 0..bank.scale_count() {
                let Some((_, i)) = statistics(&p, &bank, scale, voxel) else {
                    continue;
                };
                let (_, bi) = brute_forces(&p, &bank, scale, voxel);
                if bi > 0.0 {
                    worst = worst.max((i - bi).abs() / bi);
                }
            }
        }
        assert!(worst <= 1.0e-5, "footprint sum differs by {worst:.2e}");
    }

    #[test]
    fn a_buried_void_has_a_positive_score_and_a_sane_amplitude() {
        let (system, density, lambda0, _) = model();
        let counts: Vec<f32> = lambda0.iter().map(|l| l * 1.05).collect();
        let dlambda_dx: Vec<f32> = lambda0.iter().map(|&l| -l / 1000.0).collect();
        let p = problem(&system, &counts, &lambda0, &dlambda_dx, &density);
        let result = scan(&p, &ScanConfig::default());
        let peak = result.peak.expect("a peak");
        assert!(peak.z > 2.0, "z = {}", peak.z);
        assert!(
            peak.alpha_hat > 0.0 && peak.alpha_hat <= 1.0,
            "amplitude {} outside [0, 1]",
            peak.alpha_hat
        );
    }

    #[test]
    fn zero_count_bins_are_nan_free() {
        let (system, density, lambda0, dlambda_dx) = model();
        let counts = vec![0.0f32; system.ray_count()];
        let p = problem(&system, &counts, &lambda0, &dlambda_dx, &density);
        let result = scan(&p, &ScanConfig::default());
        for candidate in &result.candidates {
            assert!(candidate.z.is_finite(), "z = {}", candidate.z);
            assert!(candidate.alpha_hat.is_finite());
            assert!((0.0..=0.5).contains(&candidate.local_p));
        }
        if let Some(peak) = result.peak {
            assert!(peak.z.is_finite());
        }
    }

    #[test]
    fn a_template_no_ray_crosses_yields_no_candidate() {
        // Unseen voxels have no information, so they are skipped rather than
        // reported with a zero-width interval. The rays stop at row 3, so a
        // template in the last row sees nothing at all.
        let (system, density, lambda0, dlambda_dx) = model_with_rows(&[0, 1, 2, 3]);
        let counts: Vec<f32> = lambda0.iter().map(|l| l * 1.01).collect();
        let p = problem(&system, &counts, &lambda0, &dlambda_dx, &density);
        let bank = TemplateBank::new(&system.grid, &[1.0]);
        assert!(statistics(&p, &bank, 0, 9 * 10 + 9).is_none());
        // And the scan never reports it, even with a tiny top_k.
        let cfg = ScanConfig {
            scales_voxels: vec![1.0],
            top_k: 1,
            ..ScanConfig::default()
        };
        let peak = scan(&p, &cfg).peak.expect("the lit rows still score");
        assert!(peak.voxel < 5 * 10, "peak at unseen voxel {}", peak.voxel);
    }

    /// Golden-section maximisation of the profile, independent of the bisection
    /// the refinement uses.
    fn brute_profile(
        delta: &[(usize, f32)],
        counts: &[f32],
        k: &[f64],
        x0: &[f32],
        hi: f64,
    ) -> (f64, f64) {
        let f = |a: f64| log_likelihood(delta, counts, k, x0, 1000.0, a);
        let (mut a, mut b) = (0.0f64, hi);
        let phi = 0.5 * (5.0f64.sqrt() - 1.0);
        let mut c = b - phi * (b - a);
        let mut d = a + phi * (b - a);
        for _ in 0..200 {
            if f(c) > f(d) {
                b = d;
            } else {
                a = c;
            }
            c = b - phi * (b - a);
            d = a + phi * (b - a);
        }
        let alpha = 0.5 * (a + b);
        (alpha, f(alpha) - f(0.0))
    }

    #[test]
    fn the_refinement_matches_a_brute_force_profile_likelihood() {
        let (system, density, lambda0, dlambda_dx) = model();
        let bank = TemplateBank::new(&system.grid, &[2.0]);
        let centre = 4 * 10 + 4; // inside the buried block
        let cfg = ScanConfig {
            scales_voxels: vec![2.0],
            ..ScanConfig::default()
        };

        // Generate counts under a known amplitude, using the exact L0 law.
        let x0 = system.opacity(&density);
        let probe = problem(&system, &lambda0, &lambda0, &dlambda_dx, &density);
        let delta: Vec<(usize, f32)> = {
            let transposed = transpose(&system.l, system.voxel_count());
            let mut footprint = Footprint::new(system.ray_count());
            footprint.delta(&transposed, &probe, &bank.scales[0].offsets, centre)
        };
        let truth = 0.6f64;
        let mut x = x0.clone();
        for &(ray, d) in &delta {
            x[ray] -= truth as f32 * d;
        }
        let counts: Vec<f32> = x.iter().map(|&xi| LAMBDA * (-xi / 1000.0).exp()).collect();

        let p = problem(&system, &counts, &lambda0, &dlambda_dx, &density);
        let seed = ScanCandidate {
            voxel: centre,
            scale: 0,
            ts: None,
            z: 0.0,
            alpha_hat: 0.0,
            local_p: 0.5,
        };
        let refined = refine_lrt(&p, &cfg, &[seed]);
        let candidate = refined[0];

        let k: Vec<f64> = lambda0
            .iter()
            .zip(&x0)
            .map(|(&l, &xi)| f64::from(l) * (f64::from(xi) / 1000.0).exp())
            .collect();
        let (alpha, ts) = brute_profile(&delta, &counts, &k, &x0, 1.0);
        let ts = 2.0 * ts;

        assert!(
            (f64::from(candidate.alpha_hat) - alpha).abs() < 1.0e-4,
            "alpha: refinement {}, brute {alpha}",
            candidate.alpha_hat
        );
        assert!(
            (f64::from(candidate.ts.unwrap()) - ts).abs() <= 1.0e-3 * ts,
            "TS: refinement {:?}, brute {ts}",
            candidate.ts
        );
        // The mixture, not chi-squared: at the 95th percentile of chi2_1 the
        // one-sided p is 0.025, and half of it would be 0.0125.
        assert!(
            (candidate.local_p - chi2_mixture_p(ts)).abs() < 1.0e-6,
            "p = {}",
            candidate.local_p
        );
        // A removed 60% of the density must never be *less* interesting than no
        // void at all, which is the one direction the mixture's half-mass at
        // zero could get wrong.
        assert!(
            candidate.local_p < 0.5,
            "p = {} did not fall below the H0 half-mass",
            candidate.local_p
        );
    }

    #[test]
    fn the_one_sided_mixture_puts_half_its_mass_at_zero() {
        assert!((chi2_mixture_p(0.0) - 0.5).abs() < 1e-9);
        // 3.841 is the 95th percentile of chi2_1.
        assert!((chi2_mixture_p(3.841) - 0.025).abs() < 5e-4);
        assert!(chi2_mixture_p(-1.0) == 0.5);
        assert!(chi2_mixture_p(100.0) < 1e-6);
    }

    #[test]
    fn the_maximum_under_the_null_has_the_inflated_tail() {
        // Under H0 every candidate's Z is a fluctuation, so the maximum over a
        // scan is larger than any single Z: that inflation is exactly what the
        // look-evenwhere correction in #15 has to measure.
        let (system, density, lambda0, dlambda_dx) = model();
        let cfg = ScanConfig {
            scales_voxels: vec![1.0],
            top_k: 1,
            ..ScanConfig::default()
        };
        let mut rng = Lcg(11);
        let mut maxima = Vec::new();
        for _ in 0..200 {
            let counts: Vec<f32> = lambda0
                .iter()
                .map(|&l| {
                    // Knuth's method; lambda0 is ~730 here, so use a normal
                    // approximation for the bulk and cap at zero.
                    let z = normal(&mut rng) as f32;
                    (l + z * l.sqrt()).max(0.0)
                })
                .collect();
            let p = problem(&system, &counts, &lambda0, &dlambda_dx, &density);
            if let Some(peak) = scan(&p, &cfg).peak {
                maxima.push(peak.z as f64);
            }
        }
        maxima.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = maxima.iter().sum::<f64>() / maxima.len() as f64;
        let mut histogram = [0usize; 6];
        for &m in &maxima {
            let bin = ((m / 1.0).floor() as usize).min(5);
            histogram[bin] += 1;
        }
        assert!(
            mean > 1.0,
            "max Z under H0 averaged {mean:.2}: no look-elsewhere inflation"
        );
        println!("H0 max Z: mean {mean:.2}, histogram {histogram:?}");
    }

    /// Standard normal by Box–Muller, on the same LCG as everything else.
    fn normal(rng: &mut Lcg) -> f64 {
        let u1 = (rng.next() as f64 + 1.0) / (u32::MAX as f64 + 2.0);
        let u2 = (rng.next() as f64 + 1.0) / (u32::MAX as f64 + 2.0);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}
