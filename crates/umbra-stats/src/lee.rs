//! Look-elsewhere correction: turning a local `Z` into a global `p`.
//!
//! A scan over `10⁶` positions does not report the `p` of its best candidate, it
//! reports the chance that *some* candidate reaches that `Z` under `H0`. Spec
//! §5.6 names two layers and both live here:
//!
//! 1. **Monte Carlo null.** Draw background-only datasets, scan each, keep the
//!    maximum `Z`. The empirical tail of that maximum is the global `p` wherever
//!    there are toys to support it.
//! 2. **Gross–Vitells extrapolation.** From the mean Euler characteristic of the
//!    excursion set at a low threshold `u₀`, scale to high `u` with the known
//!    `χ²`-field law, so a `5σ` global `p` does not need `10⁷` toys.
//!
//! Toy `t` is drawn from `seed` and `t` alone, so a worker pool can split the
//! indices any way it likes and the calibration is bit-identical: nothing here
//! depends on draw order.

use std::f64::consts::PI;

use umbra_core::GridAny;

use crate::scan::{erfc, normal_p, z_map, ScanConfig, ScanProblem};

/// Threshold and dimensionality of the Euler extrapolation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LeeConfig {
    /// Low threshold `u₀`, in `Z`, at which the Euler characteristic is measured.
    ///
    /// Too high and few fields have an excursion, so the mean is noisy; too low
    /// and the extrapolation runs a long way from where it was calibrated.
    pub u0: f32,
    /// Search dimension: `2` for a plane, `3` for a volume.
    pub dim: u8,
}

impl Default for LeeConfig {
    /// `u₀ = 2.5σ`.
    ///
    /// Measured with `benches/lee.rs` on a 200² scene: anchoring at `1σ` puts the
    /// extrapolated `3σ` global `p` 2.7× above the Monte Carlo tail (0.93 against
    /// 0.33), because a 1σ excursion set still contains low-threshold structure
    /// that does not follow the `χ` scaling. At `2.5σ` the ratio is 1.3×. An
    /// anchor chosen too low is worse than no extrapolation: it looks like a
    /// `p`-value and is off by a factor of three.
    fn default() -> Self {
        Self { u0: 2.5, dim: 2 }
    }
}

/// One background-only toy: its peak `Z` and the Euler characteristic at `u₀`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToyResult {
    /// `max_c Z_c` over every voxel and scale.
    pub max_z: f32,
    /// Euler characteristic of `{Z > u₀}`.
    pub euler_at_u0: i32,
}

/// A calibrated null: the maximum-`Z` distribution and the extrapolation anchor.
#[derive(Debug, Clone, PartialEq)]
pub struct LeeCalibration {
    /// Number of toys behind the curve.
    pub n_toys: u32,
    /// Threshold the Euler characteristic was measured at.
    pub u0: f32,
    /// Search dimension.
    pub dim: u8,
    /// Mean Euler characteristic of the excursion set at `u0`.
    pub mean_euler_at_u0: f32,
    /// Standard error of that mean.
    pub euler_stderr: f32,
    /// Toys whose maximum exceeded `u0`.
    ///
    /// Zero means the Euler anchor was measured on empty sets and the
    /// extrapolation has nothing to stand on; the caller should lower `u0`.
    pub toys_above_u0: u32,
    /// Per-toy maxima, ascending.
    pub maxima: Vec<f32>,
    /// Content key of the configuration that produced this calibration.
    pub content_key: u64,
}

impl LeeCalibration {
    /// Assembles a calibration from per-toy results.
    ///
    /// Order-independent by construction: the maxima are sorted and the Euler
    /// mean does not depend on the sequence, so parallel workers can merge in
    /// any order and get the same object. `n_toys` must match `toys`.
    pub fn from_toys(
        lee: &LeeConfig,
        seed: u64,
        n_toys: u32,
        cells: usize,
        toys: &[ToyResult],
    ) -> Self {
        assert_eq!(
            toys.len(),
            n_toys as usize,
            "one result per toy: {n_toys} declared, {} supplied",
            toys.len()
        );
        let mut maxima: Vec<f32> = toys.iter().map(|toy| toy.max_z).collect();
        maxima.sort_by(|a, b| a.partial_cmp(b).expect("max Z is finite"));
        let n = toys.len().max(1) as f64;
        let mean = toys
            .iter()
            .map(|toy| f64::from(toy.euler_at_u0))
            .sum::<f64>()
            / n;
        let var = toys
            .iter()
            .map(|toy| (f64::from(toy.euler_at_u0) - mean).powi(2))
            .sum::<f64>()
            / n;
        Self {
            n_toys,
            u0: lee.u0,
            dim: lee.dim,
            mean_euler_at_u0: mean as f32,
            euler_stderr: (var / n).sqrt() as f32,
            toys_above_u0: toys.iter().filter(|toy| toy.max_z > lee.u0).count() as u32,
            maxima,
            content_key: content_key(lee, seed, n_toys, cells),
        }
    }

    /// Fraction of toys whose maximum reaches `z`, with the half-count
    /// correction so an empty tail reports a finite bound instead of zero.
    ///
    /// This is the layer that is only as good as its toy count: at `3σ` a
    /// 2 000-toy null has an expected count below one, which is exactly why the
    /// extrapolation exists.
    pub fn empirical_global_p(&self, z: f32) -> f32 {
        let exceed = self.maxima.iter().filter(|&&max| max >= z).count();
        ((exceed as f64 + 0.5) / (self.maxima.len() as f64 + 1.0)) as f32
    }

    /// Empirical quantile of the maximum, `q` in `[0, 1]`.
    pub fn quantile(&self, q: f32) -> f32 {
        if self.maxima.is_empty() {
            return 0.0;
        }
        let q = f64::from(q.clamp(0.0, 1.0));
        let index = ((q * self.maxima.len() as f64) as usize).min(self.maxima.len() - 1);
        self.maxima[index]
    }

    /// Global `p` from the Euler anchor, valid above `u₀`.
    ///
    /// `p_global(u) ≈ p_local(u) + ⟨χ(u₀)⟩ (u/u₀)^{d-1} exp(−(u² − u₀²)/2)`, the
    /// Gross–Vitells scaling for a `χ²` field: the local tail counts the peak
    /// itself, the second term counts the excursion regions that contain one.
    /// Below `u₀` nothing is extrapolated and the local `p` is returned.
    pub fn extrapolated_global_p(&self, z: f32) -> f32 {
        let local = f64::from(normal_p(f64::from(z)));
        let u0 = f64::from(self.u0);
        if !u0.is_finite() || u0 <= 0.0 || z <= self.u0 {
            return local.min(1.0) as f32;
        }
        let z = f64::from(z);
        let power = f64::from(self.dim.max(1) - 1);
        let chi = f64::from(self.mean_euler_at_u0).max(0.0);
        let extra = chi * (z / u0).powf(power) * (-(z * z - u0 * u0) / 2.0).exp();
        (local + extra).clamp(local, 1.0) as f32
    }
}

/// Runs `n_toys` background-only toys and assembles the calibration.
///
/// Convenience wrapper over [`lee_toy`]; a caller with a worker pool should map
/// [`lee_toy`] over the index range itself and merge with
/// [`LeeCalibration::from_toys`], which gives the same answer.
pub fn lee_calibrate(
    problem: &ScanProblem<'_>,
    cfg: &ScanConfig,
    lee: &LeeConfig,
    n_toys: u32,
    seed: u64,
) -> LeeCalibration {
    let toys: Vec<ToyResult> = (0..n_toys)
        .map(|toy| lee_toy(problem, cfg, lee, seed, toy))
        .collect();
    LeeCalibration::from_toys(lee, seed, n_toys, problem.system.voxel_count(), &toys)
}

/// One background-only toy, reproducible from `(seed, toy)` alone.
///
/// Counts are drawn from `λ⁰` with a Poisson sampler, then scanned exactly as
/// data would be. The `H0` model stays fixed, which is the point: the null is
/// the experiment with no void, not a shuffled observation.
pub fn lee_toy(
    problem: &ScanProblem<'_>,
    cfg: &ScanConfig,
    lee: &LeeConfig,
    seed: u64,
    toy: u32,
) -> ToyResult {
    let mut rng = Rng::new(toy_seed(seed, toy));
    let counts: Vec<f32> = problem
        .lambda0
        .iter()
        .map(|&lambda| rng.poisson(f64::from(lambda)) as f32)
        .collect();
    let toy_problem = ScanProblem {
        counts: &counts,
        ..*problem
    };
    let map = z_map(&toy_problem, cfg);
    let max_z = map.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let euler_at_u0 = match &problem.system.grid {
        GridAny::Two(g) => euler_characteristic_2d(&map, g.n[0] as usize, g.n[1] as usize, lee.u0),
        GridAny::Three(g) => euler_characteristic_3d(
            &map,
            [g.n[0] as usize, g.n[1] as usize, g.n[2] as usize],
            lee.u0,
        ),
    };
    ToyResult {
        max_z: if max_z.is_finite() {
            max_z.max(0.0)
        } else {
            0.0
        },
        euler_at_u0,
    }
}

/// Euler characteristic of `{field > level}` on a 2-D lattice.
///
/// Uses the pixel-complex count `χ = V − E + F` (cells minus shared edges plus
/// filled 2×2 blocks), which is the Euler characteristic of the union of closed
/// cells — the same object the excursion-set formula assumes.
pub fn euler_characteristic_2d(field: &[f32], nx: usize, ny: usize, level: f32) -> i32 {
    assert_eq!(field.len(), nx * ny, "field must be nx * ny");
    let set = |i: usize, j: usize| field[j * nx + i] > level;
    let mut v = 0i32;
    let mut e = 0i32;
    let mut f = 0i32;
    for j in 0..ny {
        for i in 0..nx {
            if !set(i, j) {
                continue;
            }
            v += 1;
            if i + 1 < nx && set(i + 1, j) {
                e += 1;
            }
            if j + 1 < ny && set(i, j + 1) {
                e += 1;
            }
            if i + 1 < nx && j + 1 < ny && set(i + 1, j) && set(i, j + 1) && set(i + 1, j + 1) {
                f += 1;
            }
        }
    }
    v - e + f
}

/// Euler characteristic of `{field > level}` on a voxel lattice: `V − E + F − C`.
pub fn euler_characteristic_3d(field: &[f32], n: [usize; 3], level: f32) -> i32 {
    let (nx, ny, nz) = (n[0], n[1], n[2]);
    assert_eq!(field.len(), nx * ny * nz, "field must be nx * ny * nz");
    let set = |i: usize, j: usize, k: usize| field[(k * ny + j) * nx + i] > level;
    let mut v = 0i32;
    let mut e = 0i32;
    let mut f = 0i32;
    let mut c = 0i32;
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                if !set(i, j, k) {
                    continue;
                }
                v += 1;
                if i + 1 < nx && set(i + 1, j, k) {
                    e += 1;
                }
                if j + 1 < ny && set(i, j + 1, k) {
                    e += 1;
                }
                if k + 1 < nz && set(i, j, k + 1) {
                    e += 1;
                }
                if i + 1 < nx
                    && j + 1 < ny
                    && set(i + 1, j, k)
                    && set(i, j + 1, k)
                    && set(i + 1, j + 1, k)
                {
                    f += 1;
                }
                if i + 1 < nx
                    && k + 1 < nz
                    && set(i + 1, j, k)
                    && set(i, j, k + 1)
                    && set(i + 1, j, k + 1)
                {
                    f += 1;
                }
                if j + 1 < ny
                    && k + 1 < nz
                    && set(i, j + 1, k)
                    && set(i, j, k + 1)
                    && set(i, j + 1, k + 1)
                {
                    f += 1;
                }
                if i + 1 < nx
                    && j + 1 < ny
                    && k + 1 < nz
                    && set(i + 1, j, k)
                    && set(i, j + 1, k)
                    && set(i, j, k + 1)
                    && set(i + 1, j + 1, k)
                    && set(i + 1, j, k + 1)
                    && set(i, j + 1, k + 1)
                    && set(i + 1, j + 1, k + 1)
                {
                    c += 1;
                }
            }
        }
    }
    v - e + f - c
}

/// FNV-1a over the configuration, so a cache can key on the calibration's inputs.
fn content_key(lee: &LeeConfig, seed: u64, n_toys: u32, cells: usize) -> u64 {
    fn eat(hash: &mut u64, value: u64) {
        for byte in value.to_le_bytes() {
            *hash ^= u64::from(byte);
            *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    eat(&mut hash, seed);
    eat(&mut hash, u64::from(n_toys));
    eat(&mut hash, u64::from(lee.dim));
    eat(&mut hash, cells as u64);
    eat(&mut hash, u64::from(lee.u0.to_bits()));
    hash
}

/// Seed of toy `toy`, mixed so adjacent indices are independent streams.
fn toy_seed(seed: u64, toy: u32) -> u64 {
    let mut rng = Rng::new(seed ^ u64::from(toy).wrapping_mul(0xa24b_aed4_963e_e407));
    rng.next_u64()
}

/// SplitMix64: small, fast, and reproducible across platforms.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform on `[0, 1)`.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Standard normal by Box–Muller.
    fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(f64::MIN_POSITIVE);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }

    /// Poisson draw by Knuth below 30 and Hörmann's PTRS above it.
    ///
    /// PTRS matters here: the tails are `λ ≈ 10³` counts, and Knuth's
    /// one-multiply-per-event loop would be a thousand steps per ray cell.
    fn poisson(&mut self, lambda: f64) -> u64 {
        if !lambda.is_finite() || lambda <= 0.0 {
            return 0;
        }
        if lambda < 30.0 {
            let limit = (-lambda).exp();
            let mut k = 0u64;
            let mut p = 1.0f64;
            loop {
                k += 1;
                p *= self.next_f64();
                if p <= limit {
                    return k - 1;
                }
            }
        }
        // Rejection sampling from a normal proposal: the discrete ratio
        // `pmf(k)/g(k)` is one at the mode, so the acceptance rate is O(1) and
        // nothing is truncated. Keying on the ratio rather than on a squeeze
        // constant is what keeps the variance exactly `lambda` — a squeeze that
        // is slightly too tight silently shaves the tails, which at `10³` counts
        // is a 25% variance deficit that no mean test would catch.
        let sigma = lambda.sqrt();
        let loglam = lambda.ln();
        let scale = 1.0 / (sigma * std::f64::consts::SQRT_2);
        // The proposal is "draw a normal, round it", so its mass at `k` is the
        // interval probability `Φ((k+½−λ)/σ) − Φ((k−½−λ)/σ)` — not the density at
        // `k`, which differs by `O(1/σ²)`.
        //
        // The envelope has to dominate the discrete ratio *everywhere*: where it
        // does not, the accept test clamps to one and the accepted density is
        // `min(pmf, envelope·mass)`, which shaves mid-range mass and shows up as
        // a variance deficit (2% at `λ = 50`, and shrinking with `λ` because the
        // ratio's oscillation does). Measured sup ratio ≈ 1 + 0.03 at `λ = 50`,
        // so the small-σ branch gets a wide envelope; it costs a few percent of
        // draws and nothing else.
        let envelope = if lambda < 200.0 {
            1.15f64.ln()
        } else {
            1.02f64.ln()
        };
        loop {
            let z = self.normal();
            let k = (lambda + sigma * z).round();
            if k < 0.0 {
                continue;
            }
            let hi = 0.5 * erfc(-(k + 0.5 - lambda) * scale);
            let lo = 0.5 * erfc(-(k - 0.5 - lambda) * scale);
            let mass = hi - lo;
            if mass <= 0.0 {
                continue;
            }
            let log_pmf = -lambda + k * loglam - ln_gamma(k + 1.0);
            if self.next_f64().ln() <= log_pmf - mass.ln() - envelope {
                return k as u64;
            }
        }
    }
}

/// Lanczos `ln Γ(x)` for `x > 0`, accurate to ~1e-10.
fn ln_gamma(x: f64) -> f64 {
    const G: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        return (PI / (PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = G[0];
    let t = x + 7.5;
    for (i, &g) in G.iter().enumerate().skip(1) {
        a += g / (x + i as f64);
    }
    0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scan::scan;
    use umbra_core::{Csr, Grid2, GridAny};
    use umbra_forward::{RayCell, System};

    fn scene_2d(n: usize) -> (System, Vec<f32>, Vec<f32>, Vec<f32>) {
        let grid = Grid2::new([n as u32, n as u32], [0.0, 0.0], 0.01);
        let mut indptr = vec![0u32];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for row in 0..n {
            for column in 0..n {
                indices.push((row * n + column) as u32);
                data.push(1.0f32);
            }
            indptr.push(indices.len() as u32);
        }
        let density = vec![2.65f32; grid.cell_count()];
        let system = System {
            l: Csr::new(indptr, indices, data),
            rays: (0..n)
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
            .map(|&x| 1.0e3 * (-x / 1.0e3).exp())
            .collect();
        let dlambda_dx: Vec<f32> = lambda0.iter().map(|&l| -l / 1.0e3).collect();
        (system, density, lambda0, dlambda_dx)
    }

    #[test]
    fn euler_counts_a_solid_square_as_one() {
        let (nx, ny) = (3, 3);
        let field = vec![1.0f32; nx * ny];
        assert_eq!(euler_characteristic_2d(&field, nx, ny, 0.5), 1);
    }

    #[test]
    fn euler_counts_a_hole_as_zero() {
        let (nx, ny) = (3, 3);
        let mut field = vec![1.0f32; nx * ny];
        field[4] = 0.0;
        assert_eq!(euler_characteristic_2d(&field, nx, ny, 0.5), 0);
    }

    #[test]
    fn euler_counts_separate_blobs() {
        let (nx, ny) = (3, 1);
        let field = [1.0f32, 0.0, 1.0];
        assert_eq!(euler_characteristic_2d(&field, nx, ny, 0.5), 2);
    }

    #[test]
    fn euler_of_an_empty_field_is_zero() {
        let field = vec![0.0f32; 16];
        assert_eq!(euler_characteristic_2d(&field, 4, 4, 0.5), 0);
        assert_eq!(euler_characteristic_3d(&field, [4, 2, 2], 0.5), 0);
    }

    #[test]
    fn euler_of_a_solid_cube_is_one() {
        let field = vec![1.0f32; 8];
        assert_eq!(euler_characteristic_3d(&field, [2, 2, 2], 0.5), 1);
    }

    #[test]
    fn poisson_matches_its_moments() {
        // Both sampler branches: Knuth under 30, rejection above it.
        for lambda in [12.0f64, 30.0, 50.0, 1.0e3] {
            let mut rng = Rng::new(7);
            let n = 200_000;
            let mut sum = 0.0f64;
            let mut sum2 = 0.0f64;
            for _ in 0..n {
                let k = rng.poisson(lambda) as f64;
                sum += k;
                sum2 += k * k;
            }
            let mean = sum / n as f64;
            let var = sum2 / n as f64 - mean * mean;
            assert!(
                (mean - lambda).abs() / lambda < 0.005,
                "λ {lambda}: mean {mean}"
            );
            assert!(
                (var - lambda).abs() / lambda < 0.02,
                "λ {lambda}: var {var} (a truncated tail shows up here, not in the mean)"
            );
        }
    }

    #[test]
    fn poisson_matches_the_exact_pmf_small_lambda() {
        let lambda = 12.0f64;
        let mut rng = Rng::new(11);
        let n = 200_000u32;
        let mut counts = [0u32; 40];
        for _ in 0..n {
            let k = rng.poisson(lambda) as usize;
            if k < counts.len() {
                counts[k] += 1;
            }
        }
        let mut pmf = (-lambda).exp();
        for (k, &count) in counts.iter().enumerate() {
            let observed = f64::from(count) / f64::from(n);
            let expected = pmf;
            if expected > 0.02 {
                assert!(
                    (observed - expected).abs() < 0.1 * expected,
                    "k = {k}: observed {observed:.4}, exact {expected:.4}"
                );
            }
            pmf *= lambda / (k as f64 + 1.0);
        }
    }

    #[test]
    fn ln_gamma_agrees_with_factorials() {
        let mut ln_fact = 0.0f64;
        for n in 1..=120 {
            if n > 1 {
                ln_fact += (n as f64 - 1.0).ln();
            }
            let got = ln_gamma(n as f64);
            assert!(
                (got - ln_fact).abs() < 1e-9,
                "ln_gamma({n}) = {got}, ln({} !) = {ln_fact}",
                n - 1
            );
        }
    }

    #[test]
    fn poisson_matches_the_exact_pmf_at_large_lambda() {
        // The exact pmf by recursion from `p(0) = e^{-λ}`, so no `ln_gamma` is
        // involved on the reference side: a sampler that loses tail mass or
        // inflates the mode shows up bin by bin, which a mean and a variance
        // cannot resolve separately.
        let lambda = 50.0f64;
        let bins = 121usize;
        let mut rng = Rng::new(3);
        let n = 400_000u32;
        let mut counts = [0u32; 121];
        for _ in 0..n {
            let k = rng.poisson(lambda) as usize;
            if k < bins {
                counts[k] += 1;
            }
        }
        let mut exacts = [0.0f64; 121];
        let mut pmf = (-lambda).exp();
        for (k, slot) in exacts.iter_mut().enumerate() {
            *slot = pmf;
            pmf *= lambda / (k as f64 + 1.0);
        }
        for (k, &expected) in exacts.iter().enumerate() {
            if expected < 0.005 {
                continue;
            }
            let observed = f64::from(counts[k]) / f64::from(n);
            let sd = (expected * (1.0 - expected) / f64::from(n)).sqrt();
            let sigma = (observed - expected) / sd;
            assert!(
                sigma.abs() < 4.0,
                "k = {k}: observed {observed:.5}, exact {expected:.5} ({sigma:.1}σ)"
            );
        }
    }

    #[test]
    fn toys_are_reproducible_and_index_independent() {
        let (system, density, lambda0, dlambda_dx) = scene_2d(12);
        let cfg = ScanConfig {
            scales_voxels: vec![1.0],
            top_k: 3,
            ..ScanConfig::default()
        };
        let lee = LeeConfig { u0: 1.0, dim: 2 };
        let problem = ScanProblem {
            system: &system,
            counts: &lambda0,
            lambda0: &lambda0,
            dlambda_dx: &dlambda_dx,
            density: &density,
            attenuation_gcm2: 1.0e3,
        };
        let a = lee_toy(&problem, &cfg, &lee, 42, 3);
        let b = lee_toy(&problem, &cfg, &lee, 42, 3);
        assert_eq!(a, b, "same seed and index must reproduce the toy");
        let c = lee_toy(&problem, &cfg, &lee, 42, 4);
        assert!(a != c, "adjacent toys must not share a stream");
    }

    #[test]
    fn calibration_reports_its_own_support() {
        let toys: Vec<ToyResult> = (0..10)
            .map(|t| ToyResult {
                max_z: 0.5 + t as f32 * 0.2,
                euler_at_u0: if t % 2 == 0 { 1 } else { 0 },
            })
            .collect();
        let lee = LeeConfig { u0: 1.0, dim: 2 };
        let calibration = LeeCalibration::from_toys(&lee, 9, 10, 1_000, &toys);
        assert!((calibration.mean_euler_at_u0 - 0.5).abs() < 1e-6);
        assert!(calibration.euler_stderr > 0.0);
        assert_eq!(calibration.toys_above_u0, 7);
        assert!(calibration.empirical_global_p(1.9) < calibration.empirical_global_p(0.4));
        assert_eq!(calibration.maxima.len(), 10);
        assert!(calibration.maxima.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn extrapolation_is_monotone_and_below_one() {
        let lee = LeeConfig { u0: 1.0, dim: 2 };
        let toys = vec![
            ToyResult {
                max_z: 1.0,
                euler_at_u0: 2
            };
            50
        ];
        let calibration = LeeCalibration::from_toys(&lee, 1, 50, 100, &toys);
        let a = calibration.extrapolated_global_p(2.0);
        let b = calibration.extrapolated_global_p(3.0);
        assert!(a > b, "{a} should exceed {b}");
        assert!(b > 0.0 && a <= 1.0);
        // At the anchor the formula adds the Euler term on top of the local tail.
        let anchor = calibration.extrapolated_global_p(0.9);
        assert!(
            (anchor - normal_p(0.9)).abs() < 1e-6,
            "no extrapolation below u0"
        );
    }

    /// Smooth unit-variance Gaussian field, squashed to `|G|` so that `Z²` is
    /// `χ²₁` at every point — the field family the Gross–Vitells scaling is
    /// derived for. Built from random plane waves, so it is smooth on a torus.
    fn chi2_field(nx: usize, ny: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng::new(seed);
        let mut acc = vec![0.0f64; nx * ny];
        for _ in 0..16 {
            let kx = (rng.next_u64() % 5) as f64 - 2.0;
            let ky = (rng.next_u64() % 5) as f64 - 2.0;
            let phase = rng.next_f64() * std::f64::consts::TAU;
            for j in 0..ny {
                for i in 0..nx {
                    let angle = std::f64::consts::TAU
                        * (kx * i as f64 / nx as f64 + ky * j as f64 / ny as f64)
                        + phase;
                    acc[j * nx + i] += angle.cos();
                }
            }
        }
        let mean = acc.iter().sum::<f64>() / acc.len() as f64;
        let var = acc.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / acc.len() as f64;
        let scale = 1.0 / var.sqrt();
        acc.iter()
            .map(|x| (((x - mean) * scale).abs()) as f32)
            .collect()
    }

    #[test]
    fn euler_scaling_matches_the_field_family() {
        let (nx, ny) = (40, 40);
        let (u0, u1) = (1.5f32, 2.5f32);
        let fields = 240;
        let mut chi0 = 0i64;
        let mut chi1 = 0i64;
        for t in 0..fields {
            let field = chi2_field(nx, ny, 0x51d3 + t as u64);
            chi0 += i64::from(euler_characteristic_2d(&field, nx, ny, u0));
            chi1 += i64::from(euler_characteristic_2d(&field, nx, ny, u1));
        }
        let m0 = chi0 as f64 / fields as f64;
        let m1 = chi1 as f64 / fields as f64;
        assert!(
            m0 > 0.0,
            "no excursion at u0, nothing to extrapolate from: {m0}"
        );
        let predicted = m0
            * (f64::from(u1) / f64::from(u0))
            * (-(f64::from(u1) * f64::from(u1) - f64::from(u0) * f64::from(u0)) / 2.0).exp();
        let ratio = m1 / predicted;
        assert!(
            (0.5..2.0).contains(&ratio),
            "measured χ ratio {ratio:.2} (χ(u0) = {m0:.2}, χ(u1) = {m1:.2}, predicted {predicted:.2})"
        );
    }

    #[test]
    fn z_map_peak_matches_the_scan_peak() {
        let (system, density, lambda0, dlambda_dx) = scene_2d(24);
        let cfg = ScanConfig {
            scales_voxels: vec![1.0, 2.0],
            top_k: 5,
            ..ScanConfig::default()
        };
        let counts: Vec<f32> = lambda0.iter().map(|&l| l * 1.02).collect();
        let problem = ScanProblem {
            system: &system,
            counts: &counts,
            lambda0: &lambda0,
            dlambda_dx: &dlambda_dx,
            density: &density,
            attenuation_gcm2: 1.0e3,
        };
        let result = scan(&problem, &cfg);
        let peak = result.peak.expect("the scene has a signal");
        let map = z_map(&problem, &cfg);
        assert!(
            (map[peak.voxel] - peak.z).abs() < 1e-4,
            "map {} vs scan {}",
            map[peak.voxel],
            peak.z
        );
    }

    #[test]
    fn calibration_from_toys_is_order_independent() {
        let lee = LeeConfig { u0: 1.0, dim: 2 };
        let toys: Vec<ToyResult> = (0..16)
            .map(|t| ToyResult {
                max_z: (t as f32 * 0.13).sin().abs() + 0.2,
                euler_at_u0: (t % 3) - 1,
            })
            .collect();
        let mut reversed = toys.clone();
        reversed.reverse();
        let a = LeeCalibration::from_toys(&lee, 5, 16, 256, &toys);
        let b = LeeCalibration::from_toys(&lee, 5, 16, 256, &reversed);
        assert_eq!(a, b, "merge order must not change the calibration");
        let other = LeeCalibration::from_toys(&lee, 6, 16, 256, &toys);
        assert_ne!(
            a.content_key, other.content_key,
            "the seed is part of the key"
        );
    }
}
