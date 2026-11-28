//! A11: simulation-based calibration (SBC).
//!
//! SBC asks one question of an inference engine: *do its posterior intervals
//! cover the truth at the rate they claim?* The construction is Talts et al.
//! (2018): draw `θ` from the prior, simulate data from `θ`, run the engine on
//! that data, and record the rank of the true `θ` among the engine's posterior
//! draws. Under a calibrated engine the rank is uniform on `[0, 1]`; a `θ` the
//! engine thinks it knows better than it does produces ranks bunched at the
//! centre, and one it second-guesses produces ranks at the ends.
//!
//! # Why this module is engine-agnostic
//!
//! The rank machinery is the instrument, and it must be checkable against an
//! engine whose calibration is known in closed form before it is pointed at a
//! pipeline whose calibration is the question. [`Engine`] is therefore a trait
//! over "prior, simulator, posterior sampler", and the tests in this file run
//! it on a Gaussian engine that is calibrated by construction — then on the
//! same engine with its posterior inflated by 2×, which must fail. A harness
//! that cannot fail is not a gate.
//!
//! # Ranks and ties
//!
//! `#{x < θ} + U·#{x = θ}` over `M` draws, with `U ~ Uniform(0, 1)`, is the
//! randomised rank. Randomising the ties is what makes the statistic continuous
//! on `[0, 1]` and lets an ordinary Kolmogorov–Smirnov test against the uniform
//! distribution be used, instead of a conservative discrete test on `M + 1`
//! buckets that would need `M` large to say anything.
//!
//! # Why the verdict is pooled and not per-parameter
//!
//! A density field is `O(10³)` parameters. Testing each at `p > 0.01` means a
//! calibrated engine fails one in ~70 runs of the gate by chance alone, which
//! makes the gate noise rather than evidence. The deciding statistic is the
//! **pooled** rank set over every parameter, whose KS `p` is compared with 0.01;
//! per-parameter `p` values are reported so a reader can see *where* an engine
//! is wrong, and the count below 0.01 is recorded rather than gated on.

use umbra_core::rng::Philox;

/// Stream id for the harness's own draws, kept apart from the engine's streams.
const SBC_STREAM: u64 = 0x5B_C0_15_CA_1B_00_00_01;

/// What an engine under test has to provide.
///
/// `simulate` takes the prior draw because an engine may carry a forward model
/// that the caller cannot reconstruct; the harness never touches the physics.
pub trait Engine {
    /// Number of parameters, i.e. the length of every vector here.
    fn parameter_count(&self) -> usize;
    /// A draw from the prior.
    fn prior(&mut self, rng: &mut Philox) -> Vec<f32>;
    /// A dataset generated at `theta`.
    fn simulate(&self, theta: &[f32], rng: &mut Philox) -> Vec<f32>;
    /// `M` posterior draws at `data`. `theta` is supplied only so an engine can
    /// report how many draws it made; a correct engine must not look at it.
    fn posterior(&mut self, theta: &[f32], data: &[f32], rng: &mut Philox) -> Vec<Vec<f32>>;
    /// Stable name for the report's verdict line.
    fn name(&self) -> &str;
}

/// Size of the run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SbcConfig {
    /// Independent prior draws. 10⁴ is spec §5.7's gate size.
    pub draws: u32,
    /// Posterior draws per dataset, `M`.
    pub posterior: u32,
    /// Seed for the harness's streams and the engine's.
    pub seed: u64,
    /// Histogram bins in the report. Ten is readable; 20 resolves the tails.
    pub bins: usize,
}

impl Default for SbcConfig {
    fn default() -> Self {
        Self {
            draws: 1000,
            posterior: 100,
            seed: 7,
            bins: 10,
        }
    }
}

impl SbcConfig {
    /// Rough cost of a decision at `ms_per_draw`, at the configured size.
    pub fn estimated_seconds(&self, ms_per_draw: f64) -> f64 {
        self.draws as f64 * ms_per_draw / 1000.0
    }
}

/// One parameter's rank histogram and uniform-fit statistic.
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterRanks {
    /// Index in the engine's parameter vector.
    pub index: usize,
    /// Counts per bin over `[0, 1]`.
    pub histogram: Vec<u32>,
    /// Kolmogorov–Smirnov statistic against `Uniform(0, 1)`.
    pub ks: f64,
    /// Its p-value. Small means the ranks are not uniform.
    pub p: f64,
    /// Ranks that went into the test. Fewer than `draws` when an engine
    /// declined to produce a posterior for some datasets.
    pub ranks: u32,
}

/// The whole run.
#[derive(Debug, Clone, PartialEq)]
pub struct SbcReport {
    /// Engine name, for the verdict line.
    pub engine: String,
    /// The configuration used.
    pub config: SbcConfig,
    /// Per-parameter results, in parameter order.
    pub parameters: Vec<ParameterRanks>,
    /// Pooled KS statistic over every parameter's ranks.
    pub pooled_ks: f64,
    /// Its p-value: the number the gate reads.
    pub pooled_p: f64,
    /// Parameters whose own `p` is below `0.01`.
    pub failures: u32,
    /// Smallest per-parameter `p`.
    pub min_p: f64,
    /// Median per-parameter `p`.
    pub median_p: f64,
    /// Posterior draws the engine declined to make, summed over datasets.
    pub missing: u64,
    /// Wall-clock cost of one draw, including the fit and the sampler.
    pub ms_per_draw: f64,
}

impl SbcReport {
    /// Whether the gate passes: the pooled ranks are uniform at `p > 0.01`.
    ///
    /// The per-parameter count is reported rather than folded in, for the reason
    /// in the module docs: with a thousand parameters, some of them failing at
    /// 0.01 is what a correct engine does.
    pub fn passed(&self) -> bool {
        // No ranks is not calibration: an engine that declined every dataset
        // would otherwise report p = 1 and pass a gate it never ran.
        self.pooled_p > 0.01 && self.parameters.iter().any(|parameter| parameter.ranks > 0)
    }

    /// The single machine-readable line a gate reads.
    ///
    /// Shape: `<!-- sbc-verdict engine=a1 draws=1000 posterior=100 ks_p=0.31
    /// min_p=0.08 fail=0 pass=true -->`, matching the LEE campaign's convention
    /// so `tools/sbc-verdict.sh` can parse it with `sed` and a field check.
    pub fn verdict_line(&self) -> String {
        format!(
            "<!-- sbc-verdict engine={} draws={} posterior={} ks_p={:.3e} min_p={:.3e} fail={} pass={} -->",
            self.engine,
            self.config.draws,
            self.config.posterior,
            self.pooled_p,
            self.min_p,
            self.failures,
            self.passed()
        )
    }

    /// Seconds a run at `draws` would take at the measured cost per draw.
    pub fn projected_seconds(&self) -> f64 {
        self.config.estimated_seconds(self.ms_per_draw)
    }
}

/// Runs SBC against `engine`.
///
/// The engine is driven once per draw: prior, simulate, posterior. Any draw for
/// which the engine returns no posterior samples is counted in
/// [`SbcReport::missing`] and contributes no ranks, rather than contributing a
/// rank of zero, which would read as a confident engine rather than an absent
/// one.
pub fn run<E: Engine>(engine: &mut E, cfg: SbcConfig) -> SbcReport {
    assert!(cfg.draws > 0, "SBC needs at least one draw");
    assert!(cfg.posterior > 1, "SBC needs at least two posterior draws");
    assert!(cfg.bins > 1, "the histogram needs at least two bins");
    let n = engine.parameter_count();
    assert!(n > 0, "SBC needs at least one parameter");

    let start = std::time::Instant::now();
    let mut rng = Philox::new(cfg.seed, SBC_STREAM);
    let mut ranks: Vec<Vec<f32>> = vec![Vec::with_capacity(cfg.draws as usize); n];
    let mut missing = 0u64;

    for _ in 0..cfg.draws {
        let theta = engine.prior(&mut rng);
        assert_eq!(theta.len(), n, "the prior must draw every parameter");
        let data = engine.simulate(&theta, &mut rng);
        let samples = engine.posterior(&theta, &data, &mut rng);
        if samples.len() < 2 {
            missing += 1;
            continue;
        }
        assert_eq!(
            samples[0].len(),
            n,
            "a posterior draw must cover every parameter"
        );
        let m = samples.len() as f64;
        for (index, truth) in theta.iter().enumerate() {
            let mut below = 0u32;
            let mut equal = 0u32;
            for sample in &samples {
                let value = sample[index];
                if value < *truth {
                    below += 1;
                } else if value == *truth {
                    equal += 1;
                }
            }
            let tie = rng.uniform() * f64::from(equal);
            ranks[index].push(((f64::from(below) + tie) / m) as f32);
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let ms_per_draw = 1000.0 * elapsed / f64::from(cfg.draws.max(1));

    // Pool first, then per parameter: the pooled set is the gate.
    let mut pooled: Vec<f64> = Vec::with_capacity(n * cfg.draws as usize);
    for set in &ranks {
        pooled.extend(set.iter().map(|value| f64::from(*value)));
    }
    pooled.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let (pooled_ks, pooled_p) = ks_uniform(&pooled);

    let mut parameters = Vec::with_capacity(n);
    let mut failures = 0u32;
    let mut min_p = 1.0f64;
    let mut ps = Vec::with_capacity(n);
    for (index, set) in ranks.iter().enumerate() {
        let mut sorted: Vec<f64> = set.iter().map(|value| f64::from(*value)).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let (ks, p) = ks_uniform(&sorted);
        if p < 0.01 {
            failures += 1;
        }
        min_p = min_p.min(p);
        ps.push(p);
        parameters.push(ParameterRanks {
            index,
            histogram: histogram(&sorted, cfg.bins),
            ks,
            p,
            ranks: sorted.len() as u32,
        });
    }
    ps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_p = ps.get(ps.len() / 2).copied().unwrap_or(1.0);

    SbcReport {
        engine: engine.name().to_string(),
        config: cfg,
        parameters,
        pooled_ks,
        pooled_p,
        failures,
        min_p,
        median_p,
        missing,
        ms_per_draw,
    }
}

/// Counts of `values` in `bins` equal buckets of `[0, 1]`.
fn histogram(sorted: &[f64], bins: usize) -> Vec<u32> {
    let mut counts = vec![0u32; bins];
    for value in sorted {
        let slot = ((value * bins as f64) as usize).min(bins - 1);
        counts[slot] += 1;
    }
    counts
}

/// Kolmogorov–Smirnov distance and p-value of `sorted` against `Uniform(0, 1)`.
///
/// The p-value uses the asymptotic Kolmogorov distribution with Stephens'
/// small-sample correction, which is accurate to a few percent for `n ≳ 35` and
/// conservative below it. Below 35 samples the p is not allowed to claim more
/// than the handful of points can support: it is capped at 1.0, and the report
/// prints the rank count beside it so a reader can see when the number is thin.
pub fn ks_uniform(sorted: &[f64]) -> (f64, f64) {
    let n = sorted.len();
    if n == 0 {
        return (0.0, 1.0);
    }
    let mut statistic = 0.0f64;
    for (position, value) in sorted.iter().enumerate() {
        let before = position as f64 / n as f64;
        let after = (position + 1) as f64 / n as f64;
        statistic = statistic.max((before - value).abs());
        statistic = statistic.max((value - after).abs());
    }
    if n < 35 {
        return (statistic, 1.0);
    }
    let root = (n as f64).sqrt();
    let lambda = (root + 0.12 + 0.11 / root) * statistic;
    (statistic, kolmogorov_sf(lambda))
}

/// `P(K > λ)` for the Kolmogorov distribution.
fn kolmogorov_sf(lambda: f64) -> f64 {
    if lambda <= 0.0 {
        return 1.0;
    }
    let mut sum = 0.0f64;
    for term in 1..=100 {
        let sign = if term % 2 == 1 { 1.0 } else { -1.0 };
        let value = sign * (-2.0 * (term as f64).powi(2) * lambda * lambda).exp();
        sum += value;
        if value.abs() < 1e-14 {
            break;
        }
    }
    // The Kolmogorov survival function carries a leading factor of two; leaving
    // it out halves every p and makes the gate twice as eager as it claims.
    (2.0 * sum).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A Gaussian engine that is calibrated by construction.
    ///
    /// Prior `θ ~ N(0, 1)`; the datum is `y = θ + N(0, 1)`; the posterior is
    /// `N(y/2, 1/2)`. Every marginal is exact, so a harness that reports anything
    /// but uniform ranks here is wrong, and `scale = 2` is the negative control:
    /// it is the same posterior with its width doubled, which must fail.
    struct Gaussian {
        scale: f64,
    }

    impl Gaussian {
        fn normal(rng: &mut Philox) -> f64 {
            // Box–Muller, with the log guarded away from zero.
            let u1 = rng.uniform().max(1e-300);
            let u2 = rng.uniform();
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        }
    }

    impl Engine for Gaussian {
        fn parameter_count(&self) -> usize {
            3
        }

        fn prior(&mut self, rng: &mut Philox) -> Vec<f32> {
            (0..3).map(|_| Self::normal(rng) as f32).collect()
        }

        fn simulate(&self, theta: &[f32], rng: &mut Philox) -> Vec<f32> {
            theta
                .iter()
                .map(|value| *value + Self::normal(rng) as f32)
                .collect()
        }

        fn posterior(&mut self, _theta: &[f32], data: &[f32], rng: &mut Philox) -> Vec<Vec<f32>> {
            let sigma = 0.5f64 * self.scale * std::f64::consts::SQRT_2;
            (0..64)
                .map(|_| {
                    data.iter()
                        .map(|value| 0.5 * f64::from(*value) + sigma * Self::normal(rng))
                        .map(|value| value as f32)
                        .collect()
                })
                .collect()
        }

        fn name(&self) -> &str {
            "gaussian"
        }
    }

    #[test]
    fn a_calibrated_engine_has_uniform_ranks() {
        let mut engine = Gaussian { scale: 1.0 };
        let report = run(
            &mut engine,
            SbcConfig {
                draws: 1000,
                posterior: 64,
                seed: 7,
                bins: 10,
            },
        );
        assert!(
            report.pooled_p > 0.01,
            "a closed-form posterior failed its own SBC: D = {}, p = {}",
            report.pooled_ks,
            report.pooled_p
        );
        assert!(report.passed());
        // Every draw contributes exactly one rank to every parameter.
        let total: u32 = report.parameters[0].histogram.iter().sum();
        assert_eq!(total, report.config.draws);
    }

    #[test]
    fn a_posterior_inflated_twice_fails_the_gate() {
        // The negative control. A harness that passes this detects nothing.
        let mut engine = Gaussian { scale: 2.0 };
        let report = run(
            &mut engine,
            SbcConfig {
                draws: 500,
                posterior: 64,
                seed: 11,
                bins: 10,
            },
        );
        assert!(
            report.pooled_p < 0.01,
            "a 2x-inflated posterior passed: D = {}, p = {}",
            report.pooled_ks,
            report.pooled_p
        );
        assert!(!report.passed());
        // Ranks pile at the centre, which is a shape the report can show: a
        // posterior twice as wide as it should be puts every rank within
        // Φ(±2·0.707) ≈ [0.08, 0.92] of the middle, and two thirds of them
        // within the middle two of ten bins.
        let histogram = &report.parameters[0].histogram;
        let middle: u32 = histogram[4..6].iter().sum();
        assert!(
            middle > 150,
            "inflated ranks should cluster at the centre, got {middle} of 500: {histogram:?}"
        );
        assert!(
            histogram[0] + histogram[9] < 20,
            "inflated ranks should not reach the ends: {histogram:?}"
        );
    }

    #[test]
    fn the_verdict_line_is_parseable_and_says_the_truth() {
        let mut engine = Gaussian { scale: 1.0 };
        let report = run(
            &mut engine,
            SbcConfig {
                draws: 200,
                posterior: 32,
                seed: 3,
                bins: 10,
            },
        );
        let line = report.verdict_line();
        assert!(line.starts_with("<!-- sbc-verdict engine=gaussian"));
        for field in ["draws=200", "posterior=32", "ks_p=", "min_p=", "pass="] {
            assert!(line.contains(field), "verdict line has no {field}: {line}");
        }
        assert!(line.contains(&format!("pass={}", report.passed())));
        assert!(line.trim_end().ends_with("-->"));
    }

    #[test]
    fn a_tie_with_the_truth_is_split_rather_than_counted_once() {
        // Truths land exactly on one of the eleven grid points the posterior
        // draws from, so every dataset has at least one tie. Without randomised
        // tie-breaking the rank would be a multiple of 1/11 and the histogram
        // would inherit that grid; with it, the ranks fill the interval.
        struct Grid;
        impl Engine for Grid {
            fn parameter_count(&self) -> usize {
                1
            }
            fn prior(&mut self, rng: &mut Philox) -> Vec<f32> {
                vec![(rng.below(11) as f32) / 10.0]
            }
            fn simulate(&self, _theta: &[f32], _rng: &mut Philox) -> Vec<f32> {
                vec![0.0]
            }
            fn posterior(&mut self, _t: &[f32], _d: &[f32], _r: &mut Philox) -> Vec<Vec<f32>> {
                (0..=10).map(|step| vec![step as f32 / 10.0]).collect()
            }
            fn name(&self) -> &str {
                "grid"
            }
        }
        let mut engine = Grid;
        let report = run(
            &mut engine,
            SbcConfig {
                draws: 1100,
                posterior: 11,
                seed: 5,
                bins: 4,
            },
        );
        let histogram = &report.parameters[0].histogram;
        assert_eq!(histogram.iter().sum::<u32>(), 1100);
        // A uniform rank set over four bins: every bin within a few sigma of a
        // quarter. A rank that ignored ties would leave the bins empty.
        for (bin, count) in histogram.iter().enumerate() {
            assert!(
                (230..320).contains(count),
                "bin {bin} holds {count} of 1100 ranks: {histogram:?}"
            );
        }
    }

    #[test]
    fn a_missing_posterior_is_counted_not_ranked() {
        struct Silent;
        impl Engine for Silent {
            fn parameter_count(&self) -> usize {
                1
            }
            fn prior(&mut self, rng: &mut Philox) -> Vec<f32> {
                vec![rng.uniform() as f32]
            }
            fn simulate(&self, _theta: &[f32], _rng: &mut Philox) -> Vec<f32> {
                vec![0.0]
            }
            fn posterior(&mut self, _t: &[f32], _d: &[f32], _r: &mut Philox) -> Vec<Vec<f32>> {
                Vec::new()
            }
            fn name(&self) -> &str {
                "silent"
            }
        }
        let mut engine = Silent;
        let report = run(
            &mut engine,
            SbcConfig {
                draws: 10,
                posterior: 8,
                seed: 1,
                bins: 4,
            },
        );
        assert_eq!(report.missing, 10);
        assert_eq!(report.parameters[0].ranks, 0);
        // No evidence is not evidence of calibration.
        assert_eq!(report.pooled_p, 1.0);
        assert!(!report.passed(), "an empty harness must not report a pass");
    }

    #[test]
    fn the_ks_statistic_matches_a_hand_computed_distance() {
        // Three points at 0.1, 0.5, 0.9: the largest gap between the empirical
        // step function and the diagonal is 1/3 − 0.1 = 0.2333, at the first
        // point where the step has not yet risen past 0.1.
        let (statistic, _) = ks_uniform(&[0.1, 0.5, 0.9]);
        assert!(
            (statistic - (1.0 / 3.0 - 0.1)).abs() < 1e-12,
            "D = {statistic}"
        );
        // A set that sits on two points has D = 0.25; the test must see it.
        let (regular, _) = ks_uniform(&[0.25, 0.25, 0.75, 0.75]);
        assert!(regular > 0.2 && regular < 0.3);
    }
}
