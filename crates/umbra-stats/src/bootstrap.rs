//! A11: the parametric bootstrap, as the empirical counterpart of the Laplace
//! error bars in [`crate::laplace`].
//!
//! The Laplace map answers "how much would this estimate move if the data were
//! re-drawn?" with the curvature of the likelihood at the MAP. That is a local
//! Gaussian statement, and it is only as good as the Gaussian part: at low
//! counts, at a bound, or where the prior is doing the work, it can be wrong by
//! a factor without anything in the code noticing.
//!
//! The bootstrap answers the same question without that assumption: take the
//! fitted `λ̂`, draw new counts from it, refit with the *same solver that
//! produced the estimate*, and take the spread of the refits. Both maps are
//! returned side by side in the campaign report because the places they
//! disagree are the places the Laplace bars should not be trusted.
//!
//! # Why the refit is a closure
//!
//! The solvers live in `umbra-recon`, which depends on this crate for the
//! Laplace map, so a solver cannot be named here. A closure is also the honest
//! signature: the bootstrap's claim is conditional on *this* pipeline, and
//! passing the pipeline in is what keeps it from being a property of whichever
//! solver this module happened to pick.
//!
//! # Which spread is reported
//!
//! `σ` is the standard deviation of the refits about their own mean, not about
//! the original fit. A biased pipeline (a TV prior pulling edges toward the
//! bulk density, say) would otherwise put its bias into the "uncertainty",
//! which is the one number a reader would then over-trust. The bias is reported
//! separately as `bias`, so it is visible instead of absorbed.

use umbra_core::rng::Philox;
use umbra_forward::expected;
use umbra_forward::System;
use umbra_physics::Fidelity;
use umbra_physics::FluxTable;

use crate::laplace::PoissonProblem;

/// Poisson draw stream, kept apart from the simulator's and the scan's.
const BOOTSTRAP_STREAM: u64 = 0xB0_07_57_2A_9C_00_00_01;

/// Per-voxel bootstrap result.
#[derive(Debug, Clone, PartialEq)]
pub struct BootstrapSigma {
    /// Mean of the refits, one entry per voxel, in g cm⁻³.
    pub mean: Vec<f32>,
    /// Standard deviation of the refits about that mean, one entry per voxel.
    pub sigma: Vec<f32>,
    /// Mean refit minus the original estimate: the pipeline's bias.
    pub bias: Vec<f32>,
    /// Replicates that produced a refit of the right shape.
    pub replicates: u32,
    /// Wall-clock cost of one replicate, refit included.
    pub ms_per_replicate: f64,
}

impl BootstrapSigma {
    /// One entry per voxel.
    pub fn len(&self) -> usize {
        self.sigma.len()
    }

    /// Whether the map is empty.
    pub fn is_empty(&self) -> bool {
        self.sigma.is_empty()
    }

    /// Replicates a ±`tolerance` (relative) decision on the mean spread would
    /// need, from the usual `σ_mean/σ = 1/√n` scaling.
    ///
    /// This is the number the report must state: a coverage figure from 50
    /// replicates and one from 10 000 are not the same claim, and the campaign
    /// should say which it made rather than leave the count in a table.
    pub fn replicates_to_decide(tolerance: f64) -> Option<u32> {
        if tolerance.is_nan() || tolerance <= 0.0 {
            return None;
        }
        Some((1.0 / (tolerance * tolerance)).ceil() as u32)
    }
}

/// Runs the parametric bootstrap at `rho_hat`.
///
/// `refit` is called once per replicate with the simulated counts and must
/// return a density field of the same length. A replicate whose refit has the
/// wrong length is skipped and counted in `replicates`, which is why that field
/// reports what actually went in rather than what was asked for.
///
/// # Panics
///
/// Panics if `rho_hat` does not match the problem's voxel count.
pub fn bootstrap_sigma<F>(
    problem: &PoissonProblem<'_>,
    rho_hat: &[f32],
    replicates: u32,
    seed: u64,
    mut refit: F,
) -> BootstrapSigma
where
    F: FnMut(&[f32]) -> Vec<f32>,
{
    let n = problem.system.voxel_count();
    assert_eq!(
        rho_hat.len(),
        n,
        "the estimate must have one entry per voxel"
    );
    assert!(replicates > 0, "the bootstrap needs at least one replicate");

    let lambda = expected(
        problem.system,
        rho_hat,
        problem.flux,
        problem.fidelity,
        problem.exposure_s,
        &problem.params,
    )
    .lambda;

    let mut rng = Philox::new(seed, BOOTSTRAP_STREAM);
    let mut mean = vec![0.0f64; n];
    let mut second = vec![0.0f64; n];
    let mut done = 0u32;
    let start = std::time::Instant::now();

    for _ in 0..replicates {
        let counts: Vec<f32> = lambda
            .iter()
            .map(|value| rng.poisson(f64::from(*value).max(0.0)) as f32)
            .collect();
        let estimate = refit(&counts);
        if estimate.len() != n {
            continue;
        }
        done += 1;
        for (index, value) in estimate.iter().enumerate() {
            let value = f64::from(*value);
            mean[index] += value;
            second[index] += value * value;
        }
    }

    let ms_per_replicate = 1000.0 * start.elapsed().as_secs_f64() / f64::from(replicates.max(1));
    let count = f64::from(done.max(1));
    let mut out_mean = vec![0.0f32; n];
    let mut sigma = vec![0.0f32; n];
    let mut bias = vec![0.0f32; n];
    if done > 0 {
        for index in 0..n {
            let average = mean[index] / count;
            // Sample variance about the mean; a single replicate has none.
            let variance = if done > 1 {
                ((second[index] - count * average * average) / (count - 1.0)).max(0.0)
            } else {
                0.0
            };
            out_mean[index] = average as f32;
            sigma[index] = variance.sqrt() as f32;
            bias[index] = (average - f64::from(rho_hat[index])) as f32;
        }
    }

    BootstrapSigma {
        mean: out_mean,
        sigma,
        bias,
        replicates: done,
        ms_per_replicate,
    }
}

/// A two-sided Wilson score interval for a binomial proportion.
///
/// The normal approximation is the wrong tool for coverage: a 68% rate from 200
/// replicates has a standard error of 3.3 points, and the Wald interval both
/// overshoots `[0, 1]` near the ends and understates the width at these counts.
/// Wilson is the interval the LEE campaigns already quote, so the two reports
/// remain comparable.
pub fn wilson(successes: u32, total: u32) -> (f64, f64) {
    if total == 0 {
        return (0.0, 1.0);
    }
    const Z: f64 = 1.959_963_984_540_054;
    let n = f64::from(total);
    let p = f64::from(successes) / n;
    let denominator = 1.0 + Z * Z / n;
    let centre = (p + Z * Z / (2.0 * n)) / denominator;
    let half = Z * ((p * (1.0 - p) / n + Z * Z / (4.0 * n * n)).sqrt()) / denominator;
    ((centre - half).max(0.0), (centre + half).min(1.0))
}

/// Fraction of voxels whose estimate is within one σ of the truth, and its
/// interval.
///
/// Voxels where the bootstrap produced no spread (a solver that returned the
/// same field every time) are excluded and counted separately: an interval of
/// zero width sits on the truth or not, and including those voxels would turn a
/// failed replicate into a pass.
#[derive(Debug, Clone, PartialEq)]
pub struct Coverage {
    /// Voxels inside the interval.
    pub inside: u32,
    /// Voxels tested.
    pub total: u32,
    /// Voxels with no usable σ.
    pub unusable: u32,
    /// Wilson 95% interval.
    pub interval: (f64, f64),
}

impl Coverage {
    /// Observed fraction.
    pub fn rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            f64::from(self.inside) / f64::from(self.total)
        }
    }

    /// Whether the nominal 68.27% lies inside the interval.
    pub fn contains_nominal(&self) -> bool {
        const NOMINAL: f64 = 0.682_7;
        self.interval.0 <= NOMINAL && NOMINAL <= self.interval.1
    }
}

/// Measures how often `|estimate − truth| ≤ sigma`.
pub fn coverage(estimate: &[f32], sigma: &[f32], truth: &[f32]) -> Coverage {
    assert_eq!(estimate.len(), sigma.len(), "one σ per voxel");
    assert_eq!(estimate.len(), truth.len(), "one truth per voxel");
    let mut inside = 0u32;
    let mut total = 0u32;
    let mut unusable = 0u32;
    for ((estimate, sigma), truth) in estimate.iter().zip(sigma).zip(truth) {
        if !sigma.is_finite() || *sigma <= 0.0 {
            unusable += 1;
            continue;
        }
        total += 1;
        if (estimate - truth).abs() <= *sigma {
            inside += 1;
        }
    }
    Coverage {
        inside,
        total,
        unusable,
        interval: wilson(inside, total),
    }
}

/// Largest relative disagreement between two σ maps, at voxels both cover.
///
/// `None` when no voxel has a positive entry in both: a ratio there would be a
/// statement about nothing.
pub fn worst_ratio(laplace: &[f32], bootstrap: &[f32]) -> Option<f32> {
    assert_eq!(
        laplace.len(),
        bootstrap.len(),
        "one σ per voxel in each map"
    );
    let mut worst: Option<f32> = None;
    for (a, b) in laplace.iter().zip(bootstrap) {
        if *a > 0.0 && *b > 0.0 && a.is_finite() && b.is_finite() {
            let ratio = if a > b { a / b } else { b / a };
            worst = Some(worst.map_or(ratio, |current: f32| current.max(ratio)));
        }
    }
    worst
}

/// Builds the observation for a system, for callers that only have one.
pub fn problem<'a>(
    system: &'a System,
    flux: &'a FluxTable,
    fidelity: Fidelity,
    exposure_s: f32,
) -> PoissonProblem<'a> {
    PoissonProblem {
        system,
        flux,
        fidelity,
        exposure_s,
        params: Default::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use umbra_core::{Csr, Grid2, GridAny};
    use umbra_forward::{ForwardParams, RayCell};
    use umbra_physics::generate_default;

    /// `rays` rows, each naming one voxel with a path length in cm.
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

    fn l0<'a>(system: &'a System, flux: &'a FluxTable) -> PoissonProblem<'a> {
        PoissonProblem {
            system,
            flux,
            fidelity: Fidelity::L0,
            exposure_s: 1.0,
            params: ForwardParams::default(),
        }
    }

    /// The exact MLE for one ray crossing one voxel at L0, inverted through the
    /// same forward law the data were drawn from.
    fn analytic_refit(path_cm: f32) -> impl FnMut(&[f32]) -> Vec<f32> {
        move |counts: &[f32]| {
            counts
                .iter()
                .map(|observed| {
                    let observed = f64::from(*observed).max(0.5);
                    let amplitude = 1.0e4 * ForwardParams::default().l0_flux0;
                    let opacity = 1000.0 * (f64::from(amplitude) / observed).ln();
                    (opacity / f64::from(path_cm)) as f32
                })
                .collect()
        }
    }

    #[test]
    fn the_bootstrap_spread_agrees_with_the_fisher_error_for_one_cell() {
        // One ray, one voxel, L0: the MLE is exact and the Fisher σ has a closed
        // form ρ/√(λ (L/Λ)²), so a bootstrap that disagrees is wrong rather than
        // merely different.
        let flux = generate_default();
        let path = 211.2f32;
        let system = independent_rays(&[path]);
        let truth = [2.65f32];
        let problem = l0(&system, &flux);
        let lambda = expected(
            &system,
            &truth,
            &flux,
            Fidelity::L0,
            1.0,
            &ForwardParams::default(),
        )
        .lambda[0];
        let fisher = f64::from(ForwardParams::default().l0_attenuation_gcm2)
            / (f64::from(path) * (f64::from(lambda) as f64).sqrt());

        let map = bootstrap_sigma(&problem, &truth, 400, 0x51ED_0001, analytic_refit(path));

        assert_eq!(map.replicates, 400);
        let relative = (f64::from(map.sigma[0]) - fisher).abs() / fisher;
        assert!(
            relative < 0.10,
            "bootstrap σ {} vs Fisher {} (relative {relative})",
            map.sigma[0],
            fisher
        );
        assert!(map.ms_per_replicate > 0.0);
    }

    #[test]
    fn the_interval_covers_the_truth_at_the_nominal_rate() {
        // Eight independent cells, 200 datasets: each dataset is fitted, its own
        // bootstrap σ is measured at that fit, and the question is whether the
        // *fit* is within σ of the truth. Comparing the bootstrap mean instead
        // would average the noise away and report a coverage near one for any
        // honest sampler, which is why the fit is the quantity carried through.
        let flux = generate_default();
        let path = 211.2f32;
        let truth = [2.65f32; 8];
        let system = independent_rays(&[path; 8]);
        let problem = l0(&system, &flux);
        let lambda = expected(
            &system,
            &truth,
            &flux,
            Fidelity::L0,
            1.0,
            &ForwardParams::default(),
        )
        .lambda;
        let mut rng = Philox::new(0xC0FF_EE00, 0x5EED);
        let mut inside = 0u32;
        let mut total = 0u32;
        for trial in 0..200u64 {
            let counts: Vec<f32> = lambda
                .iter()
                .map(|value| rng.poisson(f64::from(*value)) as f32)
                .collect();
            let estimate = analytic_refit(path)(&counts);
            let map = bootstrap_sigma(
                &problem,
                &estimate,
                40,
                0x5EED_0000 + trial,
                analytic_refit(path),
            );
            for voxel in 0..8 {
                total += 1;
                let sigma = f64::from(map.sigma[voxel]);
                if (estimate[voxel] - truth[voxel]).abs() <= sigma as f32 {
                    inside += 1;
                }
            }
        }
        let result = Coverage {
            inside,
            total,
            unusable: 0,
            interval: wilson(inside, total),
        };
        assert!(
            result.contains_nominal(),
            "68.27% is outside [{:.4}, {:.4}] for {}/{} ({:.3})",
            result.interval.0,
            result.interval.1,
            inside,
            total,
            result.rate()
        );
    }

    #[test]
    fn a_zero_spread_voxel_is_excluded_rather_than_counted() {
        let coverage = coverage(&[1.0, 1.0, 1.0], &[0.1, 0.0, 0.1], &[1.0, 9.0, 1.0]);
        assert_eq!(coverage.total, 2);
        assert_eq!(coverage.unusable, 1);
        assert_eq!(coverage.inside, 2);
    }

    #[test]
    fn wilson_stays_inside_the_unit_interval_at_the_edges() {
        let (low, high) = wilson(0, 10);
        assert_eq!(low, 0.0);
        assert!(high > 0.0 && high < 1.0, "high = {high}");
        let (low, high) = wilson(10, 10);
        assert!((high - 1.0).abs() < 1e-12, "high = {high}");
        assert!(low > 0.0 && low < 1.0, "low = {low}");
    }

    #[test]
    fn replicates_to_decide_scales_with_the_tolerance() {
        assert_eq!(BootstrapSigma::replicates_to_decide(0.02), Some(2500));
        assert_eq!(BootstrapSigma::replicates_to_decide(0.0), None);
    }

    #[test]
    fn the_worst_ratio_ignores_voxels_one_map_cannot_see() {
        let ratio = worst_ratio(&[1.0, 2.0, 0.0], &[1.5, 0.0, 4.0]);
        assert_eq!(ratio, Some(1.5));
        assert_eq!(worst_ratio(&[0.0], &[1.0]), None);
    }
}
