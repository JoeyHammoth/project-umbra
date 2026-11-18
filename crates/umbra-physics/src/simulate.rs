//! Checkpointed Poisson exposure simulator (spec §4.7).
//!
//! Scrubbing the exposure slider must feel like time passing, not like re-rolling
//! dice: counts never decrease as `t` grows, and dragging back reproduces the
//! same history. Neither property holds if you simply draw `Poisson(λ t)` for each
//! `t`, because independent draws jump around.
//!
//! The construction here is the one in the spec: pre-draw Poisson increments at
//! ~64 log-spaced checkpoints, then reach an intermediate `t` by *binomial
//! thinning* of the increment that contains it. Thinning uses a fixed uniform
//! sequence per `(ray, checkpoint)` and a conditional Bernoulli sweep, which is
//! monotone in `p` by definition rather than by approximation.

use umbra_core::{stream_id, Philox};

/// Task id mixed into stream ids for the checkpoint draw.
const TASK_CHECKPOINT: u32 = 1;
/// Task id mixed into stream ids for the thinning draw.
const TASK_THIN: u32 = 2;

/// Default number of log-spaced checkpoints (spec §4.7).
pub const DEFAULT_CHECKPOINTS: usize = 64;

/// A frozen exposure history for one expected-count vector.
///
/// Built once per `(scene, exposure, seed)`; every scrub afterwards is a lookup
/// plus a thinning draw, so the slider costs no Poisson sampling.
#[derive(Debug, Clone)]
pub struct ExposureProcess {
    /// Number of ray cells.
    pub n_rays: usize,
    /// Full exposure, in seconds; the last checkpoint.
    pub t_max: f64,
    /// Ascending checkpoint times, in seconds.
    pub checkpoints: Vec<f64>,
    /// Cumulative counts per ray at each checkpoint, laid out `ray * n_cp + c`.
    cumulative: Vec<u32>,
    seed: u64,
    scene_key: u64,
}

impl ExposureProcess {
    /// Draws the checkpoint history for the given per-ray expected counts.
    ///
    /// `lambda` is `λ_i` at the full exposure (counts, not a rate), so each
    /// increment uses the mean `λ_i · (t_c − t_{c−1}) / t_max`.
    pub fn new(
        lambda: &[f32],
        exposure_s: f32,
        seed: u64,
        scene_key: u64,
        checkpoints: usize,
    ) -> Self {
        let n_cp = checkpoints.max(2);
        let t_max = f64::from(exposure_s.max(1e-3));
        // Log spacing, floored at 1e-4 of the full exposure so short times are
        // still resolved without a checkpoint at exactly zero.
        let t_floor = (t_max * 1e-4).max(1e-6);
        let mut times = Vec::with_capacity(n_cp);
        for k in 0..n_cp {
            let frac = k as f64 / (n_cp - 1) as f64;
            let t = (t_floor.ln() + frac * (t_max.ln() - t_floor.ln())).exp();
            times.push(t);
        }
        // Ensure exact monotone endpoints despite floating-point noise.
        times[0] = t_floor.min(t_max);
        times[n_cp - 1] = t_max;

        let n_rays = lambda.len();
        let mut cumulative = vec![0u32; n_rays * n_cp];
        for (ray, lambda_i) in lambda.iter().enumerate() {
            let mut total = 0u32;
            let mut previous_t = 0.0f64;
            for (k, t) in times.iter().enumerate() {
                let dt = (t - previous_t).max(0.0);
                previous_t = *t;
                let mean = (f64::from(*lambda_i) * dt / t_max).max(0.0);
                let id = stream_id(
                    scene_key,
                    TASK_CHECKPOINT,
                    (ray as u64) * n_cp as u64 + k as u64,
                );
                let mut rng = Philox::new(seed, id);
                let increment = rng.poisson(mean).min(u64::from(u32::MAX)) as u32;
                total = total.saturating_add(increment);
                cumulative[ray * n_cp + k] = total;
            }
        }

        Self {
            n_rays,
            t_max,
            checkpoints: times,
            cumulative,
            seed,
            scene_key,
        }
    }

    /// Counts at exposure `t`, in seconds.
    ///
    /// Monotone non-decreasing in `t` for fixed everything else, and identical
    /// for identical `(scene_key, seed)` regardless of the order of calls.
    pub fn counts(&self, t: f64) -> Vec<u32> {
        let n_cp = self.checkpoints.len();
        let clamped = t.clamp(0.0, self.t_max);
        if clamped <= 0.0 {
            return vec![0; self.n_rays];
        }

        // Index of the last checkpoint at or below `t`.
        let mut k = 0usize;
        while k + 1 < n_cp && self.checkpoints[k + 1] <= clamped {
            k += 1;
        }
        let at_checkpoint = (self.checkpoints[k] - clamped).abs() <= self.t_max * 1e-12;
        let next = (k + 1).min(n_cp - 1);
        let interval = (self.checkpoints[next] - self.checkpoints[k]).max(f64::MIN_POSITIVE);
        let p = if at_checkpoint || k + 1 >= n_cp {
            0.0
        } else {
            ((clamped - self.checkpoints[k]) / interval).clamp(0.0, 1.0)
        };

        let mut out = vec![0u32; self.n_rays];
        for ray in 0..self.n_rays {
            let base = self.cumulative[ray * n_cp + k];
            if p <= 0.0 {
                out[ray] = base;
                continue;
            }
            let upper = self.cumulative[ray * n_cp + next];
            let pending = upper.saturating_sub(base);
            let id = stream_id(
                self.scene_key,
                TASK_THIN,
                (ray as u64) * n_cp as u64 + k as u64,
            );
            let mut rng = Philox::new(self.seed, id);
            let added = rng.binomial(u64::from(pending), p);
            out[ray] = base.saturating_add(added.min(u64::from(u32::MAX)) as u32);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use umbra_core::Philox as TestRng;

    fn process(lambda: &[f32], exposure: f32, seed: u64, key: u64) -> ExposureProcess {
        ExposureProcess::new(lambda, exposure, seed, key, DEFAULT_CHECKPOINTS)
    }

    #[test]
    fn counts_are_monotone_in_exposure() {
        // Property test over pseudo-random scenes: any decrease is a bug the
        // exposure slider would show the user as time running backwards.
        let mut scene_rng = TestRng::new(99, 1);
        for scene in 0..100 {
            let n_rays = 1 + (scene_rng.uniform() * 24.0) as usize;
            let lambda: Vec<f32> = (0..n_rays)
                .map(|_| (scene_rng.uniform() * 500.0) as f32)
                .collect();
            let exposure = 60.0 * (1.0 + scene_rng.uniform() as f32 * 500.0);
            let p = process(&lambda, exposure, scene as u64, 7);
            let mut previous = vec![0u32; n_rays];
            for step in 0..=40 {
                let t = f64::from(exposure) * f64::from(step) / 40.0;
                let counts = p.counts(t);
                for (i, count) in counts.iter().enumerate() {
                    assert!(
                        *count >= previous[i],
                        "scene {scene} ray {i}: counts fell from {} to {} at t={t}",
                        previous[i],
                        count
                    );
                }
                previous = counts;
            }
        }
    }

    #[test]
    fn full_exposure_reproduces_the_checkpoint_totals() {
        let p = process(&[120.0, 40.0, 900.0], 3600.0, 5, 11);
        let at_full = p.counts(p.t_max);
        let last = p.checkpoints.len() - 1;
        for (ray, value) in at_full.iter().enumerate() {
            assert_eq!(*value, p.cumulative[ray * (last + 1) + last]);
        }
    }

    #[test]
    fn thinning_back_and_forth_reproduces_history() {
        let p = process(&[300.0, 150.0], 10_000.0, 3, 4);
        let forward: Vec<Vec<u32>> = (0..=20)
            .map(|step| p.counts(f64::from(step) * p.t_max / 20.0))
            .collect();
        let mut backward: Vec<Vec<u32>> = (0..=20)
            .rev()
            .map(|step| p.counts(f64::from(step) * p.t_max / 20.0))
            .collect();
        backward.reverse();
        assert_eq!(forward, backward);
    }

    #[test]
    fn results_do_not_depend_on_evaluation_order() {
        let a = process(&[250.0], 5000.0, 8, 21);
        let b = process(&[250.0], 5000.0, 8, 21);
        let _ = b.counts(a.t_max);
        let _ = b.counts(10.0);
        assert_eq!(a.counts(1234.5), b.counts(1234.5));
    }

    #[test]
    fn mean_counts_match_the_expected_scale() {
        // 4000 statistically identical rays: the total count should sit within a
        // few sigma of n · λ, which is the only check that the thinning keeps the
        // right Poisson mean at partial exposure.
        let n_rays = 4000;
        let lambda = vec![50.0f32; n_rays];
        let exposure = 20_000.0f32;
        let p = process(&lambda, exposure, 1, 2);
        let total: f64 = p.counts(p.t_max).iter().map(|c| f64::from(*c)).sum();
        // λ is counts at full exposure, so the total is simply n · λ.
        let expected = f64::from(lambda[0]) * n_rays as f64;
        let sigma = expected.sqrt();
        assert!(
            (total - expected).abs() < 4.0 * sigma,
            "total {total} vs expected {expected} (sigma {sigma})"
        );
    }

    #[test]
    fn zero_exposure_is_zero_and_negative_clamps() {
        let p = process(&[10.0, 20.0], 1000.0, 1, 1);
        assert_eq!(p.counts(0.0), vec![0, 0]);
        assert_eq!(p.counts(-50.0), vec![0, 0]);
        assert_eq!(p.counts(1e9), p.counts(p.t_max));
    }
}
