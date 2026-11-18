//! Experimental design (algorithm A9): where should the next detector go?
//!
//! Log-likelihood ratios add across independent detectors, and the utility
//! `f(x) = 1 − exp(−x/κ)` is concave and increasing. Composing them gives a
//! monotone submodular objective, so greedy placement is within `1 − 1/e` of the
//! optimal set — the guarantee the "Suggest next detector" button relies on.

use umbra_physics::{AngularBinning, Pose};

/// A place a detector could be installed, with the directions it would accept.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidatePose {
    /// Position and orientation.
    pub pose: Pose,
    /// Angular binning it would use.
    pub binning: AngularBinning,
}

/// One hypothesis in the design's prior over voids, with its prior weight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hypothesis {
    /// Prior probability of this hypothesis among all hypotheses considered.
    pub prior_weight: f32,
    /// Expected Asimov test statistic for one detector already in place, as a
    /// function of exposure. Precomputed so utility evaluation is cheap.
    pub ts_at_reference_exposure: f32,
}

/// Saturating utility `f(x) = 1 − exp(−x/κ)`.
///
/// `kappa` sets the signal scale at which extra exposure stops buying much.
pub fn saturating_utility(x: f32, kappa: f32) -> f32 {
    if kappa <= 0.0 {
        return if x > 0.0 { 1.0 } else { 0.0 };
    }
    -(-x / kappa).exp_m1()
}

/// Expected utility of a candidate configuration under the hypothesis prior.
///
/// `ts_per_hypothesis` holds the summed Asimov statistic each hypothesis would
/// accrue from the configuration being evaluated.
pub fn expected_utility(hypotheses: &[Hypothesis], ts_per_hypothesis: &[f32], kappa: f32) -> f32 {
    assert_eq!(
        hypotheses.len(),
        ts_per_hypothesis.len(),
        "one test statistic per hypothesis"
    );
    hypotheses
        .iter()
        .zip(ts_per_hypothesis)
        .map(|(h, ts)| h.prior_weight * saturating_utility(*ts, kappa))
        .sum()
}

/// Greedy selection order over candidate indices, given a utility oracle.
///
/// `marginal_utility(&[chosen], candidate)` must return the increase in utility
/// from adding `candidate` to `chosen`. Greedy ordering is what makes the
/// submodular guarantee meaningful; the returned order is what the UI animates.
pub fn greedy_order<F>(n_candidates: usize, max_picks: usize, mut marginal_utility: F) -> Vec<usize>
where
    F: FnMut(&[usize], usize) -> f32,
{
    let mut chosen = Vec::with_capacity(max_picks.min(n_candidates));
    while chosen.len() < max_picks.min(n_candidates) {
        let mut best: Option<(usize, f32)> = None;
        for candidate in 0..n_candidates {
            if chosen.contains(&candidate) {
                continue;
            }
            let gain = marginal_utility(&chosen, candidate);
            if best.map_or(true, |(_, b)| gain > b) {
                best = Some((candidate, gain));
            }
        }
        match best {
            Some((index, _)) => chosen.push(index),
            None => break,
        }
    }
    chosen
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utility_is_monotone_and_saturates() {
        assert_eq!(saturating_utility(0.0, 1.0), 0.0);
        assert!(saturating_utility(1.0, 1.0) > saturating_utility(0.5, 1.0));
        assert!(saturating_utility(50.0, 1.0) > 0.99);
    }

    #[test]
    fn expected_utility_weights_hypotheses() {
        let hypotheses = [
            Hypothesis {
                prior_weight: 0.5,
                ts_at_reference_exposure: 0.0,
            },
            Hypothesis {
                prior_weight: 0.5,
                ts_at_reference_exposure: 0.0,
            },
        ];
        assert_eq!(expected_utility(&hypotheses, &[0.0, 0.0], 1.0), 0.0);
        assert!(expected_utility(&hypotheses, &[1.0, 1.0], 1.0) > 0.5);
    }

    #[test]
    fn greedy_picks_the_highest_marginal_gain_first() {
        let gains = [0.1f32, 0.7, 0.4];
        let order = greedy_order(3, 2, |_, candidate| gains[candidate]);
        assert_eq!(order, vec![1, 2]);
    }
}
