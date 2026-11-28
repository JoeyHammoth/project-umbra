//! A8: sequential testing, with Wald boundaries and a stopping trace.
//!
//! The Timeline asks a question the scan does not: *given the exposures we have
//! so far, how much longer until this is decided?* A sequential test answers it
//! by accumulating evidence as it arrives and stopping the moment the running
//! log-likelihood ratio leaves a band, rather than waiting for a fixed exposure
//! and then quoting a p-value.
//!
//! # The boundaries are Wald's, on the log scale
//!
//! For a false-positive rate `α` and a miss rate `β`, Wald (1945) writes the
//! band as
//!
//! ```text
//! A = ln((1 − β)/α),      B = ln(β/(1 − α)),
//! ```
//!
//! and continues the test while `B < LLR < A`. `A` is the log of the
//! likelihood-ratio evidence a crossing upward represents, so `α = 0.05`
//! buys `A = 2.94` and `α = 2.9e-7` — five sigma, one-sided — buys `A = 15.1`.
//! Both numbers are returned on every [`SrtPoint`] so a plot can draw the band
//! without recomputing it, and so a reader cannot quote a crossing against a
//! boundary the test was not using.
//!
//! # Why the alternatives are a mixture
//!
//! The scan tests a bank of templates at unknown locations, so there is no
//! single alternative to put in a simple SPRT. Mixture-SPRT (the "generalised
//! sequential probability ratio test" of the change-point literature) replaces
//! the single likelihood ratio with an equally weighted mixture over the bank:
//!
//! ```text
//! z_k = ln( (1/K) Σ_j exp(z_kj) ),     z_kj = Σ_i [ y_i ln(m_ji/b_i) − (m_ji − b_i) ]
//! ```
//!
//! where `b` is the background expectation and `m_j` the expectation under
//! template `j`. Catching a void at an unknown voxel is then a question the test
//! can answer at all; a simple SPRT would need the location supplied.
//!
//! The mixture is still a likelihood ratio against the same `H0`, so `e^{LLR}`
//! is a martingale with expectation 1 under `H0`, and Ville's inequality gives
//! `P(max LLR ≥ A) ≤ e^{−A} = α/(1 − β)`. That bound is what makes the crossing
//! rate testable rather than asserted, and the test below measures it over 2000
//! seeded streams instead of trusting the algebra.
//!
//! # Open interval, and streams that never stop
//!
//! The test continues while `B < LLR < A`, so a value *on* a boundary stops:
//! floating-point equality is a crossing, not a continue. A stream that reaches
//! the end of its exposures inside the band is **not** a crossing, and
//! [`SrtTrace::stopped_at`] is `None` rather than the last index. A forecast
//! that reported "stopped at the end" would turn an undecided experiment into a
//! decided one, which is the single most expensive mistake this module can make.

use umbra_core::rng::Philox;

/// Which boundary the running ratio left, if any.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Crossing {
    /// The ratio reached `A`: the alternative is favoured at the stated rate.
    Upper,
    /// The ratio reached `B`: the null is accepted at the stated miss rate.
    Lower,
    /// Still inside the band. With a trace, this is a stream that never stopped.
    None,
}

impl Crossing {
    /// Whether the test has stopped.
    pub fn stopped(self) -> bool {
        !matches!(self, Crossing::None)
    }
}

/// Wald's stopping band, on the log-likelihood-ratio scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WaldBoundaries {
    /// `A = ln((1 − β)/α)`.
    pub upper: f64,
    /// `B = ln(β/(1 − α))`.
    pub lower: f64,
}

impl WaldBoundaries {
    /// The band for a false-positive rate `alpha` and a miss rate `beta`.
    ///
    /// `None` when either rate is outside `(0, 1)`, when they sum to at least
    /// one, or when a rate is non-finite: those are not tests, and returning a
    /// band for them would put a plausible-looking `A` on a question nobody
    /// asked.
    ///
    /// # Panics
    ///
    /// Never; invalid input yields `None`.
    pub fn new(alpha: f64, beta: f64) -> Option<Self> {
        let valid = |rate: f64| rate.is_finite() && rate > 0.0 && rate < 1.0;
        if !valid(alpha) || !valid(beta) || alpha + beta >= 1.0 {
            return None;
        }
        Some(Self {
            upper: ((1.0 - beta) / alpha).ln(),
            lower: (beta / (1.0 - alpha)).ln(),
        })
    }

    /// The five-sigma band, one-sided: `α = 2.87e-7`, `β = 0.05`.
    ///
    /// One-sided because a void can only *raise* the count: a deficit is not the
    /// alternative the detector is built for.
    pub fn five_sigma() -> Self {
        Self::new(2.866_5e-7, 0.05).expect("5σ is a valid band")
    }

    /// Which boundary `llr` has reached, if any.
    pub fn crossing(&self, llr: f64) -> Crossing {
        if !llr.is_finite() {
            // A ratio that has become infinite is not a crossing this test can
            // report a stopping time for; treat it as still running so the
            // caller sees the non-finite value in the trace instead.
            return Crossing::None;
        }
        if llr >= self.upper {
            Crossing::Upper
        } else if llr <= self.lower {
            Crossing::Lower
        } else {
            Crossing::None
        }
    }

    /// Whether the test continues at `llr`.
    pub fn contains(&self, llr: f64) -> bool {
        !self.crossing(llr).stopped()
    }
}

/// One exposure increment: what arrived, and what each hypothesis expects.
///
/// All four lengths must match, and they are *increments* — the counts between
/// two exposures, not the cumulative ones. Cumulative counts would make the
/// increments depend on their position in the stream, and the LLR would no
/// longer be a sum of independent terms.
#[derive(Debug, Clone, Copy)]
pub struct Step<'a> {
    /// Exposure this increment covers, in seconds. Reported, never used.
    pub exposure_s: f32,
    /// Counts in the increment, one per ray cell.
    pub counts: &'a [f32],
    /// `b_i`: expected counts under `H0`.
    pub background: &'a [f32],
    /// `m_ji`: expected counts under each alternative template.
    pub alternatives: &'a [&'a [f32]],
}

/// One point of the trace: the running ratio and the band it was tested against.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SrtPoint {
    /// Exposure at this step, in seconds.
    pub exposure_s: f32,
    /// Cumulative log-likelihood ratio after this step.
    pub llr: f32,
    /// `A`, repeated so a plot or a `Float32Array` can draw it directly.
    pub upper: f32,
    /// `B`.
    pub lower: f32,
}

/// The whole stream: every point up to and including the crossing.
#[derive(Debug, Clone, PartialEq)]
pub struct SrtTrace {
    /// One entry per step consumed. The last entry is the crossing when the test
    /// stopped, so a plot never draws a point past the decision.
    pub points: Vec<SrtPoint>,
    /// Which boundary was reached. `Crossing::None` means the stream ran out of
    /// exposures inside the band: undecided, not a null result.
    pub crossing: Crossing,
    /// Index of the crossing point in [`SrtTrace::points`], if any.
    pub stopped_at: Option<usize>,
}

impl SrtTrace {
    /// The last ratio in the trace, or `None` for an empty stream.
    pub fn final_llr(&self) -> Option<f32> {
        self.points.last().map(|point| point.llr)
    }

    /// Whether the stream reached a boundary within its exposures.
    pub fn stopped(&self) -> bool {
        self.crossing.stopped()
    }

    /// Steps consumed: the stopping time when the test stopped.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether no step was consumed.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

/// The Poisson log-likelihood ratio of `alternative` against `background` at
/// `counts`.
///
/// Cells whose expectation is zero or non-finite under either hypothesis are
/// skipped: `ln(m/b)` is not defined there, and a cell the geometry never
/// illuminates carries no evidence either way rather than infinite evidence.
///
/// # Panics
///
/// Panics if the three slices differ in length.
pub fn log_likelihood_ratio(counts: &[f32], background: &[f32], alternative: &[f32]) -> f64 {
    assert_eq!(
        counts.len(),
        background.len(),
        "counts and background must share a ray-cell basis"
    );
    assert_eq!(
        counts.len(),
        alternative.len(),
        "counts and alternatives must share a ray-cell basis"
    );
    let mut total = 0.0f64;
    for ((&y, &b), &m) in counts.iter().zip(background).zip(alternative) {
        let (y, b, m) = (f64::from(y), f64::from(b), f64::from(m));
        if !y.is_finite() || !b.is_finite() || !m.is_finite() || b <= 0.0 || m <= 0.0 {
            continue;
        }
        // The `−(m − b)` term is the Poisson normalisation. Leaving it out
        // makes the ratio grow with the number of cells per step, so a stream
        // would "decide" faster purely by carrying a wider detector.
        total += y * (m / b).ln() - (m - b);
    }
    total
}

/// The increment of the mixture log-likelihood ratio for one step.
///
/// Equal weights over the alternatives: the bank is a set of candidates with no
/// prior preference, and unequal weights would be a claim about where the void
/// is. Empty alternatives are a programming error rather than a null result —
/// a mixture of nothing has no ratio.
///
/// # Panics
///
/// Panics if `alternatives` is empty or the slices disagree in length.
pub fn increment(step: &Step<'_>) -> f64 {
    assert!(
        !step.alternatives.is_empty(),
        "a mixture needs at least one alternative"
    );
    let ratios: Vec<f64> = step
        .alternatives
        .iter()
        .map(|alternative| log_likelihood_ratio(step.counts, step.background, alternative))
        .collect();
    // `ln Σ exp(z_j / K)`, in log-sum-exp form: the individual ratios reach
    // hundreds of nats on a real scene, and `exp` of that overflows to `inf`,
    // which would make every later comparison `NaN`.
    log_sum_exp(&ratios) - (ratios.len() as f64).ln()
}

/// `ln Σ exp(values)`, stable for any finite range.
fn log_sum_exp(values: &[f64]) -> f64 {
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return max;
    }
    max + values
        .iter()
        .map(|value| (value - max).exp())
        .sum::<f64>()
        .ln()
}

/// Runs the sequential test over `steps`, stopping at the first crossing.
///
/// Consumes steps lazily, so a caller that generates exposures can stop paying
/// for them once the test has decided. The trace contains every consumed step,
/// including the crossing.
pub fn sprt<'a, I>(steps: I, boundaries: WaldBoundaries) -> SrtTrace
where
    I: IntoIterator<Item = Step<'a>>,
{
    let mut points = Vec::new();
    let mut llr = 0.0f64;
    let mut crossing = Crossing::None;
    let mut stopped_at = None;
    for step in steps {
        llr += increment(&step);
        let reached = boundaries.crossing(llr);
        points.push(SrtPoint {
            exposure_s: step.exposure_s,
            llr: llr as f32,
            upper: boundaries.upper as f32,
            lower: boundaries.lower as f32,
        });
        if reached.stopped() {
            crossing = reached;
            stopped_at = Some(points.len() - 1);
            break;
        }
    }
    SrtTrace {
        points,
        crossing,
        stopped_at,
    }
}

/// Kullback–Leibler divergence `D(alternative ‖ background)`, in nats per step.
///
/// This is the expected LLR increment under the alternative, which is what
/// Wald's expected stopping time is made of: `E[N] ≈ (A(1 − β′) + Bβ′) / D`.
/// Exposed because the forecast ribbon has to be computed from the same numbers
/// the test accumulates, and a report that recomputed it elsewhere could
/// disagree with the trace it sits under.
///
/// # Panics
///
/// Panics if the two slices differ in length.
pub fn poisson_kl(alternative: &[f32], background: &[f32]) -> f64 {
    assert_eq!(
        alternative.len(),
        background.len(),
        "the two expectations must share a ray-cell basis"
    );
    let mut total = 0.0f64;
    for (&m, &b) in alternative.iter().zip(background) {
        let (m, b) = (f64::from(m), f64::from(b));
        if !m.is_finite() || !b.is_finite() || m <= 0.0 || b <= 0.0 {
            continue;
        }
        total += m * (m / b).ln() - (m - b);
    }
    total
}

/// Wald's expected number of steps to a decision, from the per-step drift.
///
/// `E[N] ≈ (A·p + B·(1 − p)) / drift`, with `p` the probability of crossing the
/// upper boundary. `drift` is the expected increment under whichever hypothesis
/// the forecast is for, from [`poisson_kl`]. Returns `None` when the drift is
/// not positive: a stream that drifts the wrong way is never expected to decide,
/// and a large number from dividing by a small one would read as a forecast.
pub fn expected_steps(boundaries: &WaldBoundaries, drift: f64, miss_rate: f64) -> Option<f64> {
    if !drift.is_finite() || drift <= 0.0 {
        return None;
    }
    let p = 1.0 - miss_rate;
    let numerator = boundaries.upper * p + boundaries.lower * (1.0 - p);
    let steps = numerator / drift;
    if steps.is_finite() && steps > 0.0 {
        Some(steps)
    } else {
        None
    }
}

/// Median expected step at which the five-sigma boundary is crossed under a
/// known per-step drift, as the Timeline's `t_5σ` ribbon quotes it.
///
/// The closed form is the inverse-quadratic scaling of
/// [`crate::time_to_five_sigma`] specialised to the sequential band: a signal
/// that earns `drift` nats per step crosses `A` after `A/drift` steps, so a
/// signal ten times weaker needs a hundred times the exposure.
pub fn steps_to_five_sigma(drift: f64) -> Option<f64> {
    let boundaries = WaldBoundaries::five_sigma();
    if !drift.is_finite() || drift <= 0.0 {
        return None;
    }
    Some(boundaries.upper / drift)
}

/// A seeded stream source, so the tests and the campaign draw from the same
/// generator and a campaign can be re-run from its seed.
///
/// The stream id is fixed: a caller's seed selects the stream, not the kind of
/// draw, and two callers with the same seed must get the same counts.
pub fn poisson_stream(seed: u64) -> Philox {
    Philox::new(seed, 0x53_50_52_54_00_00_00_01)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bootstrap::wilson;

    /// One ray cell's background expectation and the signal that raises it.
    const BACKGROUND: [f32; 4] = [40.0, 12.0, 40.0, 25.0];

    /// A void raises the transmitted count: less material, more photons.
    fn raised(scale: f32) -> Vec<f32> {
        BACKGROUND
            .iter()
            .enumerate()
            .map(|(cell, background)| {
                if cell == 2 {
                    background * (1.0 + scale)
                } else {
                    *background
                }
            })
            .collect()
    }

    /// `steps` increments of Poisson counts drawn at `lambda`.
    fn counts_at(lambda: &[f32], steps: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = poisson_stream(seed);
        (0..steps)
            .map(|_| {
                lambda
                    .iter()
                    .map(|value| rng.poisson(f64::from(*value).max(0.0)) as f32)
                    .collect()
            })
            .collect()
    }

    /// The background expectation, repeated once per increment.
    fn backgrounds(steps: usize) -> Vec<Vec<f32>> {
        vec![BACKGROUND.to_vec(); steps]
    }

    /// A one-template bank per step, so a `Step` can borrow it as a slice.
    fn bank(alternative: &[f32], steps: usize) -> Vec<Vec<&[f32]>> {
        (0..steps).map(|_| vec![alternative]).collect()
    }

    #[test]
    fn the_boundaries_are_walds_closed_form() {
        // A = ln((1 − 0.05)/0.05) = ln 19 = 2.9444, and with α = β the band is
        // symmetric.
        let bounds = WaldBoundaries::new(0.05, 0.05).expect("a valid band");
        assert!((bounds.upper - 19.0f64.ln()).abs() < 1e-12);
        assert!((bounds.lower + 19.0f64.ln()).abs() < 1e-12);
        // The five-sigma band is not symmetric: the miss rate is still 5%.
        let five = WaldBoundaries::five_sigma();
        assert!(five.upper > 15.0 && five.upper < 16.0, "A = {}", five.upper);
        // A rate outside (0, 1), a rate of zero and a pair that sums past one
        // are not tests, and must not produce a band that looks like one.
        assert!(WaldBoundaries::new(0.0, 0.05).is_none());
        assert!(WaldBoundaries::new(1.0, 0.05).is_none());
        assert!(WaldBoundaries::new(0.6, 0.5).is_none());
        assert!(WaldBoundaries::new(f64::NAN, 0.05).is_none());
    }

    #[test]
    fn the_test_continues_inside_the_band_and_stops_on_the_boundary() {
        // The documented convention is the open interval: a value *on* a
        // boundary stops. Equality is a crossing.
        let bounds = WaldBoundaries::new(0.05, 0.05).expect("a valid band");
        assert_eq!(bounds.crossing(bounds.upper), Crossing::Upper);
        assert_eq!(bounds.crossing(bounds.lower), Crossing::Lower);
        assert_eq!(bounds.crossing(0.0), Crossing::None);
        assert!(bounds.contains(0.0));
        assert!(!bounds.contains(bounds.upper));
        // A non-finite ratio is not a decision.
        assert_eq!(bounds.crossing(f64::INFINITY), Crossing::None);
    }

    #[test]
    fn the_trace_is_the_running_sum_of_its_increments() {
        // The streaming form and the batch form must agree exactly: a trace that
        // drifts from its increments is a plot that disagrees with the decision.
        let alternative = raised(0.04);
        let counts = counts_at(&alternative, 12, 0xA1);
        let background = backgrounds(12);
        let alternatives = bank(&alternative, 12);
        let step_at = |index: usize| Step {
            exposure_s: index as f32,
            counts: &counts[index],
            background: &background[index],
            alternatives: &alternatives[index],
        };
        let increments: Vec<f64> = (0..12).map(|index| increment(&step_at(index))).collect();
        let bounds = WaldBoundaries::new(0.05, 0.05).expect("a valid band");
        let trace = sprt((0..12).map(step_at), bounds);
        assert_eq!(trace.len(), 12, "no crossing was expected at this drift");
        let mut running = 0.0f64;
        for (point, increment) in trace.points.iter().zip(&increments) {
            running += increment;
            assert!(
                (f64::from(point.llr) - running).abs() < 1e-4,
                "trace {point:?} disagrees with the batch sum {running}"
            );
            assert_eq!(point.upper, bounds.upper as f32);
            assert_eq!(point.lower, bounds.lower as f32);
        }
        assert!(!trace.stopped());
        assert_eq!(trace.stopped_at, None);
    }

    #[test]
    fn an_asimov_signal_stops_near_walds_forecast() {
        // A signal of the shape the alternative predicts, at the exposure the
        // drift implies. The forecast is `E[N] ≈ A/D` with `D` the Poisson KL
        // between the two expectations, and the acceptance is 10% relative.
        let alternative = raised(0.04);
        let drift = poisson_kl(&alternative, &BACKGROUND);
        assert!(drift > 0.0, "a raised count must carry positive drift");
        let bounds = WaldBoundaries::new(0.05, 0.05).expect("a valid band");
        let forecast = expected_steps(&bounds, drift, 0.05).expect("a finite forecast");

        // 400 streams is enough to pin the mean to a few percent: the stopping
        // time has a standard error of about `E[N]/√400`.
        let mut total = 0.0f64;
        let mut stopped = 0u32;
        for seed in 0..400u64 {
            let counts = counts_at(&alternative, 2_000, 0x5EED_0000 + seed);
            let background = backgrounds(2_000);
            let alternatives = bank(&alternative, 2_000);
            let trace = sprt(
                (0..2_000).map(|index| Step {
                    exposure_s: index as f32,
                    counts: &counts[index],
                    background: &background[index],
                    alternatives: &alternatives[index],
                }),
                bounds,
            );
            if let Some(index) = trace.stopped_at {
                total += index as f64 + 1.0;
                stopped += 1;
            }
        }
        assert_eq!(stopped, 400, "a stream at the alternative must decide");
        let mean = total / f64::from(stopped);
        let relative = (mean - forecast).abs() / forecast;
        assert!(
            relative < 0.10,
            "mean stopping time {mean:.1} against Wald's {forecast:.1} ({:.1}%)",
            relative * 100.0
        );
    }

    #[test]
    fn the_false_positive_rate_matches_the_boundary() {
        // Under H0 the mixture ratio is a martingale, so Ville's inequality
        // bounds the crossing rate by `α/(1 − β)` = 0.0526 for α = β = 0.05.
        // Measure it rather than trust it: 2000 seeded background streams, and
        // the binomial interval must contain the nominal 0.05.
        //
        // The measured rate is 4.15% (83 of 2000), so the interval's upper edge
        // is 5.12% and the nominal 0.05 is inside it by 0.12 of a point. That
        // narrow margin is the answer, not a sample-size artefact: the same
        // fixture at 2000 steps per stream gives the same 83 crossings, so the
        // horizon is not truncating the tail. Wald's own result is that the
        // realised error rates sit *at or below* the nominal ones, and this is
        // the conservative direction he guaranteed rather than an overshoot to
        // tune away.
        let bounds = WaldBoundaries::new(0.05, 0.05).expect("a valid band");
        let mut crossings = 0u32;
        const STREAMS: u32 = 2_000;
        const STEPS: usize = 600;
        let background = backgrounds(STEPS);
        let alternative = raised(0.04);
        let alternatives = bank(&alternative, STEPS);
        for seed in 0..u64::from(STREAMS) {
            let counts = counts_at(&BACKGROUND, STEPS, 0xB0_0000 + seed);
            let trace = sprt(
                (0..STEPS).map(|index| Step {
                    exposure_s: index as f32,
                    counts: &counts[index],
                    background: &background[index],
                    alternatives: &alternatives[index],
                }),
                bounds,
            );
            if trace.crossing == Crossing::Upper {
                crossings += 1;
            }
        }
        let (low, high) = wilson(crossings, STREAMS);
        let rate = f64::from(crossings) / f64::from(STREAMS);
        assert!(
            low <= 0.05 && 0.05 <= high,
            "crossing rate {rate:.4} ({crossings}/{STREAMS}), Wilson [{low:.4}, {high:.4}] \
             does not contain the nominal 0.05"
        );
        // The bound is one-sided: the rate cannot sit far above `α/(1 − β)`.
        assert!(
            rate < 0.075,
            "crossing rate {rate:.4} is far above Ville's 0.0526 bound"
        );
    }

    #[test]
    fn a_stream_that_never_stops_says_so() {
        // Counts exactly at the background: every increment is zero, so the
        // ratio never leaves the band. The trace must report that as undecided
        // rather than stopping at the last exposure.
        let counts = vec![BACKGROUND.to_vec(); 5];
        let backgrounds = vec![BACKGROUND.to_vec(); 5];
        let alternative = raised(0.04);
        let alternatives = bank(&alternative, 5);
        let trace = sprt(
            (0..5).map(|index| Step {
                exposure_s: index as f32,
                counts: &counts[index],
                background: &backgrounds[index],
                alternatives: &alternatives[index],
            }),
            WaldBoundaries::new(0.05, 0.05).expect("a valid band"),
        );
        assert_eq!(trace.len(), 5, "every step must be in the trace");
        assert_eq!(trace.crossing, Crossing::None);
        assert_eq!(trace.stopped_at, None);
        assert!(!trace.stopped());
        // One step of drift per step, and finite: a count that sits exactly on
        // the background shifts the ratio by `−D(b ‖ m)` every time, which is
        // what stops a quiet stream from ever accumulating evidence.
        let drift = -poisson_kl(&BACKGROUND, &raised(0.04));
        for (index, point) in trace.points.iter().enumerate() {
            assert!(point.llr.is_finite());
            let expected = drift * (index as f64 + 1.0);
            assert!(
                (f64::from(point.llr) - expected).abs() < 1e-4,
                "step {index}: {} against {expected}",
                point.llr
            );
        }
    }

    #[test]
    fn the_mixture_does_not_prefer_one_template_by_construction() {
        // Two alternatives, one of which is the truth. A mixture of both must
        // score the data higher than the same bank with the true template
        // dropped would, and both must be finite: the mixture is an average, so
        // adding a plausible hypothesis cannot lower the evidence.
        let truth = raised(0.04);
        let decoy: Vec<f32> = BACKGROUND.iter().map(|b| b * 0.97).collect();
        let counts = counts_at(&truth, 40, 0xD00D);
        let background = backgrounds(40);
        let both: Vec<Vec<&[f32]>> = (0..40)
            .map(|_| vec![truth.as_slice(), decoy.as_slice()])
            .collect();
        let alone: Vec<Vec<&[f32]>> = (0..40).map(|_| vec![decoy.as_slice()]).collect();
        let mut with = 0.0f64;
        let mut without = 0.0f64;
        for index in 0..40 {
            let step = Step {
                exposure_s: index as f32,
                counts: &counts[index],
                background: &background[index],
                alternatives: &both[index],
            };
            with += increment(&step);
            without += increment(&Step {
                alternatives: &alone[index],
                ..step
            });
        }
        assert!(
            with > without,
            "the mixture {with} lost to a decoy {without}"
        );
        assert!(with.is_finite() && without.is_finite());
    }

    #[test]
    fn zero_expected_counts_carry_no_evidence() {
        // A cell the geometry never illuminates has `b = m = 0`. It must be
        // skipped, not divided by zero.
        let counts = [3.0f32, 0.0];
        let background = [2.0f32, 0.0];
        let alternative = [4.0f32, 0.0];
        let ratio = log_likelihood_ratio(&counts, &background, &alternative);
        assert!(ratio.is_finite());
        // The single live cell contributes `y ln(m/b) − (m − b)`.
        let expected = 3.0 * (4.0f64 / 2.0).ln() - (4.0 - 2.0);
        assert!((ratio - expected).abs() < 1e-12);
    }

    #[test]
    fn the_five_sigma_forecast_scales_inverse_quadratically() {
        // `steps_to_five_sigma` is the sequential twin of `time_to_five_sigma`:
        // a signal that carries a quarter of the drift needs four times the
        // steps, because the drift of a Poisson signal is quadratic in its
        // amplitude at fixed counts.
        let fast = steps_to_five_sigma(0.04).expect("positive drift");
        let slow = steps_to_five_sigma(0.01).expect("positive drift");
        assert!((slow / fast - 4.0).abs() < 1e-12);
        assert!(steps_to_five_sigma(0.0).is_none());
        // And the forecast agrees with Wald's band rather than a separate 25:
        // A(5σ) = 15.1, so 0.04 nats/step is 379 steps.
        assert!((fast - WaldBoundaries::five_sigma().upper / 0.04).abs() < 1e-9);
    }
}
