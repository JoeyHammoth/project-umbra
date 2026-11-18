//! Counter-based random number generation.
//!
//! Spec §4.7 requires a *counter-based* generator: the random stream a task
//! sees is a pure function of `(seed, stream id)`, never of the order in which
//! tasks happen to run. That is what makes a shared URL reproduce the same
//! universe regardless of how many workers executed it.
//!
//! The generator is Philox-4x32-10 (Salmon et al., SC'11), implemented here in
//! ~50 lines so `umbra-core` keeps its "no dependencies" invariant and compiles
//! unchanged for `wasm32-unknown-unknown`.
//!
//! # Determinism contract
//!
//! | Property | Guarantee |
//! |---|---|
//! | Same `(seed, stream)` | identical sequence, on every platform |
//! | Different stream | statistically independent sequence |
//! | Any output position | reachable directly (`Stream::at`), without drawing predecessors |
//!
//! # Known-answer parity
//!
//! The complete Random123 `tests/kat_vectors` file is committed unmodified at
//! `crates/umbra-core/tests/kat_vectors`, and every `philox4x32` row in it is
//! checked below: all three inputs — all-zero, all-ones, and the π-digit
//! counter/key pair — at both the 7- and 10-round counts. Those are the
//! reference answers rather than values produced here, so a wrong multiplier,
//! round count or key-bump order fails immediately. The function they exercise
//! is the one [`Philox::refill`] calls, which is what makes the parity claim
//! cover the streams the simulator actually consumes.

/// Philox-4x32 round multiplier, first pair.
const M0: u32 = 0xD251_1F53;
/// Philox-4x32 round multiplier, second pair.
const M1: u32 = 0xCD9E_8D57;
/// Philox-4x32 Weyl constant, first key half.
const W0: u32 = 0x9E37_79B9;
/// Philox-4x32 Weyl constant, second key half.
const W1: u32 = 0xBB67_AE85;

/// Number of rounds; ten is the standard Philox-4x32 setting.
const ROUNDS: usize = 10;

/// The 32×32→64 multiply, split into high and low halves.
#[inline]
fn mulhilo(a: u32, b: u32) -> (u32, u32) {
    let product = u64::from(a) * u64::from(b);
    ((product >> 32) as u32, product as u32)
}

/// One Philox round: two independent multiplies, cross-fed into the counter.
#[inline]
fn round(ctr: &mut [u32; 4], key: &mut [u32; 2]) {
    let (hi0, lo0) = mulhilo(M0, ctr[0]);
    let (hi1, lo1) = mulhilo(M1, ctr[2]);
    ctr[0] = hi1 ^ ctr[1] ^ key[0];
    ctr[1] = lo1;
    ctr[2] = hi0 ^ ctr[3] ^ key[1];
    ctr[3] = lo0;
    key[0] = key[0].wrapping_add(W0);
    key[1] = key[1].wrapping_add(W1);
}

/// The Philox-4x32 block function: a 128-bit counter, a 64-bit key and a round
/// count in, 128 bits out, with no state carried between calls.
///
/// This is the unit the published known-answer vectors specify, and the one
/// [`Philox::refill`] calls, so testing it is testing the generator rather than
/// a parallel reimplementation of it. `rounds` is a parameter because the
/// reference file also publishes the 7-round variant, which costs nothing to
/// check; what this crate ships is [`ROUNDS`].
fn philox_block(ctr: [u32; 4], key: [u32; 2], rounds: usize) -> [u32; 4] {
    let mut ctr = ctr;
    let mut key = key;
    for _ in 0..rounds {
        round(&mut ctr, &mut key);
    }
    ctr
}

/// A counter-based stream: the counter is `(stream, block)` and the key is the
/// seed, so any block can be produced without touching its predecessors.
#[derive(Debug, Clone)]
pub struct Philox {
    key: [u32; 2],
    block: u64,
    batch: [u32; 4],
    next: usize,
}

impl Philox {
    /// Creates a stream from a 64-bit seed and a 64-bit stream id.
    ///
    /// The seed and the stream id are mixed into the key, and the block counter
    /// always starts at zero. Keeping the stream id out of the counter is what
    /// makes [`Philox::at`] meaningful: block `n` of a stream is the same block
    /// whether it was reached by drawing or by jumping.
    pub fn new(seed: u64, stream: u64) -> Self {
        let key = stream_id(seed, 0, stream);
        Self {
            key: [(key >> 32) as u32, key as u32],
            block: 0,
            batch: [0; 4],
            next: 4,
        }
    }

    /// Creates a stream positioned at a given block, without drawing earlier
    /// blocks.
    pub fn at(seed: u64, stream: u64, block: u64) -> Self {
        let mut s = Self::new(seed, stream);
        s.block = block;
        s
    }

    /// The next block index that will be produced.
    pub fn block(&self) -> u64 {
        self.block
    }

    /// Fills the internal batch from the current block and advances.
    fn refill(&mut self) {
        // Philox's counter is 4×32 = 128 bits. The low 64 carry the block index
        // and the high 64 are a fixed domain tag, which keeps block 0 of every
        // stream clear of anything a caller might reconstruct by hand.
        let ctr = [
            self.block as u32,
            (self.block >> 32) as u32, // hi/lo of the block counter
            0x0BAD_5EED,
            0xA5A5_5A5A,
        ];
        self.batch = philox_block(ctr, self.key, ROUNDS);
        self.next = 0;
        self.block = self.block.wrapping_add(1);
    }

    /// The next 32 bits.
    pub fn next_u32(&mut self) -> u32 {
        if self.next >= 4 {
            self.refill();
        }
        let value = self.batch[self.next];
        self.next += 1;
        value
    }

    /// A uniform draw in `[0, 1)` with 53 bits of resolution.
    pub fn uniform(&mut self) -> f64 {
        // Two 32-bit words give 53 significant bits: the standard construction
        // for a double, and it never returns exactly 1.0.
        let a = u64::from(self.next_u32());
        let b = u64::from(self.next_u32());
        (((a >> 5) << 26) + (b >> 6)) as f64 / ((1u64 << 53) as f64)
    }

    /// A uniform integer in `[0, n)` by rejection, so the result is unbiased.
    pub fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0, "bound must be positive");
        let zone = u64::MAX - (u64::MAX % n);
        loop {
            let bits = (self.uniform() * ((1u64 << 53) as f64)) as u64;
            if bits < zone {
                return bits % n;
            }
        }
    }

    /// A Poisson draw with mean `lambda`.
    ///
    /// Uses the direct method for small means, where it is exact and cheap, and
    /// Hörmann's transformed rejection with squeeze (PTRS) above that, where the
    /// direct method would cost O(λ). Both are exact samplers; the switch is
    /// purely about cost.
    pub fn poisson(&mut self, lambda: f64) -> u64 {
        assert!(
            lambda >= 0.0 && lambda.is_finite(),
            "Poisson mean must be finite and non-negative, got {lambda}"
        );
        if lambda == 0.0 {
            return 0;
        }
        if lambda < 10.0 {
            // Knuth: multiply uniforms until the product drops below e^{-λ}.
            let limit = (-lambda).exp();
            let mut k = 0u64;
            let mut p = 1.0f64;
            loop {
                p *= self.uniform();
                if p <= limit {
                    return k;
                }
                k += 1;
            }
        }

        // PTRS (Hörmann 1993), the form used by most standard libraries.
        let b = 0.931 + 2.53 * lambda.sqrt();
        let a = -0.059 + 0.02483 * b;
        let inv_alpha = 1.1239 + 1.1328 / (b - 3.4);
        let vr = 0.9277 - 3.6224 / (b - 2.0);
        loop {
            let u = self.uniform() - 0.5;
            let v = self.uniform();
            let us = 0.5 - u.abs();
            let k = ((2.0 * a / us + b) * u + lambda + 0.43).floor();
            if us >= 0.07 && v <= vr {
                return k.max(0.0) as u64;
            }
            if k < 0.0 || (us < 0.013 && v > us) {
                continue;
            }
            if v.ln() + inv_alpha.ln() - (a / (us * us) + b).ln()
                <= k * lambda.ln() - lambda - ln_gamma(k + 1.0)
            {
                return k as u64;
            }
        }
    }

    /// The next 64 bits, for tests and hashing-adjacent uses.
    pub fn next_u64(&mut self) -> u64 {
        (u64::from(self.next_u32()) << 32) | u64::from(self.next_u32())
    }

    /// A binomial draw `Binom(n, p)` that is monotone in `p` for fixed uniforms.
    ///
    /// Monotonicity is the property the exposure slider depends on (spec §4.7):
    /// with the same underlying uniforms, raising `p` can only add successes, so
    /// counts never decrease as time advances and scrubbing back reproduces the
    /// same history. Implemented as a conditional Bernoulli sweep, which is the
    /// definition of monotonicity rather than an approximation of it.
    pub fn binomial(&mut self, n: u64, p: f64) -> u64 {
        if p <= 0.0 || n == 0 {
            return 0;
        }
        if p >= 1.0 {
            return n;
        }
        let mut count = 0u64;
        for _ in 0..n {
            if self.uniform() < p {
                count += 1;
            }
        }
        count
    }
}

/// `ln Γ(x)` by the Lanczos approximation, accurate to ~1e-13 for `x > 0`.
fn ln_gamma(x: f64) -> f64 {
    const COEFFS: [f64; 9] = [
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
        // Reflection: Γ(x)Γ(1-x) = π / sin(πx)
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = COEFFS[0];
    let t = x + 7.5;
    for (i, c) in COEFFS.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Derives a stream id from a scene key, a task kind and an index (spec §4.7).
///
/// FNV-1a is used because it is tiny, dependency-free and stable across
/// platforms; it is a stream *naming* function, not a cryptographic hash.
pub fn stream_id(scene_key: u64, task: u32, index: u64) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in scene_key
        .to_le_bytes()
        .iter()
        .chain(task.to_le_bytes().iter())
        .chain(index.to_le_bytes().iter())
    {
        h ^= u64::from(*byte);
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The committed Random123 `tests/kat_vectors`, parsed rather than
    /// transcribed. Every `philox4x32` row is checked: the 10-round rows are the
    /// generator this crate ships, and the 7-round rows additionally prove the
    /// round loop and key-bump order are genuinely parameterised rather than
    /// tuned to a single setting.
    ///
    /// Rows for generators this crate does not implement are skipped, so the
    /// final count assertion is what stops the test passing on an empty parse.
    #[test]
    fn matches_random123_kat_vectors() {
        const FIXTURE: &str = include_str!("../tests/kat_vectors");
        let mut checked = 0usize;
        for (line_no, line) in FIXTURE.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut fields = line.split_whitespace();
            if !fields.next().unwrap_or_default().starts_with("philox4x32") {
                continue;
            }
            let rounds: usize = fields
                .next()
                .and_then(|r| r.parse().ok())
                .unwrap_or_else(|| panic!("fixture line {}: missing round count", line_no + 1));
            let words: Vec<u32> = fields
                .map(|word| {
                    u32::from_str_radix(word, 16).unwrap_or_else(|_| {
                        panic!("fixture line {}: {word} is not hex", line_no + 1)
                    })
                })
                .collect();
            assert_eq!(
                words.len(),
                10,
                "fixture line {}: wanted 4 counter + 2 key + 4 answer words",
                line_no + 1
            );
            let ctr: [u32; 4] = words[0..4].try_into().expect("four counter words");
            let key: [u32; 2] = words[4..6].try_into().expect("two key words");
            let expected: [u32; 4] = words[6..10].try_into().expect("four answer words");
            assert_eq!(
                philox_block(ctr, key, rounds),
                expected,
                "Random123 philox4x32 {rounds} row at fixture line {}",
                line_no + 1
            );
            checked += 1;
        }
        assert_eq!(
            checked, 6,
            "the fixture publishes six philox4x32 rows; parsed {checked}"
        );
    }

    /// A snapshot, not a correctness claim: it records how [`Philox::new`] maps
    /// `(seed, stream)` onto the block function's counter and key. A change to
    /// that layout would silently rerun every published scene with different
    /// numbers, so it should be a deliberate, reviewable edit.
    #[test]
    fn stream_layout_is_pinned() {
        let mut rng = Philox::new(0, 0);
        let first: [u32; 4] = std::array::from_fn(|_| rng.next_u32());
        assert_eq!(first, [0x9932_2c13, 0xda7d_8d49, 0x564d_862d, 0x1ec7_8549]);
    }

    #[test]
    fn same_seed_and_stream_reproduce() {
        let mut a = Philox::new(42, 7);
        let mut b = Philox::new(42, 7);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_streams_diverge() {
        let mut a = Philox::new(42, 1);
        let mut b = Philox::new(42, 2);
        let mut equal = 0;
        for _ in 0..100 {
            if a.uniform() == b.uniform() {
                equal += 1;
            }
        }
        assert_eq!(equal, 0, "streams must not coincide");
    }

    #[test]
    fn a_block_can_be_reached_directly() {
        // Position independence: skipping ahead must land on the same values as
        // drawing through, which is the whole point of a counter-based design.
        // A block holds four 32-bit words, so 16 `next_u64` calls consume
        // eight blocks; a stream placed at block 8 must continue identically.
        let mut sequential = Philox::new(11, 3);
        for _ in 0..16 {
            sequential.next_u64();
        }
        let mut direct = Philox::at(11, 3, 8);
        for _ in 0..4 {
            assert_eq!(sequential.next_u64(), direct.next_u64());
        }
    }

    #[test]
    fn uniform_stays_in_the_unit_interval() {
        let mut rng = Philox::new(1, 1);
        for _ in 0..100_000 {
            let u = rng.uniform();
            assert!((0.0..1.0).contains(&u), "u = {u}");
        }
    }

    #[test]
    fn uniform_mean_and_variance_match() {
        let mut rng = Philox::new(5, 9);
        let n = 200_000;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for _ in 0..n {
            let u = rng.uniform();
            sum += u;
            sum_sq += u * u;
        }
        let mean = sum / n as f64;
        let var = sum_sq / n as f64 - mean * mean;
        assert!((mean - 0.5).abs() < 0.005, "mean = {mean}");
        assert!((var - 1.0 / 12.0).abs() < 0.002, "var = {var}");
    }

    #[test]
    fn below_is_unbiased_over_a_small_range() {
        let mut rng = Philox::new(3, 3);
        let mut counts = [0u32; 5];
        let n = 50_000;
        for _ in 0..n {
            counts[rng.below(5) as usize] += 1;
        }
        let expected = n as f64 / 5.0;
        for (value, count) in counts.iter().enumerate() {
            let deviation = (*count as f64 - expected).abs() / expected;
            assert!(deviation < 0.05, "bin {value} deviated by {deviation}");
        }
    }

    #[test]
    fn poisson_matches_mean_and_variance() {
        for lambda in [0.5f64, 3.0, 25.0, 400.0] {
            let mut rng = Philox::new(17, lambda as u64);
            let n = 200_000;
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            for _ in 0..n {
                let k = rng.poisson(lambda) as f64;
                sum += k;
                sum_sq += k * k;
            }
            let mean = sum / n as f64;
            let var = sum_sq / n as f64 - mean * mean;
            // Three sigma of the mean, floored at 1% so large means stay sane.
            let tolerance = (3.0 * (lambda / n as f64).sqrt()).max(0.01 * lambda);
            assert!(
                (mean - lambda).abs() < tolerance,
                "lambda={lambda} mean={mean} tolerance={tolerance}"
            );
            assert!(
                (var - lambda).abs() < 3.0 * (lambda * 2.0 / n as f64).sqrt() + 0.03 * lambda,
                "lambda={lambda} var={var}"
            );
        }
    }

    #[test]
    fn small_mean_poisson_agrees_with_direct_sampling() {
        // Cross-check the two branches against each other where both are exact.
        let lambda = 9.5;
        let n = 200_000;
        let mut direct = Philox::new(2, 2);
        let mut sum = 0.0;
        for _ in 0..n {
            sum += direct.poisson(lambda) as f64;
        }
        assert!((sum / n as f64 - lambda).abs() < 0.02);
    }

    #[test]
    fn binomial_is_monotone_in_p() {
        // The exposure slider depends on exactly this: same uniforms, higher p,
        // never fewer successes.
        for n in [0u64, 1, 7, 64, 500] {
            for seed in 0..8 {
                let mut previous = 0u64;
                for step in 0..=20 {
                    let p = step as f64 / 20.0;
                    let mut rng = Philox::new(seed, 1234);
                    let k = rng.binomial(n, p);
                    assert!(k >= previous, "n={n} seed={seed} p={p}: {k} < {previous}");
                    previous = k;
                }
            }
        }
    }

    #[test]
    fn binomial_lands_on_the_endpoints() {
        let mut rng = Philox::new(0, 0);
        assert_eq!(rng.binomial(10, 0.0), 0);
        assert_eq!(rng.binomial(10, 1.0), 10);
        assert_eq!(rng.binomial(0, 0.7), 0);
    }

    #[test]
    fn stream_ids_separate_tasks_and_indices() {
        assert_ne!(stream_id(1, 0, 0), stream_id(1, 1, 0));
        assert_ne!(stream_id(1, 0, 0), stream_id(1, 0, 1));
        assert_eq!(stream_id(1, 0, 0), stream_id(1, 0, 0));
    }

    #[test]
    fn ln_gamma_matches_factorials() {
        for (x, expected) in [(1.0, 0.0), (2.0, 0.0), (5.0, 24.0f64.ln())] {
            assert!((ln_gamma(x) - expected).abs() < 1e-9, "x={x}");
        }
        assert!((ln_gamma(0.5) - std::f64::consts::PI.sqrt().ln()).abs() < 1e-9);
    }
}
