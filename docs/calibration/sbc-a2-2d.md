# SBC — a2 on `default-slab`

Simulation-based calibration (Talts et al. 2018): draw the truth from the prior, simulate counts, run the engine, and rank the truth among its posterior draws. A calibrated engine gives uniform ranks.

| | |
|---|---|
| Scene | `default-slab` (Default slab with a known corridor void) |
| Grid | 32 × 32 at 0.5 m |
| Engine | a2 (Poisson MAP with Huber TV) |
| Posterior | Laplace marginals at the fit, 100 draws |
| Draws | 100 |
| Observable voxels | 641 |
| Seed | `0x7` |
| Inflation | 1× |
| Fits | 100 |
| Runtime | 168.6 s (1685.5 ms/draw) |
| Declined draws | 0 (the Laplace solve did not converge) |
| Pooled KS | D = 0.4972, p = 0.000e0 |
| Per-parameter p | median 0.000, min 0.000, 641 below 0.01 |

## Rank histograms

Ten bins over `[0, 1)`. Uniform requires the same count in each; the two end bins are where an over-confident engine shows up, and the middle pair is where an under-confident one does.

| Parameter | bins | KS D | p |
|---|---|---|---|
| 434 | [66, 0, 0, 0, 0, 0, 0, 0, 0, 34] | 0.6600 | 2.934e-39 |
| 205 | [64, 0, 0, 1, 0, 0, 0, 0, 0, 35] | 0.6400 | 6.101e-37 |
| 393 | [36, 0, 0, 0, 0, 0, 0, 0, 0, 64] | 0.6300 | 8.272e-36 |
| 264 | [63, 0, 0, 0, 0, 0, 0, 0, 0, 37] | 0.6300 | 8.272e-36 |
| 281 | [37, 0, 0, 0, 0, 0, 0, 0, 0, 63] | 0.6300 | 8.272e-36 |
| 409 | [37, 0, 0, 0, 0, 0, 0, 0, 0, 63] | 0.6300 | 8.272e-36 |
| 428 | [37, 0, 0, 0, 0, 0, 0, 0, 0, 63] | 0.6300 | 8.272e-36 |
| 17 | [38, 0, 0, 0, 0, 0, 0, 0, 0, 62] | 0.6200 | 1.076e-34 |
| 290 | [38, 0, 0, 0, 0, 0, 0, 0, 0, 62] | 0.6200 | 1.076e-34 |
| 357 | [37, 0, 0, 0, 0, 1, 0, 0, 0, 62] | 0.6200 | 1.076e-34 |

The ten lowest p-values of 641. The gate is the pooled p, not these: a thousand parameters put one below 0.01 about every ten runs even when the engine is correct.

## What a decision at 10⁴ draws would cost

At the measured 1685.5 ms/draw, 10⁴ draws take 16855 s — 4.7 h. That is a CI job, not a laptop run, which is why this report records the size it actually ran rather than the size spec §5.7 asks for.

## Verdict

Pooled KS p = 0.000e0 against the 0.01 gate: **FAIL**.

<!-- sbc-verdict engine=a2 draws=100 posterior=100 ks_p=0.000e0 min_p=2.934e-39 fail=641 pass=false -->

Reproduced by `cargo run --release -p umbra-cli -- sbc --scene web/src/scenes/default.json --engine a2 --draws 100 --posterior 100 --seed 7 --inflate 1 --report docs/calibration/sbc-a2-2d.md`.
