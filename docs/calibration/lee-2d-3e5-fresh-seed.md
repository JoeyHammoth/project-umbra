# 2-D, 3 × 10⁵ toys, a seed that had never been used

The recorded 2-D campaign
([`lee-2d-3e5.md`](lee-2d-3e5.md)) decided the M3 rate criterion **PASS** on a
seed that did not diagnose the model's fault, but "not the diagnostic seed" is a
weaker claim than it sounds: a campaign that only ever runs once cannot separate a
rate from a seed. This run repeats it at the same size — `1000²`, `3 × 10⁵` toys —
on `0x5ca1ab1e`, a seed chosen before the run and used in this repository only for
the scene pilot that picked the volume's grid.

441 of 300 000 toys clear the model's 4.87σ threshold. The rate is 1.470e-3,
7% above nominal, and the Wilson interval [1.339e-3, 1.614e-3] sits inside the
±20% band [1.080e-3, 1.620e-3], so the criterion is **PASS** again. χ̄(u₀) is
8.131 ± 0.006 against the record's 8.117 ± 0.006, and the extrapolation's
agreement at 3σ is 1.01× in both.

| | Recorded ([`lee-2d-3e5.md`](lee-2d-3e5.md)) | This run |
|---|---|---|
| Seed | `0x0000000000c0ffee` | `0x000000005ca1ab1e` |
| Toys | 300 000 | 300 000 |
| χ̄(u₀) | 8.117 ± 0.006 | 8.131 ± 0.006 |
| Exceedances at `z = 4.87` | 426 | 441 |
| Rate | 1.420e-3 | 1.470e-3 |
| Wilson 95% | [1.291e-3, 1.561e-3] | [1.339e-3, 1.614e-3] |
| Extrapolation at 3σ | 1.01× | 1.01× |
| Verdict | PASS | PASS |
| Runtime | 18 716.4 s, 16 threads (62.4 ms/toy) | 17 576.5 s, 4 threads (58.6 ms/toy) |

## What the difference means

Fifteen exceedances out of a pooled 867 is 0.51 of the counting error
(√867 = 29.4 toys per sigma), so the two samples are one measurement made twice:
pooled, the rate is 867 of 600 000 = 1.445e-3 with Wilson 95%
[1.352e-3, 1.544e-3], inside the band and within 7% of nominal. The PASS is a
property of the campaign's size rather than of its seed — which is the thing the
single recorded run could not say for itself.

## What it does not decide

At `3 × 10⁵` toys the interval is about ±9% of the rate, so a sample this size
separates 1.420e-3 from 1.470e-3 no better than 0.5σ; resolving a difference that
small from zero at 3σ confidence needs about `1.1 × 10⁶` toys, which is a
different campaign and not what this one asked. Nothing here tests whether the two
seeds share a distribution beyond what 426 and 441 counts allow — it tests
whether one seed's PASS survives a second draw, and it does.

The ratios above 3σ are a looser story on this seed: 1.01×, 1.00×, 0.95×, 0.95×
and **0.86×** at 5σ against the record's 1.01×, 1.00×, 0.95×, 0.95× and 1.04×.
The 5σ row is the one with 239 toys in it, so each ratio carries a ~6.5% counting
error and the 0.18 gap between them is about two of those — a real spread in the
measurement, not a second model. It is the row where the extrapolation overshoots
most, and it is the row a 2-D campaign of this size can say least about.

## What it changes

Nothing in [`recorded-verdict.txt`](recorded-verdict.txt). The recorded line stays
the deciding sample; a second PASS corroborates it rather than replacing it, and
the gate compares a run against its dimension's line, so this one would have
reported a moved verdict — it did not move, and no issue was opened.

Reproduced by dispatching the campaign job, which is where a run this long belongs
(a local run at `16` threads needs ~5.2 hours, and two attempts were killed by the
machine before they finished):

```text
gh workflow run ci.yml -f lee_dim=2 -f lee_toys=300000 -f lee_seed=0x5ca1ab1e
```

That is CI run `46010392675` — 4 h 53 m for the whole workflow, 17 576.5 s of it
in the campaign on four cores. The report below is that run's artifact, verbatim.

# LEE calibration — 2-D, 300000 toys

| | |
|---|---|
| Scene | 1000² voxels, 1000 rays |
| Search dimension | 2 |
| Toys | 300000 |
| Seed | `0x000000005ca1ab1e` |
| Threads | 4 (one toy index range per worker) |
| Euler anchor `u₀` | 2.5σ |
| Runtime | 17576.5 s (58.6 ms/toy) |

Mean Euler characteristic at `u₀`: **8.131 ± 0.006** (299832 of 300000 toys exceeded `u₀`).

max-`Z` quantiles: 50% 3.33, 90% 3.86, 99% 4.44, 99.9% 4.94.

The extrapolation is Gross–Vitells: `⟨N(u)⟩ = p_local(u) + ⟨χ(u₀)⟩ e^{−(u²−u₀²)/2}`,
reported as `p = 1 − e^{−⟨N⟩}` because `⟨N⟩` is a region count, not a probability.

| z | local p | empirical global p | extrapolated global p | ratio | toys above |
|---|---|---|---|---|---|
| 2.0 | 2.275e-2 | 1.000e0 | 2.275e-2 | — | 300000 |
| 2.5 | 6.210e-3 | 9.994e-1 | 6.210e-3 | — | 299832 |
| 3.0 | 1.350e-3 | 8.624e-1 | 8.722e-1 | 1.01 | 258724 |
| 3.5 | 2.327e-4 | 3.327e-1 | 3.331e-1 | 1.00 | 99806 |
| 4.0 | 3.169e-5 | 6.336e-2 | 6.022e-2 | 0.95 | 19008 |
| 4.5 | 3.401e-6 | 7.748e-3 | 7.390e-3 | 0.95 | 2324 |
| 5.0 | 2.871e-7 | 7.983e-4 | 6.897e-4 | 0.86 | 239 |

Rows at or below `u₀` show the local `p`: the extrapolation does not
apply there, so their ratio is omitted rather than computed.

**Global 3σ threshold:** `z = 4.85` (empirical global p ≤ 1.350e-3, nominal 3σ = 1.350e-3). A local 3σ peak is *not* a global 3σ result: the look-elsewhere factor here is 639×.

## M3 criterion (spec §10.3): global false-positive rate at 3σ

| | |
|---|---|
| Model 3σ threshold | z = 4.87 |
| Exceedances | 441 of 300000 |
| Empirical rate | 1.470e-3 |
| 95% interval (Wilson) | [1.339e-3, 1.614e-3] |
| Nominal band (±20%) | [1.080e-3, 1.620e-3] |
| Toys to decide | 251147 (already decided by this sample) |
| Verdict | **PASS** |

## Verdict

- Extrapolation vs Monte Carlo at `z = 3`: **1.01×** (spec target: within 20%).
- `z = 5`: 239 toys above, so only the extrapolation exists there and it is unvalidated.

<!-- lee-verdict verdict=PASS toys=300000 dim=2 rate=1.470e-3 low=1.339e-3 high=1.614e-3 threshold=4.87 needs=251147 -->
