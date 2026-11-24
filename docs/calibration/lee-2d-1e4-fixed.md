# LEE calibration — 2-D, 10000 toys

| | |
|---|---|
| Scene | 1000² voxels, 1000 rays |
| Toys | 10000 |
| Seed | `0x000000001ee6e026` |
| Threads | 16 (one toy index range per worker) |
| Euler anchor `u₀` | 2.5σ |
| Runtime | 942.6 s (94.3 ms/toy) |

Mean Euler characteristic at `u₀`: **8.128 ± 0.031** (9992 of 10000 toys exceeded `u₀`).

max-`Z` quantiles: 50% 3.33, 90% 3.86, 99% 4.44, 99.9% 5.08.

The extrapolation is Gross–Vitells: `⟨N(u)⟩ = p_local(u) + ⟨χ(u₀)⟩ e^{−(u²−u₀²)/2}`,
reported as `p = 1 − e^{−⟨N⟩}` because `⟨N⟩` is a region count, not a probability.

| z | local p | empirical global p | extrapolated global p | ratio | toys above |
|---|---|---|---|---|---|
| 2.0 | 2.275e-2 | 9.999e-1 | 2.275e-2 | — | 10000 |
| 2.5 | 6.210e-3 | 9.992e-1 | 6.210e-3 | — | 9992 |
| 3.0 | 1.350e-3 | 8.662e-1 | 8.721e-1 | 1.01 | 8662 |
| 3.5 | 2.327e-4 | 3.359e-1 | 3.330e-1 | 0.99 | 3359 |
| 4.0 | 3.169e-5 | 6.474e-2 | 6.021e-2 | 0.93 | 647 |
| 4.5 | 3.401e-6 | 8.049e-3 | 7.388e-3 | 0.92 | 80 |
| 5.0 | 2.871e-7 | 1.350e-3 | 6.895e-4 | 0.51 | 13 |

Rows at or below `u₀` show the local `p`: the extrapolation does not
apply there, so their ratio is omitted rather than computed.

**Global 3σ threshold:** `z = 4.90` (empirical global p ≤ 1.350e-3, nominal 3σ = 1.350e-3). A local 3σ peak is *not* a global 3σ result: the look-elsewhere factor here is 642×.

## M3 criterion (spec §10.3): global false-positive rate at 3σ

| | |
|---|---|
| Model 3σ threshold | z = 4.87 |
| Exceedances | 14 of 10000 |
| Empirical rate | 1.400e-3 |
| 95% interval (Wilson) | [8.342e-4, 2.349e-3] |
| Nominal band (±20%) | [1.080e-3, 1.620e-3] |
| Verdict | **UNDECIDED** |

UNDECIDED means the interval straddles the band: this campaign cannot say whether the rate is within ±20% of nominal, and the answer is more toys rather than a different threshold.

## Verdict

- Extrapolation vs Monte Carlo at `z = 3`: **1.01×** (spec target: within 20%).
- `z = 5`: 13 toys above, so only the extrapolation exists there and it is unvalidated.
