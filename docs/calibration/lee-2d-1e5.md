# LEE calibration — 2-D, 100000 toys

| | |
|---|---|
| Scene | 1000² voxels, 1000 rays |
| Toys | 100000 |
| Seed | `0x0000000000c0ffee` |
| Threads | 4 (one toy index range per worker) |
| Euler anchor `u₀` | 2.5σ |
| Runtime | 4149.9 s (41.5 ms/toy) |

Mean Euler characteristic at `u₀`: **8.119 ± 0.010** (99946 of 100000 toys exceeded `u₀`).

max-`Z` quantiles: 50% 3.33, 90% 3.86, 99% 4.44, 99.9% 4.94.

| z | local p | empirical global p | extrapolated global p | ratio | toys above |
|---|---|---|---|---|---|
| 2.0 | 2.275e-2 | 1.000e0 | 2.275e-2 | — | 100000 |
| 2.5 | 6.210e-3 | 9.995e-1 | 6.210e-3 | — | 99946 |
| 3.0 | 1.350e-3 | 8.644e-1 | 1.000e0 | 1.16 | 86442 |
| 3.5 | 2.327e-4 | 3.339e-1 | 5.662e-1 | 1.70 | 33386 |
| 4.0 | 3.169e-5 | 6.343e-2 | 9.922e-2 | 1.56 | 6343 |
| 4.5 | 3.401e-6 | 7.845e-3 | 1.333e-2 | 1.70 | 784 |
| 5.0 | 2.871e-7 | 7.650e-4 | 1.378e-3 | 1.80 | 76 |

Rows at or below `u₀` show the local `p`: the extrapolation does not
apply there, so their ratio is omitted rather than computed.

**Global 3σ threshold:** `z = 4.85` (empirical global p ≤ 1.350e-3, nominal 3σ = 1.350e-3). A local 3σ peak is *not* a global 3σ result: the look-elsewhere factor here is 640×.

## M3 criterion (spec §10.3): global false-positive rate at 3σ

| | |
|---|---|
| Model 3σ threshold | z = 5.01 |
| Exceedances | 75 of 100000 |
| Empirical rate | 7.500e-4 |
| 95% interval (Wilson) | [5.984e-4, 9.399e-4] |
| Nominal band (±20%) | [1.080e-3, 1.620e-3] |
| Verdict | **FAIL** |

## Verdict

- Extrapolation vs Monte Carlo at `z = 3`: **1.16×** (spec target: within 20%).
- `z = 5`: 76 toys above, so only the extrapolation exists there and it is unvalidated.
