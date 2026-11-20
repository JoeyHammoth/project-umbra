# LEE calibration — 2-D, 10000 toys

| | |
|---|---|
| Scene | 1000² voxels, 1000 rays |
| Toys | 10000 |
| Seed | `0x0000000000c0ffee` |
| Threads | 16 (one toy index range per worker) |
| Euler anchor `u₀` | 2.5σ |
| Runtime | 996.9 s (99.7 ms/toy) |

Mean Euler characteristic at `u₀`: **8.106 ± 0.031** (9994 of 10000 toys exceeded `u₀`).

max-`Z` quantiles: 50% 3.33, 90% 3.86, 99% 4.44, 99.9% 5.01.

| z | local p | empirical global p | extrapolated global p | ratio | toys above |
|---|---|---|---|---|---|
| 2.0 | 2.275e-2 | 9.999e-1 | 2.275e-2 | — | 10000 |
| 2.5 | 6.210e-3 | 9.994e-1 | 6.210e-3 | — | 9994 |
| 3.0 | 1.350e-3 | 8.628e-1 | 1.000e0 | 1.16 | 8628 |
| 3.5 | 2.327e-4 | 3.382e-1 | 5.652e-1 | 1.67 | 3382 |
| 4.0 | 3.169e-5 | 6.554e-2 | 9.905e-2 | 1.51 | 655 |
| 4.5 | 3.401e-6 | 8.549e-3 | 1.331e-2 | 1.56 | 85 |
| 5.0 | 2.871e-7 | 1.050e-3 | 1.375e-3 | 1.31 | 10 |

Rows at or below `u₀` show the local `p`: the extrapolation does not
apply there, so their ratio is omitted rather than computed.

**Global 3σ threshold:** `z = 4.95` (empirical global p ≤ 1.350e-3, nominal 3σ = 1.350e-3). A local 3σ peak is *not* a global 3σ result: the look-elsewhere factor here is 639×.

## Verdict

- Extrapolation vs Monte Carlo at `z = 3`: **1.16×** (spec target: within 20%).
- `z = 5`: 10 toys above, so only the extrapolation exists there and it is unvalidated.
