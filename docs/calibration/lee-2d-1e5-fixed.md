# LEE calibration — 2-D, 100000 toys, corrected model

The campaign M3's rate criterion asks for, re-run on a seed that did not diagnose
the normalisation fault: `0x51dec0de`, dispatched through the campaign job itself
(`workflow_dispatch`, `lee_toys=100000`), 4 CI cores. It is the same scene and toy
draw as [`lee-2d-1e5.md`](lee-2d-1e5.md), whose model column and FAIL belong to the
pre-fix law; this is the one to read for the corrected model.

Reproduced by `env UMBRA_LEE_TOYS=100000 UMBRA_LEE_SEED=1373552862 cargo bench -p
umbra-stats --bench lee`. The report below is that run's output, verbatim.

# LEE calibration — 2-D, 100000 toys

| | |
|---|---|
| Scene | 1000² voxels, 1000 rays |
| Search dimension | 2 |
| Toys | 100000 |
| Seed | `0x0000000051dec0de` |
| Threads | 4 (one toy index range per worker) |
| Euler anchor `u₀` | 2.5σ |
| Runtime | 4598.2 s (46.0 ms/toy) |

Mean Euler characteristic at `u₀`: **8.120 ± 0.010** (99951 of 100000 toys exceeded `u₀`).

max-`Z` quantiles: 50% 3.33, 90% 3.86, 99% 4.44, 99.9% 4.92.

The extrapolation is Gross–Vitells: `⟨N(u)⟩ = p_local(u) + ⟨χ(u₀)⟩ e^{−(u²−u₀²)/2}`,
reported as `p = 1 − e^{−⟨N⟩}` because `⟨N⟩` is a region count, not a probability.

| z | local p | empirical global p | extrapolated global p | ratio | toys above |
|---|---|---|---|---|---|
| 2.0 | 2.275e-2 | 1.000e0 | 2.275e-2 | — | 100000 |
| 2.5 | 6.210e-3 | 9.995e-1 | 6.210e-3 | — | 99951 |
| 3.0 | 1.350e-3 | 8.626e-1 | 8.718e-1 | 1.01 | 86265 |
| 3.5 | 2.327e-4 | 3.332e-1 | 3.327e-1 | 1.00 | 33324 |
| 4.0 | 3.169e-5 | 6.334e-2 | 6.015e-2 | 0.95 | 6334 |
| 4.5 | 3.401e-6 | 7.735e-3 | 7.381e-3 | 0.95 | 773 |
| 5.0 | 2.871e-7 | 6.650e-4 | 6.888e-4 | 1.04 | 66 |

Rows at or below `u₀` show the local `p`: the extrapolation does not
apply there, so their ratio is omitted rather than computed.

**Global 3σ threshold:** `z = 4.85` (empirical global p ≤ 1.350e-3, nominal 3σ = 1.350e-3). A local 3σ peak is *not* a global 3σ result: the look-elsewhere factor here is 639×.

## M3 criterion (spec §10.3): global false-positive rate at 3σ

| | |
|---|---|
| Model 3σ threshold | z = 4.87 |
| Exceedances | 128 of 100000 |
| Empirical rate | 1.280e-3 |
| 95% interval (Wilson) | [1.077e-3, 1.522e-3] |
| Nominal band (±20%) | [1.080e-3, 1.620e-3] |
| Verdict | **UNDECIDED** |

UNDECIDED means the interval straddles the band: this campaign cannot say whether the rate is within ±20% of nominal, and the answer is more toys rather than a different threshold.

## Verdict

- Extrapolation vs Monte Carlo at `z = 3`: **1.01×** (spec target: within 20%).
- `z = 5`: 66 toys above, so only the extrapolation exists there and it is unvalidated.

<!-- lee-verdict verdict=UNDECIDED toys=100000 dim=2 rate=1.280e-3 low=1.077e-3 high=1.522e-3 threshold=4.87 -->
