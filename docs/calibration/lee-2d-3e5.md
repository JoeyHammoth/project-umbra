# LEE calibration — 2-D, 300000 toys

The campaign M3's rate criterion asks for at its own size. The 10⁵ run
([`lee-2d-1e5-fixed.md`](lee-2d-1e5-fixed.md)) put the rate 5% below nominal with
a Wilson floor 0.3% under the ±20% band — a sample-size miss, not a model one —
and three times the toys halve the half-width again, which is what this run buys.

## Seed

`0x0dec1de5` was passed and **was not used**. `UMBRA_LEE_SEED` parsed through
`FromStr`, which reads decimal only, so the `0x` form fell back silently to the
default `0xc0ffee`; the report itself names the seed that ran, and that is what
is recorded here. The knob is fixed in this same change — hex or decimal, and a
value that does not parse is now an error rather than a default — but the fix
does not retroactively change this sample.

Two consequences a reader should know rather than infer:

- 100 000 of these toys are the toy draws of [`lee-2d.md`](lee-2d.md) and
  [`lee-2d-1e5.md`](lee-2d-1e5.md), whose numbers belong to the pre-correction
  model. This sample *contains* that one rather than replicating it, so it is not
  an independent confirmation of those rates; the remaining 200 000 toys, and all
  of [`lee-2d-1e5-fixed.md`](lee-2d-1e5-fixed.md) (seed `0x51dec0de`), are
  disjoint from it.
- The rate below is therefore an estimate over a 300 000-toy sample that overlaps
  a previously published 100 000-toy sample by a third. The Wilson interval is
  still correct for the sample; the overlap only means the *agreement* with the
  10⁵ corrected run is not two independent confirmations.

## Runtime

18 716.4 s on 16 worker threads, on a laptop that was also building the web
bundle for part of it, so 62.4 ms/toy of wall time is a contended-machine figure
and not comparable with the 46.0 ms/toy the 4-core CI runner measured. Sixteen
threads bought only 6.4× over the 4-thread run: at `10⁶` voxels the campaign is
bound by memory rather than by cores. That is the reason the gate sizes its
follow-ups from `needs` instead of raising the toy count by habit.

## Result

426 of 300 000 toys clear the model's 4.87σ threshold. The rate is 1.420e-3, 5%
above nominal, and the Wilson interval [1.291e-3, 1.561e-3] sits inside the ±20%
band [1.080e-3, 1.620e-3], so the 2-D rate criterion is **PASS** and M3's box is
checked. The extrapolation holds to 1.01× at 3σ and the empirical 3σ crossing is
4.85σ against the model's 4.87σ, so passing the rate criterion did not require
disturbing the correction that produced the threshold.

The 3-D criterion is untouched by this run: a volume is a different search space
at a different threshold, and its line in
[`recorded-verdict.txt`](recorded-verdict.txt) was UNDECIDED when this run was
recorded — it has since been decided by the volume's own campaign
([`lee-3d-24-1e6.md`](lee-3d-24-1e6.md)), not by this one.

Reproduced by `env UMBRA_LEE_TOYS=300000 UMBRA_LEE_SEED=0xc0ffee cargo bench -p
umbra-stats --bench lee`. The report below is that run's output, verbatim.

# LEE calibration — 2-D, 300000 toys

| | |
|---|---|
| Scene | 1000² voxels, 1000 rays |
| Search dimension | 2 |
| Toys | 300000 |
| Seed | `0x0000000000c0ffee` |
| Threads | 16 (one toy index range per worker) |
| Euler anchor `u₀` | 2.5σ |
| Runtime | 18716.4 s (62.4 ms/toy) |

Mean Euler characteristic at `u₀`: **8.117 ± 0.006** (299852 of 300000 toys exceeded `u₀`).

max-`Z` quantiles: 50% 3.33, 90% 3.86, 99% 4.44, 99.9% 4.94.

The extrapolation is Gross–Vitells: `⟨N(u)⟩ = p_local(u) + ⟨χ(u₀)⟩ e^{−(u²−u₀²)/2}`,
reported as `p = 1 − e^{−⟨N⟩}` because `⟨N⟩` is a region count, not a probability.

| z | local p | empirical global p | extrapolated global p | ratio | toys above |
|---|---|---|---|---|---|
| 2.0 | 2.275e-2 | 1.000e0 | 2.275e-2 | — | 300000 |
| 2.5 | 6.210e-3 | 9.995e-1 | 6.210e-3 | — | 299852 |
| 3.0 | 1.350e-3 | 8.630e-1 | 8.717e-1 | 1.01 | 258902 |
| 3.5 | 2.327e-4 | 3.329e-1 | 3.326e-1 | 1.00 | 99881 |
| 4.0 | 3.169e-5 | 6.358e-2 | 6.012e-2 | 0.95 | 19075 |
| 4.5 | 3.401e-6 | 7.888e-3 | 7.377e-3 | 0.94 | 2366 |
| 5.0 | 2.871e-7 | 7.650e-4 | 6.885e-4 | 0.90 | 229 |

Rows at or below `u₀` show the local `p`: the extrapolation does not
apply there, so their ratio is omitted rather than computed.

**Global 3σ threshold:** `z = 4.85` (empirical global p ≤ 1.350e-3, nominal 3σ = 1.350e-3). A local 3σ peak is *not* a global 3σ result: the look-elsewhere factor here is 639×.

## M3 criterion (spec §10.3): global false-positive rate at 3σ

| | |
|---|---|
| Model 3σ threshold | z = 4.87 |
| Exceedances | 426 of 300000 |
| Empirical rate | 1.420e-3 |
| 95% interval (Wilson) | [1.291e-3, 1.561e-3] |
| Nominal band (±20%) | [1.080e-3, 1.620e-3] |
| Toys to decide | 136483 (already decided by this sample) |
| Verdict | **PASS** |

## Verdict

- Extrapolation vs Monte Carlo at `z = 3`: **1.01×** (spec target: within 20%).
- `z = 5`: 229 toys above, so only the extrapolation exists there and it is unvalidated.

<!-- lee-verdict verdict=PASS toys=300000 dim=2 rate=1.420e-3 low=1.291e-3 high=1.561e-3 threshold=4.87 needs=136483 -->
