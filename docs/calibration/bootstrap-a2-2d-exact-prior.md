# Parametric bootstrap — a2 on `default-slab`

Counts are re-drawn from the fitted `λ̂` and the same solver refits each replicate. The spread of those refits is the empirical σ map; the Laplace map is the local Gaussian answer the cards already show. Where the two disagree is where the approximation is not enough.

| | |
|---|---|
| Scene | `default-slab` (Default slab with a known corridor void) |
| Engine | a2 (Poisson MAP with Huber TV) |
| Prior curvature | exact (β/ε inside the Huber band, 0 on a jump) |
| Datasets | 40 |
| Replicates each | 60 |
| Seed | `0x7` |
| Cost | 20.8 ms/replicate, 50.7 s total |
| Observable voxels | 641 of 1024 |
| Fisher solve | 0 unresolved, worst residual 9.99e-7 |

## Coverage at the stated rate

Nominal 68.27%: the fraction of voxels with `|ρ̂ − ρ_true| ≤ σ̂`. Voxels inside one dataset are correlated by the operator, so the Wilson interval is a voxel-count interval and not an independent-sample one — the honest summary is the rate, with the interval read as an indication of how far the count could sit from it.

| | |
|---|---|
| Observed | 0.071 |
| Wilson 95% | [0.068, 0.073] |
| Contains 0.6827 | no |
| Voxels with no usable σ | 514 |

## The same intervals, centred on the refit mean

Coverage against the truth measures two things at once: whether the bars are the right width, and whether the estimate sits on the truth at all. Centring the interval on the mean refit removes the second, so a rate near nominal here with a low rate above says the bars are fine and the *estimate* is biased; a low rate in both says the width itself is wrong.

| | |
|---|---|
| About the refit mean | 0.512 |
| Mean absolute bias | 0.0210 g cm⁻³ |

## Bootstrap σ against the Fisher bars

Median ratio of bootstrap σ to Fisher σ over every covered voxel: **2.09×**. A ratio below 1 means the Laplace bars are wider than the refits actually move, which is what a prior does where the data are weak; above 1 means they are too narrow.

## What a decision would need

A ±2% decision on the mean spread needs 2500 replicates per dataset (`1/0.02²`). At 20.8 ms each that is 0.9 min per dataset, and this report's 60 replicates give a ±12.9% interval on the spread.

## Verdict

Coverage 0.071 with Wilson [0.068, 0.073]; nominal 0.6827 is outside the interval.

<!-- bootstrap-verdict engine=a2 datasets=40 replicates=60 coverage=0.0709 low=0.0684 high=0.0735 contains_nominal=false -->

Reproduced by `cargo run --release -p umbra-cli -- bootstrap --scene web/src/scenes/default.json --engine a2 --prior exact --datasets 40 --replicates 60 --seed 7 --report docs/calibration/bootstrap-a2-2d-exact-prior.md`.
