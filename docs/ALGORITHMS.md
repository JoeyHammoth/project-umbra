# Algorithm portfolio

UMBRA's point is not one solver but a *family*, compared honestly on identical
data, with each method's failure mode visible. ★ marks the distinctive items.

| ID | Method | Crate | Status |
|---|---|---|---|
| A1 | Transmission EM family (Lange–Carson, OSTR, SPS) | `umbra-recon` | `TransmissionEm` implemented (L0), ordered subsets with a monotone guard |
| A2 | Nonlinear Poisson MAP + Huber TV, L-BFGS-B, box constraints | `umbra-recon` | `PoissonMap` implemented: projected L-BFGS over Huber TV, L0 and L1 |
| A3 | Opacity-domain inversion + Chambolle–Pock PDHG with TV | `umbra-recon` | `OpacityPdhg` implemented: safeguarded Newton per ray, delta-method weights, PDHG on `[L; ∇]` with a primal–dual gap |
| A4 ★ | Reference-anchored one-sided sparse deviation (FISTA) | `umbra-recon` | `SparseDeviation` implemented: monotone FISTA, closed-form one-sided prox, void and inclusion modes |
| A5 | Plug-and-play with a learned denoiser (ONNX) | `umbra-recon` + web worker | not started (stretch) |
| A6 ★ | Profile-likelihood scan, Asimov, Euler-characteristic LEE | `umbra-stats` | `scan` + `refine_lrt` implemented; the LEE calibration is #15 |
| A7 ★ | Parametric Bayesian void inversion: analytic chords, MCMC, nested sampling | `umbra-infer` | parameterisation only |
| A8 ★ | Sequential GLR / mixture-SPRT, `t_5σ ≈ 25/r` | `umbra-stats` | `time_to_five_sigma` implemented |
| A9 ★ | Bayesian-optimal design, monotone-submodular greedy placement | `umbra-design` | utility + greedy implemented, no Asimov oracle yet |
| A10 ★ | Fisher sensitivity, local PSF, stochastic probing, Lanczos modes, coverage roses | `umbra-vis` | Fisher diagonal implemented |
| A11 | Laplace/Fisher covariance, parametric bootstrap, SBC | `umbra-stats` | Fisher σ map implemented; bootstrap and SBC next |
| A12 | Event-level muon Monte Carlo | `umbra-physics` | not started |
| A13 ★ | Amortised SBI (NPE) via ONNX, validated with SBC/TARP | offline + `umbra-infer` | not started (stretch) |

## Measured the same way

The portfolio's numbers only mean something if they are commensurable, so every
solver is timed on one dataset at one level: the committed default scene at L0,
starting from the bulk density, with the wall-clock to a fixed **data fit** —
the Poisson NLL of the estimate. The fit is computed by the harness rather than
read from the solver, because A2–A4's own objectives include a prior and a larger
penalty would look like a better reconstruction.

`cargo bench -p umbra-recon --bench portfolio` prints per-step cost and
time-to-fit for each entry, and CI runs it on every push. A 10% regression gate
needs a controlled baseline and is M6's work; until then the number is recorded
in each run's log.

A3 is measured differently on purpose: it has a dual certificate, so "converged"
means the gap fell to a fraction of the objective rather than that the iterate
stopped moving. On the shared scene the gap falls from `1.9e8` to `2.3e5` in 200
iterations — but the primal objective *rises* 20 times along the way, by up to a
factor of 1.89. PDHG is not a descent method: the primal moves while the dual
catches up, and any claim that its objective decreases monotonically would be
false. What is guaranteed is the gap, which is why it is what the solver reports
and what the test asserts.

Its exit criterion is therefore stated about the best iterate rather than every
iterate (#12, re-scoped): the best objective the solver visits is at least two
orders of magnitude below its start, the final iterate stays within a factor of
ten of that best, and the gap falls and stays non-negative. The rise count is
still printed on every test run, so the non-monotonicity is a measured number in
the log rather than a claim nobody checks.

A4 is measured against A2 on both scenes, on the void's own footprint taken from
the scene geometry rather than from "wherever the truth is low" — the slabs are
2 m thick in a 16 m grid, so the naive mask measures the vacuum outside them.
The deviation never leaves `[−ρ_ref, 0]`, and the objective never rises: the
monotone safeguard in the step is what makes that a property of the algorithm
rather than a hope about the inertia. On the shared single-view scene A4 recovers
the corridor to a mean error of **0.099 g cm⁻³ against A2's 0.143**, while fitting
*slightly better* (`−4.665487e8` against `−4.665473e8`) — the one-sided bound says
the corridor cannot be denser than the rock around it, which is information a
smoothness prior does not have.

On the four-view scene the ordering reverses: A2 reaches **0.160** in the buried
capsule against A4's **0.374**. That is the honest boundary of the method. Four
views already supply the information TV was standing in for, and the `ℓ₁` weight
then only costs contrast. A4's advantage is real but specific — one view, or a
small cavity in a known structure — and a portfolio where each member wins
somewhere is more useful than one whose favourite wins everywhere. The test
asserts the first result and records the second rather than dropping the scene
that disagrees.

**A11, error bars.** The Laplace approximation needs the curvature of the
posterior at the estimate, and for this likelihood the data part is the Fisher
information `F = Lᵀ diag((∂λ/∂X)²/λ) L` — two sparse products with the operator
every solver already holds, so `F` is never assembled. Per-voxel σ is the
diagonal of `(F + C + 1/σ_level²)⁻¹`, obtained by one conjugate-gradient solve
per voxel rather than by inverting anything: at 1024² the inverse is already
4 MB, and the grid grows. The CG diagonal agrees with a dense Gauss–Jordan
inverse on a coupled four-voxel system to `1e-4` relative, and on independent
Poisson cells the intervals cover the truth at the rate they claim — the test
measures coverage over 3200 intervals and requires it inside 64–72%, with the
predicted σ within 7% of the empirical spread of the estimates.

Two choices are physics rather than numerics. **A voxel no ray responds to is
reported unconstrained**, not given a σ: on the shared 32-ray scene **383 of
1024 voxels are never traversed**, and their variance would otherwise be
whatever the prior asserted. And the prior curvature is TV's *quadratic
majoriser*, `β/ε` on every edge, not the exact Hessian — the exact Hessian is
zero across every detected edge, which leaves the precision near-singular at
exactly the features the bars are meant to describe (measured: every column hit
its iteration cap, residual above `10²`). The majoriser keeps the prior's
statement and slightly understates σ at a sharp boundary, which is stated here
because it is the one place the bars are deliberately not the exact Laplace
answer.

TV alone cannot make the problem invertible: a constant offset changes no edge
difference, so the uniform field is an exact null direction and the effective
condition number leaves f64 (a diverging CG residual, not a slow one). A level
prior of ±0.5 g cm⁻³ on the mean density constrains that one direction and
nothing else; it is a statement about the scene, so it is a configuration value.
The result card is `umbra_recon::ResultCard`, which also reports when its bars
did not converge, so a UI can show the estimate without inventing an interval.

The scene has one detector, so its 32 rays are a single view. That is enough to
compare solvers and not enough to test ordered-subset acceleration, which needs
several distinct views to alternate between — the criterion in #10 is re-scoped
to a multi-view scene for exactly that reason. That scene now exists:
`web/src/scenes/multiview.json`, four poses, 64 rays, with a void buried at the
mid-plane that no single view can place.

Measured on it, A1 reaches 90% of a full sweep's objective decrease in **0.88
sweeps with 8 subsets, against 16 full sweeps** — 0.05×, well inside the 0.5×
the criterion asks for. `tests/multiview.rs` asserts that bound, so a change to
the partition cannot quietly spend the acceleration. The 90% target is
load-bearing: the last parts in 10⁶ are exactly where a guarded subset step
stops descending and reports convergence instead, so demanding agreement there
would measure the stopping rule rather than the acceleration.

### A6: the scan

One back projection of weighted residuals, then a correlation with the template
bank: substituting `Δ_i = Σ_j l_ij ρ_j t_c(j)` into `U_c` gives `U_c = Σ_j ρ_j t_c(j)
(Lᵀw)_j` with `w_i = D_i (y_i − λ_i^0)/λ_i^0`, so every candidate's numerator is
read out of a single `Lᵀw`. The Fisher term has no such factorisation — it carries
`Δ_i²` — so `umbra_stats::scan` accumulates it over the rays each template
actually crosses, which is exact rather than approximate: the restriction is to
the support, and `the_footprint_restriction_agrees_with_the_full_sum` holds it to
`1e-5` against the full sum.

`refine_lrt` profiles the amplitude with the exact Poisson likelihood by
bisecting the monotone derivative, and reports
`p = ½ erfc(√(TS/2))` — the one-sided `½δ₀ + ½χ²₁` mixture that the boundary
`α ≥ 0` requires. Using the two-sided `χ²₁` value would double every
significance, and returning the NLL where the log-likelihood belongs flips the
sign of `TS` while leaving `α̂` correct, which is why both are tested against the
literal definitions: 50 random candidates against a brute-force `U`/`I`, and a
golden-section profile against the bisection.

Measured on 10⁶ candidates (`cargo bench -p umbra-stats --bench scan`), with
parallel rays through every voxel — the friendliest geometry the gather can meet:

```text
                total      correlation      spec budget
2-D 1e6 cands   76.9 ms      61.5 ms          50 ms    MISSED (1.23x)
3-D 1e6 cands  131.9 ms     103.9 ms        2000 ms    met (19x under)
```

Both were read with `UMBRA_SCAN_RUNS=25`; the bench reports the fastest of `N`
because on a busy machine the median reports the scheduler. The 2-D row comes
from a window in which the untouched `setup` stage — the transpose and the back
projection, which no optimisation in this section touched — ran at its reference
15.4 ms. The 3-D row comes from a window 1.3x slower than that, so 103.9 ms is
an upper bound and its budget is met either way. The 2-D number gets no such
excuse: at reference conditions it is 61.5 ms.

§6.5 budgets the *scan correlation*, so `ScanResult::timing` splits the two
stages: `setup` is the transpose of `L` and the back projection (≈15 ms at `10⁶`
voxels), `correlate` is the band gather plus the score loop. The 2-D number is
the correlation stage, and it is still a miss.

The first version was 154 ms because the Fisher term gathers per footprint
cell — five gathers per candidate into a transposed operator that is tens of
megabytes at `10⁶` voxels. The scan now runs in **bands**: a band of rows
gathers the union of its cells' incident rays once into a compact array, and
every candidate in the band reads that instead of the operator. Ray-keyed
accumulators keep a ray that crosses several footprint cells as one `Δ_i` —
squaring its per-cell parts separately inflates `I`, and a test comparing the
banded scan against the single-candidate path exists because that bug happened.

Banding bought 154 → 62.7 ms on the budgeted stage. Everything tried since has
been worth about ten percent, and the measurements say why:

- **Band height is not the constraint.** Heights of 4, 8, 16 and 32 rows land
  within noise of each other on the 2-D scene, so shrinking the gathered arrays
  below L2 changes nothing: the loop is not capacity-bound.
- **The per-cell range lookup was worth ~7%.** Packing each cell's entry range
  into one `u64` removed one of two dependent loads, and storing entries inline
  at a fixed stride — so a cell carrying one or two rays needs no range lookup
  at all — added a few percent more. Both are in. Two unit tests assert which
  layout each scene actually takes, because a layout that is silently never
  exercised is a bug that ships.
- **Per-candidate bookkeeping is not the cost.** `z_map`, which does nothing per
  candidate but compare and store, measures *slower* than `scan`, so there is
  nothing to win there. Deferring the amplitude division past the top-k gate is
  kept because it is free, not because it moved the number.
- **Band construction is ~13% of the stage** (≈8–11 ms), so removing it entirely
  would not close the gap either.

What remains is the gather itself: at radius 1 a candidate resolves five cells,
five ray references and five chord lengths. That is the spec's own answer — a
per-detector angular index that bounds the cells a candidate can see, rather
than one bounded by the operator's size — plus a transpose built once per
*system* instead of once per scan.

### A6: nuisance parameters

`umbra_stats::profile` is spec §5.6's "Systematics" panel. Bulk density
`κ_ρ`, flux normalisation `κ_Φ`, per-detector efficiency `κ_ε` and two alignment
offsets `δ` are Gaussian-priored and **profiled**, not marginalised by hand. The
perturbation is linear in every one of them:

```text
X_i(θ) = κ_ρ (X_i⁰ − α T_i) + δ_x Gx_i + δ_y Gy_i
λ_i(θ) = κ_Φ κ_ε,d k_i e^{−X_i(θ)/Λ}
```

with `T = L(ρ ⊙ t)` and `Gx`, `Gy` the path integrals of the density gradient, so
alignment stays inside the same response as everything else instead of
re-voxelising the scene per trial offset. The test statistic is
`2(F_H0 − F_H1)` on the penalised objectives, with `F = −(ln L + ln π)`; each
fit is a damped Newton (Levenberg-damped Gauss–Newton, damping scaled by the
diagonal) on five or six parameters, and the linear solve is a small dense
elimination.

Every derivative is written out, and one test differentiates the objective
numerically and compares all six. That test is the point: a profiler that
finite-differences its objective passes every other test while being the slow
thing it replaced.

Three identifiability lessons are baked into the tests, because each one is a
way to get a confident wrong answer:

- **`Λ` must be short enough for density to matter.** At `Λ = 10³ g cm⁻²` a 2%
  density-scale change moves the rate by 0.06%, and no fit can separate that from
  the flux. The test scene runs at `Λ = 30 g cm⁻²`, where the same change is
  percent-level.
- **One detector makes flux and efficiency exactly degenerate.** They enter only
  as `κ_Φ κ_ε`, so the data cannot tell them apart; a single-detector scene
  resolves the product and leaves the split to the priors. Efficiency is only
  testable with two detectors, which is why the recovery test has two.
- **Unconstrained means *unconstrained*.** An efficiency whose detector carries
  no rays has a flat likelihood and returns to its prior mean to machine
  precision rather than drifting — the test asserts `1e-9`.

Convergence is measured as "the objective can no longer improve relative to its
own magnitude", not "the gradient is tiny". With `Σ y ≈ 10⁸`, an improvement
below `1e-9 F` is below the resolution of the `f64` holding `F`, so a strict
decrease test stops accepting steps long before an absolute gradient threshold
would ever fire.

### A6: look-elsewhere

`umbra_stats::lee` implements both layers of §5.6. `z_map` returns the `Z`
field a scan already walks; `lee_toy(problem, cfg, lee, seed, t)` draws
background-only counts from `λ⁰` and scans them; `LeeCalibration::from_toys`
merges results in any order, because toy `t` depends only on `(seed, t)`. A
worker pool can therefore split the index range and the calibration is
bit-identical — that is the reproducibility criterion, not a hope.

The extrapolation is Gross–Vitells: `p_global(u) = 1 − exp(−⟨N(u)⟩)` with
`⟨N(u)⟩ = p_local(u) + ⟨χ(u₀)⟩ e^{−(u²−u₀²)/2}`, and `χ` the Euler characteristic
of `{Z > u₀}` (`V − E + F` on a 2-D lattice, `V − E + F − C` on a 3-D one).

Both details there are measured rather than assumed, and the first one was wrong
for a while. The count carries no `(u/u₀)^{d−1}` factor: that is the
smooth-`d`-field result, and this scan's excursions are strips — each voxel is
seen by one ray — so their number falls with the local tail. The 10⁵-toy
campaign measured the factor to over-predict by 1.70× at 3.5σ, 1.56× at 4σ and
1.80× at 5σ; without it the model and the toys agree to 0.90–1.01× over 3–5σ. And `⟨N⟩` is a rate, not a probability: at `u₀` it is eight regions per
toy, where the linear form reported `p = 1` across the whole 3–4σ span, while
`1 − e^{−⟨N⟩}` reads 0.87 against the toys' 0.86 at 3σ.

What the tests actually pin: the Euler characteristic on hand-built lattices
(square, ring, disjoint blobs, cube, empty); the Poisson sampler against the
exact pmf bin by bin at `λ = 12` and `λ = 50`, and its mean and variance at
`λ = 12, 30, 50, 1000`; the `χ` scaling law on 240 synthetic `χ²₁` fields,
within a factor of two of the smooth-field prediction; the extrapolation's own
arithmetic — that `p(5σ)` is `1 − e^{−⟨N⟩}` from the upcrossing count, that the
`(u/u₀)` factor stays absent, and that `p(3σ)` is a probability near the toys'
0.86 rather than a clamp at one; and that the calibration is unchanged when its
toys are merged in reverse order.

The law was only ever checked on a plane until the campaign learned a dimension
knob, and a plane is a weak test of it: there an excursion is a strip the ray
runs along, while in a volume it is a sheet the ray cuts through, and the `χ`
closure carries one more term. `UMBRA_LEE_DIM=3` runs the same bench on `40³`
voxels (`1600` rays, one per column), and
[`docs/calibration/lee-3d-40-2e4.md`](calibration/lee-3d-40-2e4.md) records
20 000 toys on it: `χ̄(u₀) = 19.60 ± 0.05`, and the model tracks the toys at
1.02×, 1.14×, 1.15×, 1.04× and 1.02× from 3σ to 5σ. The 2 000-toy run before it
read 1.67× at 4.5σ and 0.61× at 5σ on 21 and 5 exceedances — what a tail looks
like when the sample, not the model, is the uncertainty. The volume divides the
search space differently, so its *rates* are not comparable with the plane's; the
ratio is what the law is being checked on. `euler_counts_a_tunnel_as_zero` pins
the `− C` term on a block with a column removed (a count without it reads 1 and
calls a tunnel a ball), and `the_extrapolation_holds_on_a_volume` runs the whole
chain — Poisson toys, scan, `χ`, extrapolation — on a `16³` scene.

The campaign now acts on its own result rather than waiting for a reader. Its
report ends with a machine-readable verdict line, `tools/lee-verdict.sh` reads
it, and the campaign job dispatches the `10⁵` confirmation when a `10⁴` run
reports UNDECIDED (a confirmation never dispatches another, and the weekly
scheduled run only reports) and opens an issue when a run with at least as many
toys in the same dimension changes
[`docs/calibration/recorded-verdict.txt`](calibration/recorded-verdict.txt). The
comparison is gated on toys and dimension because a smaller sample cannot
overturn a larger decision and a volume is a different search space.

`cargo bench -p umbra-stats --bench lee` runs the campaign itself. On the 200²
scene with 200 toys, the anchor is no longer a factor-of-three decision:

```text
                  empirical p(≥ z)   extrapolated      ratio
u₀ = 1.0σ,  z = 3      0.326             0.267          0.82x
u₀ = 2.0σ,  z = 3      0.326             0.310          0.95x
u₀ = 2.5σ,  z = 3      0.326             0.304          0.93x
u₀ = 2.5σ,  z = 4      0.0124            0.0109         0.88x
```

Before the normalisation was fixed this table read 2.9×, 1.7×, 1.3× and 1.4×:
the `(u/u₀)` factor grows with the distance from the anchor, so anchoring low
turned a modelling error into an anchor-sensitivity story. With the factor gone,
`1σ` and `2.5σ` agree with the toys to better than 20%, and the default stays at
`2.5σ` for statistics rather than correctness: a full campaign needs thousands
of excursions to average, and the 1000² scene has 8.1 per toy there.

The published campaign is `1000²` voxels, `10⁵` toys, seed `0xc0ffee`, 4149.9 s
on 4 threads and mean `χ̄(u₀) = 8.119 ± 0.010`, reported at
[`docs/calibration/lee-2d-1e5.md`](calibration/lee-2d-1e5.md). Against that
campaign's own tail the corrected extrapolation reads:

```text
        empirical global p   extrapolated   ratio
z = 3         0.8644             0.8718     1.01
z = 3.5       0.3339             0.3324     1.00
z = 4         0.06343            0.06006    0.95
z = 4.5       0.007845           0.007374   0.94
z = 5         0.000765           0.000687   0.90
```

The honest global `3σ` threshold — where the empirical rate reaches the nominal
`1.35 × 10⁻³` — is `z = 4.85`, not `3.0`: on this scene a local `3σ` peak is
**640×** more common than a global one, which is the whole point of the layer.
The model's crossing is `z = 4.86`, 0.01σ from the toys'.

What it is not: certified at the far tail. The `z = 5` row rests on 76
exceedances in the campaign and the model is 10% under them, so a `5σ` global
`p` here is a model informed by a measurement rather than a measurement. The
campaign runs weekly as a CI artifact for that reason.

What no unit test can pin is the spec's calibration target itself — the global
rate at the model's `3σ` threshold, with the Wilson interval its exceedances
imply. That is the campaign's verdict, not the suite's, and #15 tracks it.

## Why these, and how they relate

The classical muography method is filtered back projection, which assumes
unlimited-angle data and a linear model. Neither holds here: detectors sit in a
few places (limited angle, sparse views) and counts depend nonlinearly on opacity.
So the portfolio spans three families:

1. **Classical Poisson reconstruction** (A1) — the lineage baseline. Note the
   terminology correction: MLEM/OSEM solve *emission* models `y ~ Pois(Aρ)`. Muon
   counts are *transmission*-type, decreasing with opacity, so the matching family
   is the ML-transmission methods in A1. A2/A3 handle the full flux law.
2. **Regularised nonlinear inversion** (A2–A5) — TV, one-sided sparsity, learned
   priors. A4 encodes the physical fact that cavities only remove material, which
   is the single biggest win available for anomaly recovery.
3. **Inference and decision** (A6–A11) — is it real (A6, A8), what is it (A7),
   where to look next (A9), what can never be known (A10), how sure are we (A11).

A12 exists to validate the expected-count model against simulated events; A13 is
the amortised-inference stretch goal.

## Shared kernels

Everything reduces to a handful of operators, which is why the workspace is small:

| Kernel | Used by |
|---|---|
| `L x` (opacity) | A1–A5, A6, A9, A10, A12 |
| `Lᵀ y` (back projection) | A1–A5, A6 (scan), A10 (PSF) |
| `Lᵀ s` correlated with a template | A6 — the entire multi-scale scan is one back projection plus separable correlations |
| Asimov `TS_A` | A6, A8, A9 |
| Angular blur convolution | L2 forward, its own adjoint |

## Novelty, stated carefully

Each method exists in the literature. The contribution is (i) one browser-native,
interactive platform exposing the chain from physics to calibrated decision to
experiment design; (ii) the one-sided anomaly prior, Euler-characteristic trial
factors and Asimov-utility detector placement applied to muography; (iii) a
transparent solver and calibration comparison. Verify the literature before
publishing the word "first" anywhere.

## References

Lange & Carson (1984); Erdoğan & Fessler (1999); Nocedal & Wright (2006);
Rudin, Osher & Fatemi (1992); Chambolle & Pock (2011); Beck & Teboulle (2009);
Venkatakrishnan et al. (2013); Romano, Elad & Milanfar (2017); Cowan et al.
(2011); Gross & Vitells (2010); Goodman & Weare (2010); Skilling (2006); Wald
(1945); Chaloner & Verdinelli (1995); Nemhauser et al. (1978); Fessler & Rogers
(1996); Talts et al. (2018); Cranmer, Brehmer & Louppe (2020); Lemos et al.
(2023).
