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
2-D 1e6 cands   78.1 ms      62.7 ms          50 ms    MISSED (1.25x)
3-D 1e6 cands  118.9 ms      96.8 ms        2000 ms    met
```

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

Banding bought 154 → 62.7 ms on the budgeted stage. The rest is per-candidate
arithmetic, not memory: six axis comparisons, an index, and two loads per
footprint cell, five cells per candidate at radius 1. Two levers remain, both
cheap relative to the angular index: an interior fast path that drops the
bounds checks for candidates further than `reach` from the grid edge, and a
transpose built once per *system* rather than once per scan. The larger one is
the spec's own answer — a per-detector angular index bounding the cells per
candidate rather than the operator's size.

### A6: look-elsewhere

`umbra_stats::lee` implements both layers of §5.6. `z_map` returns the `Z`
field a scan already walks; `lee_toy(problem, cfg, lee, seed, t)` draws
background-only counts from `λ⁰` and scans them; `LeeCalibration::from_toys`
merges results in any order, because toy `t` depends only on `(seed, t)`. A
worker pool can therefore split the index range and the calibration is
bit-identical — that is the reproducibility criterion, not a hope.

The extrapolation is Gross–Vitells:
`p_global(u) = p_local(u) + ⟨χ(u₀)⟩ (u/u₀)^{d−1} e^{−(u²−u₀²)/2}`, with `χ` the
Euler characteristic of `{Z > u₀}` (`V − E + F` on a 2-D lattice, `V − E + F − C`
on a 3-D one).

What the tests actually pin: the Euler characteristic on hand-built lattices
(square, ring, disjoint blobs, cube, empty); the Poisson sampler against the
exact pmf bin by bin at `λ = 12` and `λ = 50`, and its mean and variance at
`λ = 12, 30, 50, 1000`; the `χ` scaling law on 240 synthetic `χ²₁` fields,
within a factor of two of the prediction; and that the calibration is unchanged
when its toys are merged in reverse order.

What they do **not** pin is the spec's calibration target: the global `p` at
`3σ` within ±20% of nominal over `10⁴` toys. That is a campaign, not a unit
test, and it is the reason #15 is open.

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
