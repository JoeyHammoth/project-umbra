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
| A6 ★ | Profile-likelihood scan, Asimov, Euler-characteristic LEE | `umbra-stats` | `asimov_significance` implemented |
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
