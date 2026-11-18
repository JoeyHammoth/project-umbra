# Algorithm portfolio

UMBRA's point is not one solver but a *family*, compared honestly on identical
data, with each method's failure mode visible. ★ marks the distinctive items.

| ID | Method | Crate | Status |
|---|---|---|---|
| A1 | Transmission EM family (Lange–Carson, OSTR, SPS) | `umbra-recon` | `TransmissionEm` implemented (L0), ordered subsets with a monotone guard |
| A2 | Nonlinear Poisson MAP + Huber TV, L-BFGS-B, box constraints | `umbra-recon` | `PoissonMap` implemented: projected L-BFGS over Huber TV, L0 and L1 |
| A3 | Opacity-domain inversion + Chambolle–Pock PDHG with TV | `umbra-recon` | not started |
| A4 ★ | Reference-anchored one-sided sparse deviation (FISTA) | `umbra-recon` | not started |
| A5 | Plug-and-play with a learned denoiser (ONNX) | `umbra-recon` + web worker | not started (stretch) |
| A6 ★ | Profile-likelihood scan, Asimov, Euler-characteristic LEE | `umbra-stats` | `asimov_significance` implemented |
| A7 ★ | Parametric Bayesian void inversion: analytic chords, MCMC, nested sampling | `umbra-infer` | parameterisation only |
| A8 ★ | Sequential GLR / mixture-SPRT, `t_5σ ≈ 25/r` | `umbra-stats` | `time_to_five_sigma` implemented |
| A9 ★ | Bayesian-optimal design, monotone-submodular greedy placement | `umbra-design` | utility + greedy implemented, no Asimov oracle yet |
| A10 ★ | Fisher sensitivity, local PSF, stochastic probing, Lanczos modes, coverage roses | `umbra-vis` | Fisher diagonal implemented |
| A11 | Laplace/Fisher covariance, parametric bootstrap, SBC | `umbra-stats` | not started |
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

The scene has one detector, so its 32 rays are a single view. That is enough to
compare solvers and not enough to test ordered-subset acceleration, which needs
several distinct views to alternate between — the criterion in #10 is re-scoped
to a multi-view scene for exactly that reason. That scene now exists:
`web/src/scenes/multiview.json`, four poses, 64 rays, with a void buried at the
mid-plane that no single view can place.

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
