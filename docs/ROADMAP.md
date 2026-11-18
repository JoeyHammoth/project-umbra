# Roadmap

Milestones are ordered so that each one leaves the repository demonstrable. The
design spec is now complete (v0.2): §6–§14 cover offline tooling and the Python
oracle, the front end, assets and scenarios, build and hosting, testing and
calibration, this roadmap, the risk register, repository conventions and the
appendices. Spec §11 is the roadmap of record and this file mirrors it; where
they disagree, fix both in the same PR.

## M0 — Foundation

**Deliverable:** a workspace that builds, gates that run, and documentation that
explains the shape of the project before any science lands.

Exit criteria:

- [x] Rust workspace with the ten crates and the documented dependency rule
- [x] `cargo fmt`, `clippy -D warnings`, `cargo test` wired into CI
- [x] Web shell with the Rust ↔ TypeScript contract transcribed
- [x] README, architecture, physics, algorithms, glossary, ADRs, wiki
- [x] First green CI run on `main`

## M1 — Physics core and forward model

**Deliverable:** a 2-D scene becomes a CSR operator, and `λ` can be produced and
sampled at any exposure — verifiable against the CT lab and analytic chords.

- [x] Flux/range table generator in `umbra-cli`, validated against published curves
- [x] 2-D DDA traversal producing exact `L_ij`, with an analytic-chord oracle test
- [x] L0 and L1 forward model with exact gradients
- [x] Checkpointed binomial Poisson simulator (monotone exposure scrubbing)
- [x] Python oracle parity harness

Exit criteria: opacity from traversal matches analytic chords to tolerance for
box, prism and ellipsoid targets; MC counts agree with `λ_i` by a χ² test.

## M2 — Reconstruction portfolio *(this milestone)*

**Deliverable:** A1–A4 reconstruct the same simulated dataset, side by side, in
the Lab.

- A1 transmission EM baseline (L0)
- A2 nonlinear Poisson MAP with L-BFGS-B and Huber TV
- A3 opacity inversion + PDHG
- A4 one-sided sparse deviation with FISTA
- Uncertainty: Fisher/Laplace error bars on every result card

Exit criteria: each solver has a convergence test, a runtime benchmark and a
reconstruction on the shared default scene; A4 recovers a known void where TV
alone smooths it away.

## M3 — Detection and calibration

**Deliverable:** the Detect tab answers "is there a void?" with a significance
that survives a known false-positive test.

- A6 score scan via one back projection + template correlation
- Exact LRT refinement of the top-K candidates
- Asimov significance and `t_5σ` forecast
- Look-elsewhere calibration: MC null plus Euler-characteristic extrapolation
- Nuisance profiling (density, flux, efficiency, alignment)
- A8 sequential testing; A11 SBC rank histograms

Exit criteria: under H0 the global false-positive rate at 3σ is within ±20% of
nominal across 10⁴ simulated datasets.

## M4 — Inference, design and visibility

**Deliverable:** characterise the void, choose the next detector, and see what the
geometry cannot see.

- A7 analytic chord model, ensemble MCMC, nested-sampling evidence
- A9 Asimov utility maps with greedy placement and the submodular guarantee
- A10 local PSF, stochastic probing, Lanczos null-space modes, coverage roses

Exit criteria: posterior intervals cover truth at their stated rate on simulated
scenes; greedy placement beats the user's manual choice on time-to-5σ in a
side-by-side.

## M5 — Web experience

**Deliverable:** the app a recruiter can understand in 30 seconds and a physicist
can respect in 15 minutes.

- `ComputeBus` with content keys, priority, cancellation, IndexedDB cache
- Worker pool, transferable arrays, SIMD/non-SIMD module selection
- Observatory: 2-D reconstruction, muon rain, exposure scrubbing
- Detect, Design, Characterise tabs; corner plot; posterior volume
- Shareable URL codec; PWA; degraded modes (WebGPU → WebGL2 → static)

Exit criteria: slider-driven updates within budget; shareable URL reproduces the
scene and the counts exactly.

## M6 — Hardening and release

- Accessibility to Lighthouse ≥ 95, keyboard-complete, reduced motion honoured
- Payload budgets enforced in CI; benchmark regression thresholds
- Calibration report published (coverage, SBC, LEE)
- GitHub Pages deployment with a reproducible build

Exit criteria: published site within every §1.5 budget, `£0/month`.

## Sequencing note

M1 must land before any solver: every later milestone consumes `L` and `λ`. M2 and
M3 can then proceed in parallel; M4 depends on M3's templates; M5 depends on a
stable façade, not on M4.
