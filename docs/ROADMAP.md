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

## M2 — Reconstruction portfolio

**Deliverable:** A1–A4 reconstruct the same simulated dataset, side by side, in
the Lab.

- [x] A1 transmission EM baseline (L0)
- [x] Ordered-subset acceleration on the four-view scene: 0.88 sweeps against
      16
- [x] A2 nonlinear Poisson MAP with L-BFGS-B and Huber TV
- [x] A3 opacity inversion + PDHG
- [x] A4 one-sided sparse deviation with FISTA
- [x] Uncertainty: Fisher/Laplace error bars on every result card

Exit criteria: each solver has a convergence test, a runtime benchmark and a
reconstruction on the shared default scene; A4 recovers a known void where TV
alone smooths it away.

## M3 — Detection and calibration

**Deliverable:** the Detect tab answers "is there a void?" with a significance
that survives a known false-positive test.

- [x] A6 score scan via one back projection + template correlation
- [x] Exact LRT refinement of the top-K candidates
- [x] Scan at 10^6 candidates, measured in CI since the bench became a gate: 2-D **30.4 ms** correlation / 37.8 ms total against the 50 ms budget, 3-D 43.1 ms against 2 s, at one scale; 75.1 ms and 141.3 ms at two scales. An earlier run recorded 61.5 ms for the 2-D row on other hardware — the two are not comparable and only a controlled baseline machine (M6) can reconcile them, but the row is inside its limit on the runner that enforces it. The app's stage also stopped paying for two walks: the peak and the field come from one `scan_with_map` (2-D peak-plus-field 79.0 ms as two sweeps, 41.3 ms as one, in CI). The gather remains the thing to attack next — band height, entry packing, compact stride and deferred bookkeeping were all measured at ~10% together — so the next lever is a per-detector angular index, which needs detector poses in `System`; it carries ray directions but no poses today
- [x] Asimov significance and `t_5σ` forecast (`asimov_significance`, `time_to_five_sigma`, tested)
- [x] LEE estimator: seeded Poisson toys, max-`Z` null, Euler extrapolation (`umbra_stats::lee`)
- [ ] LEE calibration campaign: ran `10^4` toys on `10^6` voxels (report: `docs/calibration/lee-2d.md`). Extrapolation 1.16x at 3 sigma; global-3 sigma threshold is z = 4.95, where the MC rate is 22% below nominal on 10 exceedances (+-32%). Ten exceedances set the precision, and +-32% cannot establish a +-20% criterion, so this box stays open and what it needs is toys rather than code: ~10^5 of them for ~100 exceedances. The 10^4 run took 997 s inside a 45-minute job, so that is ~2.8 h. The job now allows 240 minutes and takes a `lee_toys` dispatch input, and the 10^5 run has been dispatched — so what this box waits on is that report, not more code. The run's verdict lands under "M3 criterion" in its artifact: `PASS`, `FAIL` or `UNDECIDED`, with the rate and its 95% interval against the ±20% band. Recovery, the other half of the claim, is now tested (`a_planted_void_is_found_where_it_was_placed`): a radius-4.5 full void at a known voxel of a two-view scene is found at exactly that voxel (0.00 offset, Z 15.6, fitted depth 1.00), and after a 300-toy correction the null's 99th percentile (4.62) stays below it, so the look-elsewhere penalty does not erase a real signal. A first version of that test used one parallel view, and its peak was flat along the void's whole row with Z identical at every x: a single view cannot localise along its own rays, so the fixture needed angular diversity before "where" was a question with an answer
- [x] Nuisance profiling (density, flux, efficiency, alignment): damped Newton on the penalised likelihood under `H0` and `H1`, analytic derivatives checked against central differences, six parameters with one detector; profiling costs significance and the drop is reported (`umbra_stats::profile`)
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

- [x] `ComputeBus` core: content keys, priority queue, cancellation by worker termination, cache interface with an in-memory store, transferable results, pool sized `min(core−1, 4)` — 12 tests under `node --test`
- [x] `BUILD_SYSTEM`/`EXPECTED`/`SIMULATE` WASM entry points: `umbra_wasm::Core` holds the systems and per-second rates, the façade returns `Float32Array`/`Uint32Array`, and `web/src/compute/handler.ts` forwards those buffers in a transfer list
- [x] Browser wiring: `workers.ts` spawns module workers running `worker.ts`, which loads the module and refuses one whose `api_version` disagrees; the landing page runs `BUILD_SYSTEM` → `EXPECTED` → `SIMULATE` → `RECON` through the bus
- [x] `RECON` wired end to end: `SIMULATE` keeps its counts under the task's content key, `umbra_wasm::Core::recon` recovers the operator, flux, fidelity and exposure from that key, and runs A1 or A2 to a bounded iteration budget; the density comes back as a transferred `Float32Array`, and A1 refuses non-L0 data instead of fitting the wrong model
- [x] `CacheStore` over IndexedDB (`web/src/compute/cache.ts`), with `MemoryCache` as the fallback when storage is unavailable; `web/scripts/check-browser.mjs` drives headless Chrome over CDP and asserts both the typed arrays and the stored record — and the CI web job runs that check after the build, so a worker that cannot load, a transfer list that drops the buffers or a store that never writes fails the build instead of being caught by eye; the check pins the exact counts the default scene must produce, so a silently wrong result fails as loudly as a missing one. The store keeps a budget of entries and estimated bytes, drops the oldest writes, and clears and retries once when the browser's quota refuses a write
- [x] `SCAN` wired end to end: `umbra_wasm::Core::scan` runs the A6 multi-scale score scan over stored counts — `∂λ/∂X` comes from the forward model's analytic gradient, stored per second beside the rate — and the `Z` field and amplitude map come back as transferred `Float32Array`s. `H0` is the authored scene, so the default demo's own data correctly shows no detection, and the browser check pins that null rather than a discovery. The boundary reports `-1` for its own timings in the browser: `wasm32-unknown-unknown` has no `Instant::now`, and the scan calling it aborted the worker until it learned to run untimed
- [x] The reconstruction, drawn: `web/src/DensityView.tsx` paints ρ one pixel per voxel at the scene's own grid width, windowed to the 2nd–98th percentile of its positive values with both ends printed in the legend — zero-to-peak drew a near-uniform rock field as a flat slab of the top colour, and the extremes would let one hot voxel flatten it again, and the exposure slider re-runs `SIMULATE` → `RECON` → `SCAN` on a new content key — the operator and λ keep theirs, so a drag costs one simulation and one fit, and returning to 3600 s comes back from cache. `check-browser` reads the canvas back with `getImageData` and pins 1024 painted voxels, so a field that is transferred correctly but never reaches a pixel fails the build; the scan's `Z` field is drawn beside it on a fixed 0–5σ ramp — fixed rather than normalised to its own peak, so a null stays dark — and  the check requires that canvas to differ from the density one, so a `Z` buffer that never arrived cannot pass as a picture. The window is adjustable from the page — floor and ceiling sliders over the field's own percentiles, with the number of values outside printed beside the legend — so the clipping the default states can be inspected rather than taken on trust, and `check-browser` moves both sliders and requires the canvas to change
- [x] `LEE_CALIB` wired into the page: after every scan the core draws 200 background-only toys from the same λ⁰, scans them with the same template bank, and reports where the observed peak sits in that maximum-`Z` null — so the page states the peak's empirical global `p` and how many toys matched it, instead of leaving a local `Z` to imply a search it never corrected for. The curve (global `p` every 0.05σ to 6σ), the largest toy peak and the extrapolated global-3σ threshold cross the boundary as one transferred `Float32Array` plus scalars; the threshold is `null` unless the Euler anchor fired, because extrapolating from zero excursion sets would return the local `p` and call it global. `check-browser` pins that the calibration is about the same peak the scan reported and that the index now holds six results. 200 toys is enough to say "this peak is ordinary", not enough to pin a rate — the 10⁵-toys campaign that decides M3 stays a bench and a CI job (§10.3)
- Worker pool, transferable arrays, SIMD/non-SIMD module selection
- Observatory: muon rain, detector placement and the view controls beyond a plain density image
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
