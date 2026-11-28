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
- [x] LEE calibration campaign: two runs, both recorded. The 10⁴-toys run (`docs/calibration/lee-2d.md`) left the criterion undecided at ±32%; the 10⁵-toys run (`docs/calibration/lee-2d-1e5.md`, 10⁶ voxels, 4149.9 s, 41.5 ms/toy, mean χ̄(u₀) = 8.119 ± 0.010) returns **FAIL**. At the model's 3σ threshold (z = 5.01) 75 of 100 000 toys exceeded it: rate 7.500e-4 against nominal 1.350e-3, Wilson 95% [5.984e-4, 9.399e-4], which sits entirely below the ±20% band [1.080e-3, 1.620e-3]. Seventy-five exceedances pin the rate to ±23% of itself, so this is decided rather than undecided — and the miss is the model's, not the sample's: the extrapolated global p overshoots the empirical curve everywhere above u₀ (1.16× at 3σ, 1.70× at 4.5σ, 1.80× at 5σ), so the threshold lands 0.16σ above the empirical crossing (z = 4.85) and the test buys 44% fewer false positives than 3σ promises. Closing the box needs a correction to the Gross–Vitells normalisation — the χ̄ scaling or the anchor — and a 10⁴-toys re-run is enough to confirm a fix; more toys at the present model would only measure the same overshoot more precisely. **The correction is in** (§A6 of `docs/ALGORITHMS.md` records the case): the excursion count no longer carries the smooth-field `(u/u₀)^{d−1}` factor — this scan's excursions are strips, since each voxel is seen by one ray, and their number falls with the local tail — and `⟨N⟩` is reported as `p = 1 − e^{−⟨N⟩}`, because at the anchor it is eight regions per toy, where the linear form printed `p = 1` across 3–4σ. A fresh-seed 10⁴ re-run (`docs/calibration/lee-2d-1e4-fixed.md`, seed `0x1ee6e026`, 942.6 s on 16 threads, χ̄ = 8.128 ± 0.031) agrees with the toys to 1–8% from 3σ to 4.5σ (1.01×, 0.99×, 0.93×, 0.92×), and the model's 3σ threshold moved from 5.01σ to **4.87σ** against the toys' 4.90σ. At that threshold 14 of 10 000 toys exceeded it: rate 1.400e-3, 4% above nominal, with Wilson 95% [8.342e-4, 2.349e-3] — so the point is on nominal but the interval straddles the ±20% band and the verdict is **UNDECIDED**, not PASS. Ten thousand toys cannot decide ±20%, so the box waited on a 10⁵ campaign of the corrected model rather than on another fix — and that campaign has now run on a seed that did not diagnose the fault (`docs/calibration/lee-2d-1e5-fixed.md`, seed `0x51dec0de`, 4598.2 s on 4 CI cores, χ̄ = 8.120 ± 0.010). The model tracks the toys at 1.01×, 1.00×, 0.95×, 0.95× and 1.04× from 3σ to 5σ; its 3σ threshold is 4.87σ against the toys' empirical 4.85σ; and 128 of 100 000 toys cleared it — rate 1.280e-3, 5% below nominal, Wilson 95% [1.077e-3, 1.522e-3] against the ±20% band [1.080e-3, 1.620e-3]. The interval's floor misses by 0.3%, which is a sample-size miss and not a model one: three times the toys halve the half-width again and decide it, so the box stays unchecked until a campaign says otherwise. Recovery, the other half of the claim, is now tested (`a_planted_void_is_found_where_it_was_placed`): a radius-4.5 full void at a known voxel of a two-view scene is found at exactly that voxel (0.00 offset, Z 15.6, fitted depth 1.00), and after a 300-toy correction the null's 99th percentile (4.62) stays below it, so the look-elsewhere penalty does not erase a real signal. A first version of that test used one parallel view, and its peak was flat along the void's whole row with Z identical at every x: a single view cannot localise along its own rays, so the fixture needed angular diversity before "where" was a question with an answer. A 3-D campaign of the same bench (`UMBRA_LEE_DIM=3`, 40³ voxels, 20 000 toys, `docs/calibration/lee-3d-40-2e4.md`, χ̄ = 19.60 ± 0.05) puts the law at 1.02×, 1.14×, 1.15×, 1.04× and 1.02× from 3σ to 5σ on a scene where an excursion is a sheet rather than a strip, so the normalisation is not an artefact of the plane. The campaign also stopped being something only a human reads: the report carries a machine-readable verdict, a 10⁴ run that cannot decide dispatches its own 10⁵ confirmation, and a comparable run whose verdict differs from `docs/calibration/recorded-verdict.txt` opens an issue
- [x] The campaign at the criterion's own size, and the gate that runs it: 3 × 10⁵ toys on the plane (`docs/calibration/lee-2d-3e5.md`, seed `0xc0ffee`, 18 716.4 s on 16 threads, χ̄(u₀) = 8.117 ± 0.006) put 426 toys over the model's 4.87σ threshold — rate 1.420e-3, 5% above nominal, Wilson 95% [1.291e-3, 1.561e-3] inside the ±20% band [1.080e-3, 1.620e-3], so the 2-D rate criterion is **PASS** and the box above closes. The extrapolation held to 1.01× at 3σ and the empirical 3σ crossing is 4.85σ against the model's 4.87σ, so nothing that produced the threshold had to move for the rate to pass. The campaign also stopped being something only a human reads: the report's `needs` field says how many toys its interval would need before it fits ±20%, the job dispatches that number rather than a constant (capped at 3 × 10⁵, the most its timeout can run), and `docs/calibration/recorded-verdict.txt` keeps one line per search dimension, so a volume records its own verdict instead of being compared with a plane's. A fresh seed repeats the plane's campaign at the same size (`docs/calibration/lee-2d-3e5-fresh-seed.md`): 441 of 300 000, rate 1.470e-3, Wilson [1.339e-3, 1.614e-3] inside the band, and 0.5σ of the pooled count from the recorded 426 — so the PASS belongs to the campaign's size rather than to a lucky seed. The 3-D criterion is decided by its own campaign (next bullet)
- [x] The volume's criterion, decided at a size a campaign can afford: on `40³` a toy cost 10.9 ms, so the ~1.1 × 10⁷ toys the 20 000-toy point needed was 32 hours — open by budget, not by physics. `UMBRA_LEE_GRID` now defaults to `24³` in 3-D, where a toy costs 2.2 ms and the extrapolation still sits inside the 20% target (1.08× at 3σ, 0.97× at 4σ), and (`docs/calibration/lee-3d-24-1e6.md`, seed `0x600d1de5`, 2 532.0 s on 16 threads, χ̄(u₀) = 6.091 ± 0.003) 10⁶ toys put 1233 over the model's 4.81σ threshold — rate 1.233e-3, 9% below nominal, Wilson 95% [1.166e-3, 1.304e-3] inside the ±20% band, so the 3-D rate criterion is **PASS** too, on a sample five times the 202 245 its `needs` asks for. The law holds to 1.05–1.12× from 3σ to 5σ, and the 5σ row is the first with toys in it (480 above, 1.08×), so the extrapolation there is measured rather than quoted. The volume's rate sits significantly below nominal while the plane's sits significantly above it, which is what a model good to a few percent at 3σ looks like from two sides: the criterion tests the interval, not the point
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
- [x] `LEE_CALIB` wired into the page: after every scan the core draws 200 background-only toys from the same λ⁰, scans them with the same template bank, and reports where the observed peak sits in that maximum-`Z` null — so the page states the peak's empirical global `p` and how many toys matched it, instead of leaving a local `Z` to imply a search it never corrected for. The curve (global `p` every 0.05σ to 6σ), the largest toy peak and the extrapolated global-3σ threshold cross the boundary as one transferred `Float32Array` plus scalars; the threshold is `null` unless the Euler anchor fired, because extrapolating from zero excursion sets would return the local `p` and call it global. `check-browser` pins that the calibration is about the same peak the scan reported and that the index now holds six results. 200 toys is enough to say "this peak is ordinary", not enough to pin a rate — the 10⁵-toys campaign that decides M3 stays a bench and a CI job (§10.3). The curve is drawn beside the panels (`web/src/GlobalPPlot.tsx`, geometry in `plot.ts`): global `p` on a log axis to 6σ, with the observed peak and the global-3σ rate marked, so the correction can be read rather than only quoted. The toy count is a control from 50 to 1 000, and the page prints the measured round trip beside it, which keeps the cost of a finer null visible instead of assumed; `check-browser` moves that control to 50 and requires the new count and a measured time, and requires the curve to arrive as 121 SVG points. When the toys formed an excursion at the 2.5σ Euler anchor, the Gross–Vitells extrapolation is drawn too: dashed, starting one step above the anchor so that the local `p` at the anchor does not draw a vertical jump no data has, and labelled as a model rather than a measurement. The normalisation crosses the boundary as `meanEulerAtU0` instead of being re-derived in JavaScript, where it could disagree with the threshold computed from the same number, and `check-browser` requires the drawn path and the figure's own `data-model` claim to agree. The peak's global `p` now carries its 95% Wilson half-width and the smallest `p` the toy count can express — at 200 toys, 0.0025, which is above the 3σ rate, so the figure shows where the toys run out instead of implying they reached it. The branch carries `⟨χ⟩`'s standard error as a shaded band, because the anchor is a mean over toys and a single line claimed a normalisation the campaign did not measure; and the largest maximum in the null is marked with a dotted line labelled where it stops, so the measured part of the curve visibly ends where the model begins. `check-browser` requires the band to hold exactly as many points as the curve it bands, and the marker to sit above the anchor — which is where a measured excursion at `u₀` puts it
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
