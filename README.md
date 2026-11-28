# UMBRA — An Interactive Muon-Tomography Observatory

<p align="center">
  <a href="https://github.com/JoeyHammoth/project-umbra/actions/workflows/ci.yml"><img alt="CI status" src="https://github.com/JoeyHammoth/project-umbra/actions/workflows/ci.yml/badge.svg?branch=main"></a>
  <a href="LICENSE"><img alt="Licence: MIT" src="https://img.shields.io/badge/licence-MIT-blue"></a>
</p>

Cosmic-ray muons image the inside of large structures. Stone absorbs them far
more than air, so a hidden chamber shows up as an excess of muons from a
particular direction — that is how the ScanPyramids team found the Big Void in
2017. **UMBRA** turns that into an instrument you can drive in a browser: build a
pyramid, hide a chamber, place detectors, watch muons rain down, reconstruct the
density, and then do the statistics properly — is the void real, how confident
are we, and where should the next detector go?

Everything runs client-side. Static hosting, no accounts, no backend, £0/month.

> **Status: early, but it runs end to end.** A 2-D scene is traversed into a CSR
> path-length operator, counts are simulated from it, A1 and A2 reconstruct them,
> and the A6 scan reports a global significance calibrated against a
> 3×10⁵-toy campaign on the plane and a 10⁶-toy campaign on a volume, both
> passing ([docs/ROADMAP.md](docs/ROADMAP.md)). A8/A11 close out M3;
> inference, experiment design and the observatory (M4–M6) are still open.

## Why UMBRA?

- **The problem.** Muon tomography is usually taught as a picture — muons pass
  through, dense rock blocks them — and the picture hides the three parts that
  decide whether a result means anything: the inverse problem is limited-angle,
  the statistics are a look-elsewhere effect over millions of candidate voids,
  and the next detector placement decides whether the answer was ever reachable.
  UMBRA makes each of those something you can turn a dial on and watch.
- **Who it is for.** Physicists and engineers learning muography, and anyone who
  wants to see why an uncorrected 3σ local excess is not a discovery.
- **How it differs.** Geant4, and the muography codes built on it, is the right
  tool for simulating a real detector at the particle level; it is not built to
  run in a tab or to explain a significance to you. Published monument analyses
  are bespoke pipelines fitted to one structure. UMBRA is deliberately a
  validated approximation with documented error — the approximations are listed
  in [docs/PHYSICS.md](docs/PHYSICS.md) — and the error is shown on screen rather
  than left in a paper.

## What it will do

| Layer | Responsibility |
|---|---|
| `umbra-core` | Grids, CSR path-length operator and its adjoint, primitives, RNG |
| `umbra-physics` | Muon flux, CSDA energy loss, multiple scattering, detector model |
| `umbra-forward` | System builder (DDA traversal), forward/adjoint model, analytic chords |
| `umbra-recon` | A1–A5 solvers: transmission EM, Poisson MAP, PDHG, sparse deviation, PnP |
| `umbra-stats` | A6/A8/A11: profile-likelihood scans, Asimov, look-elsewhere, SPRT, SBC |
| `umbra-infer` | A7: parametric void posteriors, ensemble MCMC, nested sampling |
| `umbra-design` | A9: Asimov utility maps and greedy detector placement |
| `umbra-vis` | A10: Fisher sensitivity, local PSF, null-space modes, coverage roses |
| `umbra-wasm` | Thin `wasm-bindgen` façade — marshals data, contains no science |
| `umbra-cli` | Native tools: table generation, oracle parity, SBC datasets, benchmarks |
| `web` | React + TypeScript front end, workers, Three.js renderer, plots |

Dependency rule: `core ← physics ← forward ← {recon, stats, infer, design, vis} ← wasm`.
Only `wasm` and `cli` are leaves. No cycles; see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quickstart

```bash
# compute core (no web dependencies; runs natively and compiles to wasm)
cargo test --workspace
cargo run -p umbra-cli

# front end
cd web && npm ci && npm run dev
```

The design specification, including all the equations, the algorithm portfolio,
the offline-tooling and testing strategy, the risk register and the appendices,
lives at [source/UMBRA_DESIGN_SPEC.md](source/UMBRA_DESIGN_SPEC.md) (v0.2,
complete through §14).
The original CT lab that seeded the ray-integral picture is preserved unmodified
in [source/ctlab](source/ctlab).

## Documentation map

| Document | Contents |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Layers, crates, task graph, Rust ↔ TS contract, invariants |
| [docs/PHYSICS.md](docs/PHYSICS.md) | Forward model, constants, fidelity levels, documented approximations |
| [docs/ALGORITHMS.md](docs/ALGORITHMS.md) | The A1–A13 portfolio, what each buys, current status |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Milestones M0–M6 with exit criteria |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Local setup, commands, CI, offline tooling |
| [docs/adr](docs/adr) | Architecture decision records |
| [GLOSSARY.md](GLOSSARY.md) | Domain vocabulary used in code, issues and docs |
| [Wiki](https://github.com/JoeyHammoth/project-umbra/wiki) | Same material, browsable |

## Contributing

Contributions are welcome, including docs fixes and a failing test for a bug you
found. See [CONTRIBUTING.md](CONTRIBUTING.md); the short version is one ticket
per branch, a conventional commit subject, and PRs that state acceptance
criteria and evidence.

New here? Look for [`good first issue`](https://github.com/JoeyHammoth/project-umbra/issues?q=is%3Aopen+label%3A%22good+first+issue%22)
for something sized to an evening, or [`help wanted`](https://github.com/JoeyHammoth/project-umbra/issues?q=is%3Aopen+label%3A%22help+wanted%22)
for work that needs a hand. Comment on the issue and it is yours.

## Support

| I want to… | Go here |
|---|---|
| Ask a question or float an idea | [Discussions](https://github.com/JoeyHammoth/project-umbra/discussions) |
| Report a bug or request work | [Issues](https://github.com/JoeyHammoth/project-umbra/issues/new/choose) |
| Report a vulnerability | [SECURITY.md](SECURITY.md) — privately, never in an issue |

More in [SUPPORT.md](SUPPORT.md).

## Scientific honesty

UMBRA is not a predictor of chambers in real monuments, and not a Geant4-class
transport code. It is a validated approximation with documented error. Every
scenario is schematic and labelled as such; every reconstruction is shown with an
indication of how much to believe it.

## Licence

MIT — see [LICENSE](LICENSE). Third-party material vendored into the tree is
listed in [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md); the only entry today
is the Random123 known-answer vector file used as RNG test data, which stays
under its own BSD-3-Clause terms.

## References

- Gaisser, *Cosmic Rays and Particle Physics*; Guan et al. (2015) for the
  large-angle flux correction.
- Groom, Mokhov & Striganov (2001), muon stopping-power and range tables.
- Morishima et al. (2017), the ScanPyramids Big Void; Procureur et al. (2023),
  the North Face Corridor.
- Lange & Carson (1984); Erdoğan & Fessler (1999); Chambolle & Pock (2011);
  Beck & Teboulle (2009); Cowan et al. (2011); Gross & Vitells (2010);
  Goodman & Weare (2010); Skilling (2006); Wald (1945); Nemhauser et al. (1978);
  Talts et al. (2018).
