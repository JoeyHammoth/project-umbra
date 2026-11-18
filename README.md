# UMBRA — An Interactive Muon-Tomography Observatory

Cosmic-ray muons image the inside of large structures. Stone absorbs them far
more than air, so a hidden chamber shows up as an excess of muons from a
particular direction — that is how the ScanPyramids team found the Big Void in
2017. **UMBRA** turns that into an instrument you can drive in a browser: build a
pyramid, hide a chamber, place detectors, watch muons rain down, reconstruct the
density, and then do the statistics properly — is the void real, how confident
are we, and where should the next detector go?

Everything runs client-side. Static hosting, no accounts, no backend, £0/month.

> **Status: scaffold.** The Rust workspace, the web shell, the CI gates and the
> documentation map exist. No physics is implemented yet beyond the range–energy
> relation and the core linear algebra; see [docs/ROADMAP.md](docs/ROADMAP.md).

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
cd web && npm install && npm run dev
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

See [CONTRIBUTING.md](CONTRIBUTING.md). Short version: one ticket per branch,
conventional commit subjects, `cargo test --workspace` and `npm run typecheck`
green, and PRs that state acceptance criteria and evidence.

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
