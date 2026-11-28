# Development

## Prerequisites

| Tool | Version | Needed for |
|---|---|---|
| Rust | stable (pinned by `rust-toolchain.toml`) | compute core |
| `wasm32-unknown-unknown` target | — | building `umbra-wasm` |
| Node.js | 20.19+ or 22.12+ (22 LTS used in CI) | front end — Vite 8's `engines` floor |
| `wasm-bindgen-cli` | 0.2.130 | local `.wasm` bundles |
| Python | 3.10+ | reference oracle and table generation |

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli
```

## Commands

```bash
# Rust
cargo build --workspace                      # native
cargo test --workspace                       # unit + integration tests
cargo run -p umbra-cli                       # workspace smoke check
cargo bench -p umbra-recon --bench portfolio   # A1–A4 per-step cost + wall-clock to shared fit target
cargo bench -p umbra-stats --bench scan        # A6 scan vs its spec §5.11 budget (50 ms 2-D / 2 s 3-D)
cargo bench -p umbra-stats --bench lee         # LEE calibration campaign (weekly/dispatch in CI)
cargo build -p umbra-wasm --target wasm32-unknown-unknown --release

# Web
cd web
npm ci                                       # lockfile committed
npm run dev                                  # compiles wasm core first, then Vite
npm test                                     # worker bus, caches, plot geometry, verdict parser
npm run typecheck
npm run build                                # wasm + tsc + vite
npm run build:wasm
npm run check:browser                        # headless-Chrome round trip; needs Chrome on PATH
npm run check:budget
npm run preview
```

The `.wasm` output is not committed. The web app loads it from the build
directory; `wasm-bindgen` generates the JS glue beside it.

## Repository layout

```text
crates/          Rust compute core (see docs/ARCHITECTURE.md)
web/             React + TypeScript front end
docs/            Architecture, physics, algorithms, roadmap, ADRs
source/          Design spec (unmodified) and the original CT lab
.github/         CI, templates, dependabot
```

`source/` is reference material. `source/ctlab` is the original teaching lab the
ray-integral picture comes from; it is preserved as-is and is what the Python
oracle parity tests compare against. Do not import it into the build.

## CI

`.github/workflows/ci.yml` runs on every push and PR:

1. `cargo fmt --all --check`
2. `cargo clippy --workspace --all-targets -- -D warnings`
3. `cargo test --workspace`
4. `cargo build -p umbra-wasm --target wasm32-unknown-unknown`
5. `bash tools/check-licensing.sh` — every manifest declares a licence; every
   non-Rust file under `crates/*/tests/` is registered in
   `tools/fixture-origins.txt` (a `third-party` fixture must also be named in
   `THIRD-PARTY-NOTICES.md`, where its notice lives); and every file under
   `web/public/` is covered by a row of the spec's §8.4 asset inventory whose
   licence is one this repo may ship
6. `bash tools/check-assets.sh` — every versioned asset directory matches its
   `manifest.json`: listed files exist at the recorded byte length and SHA-256,
   nothing unlisted has appeared beside them, and a manifest that cannot be
   parsed in full fails rather than silently checking nothing
7. `npm install --no-audit --no-fund && npm run build` in `web/`

Gates are not advisory: fix the cause, never the assertion.

## Offline tooling

The table generator, oracle parity runner and SBC dataset dumps are `umbra-cli`
subcommands. Their outputs are versioned assets checked into
`web/public/tables/<version>/`, never generated at build time.

```bash
cargo run -p umbra-cli -- gen-flux            # writes flux.bin + manifest.json
cargo run -p umbra-cli -- scene              # load, validate and report the default scene
cargo run -p umbra-cli -- golden             # regenerate the golden dataset
cargo run -p umbra-cli -- selftest           # load the default scene, evaluate L1, check the adjoint
cargo run -p umbra-cli --release -- bench    # build + forward-model timings

# Calibration campaigns (A11). Both write a report with a machine-readable
# verdict line; the recorded runs live in docs/calibration/.
cargo run -p umbra-cli --release -- sbc --engine a2 --draws 100 --posterior 100 \
    --seed 7 --report docs/calibration/sbc-a2-2d.md
cargo run -p umbra-cli --release -- bootstrap --engine a2 --datasets 40 \
    --replicates 60 --seed 7 --report docs/calibration/bootstrap-a2-2d.md
bash tools/sbc-verdict.sh docs/calibration/sbc-a2-2d.md   # the gate reads this line
```

### The default scene and its golden dataset

The shared scene lives in `web/src/scenes/default.json` in the format spec §8.1
fixes, and every solver reconstructs the same data derived from it. The loader
is `umbra_forward::scene::SceneSpec`, and it is strict on purpose: unknown keys
are rejected, unit strings are checked, and geometry that lies outside the voxel
grid fails to load rather than being silently invisible to the forward model.

The golden dataset in `crates/umbra-recon/tests/golden/default.json` records the
density field, `λ`, the sampled counts and a SHA-256 of those counts. It is
loaded by `crates/umbra-recon/tests/common` and pinned in two places:

```bash
cargo run -p umbra-cli -- golden            # rewrite it after an intended change
cargo test -p umbra-cli golden              # fails when the committed copy is stale
```

`cargo test -p umbra-recon` additionally re-derives `λ` and the counts from the
committed scene, so a physics change that moves any count cannot pass without the
golden being regenerated and reviewed. A diff in the golden is a change to every
reconstruction in the app: review it as physics, not as fixture churn.

### Oracle fixtures

The forward model's parity fixtures (spec §6.3) are generated by the independent
Python oracle in `tools/oracle/umbra_oracle` and committed to
`crates/umbra-forward/tests/fixtures/oracle/`. CI does not need Python: the
parity test reads the committed files, and a missing fixture fails it rather
than skipping.

```bash
cd tools/oracle
python -m umbra_oracle fixtures --out ../../crates/umbra-forward/tests/fixtures/oracle --seed 7
```

`python` is the repository `.venv`; its pins are in `tools/oracle/requirements.txt`.
The generator runs a self-test and refuses to write a fixture whose analytic
chord and voxel-field integral disagree beyond the tolerance that case declares,
so a wrong shape predicate fails generation instead of being committed. A diff
in these files after regeneration is a geometry change: review it as physics, not
as formatting.

The table format is fixed by spec §14.3 and carries a SHA-256 trailer; `gen-flux`
prints the digest it wrote, and the manifest records it. If you change any
physics constant, expect the digest to change — and say so in the PR, because
that changes every count in the app.

### Verifying the table against the reference quadrature

The flux parametrisation is checked against an independent `scipy` quadrature,
which lives outside the Rust build so it cannot share its mistakes:

```bash
python -m venv .venv
.venv/Scripts/pip install numpy scipy      # .venv/bin/pip on Linux/macOS
.venv/Scripts/python tools/oracle/check_flux.py
```

The script reimplements Guan et al. (2015) Eq. 3 independently, integrates it
with `scipy.integrate.quad`, and compares against the committed asset. At the
time of writing they agree to 0.08%, and both reproduce the ≈70 m⁻² s⁻¹ sr⁻¹
vertical intensity above 1 GeV. If the comparison ever fails, the table is wrong
or the citation is wrong — both are worth stopping for.

## Troubleshooting

- **`cargo: command not found`** — install the toolchain from rustup; nothing in
  this repo vendors a compiler.
- **`wasm-bindgen` version mismatch** — the CLI must match the `wasm-bindgen`
  crate version; reinstall with the version from `Cargo.lock`.
- **Blank page on GitHub Pages** — the build must use the `/project-umbra/` base
  path. Cloudflare Pages deployments set `UMBRA_BASE=/`.
