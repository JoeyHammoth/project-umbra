# Development

## Prerequisites

| Tool | Version | Needed for |
|---|---|---|
| Rust | stable (pinned by `rust-toolchain.toml`) | compute core |
| `wasm32-unknown-unknown` target | — | building `umbra-wasm` |
| Node.js | 20.19+ or 22.12+ (22 LTS used in CI) | front end — Vite 8's `engines` floor |
| `wasm-bindgen-cli` | matches the crate version | local `.wasm` bundles |
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
cargo bench                                  # criterion kernels (once added)
cargo build -p umbra-wasm --target wasm32-unknown-unknown --release

# Web
cd web
npm install
npm run dev                                  # dev server
npm run typecheck
npm run build
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
5. `npm install --no-audit --no-fund && npm run build` in `web/`

Gates are not advisory: fix the cause, never the assertion.

## Offline tooling

The table generator, oracle parity runner and SBC dataset dumps are `umbra-cli`
subcommands. Their outputs are versioned assets checked into
`web/public/tables/<version>/`, never generated at build time.

```bash
cargo run -p umbra-cli -- gen-flux            # writes flux.bin + manifest.json
cargo run -p umbra-cli -- selftest           # build a scene, evaluate L1, check the adjoint
cargo run -p umbra-cli --release -- bench    # build + forward-model timings
```

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
