# Development

## Prerequisites

| Tool | Version | Needed for |
|---|---|---|
| Rust | stable (pinned by `rust-toolchain.toml`) | compute core |
| `wasm32-unknown-unknown` target | — | building `umbra-wasm` |
| Node.js | 20+ (22 LTS recommended) | front end |
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
5. `npm ci && npm run build` in `web/`

Gates are not advisory: fix the cause, never the assertion.

## Offline tooling

The table generator, oracle parity runner and SBC dataset dumps are `umbra-cli`
subcommands (M1 onwards). Their outputs are versioned assets checked into
`web/public/tables/<version>/`, never generated at build time.

## Troubleshooting

- **`cargo: command not found`** — install the toolchain from rustup; nothing in
  this repo vendors a compiler.
- **`wasm-bindgen` version mismatch** — the CLI must match the `wasm-bindgen`
  crate version; reinstall with the version from `Cargo.lock`.
- **Blank page on GitHub Pages** — the build must use the `/project-umbra/` base
  path. Cloudflare Pages deployments set `UMBRA_BASE=/`.
