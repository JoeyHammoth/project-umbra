# ADR 0001 — Static client-side app with a Rust/WASM compute core

**Status:** accepted · **Date:** 2026-11-18

## Context

The app must do real inverse-problem science: build sparse path-length operators
over 262 k voxels, run iterative solvers, scan ~10⁶ void candidates, run MCMC
chains and bootstrap replicates. It must also cost nothing to run, work offline,
and be reproducible from a URL.

The scarce resource is information, not compute. Default 3-D runs use 4–16 k ray
cells, `L` has 5 × 10⁵ – 2 × 10⁶ non-zeros, and a CSR matvec of that size takes a
few milliseconds. A full solver run is seconds, not hours.

## Decision

Ship a **static site** — no backend, no accounts, no database — with all numerics
in a **Rust library compiled to WebAssembly**, driven from a TypeScript/React
front end, hosted on GitHub Pages and/or Cloudflare Pages.

Compute is **task-parallel across workers** (MCMC chains, null datasets, bootstrap
replicates) rather than data-parallel, so no `SharedArrayBuffer` and no
cross-origin isolation are required for the default build.

## Consequences

**Good**

- Zero running cost and no operational surface; the deployment is bytes on a CDN.
- Offline-capable PWA, shareable URLs, deterministic scene + RNG streams.
- Rust numerics are unit-testable natively and benchmarkable with `criterion`.
- The compute core is a plain library: CLI, parity runs and dataset generation all
  reuse the exact code that ships.

**Bad**

- Browser memory and threading ceilings (a few hundred MB, no shared memory by
  default) bound the high-resolution mode; hence the matrix-free operator and the
  96³ size ceiling.
- Cold start pays for a `.wasm` download (≤ 700 KB gzip budget), so tables and
  models must be lazy.
- Two build toolchains (cargo + npm) and a versioned FFI contract to keep honest.

## Alternatives rejected

- **Server-side compute:** cheap to build, but adds hosting cost, latency and an
  operational surface, violating the £0 and offline goals.
- **TypeScript-only numerics:** fine for presentation, but a 10⁶-non-zero iterative
  solver in JS does not fit the interactive latency budget, and there is no
  realistic path to SIMD plus native benchmarking.
- **Pyodide / Python in the browser:** large payload, awkward FFI, weaker
  performance for the inner loops, and it would duplicate the Python oracle.
