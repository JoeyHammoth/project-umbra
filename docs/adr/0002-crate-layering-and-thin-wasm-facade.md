# ADR 0002 — Crate layering and a thin WASM façade

**Status:** accepted · **Date:** 2026-11-18

## Context

The scientific surface is wide: grids and sparse operators, flux and energy loss,
system assembly, five reconstruction families, calibrated detection statistics,
Bayesian inference, experimental design and visibility analysis. A single crate
would make that one 20 k-line file that cannot be tested in isolation, and a
single WASM crate would put the science behind an interface that is awkward to
test natively.

## Decision

Split the core into crates with a **one-directional dependency rule**:

```text
core ← physics ← forward ← {recon, stats, infer, design, vis} ← wasm
                                        ↖ cli
```

- `umbra-wasm` **only marshals**: typed-array handoff, task entry points,
  cancellation, error mapping. It contains no algorithms.
- Only `wasm` and `cli` are leaf crates.
- A helper needed by two sibling crates belongs one level lower — never in a new
  "utils" crate, and never duplicated.

## Consequences

**Good**

- Every algorithm is testable natively, without a browser or a WASM runtime.
- The Cargo dependency graph *is* the architecture diagram, and a cycle is a build
  error rather than a design discussion.
- The CLI gets the same code for free: table generation and parity runs cannot
  drift from the shipped core.

**Bad**

- Ten manifests to keep in step; a shared version bump touches all of them
  (mitigated with `workspace.package` inheritance).
- Lower-level crates must anticipate what upper layers need, so shared types
  (`Csr`, `System`, `RayCell`) need care up front.
- Refactoring across layers is a multi-crate change rather than one edit.

## Enforcement

- `Cargo.toml` dependency lists are the guard; review rejects additions that
  violate the arrow.
- CI builds every crate for `wasm32-unknown-unknown`, so "it only works natively"
  cannot hide.
