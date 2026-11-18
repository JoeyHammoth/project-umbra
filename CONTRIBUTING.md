# Contributing to UMBRA

## Shape of the work

Work is tracked as GitHub issues, one tracer-bullet slice each: a narrow path that
cuts through every layer it touches and can be demonstrated on its own. Issues
declare what blocks them; the frontier is whatever has no open blockers.

Branch naming:

```text
<type>/<issue-number>-<slug>      e.g. feat/12-2d-dda-traversal
```

Types: `feat`, `fix`, `docs`, `chore`, `ci`, `perf`, `refactor`, `test`.

Commit subjects are conventional and imperative, ≤ 72 characters:

```text
feat(forward): build CSR path lengths with 2-D DDA
docs(adr): record crate layering rule
ci: gate wasm builds on size budget
```

## Gates

Every PR must pass, locally and in CI:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cd web && npm run typecheck && npm run build
```

Numeric work additionally needs, where it applies:

- a unit test for the kernel itself (boundary cases, not just a happy path);
- a parity test against the Python oracle in `source/ctlab/python` when the
  quantity has an analytic or reference value;
- the adjoint identity `⟨Ax, y⟩ = ⟨x, Aᵀy⟩` via `umbra_core::dot_product_test`
  for any new linear operator;
- a `criterion` benchmark when the kernel is on an interactive path.

Test coverage target on Rust numerics is ≥ 85% lines. A PR that lowers it needs a
stated reason.

## House rules

1. **No science in the façade.** `umbra-wasm` marshals; algorithms live in their
   crates where they can be tested natively.
2. **Respect the dependency rule.** `core ← physics ← forward ← {recon, stats,
   infer, design, vis} ← wasm`. If a crate needs something from a sibling, the
   shared piece belongs lower down.
3. **Determinism.** Every random draw comes from a seeded stream derived from
   `(sceneKey, taskKind, index)`. A result that changes run to run is a bug, and
   a URL that does not reproduce its universe is a bug.
4. **No panics across the boundary.** Rust returns `Result<T, UmbraError>`; the
   façade converts to a discriminated union. `panic = "abort"` in release.
5. **Budget the bytes.** Check the gzip budget in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
   before adding a dependency to a route that loads eagerly.
6. **Label your work.** Every issue and PR carries an `area:` label and a
   `type:` label; PRs also state acceptance criteria and evidence.

## Pull request checklist

- [ ] Linked issue (`Closes #NN`)
- [ ] Acceptance criteria ticked in the description
- [ ] Evidence: failing-then-passing test, benchmark delta, or screenshot
- [ ] Gates above run and green
- [ ] Docs or glossary updated when behaviour or vocabulary changed
