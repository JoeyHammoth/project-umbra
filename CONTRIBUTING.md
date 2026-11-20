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
bash tools/check-licensing.sh
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

## Licensing

UMBRA is MIT-licensed ([LICENSE](LICENSE)), and the rule for contributions is
inbound = outbound: what you send is offered under MIT, so do not paste code you
cannot relicense that way.

**What you may bring in**

| Source | Allowed | What it costs |
|---|---|---|
| Your own code | yes | nothing — it arrives under MIT |
| MIT, BSD-2/3-clause, Apache-2.0, ISC, CC0, Unlicense | yes | keep the copyright notice and disclaimer |
| SIL OFL font, as a bundled subset (spec §8.4) | yes | attribution in `THIRD-PARTY-NOTICES.md` |
| GPL, AGPL, LGPL, MPL | no | incompatible with shipping an MIT bundle; ask before even prototyping with it |
| Non-commercial, no-derivatives, "free for research" | no | not a licence this repo can honour |
| No stated licence | no | no licence means no permission |
| Photogrammetry, decorative textures, remote fonts | no | barred by spec §8.4 whatever their licence says |

Dependencies are not vendored — `cargo` and `npm` resolve them from their
registries and the versions stay pinned in `Cargo.lock` and
`web/package-lock.json`. Check a new dependency's licence against the table
before adding it.

**Registering vendored material**

Two registers, chosen by what the file is:

- **A test fixture** — any non-Rust file under `crates/*/tests/`: add one line to
  [tools/fixture-origins.txt](tools/fixture-origins.txt), marked `third-party`
  or `generated`. A `third-party` entry must also be named in
  [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md), with its licence and notice
  reproduced in full rather than summarised; a `generated` entry should say how
  to reproduce it.
- **A shipped web asset** — anything under `web/public/`: add a row to the asset
  inventory in spec §8.4, with a licence cell the repo may ship (`generated …`,
  `trained in-repo`, `SIL OFL`, `CC0`, `repo licence`).

Copy third-party files **byte-for-byte**, and record the upstream URL and the
revision you took. Never retype or hand-trim published data: the Random123
`kat_vectors` file is the worked example — complete file, upstream path, and the
SHA-256 of the vendored revision in the notices.

`bash tools/check-licensing.sh` (the **Licensing** CI job) enforces all of it: a
manifest with no `license`, an unregistered fixture, a `third-party` fixture
missing from the notices, or an unlisted or oddly licensed web asset. The
messages name the line to add. If the answer is not in this section, open an
issue rather than guessing — a licence mistake is the kind a revert does not
undo.

## Pull request checklist

- [ ] Linked issue (`Closes #NN`)
- [ ] No negated keyword: "does not close #NN" *still closes* `#NN`, because the
      matcher does not read the "not". Use "Advances #NN" for partial work.
- [ ] Acceptance criteria ticked in the description
- [ ] Evidence: failing-then-passing test, benchmark delta, or screenshot
- [ ] Gates above run and green
- [ ] Docs or glossary updated when behaviour or vocabulary changed
