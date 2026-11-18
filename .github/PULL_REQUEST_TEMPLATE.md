## Summary

<!-- The smallest view that makes the point: a tree, a diff sketch, a call tree, or one short paragraph. -->

## Acceptance criteria

<!-- Copy from the issue. Tick what this PR actually delivers. -->

- [ ]

## Evidence

- **Before:**
- **After:**

<!-- Test output, benchmark delta, or a screenshot. Prose alone is not evidence. -->

## Merge danger

**Door:** <!-- one-way or two-way -->

**Blast radius:** <!-- one word: local / package / repo / public-facing -->

## Checks

- [ ] `cargo fmt --all --check`
- [ ] `cargo clippy --workspace --all-targets -- -D warnings`
- [ ] `cargo test --workspace`
- [ ] `cd web && npm run build`  <!-- if the web app is touched -->
- [ ] Docs or glossary updated  <!-- if behaviour or vocabulary changed -->
