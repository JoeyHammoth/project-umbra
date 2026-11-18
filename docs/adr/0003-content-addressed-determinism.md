# ADR 0003 — Content-addressed tasks and seeded RNG streams

**Status:** accepted · **Date:** 2026-11-18

## Context

Every scene and random stream is supposed to be encoded in a shareable URL: same
URL, same universe. At the same time, results must be cached (some tasks take 30 s)
and tasks must be cancellable, reordered and spread across workers. A naive
"reseed the RNG when the task starts" design breaks reproducibility the moment the
scheduler changes order or a worker is reused.

## Decision

1. **Content keys.** Each task computes a hash of its inputs (`sceneKey`,
   `systemKey`, `countsKey`, …). The key is the cache key in memory and in
   IndexedDB, and the invalidation unit: changing a node invalidates every
   descendant.
2. **Stream ids, not global seeds.** A random stream is
   `H(sceneKey, taskKind, index)` with a counter-based generator (PCG64-DXSM or
   Philox-4x32, implemented in-crate). Any task, worker or replicate reproduces
   its own draws regardless of scheduling.
3. **Checkpointed binomial bridging for exposure.** Counts are pre-drawn at ~64
   log-spaced exposure checkpoints as independent Poisson increments; any
   intermediate exposure is obtained by binomial thinning between neighbouring
   checkpoints using a seed derived from `(cell, checkpoint)`. Counts are monotone
   in exposure and scrubbing backwards reproduces history exactly.

## Consequences

**Good**

- Reproducibility does not depend on thread count, worker identity or task order.
- Trivial caching: identical inputs hit, and a changed scene invalidates precisely
  its dependents.
- The exposure slider becomes a genuine time scrub rather than a re-simulation,
  which is what makes the visualisation feel physical.

**Bad**

- The hash must cover *everything* that affects the result; a missed input produces
  a stale cache hit, which is worse than a miss. Inputs are therefore hashed from
  the whole `SceneSpec`, not from hand-picked fields.
- Counter-based RNG must be implemented and tested in-crate (no `rand`
  dependency); getting the counter/stream split wrong yields correlated streams.
- Checkpoint bridging costs a little memory per ray cell and needs care at the
  endpoints.

## Verification

Property tests: the same URL twice yields byte-identical counts; counts at `t`
never decrease as `t` grows; counts at a checkpoint reached by thinning equal the
checkpoint drawn directly.
