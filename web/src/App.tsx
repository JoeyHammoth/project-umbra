import { useEffect, useState } from 'react';

import { TaskFailure, contentKey } from './compute/bus';
import { createBrowserCompute } from './compute/compute';
import type { CacheKind } from './compute/compute';
import { PIPELINE_STAGES } from './compute/protocol';
import sceneJson from './scenes/default.json?raw';

type BuildState =
  | { status: 'running' }
  | {
      status: 'ok';
      rows: number;
      cols: number;
      nnz: number;
      lambda: ArrayBufferView;
      counts: Uint32Array;
    }
  | { status: 'failed'; code: string; message: string };

function describe(state: BuildState, cache: CacheKind | null): string {
  switch (state.status) {
    case 'running':
      return 'Assembling the default scene…';
    case 'ok': {
      // The point of the transfer list is that these arrive as typed arrays and
      // not as objects; saying so in the status line makes a regression visible
      // instead of merely slower.
      const lambda = state.lambda instanceof Float32Array ? state.lambda : null;
      const counts = state.counts instanceof Uint32Array ? state.counts : null;
      if (!lambda || !counts) {
        return (
          `bulk data came back as ${state.lambda?.constructor?.name ?? 'nothing'} and ` +
          `${state.counts?.constructor?.name ?? 'nothing'}: it must be transferred as typed arrays`
        );
      }
      const events = counts.reduce((total, count) => total + count, 0);
      return (
        `Assembled ${state.rows} rays × ${state.cols} voxels, ${state.nnz} non-zero path lengths · ` +
        `λ Float32Array(${lambda.length}) · counts Uint32Array(${counts.length}), ${events} events` +
        (cache === null ? '' : ` · cached in ${cache}`)
      );
    }
    case 'failed':
      return `${state.code}: ${state.message}`;
  }
}

/**
 * The landing page, with the first three stages actually running.
 *
 * `BUILD_SYSTEM` → `EXPECTED` → `SIMULATE` runs on mount rather than behind a
 * button, so the wiring is exercised every time the page loads: a broken worker,
 * a mismatched module or a protocol slip says so instead of leaving a plausible
 * looking scaffold.
 */
export function App() {
  const [state, setState] = useState<BuildState>({ status: 'running' });
  const [cache, setCache] = useState<CacheKind | null>(null);

  useEffect(() => {
    let live = true;
    let bus: Awaited<ReturnType<typeof createBrowserCompute>>['bus'] | null = null;
    void (async () => {
      try {
        const compute = await createBrowserCompute();
        bus = compute.bus;
        if (!live) return;
        setCache(compute.cache);

        const buildPayload = { sceneJson };
        const build = await bus.submit({
          id: crypto.randomUUID(),
          kind: 'BUILD_SYSTEM',
          // The key is the hash of what the task reads, so an identical scene
          // shares a cached operator.
          key: await contentKey('BUILD_SYSTEM', buildPayload),
          priority: 0,
          payload: buildPayload,
        });

        // No density: the core evaluates the scene it just built.
        const expectedPayload = { systemKey: build.systemKey };
        const expectedKey = await contentKey('EXPECTED', expectedPayload);
        const expected = await bus.submit({
          id: crypto.randomUUID(),
          kind: 'EXPECTED',
          key: expectedKey,
          priority: 0,
          payload: expectedPayload,
        });

        const simulatePayload = { expectedKey, seed: '7', exposureSeconds: 3600 };
        const counts = await bus.submit({
          id: crypto.randomUUID(),
          kind: 'SIMULATE',
          key: await contentKey('SIMULATE', simulatePayload),
          priority: 0,
          payload: simulatePayload,
        });

        if (live) {
          setState({
            status: 'ok',
            rows: build.rows,
            cols: build.cols,
            nnz: build.nnz,
            lambda: expected.lambda,
            counts: counts.counts,
          });
        }
      } catch (error) {
        if (!live) return;
        setState({
          status: 'failed',
          code: error instanceof TaskFailure ? error.code : 'bus',
          message: error instanceof Error ? error.message : String(error),
        });
      }
    })();
    return () => {
      live = false;
      bus?.close();
    };
  }, []);

  return (
    <main className="shell">
      <header>
        <p className="eyebrow">Muon tomography observatory</p>
        <h1>UMBRA</h1>
        <p className="lede">
          Build a pyramid, hide a chamber, place detectors — then do the inverse
          problem for real: reconstruct the density, decide whether the void
          exists, and choose where the next detector should go.
        </p>
      </header>

      <section aria-labelledby="compute">
        <h2 id="compute">Compute</h2>
        <p role="status" data-testid="build-status">
          {describe(state, cache)}
        </p>
        <p>
          The first three stages run in a Web Worker against the Rust core
          compiled to WebAssembly, through the same <code>ComputeBus</code> every
          later stage will use. The operator stays in the worker; λ and the counts
          are transferred back as typed arrays rather than serialised.
        </p>
      </section>

      <section aria-labelledby="status">
        <h2 id="status">Pipeline</h2>
        <p>
          The stages below share one protocol. Those past <code>SIMULATE</code>{' '}
          are not wired yet — the scaffold exists so the store, the renderer and
          the solvers can be built against a fixed contract.
        </p>
        <ol className="stages">
          {PIPELINE_STAGES.map((stage) => (
            <li key={stage.kind}>
              <span className="kind">{stage.kind}</span>
              <span className="detail">{stage.detail}</span>
            </li>
          ))}
        </ol>
      </section>

      <footer>
        <p>
          Spec: <code>source/UMBRA_DESIGN_SPEC.md</code>. Docs:{' '}
          <code>docs/ARCHITECTURE.md</code>, <code>docs/ROADMAP.md</code>.
        </p>
      </footer>
    </main>
  );
}
