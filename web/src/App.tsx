import { useEffect, useRef, useState } from 'react';

import { DensityView } from './DensityView';
import { TaskFailure, contentKey } from './compute/bus';
import { createBrowserCompute } from './compute/compute';
import type { BrowserCompute, CacheKind } from './compute/compute';
import { PIPELINE_STAGES } from './compute/protocol';
import { NOMINAL_3_SIGMA } from './plot';
import { recordedVerdict, recordedVerdicts } from './recorded';
import { emptyTail, wilsonInterval } from './resolution';
import sceneJson from './scenes/default.json?raw';

/** The exposure the page opens on, in seconds. */
const DEFAULT_EXPOSURE = 3600;
const MIN_EXPOSURE = 600;
const MAX_EXPOSURE = 86_400;
const EXPOSURE_STEP = 600;

/**
 * Toys the page's look-elsewhere calibration runs.
 *
 * Two hundred is enough to say whether this peak is ordinary, in one worker and
 * well under a second; the campaign that pins the 3σ rate needs 10⁵ and runs in
 * CI (§10.3), not here.
 */
const LEE_TOYS = 200;

/** Where the Euler characteristic is measured, in sigma. */
const LEE_U0 = 2.5;

/**
 * Toy counts the page offers.
 *
 * The coarse end answers "is this peak ordinary?" in milliseconds; the fine end
 * costs more and buys a smaller Monte Carlo step. The measured cost is printed
 * beside the control, so the trade is visible instead of assumed.
 */
const TOY_CHOICES = [50, 200, 500, 1000];

/** A rate as `1.28e-3`, the way the campaign report quotes it. */
function formatRate(rate: number): string {
  return rate.toExponential(2);
}

/** Seconds as hours, which is how a muon exposure is usually quoted. */
function formatExposure(seconds: number): string {
  return `${(seconds / 3600).toFixed(1)} h`;
}

type BuildState =
  | { status: 'running' }
  | {
      status: 'ok';
      rows: number;
      cols: number;
      /** Voxels per row, as the rasteriser laid them out. */
      grid: number;
      nnz: number;
      lambda: ArrayBufferView;
      counts: Uint32Array;
      density: Float32Array;
      /** The scan's `Z` at every voxel, so the view can draw the statistic. */
      z: Float32Array;
      steps: number;
      objective: number | null;
      peakZ: number;
      peakVoxel: number;
      peakAlpha: number;
      /** The look-elsewhere calibration: what the whole search does under H0. */
      lee: {
        peakZ: number;
        globalPAtPeak: number;
        toysAtOrAbovePeak: number;
        toys: number;
        maxToyZ: number;
        zAt3Sigma: number | null;
        toysAboveU0: number;
        /** Mean Euler characteristic at the anchor, for the drawn model. */
        meanEulerAtU0: number;
        /** Its standard error, drawn as a band around that model. */
        eulerStderr: number;
        /** Global `p` at 0, 0.05, …, 6σ, for the drawn curve. */
        curve: Float32Array;
        /** Round trip of the calibration, measured here because wasm has no clock. */
        ms: number;
      };
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
      const density = state.density instanceof Float32Array ? state.density : null;
      if (!lambda || !counts || !density) {
        return (
          `bulk data came back as ${state.lambda?.constructor?.name ?? 'nothing'}, ` +
          `${state.counts?.constructor?.name ?? 'nothing'} and ` +
          `${state.density?.constructor?.name ?? 'nothing'}: it must be transferred as typed arrays`
        );
      }
      const events = counts.reduce((total, count) => total + count, 0);
      const objective =
        state.objective === null ? '' : `, objective ${state.objective.toPrecision(6)}`;
      // The scan's own timings are not in this line: the browser build has no
      // clock, so the worker reports `-1` rather than a measured zero.
      const scan =
        ` · Z peak ${state.peakZ.toFixed(2)} at voxel ${state.peakVoxel}` +
        ` (α̂ ${state.peakAlpha.toFixed(2)})`;
      return (
        `Assembled ${state.rows} rays × ${state.cols} voxels, ${state.nnz} non-zero path lengths · ` +
        `λ Float32Array(${lambda.length}) · counts Uint32Array(${counts.length}), ${events} events · ` +
        `ρ Float32Array(${density.length}) after ${state.steps} steps${objective}${scan}` +
        (cache === null ? '' : ` · cached in ${cache}`)
      );
    }
    case 'failed':
      return `${state.code}: ${state.message}`;
  }
}

/**
 * The landing page: five stages running for real, and the reconstruction drawn.
 *
 * `BUILD_SYSTEM` → `EXPECTED` → `SIMULATE` → `RECON` → `SCAN` runs on mount rather
 * than behind a button, so the wiring is exercised every time the page loads: a
 * broken worker, a mismatched module or a protocol slip says so instead of
 * leaving a plausible looking scaffold.
 *
 * The exposure slider re-runs the same sequence with a different `SIMULATE` key.
 * Nothing upstream of it changes identity — the operator and λ are content
 * addressed, so only the counts, the density and the scan are recomputed — and
 * the bus caches at 3600 s, so returning to the opening exposure is instant.
 */
export function App() {
  const [state, setState] = useState<BuildState>({ status: 'running' });
  const [cache, setCache] = useState<CacheKind | null>(null);
  const [requested, setRequested] = useState(DEFAULT_EXPOSURE);
  const [exposure, setExposure] = useState(DEFAULT_EXPOSURE);
  const [leeToys, setLeeToys] = useState(LEE_TOYS);
  const started = useRef<Promise<BrowserCompute> | null>(null);

  /** One bus for the page's lifetime: a slider move must not spawn a new pool. */
  const compute = (): Promise<BrowserCompute> => (started.current ??= createBrowserCompute());

  useEffect(() => {
    const created = compute();
    return () => {
      void created.then(({ bus }) => bus.close());
    };
  }, []);

  // A drag fires dozens of input events; the pipeline should run once it stops.
  // The image keeps showing the previous exposure until the new one lands.
  useEffect(() => {
    if (requested === exposure) return;
    const timer = setTimeout(() => setExposure(requested), 200);
    return () => clearTimeout(timer);
  }, [requested, exposure]);

  useEffect(() => {
    let live = true;
    void (async () => {
      try {
        const { bus, cache: kind } = await compute();
        if (!live) return;
        setCache(kind);

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

        const simulatePayload = { expectedKey, seed: '7', exposureSeconds: exposure };
        const countsKey = await contentKey('SIMULATE', simulatePayload);
        const counts = await bus.submit({
          id: crypto.randomUUID(),
          kind: 'SIMULATE',
          key: countsKey,
          priority: 0,
          payload: simulatePayload,
        });

        // A2 rather than A1 because the default scene is L1 and A1's model is
        // Beer–Lambert: the core refuses that combination instead of fitting it.
        const reconPayload = {
          countsKey,
          solver: 'a2-poisson-map' as const,
          params: {},
          iterations: 20,
        };
        const recon = await bus.submit({
          id: crypto.randomUUID(),
          kind: 'RECON',
          key: await contentKey('RECON', reconPayload),
          priority: 0,
          payload: reconPayload,
        });

        // A6 asks where a deficit of the reference density would fit best. The
        // corridor in the default scene is 1 m wide, which is two voxels at the
        // grid's 0.5 m, so the bank spans one and two.
        const scanPayload = { countsKey, scalesVoxels: [1, 2], profileNuisances: false };
        const scan = await bus.submit({
          id: crypto.randomUUID(),
          kind: 'SCAN',
          key: await contentKey('SCAN', scanPayload),
          priority: 0,
          payload: scanPayload,
        });

        // The look-elsewhere correction: the same bank over background-only toys
        // drawn from the λ⁰ the data was tested against. Without it the page
        // would quote a local `Z` as though it were a claim about one position.
        const leePayload = {
          countsKey,
          scalesVoxels: [1, 2],
          toys: leeToys,
          u0: LEE_U0,
        };
        const leeStarted = performance.now();
        const lee = await bus.submit({
          id: crypto.randomUUID(),
          kind: 'LEE_CALIB',
          key: await contentKey('LEE_CALIB', leePayload),
          priority: 0,
          payload: leePayload,
        });
        const leeMs = performance.now() - leeStarted;

        if (live) {
          setState({
            status: 'ok',
            rows: build.rows,
            cols: build.cols,
            grid: build.grid[0],
            nnz: build.nnz,
            lambda: expected.lambda,
            counts: counts.counts,
            density: recon.density as Float32Array,
            z: scan.z as Float32Array,
            steps: recon.iterations,
            objective: recon.objective,
            peakZ: scan.maxZ,
            peakVoxel: scan.voxel,
            peakAlpha: scan.peakAlpha,
            lee: {
              peakZ: lee.peakZ as number,
              globalPAtPeak: lee.globalPAtPeak as number,
              toysAtOrAbovePeak: lee.toysAtOrAbovePeak as number,
              toys: lee.toys as number,
              maxToyZ: lee.maxToyZ as number,
              zAt3Sigma: lee.zAt3Sigma as number | null,
              toysAboveU0: lee.toysAboveU0 as number,
              meanEulerAtU0: lee.meanEulerAtU0 as number,
              eulerStderr: lee.eulerStderr as number,
              curve: lee.globalPValues as Float32Array,
              ms: leeMs,
            },
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
    };
    // A new toy count re-runs the calibration. Everything upstream shares a
    // content key, so the bus answers from cache and only the toys are computed.
  }, [exposure, leeToys]);

  const shown = state.status === 'ok' ? state : null;
  // The rate the page quotes is a measurement, so it carries the interval that
  // measurement has: 161 of 200 is 0.805, but only to about ±0.055.
  const interval = shown
    ? wilsonInterval(shown.lee.toysAtOrAbovePeak, shown.lee.toys)
    : null;

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

      <section aria-labelledby="reconstruction">
        <h2 id="reconstruction">Reconstruction</h2>
        <DensityView
          density={shown?.density ?? null}
          z={shown?.z ?? null}
          width={shown?.grid ?? 1}
          summary={
            shown
              ? `Reconstructed density on a ${shown.grid} × ${shown.cols / shown.grid} voxel grid, A2 after ${shown.steps} steps at ${formatExposure(exposure)}.`
              : 'Waiting for the reconstruction…'
          }
          scanSummary={
            shown
              ? `A6 score scan of the same data, Z peak ${shown.peakZ.toFixed(2)} at voxel ${shown.peakVoxel}. The panel is fixed at 0–5σ, so brightness is comparable across exposures; the default scene's own data is H0, and this null is the honest result.`
              : 'Waiting for the scan…'
          }
          globalP={shown?.lee.curve ?? null}
          leePeakZ={shown?.lee.peakZ ?? 0}
          leeToys={shown?.lee.toys ?? 0}
          leeMeanEuler={shown?.lee.meanEulerAtU0 ?? 0}
          leeEulerStderr={shown?.lee.eulerStderr ?? 0}
          leeU0={LEE_U0}
          leeMaxZ={shown?.lee.maxToyZ ?? 0}
        />
        {shown ? (
          <p
            className="detail"
            data-testid="lee"
            data-toys={shown.lee.toys}
            data-peak-z={shown.lee.peakZ}
            data-global-p={shown.lee.globalPAtPeak}
          >
            {shown.lee.toys === leeToys ? '' : `Recalibrating with ${leeToys} toys… `}
            Look-elsewhere: the scan's local peak of {shown.lee.peakZ.toFixed(2)}σ was
            matched or beaten by {shown.lee.toysAtOrAbovePeak} of {shown.lee.toys}{' '}
            background-only toys, so its global <em>p</em> is{' '}
            {shown.lee.globalPAtPeak.toFixed(3)}
            {interval ? ` ± ${interval.half.toFixed(3)} at 95%` : ''}, and{' '}
            {shown.lee.toys} toys cannot express anything below{' '}
            {emptyTail(shown.lee.toys).toFixed(4)} — a local significance says
            nothing about a search of {shown.cols} positions.{' '}
            {shown.lee.zAt3Sigma === null
              ? `No toy reached the ${LEE_U0}σ Euler anchor, so no extrapolated 3σ threshold is quoted.`
              : `Extrapolated from the ${shown.lee.toysAboveU0} toys above ${LEE_U0}σ (mean χ = ${shown.lee.meanEulerAtU0.toFixed(2)} ± ${shown.lee.eulerStderr.toFixed(2)}), a global 3σ would need ${shown.lee.zAt3Sigma.toFixed(2)}σ.`}{' '}
            The {shown.lee.toys} toys took {Math.round(shown.lee.ms)} ms here, round trip.
          </p>
        ) : null}
        {recordedVerdict ? (
          <p
            className="detail"
            data-testid="lee-record"
            data-verdict={recordedVerdict.verdict}
          >
            The campaign behind that number is not this page's:{' '}
            {recordedVerdict.toys.toLocaleString('en-US')} background-only toys through
            the same scan in CI give a global rate of{' '}
            {formatRate(recordedVerdict.rate)} (95% [{formatRate(recordedVerdict.low)},{' '}
            {formatRate(recordedVerdict.high)}]) against nominal{' '}
            {formatRate(NOMINAL_3_SIGMA)} ± 20%
            {recordedVerdict.threshold === null
              ? ''
              : `, so a global 3σ sits at ${recordedVerdict.threshold.toFixed(2)}σ there`}
            : <strong>{recordedVerdict.verdict.toLowerCase()}</strong>
            {recordedVerdict.verdict === 'UNDECIDED'
              ? ' — the interval straddles the band, so that sample does not decide the criterion yet.'
              : ` — the interval sits ${recordedVerdict.verdict === 'PASS' ? 'inside' : 'outside'} the band.`}{' '}
            <code>docs/calibration/recorded-verdict.txt</code> holds the line this quotes.
          </p>
        ) : null}
        {recordedVerdicts.length > 0 ? (
          <table className="detail" data-testid="lee-records">
            <caption>
              The M3 rate criterion, one row per search space. A plane and a volume
              are different searches, so each is decided — or not — by its own
              campaign; the last column is what that campaign says it would still
              need, which is why the volume is not simply "more toys than the
              plane".
            </caption>
            <thead>
              <tr>
                <th scope="col">Search</th>
                <th scope="col">Toys</th>
                <th scope="col">Rate at 3σ (95%)</th>
                <th scope="col">To decide</th>
                <th scope="col">Verdict</th>
              </tr>
            </thead>
            <tbody>
              {recordedVerdicts.map((entry) => (
                <tr key={entry.dim} data-testid={`lee-record-${entry.dim}d`}>
                  <td>
                    {entry.dim === 2
                      ? 'plane (2-D)'
                      : entry.dim === 3
                        ? 'volume (3-D)'
                        : `${entry.dim}-D`}
                  </td>
                  <td>{entry.toys.toLocaleString('en-US')}</td>
                  <td>
                    {formatRate(entry.rate)} [{formatRate(entry.low)},{' '}
                    {formatRate(entry.high)}]
                  </td>
                  <td>
                    {entry.needs === null
                      ? 'no sample size — the rate is outside the band'
                      : entry.needs <= entry.toys
                        ? 'decided by this sample'
                        : `~${entry.needs.toLocaleString('en-US')} toys`}
                  </td>
                  <td>{entry.verdict}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : null}
        <label className="toys">
          <span>Look-elsewhere toys</span>
          <select
            data-testid="toys"
            value={leeToys}
            onChange={(event) => setLeeToys(Number(event.target.value))}
          >
            {TOY_CHOICES.map((count) => (
              <option key={count} value={count}>
                {count}
              </option>
            ))}
          </select>
          <output data-testid="toys-ms">
            {shown && shown.lee.toys === leeToys ? `${Math.round(shown.lee.ms)} ms` : '…'}
          </output>
        </label>
        <label className="exposure">
          <span>Exposure</span>
          <input
            type="range"
            min={MIN_EXPOSURE}
            max={MAX_EXPOSURE}
            step={EXPOSURE_STEP}
            value={requested}
            onChange={(event) => setRequested(Number(event.target.value))}
          />
          <output>{formatExposure(requested)}</output>
        </label>
        <p className="detail">
          {requested === exposure
            ? 'More exposure is more counts, and a sharper image. The operator and λ keep their content keys across a change; only the counts, the density and the scan are recomputed.'
            : 'Recomputing at the new exposure…'}
        </p>
      </section>

      <section aria-labelledby="compute">
        <h2 id="compute">Compute</h2>
        <p role="status" data-testid="build-status">
          {describe(state, cache)}
        </p>
        <p>
          The first five stages run in a Web Worker against the Rust core
          compiled to WebAssembly, through the same <code>ComputeBus</code> every
          later stage will use. The operator stays in the worker; λ, the counts
          and the reconstructed density are transferred back as typed arrays
          rather than serialised.
        </p>
      </section>

      <section aria-labelledby="status">
        <h2 id="status">Pipeline</h2>
        <p>
          The stages below share one protocol. Those past <code>SCAN</code>{' '}
          are not wired yet — the scaffold exists so the store, the renderer and
          the remaining statistics can be built against a fixed contract. Both
          fields above are drawn from typed arrays the worker transferred: the
          reconstruction, and the scan's <code>Z</code> beside it.
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
