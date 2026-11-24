/**
 * The Rust <-> TypeScript contract, transcribed from design spec §4.6.
 *
 * This file is the single source of truth for task kinds, payload shapes and
 * events. The WASM facade is versioned (`api_version()`), and the client asserts
 * the handshake at startup so a stale cached module fails loudly.
 *
 * Bulk data never passes through serde: large buffers travel as transferable
 * `ArrayBufferView`s, never as JSON.
 */

export type TaskKind =
  | 'LOAD_TABLES'
  | 'BUILD_SYSTEM'
  | 'EXPECTED'
  | 'SIMULATE'
  | 'RECON'
  | 'SCAN'
  | 'LEE_CALIB'
  | 'MCMC'
  | 'DESIGN'
  | 'PSF'
  | 'EVENT_MC';

export type FidelityLevel = 'L0' | 'L1' | 'L2';

export type SolverId =
  | 'a1-transmission-em'
  | 'a2-poisson-map'
  | 'a3-opacity-pdhg'
  | 'a4-sparse-deviation'
  | 'a5-plug-and-play';


export interface VoidPriorSpec {
  centre: Array<{ min: number; max: number }>;
  semi: Array<{ min: number; max: number }>;
  withNullModel: boolean;
}

/** Payloads keyed by task. */
export interface Payloads {
  LOAD_TABLES: { tableVersion: string };
  /**
   * The scene as its §8.1 **document**, not a reshaped JS object: the Rust
   * loader rejects unknown keys and checks units, so the wire form is the same
   * text as `web/src/scenes/default.json`. See `SceneSpec::from_json_str`.
   */
  BUILD_SYSTEM: { sceneJson: string };
  /**
   * `density` is an override, in g cm⁻³. Omitted, the core evaluates the scene
   * it built — so a client never needs its own rasteriser. `level` likewise
   * defaults to the scene's own fidelity.
   */
  EXPECTED: { systemKey: string; density?: ArrayBufferView; level?: FidelityLevel };
  SIMULATE: { expectedKey: string; seed: string; exposureSeconds: number };
  RECON: {
    countsKey: string;
    solver: SolverId;
    params: Record<string, number>;
    iterations: number;
  };
  SCAN: { countsKey: string; scalesVoxels: number[]; profileNuisances: boolean };
  LEE_CALIB: { scanKey: string; toys: number; u0: number };
  MCMC: { countsKey: string; prior: VoidPriorSpec; walkers: number; steps: number };
  DESIGN: { sceneKey: string; candidateCount: number; kappa: number };
  PSF: { systemKey: string; voxel: number };
  EVENT_MC: { sceneKey: string; seed: string; tracks: number };
}

/** Results keyed by task. */
export interface Results {
  LOAD_TABLES: { tableVersion: string; bytes: number };
  BUILD_SYSTEM: { systemKey: string; rows: number; cols: number; nnz: number };
  EXPECTED: { lambda: ArrayBufferView };
  SIMULATE: { counts: Uint32Array };
  RECON: { density: ArrayBufferView; iterations: number; objective: number | null };
  SCAN: {
    z: ArrayBufferView;
    alphaHat: ArrayBufferView;
    maxZ: number;
    voxel: number;
    peakAlpha: number;
    peakP: number;
    /**
     * Transpose and back projection, in milliseconds — or `-1` in the browser.
     *
     * `wasm32-unknown-unknown` has no `Instant::now`, so the scan runs untimed
     * there. Reporting a measured `0` would read as "too fast to measure" when
     * the truth is "not measured at all", and the spec §6.5 budget is about the
     * correlation, so a fabricated zero would be exactly the wrong answer.
     */
    setupMs: number;
    correlateMs: number;
  };
  LEE_CALIB: { globalPValues: ArrayBufferView };
  MCMC: { samples: ArrayBufferView; logEvidence: number | null };
  DESIGN: { utility: ArrayBufferView; greedyOrder: number[] };
  PSF: { psf: ArrayBufferView; coverage: ArrayBufferView };
  EVENT_MC: { tracks: ArrayBufferView };
}

export type PayloadOf<K extends TaskKind> = Payloads[K];
export type ResultOf<K extends TaskKind> = Results[K];

export type Ok<T> = { ok: true; result: T };
export type Err = { ok: false; code: string; message: string };
export type TaskResult<T> = Ok<T> | Err;

export interface TaskRequest<K extends TaskKind = TaskKind> {
  /** Request id, so progress events can be matched and cancellations routed. */
  id: string;
  kind: K;
  /** Content hash of the inputs: identical keys share a cached result. */
  key: string;
  /** 0 = interactive, 2 = background. */
  priority: 0 | 1 | 2;
  payload: PayloadOf<K>;
}

export type TaskEvent =
  | { id: string; type: 'progress'; frac: number; snapshot?: ArrayBufferView }
  | { id: string; type: 'result'; result: ResultOf<TaskKind>; transfer?: ArrayBuffer[] }
  | { id: string; type: 'error'; code: string; message: string };

/** Pipeline stages, in the order the ComputeBus runs them. */
export const PIPELINE_STAGES: ReadonlyArray<{ kind: TaskKind; detail: string }> = [
  { kind: 'LOAD_TABLES', detail: 'Flux and range tables, cached in IndexedDB' },
  { kind: 'BUILD_SYSTEM', detail: 'DDA traversal → CSR path-length operator' },
  { kind: 'EXPECTED', detail: 'Opacity → E_min → integral flux → λ' },
  { kind: 'SIMULATE', detail: 'Checkpointed Poisson counts for any exposure' },
  { kind: 'RECON', detail: 'A1–A5 density reconstruction with uncertainty' },
  { kind: 'SCAN', detail: 'Multi-scale template scan with nuisance profiling' },
  { kind: 'LEE_CALIB', detail: 'Look-elsewhere correction (MC + Euler characteristic)' },
  { kind: 'MCMC', detail: 'Parametric void posterior and Bayes factor' },
  { kind: 'DESIGN', detail: 'Asimov utility map and greedy next-detector order' },
  { kind: 'PSF', detail: 'Fisher sensitivity, local PSF and coverage rose' },
  { kind: 'EVENT_MC', detail: 'Event-level muon tracks for the rain visualisation' },
];
