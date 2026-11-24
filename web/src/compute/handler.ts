/**
 * The worker side of a task: one `TaskRequest` in, one `TaskEvent` out.
 *
 * This is where "bulk results arrive as transferred buffers, never copied
 * through serde" actually happens. The WASM façade returns a `Float32Array` or
 * a `Uint32Array` that it allocated; this module posts it back with that exact
 * buffer in the transfer list, so the main thread receives the buffer rather
 * than a copy of it. Building the list here — beside the call that produced the
 * array — is what keeps a `.slice()` or a spread from creeping in later and
 * quietly turning a 60 MB scan into two.
 *
 * The handler takes the core as an interface so the contract is testable under
 * `node --test`, with no browser and no WASM runtime.
 */

import type { PayloadOf, TaskEvent, TaskRequest } from './protocol.ts';

/**
 * The `Z` field a scan returns, before the handler transfers it.
 *
 * `Results.SCAN` widens the fields to `ArrayBufferView` because the contract
 * cannot promise which typed array arrives; here they are known to be `f32`,
 * because the core allocates them.
 */
export interface ScanField {
  z: Float32Array;
  alphaHat: Float32Array;
  maxZ: number;
  voxel: number;
  peakAlpha: number;
  peakP: number;
  setupMs: number;
  correlateMs: number;
}

/**
 * What the core reports about an operator it assembled.
 *
 * `grid` is an `ArrayLike` rather than the protocol's tuple because that is what
 * the boundary actually yields: a wasm getter returning `Vec<u32>` arrives as a
 * `Uint32Array`. Widening it here lets `worker.ts` pass the real class without a
 * cast, and `handleTask` is where it becomes the pair the protocol promises.
 */
export interface BuildSummaryOut {
  systemKey: string;
  rows: number;
  cols: number;
  nnz: number;
  grid: ArrayLike<number>;
}

/** The façade, narrowed to the tasks a worker runs. */
export interface ComputeCore {
  /** Assembles the path-length operator; the scene crosses as the §8.1 document. */
  buildSystem(systemKey: string, sceneJson: string): BuildSummaryOut;
  /** λ for a density field, keyed so `SIMULATE` can reuse the per-second rates. */
  expected(
    systemKey: string,
    expectedKey: string,
    density: Float32Array,
    level?: string,
  ): Float32Array;
  /** The scene's own density field, for a caller with no override. */
  sceneDensity(systemKey: string): Float32Array;
  /** Poisson counts at an arbitrary exposure, remembered under `countsKey`. */
  simulate(
    countsKey: string,
    expectedKey: string,
    seed: bigint,
    exposureSeconds: number,
  ): Uint32Array;
  /**
   * Runs a solver over stored counts. Parameter names and values travel as two
   * arrays because the WASM boundary takes no records; the handler zips them
   * from the protocol's plain object.
   */
  recon(
    countsKey: string,
    solver: string,
    paramNames: string[],
    paramValues: number[],
    iterations: number,
  ): { density: Float32Array; iterations: number; objective?: number };
  /**
   * The A6 score scan over stored counts. `scalesVoxels` are the template radii,
   * in voxels; the field and the amplitude map both come back for transferring.
   */
  scan(countsKey: string, scalesVoxels: number[], profileNuisances: boolean): ScanField;
  /**
   * The look-elsewhere calibration for stored counts: background-only toys drawn
   * from the same λ⁰ the data was tested against, scanned with the same bank.
   */
  leeCalib(
    countsKey: string,
    scalesVoxels: number[],
    toys: number,
    u0: number,
  ): LeeCalibrationOut;
}

/**
 * The calibration before the handler transfers its curve.
 *
 * `Results.LEE_CALIB` widens the curve to `ArrayBufferView` because the contract
 * cannot promise which typed array arrives; here it is known to be `f32`,
 * because the core allocates it.
 */
export interface LeeCalibrationOut {
  globalPValues: Float32Array;
  peakZ: number;
  globalPAtPeak: number;
  toysAtOrAbovePeak: number;
  toys: number;
  maxToyZ: number;
  /**
   * `undefined` at the boundary, because a wasm `Option<f32>` getter comes back
   * absent rather than null; the handler normalises it to the protocol's `null`.
   */
  zAt3Sigma: number | null | undefined;
  toysAboveU0: number;
  meanEulerAtU0: number;
  eulerStderr: number;
}

/**
 * `<code>: <message>` — the format `umbra-wasm` renders a `CoreError` in.
 *
 * Parsed at the first `": "` and only accepted when the prefix looks like a
 * code, so a message that happens to contain a colon keeps its wording.
 */
const CORE_FAULT = /^([a-z][a-z_]*): ([\s\S]*)$/;

/** Splits a thrown core fault into its code and message. */
export function coreFault(error: unknown): { code: string; message: string } {
  const text = error instanceof Error ? error.message : String(error);
  const match = CORE_FAULT.exec(text);
  return match
    ? { code: match[1]!, message: match[2]! }
    : { code: 'compute', message: text };
}

/** Reinterprets a density view as `f32` without copying its bytes. */
function asFloat32(view: ArrayBufferView): Float32Array {
  if (view instanceof Float32Array) return view;
  if (view.byteLength % 4 !== 0) {
    throw new Error(
      `density: ${view.byteLength} bytes is not a whole number of f32 values`,
    );
  }
  return new Float32Array(view.buffer, view.byteOffset, view.byteLength / 4);
}

/** The buffer to transfer. A transfer list must name the array's own buffer. */
function transferable(array: Float32Array | Uint32Array): ArrayBuffer {
  return array.buffer as ArrayBuffer;
}

/**
 * Runs one task and returns the event to post back.
 *
 * Never throws: a task that fails becomes an `error` event, because an exception
 * escaping here would kill the worker and the pool would have to notice a dead
 * thread instead of a failed task.
 */
export function handleTask(core: ComputeCore, request: TaskRequest): TaskEvent {
  const id = request.id;
  // The core stores every system and every expected-count vector under this
  // key. A message that arrives without one used to fail deep inside the WASM
  // glue as `undefined.length`; failing here names the actual problem.
  if (typeof request.key !== 'string' || request.key.length === 0) {
    return {
      id,
      type: 'error',
      code: 'request',
      message: `${request.kind} arrived without a content key: the core stores results by it`,
    };
  }
  try {
    switch (request.kind) {
      case 'BUILD_SYSTEM': {
        const payload = request.payload as PayloadOf<'BUILD_SYSTEM'>;
        // The operator itself stays in the worker; only its shape goes home.
        // The fields are copied out explicitly: a wasm-bindgen class exposes
        // them as prototype getters with no own properties, and structured
        // clone copies own properties only, so posting it would arrive as `{}`.
        const summary = core.buildSystem(request.key, payload.sceneJson);
        return {
          id,
          type: 'result',
          result: {
            systemKey: summary.systemKey,
            rows: summary.rows,
            cols: summary.cols,
            nnz: summary.nnz,
            // The wasm getter hands back a Uint32Array; the protocol promises a
            // two-entry pair, so the shape is fixed here rather than at every
            // consumer.
            grid: [summary.grid[0]!, summary.grid[1]!],
          },
        };
      }
      case 'EXPECTED': {
        const payload = request.payload as PayloadOf<'EXPECTED'>;
        // No density in the payload means "the scene as authored": the core
        // hands back its own field, so the pipeline never grows a second
        // rasteriser in JavaScript that could disagree with the Rust one.
        const density = payload.density
          ? asFloat32(payload.density)
          : core.sceneDensity(payload.systemKey);
        const lambda = core.expected(payload.systemKey, request.key, density, payload.level);
        return {
          id,
          type: 'result',
          result: { lambda },
          transfer: [transferable(lambda)],
        };
      }
      case 'SIMULATE': {
        const payload = request.payload as PayloadOf<'SIMULATE'>;
        // `request.key` is the counts key: the observation is what `RECON` will
        // name, and its content key is already the identity of these counts.
        const counts = core.simulate(
          request.key,
          payload.expectedKey,
          BigInt(payload.seed),
          payload.exposureSeconds,
        );
        return {
          id,
          type: 'result',
          result: { counts },
          transfer: [transferable(counts)],
        };
      }
      case 'RECON': {
        const payload = request.payload as PayloadOf<'RECON'>;
        // Object key order is insertion order, and both arrays are built from
        // the same iteration, so names and values stay paired across the call.
        const params = payload.params ?? {};
        const names = Object.keys(params);
        const outcome = core.recon(
          payload.countsKey,
          payload.solver,
          names,
          names.map((name) => params[name]!),
          payload.iterations,
        );
        // The density is read once: a wasm getter converts the vector on every
        // call, so a second read would allocate a second array and the transfer
        // list would name the wrong buffer.
        const density = outcome.density;
        return {
          id,
          type: 'result',
          result: {
            density,
            iterations: outcome.iterations,
            objective: outcome.objective ?? null,
          },
          transfer: [transferable(density)],
        };
      }
      case 'SCAN': {
        const payload = request.payload as PayloadOf<'SCAN'>;
        const outcome = core.scan(
          payload.countsKey,
          payload.scalesVoxels,
          payload.profileNuisances,
        );
        // Both arrays are read once, for the same reason the density is: a wasm
        // getter converts the vector on every call, and a second read would give
        // the transfer list a buffer nobody is looking at.
        const z = outcome.z;
        const alphaHat = outcome.alphaHat;
        return {
          id,
          type: 'result',
          result: {
            z,
            alphaHat,
            maxZ: outcome.maxZ,
            voxel: outcome.voxel,
            peakAlpha: outcome.peakAlpha,
            peakP: outcome.peakP,
            setupMs: outcome.setupMs,
            correlateMs: outcome.correlateMs,
          },
          transfer: [transferable(z), transferable(alphaHat)],
        };
      }
      case 'LEE_CALIB': {
        const payload = request.payload as PayloadOf<'LEE_CALIB'>;
        const outcome = core.leeCalib(
          payload.countsKey,
          payload.scalesVoxels,
          payload.toys,
          payload.u0,
        );
        // Read once, and read explicitly: a wasm-bindgen class exposes these as
        // prototype getters with no own properties, so `{ ...outcome }` would
        // copy the curve and drop every scalar — which is how the page ended up
        // calling `toFixed` on `undefined`.
        const globalPValues = outcome.globalPValues;
        return {
          id,
          type: 'result',
          result: {
            globalPValues,
            peakZ: outcome.peakZ,
            globalPAtPeak: outcome.globalPAtPeak,
            toysAtOrAbovePeak: outcome.toysAtOrAbovePeak,
            toys: outcome.toys,
            maxToyZ: outcome.maxToyZ,
            zAt3Sigma: outcome.zAt3Sigma ?? null,
            toysAboveU0: outcome.toysAboveU0,
            meanEulerAtU0: outcome.meanEulerAtU0,
            eulerStderr: outcome.eulerStderr,
          },
          transfer: [transferable(globalPValues)],
        };
      }
      default:
        return {
          id,
          type: 'error',
          code: 'unimplemented',
          message: `${request.kind} has no worker entry point yet`,
        };
    }
  } catch (error) {
    const fault = coreFault(error);
    return { id, type: 'error', code: fault.code, message: fault.message };
  }
}
