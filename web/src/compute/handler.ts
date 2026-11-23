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

import type { PayloadOf, Results, TaskEvent, TaskRequest } from './protocol.ts';

/** The façade, narrowed to the tasks a worker runs. */
export interface ComputeCore {
  /** Assembles the path-length operator; the scene crosses as the §8.1 document. */
  buildSystem(systemKey: string, sceneJson: string): Results['BUILD_SYSTEM'];
  /** λ for a density field, keyed so `SIMULATE` can reuse the per-second rates. */
  expected(
    systemKey: string,
    expectedKey: string,
    density: Float32Array,
    level?: string,
  ): Float32Array;
  /** The scene's own density field, for a caller with no override. */
  sceneDensity(systemKey: string): Float32Array;
  /** Poisson counts at an arbitrary exposure. */
  simulate(expectedKey: string, seed: bigint, exposureSeconds: number): Uint32Array;
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
        const counts = core.simulate(
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
