/**
 * The ComputeBus: one place where heavy work leaves the main thread.
 *
 * Spec §4.4–4.6 asks for tasks that are **content-keyed, cacheable,
 * cancellable and progressive**. Those four properties come from four small
 * mechanisms, and this file is deliberately nothing but them:
 *
 * - **Content keys.** `contentKey()` hashes a payload canonically (sorted keys,
 *   typed-array bytes included), so identical inputs produce an identical key.
 *   Tasks compose rather than re-derive: `EXPECTED`'s payload carries the
 *   `systemKey`, `RECON`'s carries the `countsKey`, so changing one scene field
 *   changes exactly the keys of its descendants.
 * - **Priority.** `0` interactive, `2` background, FIFO inside a priority, so a
 *   dragged slider never queues behind a calibration campaign.
 * - **Cancellation.** A queued task is dropped; a running task's worker is
 *   terminated and replaced, which is the only way to actually stop a WebAssembly
 *   kernel that does not poll for a token.
 * - **Progressive.** `progress` events reach the caller as they arrive.
 *
 * Bulk data never passes through serde: results arrive as transferred buffers.
 * The transfer list on a `result` event is forwarded untouched, and the worker's
 * buffer is detached by the transfer, which is what a test can observe.
 *
 * # Testability
 *
 * The bus takes a `WorkerHost` and a `CacheStore` instead of reaching for
 * `new Worker(...)` and `indexedDB` itself, so it runs under `node --test` with
 * a fake worker and an in-memory cache. The browser wiring lives in
 * `workers.ts`; the cache fallback lives in `cache.ts`.
 */

import type {
  PayloadOf,
  ResultOf,
  TaskEvent,
  TaskKind,
  TaskRequest,
} from './protocol.ts';

/** Version of the compute contract this bus speaks. */
export const BUS_API_VERSION = '0.2.0';

/** A stale cached `.wasm` or a mismatched client: always fatal, never ignored. */
export class ApiVersionMismatch extends Error {
  readonly expected: string;
  readonly actual: string;

  constructor(expected: string, actual: string) {
    super(
      `compute API mismatch: the client expects ${expected} but this build speaks ${actual}. ` +
        'Hard-reload to fetch the matching module.',
    );
    this.name = 'ApiVersionMismatch';
    this.expected = expected;
    this.actual = actual;
  }
}

/** The caller abandoned the task, or the bus was closed under it. */
export class TaskCancelled extends Error {
  readonly id: string;

  constructor(id: string) {
    super(`task ${id} was cancelled`);
    this.name = 'TaskCancelled';
    this.id = id;
  }
}

/** A task the worker refused; `code` is a stable identifier for the UI. */
export class TaskFailure extends Error {
  readonly code: string;

  constructor(code: string, message: string) {
    super(message);
    this.name = 'TaskFailure';
    this.code = code;
  }
}

/** The slice of a `Worker` the bus uses. */
export interface WorkerHandle {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  terminate(): void;
  onmessage: ((event: { data: TaskEvent }) => void) | null;
  onerror: ((event: unknown) => void) | null;
}

/** Where workers come from, and how many the machine can afford. */
export interface WorkerHost {
  /** `min(hardwareConcurrency − 1, 4)`, clamped to at least one. */
  readonly concurrency: number;
  spawn(): WorkerHandle;
}

/** A result that survived a round trip. */
export type CachedResult = { ok: true; result: unknown };

/** The cache the bus reads through. */
export interface CacheStore {
  get(key: string): Promise<CachedResult | undefined>;
  set(key: string, result: CachedResult): Promise<void>;
}

/** An in-memory cache: the fallback when IndexedDB is unavailable, and what the tests use. */
export class MemoryCache implements CacheStore {
  private readonly entries = new Map<string, CachedResult>();

  async get(key: string): Promise<CachedResult | undefined> {
    return this.entries.get(key);
  }

  async set(key: string, result: CachedResult): Promise<void> {
    this.entries.set(key, result);
  }
}

/** How many workers a machine should run: `hardwareConcurrency − 1`, capped at four. */
export function workerCount(hardwareConcurrency: number): number {
  return Math.min(Math.max(hardwareConcurrency - 1, 1), 4);
}

/** Options for a single submission. */
export interface SubmitOptions {
  /** Aborting rejects with `TaskCancelled` and frees the worker if it is running. */
  signal?: AbortSignal;
  /** Buffers to move to the worker rather than copy. They are unusable afterwards. */
  transfer?: Transferable[];
  /** Called for each `progress` event. */
  onProgress?: (frac: number, snapshot?: ArrayBufferView) => void;
}

/** Options for the bus itself. */
export interface BusOptions {
  host: WorkerHost;
  cache?: CacheStore;
  /** The version the client was built against; a mismatch throws in the constructor. */
  expectedApiVersion?: string;
}

interface Job {
  id: string;
  kind: TaskKind;
  key: string;
  priority: 0 | 1 | 2;
  payload: unknown;
  sequence: number;
  settled: boolean;
  resolve: (value: unknown) => void;
  reject: (reason: unknown) => void;
  onProgress?: (frac: number, snapshot?: ArrayBufferView) => void;
  signal?: AbortSignal;
  detachSignal?: () => void;
  transfer?: Transferable[];
}

interface Slot {
  handle: WorkerHandle;
  job: Job | null;
}

/**
 * Canonical bytes of a payload.
 *
 * Object keys are sorted and typed arrays are written as their raw bytes with
 * their type tag, so two payloads that mean the same thing hash the same and two
 * that differ anywhere hash differently. Hashing `JSON.stringify` instead would
 * make key order and float formatting part of the cache key.
 */
function canonicalBytes(value: unknown): Uint8Array<ArrayBuffer> {
  const out: number[] = [];
  const encoder = new TextEncoder();
  const pushBytes = (bytes: Uint8Array) => {
    const view = new DataView(new ArrayBuffer(4));
    view.setUint32(0, bytes.length);
    out.push(...new Uint8Array(view.buffer));
    for (const byte of bytes) out.push(byte);
  };
  const pushF64 = (number: number) => {
    const view = new DataView(new ArrayBuffer(8));
    view.setFloat64(0, number);
    for (const byte of new Uint8Array(view.buffer)) out.push(byte);
  };

  const encode = (item: unknown): void => {
    if (item === null || item === undefined) {
      out.push(0);
      return;
    }
    if (typeof item === 'number') {
      out.push(1);
      pushF64(item);
      return;
    }
    if (typeof item === 'string') {
      out.push(2);
      pushBytes(encoder.encode(item));
      return;
    }
    if (typeof item === 'boolean') {
      out.push(3, item ? 1 : 0);
      return;
    }
    if (ArrayBuffer.isView(item)) {
      const view = item as ArrayBufferView;
      out.push(4);
      pushBytes(encoder.encode(view.constructor.name));
      pushBytes(new Uint8Array(view.buffer, view.byteOffset, view.byteLength));
      return;
    }
    if (Array.isArray(item)) {
      out.push(5);
      const length = new DataView(new ArrayBuffer(4));
      length.setUint32(0, item.length);
      out.push(...new Uint8Array(length.buffer));
      for (const entry of item) encode(entry);
      return;
    }
    if (typeof item === 'object') {
      out.push(6);
      const record = item as Record<string, unknown>;
      const keys = Object.keys(record).sort();
      const length = new DataView(new ArrayBuffer(4));
      length.setUint32(0, keys.length);
      out.push(...new Uint8Array(length.buffer));
      for (const key of keys) {
        pushBytes(encoder.encode(key));
        encode(record[key]);
      }
      return;
    }
    throw new TypeError(`cannot hash a ${typeof item}`);
  };

  encode(value);
  return new Uint8Array(out);
}

/**
 * The content key of a task: SHA-256 over the canonical bytes of `kind` and
 * `payload`, as lowercase hex.
 *
 * Asynchronous because `crypto.subtle` is, and using it rather than a
 * hand-rolled hash means two machines agree on the key of the same scene.
 */
export async function contentKey(kind: TaskKind, payload: unknown): Promise<string> {
  const bytes = canonicalBytes({ kind, payload });
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('');
}

/**
 * Runs tasks on a pool of workers, in priority order, behind a content-keyed
 * cache.
 */
export class ComputeBus {
  private readonly host: WorkerHost;
  private readonly cache: CacheStore;
  private readonly slots: Slot[] = [];
  private readonly queue: Job[] = [];
  private readonly jobs = new Map<string, Job>();
  private sequence = 0;
  private closed = false;

  constructor(options: BusOptions) {
    this.host = options.host;
    this.cache = options.cache ?? new MemoryCache();
    if (
      options.expectedApiVersion !== undefined &&
      options.expectedApiVersion !== BUS_API_VERSION
    ) {
      throw new ApiVersionMismatch(options.expectedApiVersion, BUS_API_VERSION);
    }
  }

  /** Workers currently alive. */
  get workerTotal(): number {
    return this.slots.length;
  }

  /** Tasks waiting for a worker. */
  get queued(): number {
    return this.queue.length;
  }

  /**
   * Submits a task and resolves with its result.
   *
   * Rejects with `TaskCancelled` when the signal aborts or the bus closes, and
   * with `TaskFailure` when the worker reports an error.
   */
  async submit<K extends TaskKind>(
    request: TaskRequest<K>,
    options: SubmitOptions = {},
  ): Promise<ResultOf<K>> {
    if (this.closed) throw new TaskCancelled(request.id);
    if (options.signal?.aborted) throw new TaskCancelled(request.id);

    const cached = await this.cache.get(request.key);
    if (cached) {
      return cached.result as ResultOf<K>;
    }

    return new Promise<ResultOf<K>>((resolve, reject) => {
      const job: Job = {
        id: request.id,
        kind: request.kind,
        key: request.key,
        priority: request.priority,
        payload: request.payload as PayloadOf<K>,
        sequence: this.sequence++,
        settled: false,
        resolve: resolve as (value: unknown) => void,
        reject,
        onProgress: options.onProgress,
        signal: options.signal,
        transfer: options.transfer,
      };
      if (options.signal) {
        const onAbort = () => this.cancel(job);
        options.signal.addEventListener('abort', onAbort, { once: true });
        job.detachSignal = () => options.signal?.removeEventListener('abort', onAbort);
      }
      this.jobs.set(job.id, job);
      // The cache lookup above is asynchronous, so a signal can abort before this
      // executor ever runs. Checking again here is what makes a cancellation
      // issued immediately after `submit` still mean "never ran".
      if (options.signal?.aborted) {
        this.reject(job, new TaskCancelled(job.id));
        return;
      }
      this.queue.push(job);
      this.dispatch();
    });
  }

  /** Terminates every worker; in-flight tasks reject as cancelled. */
  close(): void {
    this.closed = true;
    for (const job of [...this.jobs.values()]) this.cancel(job);
    for (const slot of this.slots) slot.handle.terminate();
    this.slots.length = 0;
  }

  private dispatch(): void {
    while (true) {
      const job = this.next();
      if (!job) return;
      const slot = this.freeSlot();
      if (!slot) {
        this.queue.unshift(job);
        return;
      }
      this.run(slot, job);
    }
  }

  private next(): Job | undefined {
    if (this.queue.length === 0) return undefined;
    let best = 0;
    for (let index = 1; index < this.queue.length; index += 1) {
      const candidate = this.queue[index]!;
      const incumbent = this.queue[best]!;
      // Lower priority number wins; ties go to whoever arrived first.
      if (
        candidate.priority < incumbent.priority ||
        (candidate.priority === incumbent.priority && candidate.sequence < incumbent.sequence)
      ) {
        best = index;
      }
    }
    return this.queue.splice(best, 1)[0];
  }

  private freeSlot(): Slot | undefined {
    for (const slot of this.slots) {
      if (slot.job === null) return slot;
    }
    if (this.slots.length >= this.host.concurrency) return undefined;
    const slot: Slot = { handle: this.host.spawn(), job: null };
    this.slots.push(slot);
    return slot;
  }

  private run(slot: Slot, job: Job): void {
    slot.job = job;
    slot.handle.onmessage = (event) => this.receive(slot, job, event.data);
    slot.handle.onerror = (error) => {
      this.finish(slot, job);
      this.reject(job, new TaskFailure('worker', String(error)));
    };
    // `key` travels with the task because the core stores its systems and
    // expected counts by it: the value the broker hashed is the one the worker
    // must remember, and a message without it leaves the task unnamed.
    // `priority` stays here — it orders the queue and means nothing to a worker.
    // Input buffers travel the same way results do: transferred, not copied into
    // the worker's heap.
    slot.handle.postMessage(
      { id: job.id, kind: job.kind, key: job.key, payload: job.payload },
      job.transfer,
    );
  }

  private receive(slot: Slot, job: Job, event: TaskEvent): void {
    if (job.settled) return;
    if (event.type === 'progress') {
      job.onProgress?.(event.frac, event.snapshot);
      return;
    }
    if (event.type === 'error') {
      this.finish(slot, job);
      this.reject(job, new TaskFailure(event.code, event.message));
      return;
    }
    // A `result`: cache it, then hand the caller the buffers the worker
    // transferred. The transfer list is forwarded, not rebuilt, so the caller
    // knows which buffers are now theirs.
    this.finish(slot, job);
    void this.cache.set(job.key, { ok: true, result: event.result });
    this.settle(job, event.result);
  }

  private finish(slot: Slot, job: Job): void {
    slot.job = null;
    slot.handle.onmessage = null;
    slot.handle.onerror = null;
    this.dispatch();
    // `dispatch()` may have handed this slot a new job and cleared `job` above;
    // nothing else to do here, kept as a named step for the reader.
    void job;
  }

  private cancel(job: Job): void {
    if (job.settled) return;
    const queuedAt = this.queue.indexOf(job);
    if (queuedAt >= 0) {
      this.queue.splice(queuedAt, 1);
      this.reject(job, new TaskCancelled(job.id));
      return;
    }
    const slot = this.slots.find((entry) => entry.job === job);
    if (slot) {
      // A WebAssembly kernel cannot be interrupted from outside, so the only
      // honest cancellation is to kill the thread and spawn its replacement.
      slot.handle.terminate();
      slot.handle = this.host.spawn();
      slot.job = null;
      this.reject(job, new TaskCancelled(job.id));
      this.dispatch();
      return;
    }
    this.reject(job, new TaskCancelled(job.id));
  }

  private settle(job: Job, value: unknown): void {
    if (job.settled) return;
    job.settled = true;
    this.jobs.delete(job.id);
    job.detachSignal?.();
    job.resolve(value);
  }

  private reject(job: Job, reason: unknown): void {
    if (job.settled) return;
    job.settled = true;
    this.jobs.delete(job.id);
    job.detachSignal?.();
    job.reject(reason);
  }
}
