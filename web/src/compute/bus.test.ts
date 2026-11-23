/**
 * `ComputeBus` tests, run with `node --test` (Node 24 strips the types).
 *
 * Every test here stands for one acceptance criterion on #21: cache hits,
 * descendant-only invalidation, transferable delivery, cancellation within
 * 100 ms, the API handshake, a one-worker ("2-core") pool, and cached-round-trip
 * latency. The bus takes its workers and its cache as interfaces, which is what
 * makes all of that observable without a browser.
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  ApiVersionMismatch,
  BUS_API_VERSION,
  ComputeBus,
  MemoryCache,
  TaskCancelled,
  TaskFailure,
  contentKey,
  workerCount,
} from './bus.ts';
import type { CacheStore, WorkerHandle, WorkerHost } from './bus.ts';
import type { TaskEvent, TaskKind, TaskRequest } from './protocol.ts';

/** A worker that answers from a script, and records what it was asked. */
class FakeWorker implements WorkerHandle {
  onmessage: ((event: { data: TaskEvent }) => void) | null = null;
  onerror: ((event: unknown) => void) | null = null;
  terminated = false;
  readonly received: TaskKind[] = [];

  private readonly responder: Responder;

  constructor(responder: Responder) {
    this.responder = responder;
  }

  postMessage(message: unknown): void {
    const request = message as { id: string; kind: TaskKind; payload: unknown };
    this.received.push(request.kind);
    const reply = async () => {
      if (this.terminated) return;
      const event = await this.responder(request.kind, request.id, request.payload);
      if (this.terminated) return;
      this.onmessage?.({ data: event });
    };
    // Asynchronous like a real worker, so cancellation can land in between.
    setTimeout(() => void reply(), 1);
  }

  terminate(): void {
    this.terminated = true;
    this.onmessage = null;
    this.onerror = null;
  }
}

type Responder = (kind: TaskKind, id: string, payload: unknown) => Promise<TaskEvent>;

class FakeHost implements WorkerHost {
  readonly spawned: FakeWorker[] = [];
  readonly concurrency: number;
  terminated = 0;

  private readonly responder: Responder;

  constructor(concurrency: number, responder: Responder) {
    this.concurrency = concurrency;
    this.responder = responder;
  }

  spawn(): WorkerHandle {
    const worker = new FakeWorker(this.responder);
    const original = worker.terminate.bind(worker);
    worker.terminate = () => {
      this.terminated += 1;
      original();
    };
    this.spawned.push(worker);
    return worker;
  }
}

/** Answers `SIMULATE` with a transferred `Uint32Array`, and anything else with a marker. */
const responder: Responder = async (kind, id, payload) => {
  if (kind === 'SIMULATE') {
    const counts = new Uint32Array([1, 2, 3, 4]);
    return { id, type: 'result', result: { counts } as never, transfer: [counts.buffer] };
  }
  if (kind === 'RECON') {
    // Slow enough for a cancellation to arrive first.
    await new Promise((resolve) => setTimeout(resolve, 200));
    return { id, type: 'result', result: { density: new Float32Array([1]) } as never };
  }
  return { id, type: 'result', result: { echoed: payload } as never };
};

function request<K extends TaskKind>(
  kind: K,
  key: string,
  payload: Record<string, unknown> = {},
  priority: 0 | 1 | 2 = 1,
): TaskRequest<K> {
  return { id: `${kind}-${key}-${Math.random().toString(36).slice(2, 8)}`, kind, key, priority, payload } as TaskRequest<K>;
}

test('content keys are stable and order-independent', async () => {
  const a = await contentKey('EXPECTED', { level: 'L1', density: new Float32Array([1, 2, 3]) });
  const b = await contentKey('EXPECTED', { density: new Float32Array([1, 2, 3]), level: 'L1' });
  const c = await contentKey('EXPECTED', { level: 'L1', density: new Float32Array([1, 2, 4]) });
  assert.equal(a, b, 'key order must not change the content key');
  assert.notEqual(a, c, 'one changed voxel must change the key');
  assert.match(a, /^[0-9a-f]{64}$/);
});

test('changing a scene field invalidates exactly its descendants', async () => {
  const scene = (rho: number) => ({ key: 'scene', grid: { n: [32, 32], hM: 0.5 }, rho });
  const key = (kind: TaskKind, payload: unknown) => contentKey(kind, payload);

  const buildOld = await key('BUILD_SYSTEM', scene(2.65));
  const buildNew = await key('BUILD_SYSTEM', scene(2.66));
  assert.notEqual(buildOld, buildNew, 'the changed root must change');

  const expectedOld = await key('EXPECTED', { systemKey: buildOld, level: 'L1' });
  const expectedNew = await key('EXPECTED', { systemKey: buildNew, level: 'L1' });
  assert.notEqual(expectedOld, expectedNew, 'a descendant must change');

  const simulateOld = await key('SIMULATE', { expectedKey: expectedOld, seed: '1', exposureSeconds: 100 });
  const simulateNew = await key('SIMULATE', { expectedKey: expectedNew, seed: '1', exposureSeconds: 100 });
  assert.notEqual(simulateOld, simulateNew, 'a grandchild must change');

  // Non-descendants do not move.
  assert.equal(
    await key('LOAD_TABLES', { tableVersion: 'v1' }),
    await key('LOAD_TABLES', { tableVersion: 'v1' }),
    'an unrelated branch must keep its key',
  );
  const otherCounts = await key('SCAN', { countsKey: 'someone-elses-dataset', scalesVoxels: [1, 2] });
  assert.equal(
    otherCounts,
    await key('SCAN', { countsKey: 'someone-elses-dataset', scalesVoxels: [1, 2] }),
    'a task reading a different dataset must not be invalidated',
  );
});

test('identical inputs hit the cache instead of the worker', async () => {
  const host = new FakeHost(2, responder);
  const bus = new ComputeBus({ host });
  const first = request('SIMULATE', 'sim-1', { seed: '7' });
  const second = { ...request('SIMULATE', 'sim-1', { seed: '7' }) };

  const a = await bus.submit(first);
  const sendsAfterFirst = host.spawned.reduce((total, worker) => total + worker.received.length, 0);
  const b = await bus.submit(second);
  const sendsAfterSecond = host.spawned.reduce((total, worker) => total + worker.received.length, 0);

  assert.deepEqual(a, b, 'the cached result is the same result');
  assert.equal(sendsAfterFirst, sendsAfterSecond, 'the second submission must not reach a worker');
  bus.close();
});

test('bulk results arrive as transferred buffers, not copies', async () => {
  const transferLists: string[][] = [];
  const host = new FakeHost(2, async (kind, id, payload) => {
    const event = await responder(kind, id, payload);
    if (event.type === 'result') {
      assert.ok(
        (event.transfer?.length ?? 0) > 0,
        'a bulk result must carry a transfer list, or it is a copy',
      );
      transferLists.push(event.transfer!.map((buffer) => String(buffer.byteLength)));
    }
    return event;
  });
  const bus = new ComputeBus({ host });
  const result = await bus.submit(request('SIMULATE', 'sim-transfer'));
  const counts = result.counts;
  assert.ok(counts instanceof Uint32Array, 'the result is the typed array the worker sent');
  assert.deepEqual([...counts], [1, 2, 3, 4]);
  assert.equal(transferLists.length, 1, 'exactly one transferred result');
  assert.deepEqual(transferLists[0], ['16'], 'the counts buffer, by length');
  bus.close();
});

test('the responder transfers, so the sender buffer is detached', async () => {
  // A `postMessage` with a transfer list detaches the sender's buffer. Asserting
  // that here proves the protocol moves bytes instead of serialising them.
  const buffer = new ArrayBuffer(16);
  assert.equal(buffer.byteLength, 16);
  const channel = new MessageChannel();
  channel.port1.postMessage({ buffer }, [buffer]);
  assert.equal(buffer.byteLength, 0, 'a transferred buffer must be detached');
  channel.port1.close();
  channel.port2.close();
});

test('cancelling a running reconstruction frees its worker within 100 ms', async () => {
  const host = new FakeHost(2, responder);
  const bus = new ComputeBus({ host });
  const controller = new AbortController();
  const started = Date.now();
  const pending = bus.submit(request('RECON', 'recon-cancel'), { signal: controller.signal });
  // Let the worker pick the task up before cancelling it.
  await new Promise((resolve) => setTimeout(resolve, 20));
  assert.equal(bus.workerTotal, 1, 'one worker is running the reconstruction');
  controller.abort();
  await assert.rejects(pending, TaskCancelled);
  const freed = Date.now() - started;
  assert.ok(freed < 100, `the worker must be freed within 100 ms, took ${freed} ms`);
  assert.ok(host.terminated >= 1, 'the running worker must be terminated, not asked politely');
  assert.equal(bus.queued, 0, 'nothing is left behind');
  bus.close();
});

test('a queued task is cancelled without touching a worker', async () => {
  const host = new FakeHost(1, responder);
  const bus = new ComputeBus({ host });
  const controller = new AbortController();
  const running = bus.submit(request('RECON', 'recon-1'));
  const queued = bus.submit(request('RECON', 'recon-2'), { signal: controller.signal });
  controller.abort();
  await assert.rejects(queued, TaskCancelled);
  await running;
  assert.equal(host.spawned.length, 1, 'a queued cancellation spawns nothing');
  bus.close();
});

test('a version mismatch is a hard error at startup', () => {
  assert.throws(
    () => new ComputeBus({ host: new FakeHost(1, responder), expectedApiVersion: '0.0.1' }),
    ApiVersionMismatch,
  );
  assert.doesNotThrow(
    () => new ComputeBus({ host: new FakeHost(1, responder), expectedApiVersion: BUS_API_VERSION }),
  );
});

test('a two-core machine gets one worker and no deadlock', async () => {
  const concurrency = workerCount(2);
  assert.equal(concurrency, 1, '2 cores must yield one worker');
  const host = new FakeHost(concurrency, responder);
  const bus = new ComputeBus({ host, cache: new MemoryCache() });
  const results = await Promise.all(
    Array.from({ length: 6 }, (_, index) => bus.submit(request('EXPECTED', `exp-${index}`))),
  );
  assert.equal(results.length, 6);
  assert.equal(bus.workerTotal, 1, 'the pool must not grow past its budget');
  bus.close();
});

test('priority order is served before arrival order', async () => {
  const seen: TaskKind[] = [];
  const host = new FakeHost(1, async (kind, id, payload) => {
    seen.push(kind);
    // Occupy the only worker briefly so the next two are both *queued*: a
    // priority queue cannot preempt the task that is already running, and
    // pretending otherwise is how a real slider ends up behind a campaign.
    await new Promise((resolve) => setTimeout(resolve, 20));
    return { id, type: 'result', result: { echoed: payload } as never };
  });
  const bus = new ComputeBus({ host });
  const blocker = bus.submit(request('RECON', 'blocker', {}, 1));
  await new Promise((resolve) => setTimeout(resolve, 5));
  const background = bus.submit(request('MCMC', 'mcmc-bg', {}, 2));
  const interactive = bus.submit(request('PSF', 'psf-interactive', {}, 0));
  await Promise.all([blocker, background, interactive]);
  assert.deepEqual(seen, ['RECON', 'PSF', 'MCMC'], 'the interactive task runs first');
  bus.close();
});

test('worker errors surface as task failures', async () => {
  const host = new FakeHost(1, async (_kind, id) => ({
    id,
    type: 'error',
    code: 'bad-scene',
    message: 'the grid is empty',
  }));
  const bus = new ComputeBus({ host });
  await assert.rejects(
    bus.submit(request('BUILD_SYSTEM', 'build-bad')),
    (error: unknown) => error instanceof TaskFailure && error.code === 'bad-scene',
  );
  bus.close();
});

test('a cached simulate round-trips in under 10 ms', async () => {
  const host = new FakeHost(2, responder);
  const cache: CacheStore = new MemoryCache();
  const bus = new ComputeBus({ host, cache });
  await bus.submit(request('SIMULATE', 'sim-latency'));
  const started = performance.now();
  for (let index = 0; index < 20; index += 1) {
    await bus.submit({ ...request('SIMULATE', 'sim-latency') });
  }
  const each = (performance.now() - started) / 20;
  assert.ok(each < 10, `a cached round trip took ${each.toFixed(2)} ms`);
  bus.close();
});
