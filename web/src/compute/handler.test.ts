import assert from 'node:assert/strict';
import { test } from 'node:test';

import { coreFault, handleTask } from './handler.ts';
import type { ComputeCore } from './handler.ts';
import type { PayloadOf, TaskEvent, TaskKind, TaskRequest } from './protocol.ts';

const request = <K extends TaskKind>(
  kind: K,
  payload: PayloadOf<K>,
  key = 'content-key',
): TaskRequest<K> => ({ id: 'task-1', kind, key, priority: 0, payload });

type ResultEvent<T> = { result: T; transfer?: ArrayBuffer[] };
type ErrorEvent = { code: string; message: string };

const asResult = <T>(event: TaskEvent): ResultEvent<T> => {
  assert.equal(event.type, 'result', `expected a result, got ${event.type}`);
  return event as unknown as ResultEvent<T>;
};

const asError = (event: TaskEvent): ErrorEvent => {
  assert.equal(event.type, 'error', `expected an error, got ${event.type}`);
  return event as unknown as ErrorEvent;
};

/** Records what it was asked for, so the handler's routing is observable. */
class FakeCore implements ComputeCore {
  readonly calls: string[] = [];
  lambda = new Float32Array([1.5, 2.5, 3.5]);
  counts = new Uint32Array([7, 8, 9]);
  density = new Float32Array([2.65, 2.65]);
  failure: unknown = null;

  buildSystem(systemKey: string, sceneJson: string) {
    this.throwIfFailing();
    this.calls.push(`build ${systemKey} ${sceneJson}`);
    return { systemKey, rows: 4, cols: 9, nnz: 12 };
  }

  sceneDensity(systemKey: string): Float32Array {
    this.throwIfFailing();
    this.calls.push(`scene-density ${systemKey}`);
    return this.density;
  }

  expected(
    systemKey: string,
    expectedKey: string,
    density: Float32Array,
    level?: string,
  ) {
    this.throwIfFailing();
    this.calls.push(`expected ${systemKey} ${expectedKey} ${density.length} ${level ?? '-'}`);
    return this.lambda;
  }

  simulate(expectedKey: string, seed: bigint, exposureSeconds: number) {
    this.throwIfFailing();
    this.calls.push(`simulate ${expectedKey} ${seed} ${exposureSeconds}`);
    return this.counts;
  }

  private throwIfFailing() {
    if (this.failure !== null) throw this.failure;
  }
}

test('build system returns the operator shape and has nothing to transfer', () => {
  const core = new FakeCore();
  const event = handleTask(core, request('BUILD_SYSTEM', { sceneJson: '{"id":"x"}' }));

  const { result, transfer } = asResult<{ systemKey: string; rows: number; cols: number; nnz: number }>(
    event,
  );
  assert.deepEqual(result, { systemKey: 'content-key', rows: 4, cols: 9, nnz: 12 });
  assert.equal(transfer, undefined, 'no bulk data, so no transfer list');
  assert.deepEqual(
    Object.keys(result).sort(),
    ['cols', 'nnz', 'rows', 'systemKey'],
    'a plain object: a wasm class posts as {} because its fields are prototype getters',
  );
  assert.deepEqual(core.calls, ['build content-key {"id":"x"}']);
});

test('expected hands back the very buffer the worker allocated', () => {
  const core = new FakeCore();
  const event = handleTask(
    core,
    request('EXPECTED', {
      systemKey: 'sys',
      density: new Float32Array([2.65, 2.65]),
      level: 'L1',
    }),
  );

  const { result, transfer } = asResult<{ lambda: Float32Array }>(event);
  assert.equal(result.lambda, core.lambda, 'the array is not rebuilt');
  assert.deepEqual([...result.lambda], [1.5, 2.5, 3.5]);
  assert.equal(transfer?.length, 1, 'exactly one buffer moves');
  assert.equal(transfer?.[0], result.lambda.buffer, 'the transfer list names that buffer');
  assert.deepEqual(core.calls, ['expected sys content-key 2 L1']);
});

test('expected without a density evaluates the scene the worker already built', () => {
  const core = new FakeCore();
  const event = handleTask(core, request('EXPECTED', { systemKey: 'sys', level: 'L1' }));

  const { result, transfer } = asResult<{ lambda: Float32Array }>(event);
  assert.equal(result.lambda, core.lambda);
  assert.equal(transfer?.[0], result.lambda.buffer);
  assert.deepEqual(
    core.calls,
    ['scene-density sys', 'expected sys content-key 2 L1'],
    'the scene field is fetched, then evaluated',
  );
});

test('a density view that is not a Float32Array is reinterpreted, not copied', () => {
  const core = new FakeCore();
  const bytes = new Float32Array([2.65, 1.5]).buffer;
  handleTask(
    core,
    request('EXPECTED', {
      systemKey: 'sys',
      density: new DataView(bytes),
      level: 'L1',
    }),
  );
  assert.deepEqual(core.calls, ['expected sys content-key 2 L1']);
});

test('simulate transfers its counts and passes the seed as a BigInt', () => {
  const core = new FakeCore();
  const event = handleTask(
    core,
    request('SIMULATE', { expectedKey: 'exp', seed: '18446744073709551615', exposureSeconds: 60 }),
  );

  const { result, transfer } = asResult<{ counts: Uint32Array }>(event);
  assert.equal(result.counts, core.counts);
  assert.equal(transfer?.[0], result.counts.buffer);
  assert.deepEqual(core.calls, ['simulate exp 18446744073709551615 60']);
});

test('a task that lost its content key is a coded error, not a TypeError', () => {
  const core = new FakeCore();
  const event = handleTask(core, { ...request('BUILD_SYSTEM', { sceneJson: '{}' }), key: '' });

  const { code, message } = asError(event);
  assert.equal(code, 'request');
  assert.match(message, /content key/);
  assert.deepEqual(core.calls, [], 'nothing reaches the core unnamed');
});

test('a task with no entry point is an error event, not a thrown exception', () => {
  const core = new FakeCore();
  const event = handleTask(
    core,
    request('RECON', {
      countsKey: 'counts',
      solver: 'a1-transmission-em',
      params: {},
      iterations: 1,
    }),
  );

  const { code, message } = asError(event);
  assert.equal(code, 'unimplemented');
  assert.match(message, /RECON/);
  assert.deepEqual(core.calls, [], 'the core is never called');
});

test('a core fault becomes a coded error event with its message intact', () => {
  const core = new FakeCore();
  core.failure = new Error('density: density has 3 values but the grid has 1024 voxels');
  const event = handleTask(
    core,
    request('EXPECTED', {
      systemKey: 'sys',
      density: new Float32Array([1, 2, 3]),
      level: 'L1',
    }),
  );

  const { code, message } = asError(event);
  assert.equal(code, 'density');
  assert.equal(message, 'density has 3 values but the grid has 1024 voxels');
});

test('a message without a code prefix keeps its wording under a generic code', () => {
  const core = new FakeCore();
  core.failure = new Error('cannot read properties of undefined (reading buffer)');
  const event = handleTask(
    core,
    request('SIMULATE', { expectedKey: 'exp', seed: '1', exposureSeconds: 1 }),
  );

  const { code, message } = asError(event);
  assert.equal(code, 'compute');
  assert.equal(message, 'cannot read properties of undefined (reading buffer)');
});

test('a message that merely contains a colon is not mistaken for a code', () => {
  const fault = coreFault(new Error('scene.detectors[0].bins: need at least one bin'));
  assert.equal(fault.code, 'compute');
  assert.equal(fault.message, 'scene.detectors[0].bins: need at least one bin');

  const coded = coreFault(new Error('scene: scene.units.density: only `g/cm3` is supported'));
  assert.equal(coded.code, 'scene');
  assert.equal(coded.message, 'scene.units.density: only `g/cm3` is supported');
});
