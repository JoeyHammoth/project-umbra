import 'fake-indexeddb/auto';

import assert from 'node:assert/strict';
import { test } from 'node:test';

import type { CachedResult } from './bus.ts';
import { IdbCache } from './cache.ts';

/** A fresh database name per test: IndexedDB is global and shared across files. */
const name = () => `umbra-cache-test-${crypto.randomUUID()}`;

test('a result survives a new connection, which is what a reload is', async () => {
  const db = name();
  const lambda = new Float32Array([1.5, 2.5, 3.5]);
  const result: CachedResult = { ok: true, result: { lambda } };

  const first = new IdbCache(db);
  await first.set('EXPECTED/v1', result);
  await first.close();

  // A new instance stands in for the page after a reload: nothing but the
  // database itself is carried over.
  const second = new IdbCache(db);
  const cached = await second.get('EXPECTED/v1');
  assert.ok(cached, 'the entry must be there after reconnecting');
  const stored = (cached.result as { lambda: Float32Array }).lambda;
  assert.ok(stored instanceof Float32Array, 'structured clone keeps the type');
  assert.deepEqual([...stored], [1.5, 2.5, 3.5]);
  await second.close();
});

test('a key that was never written is a miss, not an error', async () => {
  const cache = new IdbCache(name());
  assert.equal(await cache.get('BUILD_SYSTEM/nope'), undefined);
  await cache.close();
});

test('writing a key twice keeps the later result', async () => {
  const cache = new IdbCache(name());
  await cache.set('k', { ok: true, result: { nnz: 1 } });
  await cache.set('k', { ok: true, result: { nnz: 2 } });
  assert.deepEqual((await cache.get('k'))?.result, { nnz: 2 });
  await cache.close();
});

test('separate database names do not see each other', async () => {
  const one = new IdbCache(name());
  const two = new IdbCache(name());
  await one.set('shared-key', { ok: true, result: { from: 1 } });
  assert.equal(await two.get('shared-key'), undefined);
  await one.close();
  await two.close();
});

test('a missing store degrades to a miss instead of failing the task', async () => {
  // The factory throwing is what private-mode browsers do; the bus must still
  // compute, so the cache refuses quietly.
  const broken = {
    open() {
      throw new Error('storage is disabled in this context');
    },
  } as unknown as IDBFactory;
  const cache = new IdbCache(name(), broken);
  assert.equal(await cache.get('k'), undefined);
  await cache.set('k', { ok: true, result: null });
});
