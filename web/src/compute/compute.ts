/**
 * The browser's compute entry point: one bus, real workers, a persistent cache.
 *
 * Callers get a `ComputeBus` and never learn whether the cache is IndexedDB or
 * memory — only that a result they compute once is cheap the second time.
 */

import { ComputeBus, MemoryCache } from './bus.ts';
import { IdbCache } from './cache.ts';
import { createBrowserHost } from './workers.ts';

/** Whether results will survive this page, for the UI to state plainly. */
export type CacheKind = 'indexeddb' | 'memory';

export interface BrowserCompute {
  bus: ComputeBus;
  cache: CacheKind;
}

/**
 * The app's compute bus.
 *
 * The WASM module's own handshake happens inside the worker, which refuses
 * tasks with an `api_version` fault rather than computing under a contract it
 * does not speak; `BUS_API_VERSION` is checked by the bus constructor.
 */
export async function createBrowserCompute(): Promise<BrowserCompute> {
  const persistent = typeof indexedDB !== 'undefined';
  const bus = new ComputeBus({
    host: createBrowserHost(),
    cache: persistent ? new IdbCache() : new MemoryCache(),
  });
  return { bus, cache: persistent ? 'indexeddb' : 'memory' };
}
