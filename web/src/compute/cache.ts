/**
 * The IndexedDB `CacheStore`: results that survive a reload.
 *
 * The bus is already content-keyed, so the cache is one map from a key to a
 * result. Storing it in IndexedDB rather than memory is what makes a rebuilt
 * page skip a 30-second operator assembly, and it is the store spec §4.4 asks
 * for.
 *
 * A result is stored by structured clone, which is the same algorithm the
 * worker boundary uses, so a transferred `Float32Array` round-trips as a typed
 * array and not as a plain object.
 *
 * # Failure policy
 *
 * A cache is an optimisation. A browser that refuses storage — private mode, a
 * revoked quota, a database held open by another tab — must degrade to
 * recomputation, not turn a working pipeline into an error. `get` therefore
 * returns `undefined` when the store cannot be read, and `set` resolves when it
 * cannot be written. The cost of a wrong guess here is a slow page, never a
 * wrong number.
 *
 * # Budget
 *
 * Storage is finite and the browser does not evict on our behalf: a page that
 * computes every scene the user tries would grow until a write failed. The store
 * therefore keeps at most `maxEntries` entries and `maxBytes` of estimated
 * payload, dropping the oldest writes first. A quota failure that still happens
 * clears the store and retries once, because the result the caller is waiting
 * for is worth more than the ones already cached.
 */

import type { CacheStore, CachedResult } from './bus.ts';

const DB_NAME = 'umbra-compute';
const STORE = 'results';
/**
 * v3 because a result gained a field: `LEE_CALIB` now carries `meanEulerAtU0`,
 * and a v2 record would come back without it and be drawn as if the anchor had
 * never fired. v2 records carry their own key, age and size; v1 stored the bare
 * result.
 */
const VERSION = 3;

/** How much the cache may hold before the oldest entries are dropped. */
export interface CacheBudget {
  /** Entries kept at most. Defaults to 16. */
  maxEntries?: number;
  /** Estimated payload bytes kept at most. Defaults to 64 MiB. */
  maxBytes?: number;
}

const DEFAULT_MAX_ENTRIES = 16;
const DEFAULT_MAX_BYTES = 64 * 1024 * 1024;

interface Stored {
  key: string;
  value: CachedResult;
  storedAt: number;
  bytes: number;
}

/**
 * Approximate structured-clone size of a value, in bytes.
 *
 * A budget needs a size and the platform exposes none: what a record costs is
 * its structured clone, which no browser reports. Typed arrays are counted
 * exactly — their payload dominates every real result — and numbers, strings and
 * container overhead are the obvious estimates. A value this does not model (a
 * `Date`, a `Map`) is charged its fixed overhead, which understates it rather
 * than failing.
 */
function byteLengthOf(value: unknown): number {
  if (value === null || value === undefined) return 8;
  if (typeof value === 'number' || typeof value === 'boolean') return 8;
  if (typeof value === 'string') return value.length * 2;
  if (ArrayBuffer.isView(value)) return value.byteLength + 16;
  if (value instanceof ArrayBuffer) return value.byteLength + 16;
  if (Array.isArray(value)) {
    return value.reduce<number>((total, entry) => total + byteLengthOf(entry), 16);
  }
  if (typeof value === 'object') {
    return Object.entries(value as Record<string, unknown>).reduce<number>(
      (total, [key, entry]) => total + key.length * 2 + byteLengthOf(entry),
      16,
    );
  }
  return 8;
}

export class IdbCache implements CacheStore {
  private readonly name: string;
  private readonly factory: IDBFactory | undefined;
  private readonly maxEntries: number;
  private readonly maxBytes: number;
  private database: Promise<IDBDatabase> | null = null;
  /** Monotonic write clock, so two writes in the same millisecond still order. */
  private clock = 0;

  /**
   * @param name Database name; distinct names are distinct caches.
   * @param factory The `indexedDB` global, injectable so a test can supply one.
   * @param budget How much to keep; the defaults are the browser's share of a
   *   page's storage, not of the device's disk.
   */
  constructor(name: string = DB_NAME, factory?: IDBFactory, budget: CacheBudget = {}) {
    this.name = name;
    this.factory = factory;
    this.maxEntries = Math.max(1, budget.maxEntries ?? DEFAULT_MAX_ENTRIES);
    this.maxBytes = Math.max(0, budget.maxBytes ?? DEFAULT_MAX_BYTES);
  }

  async get(key: string): Promise<CachedResult | undefined> {
    try {
      const stored = (await this.request('readonly', (store) => store.get(key))) as
        | Stored
        | undefined;
      // A record from a store this version cannot read is a miss, not a hit
      // with an undefined payload.
      return stored && 'value' in stored ? stored.value : undefined;
    } catch {
      return undefined;
    }
  }

  async set(key: string, result: CachedResult): Promise<void> {
    const record: Stored = {
      key,
      value: result,
      storedAt: this.stamp(),
      bytes: byteLengthOf(result),
    };
    try {
      await this.put(record);
    } catch {
      // A full quota surfaces here first. The cache is disposable and the caller
      // is waiting on this result, so the store is cleared and the write retried
      // once; if that fails too, the task simply was not cached.
      try {
        await this.request('readwrite', (store) => store.clear());
        await this.put(record);
      } catch {
        return;
      }
    }
    try {
      await this.evict();
    } catch {
      // Size accounting is best effort: a record that was written stays written
      // even if the trim could not run.
    }
  }

  /** Closes the connection; a later call reopens it. */
  async close(): Promise<void> {
    const open = this.database;
    this.database = null;
    if (open) {
      (await open.catch(() => null))?.close();
    }
  }

  private stamp(): number {
    this.clock = Math.max(Date.now(), this.clock + 1);
    return this.clock;
  }

  private put(record: Stored): Promise<unknown> {
    return this.request('readwrite', (store) => store.put(record, record.key));
  }

  /**
   * Drops the oldest writes until the store fits its budget.
   *
   * Reads deliberately do not refresh an entry. Rewriting a record to move its
   * timestamp would copy a 60 MB result on the hot path of a *hit*, which costs
   * more than the eviction it would avoid; the keys are content hashes, so the
   * entries that are cheap to lose are also the ones no page is asking for.
   */
  private async evict(): Promise<void> {
    const records = (await this.request('readonly', (store) => store.getAll())) as Stored[];
    let bytes = records.reduce((total, record) => total + record.bytes, 0);
    if (records.length <= this.maxEntries && bytes <= this.maxBytes) return;
    const oldest = [...records].sort((one, two) => one.storedAt - two.storedAt);
    const victims: string[] = [];
    let remaining = records.length;
    for (const record of oldest) {
      if (remaining <= this.maxEntries && bytes <= this.maxBytes) break;
      victims.push(record.key);
      remaining -= 1;
      bytes -= record.bytes;
    }
    await this.deleteMany(victims);
  }

  private async deleteMany(keys: string[]): Promise<void> {
    if (keys.length === 0) return;
    const db = await this.open();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE, 'readwrite');
      const store = tx.objectStore(STORE);
      for (const key of keys) store.delete(key);
      tx.oncomplete = () => resolve();
      tx.onabort = () => reject(tx.error ?? new Error('indexedDB transaction aborted'));
      tx.onerror = () => reject(tx.error ?? new Error('indexedDB delete failed'));
    });
  }

  private open(): Promise<IDBDatabase> {
    if (this.database) return this.database;
    const factory = this.factory ?? globalThis.indexedDB;
    if (!factory) return Promise.reject(new Error('indexedDB is unavailable'));
    this.database = new Promise<IDBDatabase>((resolve, reject) => {
      const request = factory.open(this.name, VERSION);
      request.onupgradeneeded = () => {
        const db = request.result;
        // v1 held bare results; every record this version writes carries its own
        // key, age and size. The cache is disposable, so upgrading recreates the
        // store rather than reading records it cannot account for.
        if (db.objectStoreNames.contains(STORE)) db.deleteObjectStore(STORE);
        db.createObjectStore(STORE);
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error ?? new Error('indexedDB open failed'));
      request.onblocked = () =>
        reject(new Error('indexedDB open blocked by another connection'));
    });
    return this.database;
  }

  /** Runs one request in its own transaction and resolves with its result. */
  private async request<T>(
    mode: IDBTransactionMode,
    op: (store: IDBObjectStore) => IDBRequest<T>,
  ): Promise<T> {
    const db = await this.open();
    return new Promise<T>((resolve, reject) => {
      let request: IDBRequest<T>;
      try {
        const tx = db.transaction(STORE, mode);
        request = op(tx.objectStore(STORE));
        tx.onabort = () => reject(tx.error ?? new Error('indexedDB transaction aborted'));
      } catch (error) {
        // A connection closed under us is not recoverable in place; forget it so
        // the next call reopens.
        this.database = null;
        reject(error);
        return;
      }
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error ?? new Error('indexedDB request failed'));
    });
  }
}
