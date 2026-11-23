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
 */

import type { CacheStore, CachedResult } from './bus.ts';

const DB_NAME = 'umbra-compute';
const STORE = 'results';
const VERSION = 1;

export class IdbCache implements CacheStore {
  private readonly name: string;
  private readonly factory: IDBFactory | undefined;
  private database: Promise<IDBDatabase> | null = null;

  /**
   * @param name Database name; distinct names are distinct caches.
   * @param factory The `indexedDB` global, injectable so a test can supply one.
   */
  constructor(name: string = DB_NAME, factory?: IDBFactory) {
    this.name = name;
    this.factory = factory;
  }

  async get(key: string): Promise<CachedResult | undefined> {
    try {
      const stored = await this.request('readonly', (store) => store.get(key));
      return stored as CachedResult | undefined;
    } catch {
      return undefined;
    }
  }

  async set(key: string, result: CachedResult): Promise<void> {
    try {
      await this.request('readwrite', (store) => store.put(result, key));
    } catch {
      // Quota exceeded, or storage disabled mid-session: keep computing.
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

  private open(): Promise<IDBDatabase> {
    if (this.database) return this.database;
    const factory = this.factory ?? globalThis.indexedDB;
    if (!factory) return Promise.reject(new Error('indexedDB is unavailable'));
    this.database = new Promise<IDBDatabase>((resolve, reject) => {
      const request = factory.open(this.name, VERSION);
      request.onupgradeneeded = () => {
        const db = request.result;
        if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE);
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
