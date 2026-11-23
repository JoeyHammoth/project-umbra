/**
 * The browser `WorkerHost`: real `Worker`s running `worker.ts`.
 *
 * The bus owns scheduling, cancellation and the cache; this file only knows how
 * to make a thread. Splitting it here is what lets `bus.ts` run under
 * `node --test` with a fake worker.
 */

import { workerCount } from './bus.ts';
import type { WorkerHandle, WorkerHost } from './bus.ts';
import type { TaskEvent } from './protocol.ts';

/** One `Worker`, adapted to the handle the bus holds. */
class BrowserWorker implements WorkerHandle {
  onmessage: ((event: { data: TaskEvent }) => void) | null = null;
  onerror: ((event: unknown) => void) | null = null;

  private readonly thread: Worker;

  constructor(thread: Worker) {
    this.thread = thread;
    // The bus assigns these after construction, so the forwarding closures read
    // the current field rather than capturing a handler that does not exist yet.
    thread.onmessage = (event: MessageEvent) => this.onmessage?.({ data: event.data as TaskEvent });
    thread.onerror = (event: ErrorEvent) => this.onerror?.(event);
  }

  postMessage(message: unknown, transfer?: Transferable[]): void {
    this.thread.postMessage(message, transfer ?? []);
  }

  terminate(): void {
    this.thread.terminate();
  }
}

/**
 * A host that spawns module workers.
 *
 * `hardwareConcurrency` is left as an injectable number so a caller (or a test
 * of the policy) can pin it; the default is the machine's own count minus one,
 * because the main thread is doing the rendering.
 */
export function createBrowserHost(hardwareConcurrency?: number): WorkerHost {
  const cores =
    hardwareConcurrency ??
    (typeof navigator === 'undefined' ? 2 : (navigator.hardwareConcurrency ?? 2));
  return {
    concurrency: workerCount(cores),
    spawn() {
      const thread = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
      return new BrowserWorker(thread);
    },
  };
}
