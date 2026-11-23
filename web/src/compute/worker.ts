/**
 * The compute worker: loads the WASM core, checks the handshake, then runs one
 * task per message.
 *
 * This file is the only place the generated bindings are imported. Everything
 * else talks to `ComputeCore` through the interface in `handler.ts`, which is
 * also what keeps the task contract testable under `node --test`.
 */

import init, { ComputeCore, api_version } from '../wasm/umbra_wasm.js';

import { BUS_API_VERSION } from './bus.ts';
import { handleTask } from './handler.ts';
import type { TaskEvent, TaskRequest } from './protocol.ts';

// A dedicated worker's global scope. Declared narrowly rather than pulling in
// the `webworker` lib, which conflicts with the DOM lib the rest of the app
// compiles against.
declare const self: {
  postMessage(message: TaskEvent, transfer?: Transferable[]): void;
  onmessage: ((event: MessageEvent<TaskRequest>) => void) | null;
};

/**
 * The core, or the reason there is none.
 *
 * Boot is memoised so N queued tasks pay for one module fetch, and a failure
 * stays a failure: retrying `init()` after a version mismatch would just fail
 * again, so the fault is cached alongside the promise.
 */
let core: Promise<ComputeCore> | null = null;

function boot(): Promise<ComputeCore> {
  core ??= (async () => {
    await init();
    const actual = api_version();
    if (actual !== BUS_API_VERSION) {
      // A stale cached module is always fatal. Computing physics under a
      // mismatched contract would silently produce the wrong answer, which is
      // worse than refusing.
      throw new Error(
        `api_version: the client expects ${BUS_API_VERSION} but this module speaks ${actual}`,
      );
    }
    return new ComputeCore();
  })();
  return core;
}

self.onmessage = async (event: MessageEvent<TaskRequest>) => {
  const request = event.data;
  const id = request?.id ?? 'unknown';
  let reply: TaskEvent;
  try {
    reply = handleTask(await boot(), request);
  } catch (error) {
    const text = error instanceof Error ? error.message : String(error);
    const [head, ...rest] = text.split(': ');
    const coded = rest.length > 0 && /^[a-z][a-z_]*$/.test(head ?? '');
    reply = {
      id,
      type: 'error',
      // A module that will not load is not a compute fault: the UI can only
      // offer a reload, and the code is what tells it that.
      code: coded ? head! : 'module',
      message: coded ? rest.join(': ') : text,
    };
  }
  // Results carry their transfer list from the handler; the buffers were
  // allocated by the core, so naming them here is what makes the move real.
  self.postMessage(reply, reply.type === 'result' ? reply.transfer : undefined);
};
