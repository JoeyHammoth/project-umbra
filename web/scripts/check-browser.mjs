#!/usr/bin/env node
/**
 * Browser proof for the compute boundary.
 *
 * Loads the built app in headless Chrome, waits for `BUILD_SYSTEM` to come back
 * from the worker, and fails on anything else. `chrome --dump-dom` cannot do
 * this: it snapshots the page once timers run out, which is before a module
 * worker has fetched and instantiated the `.wasm`.
 *
 * Usage:
 *   npm run build && npx vite preview --port 4174 &
 *   CHROME_PATH="/c/Program Files/Google/Chrome/Application/chrome.exe" \
 *     node scripts/check-browser.mjs http://localhost:4174/project-umbra/
 */

import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

const url = process.argv[2] ?? 'http://localhost:4174/project-umbra/';
const chromePath = process.env.CHROME_PATH ?? 'chrome';
const port = Number(process.env.CDP_PORT ?? 9222);
const profile = mkdtempSync(path.join(tmpdir(), 'umbra-chrome-'));
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// Console output from the *worker* never reaches the page's CDP session, so
// Chrome's own stderr log is the only place a worker exception surfaces.
const logLines = [];
const chrome = spawn(
  chromePath,
  [
    '--headless=new',
    '--disable-gpu',
    '--no-first-run',
    '--no-default-browser-check',
    '--enable-logging=stderr',
    '--v=1',
    `--remote-debugging-port=${port}`,
    `--user-data-dir=${profile}`,
    'about:blank',
  ],
  { stdio: ['ignore', 'ignore', 'pipe'] },
);
chrome.stderr.on('data', (chunk) => {
  for (const line of String(chunk).split('\n')) {
    if (/CONSOLE|Uncaught|TypeError/.test(line)) logLines.push(line.trim());
  }
});

/** Console output from the page and from Chrome itself, for a failed run. */
const diagnostics = (client) =>
  [...client.consoleLines, ...logLines].map((line) => `  console: ${line}`).join('\n') ||
  '  (no console output)';

const cleanup = () => {
  chrome.kill();
  try {
    rmSync(profile, { recursive: true, force: true });
  } catch {
    // Chrome may still hold a file; the temp directory is not worth retrying.
  }
};

/** The page target Chrome opens at startup. */
async function pageTarget() {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    try {
      const response = await fetch(`http://127.0.0.1:${port}/json/list`);
      const targets = await response.json();
      const page = targets.find((target) => target.type === 'page');
      if (page) return page;
    } catch {
      // Chrome is not listening yet.
    }
    await sleep(200);
  }
  throw new Error('Chrome never exposed a page target');
}

/** A minimal CDP client over Node's built-in WebSocket. */
function connect(wsUrl) {
  const socket = new WebSocket(wsUrl);
  const pending = new Map();
  const consoleLines = [];
  let nextId = 0;

  socket.addEventListener('message', (event) => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
      pending.get(message.id)(message);
      pending.delete(message.id);
      return;
    }
    if (message.method === 'Runtime.consoleAPICalled' && message.params.type === 'error') {
      consoleLines.push(
        message.params.args
          .map((arg) => arg.value ?? arg.description ?? '')
          .join(' '),
      );
    }
    if (message.method === 'Runtime.exceptionThrown') {
      consoleLines.push(message.params.exceptionDetails.text);
    }
  });

  const send = (method, params = {}) =>
    new Promise((resolve) => {
      const id = (nextId += 1);
      pending.set(id, resolve);
      socket.send(JSON.stringify({ id, method, params }));
    });

  const open = new Promise((resolve, reject) => {
    socket.addEventListener('open', resolve);
    socket.addEventListener('error', () => reject(new Error('CDP socket failed')));
  });

  return { send, open, consoleLines, close: () => socket.close() };
}

async function main() {
  const target = await pageTarget();
  const client = connect(target.webSocketDebuggerUrl);
  await client.open;
  await client.send('Runtime.enable');
  await client.send('Page.enable');
  await client.send('Page.navigate', { url });

  const expression =
    "document.querySelector('[data-testid=\"build-status\"]')?.textContent ?? ''";
  let text = '';
  for (let attempt = 0; attempt < 240; attempt += 1) {
    const reply = await client.send('Runtime.evaluate', { expression, returnByValue: true });
    text = reply.result?.result?.value ?? '';
    if (text && !text.includes('Assembling')) break;
    await sleep(250);
  }

  if (!text || text.includes('Assembling')) {
    throw new Error(
      `the build never finished (last status: ${JSON.stringify(text)})\n` +
        diagnostics(client),
    );
  }
  // The whole boundary, not just the first stage: `Float32Array` and
  // `Uint32Array` in the status line are λ and the counts as the main thread
  // received them, after the worker transferred their buffers.
  if (
    !/^Assembled \d+ rays × \d+ voxels, \d+ non-zero path lengths · λ Float32Array\(\d+\) · counts Uint32Array\(\d+\), \d+ events/.test(
      text,
    )
  ) {
    throw new Error(`the build reported a failure: ${text}\n` + diagnostics(client));
  }
  // The store is only real if the browser's own IndexedDB holds the result:
  // the unit tests run against a fake, this runs against the engine.
  const stored = await client.send('Runtime.evaluate', {
    expression: `new Promise((resolve, reject) => {
      const open = indexedDB.open('umbra-compute');
      open.onerror = () => reject(open.error);
      open.onsuccess = () => {
        const request = open.result
          .transaction('results', 'readonly')
          .objectStore('results')
          .count();
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
      };
    })`,
    awaitPromise: true,
    returnByValue: true,
  });
  const records = stored.result?.result?.value;
  if (typeof records !== 'number' || records < 1) {
    throw new Error(
      `the browser's IndexedDB holds no cached result (${JSON.stringify(stored.result)})`,
    );
  }

  client.close();
  return `${text} [indexeddb records: ${records}]`;
}

try {
  const text = await main();
  console.log(`browser round trip: ${text}`);
} catch (error) {
  console.error(`browser round trip FAILED: ${error.message}`);
  process.exitCode = 1;
} finally {
  cleanup();
}
