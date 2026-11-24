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
 *   npm run build && npm run check:browser
 *   CHROME_PATH=/path/to/chrome node scripts/check-browser.mjs [url]
 *
 * With a `url` argument no server is started: the check points at one already
 * running. Without it the script serves `dist/` itself, so CI needs one step
 * and no background process.
 */

import { spawn, spawnSync } from 'node:child_process';
import { existsSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

const webRoot = path.resolve(import.meta.dirname, '..');
const explicitUrl = process.argv[2];
const previewPort = Number(process.env.PREVIEW_PORT ?? 4174);
const url = explicitUrl ?? `http://127.0.0.1:${previewPort}/project-umbra/`;
const port = Number(process.env.CDP_PORT ?? 9222);
const profile = mkdtempSync(path.join(tmpdir(), 'umbra-chrome-'));
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * The numbers this build must produce, pinned rather than matched by shape.
 *
 * A shape-only check passes on a density that lost its transfer, a simulation at
 * the wrong exposure or a reconstruction of the wrong voxel count, because every
 * one of those still prints a plausible looking line. When the physics or the
 * default scene changes on purpose, these move in the same commit — which is
 * where the change has to be stated anyway.
 */
const EXPECTED = {
  rays: 32,
  voxels: 1024,
  nnz: 1280,
  lambda: 32,
  counts: 32,
  events: 50074,
  density: 1024,
  steps: 20,
  // The scan holds the scene itself as H0, and the counts were drawn from it, so
  // the honest result is no detection. Pinning that null is stronger than
  // pinning a discovery: a scan that started reporting 20σ on its own data would
  // fail here, and so would one that silently returned zeros.
  peakZ: 1.66,
  voxel: 529,
  peakAlpha: 1.0,
  records: 5,
};

/** The status line, with every number captured so it can be compared. */
const STATUS =
  /^Assembled (\d+) rays × (\d+) voxels, (\d+) non-zero path lengths · λ Float32Array\((\d+)\) · counts Uint32Array\((\d+)\), (\d+) events · ρ Float32Array\((\d+)\) after (\d+) steps, objective (-?[\d.]+(?:e[+-]?\d+)?) · Z peak (-?[\d.]+) at voxel (\d+) \(α̂ (-?[\d.]+)\) · cached in indexeddb$/;

/** `CHROME_PATH`, or the first Chrome-like binary on this machine's PATH. */
function resolveChrome() {
  const candidates = [
    process.env.CHROME_PATH,
    'google-chrome',
    'chromium',
    'chromium-browser',
    'chrome',
  ].filter(Boolean);
  for (const candidate of candidates) {
    if (spawnSync(candidate, ['--version'], { stdio: 'ignore' }).status === 0) return candidate;
  }
  throw new Error(
    `no Chrome found; set CHROME_PATH (tried: ${candidates.join(', ')})`,
  );
}

/**
 * Serves `dist/` for the duration of the run and resolves once it answers.
 * Returns a stop function; `strictPort` makes a busy port an error instead of
 * a silent move to another one, which would leave the check on the wrong page.
 */
async function startPreview() {
  if (!existsSync(path.join(webRoot, 'dist', 'index.html'))) {
    throw new Error('dist/ is missing; run `npm run build` first');
  }
  // The vite bin runs in this process, so killing it stops the server.
  const server = spawn(
    process.execPath,
    [
      path.join(webRoot, 'node_modules', 'vite', 'bin', 'vite.js'),
      'preview',
      // Explicit IPv4: left to itself vite binds `localhost`, which on a dual
      // stack host can mean `::1` only, and 127.0.0.1 then refuses.
      '--host',
      '127.0.0.1',
      '--port',
      String(previewPort),
      '--strictPort',
    ],
    { cwd: webRoot, stdio: ['ignore', 'ignore', 'pipe'] },
  );
  let stderr = '';
  server.stderr.on('data', (chunk) => {
    stderr += chunk;
  });
  const stop = () => server.kill();
  for (let attempt = 0; attempt < 150; attempt += 1) {
    if (server.exitCode !== null) {
      throw new Error(`the preview server exited (${server.exitCode}):\n${stderr}`);
    }
    try {
      if ((await fetch(url)).ok) return stop;
    } catch {
      // Not listening yet.
    }
    await sleep(200);
  }
  stop();
  throw new Error(`the preview server never answered ${url}\n${stderr}`);
}

/** Console output from the page and from Chrome itself, for a failed run. */
const diagnostics = (client, logLines) =>
  [...client.consoleLines, ...logLines].map((line) => `  console: ${line}`).join('\n') ||
  '  (no console output)';

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
  const chromePath = resolveChrome();
  const stopPreview = explicitUrl ? () => {} : await startPreview();

  // Console output from the *worker* never reaches the page's CDP session, so
  // Chrome's own stderr log is the only place a worker exception surfaces.
  const logLines = [];
  const isRoot = (process.getuid?.() ?? null) === 0;
  const chrome = spawn(
    chromePath,
    [
      '--headless=new',
      '--disable-gpu',
      '--no-first-run',
      '--no-default-browser-check',
      '--enable-logging=stderr',
      '--v=1',
      // Sandboxing needs a non-root user namespace; a root container has none.
      ...(isRoot ? ['--no-sandbox'] : []),
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

  try {
    const target = await pageTarget();
    const client = connect(target.webSocketDebuggerUrl);
    await client.open;
    await client.send('Runtime.enable');
    await client.send('Page.enable');
    await client.send('Page.navigate', { url });

    const expression =
      "document.querySelector('[data-testid=\\\"build-status\\\"]')?.textContent ?? ''";
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
          diagnostics(client, logLines),
      );
    }
    // The whole boundary, not just the first stage: `Float32Array` and
    // `Uint32Array` in the status line are λ, the counts and the density as the
    // main thread received them, after the worker transferred their buffers.
    const match = STATUS.exec(text);
    if (!match) {
      throw new Error(`the build reported a failure: ${text}\n` + diagnostics(client, logLines));
    }
    const observed = {
      rays: Number(match[1]),
      voxels: Number(match[2]),
      nnz: Number(match[3]),
      lambda: Number(match[4]),
      counts: Number(match[5]),
      events: Number(match[6]),
      density: Number(match[7]),
      steps: Number(match[8]),
      peakZ: Number(match[10]),
      voxel: Number(match[11]),
      peakAlpha: Number(match[12]),
    };
    const misplaced = Object.entries(EXPECTED)
      .filter(([field]) => field in observed)
      .filter(([field, want]) => observed[field] !== want)
      .map(([field, want]) => `${field}: expected ${want}, got ${observed[field]}`);
    if (!Number.isFinite(Number(match[9]))) {
      misplaced.push(`objective: not a finite number (${match[9]})`);
    }
    if (misplaced.length > 0) {
      throw new Error(
        `the build returned different numbers than this check pins:\n  ${misplaced.join('\n  ')}\n` +
          `  status: ${text}`,
      );
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
    if (records !== EXPECTED.records) {
      throw new Error(
        `the browser's IndexedDB holds ${String(records)} results, not ` +
          `${EXPECTED.records} (one per completed stage)\n` +
          diagnostics(client, logLines),
      );
    }

    client.close();
    return `${text} [indexeddb records: ${records}]`;
  } finally {
    chrome.kill();
    stopPreview();
    try {
      rmSync(profile, { recursive: true, force: true });
    } catch {
      // Chrome may still hold a file; the temp directory is not worth retrying.
    }
  }
}

try {
  const text = await main();
  console.log(`browser round trip: ${text}`);
} catch (error) {
  console.error(`browser round trip FAILED: ${error.message}`);
  process.exitCode = 1;
}
