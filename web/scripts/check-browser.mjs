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
  records: 6,
  /** Toys the page's look-elsewhere calibration runs; see `LEE_TOYS` in App. */
  leeToys: 200,
  // Every voxel of a 32 x 32 grid carries a value, so every pixel is painted.
  // Fewer means part of the field never reached the canvas. "Lit" is a non-zero
  // red channel, which the heatmap's own first stop guarantees for any density
  // above zero (it starts at 16, not 0) while zero stays black.
  litVoxels: 1024,
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
  const stop = () => {
    server.kill();
    // Detach the killed server from the event loop: the check is over either
    // way, and waiting on a process the OS has already reaped is how a one-shot
    // harness ends up hanging after printing its result.
    server.unref();
  };
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

    // The canvas is the deliverable, not the status line. A field can be
    // transferred correctly and still never reach a pixel, so read back what the
    // page painted rather than trusting that a `density` prop was passed.
    const painted = await client.send('Runtime.evaluate', {
      expression: `(() => {
        const canvas = document.querySelector('canvas[data-testid="density"]');
        if (!canvas) return { found: false };
        const context = canvas.getContext('2d');
        if (!context) return { found: true, width: canvas.width, height: canvas.height, lit: -1 };
        const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
        let lit = 0;
        for (let i = 0; i < pixels.length; i += 4) { if (pixels[i] > 0) lit += 1; }
        return { found: true, width: canvas.width, height: canvas.height, lit };
      })()`,
      returnByValue: true,
    });
    const drawn = painted.result?.result?.value;
    if (!drawn?.found) {
      throw new Error('the page has no density canvas\n' + diagnostics(client, logLines));
    }
    if (drawn.width !== 32 || drawn.height !== 32) {
      throw new Error(
        `the density canvas is ${drawn.width} x ${drawn.height}, not the 32 x 32 grid`,
      );
    }
    if (drawn.lit !== EXPECTED.litVoxels) {
      throw new Error(
        `the density canvas painted ${String(drawn.lit)} of ${EXPECTED.litVoxels} voxels`,
      );
    }

    // The scan's field is a second canvas, and it is only a picture if `Z`
    // reached a pixel. Comparing it with the density canvas proves the two hold
    // different data: a fallback that painted rho twice would pass a brightness
    // count and fail this.
    const scanned = await client.send('Runtime.evaluate', {
      expression: `(() => {
        const density = document.querySelector('canvas[data-testid="density"]');
        const scan = document.querySelector('canvas[data-testid="scan"]');
        if (!scan) return { found: false };
        const d = density.getContext('2d').getImageData(0, 0, density.width, density.height).data;
        const s = scan.getContext('2d').getImageData(0, 0, scan.width, scan.height).data;
        let identical = d.length === s.length;
        for (let i = 0; identical && i < s.length; i += 1) {
          if (d[i] !== s[i]) identical = false;
        }
        return { found: true, width: scan.width, height: scan.height, identical };
      })()`,
      returnByValue: true,
    });
    const scanDrawn = scanned.result?.result?.value;
    if (!scanDrawn?.found) {
      throw new Error('the page has no scan canvas\n' + diagnostics(client, logLines));
    }
    if (scanDrawn.width !== 32 || scanDrawn.height !== 32) {
      throw new Error(
        `the scan canvas is ${scanDrawn.width} x ${scanDrawn.height}, not the 32 x 32 grid`,
      );
    }
    if (scanDrawn.identical) {
      throw new Error(
        'the scan canvas is a copy of the density canvas: the Z field never arrived',
      );
    }

    // The look-elsewhere stage is what turns a local Z into a statement about
    // the whole search. It must be about the same peak the scan reported: a
    // calibration paired with another dataset would still print a plausible
    // number, so the peak is compared rather than merely present.
    const lee = await client.send('Runtime.evaluate', {
      expression: `(() => {
        const node = document.querySelector('[data-testid="lee"]');
        if (!node) return { found: false };
        return {
          found: true,
          toys: Number(node.dataset.toys),
          peakZ: Number(node.dataset.peakZ),
          globalP: Number(node.dataset.globalP),
        };
      })()`,
      returnByValue: true,
    });
    const calibrated = lee.result?.result?.value;
    if (!calibrated?.found) {
      throw new Error(
        'the page has no look-elsewhere readout\n' + diagnostics(client, logLines),
      );
    }
    if (calibrated.toys !== EXPECTED.leeToys) {
      throw new Error(
        `the calibration ran ${String(calibrated.toys)} toys, not ${EXPECTED.leeToys}`,
      );
    }
    // The status line carries the peak to two decimals, so compare at that
    // precision rather than demanding the full f32.
    if (Math.abs(calibrated.peakZ - observed.peakZ) > 0.005) {
      throw new Error(
        `the calibration is about a ${calibrated.peakZ.toFixed(2)}σ peak, but the scan reported ${observed.peakZ}σ`,
      );
    }
    if (!(calibrated.globalP > 0 && calibrated.globalP <= 1)) {
      throw new Error(
        `the calibration reported a global p of ${String(calibrated.globalP)}`,
      );
    }

    // The correction is drawn as well as quoted: the curve has to reach an SVG
    // path, one point per threshold, or the figure is an empty frame that still
    // looks finished.
    const curve = await client.send('Runtime.evaluate', {
      expression: `(() => {
        const svg = document.querySelector('[data-testid="global-p"]');
        if (!svg) return { found: false };
        const path = svg.querySelector('path.global-p-curve');
        const d = path?.getAttribute('d') ?? '';
        const model = svg.querySelector('path.global-p-model')?.getAttribute('d') ?? '';
        const band = svg.querySelector('path.global-p-band')?.getAttribute('d') ?? '';
        const toys = svg.querySelector('[data-testid="global-p-toys"]');
        return {
          found: true,
          points: (d.match(/[ML]/g) ?? []).length,
          label: svg.getAttribute('aria-label') ?? '',
          model: svg.getAttribute('data-model'),
          modelPoints: (model.match(/[ML]/g) ?? []).length,
          bandPoints: (band.match(/[ML]/g) ?? []).length,
          toysMark: toys ? Number(toys.dataset.maxZ) : 0,
        };
      })()`,
      returnByValue: true,
    });
    const curveDrawn = curve.result?.result?.value;
    if (!curveDrawn?.found) {
      throw new Error('the page has no global-p curve\n' + diagnostics(client, logLines));
    }
    if (curveDrawn.points !== 121) {
      throw new Error(
        `the curve has ${String(curveDrawn.points)} points, not the 121 samples 0-6σ needs`,
      );
    }
    if (!curveDrawn.label.includes('background-only toys')) {
      throw new Error(`the curve's label does not say what it is from: ${curveDrawn.label}`);
    }
    // The model branch is drawn only when the toys measured an excursion at the
    // anchor. Either way the SVG must agree with itself: a `model` label with no
    // path is an empty promise, and a path under a `none` label is a model drawn
    // from nothing.
    if (curveDrawn.model === 'model') {
      if (!(curveDrawn.modelPoints >= 40)) {
        throw new Error(
          `the page says it draws the Gross–Vitells model but its path has ${String(curveDrawn.modelPoints)} points`,
        );
      }
      // As many points as the centre line: a band with half the samples would be
      // a different curve, not the width of this one.
      if (curveDrawn.bandPoints !== 2 * curveDrawn.modelPoints) {
        throw new Error(
          `the model's band has ${String(curveDrawn.bandPoints)} points for a ${String(curveDrawn.modelPoints)}-point curve`,
        );
      }
      // The measured part of the curve ends at the largest maximum in the null,
      // and with an excursion at the anchor that maximum is above u0 = 2.5σ.
      if (!(curveDrawn.toysMark > 2.5 && curveDrawn.toysMark < 6)) {
        throw new Error(
          `the page does not mark where the toys stop (max Z ${String(curveDrawn.toysMark)})`,
        );
      }
      if (!curveDrawn.label.includes('largest maximum among the toys')) {
        throw new Error(`the curve's label does not say where the toys stop: ${curveDrawn.label}`);
      }
    } else if (
      curveDrawn.model !== 'none' ||
      curveDrawn.modelPoints !== 0 ||
      curveDrawn.bandPoints !== 0 ||
      curveDrawn.toysMark !== 0
    ) {
      throw new Error(
        `the model branch is ${String(curveDrawn.model)} with ${String(curveDrawn.modelPoints)} points`,
      );
    }

    // A rate without its interval reads as precise. The peak's p must arrive
    // with its Wilson width and the floor of what this toy count can express.
    const reading = await client.send('Runtime.evaluate', {
      expression: `document.querySelector('[data-testid="lee"]')?.textContent ?? ''`,
      returnByValue: true,
    });
    const paragraph = reading.result?.result?.value ?? '';
    if (!/± 0\.\d{3} at 95%/.test(paragraph)) {
      throw new Error(
        `the global p does not quote its 95% interval: ${paragraph.slice(0, 160)}`,
      );
    }
    if (!/cannot express anything below 0\.\d{4}/.test(paragraph)) {
      throw new Error(
        `the readout does not say what the toy count can resolve: ${paragraph.slice(0, 160)}`,
      );
    }

    // The window is a control, not a caption. Moving both sliders to the full
    // positive range must change the pixels: a slider wired only to the legend
    // would leave this canvas identical and read as working.
    const moved = await client.send('Runtime.evaluate', {
      expression: `(async () => {
        const canvas = document.querySelector('canvas[data-testid="density"]');
        const context = canvas.getContext('2d');
        const before = context.getImageData(0, 0, canvas.width, canvas.height).data;
        const sliders = [...document.querySelectorAll('.window input[type="range"]')];
        if (sliders.length !== 2) return { found: false, sliders: sliders.length };
        const set = (input, value) => {
          const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
          setter.call(input, String(value));
          input.dispatchEvent(new Event('input', { bubbles: true }));
        };
        set(sliders[0], 0);
        set(sliders[1], 100);
        await new Promise((resolve) => setTimeout(resolve, 250));
        const after = context.getImageData(0, 0, canvas.width, canvas.height).data;
        let changed = 0;
        for (let i = 0; i < after.length; i += 1) { if (before[i] !== after[i]) changed += 1; }
        return { found: true, changed };
      })()`,
      awaitPromise: true,
      returnByValue: true,
    });
    const windowed = moved.result?.result?.value;
    if (!windowed?.found) {
      throw new Error(
        `the page has ${String(windowed?.sliders)} window sliders, not two\n` +
          diagnostics(client, logLines),
      );
    }
    if (!(windowed.changed > 0)) {
      throw new Error('moving the window to 0–100% painted the same pixels');
    }

    // The toy count is a control with a measured cost, not a constant. Changing
    // it must re-run the calibration and report the new count and its time; a
    // select wired to nothing would leave the old answer on the page.
    const retooled = await client.send('Runtime.evaluate', {
      expression: `(async () => {
        const select = document.querySelector('[data-testid="toys"]');
        if (!select) return { found: false };
        const setter = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set;
        setter.call(select, '50');
        select.dispatchEvent(new Event('change', { bubbles: true }));
        for (let attempt = 0; attempt < 100; attempt += 1) {
          await new Promise((resolve) => setTimeout(resolve, 100));
          const node = document.querySelector('[data-testid="lee"]');
          if (node?.dataset.toys === '50') {
            return {
              found: true,
              toys: 50,
              ms: document.querySelector('[data-testid="toys-ms"]')?.textContent ?? '',
            };
          }
        }
        return { found: true, toys: -1, ms: '' };
      })()`,
      awaitPromise: true,
      returnByValue: true,
    });
    const retuned = retooled.result?.result?.value;
    if (!retuned?.found) {
      throw new Error('the page has no toy-count control\n' + diagnostics(client, logLines));
    }
    if (retuned.toys !== 50) {
      throw new Error(
        `changing the toy count left ${String(retuned.toys)} toys on the page`,
      );
    }
    if (!/^\d+ ms$/.test(retuned.ms)) {
      throw new Error(
        `the toy count reports no measured time: ${JSON.stringify(retuned.ms)}`,
      );
    }

    client.close();
    return `${text} [indexeddb records: ${records}] [canvas 32x32, ${drawn.lit} lit] [scan 32x32, distinct] [lee ${calibrated.toys} toys, p ${calibrated.globalP}] [window moved, ${windowed.changed} bytes] [curve ${curveDrawn.points} points] [model ${curveDrawn.model === 'model' ? `${curveDrawn.modelPoints} points` : 'none'}] [band ${curveDrawn.bandPoints} points] [toys stop ${curveDrawn.toysMark}σ] [toys 50, ${retuned.ms}]`;
  } finally {
    chrome.kill();
    // A killed Chrome can leave a renderer holding the inherited stderr pipe,
    // and on Windows that keeps Node's loop alive after the result is printed.
    // Release the handles rather than waiting for processes this script no
    // longer needs.
    chrome.stderr?.destroy();
    chrome.unref();
    stopPreview();
    try {
      rmSync(profile, { recursive: true, force: true });
    } catch {
      // Chrome may still hold a file; the temp directory is not worth retrying.
    }
  }
}

// Exit once the line is flushed. A killed headless Chrome can leave a renderer
// holding the stderr pipe it inherited, and waiting for that process to die is
// how a one-shot check hangs after it already has its answer -- which reads as a
// timeout in CI even though the round trip passed. The write callback runs after
// the pipe has taken the bytes, so exiting there loses nothing.
try {
  const text = await main();
  process.stdout.write(`browser round trip: ${text}\n`, () => process.exit(0));
} catch (error) {
  process.stderr.write(`browser round trip FAILED: ${error.message}\n`, () => process.exit(1));
}
