/**
 * Enforces the payload budgets from spec §1.5 and §7.7 against a real build.
 *
 * The budgets — landing-route JavaScript ≤ 250 KB gzip, wasm core ≤ 700 KB
 * gzip — were documented and not measured, which is the same as not having
 * them: the first feature to cross one would be found by whoever noticed the
 * page got slow. This runs on `dist/`, so it checks the bytes this build
 * produces rather than the bytes someone estimated.
 *
 * What counts:
 *   - landing JS: every module `dist/index.html` loads eagerly — the entry
 *     `<script>` plus any `<link rel="modulepreload">`, which is what Vite
 *     writes for the static import graph. Lazily imported chunks (3D, plots)
 *     are deliberately not charged to the landing route.
 *   - wasm core: the `umbra_wasm*` module in `dist/assets/`. Matching the name
 *     rather than summing every `.wasm` keeps a future, separately budgeted
 *     payload (an ONNX model loads at 1.5 MB) from failing this gate; a build
 *     that emits no core file at all fails, so a rename cannot pass by
 *     measuring nothing.
 *
 * Sizes are gzip level 9, which is what a static host's gzip configuration
 * approaches; a host serving brotli will be smaller than these numbers, so the
 * gate errs towards failing early.
 */

import { readFileSync, readdirSync, statSync } from 'node:fs';
import { gzipSync } from 'node:zlib';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const dist = path.join(web, 'dist');
const assets = path.join(dist, 'assets');

const KIB = 1024;
/** Spec §7.7: landing route JS ≤ 250 KB gzip. */
const LANDING_JS_BUDGET = 250 * KIB;
/** Spec §7.7: wasm core ≤ 700 KB gzip. */
const WASM_BUDGET = 700 * KIB;

const gzipSize = (file) => gzipSync(readFileSync(file), { level: 9 }).length;
const kib = (bytes) => `${(bytes / KIB).toFixed(1)} KiB`;
const rawKib = (file) => kib(statSync(file).size);

/**
 * URLs from `index.html` that the browser fetches before the first paint.
 * Attribute order is not fixed by the HTML spec, so each tag is read as a
 * whole instead of matching `rel` and `href` as one pattern.
 */
function eagerModules(html) {
  const urls = [];
  const push = (url) => {
    if (url && !urls.includes(url)) urls.push(url);
  };

  for (const [tag] of html.matchAll(/<script\b[^>]*>/g)) {
    if (/\bsrc\s*=/.test(tag)) push(tag.match(/\bsrc\s*=\s*"([^"]+)"/)?.[1]);
  }
  for (const [tag] of html.matchAll(/<link\b[^>]*>/g)) {
    if (!/\brel\s*=\s*"[^"]*modulepreload[^"]*"/.test(tag)) continue;
    push(tag.match(/\bhref\s*=\s*"([^"]+)"/)?.[1]);
  }
  return urls;
}

/** `dist/assets/index-abc.js` from `/project-umbra/assets/index-abc.js`. */
function buildFile(url) {
  const [, relative] = url.split('/assets/');
  if (!relative) return null;
  const file = path.join(assets, relative.split('?')[0]);
  return statSync(file, { throwIfNoEntry: false }) ? file : null;
}

function main() {
  const indexHtml = path.join(dist, 'index.html');
  if (!statSync(indexHtml, { throwIfNoEntry: false })) {
    console.error(`no ${path.relative(web, indexHtml)}: run \`npm run build\` first`);
    process.exit(1);
  }

  const failures = [];
  const report = (label, measured, budget, detail) => {
    const over = measured > budget;
    console.log(
      `${label.padEnd(11)} ${kib(measured).padStart(9)} of ${kib(budget).padEnd(9)} gzip` +
        `  ${over ? 'OVER' : 'ok'}` +
        (detail ? `  (${detail})` : ''),
    );
    if (over) {
      failures.push(`${label} is ${kib(measured)}, ${kib(measured - budget)} over its ${kib(budget)} budget`);
    }
  };

  const modules = eagerModules(readFileSync(indexHtml, 'utf8')).map(buildFile);
  const missing = modules.filter((file) => file === null).length;
  if (modules.length === 0 || missing > 0) {
    // A missing file means the URL scheme moved and this check read nothing;
    // reporting 0 KiB of JavaScript would be a pass on no evidence.
    console.error(
      `${indexHtml} references ${modules.length} eager module(s), ${missing} of which do not exist in dist/assets`,
    );
    process.exit(1);
  }
  const landingJs = modules.reduce((total, file) => total + gzipSize(file), 0);
  report(
    'landing JS',
    landingJs,
    LANDING_JS_BUDGET,
    modules.map((file) => `${path.basename(file)} ${rawKib(file)} raw`).join(', '),
  );

  const cores = readdirSync(assets).filter((name) => /^umbra_wasm.*\.wasm$/.test(name));
  if (cores.length === 0) {
    console.error('no umbra_wasm*.wasm in dist/assets: the core did not reach the build');
    process.exit(1);
  }
  const wasm = cores.reduce((total, name) => total + gzipSize(path.join(assets, name)), 0);
  report(
    'wasm core',
    wasm,
    WASM_BUDGET,
    cores.map((name) => `${name} ${rawKib(path.join(assets, name))} raw`).join(', '),
  );

  if (failures.length > 0) {
    for (const failure of failures) console.error(`FAIL: ${failure}`);
    console.error('Spec §1.5 makes these budgets success criteria; split or lazy-load before merging.');
    process.exit(1);
  }
}

main();
