/**
 * Builds the compute core for the browser and generates its JS bindings.
 *
 * `cargo build --target wasm32-unknown-unknown` produces a `.wasm` with no
 * JavaScript around it; `wasm-bindgen` writes the glue that turns `Vec<f32>`
 * into a `Float32Array` and `Option<String>` into `string | undefined`. The
 * output goes to `src/wasm/`, which is generated and gitignored — run this
 * before `npm run typecheck`, `npm test` or `npm run dev`.
 */

import { execFileSync } from 'node:child_process';
import { mkdirSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const root = path.resolve(web, '..');
const manifest = path.join(root, 'Cargo.toml');
const outDir = path.join(web, 'src', 'wasm');
const { target_directory: targetDir } = JSON.parse(
  execFileSync(
    'cargo',
    ['metadata', '--manifest-path', manifest, '--format-version', '1', '--no-deps'],
    { encoding: 'utf8' },
  ),
);
const artifact = path.join(
  targetDir,
  'wasm32-unknown-unknown',
  'release',
  'umbra_wasm.wasm',
);

execFileSync(
  'cargo',
  [
    'build',
    '--manifest-path',
    manifest,
    '-p',
    'umbra-wasm',
    '--target',
    'wasm32-unknown-unknown',
    '--release',
  ],
  { stdio: 'inherit' },
);

mkdirSync(outDir, { recursive: true });

try {
  execFileSync(
    'wasm-bindgen',
    ['--target', 'web', '--out-dir', outDir, '--out-name', 'umbra_wasm', artifact],
    { stdio: 'inherit' },
  );
} catch (error) {
  console.error(
    '\nwasm-bindgen failed. It must match the `wasm-bindgen` version in Cargo.lock:\n' +
      '  cargo install wasm-bindgen-cli --version 0.2.130 --locked\n',
  );
  throw error;
}

const wasm = path.join(outDir, 'umbra_wasm_bg.wasm');
console.log(
  `umbra_wasm_bg.wasm: ${(statSync(wasm).size / 1024).toFixed(0)} KiB (before gzip)`,
);
