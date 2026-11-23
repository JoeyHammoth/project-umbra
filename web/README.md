# umbra-web

Front end for UMBRA: React 19 + TypeScript + Vite, built to a static bundle and
served from GitHub Pages or Cloudflare Pages. No backend, no accounts.

```bash
npm install
npm run build:wasm   # crates/umbra-wasm -> src/wasm/, gitignored
npm run dev          # http://localhost:5173/project-umbra/
npm run typecheck
npm run build        # build:wasm, then tsc, then vite
npm run check:browser # build first: headless Chrome over CDP drives the real bundle
```

`npm run build:wasm` needs the `wasm32-unknown-unknown` target and a
`wasm-bindgen` CLI whose version matches `Cargo.lock`:

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.130 --locked
```

`src/compute/protocol.ts` is the transcription of the Rust ↔ TypeScript contract
(design spec §4.6). Change it in the same PR as the corresponding `umbra-wasm`
change, and bump `API_VERSION` when the contract breaks.

## The compute layer

| File | Role |
| --- | --- |
| `protocol.ts` | Task kinds, payloads, results, events |
| `bus.ts` | Priority queue, cancellation, content keys, cache lookup |
| `workers.ts` | The browser `WorkerHost`: real module workers |
| `worker.ts` | Inside the worker: loads WASM, checks the handshake, calls `handleTask` |
| `handler.ts` | One task in, one event out — and the transfer list that moves the buffers |
| `cache.ts` | `CacheStore` over IndexedDB, so results survive a reload |
| `compute.ts` | What the app calls: bus + workers + persistent cache |

`bus.ts`, `handler.ts` and `cache.ts` are tested under `node --test`
(`npm test`); `worker.ts` needs a browser because it fetches a `.wasm` module.
That is `scripts/check-browser.mjs`: it starts `vite preview` on `dist/`, points
headless Chrome at the page over CDP, waits for the worker's reply, and checks
that the main thread holds real `Float32Array`/`Uint32Array` results and that
Chrome's own IndexedDB has the record. It takes `CHROME_PATH` (default: scan the
PATH for `google-chrome`/`chromium`) and an optional URL to skip starting a
server. CI runs it in the web job, so a broken boundary fails the build.

The heavy layers (Three.js, uPlot, ONNX) are not installed yet; they arrive as
lazy chunks so the landing route stays inside its 250 KB gzip budget.
