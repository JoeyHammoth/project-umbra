# umbra-web

Front end for UMBRA: React 19 + TypeScript + Vite, built to a static bundle and
served from GitHub Pages or Cloudflare Pages. No backend, no accounts.

```bash
npm install
npm run dev        # http://localhost:5173/project-umbra/
npm run typecheck
npm run build
```

`src/compute/protocol.ts` is the transcription of the Rust ↔ TypeScript contract
(design spec §4.6). Change it in the same PR as the corresponding `umbra-wasm`
change, and bump `API_VERSION` when the contract breaks.

The heavy layers (Three.js, uPlot, the WASM core, ONNX) are not installed yet;
they arrive as lazy chunks so the landing route stays inside its 250 KB gzip
budget.
