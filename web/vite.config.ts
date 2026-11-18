import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Project-site base path: served from https://<user>.github.io/project-umbra/.
// Cloudflare Pages deployments override this with UMBRA_BASE=/.
export default defineConfig({
  base: process.env.UMBRA_BASE ?? '/project-umbra/',
  plugins: [react()],
  build: {
    target: 'es2022',
    sourcemap: true,
  },
  worker: {
    format: 'es',
  },
});
