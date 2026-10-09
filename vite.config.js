import { defineConfig } from 'vite';
import tailwindcss from '@tailwindcss/vite';
// Cross-origin isolation lets the in-browser model use several CPU threads (production adds it with public/coi-sw.js).
const isolation = {'Cross-Origin-Opener-Policy':'same-origin','Cross-Origin-Embedder-Policy':'require-corp'};
// BASE_PATH lets GitHub Pages serve the site from /<repo>/. onnxruntime-web must not be
// pre-bundled, or its .wasm files cannot be located in development.
export default defineConfig({base:process.env.BASE_PATH||'/',plugins:[tailwindcss()],optimizeDeps:{exclude:['onnxruntime-web']},worker:{format:'es'},server:{headers:isolation}});
