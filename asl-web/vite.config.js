// vite.config.js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { viteStaticCopy } from 'vite-plugin-static-copy'

export default defineConfig({
  plugins: [
    react(),
    viteStaticCopy({
      targets: [
        {
          // Copy the WASM binary
          src: 'node_modules/onnxruntime-web/dist/*.wasm',
          dest: 'ort',
        },
        {
          // CRITICAL: Copy the .mjs JavaScript glue module
          src: 'node_modules/onnxruntime-web/dist/*.mjs',
          dest: 'ort',
        },
      ],
    }),
  ],
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
})