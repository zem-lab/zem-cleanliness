import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  base: './',
  plugins: [react()],
  optimizeDeps: { entries: ['index.html'] },
  server: {
    watch: { ignored: ['**/artifacts/**', '**/output/**', '**/.venv/**', '**/venv/**', '**/backend/**', '**/resources/models/**'] },
  },
})
