import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
// Keep this app isolated from any other local FastAPI services using port 8000.
export default defineConfig({ plugins: [react()], server: { proxy: { '/api': { target: 'http://127.0.0.1:8001', rewrite: p => p.replace(/^\/api/, '') } } } });
