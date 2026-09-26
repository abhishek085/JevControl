import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The Python server serves the built bundle from jevcontrol/webui; in dev, proxy /api to it.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "../jevcontrol/webui", emptyOutDir: true },
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8600" } },
});
