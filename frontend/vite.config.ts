import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // changeOrigin sends the target's own host name, which hosts such as Fly.io route on, so
      // API_PROXY_TARGET can also be the hosted app.
      "/api": { target: process.env.API_PROXY_TARGET ?? "http://localhost:8000", changeOrigin: true },
    },
  },
  test: {
    environment: "jsdom",
  },
});
