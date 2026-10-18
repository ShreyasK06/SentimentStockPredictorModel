import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// GitHub Pages serves a project site from /<repo>/. Override with SITE_BASE for a
// custom domain or a different repo name (e.g. SITE_BASE=/ npm run build).
const base = process.env.SITE_BASE ?? "/SentimentStockPredictorModel/";

export default defineConfig({
  base,
  plugins: [react()],
  build: { outDir: "dist", sourcemap: false, target: "es2020" },
});
