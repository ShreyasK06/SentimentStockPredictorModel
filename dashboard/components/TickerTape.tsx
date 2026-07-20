"use client";

import { useEffect, useRef } from "react";
import { UNIVERSE, SECTOR_MAP } from "@/lib/data/universe";
import { SECTOR_COLOR_VARS } from "@/lib/ui/sectorColors";

const SPEED_PX_PER_SEC = 22;
const GAP_PX = 30;

// Ambient, decorative canvas ticker tape for the dashboard header — the
// 28-ticker universe drifting slowly left with a sector-colored dot per
// symbol. Purely presentational: draws from static in-memory constants
// only, no data fetching. Freezes on a single static frame (no rAF loop
// at all) under prefers-reduced-motion, matching the pattern used
// elsewhere in this project for reduced-motion handling.
export function TickerTape() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);

    let width = 0;
    let height = 0;
    let raf = 0;

    const rootStyles = getComputedStyle(document.documentElement);
    function cssVar(name: string, fallback: string) {
      const v = rootStyles.getPropertyValue(name).trim();
      return v || fallback;
    }
    const inkMuted = cssVar("--ink-muted", "#7c8c81");
    const fontStack = cssVar("--font-mono-stack", "monospace");
    const font = `11px ${fontStack}`;

    function resize() {
      if (!canvas) return;
      width = canvas.clientWidth;
      height = canvas.clientHeight;
      canvas.width = Math.max(1, Math.round(width * dpr));
      canvas.height = Math.max(1, Math.round(height * dpr));
      ctx!.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    resize();
    window.addEventListener("resize", resize);

    ctx.font = font;
    const segments = UNIVERSE.map((ticker) => {
      const sector = SECTOR_MAP[ticker];
      const color = cssVar(SECTOR_COLOR_VARS[sector], inkMuted);
      const textWidth = ctx.measureText(ticker).width;
      return { ticker, color, segWidth: 8 /* dot */ + 8 /* dot gap */ + textWidth + GAP_PX };
    });
    const totalWidth = segments.reduce((sum, s) => sum + s.segWidth, 0) || 1;

    let offset = 0;
    let lastTime = performance.now();

    function drawFrame() {
      ctx!.clearRect(0, 0, width, height);
      ctx!.font = font;
      ctx!.textBaseline = "middle";

      let x = -offset;
      while (x < width) {
        for (const seg of segments) {
          if (x + seg.segWidth > 0 && x < width) {
            const dotY = height / 2;
            ctx!.beginPath();
            ctx!.fillStyle = seg.color;
            ctx!.arc(x + 4, dotY, 3, 0, Math.PI * 2);
            ctx!.fill();
            ctx!.fillStyle = inkMuted;
            ctx!.fillText(seg.ticker, x + 14, dotY + 1);
          }
          x += seg.segWidth;
        }
      }
    }

    function tick(now: number) {
      const dt = (now - lastTime) / 1000;
      lastTime = now;
      offset = (offset + SPEED_PX_PER_SEC * dt) % totalWidth;
      drawFrame();
      raf = requestAnimationFrame(tick);
    }

    if (reducedMotion) {
      // Freeze on a single static frame — no animation loop at all.
      drawFrame();
    } else {
      raf = requestAnimationFrame(tick);
    }

    return () => {
      if (raf) cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="block h-7 w-full"
      role="presentation"
      aria-hidden="true"
    />
  );
}
