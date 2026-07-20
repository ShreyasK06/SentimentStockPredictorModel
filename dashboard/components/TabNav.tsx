"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useLayoutEffect, useRef, useState } from "react";

interface Tab {
  href: string;
  label: string;
}

// Sub-pages nested under "Models". Rendered as a persistent pill strip
// (tier 2, below) whenever the current route is one of these four, so they
// stay discoverable without a click-to-open menu, while not permanently
// consuming top-level nav space on every other page (Paper Trading, Stocks,
// About, etc).
const MODEL_TABS: Tab[] = [
  { href: "/predictions", label: "Predictions" },
  { href: "/weekly", label: "Weekly" },
  { href: "/twenty-day", label: "20-Day" },
  { href: "/volatility", label: "Volatility" },
];

// Tier 1: primary row, always visible, same 4 top-level destinations (and
// same visual weight) as before a dropdown was introduced — a prior pass
// measured this row against the header's max-w-6xl (1152px) container
// alongside the brand label and the email/sign-out/theme controls on the
// right, and found a fully flat 7-item bar didn't reliably fit, hence
// grouping the 4 model pages under one "Models" entry. That grouping (and
// its width math) is unchanged here: "Models" is still a single top-level
// item — it's just no longer a click-to-open menu trigger. It's a plain
// link to the default model page (/predictions), active whenever the route
// is any MODEL_TABS entry rather than only /predictions itself.
// Discoverability of the four model pages now comes from tier 2 below,
// not from a dropdown.
const PRIMARY_TABS: Tab[] = [
  { href: "/predictions", label: "Models" },
  { href: "/paper-trading", label: "Paper Trading" },
  { href: "/stocks", label: "Stocks" },
  { href: "/about", label: "About" },
];

function isModelRoute(pathname: string): boolean {
  return MODEL_TABS.some((tab) => tab.href === pathname);
}

export function TabNav() {
  const pathname = usePathname();
  const navRef = useRef<HTMLElement>(null);
  const triggerRefs = useRef<Record<string, HTMLElement | null>>({});
  // Sliding active-tab indicator position for tier 1, in pixels relative to
  // the tier-1 nav container. Purely presentational — does not affect which
  // tab is "active" (derived from usePathname()/isModelRoute below), only
  // how the underline visually transitions between tabs. Slides under
  // "Models" for any of the four model routes, not just /predictions.
  const [indicator, setIndicator] = useState<{ left: number; width: number } | null>(null);

  const showModelTabs = isModelRoute(pathname);

  useLayoutEffect(() => {
    function measure() {
      const container = navRef.current;
      const activeTab = PRIMARY_TABS.find((tab) =>
        tab.label === "Models" ? showModelTabs : tab.href === pathname
      );
      const el = activeTab ? triggerRefs.current[activeTab.href] : null;
      if (!container || !el) {
        setIndicator(null);
        return;
      }
      const containerRect = container.getBoundingClientRect();
      const elRect = el.getBoundingClientRect();
      setIndicator({ left: elRect.left - containerRect.left, width: elRect.width });
    }
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pathname]);

  return (
    <div className="flex flex-col">
      <nav ref={navRef} className="relative flex gap-1 text-sm">
        {PRIMARY_TABS.map((tab) => {
          const active = tab.label === "Models" ? showModelTabs : tab.href === pathname;
          return (
            <Link
              key={tab.href}
              href={tab.href}
              ref={(el) => {
                triggerRefs.current[tab.href] = el;
              }}
              aria-current={active ? "page" : undefined}
              className={
                active
                  ? "px-3 py-2 font-medium text-ink"
                  : "px-3 py-2 text-ink-secondary transition-colors hover:text-ink"
              }
            >
              {tab.label}
            </Link>
          );
        })}
        <span
          className="pointer-events-none absolute bottom-0 h-0.5 bg-accent transition-[left,width] duration-200 ease-out"
          style={{
            left: indicator?.left ?? 0,
            width: indicator?.width ?? 0,
            opacity: indicator ? 1 : 0,
          }}
          aria-hidden="true"
        />
      </nav>

      {/* Tier 2: persistent pill strip for the four model pages, rendered
          only while on one of them — this (not a dropdown) is what makes
          them discoverable, without permanently consuming top-level nav
          space on every other page. Uses a solid-fill active pill rather
          than a second sliding underline: the fill alone is a clear enough
          active indicator here, and it reuses the exact pill styling
          already established by PredictionsStrategySwitcher elsewhere in
          this app, so no new visual language is introduced. */}
      {showModelTabs && (
        <nav
          aria-label="Model pages"
          className="mt-1 flex w-fit gap-1 overflow-hidden rounded-[var(--radius-md)] border border-rule bg-surface p-1 text-sm"
        >
          {MODEL_TABS.map((tab) => {
            const active = tab.href === pathname;
            return (
              <Link
                key={tab.href}
                href={tab.href}
                aria-current={active ? "page" : undefined}
                className={
                  active
                    ? "rounded-[var(--radius-md)] bg-accent px-3 py-1.5 font-medium text-bg"
                    : "rounded-[var(--radius-md)] px-3 py-1.5 text-ink-secondary transition-colors hover:text-ink"
                }
              >
                {tab.label}
              </Link>
            );
          })}
        </nav>
      )}
    </div>
  );
}
