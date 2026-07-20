"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useLayoutEffect, useRef, useState } from "react";

interface Tab {
  href: string;
  label: string;
}

interface TabGroup {
  label: string;
  children: Tab[];
}

// Flat entries render as plain links. "Models" groups every
// prediction-producing strategy page behind a dropdown — with 7 total
// destinations, a fully flat bar doesn't reliably fit inside the header's
// max-w-6xl (1152px) container alongside the brand label and the
// email/sign-out/theme controls on the right, so strategy pages are grouped
// and Paper Trading/Stocks/About stay flat top-level tabs.
const TABS: (Tab | TabGroup)[] = [
  {
    label: "Models",
    children: [
      { href: "/predictions", label: "Predictions" },
      { href: "/weekly", label: "Weekly" },
      { href: "/twenty-day", label: "20-Day" },
      { href: "/volatility", label: "Volatility" },
    ],
  },
  { href: "/paper-trading", label: "Paper Trading" },
  { href: "/stocks", label: "Stocks" },
  { href: "/about", label: "About" },
];

function isGroup(tab: Tab | TabGroup): tab is TabGroup {
  return "children" in tab;
}

export function TabNav() {
  const pathname = usePathname();
  const navRef = useRef<HTMLElement>(null);
  const triggerRefs = useRef<Record<string, HTMLElement | null>>({});
  const groupRef = useRef<HTMLDivElement>(null);
  // Sliding active-tab indicator position, in pixels relative to the nav
  // container. Purely presentational — does not affect which tab is
  // "active" (that's still derived from usePathname() below), only how the
  // underline visually transitions between tabs. Positioned under the
  // group's own trigger button when a child route is active, so the
  // indicator still lands somewhere meaningful even though the dropdown
  // itself is collapsed most of the time.
  const [indicator, setIndicator] = useState<{ left: number; width: number } | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);

  function keyFor(tab: Tab | TabGroup): string {
    return isGroup(tab) ? `group:${tab.label}` : tab.href;
  }
  // A group is "active" (and so is its trigger, for indicator purposes)
  // when the current route matches one of its children.
  function isActive(tab: Tab | TabGroup): boolean {
    return isGroup(tab)
      ? tab.children.some((child) => child.href === pathname)
      : tab.href === pathname;
  }

  useLayoutEffect(() => {
    function measure() {
      const container = navRef.current;
      const activeTab = TABS.find((tab) => isActive(tab));
      const el = activeTab ? triggerRefs.current[keyFor(activeTab)] : null;
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

  // Close the dropdown on outside click / Escape while open. Navigating via
  // a child link closes it directly in that link's onClick (below) rather
  // than via a pathname-keyed effect, since setState-in-effect triggers a
  // cascading-render lint error and isn't needed here anyway.
  useEffect(() => {
    if (!menuOpen) return;
    function handlePointerDown(e: MouseEvent) {
      if (groupRef.current && !groupRef.current.contains(e.target as Node)) {
        setMenuOpen(false);
      }
    }
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") setMenuOpen(false);
    }
    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [menuOpen]);

  return (
    <nav ref={navRef} className="relative flex gap-1 text-sm">
      {TABS.map((tab) => {
        const active = isActive(tab);
        const key = keyFor(tab);

        if (isGroup(tab)) {
          return (
            <div key={key} ref={groupRef} className="relative">
              <button
                type="button"
                ref={(el) => {
                  triggerRefs.current[key] = el;
                }}
                onClick={() => setMenuOpen((open) => !open)}
                aria-haspopup="menu"
                aria-expanded={menuOpen}
                className={
                  active
                    ? "flex items-center gap-1 px-3 py-2 font-medium text-ink"
                    : "flex items-center gap-1 px-3 py-2 text-ink-secondary transition-colors hover:text-ink"
                }
              >
                {tab.label}
                <span
                  className={`text-[10px] transition-transform ${menuOpen ? "rotate-180" : ""}`}
                  aria-hidden="true"
                >
                  ▾
                </span>
              </button>
              {menuOpen && (
                <div
                  role="menu"
                  className="absolute left-0 top-full z-50 mt-1 flex w-40 flex-col overflow-hidden rounded-[var(--radius-lg)] border border-rule bg-surface py-1 shadow-[var(--shadow-float)]"
                >
                  {tab.children.map((child) => {
                    const childActive = pathname === child.href;
                    return (
                      <Link
                        key={child.href}
                        href={child.href}
                        role="menuitem"
                        onClick={() => setMenuOpen(false)}
                        className={
                          childActive
                            ? "px-3 py-2 text-sm font-medium text-ink"
                            : "px-3 py-2 text-sm text-ink-secondary transition-colors hover:bg-bg hover:text-ink"
                        }
                      >
                        {child.label}
                      </Link>
                    );
                  })}
                </div>
              )}
            </div>
          );
        }

        return (
          <Link
            key={key}
            href={tab.href}
            ref={(el) => {
              triggerRefs.current[key] = el;
            }}
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
  );
}
