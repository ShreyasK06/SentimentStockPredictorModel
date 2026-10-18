import { useEffect, useRef, useState } from "react";
import { useAllData } from "./lib/data";
import { Nav } from "./components/Nav";
import { Hero } from "./components/Hero";
import { Pipeline } from "./components/Pipeline";
import { SentimentSection } from "./components/SentimentSection";
import { Demo } from "./components/Demo";
import { Direction } from "./components/Direction";
import { Learned } from "./components/Learned";
import { Method } from "./components/Method";
import { Footer } from "./components/Footer";
import { Explorer } from "./components/Explorer";

type Route = "home" | "results";
const routeOf = (): Route => (window.location.hash.startsWith("#/results") ? "results" : "home");

function useRoute(): Route {
  const [r, setR] = useState<Route>(routeOf);
  useEffect(() => {
    const on = () => setR(routeOf());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return r;
}

export default function App() {
  const state = useAllData();
  const route = useRoute();
  const mainRef = useRef<HTMLElement | null>(null);
  const ready = state.status === "ready";

  useEffect(() => {
    document.title = route === "results" ? "Results explorer | Sentiment Stock Predictor" : "Sentiment Stock Predictor";
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const behavior: ScrollBehavior = reduce ? "auto" : "smooth";
    if (route === "results") {
      window.scrollTo({ top: 0, behavior: "auto" });
      document.getElementById("ex-title")?.focus({ preventScroll: true });
      return;
    }
    const id = window.location.hash.replace("#", "");
    if (!id || id === "top") {
      window.scrollTo({ top: 0, behavior: "auto" });
      return;
    }
    // Sections render after the data arrives; wait for them, then align to the hash.
    const t = window.setTimeout(() => document.getElementById(id + "-sec")?.scrollIntoView({ behavior, block: "start" }) ?? document.getElementById(id)?.scrollIntoView({ behavior, block: "start" }), 30);
    return () => window.clearTimeout(t);
  }, [route, ready]);

  const p8 = ready ? state.data.phase8 : null;

  return (
    <>
      <a
        className="skip-link"
        href="#main"
        onClick={(e) => {
          e.preventDefault();
          mainRef.current?.focus();
        }}
      >
        Skip to content
      </a>
      <Nav route={route} />
      <main id="main" ref={mainRef} tabIndex={-1}>
        {route === "home" && (
          <>
            <div className="wrap">
              <Hero p8={p8} />
            </div>
            {state.status === "loading" && (
              <div className="wrap">
                <p className="loading" role="status">Loading measured results&hellip;</p>
              </div>
            )}
            {state.status === "error" && (
              <div className="wrap">
                <p className="loading err" role="alert">
                  The data files could not be loaded ({state.error}). Run <code>python scripts/export_site_data.py</code> and rebuild.
                </p>
              </div>
            )}
            {state.status === "ready" && (
              <>
                <div className="wrap">
                  <Pipeline />
                  <SentimentSection p9={state.data.phase9} />
                  <Demo ex={state.data.examples} />
                  <Direction p8={state.data.phase8} p7={state.data.phase7} />
                </div>
                <Learned p8={state.data.phase8} />
                <div className="wrap">
                  <Method p8={state.data.phase8} p9={state.data.phase9} />
                </div>
              </>
            )}
          </>
        )}
        {route === "results" && (
          <div className="wrap">
            {state.status === "ready" ? (
              <Explorer p8={state.data.phase8} p7={state.data.phase7} />
            ) : state.status === "error" ? (
              <p className="loading err" role="alert">The data files could not be loaded ({state.error}).</p>
            ) : (
              <p className="loading" role="status">Loading measured results&hellip;</p>
            )}
          </div>
        )}
      </main>
      <Footer meta={ready ? state.data.meta : null} />
    </>
  );
}
