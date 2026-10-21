import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { loadModel, type Prediction, type SentimentModel } from "./lib/model";

const EXAMPLES = [
  "Microsoft cloud growth slows for a third straight quarter",
  "Boeing wins large aircraft order from Asian carrier",
  "Walmart cuts annual outlook citing weaker discretionary spending",
  "Johnson & Johnson receives regulatory approval for new therapy",
  "Home Depot names a new chief financial officer",
  "Caterpillar lowers forecast as construction demand weakens",
  "Meta to cut thousands of jobs in a new round of restructuring",
  "Exxon profit falls as crude prices slide",
];
const MAX_LINES = 200;

type Status = { kind: "loading"; progress: number } | { kind: "ready" } | { kind: "error"; message: string };
interface Row {
  headline: string;
  pred: Prediction;
}

const titleCase = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
const pct = (p: number) => `${(p * 100).toFixed(p >= 0.995 || p < 0.005 ? 0 : 1)}%`;

function toCsv(rows: Row[], labels: string[]): string {
  const esc = (s: string) => `"${s.replace(/"/g, '""')}"`;
  const head = ["headline", "prediction", ...labels.map((l) => `p_${l}`)].join(",");
  const body = rows.map((r) => [esc(r.headline), labels[r.pred.top], ...r.pred.probs.map((p) => p.toFixed(4))].join(","));
  return [head, ...body].join("\n");
}

export default function App() {
  const [status, setStatus] = useState<Status>({ kind: "loading", progress: 0 });
  const [model, setModel] = useState<SentimentModel | null>(null);
  const [text, setText] = useState(EXAMPLES.slice(0, 5).join("\n"));
  const [rows, setRows] = useState<Row[]>([]);
  const [busy, setBusy] = useState(false);
  const [truncated, setTruncated] = useState(false);
  const autoRan = useRef(false);

  const start = useCallback(() => {
    setStatus({ kind: "loading", progress: 0 });
    loadModel((progress) => setStatus({ kind: "loading", progress }))
      .then((m) => {
        setModel(m);
        setStatus({ kind: "ready" });
      })
      .catch((err: unknown) => setStatus({ kind: "error", message: err instanceof Error ? err.message : String(err) }));
  }, []);
  useEffect(start, [start]);

  const analyze = useCallback(
    async (source: string) => {
      if (!model) return;
      const lines = source.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
      setTruncated(lines.length > MAX_LINES);
      const headlines = lines.slice(0, MAX_LINES);
      if (!headlines.length) {
        setRows([]);
        return;
      }
      setBusy(true);
      try {
        // Let the browser paint the busy state before the model blocks the main thread.
        await new Promise((resolve) => setTimeout(resolve, 20));
        const preds = await model.classify(headlines);
        setRows(headlines.map((headline, i) => ({ headline, pred: preds[i] })));
      } finally {
        setBusy(false);
      }
    },
    [model],
  );

  // Score the starter headlines as soon as the model is ready, so the page shows a result straight away.
  useEffect(() => {
    if (model && !autoRan.current) {
      autoRan.current = true;
      void analyze(text);
    }
  }, [model, analyze, text]);

  const labels = model?.labels ?? ["negative", "neutral", "positive"];
  const summary = useMemo(() => {
    if (!rows.length) return null;
    const counts = labels.map(() => 0);
    const mean = labels.map(() => 0);
    for (const r of rows) {
      counts[r.pred.top]++;
      r.pred.probs.forEach((p, i) => (mean[i] += p / rows.length));
    }
    const pos = labels.indexOf("positive");
    const neg = labels.indexOf("negative");
    return { counts, net: mean[pos] - mean[neg] };
  }, [rows, labels]);

  const ready = status.kind === "ready";
  const lineCount = text.split(/\r?\n/).filter((l) => l.trim()).length;

  const download = () => {
    const url = URL.createObjectURL(new Blob([toCsv(rows, labels)], { type: "text/csv" }));
    const a = Object.assign(document.createElement("a"), { href: url, download: "sentiment.csv" });
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <>
      <header className="top">
        <div className="wrap top-in">
          <span className="brand">Headline Sentiment</span>
          <a className="src" href="https://github.com/ShreyasK06/SentimentStockPredictorModel">
            Source on GitHub
          </a>
        </div>
      </header>

      <main className="wrap" id="main">
        <h1>
          Score financial headlines<span className="dot">.</span>
        </h1>
        <p className="lede">
          Paste headlines, one per line. A fine-tuned DistilBERT model reads each one and returns positive, neutral or negative with its
          probabilities. It runs in your browser, so nothing you paste leaves this page.
        </p>

        <section className="card" aria-labelledby="in-h">
          <div className="card-head">
            <h2 id="in-h">Headlines</h2>
            <div className="model-state" role="status" aria-live="polite">
              {status.kind === "loading" && (
                <>
                  <span
                    className="bar"
                    role="progressbar"
                    aria-label="Loading the model"
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuenow={status.progress < 0 ? undefined : Math.round(status.progress * 100)}
                  >
                    <span className={status.progress < 0 ? "bar-fill indet" : "bar-fill"} style={status.progress < 0 ? undefined : { width: `${status.progress * 100}%` }} />
                  </span>
                  <span>{status.progress < 0 ? "Loading model" : `Loading model ${Math.round(status.progress * 100)}%`}</span>
                </>
              )}
              {status.kind === "ready" && (
                <>
                  <span className="ok-dot" aria-hidden="true" />
                  <span>Model ready</span>
                </>
              )}
              {status.kind === "error" && (
                <>
                  <span>Model failed to load.</span>
                  <button type="button" className="link" onClick={start}>
                    Retry
                  </button>
                </>
              )}
            </div>
          </div>

          <label htmlFor="headlines" className="sr-only">
            Headlines, one per line
          </label>
          <textarea
            id="headlines"
            value={text}
            rows={7}
            spellCheck={false}
            placeholder="Paste one headline per line"
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
                e.preventDefault();
                void analyze(text);
              }
            }}
          />

          <div className="examples" role="group" aria-label="Add an example headline">
            <span className="muted">Try</span>
            {EXAMPLES.slice(0, 6).map((ex) => (
              <button key={ex} type="button" className="chip" onClick={() => setText((t) => (t.trim() ? `${t.trimEnd()}\n${ex}` : ex))}>
                {ex.split(" ").slice(0, 3).join(" ")}...
              </button>
            ))}
          </div>

          <div className="actions">
            <button type="button" className="btn primary" disabled={!ready || busy || lineCount === 0} onClick={() => void analyze(text)}>
              {busy ? "Scoring..." : ready ? `Score ${lineCount} headline${lineCount === 1 ? "" : "s"}` : "Loading model..."}
            </button>
            <button
              type="button"
              className="btn"
              onClick={() => {
                setText("");
                setRows([]);
              }}
              disabled={!text && !rows.length}
            >
              Clear
            </button>
            <span className="hint">Ctrl or Cmd + Enter</span>
          </div>
        </section>

        {rows.length > 0 && summary && (
          <section className="results" aria-labelledby="res-h" aria-live="polite">
            <div className="res-head">
              <h2 id="res-h">Results</h2>
              <button type="button" className="btn small" onClick={download}>
                Download CSV
              </button>
            </div>

            {rows.length > 1 && (
              <div className="summary">
                <div className="mix" role="img" aria-label={labels.map((l, i) => `${summary.counts[i]} ${l}`).join(", ")}>
                  {labels.map((l, i) => (summary.counts[i] ? <span key={l} className={`seg ${l}`} style={{ flexGrow: summary.counts[i] }} /> : null))}
                </div>
                <ul className="counts">
                  {labels.map((l, i) => (
                    <li key={l}>
                      <span className={`swatch ${l}`} aria-hidden="true" />
                      <strong>{summary.counts[i]}</strong> {l}
                    </li>
                  ))}
                  <li className="net">
                    Net tone{" "}
                    <strong>
                      {summary.net >= 0 ? "+" : ""}
                      {summary.net.toFixed(2)}
                    </strong>
                  </li>
                </ul>
              </div>
            )}

            <ol className="rows">
              {rows.map((r, i) => (
                <li key={`${i}-${r.headline}`} className="row">
                  <div className="row-top">
                    <p className="hl">{r.headline}</p>
                    <span className={`pill ${labels[r.pred.top]}`}>
                      {titleCase(labels[r.pred.top])} {pct(r.pred.probs[r.pred.top])}
                    </span>
                  </div>
                  <ul className="probs" aria-label="Probabilities">
                    {labels.map((l, k) => (
                      <li key={l} className={k === r.pred.top ? "top" : ""}>
                        <span className="p-name">{titleCase(l)}</span>
                        <span className="track" role="presentation">
                          <span className={`fill ${l}`} style={{ width: `${Math.max(0.6, r.pred.probs[k] * 100)}%` }} />
                        </span>
                        <span className="p-val">{pct(r.pred.probs[k])}</span>
                      </li>
                    ))}
                  </ul>
                </li>
              ))}
            </ol>
            {truncated && <p className="muted small-note">Showing the first {MAX_LINES} lines.</p>}
          </section>
        )}
      </main>
    </>
  );
}
