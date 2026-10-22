import { useState } from "react";
import type { Prediction, SentimentModel } from "../lib/model";

const START = [
  "Microsoft cloud growth slows for a third straight quarter",
  "Boeing wins large aircraft order from Asian carrier",
  "Walmart cuts annual outlook citing weaker discretionary spending",
].join("\n");

type Status = { kind: "idle" } | { kind: "loading"; progress: number } | { kind: "ready" } | { kind: "error"; message: string };

/**
 * Score any headline with the same fine-tuned sentiment model the dashboard uses, in the browser.
 * The 67 MB model is only downloaded when the visitor asks for it, and the runtime is imported lazily.
 */
export function HeadlineLab() {
  const [status, setStatus] = useState<Status>({ kind: "idle" });
  const [model, setModel] = useState<SentimentModel | null>(null);
  const [text, setText] = useState(START);
  const [rows, setRows] = useState<{ headline: string; pred: Prediction }[]>([]);
  const [busy, setBusy] = useState(false);

  const load = async () => {
    setStatus({ kind: "loading", progress: 0 });
    try {
      const { loadModel } = await import("../lib/model");
      const m = await loadModel((progress) => setStatus({ kind: "loading", progress }));
      setModel(m);
      setStatus({ kind: "ready" });
    } catch (err) {
      setStatus({ kind: "error", message: err instanceof Error ? err.message : String(err) });
    }
  };

  const score = async () => {
    if (!model) return;
    const headlines = text.split(/\r?\n/).map((l) => l.trim()).filter(Boolean).slice(0, 50);
    if (!headlines.length) return setRows([]);
    setBusy(true);
    try {
      await new Promise((resolve) => setTimeout(resolve, 20));
      const preds = await model.classify(headlines);
      setRows(headlines.map((headline, i) => ({ headline, pred: preds[i] })));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="lab">
      {status.kind !== "ready" ? (
        <div className="lab-load">
          <p className="muted">Runs the sentiment model in your browser. Nothing you type is sent anywhere.</p>
          <button type="button" className="btn primary" disabled={status.kind === "loading"} onClick={load}>
            {status.kind === "loading" ? (status.progress < 0 ? "Loading model..." : `Loading model ${Math.round(status.progress * 100)}%`) : "Load the model (67 MB)"}
          </button>
          {status.kind === "error" && <p role="alert">{status.message}</p>}
        </div>
      ) : (
        <>
          <label htmlFor="lab-text" className="sr-only">
            Headlines, one per line
          </label>
          <textarea id="lab-text" rows={4} value={text} spellCheck={false} onChange={(e) => setText(e.target.value)} />
          <button type="button" className="btn primary" disabled={busy} onClick={score}>
            {busy ? "Scoring..." : "Score"}
          </button>
          {rows.length > 0 && (
            <ul className="lab-rows">
              {rows.map((r, i) => (
                <li key={`${i}-${r.headline}`}>
                  <span className={`pill ${model!.labels[r.pred.top]}`}>
                    {model!.labels[r.pred.top]} {(r.pred.probs[r.pred.top] * 100).toFixed(0)}%
                  </span>
                  <span>{r.headline}</span>
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </div>
  );
}
