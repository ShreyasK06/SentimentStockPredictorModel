import { useState } from "react";
import type { Examples } from "../lib/data";
import { Reveal, SectionHead } from "./ui";

const CLASSES = [
  { key: "positive", label: "Positive" },
  { key: "negative", label: "Negative" },
  { key: "neutral", label: "Neutral" },
] as const;

function Probs({ title, kind, s }: { title: string; kind: string; s: { positive: number; negative: number; neutral: number } }) {
  const top = CLASSES.reduce((a, b) => (s[b.key] > s[a.key] ? b : a));
  return (
    <div className="probs" role="group" aria-label={`${title} probabilities`}>
      <div className="probs-head">
        <span className="probs-title">{title}</span>
        <span className="tag">{kind}</span>
      </div>
      <ul>
        {CLASSES.map((c) => (
          <li key={c.key} className={c.key === top.key ? "top" : ""}>
            <span className="p-label">{c.label}</span>
            <span className="p-track" role="presentation">
              <span className="p-fill" style={{ width: `${Math.max(0.5, s[c.key] * 100)}%` }} />
            </span>
            <span className="p-val">{(s[c.key] * 100).toFixed(1)}%</span>
          </li>
        ))}
      </ul>
      <p className="p-verdict">
        Reads as <strong>{top.label.toLowerCase()}</strong> ({(s[top.key] * 100).toFixed(0)}%). That is a statement about the words, not a price call.
      </p>
    </div>
  );
}

export function Demo({ ex }: { ex: Examples }) {
  const [sel, setSel] = useState(1);
  const cur = ex.examples.find((e) => e.id === sel) ?? ex.examples[0];
  return (
    <section id="demo-sec" className="sec" aria-labelledby="demo">
      <SectionHead
        id="demo"
        title="Pick a headline, read its sentiment"
        lead="This is a precomputed gallery, not a live model. The scores below were produced offline by the models named on each card, so nothing you do here is sent anywhere or computed in your browser."
      />
      <div className="demo">
        <Reveal className="demo-list-wrap">
          <h3 id="demo-list-h" className="sr-only">
            Example headlines
          </h3>
          <ul className="demo-list" aria-labelledby="demo-list-h">
            {ex.examples.map((e) => (
              <li key={e.id}>
                <button type="button" aria-pressed={e.id === cur.id} onClick={() => setSel(e.id)}>
                  <span className="tkr">{e.ticker}</span>
                  <span className="hl-text">{e.headline}</span>
                </button>
              </li>
            ))}
          </ul>
        </Reveal>
        <Reveal className="demo-panel" delay={80}>
          <blockquote className="demo-headline" aria-live="polite">
            <span className="tkr">{cur.ticker}</span>
            {cur.headline}
          </blockquote>
          <div className={`probs-grid n${ex.models.length}`}>
            {ex.models.map((m) => (
              <Probs key={m.id} title={m.label} kind={m.kind === "fine-tuned" ? "Fine-tuned" : "Baseline"} s={cur.scores[m.id]} />
            ))}
          </div>
          {ex.models.length === 1 && (
            <p className="card-sub">
              Only the released FinBERT has scored these so far. A fine-tuned column appears here once the Phase 9 model exists and the data export is re-run.
            </p>
          )}
          <p className="card-sub"><strong>Precomputed, not live.</strong> {ex.note} Texts are cut at {ex.max_length} tokens. To score your own text, edit the headline list in <code>scripts/export_site_data.py</code> and re-run it.</p>
        </Reveal>
      </div>
    </section>
  );
}
