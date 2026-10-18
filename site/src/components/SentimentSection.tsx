import type { Phase9 } from "../lib/data";
import { f3, signed } from "../lib/format";
import { readPhase9 } from "../lib/phase9";
import { Bars } from "../charts/Bars";
import { Confusion } from "../charts/Confusion";
import { Reveal, SectionHead } from "./ui";

export function SentimentSection({ p9 }: { p9: Phase9 }) {
  const v = readPhase9(p9.raw);
  const baselines = v.bars.filter((b) => !b.highlight && b.value != null) as { value: number; label: string }[];
  const best = baselines.length ? baselines.reduce((a, b) => (b.value > a.value ? b : a)) : null;
  const delta = v.testAcc != null && best ? v.testAcc - best.value : null;
  const f1Bars = (v.perClass.length ? v.perClass : ["negative", "neutral", "positive"].map((l) => ({ label: l, f1: null, precision: null, recall: null, support: null }))).map((c) => ({
    key: c.label,
    label: c.label[0].toUpperCase() + c.label.slice(1),
    note: c.support != null ? `${c.support.toLocaleString("en-US")} examples` : undefined,
    value: c.f1,
  }));

  return (
    <section id="sentiment-sec" className="sec" aria-labelledby="sentiment">
      <SectionHead
        id="sentiment"
        title="Can fine-tuning make FinBERT read better?"
        lead="Everything downstream consumes sentiment, so Phase 9 asks the narrower question first: on labelled financial text, does fine-tuning beat the untouched model and the dull baselines, on data it has never seen?"
      />

      {!v.ready ? (
        <Reveal>
          <div className="status pending" role="status">
            <span className="status-dot" aria-hidden="true" />
            <div>
              <strong>Run pending.</strong> The fine-tuning job has not written its numbers yet. The panels below keep their layout and fill in on their own once <code>reports/phase9/finetune_results.json</code> contains a test accuracy. Until then, nothing here is a result.
            </div>
          </div>
        </Reveal>
      ) : (
        <Reveal>
          <div className="status ready" role="status">
            <span className="status-dot" aria-hidden="true" />
            <div>
              <strong>Measured.</strong>{" "}
              {delta != null && best ? (
                <>
                  Fine-tuned test accuracy is {f3(v.testAcc)}, {signed(delta)} versus the strongest baseline ({best.label}, {f3(best.value)}).
                </>
              ) : (
                <>Fine-tuned test accuracy is {f3(v.testAcc)}.</>
              )}{" "}
              {v.cv ? <>Cross-validation mean {f3(v.cv.mean)}{v.cv.std != null ? ` (sd ${f3(v.cv.std)})` : ""}. </> : null}
              These numbers come from different splits, labelled below.
            </div>
          </div>
        </Reveal>
      )}

      <div className="grid-2 sent-grid">
        <Reveal className="card">
          <h3>Accuracy on the held-out test split</h3>
          <p className="card-sub">Bars start at zero. The line on the fine-tuned bar is its 95% interval, when reported.</p>
          <Bars bars={v.bars} ariaLabel="Sentiment accuracy by model on the held-out test split" />
          {v.caveat && <p className="caveat"><strong>Caveat.</strong> {v.caveat}</p>}
          {v.valAcc != null && <p className="card-sub">Validation accuracy (used to pick the checkpoint): {f3(v.valAcc)}.</p>}
          {v.crossDataset.length > 0 && (
            <p className="card-sub">
              Cross-dataset accuracy: {v.crossDataset.map((c) => `${c.label} ${f3(c.value)}`).join("; ")}.
            </p>
          )}
        </Reveal>

        <Reveal className="card" delay={80}>
          <h3>Where it is right and wrong</h3>
          <p className="card-sub">Rows are the true label. Each cell prints its share of the row and its count.</p>
          <Confusion labels={v.confusion?.labels ?? null} matrix={v.confusion?.matrix ?? null} pending={!v.confusion} />
          <h3 className="sub-h">F1 by class</h3>
          <Bars bars={f1Bars} ariaLabel="F1 score by sentiment class" />
        </Reveal>
      </div>

      {(v.facts.length > 0 || v.notes.length > 0) && (
        <Reveal className="facts">
          <dl>
            {v.facts.map((f) => (
              <div key={f.label}>
                <dt>{f.label}</dt>
                <dd>{f.value}</dd>
              </div>
            ))}
          </dl>
          {v.notes.map((n) => (
            <p key={n} className="card-sub">{n}</p>
          ))}
        </Reveal>
      )}
    </section>
  );
}
