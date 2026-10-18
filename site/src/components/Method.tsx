import type { Phase8, Phase9 } from "../lib/data";
import { doc } from "../lib/data";
import { int } from "../lib/format";
import { readPhase9 } from "../lib/phase9";
import { DataTable, Reveal, SectionHead } from "./ui";

const DOCS = [
  { href: "docs/phase8_results.md", t: "Phase 8 results", d: "News-sentiment next-day direction classifier with selective prediction." },
  { href: "docs/phase7_tft_results.md", t: "Phase 7 results", d: "FinBERT plus Temporal Fusion Transformer, staged evaluation." },
  { href: "docs/phase9_finetune_results.md", t: "Phase 9 results", d: "Sentiment fine-tuning. Written by the fine-tuning job." },
  { href: "docs/phase5_cross_sectional_results.md", t: "Phase 5 results", d: "Cross-sectional baselines." },
  { href: "docs/phase4_lstm_results.md", t: "Phase 4 results", d: "LSTM experiments." },
  { href: "README.md", t: "Repository README", d: "Setup, layout and how to reproduce each phase." },
];

export function Method({ p8, p9 }: { p8: Phase8; p9: Phase9 }) {
  const v9 = readPhase9(p9.raw);
  const s9 = (k: string) => v9.facts.find((f) => f.label.toLowerCase().startsWith(k))?.value ?? "run pending";
  return (
    <section id="method-sec" className="sec" aria-labelledby="method">
      <SectionHead id="method" title="Method and data" lead="Every number on this site comes from one of these splits. Each table names the split so a figure can never be quoted without it." />

      <Reveal className="card wide">
        <h3>Splits</h3>
        <DataTable
          caption="Training, validation and test windows by phase"
          head={["Phase and split", "Window", "Used for"]}
          rows={[
            ["Phase 8 development", `${p8.cv.dev_start} to 2021`, "Training and selection. Start fixed by a 20% news-coverage rule."],
            ["Phase 8 walk-forward CV", `Evaluation years ${p8.cv.eval_years[0]} to ${p8.cv.eval_years[p8.cv.eval_years.length - 1]}`, "Each year predicted from earlier years only; the prior year is the validation set."],
            ["Phase 8 final validation", p8.test.validation, "Sets the confidence threshold for the test run."],
            ["Phase 8 held-out test", `${p8.test.window}, n=${int(p8.test.n)}`, "Evaluated once and locked (TEST_LOCK.json)."],
            ["Phase 7 CV folds F1, F2, F3", "Validation 2021, 2022, 2023", "Expanding window. F3 is the selection fold for tuning and thresholds."],
            ["Phase 7 held-out test", "2024-01-01 to 2025-12-31", "Model trained through 2022, early-stopped on 2023, run once per stage."],
            ["Phase 9 train", s9("train"), "Fine-tuning."],
            ["Phase 9 validation", s9("validation"), "Checkpoint selection."],
            ["Phase 9 test", s9("test"), "Reported accuracy, confusion matrix and F1."],
          ]}
        />
      </Reveal>

      <div className="grid-2 method-grid">
        <Reveal className="card">
          <h3>Leakage controls</h3>
          <ul className="plain">
            <li>Features at day t use only data available at the close of t; news after 16:00 ET moves to the next session.</li>
            <li>Training rows satisfy t + h &le; train end, so no label window overlaps evaluation.</li>
            <li>Unit tests perturb future prices and articles and assert today&rsquo;s features are unchanged.</li>
            <li>Thresholds and hyperparameters use validation or CV only. Test files are run once and locked.</li>
            <li>A permutation check and a naive-clock variant probe for timing leaks, with a 70% alarm.</li>
            <li>Intervals are block bootstraps over 20-day blocks, because overlapping horizons and correlated tickers make binomial intervals far too narrow.</li>
          </ul>
        </Reveal>
        <Reveal className="card" delay={80}>
          <h3>Datasets</h3>
          <ul className="plain">
            <li>
              <strong>FNSPID</strong> (Hugging Face): 162,704 de-duplicated articles for 28 tickers, stamped 2009-06-22 to 2023-12-16. 99.8% carry a date only.
            </li>
            <li>
              <strong>Finnhub</strong> company news from August 2025, for live inference and the Phase 7 sentiment window. Not used to train or score Phase 8.
            </li>
            <li>
              <strong>Price panel</strong> for Phase 7: 28 tickers, 2019-12-31 to 2025-12-31, with a fixed 16-ticker subset (two largest per sector).
            </li>
            <li>
              <strong>Phase 9 sentiment data</strong>: {v9.facts.find((f) => f.label === "Datasets")?.value ?? "run pending"}.
            </li>
          </ul>
        </Reveal>
        <Reveal className="card wide-md" delay={120}>
          <h3>Limitations</h3>
          <ul className="plain cols">
            <li>Source shift: 2021 to 2023 news comes only from the Nasdaq file, while earlier years mix two files.</li>
            <li>Date-only timestamps force the conservative next-session rule and discard same-day reaction.</li>
            <li>Single seed, 28 large-cap tickers, daily headlines only, no intraday data.</li>
            <li>Phase 7 uses a 16-ticker subset and CPU-sized training, one seed.</li>
            <li>Accuracy is not a trading result: no costs, slippage or position sizing are modelled.</li>
            <li>Results describe the past windows above and say nothing about future returns.</li>
          </ul>
        </Reveal>
      </div>

      <Reveal>
        <h3 className="docs-h">Read the full write-ups</h3>
        <ul className="docs">
          {DOCS.map((d) => (
            <li key={d.href}>
              <a href={doc(d.href)} target="_blank" rel="noreferrer">
                <strong>{d.t}</strong>
                <span>{d.d}</span>
                <span className="sr-only"> (opens in a new tab)</span>
              </a>
            </li>
          ))}
        </ul>
      </Reveal>
    </section>
  );
}
