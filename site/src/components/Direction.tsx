import { useMemo, useState } from "react";
import type { Phase7, Phase8 } from "../lib/data";
import { ci, f3, int, pct, signed } from "../lib/format";
import { nameOf } from "../lib/models";
import { modelRows, stageRows } from "../lib/rows";
import { DotCI, DotLegend } from "../charts/DotCI";
import { DataTable, Notice, Reveal, SectionHead, Segmented } from "./ui";

export function Direction({ p8, p7 }: { p8: Phase8; p7: Phase7 }) {
  const [split, setSplit] = useState<"test" | "cv">("test");
  const S = split === "test" ? p8.test : p8.cv;
  const rows = useMemo(() => modelRows(S.models, "both"), [S]);

  const T = p8.test.models;
  const un = T.map((m) => m.unrestricted.acc as number);
  const co = T.map((m) => m.confident.acc as number);
  const cov = T.map((m) => m.confident.coverage);
  const clear = T.filter((m) => m.confident.ci && m.confident.local_majority != null && m.confident.ci[0] > (m.confident.local_majority as number)).length;
  const below = p7.stages.filter((s) => (s.test_minus_maj ?? 0) < 0).length;

  return (
    <section id="direction-sec" className="sec" aria-labelledby="direction">
      <SectionHead
        id="direction"
        title="Predicting tomorrow: no measurable edge"
        lead="The headline number to beat is not 50%. Stocks drifted up in the sample, so always guessing UP is right about 52% of the time. A model has to clear that, and its error bars, on data it never saw."
      />

      <Reveal className="stats">
        <div className="stat">
          <div className="stat-n">
            {f3(Math.min(...un))} to {f3(Math.max(...un))}
          </div>
          <div className="stat-l">Accuracy on every day, six models, versus a {f3(p8.test.majority_rate)} majority-class rate.</div>
        </div>
        <div className="stat">
          <div className="stat-n">
            {f3(Math.min(...co))} to {f3(Math.max(...co))}
          </div>
          <div className="stat-l">
            Accuracy when a model chooses to call, at {pct(Math.min(...cov))} to {pct(Math.max(...cov))} coverage.
          </div>
        </div>
        <div className="stat">
          <div className="stat-n">
            {clear} of {T.length}
          </div>
          <div className="stat-l">Models whose confident-call interval clears its own matched baseline.</div>
        </div>
      </Reveal>

      <Reveal className="card wide">
        <div className="card-top">
          <div>
            <h3>Accuracy against the majority-class rate</h3>
            <p className="card-sub">
              {split === "test"
                ? `Held-out test ${p8.test.window}, n=${int(p8.test.n)}, evaluated once. Training ${p8.test.train}, thresholds from ${p8.test.validation}.`
                : `Walk-forward CV, evaluation years ${p8.cv.eval_years[0]} to ${p8.cv.eval_years[p8.cv.eval_years.length - 1]} pooled, n=${int(p8.cv.n)}. Each year is predicted by a model trained on earlier years only.`}
            </p>
          </div>
          <Segmented
            label="Evaluation split"
            value={split}
            onChange={setSplit}
            options={[
              { value: "test", label: "Held-out test" },
              { value: "cv", label: "Walk-forward CV" },
            ]}
          />
        </div>
        <DotLegend
          items={[
            { kind: "solid", text: "Every day (with 95% interval)" },
            { kind: "hollow", text: "Confident calls only (with 95% interval)" },
            { kind: "line", text: `Majority-class rate ${f3(S.majority_rate)}` },
          ]}
        />
        <DotCI
          rows={rows}
          domain={[0.44, 0.6]}
          ticks={[0.44, 0.48, 0.52, 0.56, 0.6]}
          baseline={{ value: S.majority_rate, label: `Majority-class rate ${f3(S.majority_rate)}` }}
          fmt={f3}
          axisTitle="Directional accuracy"
          ariaLabel={`Accuracy of six models on the ${split === "test" ? "held-out test" : "walk-forward cross-validation"} split against a majority-class rate of ${f3(S.majority_rate)}`}
          desc={S.models.map((m) => `${nameOf(m.id, m.label)}: every day ${f3(m.unrestricted.acc)}, confident calls ${f3(m.confident.acc)}`).join(". ")}
        />
        <p className="card-sub">
          Confident calls are judged against the majority rate <em>within the same rows</em>, which differs by model. The table shows each matched baseline. Single folds sometimes look strong; pooled, they do not.
        </p>
        <details className="tbl-toggle">
          <summary>View as a table</summary>
          <DataTable
            caption={`Accuracy by model, ${split === "test" ? "held-out test" : "walk-forward CV"}`}
            head={["Model", "Every day", "95% interval", "Confident calls", "95% interval", "Coverage", "Matched baseline"]}
            rows={S.models.map((m) => [nameOf(m.id, m.label), f3(m.unrestricted.acc), ci(m.unrestricted.ci), f3(m.confident.acc), ci(m.confident.ci), pct(m.confident.coverage), f3(m.confident.local_majority)])}
          />
        </details>
        <p className="more">
          <a href="#/results">Open the results explorer</a> for coverage curves, per-year folds and every number.
        </p>
      </Reveal>

      <div className="split-note">
        <Reveal>
          <h3>A deeper model did not help</h3>
          <p>
            Phase 7 swapped the classifiers for a Temporal Fusion Transformer forecasting 1, 3, 5 and 7 trading days ahead on {p7.universe}. On the held-out {p7.test_window}, {below} of {p7.stages.length} configurations finished below the majority-class baseline, including the version with FinBERT sentiment. The tuned model is not distinguishable from the untuned one.
          </p>
        </Reveal>
        <Reveal delay={80}>
          <Notice>
            <strong>Reading the chart.</strong> Zero means &ldquo;as good as always guessing the majority class&rdquo;. Dots left of zero are worse than that. Whiskers are 95% block-bootstrap intervals; a whisker that crosses zero is inconclusive, not a win.
          </Notice>
        </Reveal>
      </div>

      <Reveal className="card wide">
        <h3>Phase 7 TFT stages: held-out accuracy minus the baseline</h3>
        <p className="card-sub">Mean over four horizons, test window {p7.test_window}. Stage e is the confident subset only, so it has no interval here and is judged against its own rows.</p>
        <DotCI
          rows={stageRows(p7.stages)}
          domain={[-0.09, 0.03]}
          ticks={[-0.08, -0.06, -0.04, -0.02, 0, 0.02]}
          baseline={{ value: 0, label: "Equals majority-class baseline" }}
          fmt={(n) => signed(n, 3)}
          axisTitle="Accuracy minus majority-class rate"
          ariaLabel="Phase 7 temporal fusion transformer stages: held-out accuracy minus the majority-class baseline. Every stage is below zero."
          desc={p7.stages.map((s) => `${s.stage}: ${signed(s.test_minus_maj)}`).join(". ")}
        />
        <details className="tbl-toggle">
          <summary>View as a table</summary>
          <DataTable
            caption="Phase 7 staged results, mean over horizons 1, 3, 5 and 7"
            head={["Stage", "CV acc", "CV baseline", "Test acc", "Test baseline", "Test minus baseline", "95% interval of difference"]}
            rows={p7.stages.map((s) => [s.stage, f3(s.cv_acc), f3(s.cv_maj), f3(s.test_acc), f3(s.test_maj), signed(s.test_minus_maj), ci(s.diff_ci, (n) => signed(n))])}
          />
        </details>
      </Reveal>
    </section>
  );
}
