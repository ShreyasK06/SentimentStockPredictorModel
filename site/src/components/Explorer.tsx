import { useMemo, useState } from "react";
import { dataUrl, type Phase7, type Phase8 } from "../lib/data";
import { ci, f3, int, MODEL_STYLE, pct, signed } from "../lib/format";
import { nameOf } from "../lib/models";
import { COVERAGES, covLabel, curveAt, modelRows, stageRows, type Metric } from "../lib/rows";
import { DotCI, DotLegend } from "../charts/DotCI";
import { LineChart, type LineSeries } from "../charts/LineChart";
import { Marker } from "../charts/Marker";
import { DataTable, Notice, Reveal, Segmented } from "./ui";

const pad = (lo: number, hi: number): [number, number] => [Math.floor((lo - 0.01) * 50) / 50, Math.ceil((hi + 0.01) * 50) / 50];
const ticksOf = (d: [number, number]) => {
  const out: number[] = [];
  for (let v = d[0]; v <= d[1] + 1e-9; v += 0.04) out.push(Math.round(v * 100) / 100);
  return out;
};

export function Explorer({ p8, p7 }: { p8: Phase8; p7: Phase7 }) {
  const [split, setSplit] = useState<"test" | "cv">("test");
  const [metric, setMetric] = useState<Metric>("both");
  const [cov, setCov] = useState(0.2);
  const [sel, setSel] = useState<string[]>(() => p8.test.models.map((m) => m.id));

  const S = split === "test" ? p8.test : p8.cv;
  const models = S.models.filter((m) => sel.includes(m.id));

  const toggle = (id: string) => setSel((cur) => (cur.includes(id) ? (cur.length > 1 ? cur.filter((x) => x !== id) : cur) : [...cur, id]));

  const rows = useMemo(() => modelRows(models, metric), [models, metric]);

  const curveSeries: LineSeries[] = models.map((m) => ({
    key: m.id,
    label: nameOf(m.id, m.label),
    color: MODEL_STYLE[m.id].color,
    shape: MODEL_STYLE[m.id].shape,
    emphasis: m.primary,
    points: COVERAGES.map((c) => {
      const p = curveAt(m, c);
      return { x: c, y: p ? p.accuracy : null, hollow: p ? p.n < 300 : false };
    }),
  }));
  const allY = curveSeries.flatMap((s) => s.points.map((p) => p.y).filter((y): y is number => y != null));
  const yd = pad(Math.min(...allY, S.majority_rate), Math.max(...allY, S.majority_rate));

  const foldSeries: LineSeries[] = models.map((m) => ({
    key: m.id,
    label: nameOf(m.id, m.label),
    color: MODEL_STYLE[m.id].color,
    shape: MODEL_STYLE[m.id].shape,
    emphasis: m.primary,
    points: (m.folds ?? []).map((f) => ({ x: f.year, y: f.unrestricted_acc })),
  }));
  const majFolds = (p8.cv.models[0].folds ?? []).map((f) => ({ x: f.year, y: f.majority }));
  const foldAll = [...foldSeries.flatMap((s) => s.points.map((p) => p.y as number)), ...majFolds.map((p) => p.y as number)];
  const fd = pad(Math.min(...foldAll), Math.max(...foldAll));
  const years = p8.cv.eval_years;

  return (
    <div className="explorer">
      <header className="ex-head">
        <h1 id="ex-title" tabIndex={-1}>
          Results explorer
        </h1>
        <p className="lead">
          Every measured number for the direction classifiers and the Phase 7 forecasters. Choose a split, choose models, and read each result next to the baseline it has to beat.
        </p>
        <Notice tone="strong">
          <strong>Research, not financial advice.</strong> No model here demonstrated an edge over the majority-class baseline.
        </Notice>
      </header>

      <div className="controls" role="region" aria-label="Chart controls">
        <div className="ctl">
          <span className="ctl-l">Split</span>
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
        <div className="ctl">
          <span className="ctl-l">Metric</span>
          <Segmented
            label="Metric shown in the accuracy chart"
            value={metric}
            onChange={setMetric}
            options={[
              { value: "both", label: "Both" },
              { value: "all", label: "Every day" },
              { value: "confident", label: "Confident calls" },
            ]}
          />
        </div>
        <div className="ctl models-ctl">
          <span className="ctl-l">Models</span>
          <ul className="chips" aria-label="Models to show">
            {S.models.map((m) => {
              const st = MODEL_STYLE[m.id];
              const on = sel.includes(m.id);
              return (
                <li key={m.id}>
                  <button type="button" className="chip" aria-pressed={on} onClick={() => toggle(m.id)}>
                    <svg width="18" height="18" aria-hidden="true">
                      <Marker shape={st.shape} x={9} y={9} r={4.5} color={on ? st.color : "var(--ink-3)"} hollow={!on} />
                    </svg>
                    {nameOf(m.id, m.label)}
                  </button>
                </li>
              );
            })}
          </ul>
        </div>
      </div>

      <Reveal className="card wide">
        <h2 className="h3">Accuracy against the majority-class rate</h2>
        <p className="card-sub">
          {split === "test"
            ? `Held-out test ${p8.test.window}, n=${int(p8.test.n)}. Trained on ${p8.test.train}, thresholds from ${p8.test.validation}.`
            : `Walk-forward CV, ${years[0]} to ${years[years.length - 1]} pooled, n=${int(p8.cv.n)}.`}
        </p>
        <DotLegend
          items={[
            ...(metric !== "confident" ? [{ kind: "solid" as const, text: "Every day" }] : []),
            ...(metric !== "all" ? [{ kind: "hollow" as const, text: "Confident calls only" }] : []),
            { kind: "line" as const, text: `Majority-class rate ${f3(S.majority_rate)}` },
          ]}
        />
        <DotCI
          rows={rows}
          domain={[0.44, 0.6]}
          ticks={[0.44, 0.48, 0.52, 0.56, 0.6]}
          baseline={{ value: S.majority_rate, label: `Majority-class rate ${f3(S.majority_rate)}` }}
          fmt={f3}
          axisTitle="Directional accuracy, 95% interval"
          ariaLabel={`Accuracy of ${models.length} selected models against a majority-class rate of ${f3(S.majority_rate)}`}
          desc={models.map((m) => `${nameOf(m.id, m.label)}: every day ${f3(m.unrestricted.acc)}, confident ${f3(m.confident.acc)}`).join(". ")}
        />
      </Reveal>

      <Reveal className="card wide">
        <h2 className="h3">Does confidence carry signal? Coverage against accuracy</h2>
        <p className="card-sub">
          Moving right, a model keeps only its most confident calls. If confidence meant something, accuracy would climb. Hollow marks have fewer than 300 calls and are noisy.
        </p>
        <DotLegend
          items={[
            { kind: "line", text: `Majority-class rate on all rows ${f3(S.majority_rate)}` },
            { kind: "hollow", text: "Hollow marks: under 300 calls" },
          ]}
        />
        <LineChart
          series={curveSeries}
          xScale="log"
          xDomain={[0.01, 1]}
          reverseX
          xTicks={[1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01].map((v) => ({ v, label: covLabel(v) }))}
          yDomain={yd}
          yTicks={ticksOf(yd)}
          baseline={{ value: S.majority_rate, label: "" }}
          fmtY={f3}
          selectedX={cov}
          xTitle="Share of days with a call (coverage); more selective to the right"
          yTitle="Accuracy of calls"
          ariaLabel={`Accuracy of calls as coverage shrinks, for ${models.length} selected models`}
          desc={models.map((m) => `${nameOf(m.id, m.label)}: ` + COVERAGES.map((c) => `${covLabel(c)} coverage ${f3(curveAt(m, c)?.accuracy)}`).join(", ")).join(". ")}
        />
        <div className="cov-pick">
          <span className="ctl-l">Read at coverage</span>
          <Segmented
            label="Coverage level for the readout"
            value={String(cov)}
            onChange={(v) => setCov(Number(v))}
            options={COVERAGES.map((c) => ({ value: String(c), label: covLabel(c) }))}
          />
        </div>
        <DataTable
          caption={`Readout at ${covLabel(cov)} coverage`}
          head={["Model", "Accuracy", "Calls", "Matched baseline", "Accuracy minus baseline"]}
          rows={models.map((m) => {
            const p = curveAt(m, cov);
            return [
              nameOf(m.id, m.label),
              p ? f3(p.accuracy) : "n/a",
              p ? int(p.n) : "n/a",
              p ? f3(p.local_majority) : "n/a",
              p && p.local_majority != null ? signed(p.accuracy - p.local_majority) : "n/a",
            ];
          })}
        />
        <p className="card-sub">
          The matched baseline is the majority-class rate among exactly the days the model called. Where the line falls rather than rises as coverage shrinks, confidence is not informative out of sample.
        </p>
      </Reveal>

      {split === "cv" && (
        <Reveal className="card wide">
          <h2 className="h3">Year by year</h2>
          <p className="card-sub">Each point is one walk-forward evaluation year, every day counted. The dashed line is the majority-class rate of that year. Models swap places from year to year; none stays above the line.</p>
          <LineChart
            series={[
              ...foldSeries,
              { key: "maj", label: "Majority-class rate", color: "var(--ink)", shape: "square", dash: "5 4", points: majFolds },
            ]}
            xScale="linear"
            xDomain={[years[0] - 0.4, years[years.length - 1] + 0.4]}
            xTicks={years.map((y) => ({ v: y, label: String(y) }))}
            yDomain={fd}
            yTicks={ticksOf(fd)}
            fmtY={f3}
            xTitle="Evaluation year"
            yTitle="Accuracy, every day"
            ariaLabel="Walk-forward accuracy by year for the selected models, with the majority-class rate of each year"
            desc={foldSeries.map((s) => `${s.label}: ` + s.points.map((p) => `${p.x} ${f3(p.y)}`).join(", ")).join(". ")}
          />
        </Reveal>
      )}

      <Reveal className="card wide">
        <h2 className="h3">All numbers</h2>
        <DataTable
          caption={`Every selected model on the ${split === "test" ? "held-out test" : "walk-forward CV"} split`}
          head={["Model", "Every day", "95% interval", "Confident calls", "95% interval", "Coverage", "Calls", "Matched baseline"]}
          rows={models.map((m) => [nameOf(m.id, m.label), f3(m.unrestricted.acc), ci(m.unrestricted.ci), f3(m.confident.acc), ci(m.confident.ci), pct(m.confident.coverage), int(m.confident.n), f3(m.confident.local_majority)])}
        />
        <p className="card-sub">
          Majority-class rate on all rows: {f3(S.majority_rate)}. Unrestricted rows: {int(S.models[0].unrestricted.n)}. Confident calls need coverage of at least {pct(p8.cv.min_coverage)} and {p8.cv.min_n} calls to count as feasible.
        </p>
      </Reveal>

      <Reveal className="card wide">
        <h2 className="h3">Phase 7: TFT multi-horizon stages</h2>
        <p className="card-sub">Mean over horizons 1, 3, 5 and 7 trading days, held-out {p7.test_window}. Difference from the majority-class baseline, with 95% block-bootstrap intervals where reported.</p>
        <DotCI
          rows={stageRows(p7.stages)}
          domain={[-0.09, 0.03]}
          ticks={[-0.08, -0.06, -0.04, -0.02, 0, 0.02]}
          baseline={{ value: 0, label: "Equals majority-class baseline" }}
          fmt={(n) => signed(n, 3)}
          axisTitle="Accuracy minus majority-class rate"
          ariaLabel="Phase 7 stages: held-out accuracy minus the majority-class baseline"
          desc={p7.stages.map((s) => `${s.stage}: ${signed(s.test_minus_maj)}`).join(". ")}
        />
        <DataTable
          caption="Phase 7 staged results"
          head={["Stage", "CV acc", "CV baseline", "Test acc", "Test baseline", "Test minus baseline", "95% interval"]}
          rows={p7.stages.map((s) => [s.stage, f3(s.cv_acc), f3(s.cv_maj), f3(s.test_acc), f3(s.test_maj), signed(s.test_minus_maj), ci(s.diff_ci, (n) => signed(n))])}
        />
        <p className="card-sub">Stage e keeps only confident predictions, so its baseline is the majority rate within those rows. {p7.summary}</p>
      </Reveal>

      <Reveal className="card wide dl">
        <h2 className="h3">Get the data</h2>
        <p className="card-sub">
          The same JSON this page renders. Regenerate it with <code>python scripts/export_site_data.py</code>.
        </p>
        <ul className="plain cols">
          {["phase8.json", "phase7.json", "phase9.json", "examples.json", "meta.json"].map((n) => (
            <li key={n}>
              <a href={dataUrl(n)} download>
                {n}
              </a>
            </li>
          ))}
        </ul>
      </Reveal>
    </div>
  );
}
