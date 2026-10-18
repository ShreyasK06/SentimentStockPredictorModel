import type { Phase8 } from "../lib/data";
import { f3, pct } from "../lib/format";
import { Reveal } from "./ui";

export function Learned({ p8 }: { p8: Phase8 }) {
  const acn = p8.test.models.find((m) => m.id === "a_logit");
  const guards = [
    { t: "Align", d: "Articles dated D are used for the first trading session after D, and the label starts after that. Same-day reaction is deliberately thrown away rather than risk peeking." },
    { t: "Fence", d: "Feature code is unit-tested by perturbing future prices and future articles and asserting that today's features do not change. A guard rejects any column that could encode the target." },
    { t: "Lock", d: "Hyperparameters and confidence thresholds are chosen on earlier years. The test window is run once, recorded in a lock file, and never re-run to improve a number." },
    { t: "Probe", d: `A label-permutation check and a deliberately naive-clock variant look for timing leaks. Any result above 70% would raise an alarm; none did, and the naive variant was no better (${f3(p8.diagnostics.naive_alignment_acc)}).` },
  ];
  return (
    <section id="learned-sec" className="sec band" aria-labelledby="learned">
      <div className="band-in">
        <header className="sec-head">
          <h2 id="learned">Why next-day direction is close to a coin flip</h2>
          <p className="lead">None of this is surprising in hindsight. It is the reason the project reports baselines and intervals before it reports anything else.</p>
        </header>

        <ol className="lessons">
          <Reveal as="li">
            <h3>The baseline is {pct(p8.test.majority_rate)}, not 50%</h3>
            <p>
              Prices drifted up across the sample, so a model that says UP most of the time looks decent. On the test set the logistic model&rsquo;s confident calls were {acn?.confident.pred_up_share != null ? pct(acn.confident.pred_up_share) : "mostly"} UP, which mostly encodes the drift. Accuracy only means something next to the majority-class rate.
            </p>
          </Reveal>
          <Reveal as="li" delay={70}>
            <h3>Headlines are public to everyone at once</h3>
            <p>
              A sentiment score summarises information that traders already acted on. Our date-only timestamps also force a conservative next-session rule, which discards same-day reactions. What remains is a weak signal under a lot of noise.
            </p>
          </Reveal>
          <Reveal as="li" delay={140}>
            <h3>Selection manufactures phantom skill</h3>
            <p>
              Single folds reached about 0.62 on roughly 5% of calls, then vanished when pooled. Picking thresholds on a validation year rewards noise, which is why every confident-call number sits beside its matched baseline and an interval.
            </p>
          </Reveal>
          <Reveal as="li" delay={210}>
            <h3>An ablation needs the data it ablates</h3>
            <p>
              In the Phase 7 panel, sentiment inputs are exactly zero before November 2025 for the 16-ticker subset. The stage with sentiment therefore differs from the one without it only by training noise of about three points. We say so rather than read it as an effect.
            </p>
          </Reveal>
        </ol>

        <div className="guards">
          <h3>How the leakage controls work</h3>
          <ol>
            {guards.map((g, i) => (
              <Reveal as="li" key={g.t} delay={i * 60}>
                <strong>{g.t}</strong>
                <span>{g.d}</span>
              </Reveal>
            ))}
          </ol>
        </div>
        <p className="band-note">
          An early goal of roughly 85% accuracy on confident calls was not reached. Nothing was relaxed, re-run or hand-picked to approach it, and the pre-declared minimum of 5% coverage and 300 calls was kept as set.
        </p>
      </div>
    </section>
  );
}
