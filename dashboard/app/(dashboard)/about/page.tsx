import { Reveal } from "@/components/Reveal";

export default function AboutPage() {
  return (
    <div className="mx-auto flex max-w-2xl flex-col gap-5 font-serif text-[15px] leading-relaxed text-ink">
      <Reveal index={0}>
        <h1 className="font-mono text-lg font-semibold tracking-tight text-ink">About</h1>
        <p className="mt-1 font-mono text-xs uppercase tracking-widest text-ink-muted">
          Research ledger — project notes
        </p>
      </Reveal>

      <Reveal as="p" index={1}>
        This project set out to test whether sentiment and technical signals
        could predict short-horizon stock direction across a 28-stock
        large-cap universe spanning six sectors. Over five research phases,
        we trained and evaluated five different algorithms — logistic
        regression, LightGBM, a random forest, XGBoost, and a small LSTM —
        against walk-forward, out-of-sample splits. The finding, stated
        plainly because it is the load-bearing fact of the whole project: no
        configuration of any algorithm produced a validated edge over the
        simplest possible baseline, always guessing that the stock goes up.
        Confidence gating, cross-sectional features, sector-relative
        z-scores, alternate feature sets — none of it moved the needle in a
        way that survived honest evaluation.
      </Reveal>

      <Reveal as="p" index={2}>
        Midway through the project we found a real bug in our own evaluation
        code: confidence-filtered accuracy was being compared against the
        wrong baseline, which made some early results look better than they
        were. We fixed the bug and retroactively re-ran every earlier
        comparison under the corrected methodology rather than letting the
        original numbers stand. That correction is part of why we trust the
        final &ldquo;no edge&rdquo; conclusion — it survived us actively
        trying to find the mistake that would have overturned it.
      </Reveal>

      <Reveal as="p" index={3}>
        The paper-trading system this dashboard displays is a direct
        consequence of that finding, not a contradiction of it. Rather than
        shelve the work, the team built a small daily pipeline that runs the
        LightGBM model (10-day horizon, confidence-gated at 0.6) across the
        same 28-ticker universe, logs every prediction it makes, executes
        paper trades through a deliberately conservative decision rule, and
        evaluates its own hit rate against real baselines over time. Most
        days it trades nothing at all — that is the decision rule working as
        intended, not a malfunction. Nothing here is a claim that the model
        makes money. It exists so that if the model&apos;s behavior ever
        drifts — for better or worse — there is an honest, continuously
        running record of it, checked against reality instead of a backtest.
      </Reveal>
    </div>
  );
}
