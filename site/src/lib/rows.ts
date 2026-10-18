import type { DotRow } from "../charts/DotCI";
import type { CurvePt, ModelRes, Stage } from "./data";
import { MODEL_STYLE } from "./format";
import { nameOf } from "./models";

export type Metric = "both" | "all" | "confident";

export function modelRows(models: ModelRes[], metric: Metric = "both"): DotRow[] {
  return models.map((m) => {
    const marks: DotRow["marks"] = [];
    if (metric !== "confident") marks.push({ kind: "solid", value: m.unrestricted.acc, ci: m.unrestricted.ci });
    if (metric !== "all") marks.push({ kind: "hollow", value: m.confident.acc, ci: m.confident.ci });
    return {
      key: m.id,
      label: nameOf(m.id, m.label),
      sub: m.primary ? "primary model" : undefined,
      color: MODEL_STYLE[m.id]?.color ?? "var(--s1)",
      emphasis: m.primary,
      marks,
    };
  });
}

export function stageRows(stages: Stage[]): DotRow[] {
  return stages.map((s, i) => ({
    key: s.stage,
    label: s.stage.startsWith("e.") ? "Confident calls only (threshold)" : s.stage.replace(/^[a-e]\. /, ""),
    sub: s.stage.startsWith("e.") ? "confident subset vs its own baseline" : undefined,
    color: i === 5 ? "var(--s2)" : "var(--s1)",
    marks: [{ kind: "solid", value: s.test_minus_maj, ci: s.diff_ci }],
  }));
}

/** Snap a realised coverage to its nominal target so curves from different models share x positions. */
const TARGETS = [1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01];
export function snapCov(c: number): number {
  return TARGETS.reduce((a, b) => (Math.abs(Math.log(b / c)) < Math.abs(Math.log(a / c)) ? b : a));
}
export const COVERAGES = TARGETS;
export const covLabel = (c: number) => `${Math.round(c * 100)}%`;
export const curveAt = (m: ModelRes, cov: number): CurvePt | undefined => m.curve.find((p) => snapCov(p.coverage) === cov);
