/**
 * Tolerant reader for reports/phase9/finetune_results.json.
 * The file starts as a stub of nulls and is filled by the fine-tuning job. Every field may be
 * null/missing; the UI shows a "run pending" state for anything that is not a number yet.
 */
import type { Range } from "./data";

export interface BarItem {
  key: string;
  label: string;
  note?: string;
  value: number | null;
  ci?: Range | null;
  highlight?: boolean;
}
export interface ClassRow {
  label: string;
  f1: number | null;
  precision: number | null;
  recall: number | null;
  support: number | null;
}
export interface Phase9View {
  ready: boolean;
  bars: BarItem[];
  caveat: string | null;
  model: string | null;
  testAcc: number | null;
  valAcc: number | null;
  cv: { mean: number; std: number | null } | null;
  crossDataset: { label: string; value: number }[];
  perClass: ClassRow[];
  confusion: { labels: string[]; matrix: number[][] } | null;
  facts: { label: string; value: string }[];
  notes: string[];
}

type Obj = Record<string, unknown>;
const isObj = (x: unknown): x is Obj => typeof x === "object" && x !== null && !Array.isArray(x);

/** Accept 0..1 fractions or 0..100 percentages. */
export function num(x: unknown): number | null {
  if (typeof x === "string" && x.trim() !== "" && !Number.isNaN(Number(x))) x = Number(x);
  if (typeof x !== "number" || !Number.isFinite(x)) return null;
  return x > 1.0001 && x <= 100 ? x / 100 : x;
}
const rawNum = (x: unknown): number | null => (typeof x === "number" && Number.isFinite(x) ? x : null);
const rng = (x: unknown): Range | null => {
  if (Array.isArray(x) && x.length === 2) {
    const a = num(x[0]);
    const b = num(x[1]);
    if (a != null && b != null) return [a, b];
  }
  return null;
};
const accOf = (x: unknown): number | null => (isObj(x) ? num(x.acc ?? x.accuracy) : num(x));

const CLASS_ORDER = ["negative", "neutral", "positive"];

function normPerClass(x: unknown): ClassRow[] {
  if (!x) return [];
  const rows: ClassRow[] = [];
  if (Array.isArray(x)) {
    for (const it of x) {
      if (!isObj(it)) continue;
      rows.push({
        label: String(it.label ?? it.class ?? it.name ?? "?"),
        f1: num(it.f1 ?? it["f1-score"] ?? it.f1_score),
        precision: num(it.precision),
        recall: num(it.recall),
        support: rawNum(it.support ?? it.n),
      });
    }
  } else if (isObj(x)) {
    const keys = Object.keys(x);
    if (keys.includes("f1") && isObj(x.f1)) {
      const f1 = x.f1 as Obj;
      for (const c of Object.keys(f1)) {
        const g = (m: string) => (isObj(x[m]) ? num((x[m] as Obj)[c]) : null);
        rows.push({ label: c, f1: num(f1[c]), precision: g("precision"), recall: g("recall"), support: isObj(x.support) ? rawNum((x.support as Obj)[c]) : null });
      }
    } else {
      for (const c of keys) {
        const v = x[c];
        if (isObj(v)) {
          rows.push({
            label: c,
            f1: num(v.f1 ?? v["f1-score"] ?? v.f1_score),
            precision: num(v.precision),
            recall: num(v.recall),
            support: rawNum(v.support ?? v.n),
          });
        } else rows.push({ label: c, f1: num(v), precision: null, recall: null, support: null });
      }
    }
  }
  const rank = (l: string) => {
    const i = CLASS_ORDER.indexOf(l.toLowerCase());
    return i < 0 ? 9 : i;
  };
  return rows.filter((r) => !/avg|accuracy/i.test(r.label)).sort((a, b) => rank(a.label) - rank(b.label));
}

function normConfusion(x: unknown, classNames: string[]): { labels: string[]; matrix: number[][] } | null {
  let matrix: unknown = x;
  let labels: string[] | null = null;
  if (isObj(x)) {
    matrix = x.matrix ?? x.values ?? x.data;
    if (Array.isArray(x.labels)) labels = x.labels.map(String);
  }
  if (!Array.isArray(matrix) || matrix.length === 0) return null;
  const m = matrix.map((r) => (Array.isArray(r) ? r.map((v) => rawNum(v) ?? NaN) : []));
  if (m.some((r) => r.length !== m.length || r.some((v) => Number.isNaN(v)))) return null;
  if (!labels || labels.length !== m.length) {
    labels = classNames.length === m.length ? classNames : m.length === 3 ? CLASS_ORDER : m.map((_, i) => `class ${i}`);
  }
  return { labels, matrix: m };
}

function factText(v: unknown): string | null {
  if (v == null) return null;
  if (typeof v === "string" || typeof v === "number") return String(v);
  if (Array.isArray(v)) return v.map((i) => (typeof i === "string" ? i : isObj(i) ? String(i.name ?? i.id ?? JSON.stringify(i)) : String(i))).join(", ");
  if (isObj(v)) return Object.entries(v).map(([k, val]) => `${k}: ${typeof val === "object" ? JSON.stringify(val) : val}`).join("; ");
  return null;
}

export function readPhase9(raw: Record<string, unknown> | undefined): Phase9View {
  const r = isObj(raw) ? raw : {};
  const base = isObj(r.baselines) ? r.baselines : {};
  const ft = isObj(r.finetuned) ? r.finetuned : {};
  const testAcc = num(ft.test_acc);
  const prosus = base.prosus_finbert_as_is;

  const bars: BarItem[] = [
    { key: "majority", label: "Majority class", note: "always predict the most common label", value: accOf(base.majority) },
    { key: "tfidf", label: "TF-IDF + logistic regression", note: "bag-of-words baseline", value: accOf(base.tfidf_logreg) },
    { key: "prosus", label: "FinBERT as released", note: "ProsusAI/finbert, no fine-tuning", value: accOf(prosus) },
    { key: "head", label: "FinBERT, untuned head", note: "new classifier head, encoder frozen", value: accOf(base.untuned_head) },
    {
      key: "ft",
      label: "Fine-tuned FinBERT",
      note: typeof ft.model === "string" ? ft.model : "held-out test split",
      value: testAcc,
      ci: rng(ft.test_acc_ci95),
      highlight: true,
    },
  ];

  const perClass = normPerClass(ft.per_class);
  const confusion = normConfusion(ft.confusion_matrix, perClass.map((p) => p.label));
  const cm = rawNum(ft.cv_mean) != null ? num(ft.cv_mean) : null;

  const crossDataset: { label: string; value: number }[] = [];
  if (isObj(ft.cross_dataset_acc)) {
    for (const [k, v] of Object.entries(ft.cross_dataset_acc)) {
      const a = accOf(v);
      if (a != null) crossDataset.push({ label: k, value: a });
    }
  } else if (num(ft.cross_dataset_acc) != null) crossDataset.push({ label: "Other dataset", value: num(ft.cross_dataset_acc) as number });

  const facts: { label: string; value: string }[] = [];
  const ds = factText(r.datasets);
  if (ds) facts.push({ label: "Datasets", value: ds });
  if (isObj(r.splits)) {
    for (const k of ["train", "val", "test"]) {
      const t = factText((r.splits as Obj)[k]);
      if (t) facts.push({ label: `${k === "val" ? "Validation" : k[0].toUpperCase() + k.slice(1)} split`, value: t });
    }
  }

  return {
    ready: testAcc != null,
    bars,
    caveat: isObj(prosus) && typeof prosus.caveat === "string" ? prosus.caveat : null,
    model: typeof ft.model === "string" ? ft.model : null,
    testAcc,
    valAcc: num(ft.val_acc),
    cv: cm != null ? { mean: cm, std: num(ft.cv_std) } : null,
    crossDataset,
    perClass,
    confusion,
    facts,
    notes: Array.isArray(r.notes) ? r.notes.filter((n): n is string => typeof n === "string" && !/^stub:/i.test(n)) : [],
  };
}
