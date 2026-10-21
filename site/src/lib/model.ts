import * as ort from "onnxruntime-web/wasm";
import wasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url";
import { createTokenizer, type Tokenizer } from "./tokenizer";

/**
 * The fine-tuned sentiment model, run entirely in the browser with ONNX Runtime (WebAssembly).
 * Nothing the visitor types leaves the page. Files live in `public/model/` (see scripts/export_site_model.py).
 */
const BASE = import.meta.env.BASE_URL;
const MODEL_URL = `${BASE}model/model.onnx`;
const VOCAB_URL = `${BASE}model/vocab.txt`;
const LABELS_URL = `${BASE}model/labels.json`;
const CACHE_NAME = "ssp-model-v1";
const BATCH = 16;

// Serve the WebAssembly runtime from this site instead of a CDN. Without cross-origin isolation it runs single-threaded.
ort.env.wasm.wasmPaths = { wasm: wasmUrl };
ort.env.wasm.numThreads = 1;

export interface Prediction {
  /** Index into `labels` with the highest probability. */
  top: number;
  /** One probability per label, in `labels` order; they sum to 1. */
  probs: number[];
}

export interface SentimentModel {
  labels: string[];
  classify(texts: string[]): Promise<Prediction[]>;
}

/** Fraction downloaded, 0 to 1, or -1 when the server does not report a size. */
export type ProgressFn = (fraction: number) => void;

async function downloadModel(onProgress: ProgressFn): Promise<Uint8Array> {
  const cache = typeof caches === "undefined" ? null : await caches.open(CACHE_NAME).catch(() => null);
  const hit = await cache?.match(MODEL_URL);
  if (hit) {
    const bytes = new Uint8Array(await hit.arrayBuffer());
    onProgress(1);
    return bytes;
  }

  const res = await fetch(MODEL_URL);
  if (!res.ok || !res.body) throw new Error(`Could not download the model (HTTP ${res.status}).`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  let lastPercent = -2;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    // Report whole percents only: a network chunk is a few KB, and re-rendering the page for each one is slow.
    const percent = total ? Math.floor(Math.min(loaded / total, 1) * 100) : -1;
    if (percent !== lastPercent) {
      lastPercent = percent;
      onProgress(percent < 0 ? -1 : percent / 100);
    }
  }
  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  // Keep a copy so the next visit starts instantly; a full or blocked cache is not an error.
  cache?.put(MODEL_URL, new Response(bytes)).catch(() => undefined);
  onProgress(1);
  return bytes;
}

function softmax(logits: ArrayLike<number>, from: number, count: number): number[] {
  let max = -Infinity;
  for (let i = 0; i < count; i++) max = Math.max(max, logits[from + i]);
  const exps: number[] = [];
  let sum = 0;
  for (let i = 0; i < count; i++) {
    const e = Math.exp(logits[from + i] - max);
    exps.push(e);
    sum += e;
  }
  return exps.map((e) => e / sum);
}

let loading: Promise<SentimentModel> | null = null;

/** Download and start the model once; later calls share the same promise. */
export function loadModel(onProgress: ProgressFn): Promise<SentimentModel> {
  if (!loading) {
    loading = (async () => {
      const [meta, vocab, bytes] = await Promise.all([
        fetch(LABELS_URL).then((r) => r.json() as Promise<{ labels: string[]; max_length: number }>),
        fetch(VOCAB_URL).then((r) => r.text()),
        downloadModel(onProgress),
      ]);
      const tokenizer: Tokenizer = createTokenizer(vocab, meta.max_length);
      const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
      const n = meta.labels.length;

      return {
        labels: meta.labels,
        async classify(texts: string[]): Promise<Prediction[]> {
          const out: Prediction[] = [];
          for (let start = 0; start < texts.length; start += BATCH) {
            const rows = texts.slice(start, start + BATCH).map((t) => tokenizer.encode(t));
            const width = Math.max(...rows.map((r) => r.length));
            const ids = new BigInt64Array(rows.length * width); // zero-filled: [PAD] has id 0
            const mask = new BigInt64Array(rows.length * width);
            rows.forEach((row, r) =>
              row.forEach((id, c) => {
                ids[r * width + c] = BigInt(id);
                mask[r * width + c] = 1n;
              }),
            );
            const result = await session.run({
              input_ids: new ort.Tensor("int64", ids, [rows.length, width]),
              attention_mask: new ort.Tensor("int64", mask, [rows.length, width]),
            });
            const logits = result.logits.data as Float32Array;
            for (let r = 0; r < rows.length; r++) {
              const probs = softmax(logits, r * n, n);
              out.push({ top: probs.indexOf(Math.max(...probs)), probs });
            }
          }
          return out;
        },
      };
    })().catch((err) => {
      loading = null; // allow a retry
      throw err;
    });
  }
  return loading;
}
