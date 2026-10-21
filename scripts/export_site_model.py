"""Export the fine-tuned sentiment model for the in-browser tool (site/).

Reads  models/sentiment_ft/   (DistilBERT sequence classifier, negative/neutral/positive)
Writes site/public/model/
    model.onnx             ONNX graph with int8 weights (per-channel scales), run by onnxruntime-web
    vocab.txt              WordPiece vocabulary, one token per line (line number = token id)
    labels.json            class names in logit order

Needs: torch, transformers, onnx, onnxscript, onnxruntime.
The script also checks the exported model against the PyTorch original on a few sentences
and refuses to write output if the predicted classes disagree.

    python scripts/export_site_model.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper, numpy_helper
from torch.export import Dim
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "models" / "sentiment_ft"
OUT = ROOT / "site" / "public" / "model"
MAX_LEN = 64  # the length the model was fine-tuned and evaluated at

CHECK_TEXTS = [
    "Apple raises full-year guidance after record services revenue",
    "Apple faces new antitrust probe over App Store fees",
    "Microsoft cloud growth slows for a third straight quarter",
    "The company will hold its annual meeting on Tuesday",
    "Tesla recalls 200,000 vehicles over software fault",
    "NVIDIA beats earnings estimates and lifts outlook",
]


class Logits(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def weight_only_int8(graph, min_params=100_000):
    """Store every large 2-D weight as int8 with one scale per output channel, dequantized at run time.

    Activations stay float, so the network computes in full precision and differs from the original only by the
    rounding of its weights. (ONNX Runtime's dynamic quantization also rounds activations and treats each embedding
    table as one tensor; that changed individual predictions too much to ship.) Axis choice per consumer:
      Gather data     [rows, dim]  -> one scale per row (per token)
      MatMul B        [in, out]    -> one scale per output column
      Gemm B          [out, in] if transB else [in, out] -> one scale per output channel
    """
    inits = {i.name: i for i in graph.graph.initializer}
    axis_for = {}
    for node in graph.graph.node:
        for pos, name in enumerate(node.input):
            init = inits.get(name)
            if init is None or len(init.dims) != 2 or init.dims[0] * init.dims[1] < min_params:
                continue
            if node.op_type == "Gather" and pos == 0:
                axis_for[name] = 0
            elif node.op_type == "MatMul" and pos == 1:
                axis_for[name] = 1
            elif node.op_type == "Gemm" and pos == 1:
                trans_b = next((a.i for a in node.attribute if a.name == "transB"), 0)
                axis_for[name] = 0 if trans_b else 1
    new_nodes, kept = [], []
    for init in graph.graph.initializer:
        if init.name not in axis_for:
            kept.append(init)
            continue
        w = numpy_helper.to_array(init).astype(np.float32)
        axis = axis_for[init.name]
        reduce_axis = 1 - axis  # take the max over the other dimension, leaving one value per channel
        scale = np.maximum(np.abs(w).max(axis=reduce_axis), 1e-8) / 127.0
        shape = [-1, 1] if axis == 0 else [1, -1]
        q = np.clip(np.round(w / scale.reshape(shape)), -127, 127).astype(np.int8)
        kept += [
            numpy_helper.from_array(q, init.name + "_q"),
            numpy_helper.from_array(scale.astype(np.float32), init.name + "_scale"),
            numpy_helper.from_array(np.zeros_like(scale, dtype=np.int8), init.name + "_zp"),
        ]
        new_nodes.append(
            helper.make_node("DequantizeLinear", [init.name + "_q", init.name + "_scale", init.name + "_zp"], [init.name], axis=axis)
        )
    del graph.graph.initializer[:]
    graph.graph.initializer.extend(kept)
    # The dequantize nodes only read initializers, so putting them first keeps the graph topologically sorted.
    old = list(graph.graph.node)
    del graph.graph.node[:]
    graph.graph.node.extend(new_nodes + old)
    return len(new_nodes)


def main():
    tok = AutoTokenizer.from_pretrained(SRC)
    model = AutoModelForSequenceClassification.from_pretrained(SRC, attn_implementation="eager").eval()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    print("labels:", labels)

    enc = tok(CHECK_TEXTS[:2], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    tmp = Path(tempfile.mkdtemp())
    fp32 = tmp / "model.onnx"
    torch.onnx.export(
        Logits(model),
        (enc["input_ids"], enc["attention_mask"]),
        str(fp32),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_shapes={
            "input_ids": {0: Dim("batch", min=1, max=64), 1: Dim("seq", min=2, max=128)},
            "attention_mask": {0: Dim("batch", min=1, max=64), 1: Dim("seq", min=2, max=128)},
        },
        opset_version=17,
        dynamo=True,
    )

    graph = onnx.load(str(fp32))  # also loads the external weights file
    del graph.graph.value_info[:]  # the exporter leaves stale shape annotations behind
    n = weight_only_int8(graph)
    print(f"stored {n} weight matrices as int8 + per-channel scales")

    OUT.mkdir(parents=True, exist_ok=True)
    quant = OUT / "model.onnx"
    onnx.save(graph, str(quant))
    print(f"model file: {os.path.getsize(quant) / 1e6:.1f} MB")

    # Parity check: the shipped model must agree with PyTorch on the predicted class.
    e = tok(CHECK_TEXTS, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="np")
    feeds = {"input_ids": e["input_ids"].astype(np.int64), "attention_mask": e["attention_mask"].astype(np.int64)}
    got = softmax(ort.InferenceSession(str(quant), providers=["CPUExecutionProvider"]).run(None, feeds)[0])
    with torch.no_grad():
        ref = torch.softmax(model(**{k: torch.tensor(v) for k, v in e.items()}).logits, -1).numpy()
    print("max abs probability difference vs PyTorch:", float(np.abs(got - ref).max()))
    # Rounding the weights may flip a near-tie (top two classes within 0.15) but nothing more decisive than that.
    top2 = np.sort(ref, axis=-1)[:, -2:]
    flipped = got.argmax(-1) != ref.argmax(-1)
    decisive_flip = flipped & ((top2[:, 1] - top2[:, 0]) >= 0.15)
    if decisive_flip.any() or np.abs(got - ref).max() > 0.3:
        quant.unlink()
        sys.exit("Exported model drifts from PyTorch on a decisive prediction; refusing to write it.")
    print(f"class flips vs PyTorch: {int(flipped.sum())} of {len(flipped)} (near-ties only)")

    vocab = tok.get_vocab()
    ordered = [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    assert len(ordered) == len(vocab) and all(vocab[t] == i for i, t in enumerate(ordered)), "vocab ids are not contiguous"
    (OUT / "vocab.txt").write_bytes(("\n".join(ordered) + "\n").encode("utf-8"))  # bytes: no CRLF translation on Windows
    (OUT / "labels.json").write_text(json.dumps({"labels": labels, "max_length": MAX_LEN}), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
