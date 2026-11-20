"""Collect everything the slides need into results/deck_data.json (all numbers from our own runs).

Usage:  python -m src.report        (after python -m src.evaluate)
"""
import json
import time

import pandas as pd
import torch

from .config import LATENCY_BUDGET_MS, RESULTS
from .masking import mask
from .normalize import normalize_full, normalize_light

NORM_EXAMPLES = [
    "@USER Sen tam bir GERİZEKALISIN!!! https://t.co/x",
    "s1kt1r g1t b u r a d a n",
    "çoooook güzelll olmuşşş 👏",
    "@pt@l mısın sen",
]
DEMO_EXAMPLES = [
    "iyi oyundu arkadaşlar, bir el daha atalım mı?",
    "s1kt1r g1t buradan ezik",
    "takımı sen batırdın, senin gibi adamlar bu oyunun yüz karası",
]


def main():
    res = pd.read_csv(RESULTS / "results.csv")
    preds = pd.read_csv(RESULTS / "test_predictions.csv", dtype={"id": str})
    md = (RESULTS / "results.md").read_text().splitlines()
    import onnxruntime
    hw_line = next(l for l in md if l.startswith("Hardware:"))
    hw_line = hw_line.split(". Latency")[0].replace("Hardware: ", "") + f", ONNX Runtime {onnxruntime.__version__}"
    agree_line = next((l for l in md if l.startswith("Label agreement")), "")

    thr = res.set_index("key").threshold
    q = preds.p_berturk_onnx_int8 >= thr["berturk_onnx_int8"]
    t = preds.p_tfidf >= thr["tfidf"]
    fn = preds[(preds.y == 1) & ~q].copy()
    fp = preds[(preds.y == 0) & q].copy()
    fn["len"] = fn.text.str.len(); fp["len"] = fp.text.str.len()
    # short, readable errors (no @USER-only messages)
    fn = fn[(fn.len.between(25, 110))].sort_values("p_berturk_onnx_int8").head(12)
    fp = fp[(fp.len.between(25, 110))].sort_values("p_berturk_onnx_int8", ascending=False).head(12)
    wins = preds[(preds.y == 1) & q & ~t]
    wins = wins[wins.text.str.len().between(25, 110)].head(12)

    def rows(df):
        return [{"text": r.text, "y": int(r.y), "p_int8": float(r.p_berturk_onnx_int8),
                 "p_tfidf": float(r.p_tfidf)} for r in df.itertuples()]

    from .predictors import BertPredictor, OnnxBertPredictor
    torch.set_num_threads(1)
    m = OnnxBertPredictor("int8", threads=1)
    demo = []
    for s in DEMO_EXAMPLES:
        m.predict_one(s)
        t0 = time.perf_counter(); p = m.predict_one(s); ms = (time.perf_counter() - t0) * 1000
        off = bool(p >= m.threshold)
        demo.append({"text": s, "shown": mask(s, star="*").replace("*mesaj moderasyon tarafından gizlendi*",
                                                                    "message hidden by moderation") if off else s,
                     "p_off": p, "offensive": off, "ms": ms})

    from . import lexicon
    from .data import load_split
    tq = BertPredictor(quantized=True)
    bert_params = sum(p.numel() for p in BertPredictor(quantized=False).model.parameters())
    n_q = sum(type(mod).__name__ == "Linear" and "quantized" in type(mod).__module__
              for mod in tq.model.modules())
    import onnx
    g = onnx.load(str(m.path), load_external_data=False).graph
    ops = pd.Series([n.op_type for n in g.node]).value_counts().to_dict()
    data = {
        "results": res.to_dict(orient="records"),
        "n_quantized_linear": int(n_q),
        "bert_params": int(bert_params),
        "onnx_int8_ops": {k: int(v) for k, v in ops.items()
                          if k in ("MatMulInteger", "DynamicQuantizeLinear", "DynamicQuantizeMatMul",
                                   "MatMul", "Gather", "DequantizeLinear")},
        "lexicon_size": len(lexicon.ROOTS) + len(lexicon.SHORT_EXACT),
        "split_sizes": {k: int(len(load_split(k))) for k in ["train", "val", "test"]},
        "split_off_rate": {k: float(load_split(k).y.mean()) for k in ["train", "val", "test"]},
        "hardware": hw_line, "agreement": agree_line, "budget_ms": LATENCY_BUDGET_MS,
        "n_test": int(len(preds)), "off_rate_test": float(preds.y.mean()),
        "normalization": [{"raw": s, "light": normalize_light(s), "full": normalize_full(s)}
                          for s in NORM_EXAMPLES],
        "false_negatives": rows(fn), "false_positives": rows(fp), "bert_wins": rows(wins),
        "demo": demo,
    }
    (RESULTS / "deck_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(json.dumps({k: data[k] for k in ["hardware", "agreement", "demo"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
