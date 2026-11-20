"""Evaluate every model of the ladder on the SAME held-out test set (official OffensEval-TR test).

For each model:  macro-F1 / accuracy / OFF-class P-R-F1 (threshold tuned on validation),
                 model size (MB), latency per message on CPU (batch size 1, mean and p95 over
                 >= 100 test messages, end-to-end incl. normalization + tokenization).

Writes results/results.csv, results/results.md and results/test_predictions.csv.

Usage:  python -m src.evaluate [--threads 1] [--n-latency 200]
"""
import argparse
import os
import platform
import subprocess
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .config import LATENCY_BUDGET_MS, RESULTS
from .data import load_split
from .predictors import load_all

BULK_THREADS = max(1, (os.cpu_count() or 4))


def hardware():
    try:
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        cpu = platform.processor()
    return f"{cpu}, {platform.system()} {platform.machine()}, PyTorch {torch.__version__}"


def latency(pred, warmup, texts):
    from .lexicon import match_token
    match_token.cache_clear()          # no help from tokens cached during the bulk prediction
    for t in warmup:
        pred.predict_one(t)
    times = []
    for t in texts:
        t0 = time.perf_counter()
        pred.predict_one(t)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.percentile(times, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--n-latency", type=int, default=200)
    ap.add_argument("--no-bert", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    test = load_split("test")
    y = test.y.to_numpy()
    sample = test.text.sample(args.n_latency + 10, random_state=0).tolist()
    warm_texts, lat_texts = sample[:10], sample[10:]      # warm-up messages are not timed
    preds_df = test[["id", "text", "y"]].copy()
    rows, cms = [], {}
    for p in load_all(include_bert=not args.no_bert):
        t0 = time.time()
        torch.set_num_threads(BULK_THREADS)            # bulk test-set predictions: all cores
        prob = np.asarray(p.predict_proba(test.text.tolist()))
        torch.set_num_threads(args.threads)            # latency: the deployment setting
        yhat = (prob >= p.threshold).astype(int)
        preds_df[f"p_{p.key}"] = prob.round(4)
        pr, rc, f1, _ = precision_recall_fscore_support(y, yhat, average=None, labels=[0, 1])
        mean_ms, p95_ms = latency(p, warm_texts, lat_texts)
        rows.append({
            "model": p.name, "key": p.key, "precision": p.precision,
            "macro_F1": f1_score(y, yhat, average="macro"), "accuracy": accuracy_score(y, yhat),
            "OFF_precision": pr[1], "OFF_recall": rc[1], "OFF_F1": f1[1],
            "size_MB": p.size_mb(), "latency_ms_mean": mean_ms, "latency_ms_p95": p95_ms,
            "meets_budget": mean_ms <= LATENCY_BUDGET_MS, "threshold": p.threshold,
        })
        cms[p.name] = confusion_matrix(y, yhat).tolist()
        print(f"{p.name:40s} macro-F1={rows[-1]['macro_F1']:.4f} size={rows[-1]['size_MB']:.1f}MB "
              f"lat={mean_ms:.2f}ms (p95 {p95_ms:.2f}) [{time.time() - t0:.0f}s]", flush=True)

    res = pd.DataFrame(rows)
    agree = []
    if "berturk" in set(res.key):
        thr = res.set_index("key").threshold["berturk"]
        ref = preds_df.p_berturk >= thr
        for k in [k for k in res.key if k.startswith("berturk_") and k != "berturk"]:
            name = res.set_index("key").model[k]
            agree.append(f"{name} vs FP32 PyTorch: {((preds_df[f'p_{k}'] >= thr) == ref).mean():.2%} same labels")
    suffix = "" if not args.no_bert else "_nobert"
    res.to_csv(RESULTS / f"results{suffix}.csv", index=False)
    preds_df.to_csv(RESULTS / f"test_predictions{suffix}.csv", index=False)

    hw = hardware()
    md = [f"# Results on the OffensEval-TR 2020 test set (n={len(test)}, OFF={y.mean():.1%})", "",
          f"Hardware: {hw}. Latency = batch size 1, {args.threads} CPU thread(s), "
          f"mean over {args.n_latency} test messages (after 10 untimed warm-up messages), "
          f"end-to-end (normalization + tokenization + model). Budget: {LATENCY_BUDGET_MS:.0f} ms/message.", "",
          "| Model | Precision | Macro-F1 | Accuracy | OFF-F1 | Size (MB) | Latency mean (ms) | Latency p95 (ms) | ≤ budget |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['precision']} | {r['macro_F1']:.4f} | {r['accuracy']:.4f} | "
                  f"{r['OFF_F1']:.4f} | {r['size_MB']:.2f} | {r['latency_ms_mean']:.2f} | "
                  f"{r['latency_ms_p95']:.2f} | {'yes' if r['meets_budget'] else 'no'} |")
    if agree:
        md += ["", "Label agreement with the FP32 BERTurk on the test set: " + "; ".join(agree)]
    md += ["", "## Confusion matrices (rows = true NOT/OFF, cols = predicted NOT/OFF)", ""]
    md += [f"- {k}: {v}" for k, v in cms.items()]
    (RESULTS / f"results{suffix}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
