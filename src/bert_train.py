"""Fine-tune BERTurk (dbmdz/bert-base-turkish-cased) for offensive-message classification (Area C).

Input text is normalized with normalize_light (keeps case + Turkish letters, fixes leetspeak etc.).
A linear classification head on the [CLS] token is trained jointly with the encoder.
Best epoch by validation macro-F1 is saved to artifacts/berturk-offenseval/.

Usage:  python -m src.bert_train [--epochs 3] [--bs 32] [--lr 2e-5]
"""
import argparse
import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)

from .common import best_threshold, macro_f1, save_threshold, set_seed
from .config import ARTIFACTS, BERT_BASE, LABELS, MAX_LEN, RESULTS
from .data import load_split
from .normalize import normalize_light

OUT_DIR = ARTIFACTS / "berturk-offenseval"


def encode(tok, df):
    enc = tok([normalize_light(t) for t in df.text], truncation=True, max_length=MAX_LEN)
    return [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i],
             "labels": int(df.y.iloc[i])} for i in range(len(df))]


@torch.no_grad()
def predict_proba(model, loader, device):
    model.eval()
    out = []
    for b in loader:
        b = {k: v.to(device) for k, v in b.items() if k != "labels"}
        logits = model(**b).logits
        out.append(torch.softmax(logits.float(), -1)[:, 1].cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N training rows")
    args = ap.parse_args()

    set_seed()
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    log = open(RESULTS / "bert_train_log.txt", "a")

    def p(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.write(s + "\n"); log.flush()

    tok = AutoTokenizer.from_pretrained(BERT_BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_BASE, num_labels=2, id2label=dict(enumerate(LABELS)),
        label2id={l: i for i, l in enumerate(LABELS)}).to(device)

    train, val = load_split("train"), load_split("val")
    if args.limit:
        train = train.sample(args.limit, random_state=0)
    coll = DataCollatorWithPadding(tok)
    tr_loader = DataLoader(encode(tok, train), batch_size=args.bs, shuffle=True, collate_fn=coll)
    va_loader = DataLoader(encode(tok, val), batch_size=128, shuffle=False, collate_fn=coll)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(tr_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    p(f"=== device={device} train={len(train)} val={len(val)} epochs={args.epochs} bs={args.bs} "
      f"lr={args.lr} steps={total}")

    best = -1.0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        for step, b in enumerate(tr_loader, 1):
            b = {k: v.to(device) for k, v in b.items()}
            loss = model(**b).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            run_loss += loss.item()
            if step % 100 == 0:
                el = time.time() - t0
                done = (ep - 1) * len(tr_loader) + step
                p(f"ep{ep} step {step}/{len(tr_loader)} loss={run_loss / 100:.4f} "
                  f"elapsed={el / 60:.1f}m eta={el / done * (total - done) / 60:.1f}m")
                run_loss = 0.0
        pv = predict_proba(model, va_loader, device)
        f1 = macro_f1(val.y, pv)
        thr, f1_thr = best_threshold(val.y, pv)
        p(f"ep{ep} VAL macro-F1@0.5={f1:.4f}  best-thr={thr:.2f} macro-F1={f1_thr:.4f}")
        if f1_thr > best:
            best = f1_thr
            model.save_pretrained(OUT_DIR)
            tok.save_pretrained(OUT_DIR)
            save_threshold("berturk", thr, f1_thr)
            p(f"  saved -> {OUT_DIR}")
    p(f"done in {(time.time() - t0) / 60:.1f} min, best val macro-F1={best:.4f}")


if __name__ == "__main__":
    main()
