"""Train our own word2vec on the training messages, then two neural classifiers on top (Area A):

  1. AvgFFN  - feed-forward net on averaged word2vec vectors, trained with SGD + backpropagation
  2. TextCNN - 1D CNN over the word2vec sequence (dropout), trained with SGD (momentum) + backprop

Only the TRAIN split is used for word2vec and for training; VAL for early stopping + threshold.

Usage:  python -m src.w2v_train
"""
import time

import numpy as np
import torch
import torch.nn as nn
from gensim.models import Word2Vec

from .common import best_threshold, save_threshold, set_seed
from .config import ARTIFACTS, SEED
from .data import load_split
from .nn_models import PAD, UNK, AvgFFN, TextCNN, encode_tokens, pad_batch
from .normalize import tokenize

W2V_PATH = ARTIFACTS / "w2v_offenseval.model"
EMB_DIM = 200


def train_word2vec(token_lists):
    w2v = Word2Vec(sentences=token_lists, vector_size=EMB_DIM, window=5, min_count=2, sg=0,
                   negative=5, epochs=20, workers=4, seed=SEED)  # CBOW
    w2v.save(str(W2V_PATH))
    return w2v


def build_vocab(w2v):
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for w in w2v.wv.index_to_key:
        vocab[w] = len(vocab)
    emb = np.zeros((len(vocab), EMB_DIM), dtype=np.float32)
    emb[UNK] = np.random.default_rng(SEED).normal(0, 0.1, EMB_DIM)
    for w, i in vocab.items():
        if i > UNK:
            emb[i] = w2v.wv[w]
    return vocab, torch.tensor(emb)


@torch.no_grad()
def predict(model, seqs, bs=512):
    model.eval()
    out = []
    for i in range(0, len(seqs), bs):
        out.append(torch.softmax(model(pad_batch(seqs[i:i + bs])), -1)[:, 1])
    return torch.cat(out).numpy()


def fit(model, tr_seqs, ytr, va_seqs, yva, lr, epochs, bs=64, name=""):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    y = torch.tensor(ytr)
    best = (-1, None, 0.5)
    for ep in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(len(tr_seqs))
        tot = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            logits = model(pad_batch([tr_seqs[j] for j in idx]))
            loss = loss_fn(logits, y[idx])
            opt.zero_grad(); loss.backward(); opt.step()            # backpropagation + SGD step
            tot += loss.item() * len(idx)
        sched.step()
        pv = predict(model, va_seqs)
        thr, f1 = best_threshold(yva, pv)
        print(f"  [{name}] ep{ep:2d} train-loss={tot / len(perm):.4f} val macro-F1={f1:.4f} thr={thr:.2f}")
        if f1 > best[0]:
            best = (f1, {k: v.clone() for k, v in model.state_dict().items()}, thr)
    model.load_state_dict(best[1])
    return best[0], best[2]


def main():
    set_seed()
    torch.set_num_threads(4)
    train, val = load_split("train"), load_split("val")
    tr_tok = [tokenize(t) for t in train.text]
    va_tok = [tokenize(t) for t in val.text]

    t0 = time.time()
    w2v = train_word2vec(tr_tok)
    print(f"word2vec: vocab={len(w2v.wv)} dim={EMB_DIM} ({time.time() - t0:.0f}s)")
    for probe in ["salak", "aptal", "guzel", "erdogan"]:
        if probe in w2v.wv:
            print(f"  most similar to {probe!r}: {[w for w, _ in w2v.wv.most_similar(probe, topn=6)]}")

    vocab, emb = build_vocab(w2v)
    tr_seqs = [encode_tokens(t, vocab) for t in tr_tok]
    va_seqs = [encode_tokens(t, vocab) for t in va_tok]
    ytr, yva = train.y.to_numpy(), val.y.to_numpy()

    for key, model, lr, epochs in [
        ("w2v_ffn", AvgFFN(emb), 0.05, 30),
        ("w2v_cnn", TextCNN(emb), 0.02, 15),
    ]:
        set_seed()
        t0 = time.time()
        f1, thr = fit(model, tr_seqs, ytr, va_seqs, yva, lr, epochs, name=key)
        print(f"{key}: best val macro-F1={f1:.4f} thr={thr:.2f} ({time.time() - t0:.0f}s)")
        torch.save({"state_dict": model.state_dict(), "vocab": vocab,
                    "arch": type(model).__name__, "emb_shape": tuple(emb.shape)},
                   ARTIFACTS / f"{key}.pt")
        save_threshold(key, thr, f1)


if __name__ == "__main__":
    main()
