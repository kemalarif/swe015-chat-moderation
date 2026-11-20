"""Classical baseline: tf-idf (word 1-2 grams + char 2-5 grams) + logistic regression (Area B).

Text is normalized with normalize_full (regex: leetspeak, repeated letters, Turkish folding).
C is tuned on the validation set; the decision threshold too (macro-F1).

Usage:  python -m src.tfidf_train
"""
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .common import best_threshold, save_threshold
from .config import ARTIFACTS, SEED
from .data import load_split
from .normalize import normalize_full

OUT = ARTIFACTS / "tfidf_lr.pkl"


def build(C):
    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                 sublinear_tf=True, max_features=300_000)),
    ])
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=SEED)
    return Pipeline([("tfidf", feats), ("lr", clf)])


def main():
    train, val = load_split("train"), load_split("val")
    Xtr = [normalize_full(t) for t in train.text]
    Xva = [normalize_full(t) for t in val.text]
    best = None
    for C in [0.5, 1, 2, 4, 8]:
        m = build(C).fit(Xtr, train.y)
        pv = m.predict_proba(Xva)[:, 1]
        thr, f1 = best_threshold(val.y, pv)
        print(f"C={C:<4} val macro-F1={f1:.4f} (thr={thr:.2f})")
        if best is None or f1 > best[0]:
            best = (f1, C, thr, m)
    f1, C, thr, m = best
    print(f"best C={C} thr={thr:.2f} val macro-F1={f1:.4f}")
    with open(OUT, "wb") as f:
        pickle.dump(m, f)
    save_threshold("tfidf", thr, f1)


if __name__ == "__main__":
    main()
