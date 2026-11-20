"""Small helpers shared by the training scripts."""
import json
import random

import numpy as np
import torch
from sklearn.metrics import f1_score

from .config import ARTIFACTS, SEED


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def macro_f1(y_true, p_off, threshold=0.5):
    return f1_score(y_true, (np.asarray(p_off) >= threshold).astype(int), average="macro")


def best_threshold(y_true, p_off):
    """Pick the decision threshold on P(offensive) that maximizes macro-F1 on the VALIDATION set."""
    grid = np.linspace(0.05, 0.95, 91)
    scores = [macro_f1(y_true, p_off, t) for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def save_threshold(model_key: str, threshold: float, val_f1: float):
    path = ARTIFACTS / "thresholds.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[model_key] = {"threshold": threshold, "val_macro_f1": val_f1}
    path.write_text(json.dumps(data, indent=2))


def load_threshold(model_key: str, default: float = 0.5) -> float:
    path = ARTIFACTS / "thresholds.json"
    if path.exists():
        return json.loads(path.read_text()).get(model_key, {}).get("threshold", default)
    return default
