"""Shared paths and constants for the chat moderation project."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"

DATA_URL = "https://coltekin.github.io/offensive-turkish/offenseval2020-turkish.zip"

LABELS = ["NOT", "OFF"]          # 0 = clean, 1 = offensive
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
SEED = 42
VAL_FRACTION = 0.10

BERT_BASE = "dbmdz/bert-base-turkish-cased"
MAX_LEN = 96
LATENCY_BUDGET_MS = 50.0         # per message, CPU, batch size 1

# Hugging Face Hub model repo that hosts the trained weights (see src/hub.py)
HF_REPO = os.environ.get("MODERATION_HF_REPO", "kemalarif/berturk-offenseval-tr")

for d in (RAW_DIR, PROC_DIR, ARTIFACTS, RESULTS):
    d.mkdir(parents=True, exist_ok=True)
