# Real-Time Chat Moderation for Turkish (SWE015 – Use Case 4)

A gaming platform must flag offensive chat messages in Turkish within a strict time limit on a
CPU-only server. This repository trains four models of increasing sophistication, compares them
on the **same held-out test set** (the official OffensEval-TR 2020 test set), and serves the
**INT8-quantized BERTurk (ONNX Runtime)** in a live chat app built with Gradio.

- Model weights: https://huggingface.co/kemalarif/berturk-offenseval-tr (downloaded automatically)
- Author: Kemal Arif Karabacıoğlu (220911802)

## Quick start

Tested with Python 3.12 on macOS (Apple M2). Python 3.10–3.12 should work.

```bash
git clone https://github.com/kemalarif/swe015-chat-moderation.git
cd swe015-chat-moderation
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:7860. On first start the trained weights (≈ 0.6 GB) are downloaded
from the Hugging Face Hub (`src/hub.py`), so the first start takes a few minutes. Nothing large
is stored in this repository. If the first message in the browser shows a spinner forever,
reload the page once.

In the app: type a message and press Enter. The selected model (default: BERTurk INT8 on ONNX
Runtime) labels it *clean / offensive* with a confidence score. Offensive messages are masked. The
side panel logs the response time of every message against the 50 ms budget, lets you switch
models, and shows the prediction of every model for the last message.

## The model ladder

| Ladder step | Model | Area |
|---|---|---|
| Rule baseline (extra) | regex normalization + hand-written lexicon of 91 offensive roots, matched with our own **minimum edit distance** | B |
| Classical baseline | **tf-idf** (word 1–2-grams + char 2–5-grams) + logistic regression | B |
| Neural network | **our own word2vec** (CBOW, 200-d, trained on the training messages) → **feed-forward net** and **1D CNN**, SGD + backpropagation, dropout | A |
| Transformer | **BERTurk** (`dbmdz/bert-base-turkish-cased`) fine-tuned with a classification head | C |
| Quantized Transformer | BERTurk exported to **ONNX** and **dynamically quantized to INT8** with ONNX Runtime (also tried: PyTorch dynamic INT8) | D |

Every model outputs P(offensive). Its decision threshold is tuned on the validation split
(best macro-F1); the test split is never used for tuning.

## Results (official OffensEval-TR 2020 test set, n = 3,528, 20.3% offensive)

Hardware: Apple M2, macOS (arm64), PyTorch 2.15.0, ONNX Runtime 1.32.0. Latency = batch size 1,
**1 CPU thread**, mean / 95th percentile over 200 test messages after 10 untimed warm-up messages,
end-to-end (normalization + tokenization + model). Latency budget: **50 ms per message**.

| Model | Precision | Macro-F1 | Accuracy | OFF-F1 | Size (MB) | Latency mean (ms) | Latency p95 (ms) | ≤ budget |
|---|---|---|---|---|---|---|---|---|
| Lexicon + edit distance (rules) | - | 0.6724 | 0.8141 | 0.4570 | 0.00 | 1.81 | 4.32 | yes |
| tf-idf + logistic regression | FP64 | 0.7751 | 0.8614 | 0.6359 | 4.84 | 0.64 | 0.82 | yes |
| word2vec + feed-forward net | FP32 | 0.6594 | 0.7840 | 0.4534 | 22.61 | 0.10 | 0.15 | yes |
| word2vec + 1D CNN | FP32 | 0.6568 | 0.8160 | 0.4231 | 23.29 | 0.20 | 0.29 | yes |
| BERTurk fine-tuned (PyTorch) | FP32 | 0.8159 | 0.8858 | 0.7026 | 422.05 | 30.18 | 37.77 | yes |
| BERTurk + dynamic INT8 (PyTorch) | INT8 Linear (qnnpack) | 0.8166 | 0.8883 | 0.7020 | 177.41 | 66.05 | 139.04 | no |
| BERTurk FP32 (ONNX Runtime) | FP32 | 0.8159 | 0.8858 | 0.7026 | 422.18 | 63.52 | 133.33 | no |
| **BERTurk INT8 (ONNX Runtime) — used by the app** | INT8 MatMul (per-channel) | 0.8143 | 0.8866 | 0.6983 | 106.62 | 21.73 | 46.10 | yes |

Full output, confusion matrices and label agreement: [`results/results.md`](results/results.md);
per-message test predictions: [`results/test_predictions.csv`](results/test_predictions.csv);
training logs: `results/*_log.txt`.

## Architecture

```
Player (browser) ──► Gradio UI (app.py) ──► Preprocessing (src/normalize.py)
                                              │  normalize_light → BERTurk
                                              │  normalize_full  → lexicon / tf-idf / word2vec
                                              ▼
                                Models (src/predictors.py), 1 CPU thread
                                  · Lexicon + min. edit distance (rules)
                                  · tf-idf + logistic regression (FP64, scikit-learn)
                                  · word2vec + FFN / 1D CNN (FP32, PyTorch)
                                  · BERTurk FP32 (PyTorch) and dynamic INT8 (PyTorch qnnpack)
                                  · BERTurk INT8 (ONNX Runtime)  ★ default
                                              ▼
                     Post-processing: P(offensive) ≥ validation threshold → label + confidence,
                     masking (src/masking.py), latency log vs. 50 ms budget ──► chat bubble + side panel
```

Weights come from the Hugging Face Hub (`src/hub.py`). Training ran offline: BERTurk on the
Apple M2 GPU (MPS), everything else on CPU.

## Reproduce everything from scratch

```bash
python -m src.data          # download OffensEval-TR 2020, build train/val/test splits
python -m src.tfidf_train   # tf-idf + logistic regression baseline           (~30 s)
python -m src.w2v_train     # word2vec + feed-forward net + 1D CNN             (~35 min on CPU)
python -m src.bert_train    # fine-tune BERTurk (MPS / CUDA / CPU)             (~70 min on M2 GPU)
python -m src.onnx_export   # ONNX export + INT8 quantization with ONNX Runtime
python -m src.evaluate      # all models on the same test set -> results/results.md
python -m src.report        # numbers for the slides -> results/deck_data.json
python app.py
```

## Repository layout

```
app.py                 Gradio chat UI (default model: BERTurk INT8 on ONNX Runtime)
src/config.py          paths, constants, latency budget, Hugging Face repo id
src/data.py            download + splits (90/10 stratified split of the official train set; official test set)
src/normalize.py       regex normalization: mentions/URLs, leetspeak, repeated letters, spaced letters, Turkish casing/folding
src/lexicon.py         rule baseline: lexicon + minimum edit distance (own dynamic-programming implementation)
src/tfidf_train.py     tf-idf + logistic regression
src/nn_models.py       feed-forward net and 1D CNN (PyTorch)
src/w2v_train.py       word2vec (gensim, CBOW) + training of the two networks with SGD
src/bert_train.py      BERTurk fine-tuning
src/onnx_export.py     ONNX export + ONNX Runtime dynamic INT8 quantization
src/predictors.py      uniform inference wrappers (incl. PyTorch dynamic INT8)
src/masking.py         masking of flagged messages
src/evaluate.py        macro-F1, size and latency benchmark on the test set
src/report.py          collects the numbers used in the slides
src/hub.py             download / upload of the weights (Hugging Face Hub)
results/               results table, logs, test predictions
slides/                slide generator (pptxgenjs) reading results/deck_data.json
```

## Data and credits

- OffensEval 2020 Turkish: Ç. Çöltekin, *A Corpus of Turkish Offensive Language on Social Media*,
  LREC 2020. CC-BY. 31,756 training tweets (split 90/10 into train / validation, stratified, seed 42)
  and 3,528 test tweets. Downloaded from https://coltekin.github.io/offensive-turkish/.
- BERTurk: `dbmdz/bert-base-turkish-cased` (Stefan Schweter, MDZ Digital Library team).
- The test predictions file contains tweets from the data set, some of them offensive.

## AI usage

Claude Code (Anthropic) was used to write and debug the code, diagnose problems and draft the
README and the slides. All training and benchmark runs were executed on my own machine; every
number above comes from those runs.
