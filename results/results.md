# Results on the OffensEval-TR 2020 test set (n=3528, OFF=20.3%)

Hardware: Apple M2, Darwin arm64, PyTorch 2.15.0. Latency = batch size 1, 1 CPU thread(s), mean over 200 test messages (after 10 untimed warm-up messages), end-to-end (normalization + tokenization + model). Budget: 50 ms/message.

| Model | Precision | Macro-F1 | Accuracy | OFF-F1 | Size (MB) | Latency mean (ms) | Latency p95 (ms) | ≤ budget |
|---|---|---|---|---|---|---|---|---|
| Lexicon + edit distance (rules) | - | 0.6724 | 0.8141 | 0.4570 | 0.00 | 1.81 | 4.32 | yes |
| tf-idf + logistic regression | FP64 | 0.7751 | 0.8614 | 0.6359 | 4.84 | 0.64 | 0.82 | yes |
| word2vec + feed-forward net | FP32 | 0.6594 | 0.7840 | 0.4534 | 22.61 | 0.10 | 0.15 | yes |
| word2vec + 1D CNN | FP32 | 0.6568 | 0.8160 | 0.4231 | 23.29 | 0.20 | 0.29 | yes |
| BERTurk fine-tuned (PyTorch) | FP32 | 0.8159 | 0.8858 | 0.7026 | 422.05 | 30.18 | 37.77 | yes |
| BERTurk + dynamic INT8 (PyTorch) | INT8 Linear (qnnpack) | 0.8166 | 0.8883 | 0.7020 | 177.41 | 66.05 | 139.04 | no |
| BERTurk FP32 (ONNX Runtime) | FP32 | 0.8159 | 0.8858 | 0.7026 | 422.18 | 63.52 | 133.33 | no |
| BERTurk INT8 (ONNX Runtime) | INT8 MatMul (per-channel) | 0.8143 | 0.8866 | 0.6983 | 106.62 | 21.73 | 46.10 | yes |

Label agreement with the FP32 BERTurk on the test set: BERTurk + dynamic INT8 (PyTorch) vs FP32 PyTorch: 98.78% same labels; BERTurk FP32 (ONNX Runtime) vs FP32 PyTorch: 100.00% same labels; BERTurk INT8 (ONNX Runtime) vs FP32 PyTorch: 98.84% same labels

## Confusion matrices (rows = true NOT/OFF, cols = predicted NOT/OFF)

- Lexicon + edit distance (rules): [[2596, 216], [440, 276]]
- tf-idf + logistic regression: [[2612, 200], [289, 427]]
- word2vec + feed-forward net: [[2450, 362], [400, 316]]
- word2vec + 1D CNN: [[2641, 171], [478, 238]]
- BERTurk fine-tuned (PyTorch): [[2649, 163], [240, 476]]
- BERTurk + dynamic INT8 (PyTorch): [[2670, 142], [252, 464]]
- BERTurk FP32 (ONNX Runtime): [[2649, 163], [240, 476]]
- BERTurk INT8 (ONNX Runtime): [[2665, 147], [253, 463]]
