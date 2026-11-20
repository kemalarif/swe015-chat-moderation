"""Uniform inference wrappers for every model of the ladder (used by evaluate.py and app.py).

Every predictor exposes:
    key, name, precision
    threshold                      decision threshold on P(offensive), tuned on the validation set
    predict_proba(list[str])       -> np.ndarray of P(offensive)
    predict_one(str)               -> float P(offensive)   (batch size 1, what the app does)
    size_mb()                      -> serialized model size
"""
import io
import pickle
from pathlib import Path

import numpy as np
import torch

from .common import load_threshold
from .config import ARTIFACTS, MAX_LEN

BERT_DIR = ARTIFACTS / "berturk-offenseval"


def _file_mb(path) -> float:
    return Path(path).stat().st_size / 2**20


def _state_dict_mb(model) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 2**20


class Predictor:
    key = ""
    name = ""
    precision = ""

    @property
    def threshold(self):
        return load_threshold(self.threshold_key)

    @property
    def threshold_key(self):
        return self.key

    def predict_proba(self, texts):
        return np.array([self.predict_one(t) for t in texts])

    def predict_one(self, text):
        return float(self.predict_proba([text])[0])


class LexiconPredictor(Predictor):
    key, name, precision = "lexicon", "Lexicon + edit distance (rules)", "-"

    def __init__(self):
        from .lexicon import LexiconClassifier
        self.clf = LexiconClassifier()

    @property
    def threshold(self):
        return 0.5

    def predict_one(self, text):
        return self.clf.predict_proba_one(text)

    def size_mb(self):
        from . import lexicon
        return len(pickle.dumps((lexicon.ROOTS, lexicon.SHORT_EXACT))) / 2**20


class TfidfPredictor(Predictor):
    key, name, precision = "tfidf", "tf-idf + logistic regression", "FP64"
    path = ARTIFACTS / "tfidf_lr.pkl"

    def __init__(self):
        from .normalize import normalize_full
        self.norm = normalize_full
        with open(self.path, "rb") as f:
            self.model = pickle.load(f)

    def predict_proba(self, texts):
        return self.model.predict_proba([self.norm(t) for t in texts])[:, 1]

    def size_mb(self):
        return _file_mb(self.path)


class W2VPredictor(Predictor):
    precision = "FP32"

    def __init__(self, key):
        from .nn_models import AvgFFN, TextCNN
        from .normalize import tokenize
        self.key = key
        self.path = ARTIFACTS / f"{key}.pt"
        ck = torch.load(self.path, map_location="cpu", weights_only=False)
        arch = {"AvgFFN": AvgFFN, "TextCNN": TextCNN}[ck["arch"]]
        self.model = arch(torch.zeros(ck["emb_shape"]))
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.vocab = ck["vocab"]
        self.tokenize = tokenize
        self.name = {"AvgFFN": "word2vec + feed-forward net",
                     "TextCNN": "word2vec + 1D CNN"}[ck["arch"]]

    @torch.no_grad()
    def predict_proba(self, texts):
        from .nn_models import encode_tokens, pad_batch
        seqs = [encode_tokens(self.tokenize(t), self.vocab) for t in texts]
        out = []
        for i in range(0, len(seqs), 256):
            out.append(torch.softmax(self.model(pad_batch(seqs[i:i + 256])), -1)[:, 1])
        return torch.cat(out).numpy()

    def size_mb(self):
        return _file_mb(self.path)


def quantize_int8(model):
    """Post-training dynamic quantization: every nn.Linear -> INT8 weights (activations quantized
    on the fly). Embeddings / LayerNorm stay FP32."""
    torch.backends.quantized.engine = "qnnpack" if "qnnpack" in torch.backends.quantized.supported_engines \
        else torch.backends.quantized.supported_engines[-1]
    return torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


class BertPredictor(Predictor):
    def __init__(self, quantized=False):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from .normalize import normalize_light
        path = str(BERT_DIR)
        self.tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path).eval()
        self.quantized = quantized
        self.model = quantize_int8(model) if quantized else model
        self.norm = normalize_light
        self.key = "berturk_int8_torch" if quantized else "berturk"
        self.name = "BERTurk + dynamic INT8 (PyTorch)" if quantized else "BERTurk fine-tuned (PyTorch)"
        self.precision = "INT8 Linear (qnnpack)" if quantized else "FP32"

    @property
    def threshold_key(self):
        return "berturk"          # the INT8 model reuses the FP32 model's validation threshold

    @torch.no_grad()
    def predict_proba(self, texts, bs=64):
        out = []
        for i in range(0, len(texts), bs):
            enc = self.tok([self.norm(t) for t in texts[i:i + bs]], truncation=True,
                           max_length=MAX_LEN, padding=True, return_tensors="pt")
            out.append(torch.softmax(self.model(**enc).logits, -1)[:, 1])
        return torch.cat(out).numpy()

    def size_mb(self):
        return _state_dict_mb(self.model)


class OnnxBertPredictor(Predictor):
    """BERTurk exported to ONNX and run with ONNX Runtime (CPU). variant = "int8" or "fp32"."""

    def __init__(self, variant="int8", threads=None):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        from .normalize import normalize_light
        self.path = ARTIFACTS / "onnx" / f"berturk_{variant}.onnx"
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads or torch.get_num_threads()
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(self.path), opts, providers=["CPUExecutionProvider"])
        self.tok = AutoTokenizer.from_pretrained(str(BERT_DIR))
        self.norm = normalize_light
        self.key = f"berturk_onnx_{variant}"
        self.name = f"BERTurk {variant.upper()} (ONNX Runtime)"
        self.precision = "INT8 MatMul (per-channel)" if variant == "int8" else "FP32"

    @property
    def threshold_key(self):
        return "berturk"          # same validation threshold as the FP32 PyTorch model

    def predict_proba(self, texts, bs=64):
        out = []
        for i in range(0, len(texts), bs):
            enc = self.tok([self.norm(t) for t in texts[i:i + bs]], truncation=True,
                           max_length=MAX_LEN, padding=True, return_tensors="np")
            feed = {k: enc[k].astype(np.int64) for k in ("input_ids", "attention_mask", "token_type_ids")}
            logits = self.sess.run(["logits"], feed)[0]
            e = np.exp(logits - logits.max(-1, keepdims=True))
            out.append(e[:, 1] / e.sum(-1))
        return np.concatenate(out)

    def size_mb(self):
        return _file_mb(self.path)


APP_KEYS = ["lexicon", "tfidf", "w2v_ffn", "w2v_cnn", "berturk", "berturk_int8_torch",
            "berturk_onnx_int8"]


def load_all(include_bert=True, threads=None, keys=None):
    """Load the predictors. keys=None -> every model (evaluation); APP_KEYS for the app."""
    from .hub import ensure_artifacts
    ensure_artifacts()
    makers = {
        "lexicon": LexiconPredictor, "tfidf": TfidfPredictor,
        "w2v_ffn": lambda: W2VPredictor("w2v_ffn"), "w2v_cnn": lambda: W2VPredictor("w2v_cnn"),
        "berturk": lambda: BertPredictor(quantized=False),
        "berturk_int8_torch": lambda: BertPredictor(quantized=True),
        "berturk_onnx_fp32": lambda: OnnxBertPredictor("fp32", threads),
        "berturk_onnx_int8": lambda: OnnxBertPredictor("int8", threads),
    }
    keys = keys or [k for k in makers if include_bert or not k.startswith("berturk")]
    return [makers[k]() for k in keys]
