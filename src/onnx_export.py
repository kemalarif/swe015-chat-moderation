"""Export the fine-tuned BERTurk to ONNX and quantize it to INT8 with ONNX Runtime (Area D).

    artifacts/onnx/berturk_fp32.onnx   FP32 graph (torch.onnx.export)
    artifacts/onnx/berturk_int8.onnx   dynamic INT8: the 74 MatMul weight matrices (per-channel) and the
                                       3 embedding tables are stored as INT8; activations are quantized
                                       on the fly; LayerNorm and biases stay FP32

ONNX Runtime's INT8 kernels use the ARM dot-product / I8MM instructions of the CPU.

Usage:  python -m src.onnx_export
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import ARTIFACTS

BERT_DIR = ARTIFACTS / "berturk-offenseval"
ONNX_DIR = ARTIFACTS / "onnx"
FP32_PATH = ONNX_DIR / "berturk_fp32.onnx"
INT8_PATH = ONNX_DIR / "berturk_int8.onnx"


def export_fp32():
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(BERT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_DIR).eval()
    enc = tok(["örnek bir mesaj", "ikinci örnek mesaj biraz daha uzun"], padding=True, return_tensors="pt")
    args = (enc["input_ids"], enc["attention_mask"], enc["token_type_ids"])
    axes = {n: {0: "batch", 1: "seq"} for n in ["input_ids", "attention_mask", "token_type_ids"]}
    axes["logits"] = {0: "batch"}
    torch.onnx.export(model, args, str(FP32_PATH), input_names=["input_ids", "attention_mask", "token_type_ids"],
                      output_names=["logits"], dynamic_axes=axes, opset_version=17, dynamo=False)
    print(f"exported {FP32_PATH} ({FP32_PATH.stat().st_size / 2**20:.1f} MB)")


def quantize_int8():
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(FP32_PATH), str(INT8_PATH), weight_type=QuantType.QInt8, per_channel=True)
    print(f"quantized {INT8_PATH} ({INT8_PATH.stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    export_fp32()
    quantize_int8()
