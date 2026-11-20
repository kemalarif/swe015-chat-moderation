"""Real-time chat moderation demo for a gaming platform (Turkish).

Every message typed into the chat is labeled live (clean / offensive) with a confidence score by
the selected model (default: the quantized INT8 BERTurk). Flagged messages are masked. The side
panel logs the response time of every message against the latency budget and lets the user
switch models or compare all models side by side.

Run:  python app.py      (then open http://127.0.0.1:7860)
"""
import datetime
import time

import gradio as gr
import pandas as pd
import torch

from src.config import LATENCY_BUDGET_MS
from src.masking import mask
from src.predictors import APP_KEYS, load_all

torch.set_num_threads(1)          # same setting as the benchmark: one CPU core per message
MODELS = {p.name: p for p in load_all(keys=APP_KEYS, threads=1)}
DEFAULT_MODEL = next(n for n, p in MODELS.items() if p.key == "berturk_onnx_int8")
SHORT = {"lexicon": "Lexicon", "tfidf": "tf-idf+LR", "w2v_ffn": "w2v+FFN", "w2v_cnn": "w2v+CNN",
         "berturk": "BERTurk FP32", "berturk_int8_torch": "BERTurk INT8 (PyTorch)",
         "berturk_onnx_fp32": "BERTurk FP32 (ONNX)", "berturk_onnx_int8": "BERTurk INT8 (ONNX RT)"}
LOG_COLS = ["saat", "model", "karar", "güven", "ms", f"≤{LATENCY_BUDGET_MS:.0f} ms"]
CMP_COLS = ["model", "hassasiyet", "karar", "P(saldırgan)", "ms"]

EXAMPLES = [
    "iyi oyundu arkadaşlar, bir el daha atalım mı?",
    "s1kt1r g1t buradan ezik",
    "g e r i z e k a l ı  mısın sen",
    "takımı sen batırdın, senin gibi adamlar bu oyunun yüz karası",
    "orspu çocuğu kamp yapma",
    "bu maçta çok kötü oynadım ya, kendime kızıyorum",
]


def classify(pred, text):
    t0 = time.perf_counter()
    p = pred.predict_one(text)
    ms = (time.perf_counter() - t0) * 1000
    off = p >= pred.threshold
    return off, p, ms


def on_send(msg, history, log, model_name, compare):
    msg = (msg or "").strip()
    if not msg:
        return "", history, log, gr.update(), gr.update()
    pred = MODELS[model_name]
    off, p, ms = classify(pred, msg)
    conf = p if off else 1 - p
    ok = ms <= LATENCY_BUDGET_MS
    badge = (f"🚫 SALDIRGAN · güven {conf:.2f}" if off else f"✅ temiz · güven {conf:.2f}") + \
            f" · {ms:.1f} ms · {SHORT[pred.key]}"
    shown = mask(msg) if off else msg
    history = history + [{"role": "user", "content": f"{shown}\n\n`{badge}`"}]

    row = [datetime.datetime.now().strftime("%H:%M:%S"), SHORT[pred.key],
           "saldırgan" if off else "temiz", round(conf, 3), round(ms, 2), "✔" if ok else "✘"]
    log = pd.concat([pd.DataFrame([row], columns=LOG_COLS), log], ignore_index=True)

    cur = log[log.model == SHORT[pred.key]]
    stats = (f"**{SHORT[pred.key]}** · {len(cur)} mesaj · ortalama **{cur.ms.mean():.1f} ms** · "
             f"maks {cur.ms.max():.1f} ms · bütçe {LATENCY_BUDGET_MS:.0f} ms "
             f"({(cur.ms <= LATENCY_BUDGET_MS).mean():.0%} bütçe içinde)")

    cmp = gr.update()
    if compare:
        rows = []
        for m in MODELS.values():
            o, pp, t = classify(m, msg)
            rows.append([SHORT[m.key], m.precision, "🚫 saldırgan" if o else "✅ temiz",
                         round(pp, 3), round(t, 2)])
        cmp = pd.DataFrame(rows, columns=CMP_COLS)
    return "", history, log, cmp, stats


def on_clear():
    return [], pd.DataFrame(columns=LOG_COLS), pd.DataFrame(columns=CMP_COLS), ""


with gr.Blocks(title="Oyun Sohbeti Moderasyonu") as demo:
    gr.Markdown("## 🎮 Gerçek Zamanlı Sohbet Moderasyonu (Türkçe) — OffensEval-TR\n"
                "Her mesaj seçili modelle anında **temiz / saldırgan** olarak etiketlenir; "
                "saldırgan mesajlar maskelenir. Varsayılan model: **quantize edilmiş (INT8) BERTurk, ONNX Runtime**, "
                f"CPU, 1 çekirdek, bütçe {LATENCY_BUDGET_MS:.0f} ms/mesaj.")
    with gr.Row():
        with gr.Column(scale=3):
            chat = gr.Chatbot(label="Sohbet odası", height=480, layout="bubble")
            with gr.Row():
                box = gr.Textbox(placeholder="Mesaj yaz ve Enter'a bas…", show_label=False, scale=5,
                                 autofocus=True)
                send = gr.Button("Gönder", variant="primary", scale=1)
            gr.Examples(EXAMPLES, inputs=box, label="Örnek mesajlar")
        with gr.Column(scale=2):
            model_dd = gr.Dropdown(list(MODELS), value=DEFAULT_MODEL, label="Model")
            compare = gr.Checkbox(value=True, label="Tüm modelleri yan yana karşılaştır")
            stats = gr.Markdown()
            log = gr.Dataframe(pd.DataFrame(columns=LOG_COLS), label="Yanıt süresi kaydı",
                               interactive=False, max_height=260)
            cmp = gr.Dataframe(pd.DataFrame(columns=CMP_COLS), label="Son mesaj — tüm modeller",
                               interactive=False)
            clear = gr.Button("Temizle")

    ins, outs = [box, chat, log, model_dd, compare], [box, chat, log, cmp, stats]
    box.submit(on_send, ins, outs)
    send.click(on_send, ins, outs)
    clear.click(on_clear, None, [chat, log, cmp, stats])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
