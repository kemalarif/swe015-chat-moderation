// Builds slides/SWE015_ChatModeration.pptx from results/deck_data.json (numbers from our own runs).
// Usage: cd slides && node build.js
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require("./apply_theme.js");

const D = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "results", "deck_data.json"), "utf8"));
const N = require("./narrative.js"); // slide wording that depends on the results
const R = Object.fromEntries(D.results.map((r) => [r.key, r]));

const META = {
  name: "Kemal Arif Karabacıoğlu",
  id: "220911802",
  repo: "https://github.com/kemalarif/swe015-chat-moderation",
  video: process.env.VIDEO_URL || "VIDEO LINK – add before export",
  hf: "https://huggingface.co/kemalarif/berturk-offenseval-tr",
};

const THEME = {
  name: "Chat Moderation",
  headFontFace: "Avenir Next",
  bodyFontFace: "Helvetica Neue",
  colors: {
    dk1: "1C1B2E", lt1: "FFFFFF", dk2: "3A3858", lt2: "F2F1F8",
    accent1: "6C5CE7", accent2: "E5484D", accent3: "17A67E", accent4: "F2A93B",
    accent5: "8C8AA6", accent6: "2F9BD6", hlink: "6C5CE7", folHlink: "8C8AA6",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Real-Time Chat Moderation for Turkish";
pres.author = META.name;
const C = pres.SchemeColor;
const W = 13.333;

// ---------- layouts ----------
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "kicker", type: "body", x: 0.7, y: 0.9, w: 7.2, h: 0.4, fontSize: 14, color: C.accent4, bold: true, margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 1.35, w: 7.2, h: 1.9, fontSize: 44, bold: true, color: C.background1, valign: "top", align: "left", margin: 0, fontFace: THEME.headFontFace }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "kicker", type: "body", x: 0.6, y: 0.35, w: 9, h: 0.35, fontSize: 12, color: C.accent1, bold: true, margin: 0, charSpacing: 2 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.7, w: 12.1, h: 0.8, fontSize: 32, bold: true, color: C.text1, valign: "top", align: "left", margin: 0, fontFace: THEME.headFontFace }, text: "" } },
    { text: { text: `SWE015 · Use Case 4 · ${META.name}`, options: { x: 0.6, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: C.accent5, margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: C.accent5, align: "right" },
});
pres.defineSlideMaster({
  title: "CLOSING_DARK",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "kicker", type: "body", x: 0.6, y: 0.35, w: 9, h: 0.35, fontSize: 12, color: C.accent4, bold: true, margin: 0, charSpacing: 2 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.7, w: 12.1, h: 0.8, fontSize: 32, bold: true, color: C.background1, valign: "top", align: "left", margin: 0, fontFace: THEME.headFontFace }, text: "" } },
    { text: { text: `SWE015 · Use Case 4 · ${META.name}`, options: { x: 0.6, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: C.accent5, margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: C.accent5, align: "right" },
});

// ---------- helpers ----------
const fmt = (x, d = 3) => Number(x).toFixed(d);
const ms = (x) => (Number(x) < 10 ? Number(x).toFixed(2) : Number(x).toFixed(1));
const mb = (x) => (Number(x) < 0.1 ? Number(x).toFixed(3) : Number(x) < 10 ? Number(x).toFixed(1) : Number(x).toFixed(0));
let objId = 0;
const oid = (s) => `${s}-${++objId}`;
const pct = (x) => `${(100 * x).toFixed(1)}%`;

function text(slide, t, o) {
  slide.addText(t, { isTextBox: true, margin: 0, fontSize: 15, color: C.text1, valign: "top", objectName: oid("text"), ...o });
}
function card(slide, x, y, w, h, fill = C.background2) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill }, line: { color: fill }, objectName: oid("card") });
}
function badge(slide, x, y, label, fill, d = 0.5) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill }, objectName: oid("badge") });
  slide.addText(label, { isTextBox: true, x, y, w: d, h: d, align: "center", valign: "middle", fontSize: 16, bold: true, color: C.background1, margin: 0, objectName: oid("badge-label") });
}
function arrow(slide, x, y, w, h, color = HEX.accent5) {
  slide.addShape(pres.shapes.LINE, { x, y, w, h, line: { color, width: 1.75, endArrowType: "triangle" }, objectName: oid("arrow") });
}

// Bar charts drawn with shapes: Keynote (used to export the PDF) does not import pptxgenjs charts.
function colChart(slide, { x, y, w, h, title, items, vmax, budget, fmtv }) {
  text(slide, title, { x, y, w, h: 0.35, fontSize: 13, bold: true, color: C.text1 });
  const top = y + 0.75, bottom = y + h - 0.75, plotH = bottom - top, slot = w / items.length, bw = slot * 0.55;
  items.forEach((it, i) => {
    const bh = Math.max(0.02, (it.value / vmax) * plotH), bx = x + i * slot + (slot - bw) / 2;
    slide.addShape(pres.shapes.RECTANGLE, { x: bx, y: bottom - bh, w: bw, h: bh, fill: { color: it.color }, line: { color: it.color }, objectName: oid("bar") });
    text(slide, fmtv(it.value), { x: bx - 0.2, y: bottom - bh - 0.36, w: bw + 0.4, h: 0.32, fontSize: 13, bold: true, align: "center", valign: "bottom", color: C.text1 });
    text(slide, it.label, { x: x + i * slot, y: bottom + 0.08, w: slot, h: 0.6, fontSize: 11, align: "center", color: C.text2 });
  });
  slide.addShape(pres.shapes.LINE, { x, y: bottom, w, h: 0, line: { color: "C9C7D9", width: 1 }, objectName: oid("axis") });
  if (budget) {
    const yb = bottom - (budget / vmax) * plotH;
    slide.addShape(pres.shapes.LINE, { x, y: yb, w, h: 0, line: { color: HEX.accent2, width: 1.5, dashType: "dash" }, objectName: oid("budget") });
    text(slide, `budget ${budget} ms`, { x: x + w - 1.6, y: yb - 0.3, w: 1.6, h: 0.26, fontSize: 11, bold: true, align: "right", color: C.accent2 });
  }
}
function barChartH(slide, { x, y, w, h, title, items, vmax, fmtv, labelW = 1.8 }) {
  text(slide, title, { x, y, w, h: 0.35, fontSize: 13, bold: true, color: C.text1 });
  const top = y + 0.55, rowH = (h - 0.55) / items.length, bw = rowH * 0.62, plotW = w - labelW - 0.75;
  items.forEach((it, i) => {
    const ry = top + i * rowH + (rowH - bw) / 2, len = Math.max(0.02, (it.value / vmax) * plotW);
    text(slide, it.label, { x, y: ry, w: labelW - 0.1, h: bw, fontSize: 12, align: "right", valign: "middle", color: C.text2, bold: !!it.bold });
    slide.addShape(pres.shapes.RECTANGLE, { x: x + labelW, y: ry, w: len, h: bw, fill: { color: it.color }, line: { color: it.color }, objectName: oid("bar") });
    text(slide, fmtv(it.value), { x: x + labelW + len + 0.08, y: ry, w: 0.7, h: bw, fontSize: 12, bold: true, valign: "middle", color: C.text1 });
  });
}

// chat bubble = the deck's visual motif
function bubble(slide, x, y, w, msg, verdict, offensive, opts = {}) {
  const h = opts.h || 0.95;
  const fill = opts.dark ? HEX.dk2 : HEX.lt2;
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.18, fill: { color: fill }, line: { color: offensive ? HEX.accent2 : HEX.accent3, width: 1.5 }, objectName: oid("bubble") });
  text(slide, msg, { x: x + 0.2, y: y + 0.1, w: w - 0.4, h: h * 0.5, fontSize: opts.fs || 14, color: opts.dark ? C.background1 : C.text1, valign: "middle" });
  text(slide, verdict, { x: x + 0.2, y: y + h * 0.58, w: w - 0.4, h: h * 0.32, fontSize: 11, bold: true, color: offensive ? C.accent2 : C.accent3, valign: "middle" });
}

// =====================================================================
// 1. Title
pres.addSection({ title: "Title" });
{
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Title" });
  s.addText("SWE015 · USE CASE 4 – REAL-TIME CHAT MODERATION", { placeholder: "kicker" });
  s.addText("Real-Time Chat Moderation for Turkish", { placeholder: "title" });
  text(s, "From lexicon rules to a quantized BERTurk: four models compared on the same test set, the INT8 model served in a live chat app on one CPU core.", { x: 0.7, y: 3.35, w: 7.0, h: 0.9, fontSize: 16, color: C.background2 });
  const info = [
    [{ text: META.name, options: { bold: true, color: C.background1 } }, { text: `   ·   Student ID ${META.id}`, options: { color: C.background2 } }],
    [{ text: "Repository  ", options: { bold: true, color: C.accent4 } }, { text: META.repo, options: { color: C.background1, hyperlink: { url: META.repo } } }],
    [{ text: "Demo video  ", options: { bold: true, color: C.accent4 } }, { text: META.video, options: { color: C.background1, ...(META.video.startsWith("http") ? { hyperlink: { url: META.video } } : {}) } }],
  ];
  info.forEach((runs, i) => s.addText(runs, { isTextBox: true, x: 0.7, y: 4.75 + i * 0.5, w: 7.6, h: 0.4, fontSize: 15, margin: 0, objectName: oid("info") }));
  // chat mock-up with the INT8 model's real outputs (src/report.py)
  card(s, 8.55, 0.9, 4.2, 5.7, HEX.dk2);
  text(s, "LIVE CHAT · BERTurk INT8 (ONNX Runtime)", { x: 8.85, y: 1.1, w: 3.6, h: 0.3, fontSize: 11, bold: true, color: C.accent4 });
  D.demo.forEach((d, i) => {
    const shown = d.shown;
    const verdict = `${d.offensive ? "🚫 offensive" : "✅ clean"} · conf ${fmt(d.offensive ? d.p_off : 1 - d.p_off, 2)} · ${ms(d.ms)} ms`;
    bubble(s, 8.8, 1.6 + i * 1.6, 3.7, shown, verdict, d.offensive, { dark: false, h: 1.3, fs: 13 });
  });
}

// =====================================================================
// 2. Introduction
pres.addSection({ title: "Introduction" });
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  s.addText("INTRODUCTION", { placeholder: "kicker" });
  s.addText("Flag offensive Turkish chat within a strict CPU time budget", { placeholder: "title" });
  text(s, "The problem", { x: 0.6, y: 1.8, w: 5.6, h: 0.4, fontSize: 20, bold: true, color: C.accent1 });
  text(s, [
    { text: "A gaming platform's chat receives a constant stream of short, noisy Turkish messages: slang, leetspeak (s1kt1r), repeated letters, spaced-out insults.", options: { bullet: true, breakLine: true } },
    { text: "Offensive messages must be labeled live and masked before other players see them.", options: { bullet: true, breakLine: true } },
    { text: "The server is CPU-only, so every message must be classified within a fixed time limit.", options: { bullet: true, breakLine: true } },
    { text: "Users: players (safer chat) and moderators (fewer manual reports).", options: { bullet: true } },
  ], { x: 0.6, y: 2.3, w: 5.6, h: 3.6, fontSize: 15, paraSpaceAfter: 10 });
  text(s, "What success looks like", { x: 6.9, y: 1.8, w: 5.8, h: 0.4, fontSize: 20, bold: true, color: C.accent1 });
  const goals = [
    ["1", "Accurate", `Macro-F1 on the held-out OffensEval-TR test set (classes are imbalanced: ${(D.off_rate_test * 100).toFixed(0)}% offensive).`, C.accent1],
    ["2", "Fast", `≤ ${D.budget_ms} ms per message on one CPU core for the deployed model, reached with INT8 quantization.`, C.accent4],
    ["3", "Usable", "A live chat app: label + confidence, masking, latency log, model switch.", C.accent3],
  ];
  goals.forEach(([n, h, b, col], i) => {
    const y = 2.35 + i * 1.5;
    card(s, 6.9, y, 5.8, 1.3);
    badge(s, 7.1, y + 0.4, n, col);
    text(s, h, { x: 7.85, y: y + 0.15, w: 4.6, h: 0.35, fontSize: 17, bold: true });
    text(s, b, { x: 7.85, y: y + 0.5, w: 4.7, h: 0.75, fontSize: 13, color: C.text2 });
  });
}

// =====================================================================
// 3. Data
pres.addSection({ title: "Development" });
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Development" });
  s.addText("DEVELOPMENT · DATA", { placeholder: "kicker" });
  s.addText("OffensEval-TR 2020: 35k labeled Turkish tweets", { placeholder: "title" });
  const stats = [
    [D.split_sizes.train.toLocaleString("en-US"), "train", "90% of the official training set (stratified, seed 42)"],
    [D.split_sizes.val.toLocaleString("en-US"), "validation", "10% of the training set: early stopping and decision thresholds"],
    [D.n_test.toLocaleString("en-US"), "test (held out)", "official OffensEval-TR test set; the same test set for every model"],
  ];
  stats.forEach(([v, l, d], i) => {
    const y = 1.85 + i * 1.5;
    text(s, v, { x: 0.6, y, w: 2.4, h: 0.75, fontSize: 40, bold: true, color: C.accent1, fontFace: THEME.headFontFace });
    text(s, l, { x: 3.1, y: y + 0.05, w: 2.6, h: 0.35, fontSize: 16, bold: true });
    text(s, d, { x: 3.1, y: y + 0.42, w: 2.7, h: 0.75, fontSize: 12, color: C.text2 });
  });
  text(s, `Source: Çöltekin (2020), A Corpus of Turkish Offensive Language on Social Media, LREC (CC-BY). Labels NOT / OFF (sub-task A). OFF share: train ${pct(D.split_off_rate.train)}, val ${pct(D.split_off_rate.val)}, test ${pct(D.split_off_rate.test)}.`, { x: 0.6, y: 6.25, w: 5.4, h: 0.6, fontSize: 10, color: C.accent5 });
  text(s, "Preprocessing (regex, src/normalize.py)", { x: 6.4, y: 1.8, w: 6.3, h: 0.4, fontSize: 18, bold: true, color: C.accent1 });
  const rows = [[
    { text: "Raw message", options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk1 } } },
    { text: "normalize_light → BERTurk", options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk1 } } },
    { text: "normalize_full → lexicon, tf-idf, word2vec", options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk1 } } },
  ]];
  D.normalization.forEach((n) => rows.push([n.raw, n.light, n.full]));
  s.addTable(rows, { x: 6.4, y: 2.3, w: 6.3, colW: [2.2, 2.0, 2.1], fontSize: 11, color: HEX.dk1, fontFace: THEME.bodyFontFace, border: { type: "solid", pt: 0.5, color: "D9D7E6" }, fill: { color: HEX.lt1 }, margin: 0.06, objectName: oid("table") });
  text(s, "Handles @mentions/URLs, leetspeak (1→i, 3→e, @→a), 3+ repeated letters, spaced-out letters, Turkish casing (I→ı, İ→i) and, for the classical models, ASCII folding (ş→s, ı→i).", { x: 6.4, y: 5.55, w: 6.3, h: 0.9, fontSize: 12, color: C.text2 });
}

// =====================================================================
// 4. Architecture
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Development" });
  s.addText("DEVELOPMENT · ARCHITECTURE (R6)", { placeholder: "kicker" });
  s.addText("From a typed message to a masked, labeled chat line", { placeholder: "title" });
  const y0 = 1.8, H = 4.0;
  const box = (x, w, title, body, fill = HEX.lt2, tcol = C.text1) => {
    card(s, x, y0, w, H, fill);
    text(s, title, { x: x + 0.15, y: y0 + 0.15, w: w - 0.3, h: 0.6, fontSize: 13, bold: true, color: tcol });
    text(s, body, { x: x + 0.15, y: y0 + 0.85, w: w - 0.3, h: H - 1.0, fontSize: 11, color: tcol === C.text1 ? C.text2 : tcol });
  };
  box(0.5, 1.75, "Player\n(web browser)", "Types a chat message and picks a model in the side panel.");
  arrow(s, 2.3, y0 + H / 2, 0.35, 0);
  box(2.7, 1.95, "Gradio UI\napp.py", "Chat room + side panel: latency log, model dropdown, side-by-side comparison of all models.");
  arrow(s, 4.7, y0 + H / 2, 0.35, 0);
  box(5.1, 2.25, "Preprocessing\nsrc/normalize.py", "normalize_light: @mentions, URLs, spaced letters, leetspeak, 3+ repeats\n\nnormalize_full: + Turkish lower-case, ASCII folding, punctuation → tokens");
  arrow(s, 7.4, y0 + H / 2, 0.35, 0);
  // model stack
  card(s, 7.8, y0, 2.85, H, HEX.lt2);
  text(s, "Models · src/predictors.py", { x: 7.95, y: y0 + 0.12, w: 2.6, h: 0.3, fontSize: 13, bold: true });
  const models = [
    ["Lexicon + min. edit distance", "rules"],
    ["tf-idf + log. regression", "FP64 · scikit-learn"],
    ["word2vec + feed-forward", "FP32 · PyTorch"],
    ["word2vec + 1D CNN", "FP32 · PyTorch"],
    ["BERTurk (WordPiece)", "FP32 · PyTorch"],
    ["BERTurk dynamic INT8", "INT8 Linear · PyTorch qnnpack"],
    ["BERTurk INT8 ★ default", "INT8 MatMul · ONNX Runtime"],
  ];
  models.forEach(([m, p], i) => {
    const y = y0 + 0.48 + i * 0.48;
    const last = i === models.length - 1;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.95, y, w: 2.55, h: 0.42, rectRadius: 0.06, fill: { color: last ? HEX.accent1 : HEX.lt1 }, line: { color: last ? HEX.accent1 : "D9D7E6" }, objectName: oid("model") });
    text(s, [{ text: m, options: { bold: true, breakLine: true } }, { text: p, options: { fontSize: 9 } }], { x: 8.05, y: y + 0.01, w: 2.4, h: 0.4, fontSize: 10, color: last ? C.background1 : C.text1, valign: "middle" });
  });
  arrow(s, 10.68, y0 + H / 2, 0.3, 0);
  box(11.02, 1.85, "Post-processing\napp.py", `P(offensive) ≥ threshold (tuned on validation) → label + confidence\n\nMask flagged words or the whole message (src/masking.py)\n\nLog latency vs ${D.budget_ms} ms budget`);
  // return path to the UI
  s.addShape(pres.shapes.LINE, { x: 11.95, y: y0 + H, w: 0, h: 0.3, line: { color: HEX.accent5, width: 1.75 }, objectName: oid("ret") });
  s.addShape(pres.shapes.LINE, { x: 3.65, y: y0 + H + 0.3, w: 8.3, h: 0, line: { color: HEX.accent5, width: 1.75 }, objectName: oid("ret") });
  s.addShape(pres.shapes.LINE, { x: 3.65, y: y0 + H, w: 0, h: 0.3, line: { color: HEX.accent5, width: 1.75, beginArrowType: "triangle" }, objectName: oid("ret") });
  text(s, "labeled / masked chat bubble + latency row", { x: 5.2, y: y0 + H + 0.33, w: 5, h: 0.3, fontSize: 10, color: C.accent5, align: "center" });
  text(s, [
    { text: "Runs on: ", options: { bold: true } },
    { text: `local CPU, 1 thread per message (${D.hardware.split(",")[0]}). ` },
    { text: "Weights: ", options: { bold: true } },
    { text: "Hugging Face Hub (kemalarif/berturk-offenseval-tr), downloaded at start-up by src/hub.py. " },
    { text: "Offline: ", options: { bold: true } },
    { text: "BERTurk trained on the M2 GPU (MPS), other models on CPU; ONNX export + INT8 quantization in src/onnx_export.py." },
  ], { x: 0.5, y: 6.45, w: 12.3, h: 0.5, fontSize: 11, color: C.text2 });
}

// =====================================================================
// 5. Technique map
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Development" });
  s.addText("DEVELOPMENT · TECHNIQUE MAP (R2)", { placeholder: "kicker" });
  s.addText("Every area of the course is used by a component", { placeholder: "title" });
  const hdr = (t) => ({ text: t, options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk1 } } });
  const area = (t, col) => ({ text: t, options: { bold: true, color: HEX.lt1, fill: { color: col }, valign: "middle" } });
  const rows = [
    [hdr("Area"), hdr("Technique"), hdr("Component in the application"), hdr("Code")],
    [area("A · Neural networks", HEX.accent6), "Own word2vec (CBOW, 200-d) trained on the training messages; feed-forward net trained with SGD + backpropagation; 1D CNN (convolution, max-pooling) with dropout", "Neural-network rung of the ladder: word2vec + FFN and word2vec + 1D CNN classifiers", "src/w2v_train.py\nsrc/nn_models.py"],
    [area("B · Classical NLP", HEX.accent3), "Regex normalization (leetspeak, repeated letters, Turkish characters); tokenization; minimum edit distance; tf-idf", "Preprocessing for every model; lexicon rule baseline with edit-distance matching; tf-idf + logistic-regression baseline; masking of flagged words", "src/normalize.py\nsrc/lexicon.py\nsrc/tfidf_train.py"],
    [area("C · Transformers", HEX.accent1), "Fine-tuning BERTurk (bert-base-turkish-cased, 12-layer self-attention) with a classification head; WordPiece subword tokenizer", "Transformer rung: offensive / clean classifier", "src/bert_train.py"],
    [area("D · Efficient models", HEX.accent4), `Post-training dynamic INT8 quantization (PyTorch and ONNX Runtime); ONNX export; latency budget ${D.budget_ms} ms / message on 1 CPU core; benchmarking size, latency, accuracy`, "INT8 BERTurk on ONNX Runtime = the model the app runs by default; results table", "src/onnx_export.py\nsrc/predictors.py\nsrc/evaluate.py"],
  ];
  s.addTable(rows, { x: 0.6, y: 1.8, w: 12.1, colW: [2.0, 4.6, 3.6, 1.9], fontSize: 12, color: HEX.dk1, fontFace: THEME.bodyFontFace, border: { type: "solid", pt: 0.5, color: "D9D7E6" }, fill: { color: HEX.lt1 }, margin: 0.08, valign: "middle", objectName: oid("table") });
}

// =====================================================================
// 6. Model ladder
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Development" });
  s.addText("DEVELOPMENT · THE FOUR MODELS", { placeholder: "kicker" });
  s.addText("A model ladder of increasing sophistication", { placeholder: "title" });
  const th = (k) => fmt(R[k].threshold, 2);
  const cards = [
    ["1", "Classical baseline", C.accent3, [
      "tf-idf: word 1–2-grams + char 2–5-grams on normalize_full text",
      "Logistic regression, class-balanced; C = 4 picked on validation",
      `Threshold ${th("tfidf")}`,
      `Extra rule baseline: ${D.lexicon_size} hand-written offensive roots, edit distance ≤ 1–2`,
    ]],
    ["2", "Neural network", C.accent6, [
      `Own word2vec (CBOW, 200-d, 20 epochs) on the ${D.split_sizes.train.toLocaleString("en-US")} training messages`,
      "1D CNN: filters of width 2/3/4 × 128, max-pool, dropout 0.5",
      "Also: FFN on averaged vectors",
      "SGD + momentum, backprop; best epoch on validation",
    ]],
    ["3", "Transformer", C.accent1, [
      `BERTurk (dbmdz/bert-base-turkish-cased), ${(D.bert_params / 1e6).toFixed(1)}M parameters, 12 layers`,
      "Classification head on [CLS], max 96 WordPiece tokens",
      "AdamW 2e-5, batch 32, 3 epochs, warm-up 6%; Apple M2 GPU (MPS)",
      `Best epoch on validation; threshold ${th("berturk")}`,
    ]],
    ["4", "Quantized Transformer", C.accent4, [
      "The same fine-tuned BERTurk, no retraining",
      "Exported to ONNX, ONNX Runtime dynamic quantization: weights INT8 (per channel)",
      "Also tried: PyTorch dynamic INT8 (qnnpack)",
      "Same threshold as FP32; ONNX INT8 runs in the app",
    ]],
  ];
  cards.forEach(([n, h, col, items], i) => {
    const x = 0.6 + i * 3.08;
    card(s, x, 1.85, 2.88, 4.55);
    badge(s, x + 0.2, 2.05, n, col);
    text(s, h, { x: x + 0.85, y: 2.08, w: 1.95, h: 0.5, fontSize: 15, bold: true, valign: "middle" });
    text(s, items.map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < items.length - 1 } })), { x: x + 0.2, y: 2.8, w: 2.55, h: 3.8, fontSize: 12, color: C.text2, paraSpaceAfter: 8 });
  });
  text(s, "Every model outputs P(offensive); its decision threshold is tuned on the validation set (best macro-F1). The test set is never used for tuning.", { x: 0.6, y: 6.6, w: 12.1, h: 0.3, fontSize: 11, color: C.accent5 });
}

// =====================================================================
// 7. Quantization
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Development" });
  s.addText("DEVELOPMENT · QUANTIZATION (D)", { placeholder: "kicker" });
  s.addText(N.quantTitle(R, D), { placeholder: "title" });
  text(s, "Method", { x: 0.6, y: 1.8, w: 5.6, h: 0.4, fontSize: 20, bold: true, color: C.accent1 });
  text(s, [
    { text: "Post-training dynamic quantization: no retraining, no calibration data.", options: { bullet: true, breakLine: true } },
    { text: `PyTorch: quantize_dynamic on all ${D.n_quantized_linear} nn.Linear layers (INT8 weights, per-tensor scale, qnnpack backend).`, options: { bullet: true, breakLine: true } },
    { text: `ONNX Runtime: exported to ONNX (opset 17), then quantize_dynamic: ${D.onnx_int8_ops.MatMulInteger} MatMul weights (per channel) and the ${D.onnx_int8_ops.DequantizeLinear} embedding tables in INT8; activations quantized on the fly; LayerNorm stays FP32.`, options: { bullet: true, breakLine: true } },
    { text: "Latency: end-to-end, batch size 1, 1 CPU thread, mean over 200 test messages.", options: { bullet: true } },
  ], { x: 0.6, y: 2.3, w: 5.6, h: 3.0, fontSize: 13, paraSpaceAfter: 8 });
  const f = R.berturk, q = R.berturk_onnx_int8;
  const callouts = [
    [`${mb(f.size_MB)} → ${mb(q.size_MB)} MB`, "model size"],
    [`${ms(f.latency_ms_mean)} → ${ms(q.latency_ms_mean)} ms`, "latency / message"],
    [`${fmt(f.macro_F1)} → ${fmt(q.macro_F1)}`, "test macro-F1"],
  ];
  text(s, "FP32 PyTorch → INT8 ONNX Runtime (deployed)", { x: 0.6, y: 5.05, w: 5.6, h: 0.3, fontSize: 12, bold: true, color: C.text2 });
  callouts.forEach(([v, l], i) => {
    const x = 0.6 + i * 1.9;
    card(s, x, 5.4, 1.8, 1.3);
    text(s, v, { x: x + 0.08, y: 5.48, w: 1.64, h: 0.7, fontSize: 15, bold: true, color: C.accent1, align: "center", valign: "middle" });
    text(s, l, { x: x + 0.08, y: 6.2, w: 1.64, h: 0.35, fontSize: 11, color: C.text2, align: "center" });
  });
  const keys = ["berturk", "berturk_int8_torch", "berturk_onnx_fp32", "berturk_onnx_int8"].filter((k) => R[k]);
  const lab = { berturk: "FP32\nPyTorch", berturk_int8_torch: "INT8\nPyTorch", berturk_onnx_fp32: "FP32\nONNX Runtime", berturk_onnx_int8: "INT8 ★ app\nONNX Runtime" };
  const col = { berturk: HEX.accent5, berturk_int8_torch: HEX.accent5, berturk_onnx_fp32: HEX.accent5, berturk_onnx_int8: HEX.accent1 };
  colChart(s, { x: 6.6, y: 1.85, w: 6.1, h: 4.85, title: `BERTurk latency per message, 1 CPU core (ms)`,
    items: keys.map((k) => ({ label: lab[k], value: R[k].latency_ms_mean, color: col[k] })),
    vmax: Math.max(...keys.map((k) => R[k].latency_ms_mean)) * 1.15, budget: D.budget_ms, fmtv: (v) => ms(v) });
}

// =====================================================================
// 8. Results table
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Development" });
  s.addText("DEVELOPMENT · RESULTS (R4)", { placeholder: "kicker" });
  s.addText(`All models on the same held-out test set (n = ${D.n_test.toLocaleString("en-US")})`, { placeholder: "title" });
  const hdr = (t) => ({ text: t, options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk1 }, align: "center" } });
  const rows = [[hdr("Model"), hdr("Precision"), hdr("Macro-F1"), hdr("Accuracy"), hdr("OFF-F1"), hdr("Size (MB)"), hdr("Latency mean (ms)"), hdr("Latency p95 (ms)"), hdr(`≤ ${D.budget_ms} ms`)]];
  const order = ["lexicon", "tfidf", "w2v_ffn", "w2v_cnn", "berturk", "berturk_int8_torch", "berturk_onnx_fp32", "berturk_onnx_int8"];
  const NAME = { lexicon: "Lexicon + edit distance (rules)", tfidf: "tf-idf + logistic regression", w2v_ffn: "word2vec + feed-forward net", w2v_cnn: "word2vec + 1D CNN",
    berturk: "BERTurk · PyTorch", berturk_int8_torch: "BERTurk · PyTorch dynamic", berturk_onnx_fp32: "BERTurk · ONNX Runtime", berturk_onnx_int8: "BERTurk · ONNX Runtime" };
  const PREC = { lexicon: "rules", tfidf: "FP64", w2v_ffn: "FP32", w2v_cnn: "FP32", berturk: "FP32", berturk_int8_torch: "INT8", berturk_onnx_fp32: "FP32", berturk_onnx_int8: "INT8" };
  const bestF1 = Math.max(...D.results.map((r) => r.macro_F1));
  order.filter((k) => R[k]).forEach((k) => {
    const r = R[k];
    const hl = k === "berturk_onnx_int8";
    const o = (extra = {}) => ({ align: "center", ...(hl ? { fill: { color: "E7E4FB" }, bold: true } : {}), ...extra });
    rows.push([
      { text: NAME[k] + (hl ? "  ★ app" : ""), options: { ...o({ align: "left" }) } },
      { text: PREC[k], options: o() },
      { text: fmt(r.macro_F1), options: o(r.macro_F1 === bestF1 ? { color: HEX.accent1, bold: true } : {}) },
      { text: fmt(r.accuracy), options: o() },
      { text: fmt(r.OFF_F1), options: o() },
      { text: mb(r.size_MB), options: o() },
      { text: ms(r.latency_ms_mean), options: o() },
      { text: ms(r.latency_ms_p95), options: o() },
      { text: r.meets_budget === true || r.meets_budget === "True" ? "yes" : "no", options: o({ color: r.meets_budget === true || r.meets_budget === "True" ? HEX.accent3 : HEX.accent2, bold: true }) },
    ]);
  });
  s.addTable(rows, { x: 0.6, y: 1.8, w: 12.1, colW: [3.75, 1.05, 1.0, 1.0, 0.95, 1.0, 1.3, 1.2, 0.85], fontSize: 12, color: HEX.dk1, fontFace: THEME.bodyFontFace, border: { type: "solid", pt: 0.5, color: "D9D7E6" }, fill: { color: HEX.lt1 }, margin: 0.06, valign: "middle", rowH: 0.42, objectName: oid("table") });
  text(s, [
    { text: "Hardware: " + D.hardware + ".", options: { breakLine: true } },
    { text: "Latency: batch size 1, 1 CPU thread, mean and 95th percentile over 200 test messages (10 other messages as untimed warm-up), end-to-end (normalization + tokenization + model). Size: serialized model file / state dict. Macro-F1 with the threshold tuned on validation.", options: { breakLine: !!D.agreement } },
    ...(D.agreement ? [{ text: D.agreement + "." }] : []),
  ], { x: 0.6, y: 5.85, w: 12.1, h: 1.05, fontSize: 10, color: C.accent5 });
}

// =====================================================================
// 9. Comparison takeaways
pres.addSection({ title: "Conclusion" });
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Conclusion" });
  s.addText("CONCLUSION · WHAT THE COMPARISON SHOWED", { placeholder: "kicker" });
  s.addText(N.compareTitle(R, D), { placeholder: "title" });
  const order = ["lexicon", "tfidf", "w2v_ffn", "w2v_cnn", "berturk", "berturk_int8_torch", "berturk_onnx_int8"].filter((k) => R[k]);
  const short = { lexicon: "Lexicon rules", tfidf: "tf-idf + LR", w2v_ffn: "w2v + FFN", w2v_cnn: "w2v + 1D CNN", berturk: "BERTurk FP32", berturk_int8_torch: "INT8 PyTorch", berturk_onnx_int8: "INT8 ONNX RT ★" };
  const colr = { lexicon: HEX.accent3, tfidf: HEX.accent3, w2v_ffn: HEX.accent6, w2v_cnn: HEX.accent6, berturk: HEX.accent5, berturk_int8_torch: HEX.accent5, berturk_onnx_int8: HEX.accent1 };
  barChartH(s, { x: 0.6, y: 1.85, w: 6.1, h: 4.85, title: "Test macro-F1 (higher is better)",
    items: order.map((k) => ({ label: short[k], value: R[k].macro_F1, color: colr[k], bold: k === "berturk_onnx_int8" })),
    vmax: 1.0, fmtv: (v) => fmt(v) });
  N.takeaways(R, D).forEach(([h, b, col], i) => {
    const y = 1.8 + i * 1.25;
    card(s, 7.0, y, 5.7, 1.1);
    badge(s, 7.15, y + 0.3, String(i + 1), col, 0.48);
    text(s, h, { x: 7.8, y: y + 0.1, w: 4.75, h: 0.35, fontSize: 14, bold: true });
    text(s, b, { x: 7.8, y: y + 0.45, w: 4.75, h: 0.6, fontSize: 11, color: C.text2 });
  });
}

// =====================================================================
// 10. Limitations, failure examples, next steps
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Conclusion" });
  s.addText("CONCLUSION · LIMITATIONS AND NEXT STEPS", { placeholder: "kicker" });
  s.addText("Where the deployed model still fails", { placeholder: "title" });
  text(s, "Test-set errors of the deployed INT8 model", { x: 0.6, y: 1.8, w: 6.2, h: 0.4, fontSize: 16, bold: true, color: C.accent1 });
  N.failures(D).forEach((f, i) => {
    bubble(s, 0.6, 2.3 + i * 1.45, 6.2, f.text, f.verdict, f.truthOffensive, { h: 1.25, fs: 12 });
  });
  text(s, "Limitations", { x: 7.2, y: 1.8, w: 5.5, h: 0.4, fontSize: 16, bold: true, color: C.accent1 });
  text(s, N.limitations(R, D).map((t, j, a) => ({ text: t, options: { bullet: true, breakLine: j < a.length - 1 } })), { x: 7.2, y: 2.25, w: 5.5, h: 2.3, fontSize: 12, color: C.text2, paraSpaceAfter: 6 });
  text(s, "Next steps", { x: 7.2, y: 4.6, w: 5.5, h: 0.4, fontSize: 16, bold: true, color: C.accent1 });
  text(s, N.nextSteps(R, D).map((t, j, a) => ({ text: t, options: { bullet: true, breakLine: j < a.length - 1 } })), { x: 7.2, y: 5.05, w: 5.5, h: 1.8, fontSize: 12, color: C.text2, paraSpaceAfter: 6 });
}

// =====================================================================
// 11. AI usage
{
  const s = pres.addSlide({ masterName: "CLOSING_DARK", sectionTitle: "Conclusion" });
  s.addText("CONCLUSION · AI USAGE STATEMENT (R8)", { placeholder: "kicker" });
  s.addText("How AI assistants were used", { placeholder: "title" });
  const cols = [
    ["Tool", ["Claude Code (Anthropic, Claude Opus 5.5) in the Claude desktop app"]],
    ["Used for", N.aiUsedFor],
    ["Done and checked by me", N.aiMine],
  ];
  cols.forEach(([h, items], i) => {
    const x = 0.6 + i * 4.1;
    card(s, x, 1.85, 3.9, 4.2, HEX.dk2);
    text(s, h, { x: x + 0.25, y: 2.05, w: 3.4, h: 0.4, fontSize: 17, bold: true, color: C.accent4 });
    text(s, items.map((t, j) => ({ text: t, options: { bullet: items.length > 1, breakLine: j < items.length - 1 } })), { x: x + 0.25, y: 2.6, w: 3.45, h: 3.9, fontSize: 13, color: C.background1, paraSpaceAfter: 8 });
  });
}

(async () => {
  const out = path.join(__dirname, "SWE015_ChatModeration.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("wrote", out);
})();
