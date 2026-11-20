// Slide wording that interprets the results. Every number is read from results/deck_data.json.
const f3 = (x) => Number(x).toFixed(3);
const f1 = (x) => Number(x).toFixed(1);
const pct = (x) => `${(100 * x).toFixed(1)}%`;

// Failure examples shown on the limitations slide: picked by hand from the test-set errors that
// src/report.py lists (index into false_negatives / false_positives).
const PICK = {
  fn: 10, fnWhy: (r) => `insult inside a complaint; tf-idf caught it (P = ${f3(r.p_tfidf)})`,
  fp: 1, fpWhy: () => "the keyboard mash “aqsdf” looks like the slur abbreviation “aq”",
};

module.exports = {
  quantTitle: (R) => {
    const f = R.berturk, q = R.berturk_onnx_int8;
    const smaller = f.size_MB / q.size_MB, faster = f.latency_ms_mean / q.latency_ms_mean;
    const dF1 = q.macro_F1 - f.macro_F1;
    return `INT8 on ONNX Runtime: ${f1(smaller)}× smaller, ${f1(faster)}× faster, ${dF1 >= 0 ? "+" : "−"}${f3(Math.abs(dF1))} macro-F1`;
  },

  compareTitle: (R) =>
    `BERTurk is the most accurate; tf-idf is a strong, very cheap baseline`,

  takeaways: (R, D) => {
    const b = R.berturk, t = R.tfidf, l = R.lexicon, c = R.w2v_cnn, n = R.w2v_ffn;
    const qt = R.berturk_int8_torch, qo = R.berturk_onnx_int8;
    return [
      ["Transformer wins on accuracy",
        `BERTurk ${f3(b.macro_F1)} vs tf-idf ${f3(t.macro_F1)} macro-F1: pre-trained Turkish context and WordPiece subwords handle insults without swear words.`,
        "6C5CE7"],
      ["Classical baseline: best value per ms",
        `tf-idf + char n-grams reaches ${f3(t.macro_F1)} at ${f1(t.latency_ms_mean)} ms and ${f1(t.size_MB)} MB; char n-grams catch suffixes and misspellings.`,
        "17A67E"],
      ["Our word2vec networks lose",
        `CNN ${f3(c.macro_F1)} / FFN ${f3(n.macro_F1)}, below the lexicon (${f3(l.macro_F1)}): word2vec saw only ~0.44M training tokens, too little for good vectors.`,
        "2F9BD6"],
      ["Quantization: the runtime matters",
        `PyTorch INT8 kept accuracy but took ${f1(qt.latency_ms_mean)} ms (slower than FP32 on ARM); ONNX Runtime INT8 needs ${f1(qo.latency_ms_mean)} ms, within the ${D.budget_ms} ms budget.`,
        "F2A93B"],
    ];
  },

  failures: (D) => {
    const fn = D.false_negatives[PICK.fn], fp = D.false_positives[PICK.fp];
    return [
      { text: `“${fn.text}”`, verdict: `MISSED (true: offensive) · P(off) = ${f3(fn.p_int8)} · ${PICK.fnWhy(fn)}`, truthOffensive: true },
      { text: `“${fp.text}”`, verdict: `FALSE ALARM (true: clean) · P(off) = ${f3(fp.p_int8)} · ${PICK.fpWhy(fp)}`, truthOffensive: false },
    ];
  },

  limitations: (R, D) => [
    "Domain shift: trained on 2019 Turkish tweets, deployed on gaming chat (different slang, names, topics).",
    "Labels are subjective; the errors we inspected are mostly implicit insults, slang and keyboard noise.",
    "Masking uses the lexicon, so offensive messages without lexicon words are hidden completely.",
    "Latency measured on one machine (Apple M2, ARM); an x86 server would give different numbers.",
  ],

  nextSteps: (R, D) => [
    "Distill BERTurk into a smaller student (4–6 layers) for more latency headroom.",
    "Collect in-domain chat messages and moderator feedback; retrain and re-tune thresholds.",
    "Train word2vec / fastText on a large Turkish corpus to strengthen the neural rung.",
  ],

  aiUsedFor: [
    "Writing and debugging the Python code: data pipeline, training scripts, ONNX export, evaluation and the Gradio app",
    "Diagnosing problems (gensim skip-gram BLAS error, slow PyTorch INT8 on ARM)",
    "Drafting the README and this slide deck",
  ],
  aiMine: [
    "Chose the use case, the data set and the 50 ms latency budget",
    "Ran every training and benchmark myself; all numbers come from these runs (results/ in the repo)",
    "Reviewed the code and outputs and can explain every component of the architecture",
  ],
};
