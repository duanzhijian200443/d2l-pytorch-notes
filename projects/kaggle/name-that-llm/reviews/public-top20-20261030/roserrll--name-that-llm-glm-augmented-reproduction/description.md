# Name That LLM | Who am I? — V31 direct improvement candidate

**Starting point:** the uploaded V30 notebook reports Public LB **0.654**. This V31 version keeps the original public model checkpoints and adapters, and makes targeted code/method improvements; its final score is **not yet verified** until a Kaggle commit-and-run completes.

**Original acknowledgements and licenses:** [dhruvtibarewal — nlm-v27](https://www.kaggle.com/code/dhruvtibarewal/nlm-v27), [kirilltym — 0.639 Teach More, Keep the Signature](https://www.kaggle.com/code/kirilltym/0-639-teach-more-keep-the-signature), [Spectral Style Heads Paired 4404](https://www.kaggle.com/datasets/kirilltym/ntl-spectral-style-heads-paired-4404), [fleongg — NTL Part Two Complement](https://www.kaggle.com/code/fleongg/ntl-part-two-complement), [Nishant Dahal — Two-Stage LoRA Part Two](https://www.kaggle.com/code/nishantdahal/two-stage-lora-part-two), and the original authors of all public model artifacts. All original artifacts remain credited to their authors; do not redistribute gated weights outside their terms.

## Direct changes in V31

1. **Use the matched-clean external data instead of loading and ignoring it:** the existing TF-IDF and topic-distortion branches are enabled at low auxiliary weights; exact normalized duplicates are de-duplicated with official rows taking precedence. The four-view SVC now uses official + matched-clean + pool rows with source weights.
2. **Activate the trained family gates already present in `comp_stack.py`:** previously the gate models and `GATE_LAM` were created but never affected the final logits. V31 applies them only to within-family ranking, with confidence gating, while preserving each family's overall probability mass.
3. **Restore Qwen3.5 Token-Attention when its head and matching backbone are available.** If either asset is missing, the notebook logs a skip rather than failing. This may add roughly 1.7–2.7 hours; the ATLAS budget is reduced dynamically, but the full runtime must still be checked against Kaggle's 9-hour limit.
4. **Replace full uniform Sinkhorn with damped adaptive marginal balancing:** the competition states test class counts are unequal and Public/Private mixes differ, so V31 does not force every class to exactly 1/18 of test predictions.
5. **Strengthen Bayesian typography evidence:** class-specific marker likelihoods shrink toward global marker rates; correction strength is capped and reduced for short texts, sparse style markers, and high-confidence base predictions. Diagnostics no longer claim this is OOF-validated.

The changes are intentionally targeted; they do not retrain the public Decoder adapters. The output is still `/kaggle/working/submission.parquet`; the original V30 notebook is left untouched.
