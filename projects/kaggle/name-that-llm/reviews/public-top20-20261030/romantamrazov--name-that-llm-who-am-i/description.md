# Name That LLM | Who am I? — V40A: OOF sibling-style residual

**Reference:** user-reported 0.660 submission, derived directly from the uploaded `name-that-llm-submission.ipynb`. The file's inherited introductory Markdown mentions an older 0.654 checkpoint; this version uses its actual code, not an assumed leaderboard verification.

**Change:** add a conservatively weighted family-only stylometry expert. Its reliability is estimated with **5-fold `GroupKFold(prompt_id)`**, and it reranks siblings while preserving the probability mass assigned to each model family. Existing LoRA, ATLAS, Complement, training datasets and their weights are unchanged. Writes `candidate_reference_0660.parquet` before the new correction and writes `submission.parquet` after it.

**Acknowledgements:**
- [mercurydust — Name That LLM Submission](https://www.kaggle.com/code/mercurydust/name-that-llm-submission) — the supplied public notebook and ensemble design, used as the direct basis.
- [dhruvtibarewal — nlm-v27](https://www.kaggle.com/code/dhruvtibarewal/nlm-v27) — two-stage LoRA ensemble components.
- [kirilltym — 0.639 Teach More, Keep the Signature](https://www.kaggle.com/code/kirilltym/0-639-teach-more-keep-the-signature) and [Spectral Style Heads Paired 4404](https://www.kaggle.com/datasets/kirilltym/ntl-spectral-style-heads-paired-4404) — style/representation components.
- [fleongg — NTL Part Two Complement](https://www.kaggle.com/code/fleongg/ntl-part-two-complement) — complementary stack.
- [Nishant Dahal — Two-Stage LoRA Part Two](https://www.kaggle.com/code/nishantdahal/two-stage-lora-part-two) — additional two-stage components.
- Original authors of every public checkpoint, head and external dataset remain credited; their original licensing terms apply.