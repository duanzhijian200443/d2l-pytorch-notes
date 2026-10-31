
# CELL 0
# Name That LLM | Who am I? — V31 (Public LB 0.660)

**Reported result:** the user reports that this V31 candidate achieved **0.660 Public LB** on Kaggle. V30's **0.654** result was separately reproduced. This documentation update does not change the inference code or claim a new Kaggle run.

**Purpose:** reuse public pretrained checkpoints, LoRA adapters, fitted feature heads and public data assets, then improve their inference-time use. The V31 changes are limited to auxiliary data usage, family-level gating, adaptive probability balancing, and confidence-aware Bayesian typography. No original pretrained weights or adapters are modified.

**Acknowledgements:** this notebook builds on [dhruvtibarewal's nlm-v27](https://www.kaggle.com/code/dhruvtibarewal/nlm-v27), [kirilltym's 0.639 Teach More, Keep the Signature](https://www.kaggle.com/code/kirilltym/0-639-teach-more-keep-the-signature), [Spectral Style Heads Paired 4404](https://www.kaggle.com/datasets/kirilltym/ntl-spectral-style-heads-paired-4404), [fleongg's NTL Part Two Complement](https://www.kaggle.com/code/fleongg/ntl-part-two-complement), and [Nishant Dahal's Two-Stage LoRA Part Two](https://www.kaggle.com/code/nishantdahal/two-stage-lora-part-two). The detailed method and asset attribution table follows this introduction.


# CELL 1
## Methods used in this notebook

The final predictor is a **multi-model, 18-class ensemble**. It combines public task-trained Decoder adapters, Encoder classifiers, frozen-hidden-state readout heads, sparse text-style classifiers, prompt-aware features, a low-confidence ATLAS correction, adaptive marginal balancing, and a confidence-gated typography correction. It does not train a new large Decoder in this notebook.

| Component | What the code does | Attribution / origin |
|---|---|---|
| Multi-fold Decoder ensemble | Fold-averaged Qwen3-14B v5, Qwen3-8B v5 and Llama-3.1-8B v5 adapters, plus two-stage Qwen3-14B and Qwen2.5-14B adapters; multi-layer hidden-state pooling and weighted log-probability fusion | dhruvtibarewal; Nishant Dahal; original Qwen and Meta model authors |
| Four-view N-gram SVC | Character, word, shape/punctuation and function-word style views; uses official labels and selected public pools with source-aware weights and duplicate filtering | Public N-gram / style-classification recipe; Yang Kuang Ou and upstream recipe authors; `nlm_pool_v5` from dhruvtibarewal |
| Complementary stack | StyleGram, Stylometry LightGBM, UltraBERT-380M, DeBERTa/ELECTRA/Desklib heads, frozen Qwen/GLM feature readouts, LoRA feature heads and Token-Attention heads | prvsiyan, Lucas Boesen, Travis Nguyen (`vnton37nguyn`), Kirill Tymchishin, Zbigniew Kirchner, pjmathematician and other checkpoint authors listed below |
| Prompt-aware features | Prompt-centred twins and shared-prompt Gaussian group heads; test prompt groups are inferred from text rather than read from hidden test labels | prvsiyan; Kirill Tymchishin and upstream group-Gaussian implementation authors |
| ATLAS correction | Adds source-conditioned likelihood evidence for low-margin rows (margin threshold 0.3; configured mix weight 0.10) | Ko et al., *Who's Behind the Text? Robust Model Authorship Attribution*, arXiv:2610.19583; Joseph Adamski's public bundle/notebook |
| Text distortion / TF-IDF auxiliaries | Low-weight TF-IDF and topic-masked text-distortion classifier branches | Efstathios Stamatatos, “Authorship Attribution Using Text Distortion” (EACL 2017); upstream implementation authors |
| Adaptive Sinkhorn | Partial marginal balancing (`sinkhorn=0.45`, target-mix parameter `0.35`) rather than forcing exact uniform 1/18 test-class proportions | Current notebook's adaptation of the upstream probability-balancing stage |
| Family gates | Applies confidence-aware corrections within a model family while preserving the family's total probability mass | Current V31 code activates previously fitted gate outputs; underlying family/style features are from the public complementary stack |
| Bayesian typography | Hierarchical Beta-Bernoulli-smoothed class-specific marker likelihoods, discriminative-marker weighting, bounded evidence, and confidence/length/coverage gates | New V31 inference-time adaptation; inspired by general Bayesian likelihood modelling, not claimed as a verbatim upstream method |
| Final assignment | The V31 command sets `--post 0`; it does **not** force the original hard typography guards or Hungarian one-to-one assignment in the final candidate | Explicit V31 configuration choice |

## Public notebooks, datasets and model-artifact credits

| Author / project | Reused idea or asset | Public reference |
|---|---|---|
| dhruvtibarewal | Original `nlm-v27`, v5 multi-fold LoRA adapters, matched-clean corpus, `nlm_pool_v5` | [Notebook](https://www.kaggle.com/code/dhruvtibarewal/nlm-v27) · [v5 adapters](https://www.kaggle.com/datasets/dhruvtibarewal/nlm-l24-adapters-v5) · [matched-clean](https://www.kaggle.com/datasets/dhruvtibarewal/nlm-matched-clean) · [older adapters](https://www.kaggle.com/datasets/dhruvtibarewal/nlm-l24-adapters) |
| Kirill Tymchishin (`kirilltym`) | Spectral Style Heads and paired synthetic feature bank; style-feature research | [0.639 notebook](https://www.kaggle.com/code/kirilltym/0-639-teach-more-keep-the-signature) · [paired 4404 dataset](https://www.kaggle.com/datasets/kirilltym/ntl-spectral-style-heads-paired-4404) · [Prompt-removal / two-branch work](https://www.kaggle.com/code/kirilltym/0-600-teach-the-model-remove-the-prompt) |
| fleongg | NTL Part Two Complement reference ensemble | [Notebook](https://www.kaggle.com/code/fleongg/ntl-part-two-complement) |
| Nishant Dahal | Two-stage public-pool LoRA and DeBERTa-related assets | [Two-Stage LoRA Part Two](https://www.kaggle.com/code/nishantdahal/two-stage-lora-part-two) · [handwriting blend assets](https://www.kaggle.com/datasets/nishantdahal/ntl-handwriting-blend) |
| prvsiyan | Shared-prompt Gaussian and prompt-aware attribution | [The Prompt Is Not the Author](https://www.kaggle.com/code/prvsiyan/the-prompt-is-not-the-author) |
| Lucas Boesen (`lucasboesen`) | LoRA24 seed-7 training, Token-Attention heads, ELECTRA and Desklib feature/readout assets | [LoRA24 seed 7](https://www.kaggle.com/code/lucasboesen/name-that-llm-train-lora24-s7) · [Token-Attention](https://www.kaggle.com/code/lucasboesen/name-that-llm-train-tokattn) · [Token-Attention v2](https://www.kaggle.com/code/lucasboesen/name-that-llm-train-tokattn2) · [ELECTRA assets](https://www.kaggle.com/code/lucasboesen/name-that-llm-electra-large-weights) · [Desklib assets](https://www.kaggle.com/code/lucasboesen/name-that-llm-desklib-weights) |
| Travis Nguyen (`vnton37nguyn`) | Qwen2.5 In-Style LoRA / training assets | [In-Style ensemble](https://www.kaggle.com/code/vnton37nguyn/ntl-lora-instyle-0-563) |
| Zbigniew Kirchner (`zbigniewkirchner`) | Claude-specialist and Qwen3-8B In-Style LoRA heads | [Claude head dataset](https://www.kaggle.com/datasets/zbigniewkirchner/ntl-qwen3-8b-lora-claude-head) · [In-Style Qwen3-8B](https://www.kaggle.com/code/zbigniewkirchner/0-574-in-style-qwen3-8b-lora-fully-public) |
| pjmathematician | UltraBERT-380M trained classifier | [UltraBERT dataset](https://www.kaggle.com/datasets/pjmathematician/ultrabert-380m-ntl) |
| Joseph Adamski | ATLAS implementation bundle and early-layer / low-confidence inference assets | [ATLAS Qwen3-8B bundle](https://www.kaggle.com/datasets/josephadamski91/ntl-atlas-qwen3-8b-v1) · [Teach the Model + 10% ATLAS](https://www.kaggle.com/code/josephadamski91/teach-the-model-10-atlas) |
| Ko, Lee, Song, Kim, Shin & Kim (2026) | ATLAS method: source-conditioned likelihood for robust model authorship attribution | [arXiv:2610.19583](https://arxiv.org/abs/2610.19583) · [reference code](https://github.com/komseok0109/ATLAS) |
| Stamatatos (2017) | Text-distortion authorship-attribution method referenced by the auxiliary branch | E. Stamatatos, “Authorship Attribution Using Text Distortion,” EACL 2017 |
| Original base-model authors | Qwen, Meta/Llama, Microsoft/DeBERTa, Google/ELECTRA and the other original model developers | Each attached base model retains its original model card and license |

The notebook also consumes public extra-passage datasets when they are attached as Kaggle Inputs, including resources from Seyit Kaan Güneş, Abdullah Afify and Do Phuc Hung. The loader filters labels, response length, stop/finish status where available, and duplicate `(prompt_id, model)` pairs. Each such dataset retains its own original provenance and terms.

## Attribution and license notes

- This notebook reuses and adapts public code, model weights, trained adapters, fitted feature banks and datasets. It is not a claim of authorship over those upstream assets. Citations above identify the major reused components; the original Kaggle Input metadata remains part of the provenance record.
- Each asset retains its own license and terms. Do not redistribute checkpoints, fitted banks or datasets unless their terms allow it. In particular, Llama 3.1-derived weights/assets remain subject to Meta's Llama 3.1 Community License and its applicable notice requirements; this notebook does not relicense them.
- The `nlm_pool_v5` / v5-adapter training recipe is not fully documented in the upstream notebook. The current score is a Kaggle Public LB result, not a strict OOF estimate and not proof that any one component caused the full gain.
- V31 uses the official competition training labels for final inference-time typography statistics. This is part of the submission pipeline, not an OOF validation claim. The source code does not establish independent OOF validation for that correction.
- Reported Public LB **0.660** is the user's observed result. This notebook's documentation is not a separate Kaggle execution record.
