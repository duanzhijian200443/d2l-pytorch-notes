# NTL v11 Candidate | Sinkhorn alpha=0.25

**Public LB 0.552.** The base notebook, Lucas Boesen's [LoRA x Adaptive Centering](https://www.kaggle.com/code/lucasboesen/lora-x-adaptive-centering), scores 0.549.

## What I added on top of the 0.549 blend
1. **A third Qwen3-8B LoRA with a within-Claude head:** [dataset](https://www.kaggle.com/datasets/zbigniewkirchner/ntl-qwen3-8b-lora-claude-head). Same 24-layer recipe as the public LoRAs, plus an auxiliary **5-way Claude head** trained on the original Claude rows + the free public [Claude API passages](https://www.kaggle.com/datasets/seyitkaangunes/ntl-extra-api-passages). The 18-way heads see only original data. On the public LoRA's own `outer0` validation rows:

   | | balanced acc |
   |---|---|
   | public lora24 (plain) | 0.4559 |
   | **this LoRA (plain)** | **0.4838** |
   | **this LoRA + Claude-head gate** | **0.4932** |

   It joins the blend as `lora3` (weight 0.177, the same as the public LoRA) and adds its centered twin on clustered rows.
2. **Claude gates.** When the final blend predicts *a* Claude model, two within-Claude models decide *which* one: a stylegram LinearSVC trained on Claude-only text (original + API passages), and the LoRA's Claude head. Non-Claude predictions are never touched, so the extra data can't bias the other 13 classes. In my grouped CV on the stylegram model: 0.3845 → 0.3967.

| Version | Public LB |
|---|---|
| fork of public 0.541 | 0.541 |
| + Claude gate | 0.544 |
| + own LoRA instead of the public one | 0.547 |
| **0.549 base + own LoRA as `lora3` + both gates (this)** | **0.552** |

Please also upvote the work this builds on: Lucas Boesen (blend, adaptive centering), Vladislav Lavrentev (public LoRA), kirilltym (token attention), pjmathematician (UltraBERT), seyitkaangunes (Claude API passages). Apache 2.0, as the source.

If this helps you, an upvote is appreciated 🙂

---
### Original description (Lucas Boesen)

# LoRA x Adaptive Centering

A log-probability blend of style models, frozen language-model heads, frozen encoders, three fine-tuned classifiers and two learned token-attention heads, plus **adaptive prompt-centering** at test time. Each version of this notebook is one of our leaderboard submissions.

## Version notes
- **Version 3 (public LB 0.549):** version 2 plus two members.
  - Our own run of the LoRA24 recipe (seed 7, on our folds), whose out-of-fold accuracy (0.480) matches the public run's.
  - The same learned token-attention head, on frozen DeBERTa-v3-large layer-23 token states.

  Held-out blend +0.006; pooled sim about even with version 2. Public LB went from 0.547 to 0.549.
- **Version 2 (public LB 0.547):** version 1 plus a token-attention head on frozen Qwen3.5-9B layer-18 token states (kirilltym's recipe, re-trained on our folds), plus the LoRA's centered readout as an extra twin on clustered test rows.
- **Version 1 (public LB 0.541):** frozen encoders + adaptive centering (0.532) plus the public LoRA24 classifier.

## How it works
1. **Style models.**
   - Style-aware character/word/shape/function-word n-grams with a LinearSVC.
   - Stylometric features with LightGBM.
2. **Frozen LM heads.** Mean-pooled hidden states with StandardScaler + LogisticRegression, from several open models (Qwen3-8B/4B-Base, Qwen2.5-7B, GLM-4-9B-0414). Each model is truncated to the last layer it needs.
3. **Frozen bidirectional encoders.** DeBERTa-v3-large, ELECTRA-large and an AI-text-detector DeBERTa, mean- and style-token-pooled, each with a LogisticRegression head.
4. **Fine-tuned classifiers.**
   - The public UltraBERT-380M 5-fold ensemble.
   - A public LoRA classifier on Qwen3-8B-Base: 24 layers, 8 pooled layers, each with its own LayerNorm.
5. **Learned token attention.** A small attention-pooling head over frozen Qwen3.5-9B layer-18 token states.
6. **Test-time steps.** These use no labels.
   - Test-set standardization of the LM features.
   - Topic clustering of the test passages: TF-IDF plus GLM-4-9B layer-24 cosine, average linkage.
   - Inside clusters of 2-18 passages, prompt-centered "twin" heads add evidence. The twins are trained on prompt-centered train features and applied to cluster-centered test features.
   - Sinkhorn class balancing.

Blend weights are fit with L-BFGS on 5-fold prompt-grouped out-of-fold predictions. Candidates are chosen on a pooled simulation with 1-8 passages per prompt. Its predicted gains have matched our public-LB gains to within about 0.002-0.004 per version.

## Credits (thank you!)
- **LoRA24 classifier and checkpoint:** Vladislav Lavrentev, [lightningv08/ntl-lora24-fivefold-training](https://www.kaggle.com/code/lightningv08/ntl-lora24-fivefold-training). We use its public `final` checkpoint for inference and its 5-fold OOF to fit our blend weight.
- **Learned token attention (recipe):** Кирилл Тымчишин, [kirilltym/0-525-learned-token-attention](https://www.kaggle.com/code/kirilltym/0-525-learned-token-attention). We re-trained the head on our folds.
- **Frozen-encoder idea (DeBERTa + LR):** Rudra, [rudrabhaskar](https://www.kaggle.com/rudrabhaskar)'s fusion notebooks, which we found via Kizabgd123's [Apex Quintuple Ensemble](https://www.kaggle.com/code/kiza123123/name-that-llm-apex-quintuple-ensemble).
- **Adding centered evidence from cluster size 2:** Rudra, [0.490 | Qwen3 Fusion](https://www.kaggle.com/code/rudrabhaskar/0-490-highest-public-notebook-qwen3-fusion).
- **UltraBERT-380M 5-fold weights:** pjmathematician, [pjmathematician/ultrabert-380m-ntl](https://www.kaggle.com/datasets/pjmathematician/ultrabert-380m-ntl). The model code is from dmis-lab.
- **Style-aware n-gram recipe:** Yang Kuang Ou, [yangkuangou/name-that-llm-style-aware-char-n-grams-cv-0-384](https://www.kaggle.com/code/yangkuangou/name-that-llm-style-aware-char-n-grams-cv-0-384).
- **Qwen3.5-9B weights:** [bbucxi/qwen35-9b-safetensors](https://www.kaggle.com/datasets/bbucxi/qwen35-9b-safetensors).
- **Models:**
  - Qwen3 and Qwen2.5 (Qwen team)
  - GLM-4-9B-0414 (Zhipu AI / zai-org)
  - microsoft/deberta-v3-large
  - google/electra-large-discriminator
  - desklib/ai-text-detector-v1.01
- **Our own LoRA24 run (seed 7):** [name-that-llm-train-lora24-s7](https://www.kaggle.com/code/lucasboesen/name-that-llm-train-lora24-s7). It follows lightningv08's recipe exactly.
- **DeBERTa token-attention training:** [name-that-llm-train-tokattn2](https://www.kaggle.com/code/lucasboesen/name-that-llm-train-tokattn2)
- **Our own training and weight notebooks (all public):**
  - [GLM-4-9B truncation](https://www.kaggle.com/code/lucasboesen/name-that-llm-glm4-9b-trunc25)
  - [encoder weights](https://www.kaggle.com/code/lucasboesen/name-that-llm-extract-encoders2)
  - [ELECTRA weights](https://www.kaggle.com/code/lucasboesen/name-that-llm-electra-large-weights)
  - [desklib weights](https://www.kaggle.com/code/lucasboesen/name-that-llm-desklib-weights)
  - [token-attention training](https://www.kaggle.com/code/lucasboesen/name-that-llm-train-tokattn)

Settings: GPU T4 x2, internet off. The full test run takes about 1.5 hours.
