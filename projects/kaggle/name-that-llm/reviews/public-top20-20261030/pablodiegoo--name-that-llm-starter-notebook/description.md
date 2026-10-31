# Name That LLM: Transductive Adaptive Centering, Sinkhorn-Knopp & Multi-Backbone LoRA Attribution

This **Starter Notebook** implements an offline-ready (`enable_internet: false`), multi-stage authorship attribution pipeline for the **Name That LLM** competition (`2,699` training generations across `18` frontier and open-weights Large Language Models and `180` prompts, evaluated by multiclass **Balanced Accuracy**).

### Architectural & Empirical Progression

| Stage / Specification | Representation & Post-Processing | 5-Fold `GroupKFold` (`prompt_id`) OOF | Public LB Benchmark |
| :--- | :--- | ---: | ---: |
| **1. Lexical Baseline** | Word `TF-IDF` (`1..3`-grams, sublinear) + Multinomial `LogisticRegression` | `0.3145 +/- 0.0153` | — |
| **2. Stylometric + Multi-Gram Union** | `20` punctuation/cadence ratios (`StylometricTransformer`) + Word & Char `TF-IDF` | `0.3367 +/- 0.0166` | `0.330` (`v1`) |
| **3. Multi-View + Stamatatos Distortion** | `LogisticRegression` + `LinearSVC` + Topic-Masked Text Distortion (`top_k=100` function words) | `0.3686 +/- 0.0155` | — |
| **4. Prompt-Clustered Adaptive Centering** | Average-linkage cosine clustering on content words (`sim_thr=0.15`) + cluster-mean subtraction (`alpha=0.65`) | `0.3812 +/- 0.0173` | — |
| **5. Damped Sinkhorn-Knopp + Centering** | Iterative biproportional marginal scaling (`alpha=0.45`, `rho=0.35`, `100` steps) + Cluster Hungarian (`linear_sum_assignment`) | `0.3834` (`0.4345` Hungarian) | — |
| **6. Full Multi-Backbone GPU Ensemble (`v2`)** | Fold-averaged Two-Stage `LoRA24` (`Qwen3-14B`, `Qwen3-8B`, `Llama-3.1-8B`, `Qwen2.5-14B`, `DeBERTa-v3-Large`) + Shared-Prompt Gaussian (`gauss.py`) + Complementary Stack (`15%`) + `ATLAS` (`20%`) + Damped Sinkhorn-Knopp (`0.45 / 0.35`) + `V31` Bayesian Typography + `V40A` OOF Sibling-Style Residual | `0.6865` (nested CV) | **`0.662 - 0.663`** |

## 1. Exploratory Stylometric Profile Across the 18 LLMs

Before fitting lexical and neural models, we inspect how response length (word count), sentence cadence, and typographical markers (`\u2014`, `;`, `:`) vary systematically across the 18 candidate language models.

## 2. 5-Fold `GroupKFold` Transductive Ablation (Adaptive Centering, Sinkhorn-Knopp & Hungarian)

Because responses to the same prompt share topical vocabulary and length tendencies, raw classifiers conflate *what was asked* (prompt effect $p_g$) with *who answered* (author signature $\mu_c$). On unseen validation prompts (`5-Fold GroupKFold` grouped by `prompt_id`), we evaluate:
1. **Multi-View Linear Ensemble + Stamatatos Text Distortion:** Combining case-sensitive word/character n-grams (`LogisticRegression` + `LinearSVC`), 20 stylometric ratios, and a topic-masked text-distortion view that retains only the top 100 function words and punctuation.
2. **Prompt-Clustered Adaptive Centering (`adaptive_center_log_probs`):** Recovering latent prompt clusters via average-linkage cosine similarity on stop-word-filtered TF-IDF (`sim_thr = 0.15`) and subtracting the cluster mean log-probability profile.
3. **Damped Sinkhorn-Knopp Marginal Balancing (`sinkhorn_knopp_balance`):** Iterative biproportional scaling (`100` rounds, $\alpha = 0.45$) toward a shrinkage target $(1 - \rho)\hat{\pi}_{\text{pred}} + \rho \pi_{\text{train}}$ ($\rho = 0.35$) that respects unequal class mixes on the test board.

## 3. Multi-Backbone Input Verification & Hardware Setup

When running on Kaggle with attached GPU models and datasets, we verify the mounted checkpoints for the fold-averaged two-stage `LoRA24` trunks (`Qwen3-14B-Base`, `Qwen3-8B-Base`, `Llama-3.1-8B-Base` from `alexchilton/llama-3-1-8b-base`, `Qwen2.5-14B`, and `DeBERTa-v3-Large`), the `ATLAS` source-conditioned likelihood bundle (`josephadamski91/ntl-atlas-qwen3-8b-v1`), and the external matched pools (`nlm_matched_clean.parquet` and `nlm_pool_v5.parquet`).

## 4. Shared-Prompt Gaussian Group Head (`gauss.py`) & OOF Sibling-Style Residual (`sibling_style.py`)

We materialize two specialized transductive modules:
- **`gauss.py`:** Models hidden-state embeddings as $x = \mu_c + p_g + \varepsilon$, jointly diagonalizing within-class noise covariance $\Sigma_e$ and prompt covariance $\Sigma_p$ via Ledoit-Wolf shrinkage and computing exact injective group marginals ($n \le 3$) or cavity-EM marginals ($n > 3$) across recovered prompt clusters.
- **`sibling_style.py`:** Estimates within-family sibling stylometry classifiers (`claude`, `gemini`, `gpt`, `glm`) using 5-fold `GroupKFold(prompt_id)` and refines disputed sibling ranks while strictly preserving the probability mass assigned to each provider family.

## 5. Complementary Frozen-Feature Stack with Conditional Family Gates (`comp_stack.py`)

The complementary stack blends frozen hidden-state readouts (`Qwen3-8B-Base`, `Qwen2.5-7B`, `Qwen3-4B`, `GLM-4-9B`), `UltraBERT-380M` 5-fold checkpoints, and conditional within-family gates (`claude_style`, `claude_lora`, `gpt`, `gemini`, `glm`). It executes in an isolated subprocess during the competition rerun so all GPU memory is released before the main `LoRA24` + `ATLAS` stage.

## 6. Main Multi-Backbone `LoRA24` + `ATLAS 20%` + Damped Sinkhorn-Knopp + `V31`/`V40A` Pipeline (`infer.py`)

The main inference script (`infer.py`) integrates:
1. Fold-averaged two-stage `LoRA24` hidden-state extraction (`q14v5`, `q8v5`, `ll8v5`, `n14b`, `q25_14`, and `deb`) with dual-T4 data parallelism (`dp=True`) on the 8B trunks.
2. Prompt-clustered Adaptive Centering (`--adapt_w 1.0`, `--adapt_thr 0.15`, `twin=0.35`) and Shared-Prompt Gaussian group scoring (`gauss=0.70` on `q14v5`, `gauss=0.44` on `q8v5`).
3. 4-view n-gram `LinearSVC` (`--svc_w 1.5`), prompt-centered `TF-IDF` (`--tfidf_w 0.25`), and Stamatatos text distortion (`--distort_w 0.10`).
4. `15%` Complementary Stack blend + `20%` `ATLAS` source-conditioned likelihood on disputed rows (`top-2 margin < 0.30`).
5. **Damped Sinkhorn-Knopp Transductive Balancing (`--sinkhorn 0.45 --sinkhorn_target_mix 0.35`)** followed by **Hierarchical Beta-Bernoulli Typography (`V31`)** and **OOF Sibling-Style Residual (`V40A`)**.

## 7. Full Inference Execution & `submission.parquet` Export