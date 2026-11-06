# NTL Sovereign Gold Apex (198k Clean Pool + DP LoRA24 + 20% ATLAS + V40A Sibling)

> **Sovereign Gold-Tier Frontier Architecture** uniting all verified empirical breakthroughs:
> 1. **198k Clean Open Pool (crtproduction v11)**: Dynamically routed Gemini 3.7 risk and prompt regen leaks pruned; increases LinearSVC solo CV from 0.4805 to 0.5008 (+0.020).
> 2. **Re-scaled Stylometric Weight (`--svc_w 12`)**: Calibrates the 198k-trained 4-view LinearSVC decision values to have appropriate influence alongside deep neural logits.
> 3. **Dual-GPU DataParallel 8B Sharding (`dp: True`)**: Runs Qwen3-8B and Llama-3.1-8B simultaneously across both T4 GPUs, cutting inference latency in half and freeing ~40 GPU minutes.
> 4. **20% Autoregressive ATLAS Likelihood**: Token-level source-conditioned likelihood evaluation on candidate passages with close ensemble margin (< 0.30).
> 5. **V40A Intra-Family Sibling Style Residual (`sibling_style.py`)**: 5-fold GroupKFold(prompt_id) stylistic regression resolving the primary 55% error bottleneck across Claude, Gemini, GPT, and GLM while mathematically preserving 100% provider probability mass.
> 6. **Calibrated Typography Evidence (Scale 1.2, kappa=24.0)**: Softer Bayesian prior smoothing preventing over-penalization of neural representations.
> 7. **Damped Sinkhorn Marginal Regularization (`--sinkhorn 0.60`, `--sinkhorn_target_mix 0.35`)**: Regularizes test predictions toward official prior distribution without artificial one-hot distortion.
> 8. **Fast Commit Contract**: Sub-minute commit phase with full execution automatically triggered on hidden competition test rerun.


## 1. Inputs
Finds every base model, adapter and checkpoint under `/kaggle/input`. Llama-3.1-8B comes from avazaku's hash-checked cache (shard sizes asserted) and is copied to local disk first. `FAST_COMMIT` is true only on the 3 visible test rows.

## 1b. Clean Open Pool Setup (`crtproduction/name-that-llm-open-pool`)
Filters out dynamic routing risks (Gemini 3.7 fallback to 3.6) and prompt regeneration leaks from the 198k passage pool.

## 2. Shared-prompt Gaussian head (`gauss.py`)
A two-way model *x = author mean + prompt effect + noise*, integrated over the unknown prompt, with injective group marginals (prvsiyan / Kirill Tymchyshyn / sangam21, Apache 2.0).

## 2b. Sibling-Style Residual (`sibling_style.py`)
Roman Tamrazov's V40A group-calibrated residual for resolving intra-family sibling confusion (Claude, Gemini, GPT, GLM) while preserving total provider mass.

## 3. Main pipeline (`infer.py`)
Dhruv Tibarewal's pipeline with Roman Tamrazov's and mercurydust's changes: pooled hidden states of each LoRA24 member, plain and prompt-centred heads, the n-gram SVM (on official + external + pool rows), the TF-IDF and text-distortion members, test prompt recovery, the weighted log-probability blend, the complementary stack, ATLAS, V31's soft Sinkhorn and its typography step, which writes the submission. Our changes: in `l24_predict`, members marked `"dp"` (the 8B ones) load a copy on each GPU and alternate batches between them, and every member averages its hidden states on the GPU; ATLAS no longer stops on a projected overrun.

## 4. Complementary stack (`comp_stack.py`)
The public frozen-feature ensemble (prvsiyan, Lucas Boesen, vnton37nguyn, Kirill Tymchyshyn), mixed in at 15%, with V31's within-family gates. Optional token-attention heads are skipped when their backbone is not attached.

## 5. Run the complementary stack
In its own process, so all GPU memory is free again afterwards; its wall time is charged to ATLAS's budget.

## 6. Members and run
The 0.651 members and weights with V31's run flags (`--tfidf_w 0.25 --distort_w 0.10 --sinkhorn 0.45 --sinkhorn_target_mix 0.35 --post 0`). On the 3 visible rows the commit run writes a placeholder; Kaggle's rerun on the ~4,000 hidden rows runs every cell. The run command passes `'--atlas_w', '0.20'` (V31: 0.10).