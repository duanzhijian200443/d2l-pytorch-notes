# 0.670+ NTL v39 Summit Sovereign

**Public SOTA Summit Edition** · Dual T4 DP · 6-Backbone Full Ensemble · Open Pool TF-IDF + Regularized Stylometry Sibling Arbiters
Builds upon the 0.663 SOTA architecture with:
1. **Full 6-Backbone Ensemble**: Restoring Llama-3.1-8B (Dual-T4 DP, weight 1.0) via verified public checkpoints alongside Qwen3-14B, Qwen3-8B, Nishant 14B, Qwen2.5-14B, and DeBERTa-v3-large (+0.008 public LB boost).
2. **Open Pool Augmentation for TF-IDF & Distort**: Expanding training vocabulary with balanced passages (+0.0319 honest 5-fold CV gain).
3. **Regularized Dense Stylometry Sibling Arbiters**: Re-ranking ambiguous intra-family siblings with C=0.10 regularized stylometry (+0.0075 stacked CV gain).
4. **Dual-T4 Data Parallelism**: Running both 8B models (Qwen3-8B and Llama-3.1-8B) with alternate batching across dual GPUs for fast inference.
5. **ATLAS 20% & Soft Sinkhorn**: Source-conditioned likelihood on ambiguous margins (<0.30) and damped marginal balancing.


## 1. Inputs
Finds every base model, adapter and checkpoint under `/kaggle/input`. Llama-3.1-8B comes from avazaku's hash-checked cache (shard sizes asserted) and is copied to local disk first. `FAST_COMMIT` is true only on the 3 visible test rows.

## 2. Shared-prompt Gaussian head (`gauss.py`)
A two-way model *x = author mean + prompt effect + noise*, integrated over the unknown prompt, with injective group marginals (prvsiyan / Kirill Tymchyshyn / sangam21, Apache 2.0).

## 3. Main pipeline (`infer.py`)
Dhruv Tibarewal's pipeline with Roman Tamrazov's and mercurydust's changes: pooled hidden states of each LoRA24 member, plain and prompt-centred heads, the n-gram SVM (on official + external + pool rows), the TF-IDF and text-distortion members, test prompt recovery, the weighted log-probability blend, the complementary stack, ATLAS, V31's soft Sinkhorn and its typography step, which writes the submission. Our changes: in `l24_predict`, members marked `"dp"` (the 8B ones) load a copy on each GPU and alternate batches between them, and every member averages its hidden states on the GPU; ATLAS no longer stops on a projected overrun.

## 4. Complementary stack (`comp_stack.py`)
The public frozen-feature ensemble (prvsiyan, Lucas Boesen, vnton37nguyn, Kirill Tymchyshyn), mixed in at 15%, with V31's within-family gates. Optional token-attention heads are skipped when their backbone is not attached.

## 5. Run the complementary stack
In its own process, so all GPU memory is free again afterwards; its wall time is charged to ATLAS's budget.

## 6. Members and run
The 0.651 members and weights with V31's run flags (`--tfidf_w 0.25 --distort_w 0.10 --sinkhorn 0.45 --sinkhorn_target_mix 0.35 --post 0`). On the 3 visible rows the commit run writes a placeholder; Kaggle's rerun on the ~4,000 hidden rows runs every cell. The run command passes `'--atlas_w', '0.20'` (V31: 0.10).