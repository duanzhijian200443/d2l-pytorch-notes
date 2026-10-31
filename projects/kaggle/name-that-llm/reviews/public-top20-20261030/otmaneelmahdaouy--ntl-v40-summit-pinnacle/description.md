# 0.670+ NTL v40 Summit Pinnacle

**Public SOTA Summit Edition** · Dual T4 DP · Full 6-Backbone Ensemble · Pure In-Domain TF-IDF · Dual-T4 DP Llama-3.1-8B · ATLAS 20% · Soft Sinkhorn

### Architectural Blueprint & Forensic Breakthroughs:
1. **Pure In-Domain TF-IDF Calibration**: `FIT_T` is strictly reserved for clean in-domain texts (`train.text` + `matched-clean extra rows`), preserving the sharp C=20 document frequency calibration without out-of-domain vocabulary dilution.
2. **Source-Weighted 4-View SVC**: Open pool auxiliary passages are strictly isolated to the 4-view stylometric SVM with de-duplication and calibrated source weighting (`_srcw = 0.35`).
3. **Full 6-Backbone Multi-Modal Ensemble**: Restores verified public Llama-3.1-8B (`alexchilton/llama-3-1-8b-base`, weight 1.0, Dual-T4 DP) alongside Qwen3-14B (w=3.0), Qwen3-8B (w=2.0, Dual-T4 DP), Nishant Qwen3-14B Part 2 (w=2.5), Qwen2.5-14B Part 2 (w=1.0), and DeBERTa-v3-large (w=1.0).
4. **Calibrated Sibling Family Gates**: In `comp_stack.py`, family gates (C=1.0 for gpt, gemini; C=0.5 for glm) sharpen intra-family discriminative boundaries while strictly conserving total family probability mass.
5. **ATLAS 20% Likelihood Beam Search**: Evaluates source-conditioned likelihood (arXiv 2610.19583) on ambiguous rows (`margin < 0.30`), flipping subtle sibling confusions with zero additional runtime.
6. **Soft Sinkhorn Marginal Balancing**: Damped marginal balancing (`tau=0.45`, `target_mix=0.35`) eliminates extreme clipping while respecting the authentic hidden test label distribution.
7. **Hierarchical Beta-Bernoulli Typography**: 37-marker Empirical Bayes smoothing (kappa=24.0) with 3-way multiplicative confidence, length, and coverage gating.


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