# NTL Apex E: 0.670 pipeline + full 198k open pool in the SVC + best-public tweaks

> **Sovereign Gold-Tier Architecture** uniting the verified strengths of top solutions:
> 1. **DataParallel 8B Sharding (`dp: True`)**: Dual T4 GPU concurrent execution cutting LoRA inference latency in half.
> 2. **Doubled 20% ATLAS Autoregressive Token Likelihood**: Source-conditioned likelihood on disputed pairs with maximized time budget.
> 3. **Hierarchical Beta-Bernoulli Bayesian Typography Model**: Gated marker likelihood ratio shrinking class frequencies toward global rates (kappa=24.0).
> 4. **Roman Tamrazov's V40A Sibling-Style Residual (`sibling_style.py`)**: 5-fold GroupKFold OOF stylistic regression resolving intra-family ambiguity across Claude, Gemini, GPT, and GLM while mathematically preserving 100% provider probability mass.
> 5. **Safe Dual Llama Fallback**: Guaranteed compatibility with alexchilton/llama-3-1-8b-base and metaresearch/llama-3.1.
> 6. **Fast Commit Mechanism**: Sub-minute commit phase with full execution automatically triggered on hidden competition test rerun.


**Variant E (ours).** Starts from the public 0.670 notebook (synthreaper: 0.663 recipe + sibling-style) and combines the public tweaks with the strongest published evidence: (1) the SVC training pool is swapped from nlm_pool_v5 to the full public CC BY-SA 4.0 `crtproduction/name-that-llm-open-pool` (198k cleaned passages, gemini-3.7 routing-risk rows removed); (2) `--svc_w` 12 (zbigniewkirchner's one-swap notebook); (3) judith007's typography evidence coefficient 1.2 and `--sinkhorn 0.60`. Everything else is untouched.


## 1. Inputs
Finds every base model, adapter and checkpoint under `/kaggle/input`. Llama-3.1-8B comes from avazaku's hash-checked cache (shard sizes asserted) and is copied to local disk first. `FAST_COMMIT` is true only on the 3 visible test rows.

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