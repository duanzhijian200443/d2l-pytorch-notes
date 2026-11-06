# Name That LLM — the 0.664 pipeline, restructured and hardened (derivative)

> ## Source and attribution
> A restructured copy of the public notebook **"0.664 · hyperparameter tuned · who are u"** (Public LB 0.664). Its pipeline is
> **Dhruv Tibarewal's** `nlm-v27` ([v5 adapters](https://www.kaggle.com/datasets/dhruvtibarewal/nlm-l24-adapters-v5) ·
> [pool](https://www.kaggle.com/datasets/dhruvtibarewal/nlm-matched-clean) · [v1 adapters](https://www.kaggle.com/datasets/dhruvtibarewal/nlm-l24-adapters)), with
> changes by **Roman Tamrazov**, **mercurydust** and the **V31** notebook, and the 0.664 author's data-parallel and ATLAS fixes. Full
> credits are at the end. Please upvote the originals.
>
> **What this derivative changes:** structure, input handling, failure handling, one latent bug fix, and experiment switches.
> **With the default settings the models, weights and blend are byte-for-byte those of the 0.664 notebook.**

### Expectation, honestly
By the 0.664 author's own resubmissions, *identical code* moves ±0.01 on the public LB (`nlm-v27` scored 0.651 / 0.651 / 0.643 /
0.651). The public LB holds ~1,600 rows, which puts the standard error on balanced accuracy at ≈ 0.012. A rerun of this notebook
should land around 0.66, not exactly 0.664. No switch below is validated out-of-fold.

### What changed
| | 0.664 original | This version | Prediction change at defaults |
|---|---|---|---|
| Layout | Scripts as bare cells, flags inline in one long command | Documented sections, one config cell | none |
| Required inputs | `assert`s on 10 inputs it never uses (v1 adapter folds, Nishant v1, Qwen3 instruct, heads bank) | Only inputs actually used are required | none |
| Llama-3.1-8B | Public re-upload only, size-checked | Public re-upload **or** the official gated model, same size check; clear message if missing | none |
| Commit run | Always a placeholder, so the real code first runs in the hidden rerun | Same by default; `FULL_COMMIT_TEST = True` runs everything on the 3 visible rows (in-sample LoRA checks included), so you can test it before submitting | none |
| Failure | Any crash or timeout → failed submission | Comp-stack submission written first; main stage under a watchdog that keeps it if time runs out | only on failure |
| `--post` | Silently discarded by the typography step (latent bug) | Respected | none (`post` is 0) |
| Outputs | — | Per-member log-probs saved (`members_logprob.npz`); DeBERTa and cache copies moved out of `/kaggle/working` | none |

```
 A. comp_stack.py (subprocess, both T4s) ── public frozen-feature ensemble + V31 family gates ──────► comp_logit.npy ─► fallback submission
 B. infer.py: n-gram SVC (train + matched + pool) · TF-IDF · text distortion · v5 LoRA folds (Q3-14B ×3.0, Q3-8B ×2.0, Llama-8B ×1.0)
             · Nishant Q3-14B ×2.5, Q2.5-14B ×1.0, DeBERTa ×1.0 · centred twins · Gaussian group heads
     Σ w·log p / #members → softmax → .85·main + .15·comp → ATLAS 20% on rows with margin < .3 (partial if time runs short)
     → soft Sinkhorn (0.6, target 35% toward train prior) → V31 Bayesian typography → submission
```

## 0 · Configuration
Defaults reproduce the 0.664 notebook exactly. **Change one thing per submission.**

## 1 · Inputs
Only inputs that are actually used are required. Llama-3.1-8B comes from a public re-upload (`alexchilton/llama-3-1-8b-base`, as in
0.664) or from Meta's gated Kaggle model if you have accepted its licence. Either way the four shard sizes must match Meta's release.
It is copied to local disk first (≈ 3 min).

## 2 · Scripts
`gauss.py` is the shared-prompt Gaussian group head (prvsiyan, Kirill Tymchyshyn, sangam21; Apache-2.0). `comp_stack.py` is the 0.664
file, unchanged. In `infer.py` the edited lines are marked `# [restructure]`. They add the `--v31_typo` switch (default on), fix the
`--post` bug, and save per-member log-probs.

## 3 · Stage A — complementary stack, then a fallback submission
The stack runs in its own process, so its GPU memory is freed before the 14B models load. Its blend is written as `submission.parquet`
straight away, so a crash or timeout in stage B still leaves a valid (lower-scoring, ≈0.6) submission.

## 4 · Stage B — main blend (`infer.py`)
Same command and flags as the 0.664 notebook, now built from the config cell. The watchdog stops the stage `SAFETY_S` seconds
before Kaggle's limit and keeps the stage-A submission. ATLAS already stops itself on its own budget and keeps the rows it
finished.

## Trying for more than 0.664
Every switch in the config cell defaults to the 0.664 setting. Nothing here has out-of-fold evidence, so each idea costs one
full submission and needs a gain of ≳ 0.01 to mean anything. Ranked by how much I'd expect from each:

1. **Spend your two final picks as a hedge.** Kaggle scores two chosen submissions on the private 60%. Pick this notebook at defaults,
   plus one with the *least-tuned* blend: `SINKHORN = 1.0, SINKHORN_TARGET_MIX = 0.0, V31_TYPO = 0, TFIDF_W = DISTORT_W = 0, ATLAS_W = 0.10`
   (≈ `nlm-v27`, 0.651 public). Public-LB-tuned additions are the ones most likely to reorder on the private board.
2. **`V31_TYPO = 0`.** The typography step has no out-of-fold check (its own log says so), and its 37 markers overlap with what the
   n-gram SVC already sees. If removing it costs < 0.005 public, the simpler blend is the safer pick.
3. **`ATLAS_MARGIN = 0.4`.** ATLAS scores more uncertain rows. It is time-bounded and keeps partial results, so the risk is runtime only.

Not worth a probe: member weights (fit by the upstream authors on OOF) and `TWIN`/`GAUSS` (only active on detected same-prompt groups).

## Credits
* **Pipeline (`nlm-v27`) and v5 adapters/pool:** Dhruv Tibarewal ([v5 adapters](https://www.kaggle.com/datasets/dhruvtibarewal/nlm-l24-adapters-v5), [matched pool](https://www.kaggle.com/datasets/dhruvtibarewal/nlm-matched-clean))
* **Changes:** Roman Tamrazov, mercurydust, the V31 notebook (soft Sinkhorn, Bayesian typography, family gates), and the 0.664 author (data-parallel 8B members, partial ATLAS)
* **Two-stage LoRA members (Qwen3-14B, Qwen2.5-14B, DeBERTa):** Nishant Dahal ([ntl-handwriting-blend](https://www.kaggle.com/datasets/nishantdahal/ntl-handwriting-blend)). The 0.664 notebook did not credit these.
* **ATLAS:** Ko, Lee, Song, Kim, Shin, Kim, *Who's Behind the Text? Robust Model Authorship Attribution*, arXiv:2610.19583 ([code](https://github.com/komseok0109/ATLAS)); Kaggle bundle by [josephadamski91](https://www.kaggle.com/code/josephadamski91/teach-the-model-10-atlas)
* **Complementary stack:** prvsiyan, Lucas Boesen, vnton37nguyn, Kirill Tymchyshyn ([0.639](https://www.kaggle.com/code/kirilltym/0-639-teach-more-keep-the-signature)), zbigniewkirchner ([Claude-head LoRA](https://www.kaggle.com/datasets/zbigniewkirchner/ntl-qwen3-8b-lora-claude-head)), pjmathematician (UltraBERT); Gaussian head: prvsiyan / Kirill Tymchyshyn / sangam21
* **Recipes:** LoRA24 by [lightningv08](https://www.kaggle.com/code/lightningv08/ntl-lora24-fivefold-training) and [Lucas Boesen](https://www.kaggle.com/code/lucasboesen/name-that-llm-train-lora24-s7); style n-grams by yangkuangou; centring and Sinkhorn by [Lucas Boesen](https://www.kaggle.com/code/lucasboesen/lora-x-adaptive-centering)
* **Models:** Qwen3 / Qwen2.5 (Apache-2.0); DeBERTa-v3 (MIT); **Llama 3.1 — Built with Llama** (Llama 3.1 Community License; weights via Meta or the public re-upload by alexchilton)