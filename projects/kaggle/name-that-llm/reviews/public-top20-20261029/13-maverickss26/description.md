
# CELL 0
# Name That LLM — nlm-v27, restructured (derivative)

> ## Source and attribution
> **This is a restructured copy of the notebook `nlm-v27`, Public LB 0.651.** Every modelling decision,
> checkpoint and weight is the original author's or the upstream authors' listed below. The original notebook has no
> author credits or markdown. The three datasets it is built around belong to **dhruvtibarewal**:
> [nlm-l24-adapters-v5](https://www.kaggle.com/datasets/dhruvtibarewal/nlm-l24-adapters-v5) (the Qwen3-14B / Qwen3-8B / Llama-3.1-8B "v5" LoRA folds),
> [nlm-matched-clean](https://www.kaggle.com/datasets/dhruvtibarewal/nlm-matched-clean) (`nlm_pool_v5.parquet`), and
> [nlm-l24-adapters](https://www.kaggle.com/datasets/dhruvtibarewal/nlm-l24-adapters). Please upvote the original notebook and those datasets.
>
> **What this derivative changes:** layout, removal of computation whose output was never used, removal of required inputs
> that are never read, a fallback submission, and diagnostics. **No model, adapter, weight or blend setting is changed.**
> Details are in *What changed* below.

### Where the score comes from
| Block | Members (log-prob weight) | Source |
|---|---|---|
| v5 LoRA folds (5 folds + full, fold-averaged) | Qwen3-14B **3.0**, Qwen3-8B **2.0**, Llama-3.1-8B **1.0**, + centred twins and shared-prompt Gaussian heads | dhruvtibarewal |
| Two-stage LoRA part two | Qwen3-14B 2.5, Qwen2.5-14B 1.0, DeBERTa-v3-large 1.0 | Nishant Dahal |
| 4-view n-gram SVC on train + `nlm_pool_v5` | 1.5 | recipe: yangkuangou / Nishant Dahal; pool: dhruvtibarewal |
| Complementary stack (separate process) | linear mix 15% | prvsiyan, lucasboesen, vnton37nguyn, kirilltym, zbigniewkirchner, pjmathematician |
| ATLAS on low-margin rows | linear mix 10% | method: Ko et al. 2026; bundle: josephadamski91 |
| Sinkhorn → typography guards → Hungarian in dense clusters | post-processing | lucasboesen (Sinkhorn); original author (guards, Hungarian) |

**Where the gain over 0.639 comes from.** The other blocks in this table are close to kirilltym's 0.639 stack. The step to 0.651
comes from adding the **v5 adapter folds and `nlm_pool_v5`** on top. Their training recipe is not documented in the notebook.
Cell **0d** prints basic data-hygiene stats for the pool.

```
 train / test ─┬─► A. complementary stack (subprocess) ──────────────────────────► comp_logit.npy ─► fallback submission
               └─► B. infer.py: n-gram SVC (train + pool v5) · v5 LoRA folds (Q3-14B, Q3-8B, Llama-8B)
                                 · Nishant Q3-14B / Q2.5-14B / DeBERTa · twins · Gaussian heads
                                   Σ w·log p / #members → softmax → .85·main + .15·comp → ATLAS 10% (margin < .3)
                                   → Sinkhorn → typography guards → Hungarian (clusters 12–18) → submission
```

# CELL 1
### What changed from the original
| | Original | This version | Effect on predictions |
|---|---|---|---|
| Layout | No markdown; 3 scripts as bare `%%writefile` cells; weights inline in a shell command | Documented sections; one config cell | none |
| Unused required inputs | `assert`s on the v1 `nlm-l24-adapters` folds, Nishant's v1 `lora8b/14b`, Qwen3-8B/14B *instruct*, and the `heads/` bank, none of which is used | Not required (still listed in the metadata) | none |
| Weight-0 TF-IDF member | Fitted on train + 24k matched rows (~6 min), multiplied by 0 | Skipped | none (it added exact zeros) |
| Comp-stack gates | Claude/gpt/gemini/glm gate SVCs fitted, never read (~2 min) | Removed | none |
| DeBERTa copy | Into `/kaggle/working` (bloats the output) | Into `/kaggle/temp` | none |
| Failure | If `infer.py` crashes there is no `submission.parquet` → failed submission | Comp-stack submission written first as a fallback; exit code checked and reported | only on failure |
| Diagnostics | — | Pool/test overlap check on the public file; per-member log-probs saved (`members_logprob.npz`) | none |

The edited lines inside the scripts are marked `# [restructure]`. With the default config, predictions are identical to the
original. I tested the blending and post-processing path against the original `infer.py` on synthetic data.

# CELL 2
## 0 · Configuration
Every weight in one place. Defaults = original `nlm-v27`.

# CELL 4
## 0b · Offline installs and input discovery
Only inputs that are actually used are required. The rest are reported.

# CELL 6
## 0d · Pool data-hygiene check (optional, print only)
This runs only on the 3-row public `test.csv` and only prints. It reports, for `nlm_pool_v5` (the SVC's extra training data):
* exact matches with the visible test passages, and the nearest-neighbour cosine on character 5-grams (≥ 0.9 ≈ near-duplicate);
* how many **train** passages appear in the pool. The upstream pools filter these out, so a large number means the filter is missing.

A good pool should show no exact test matches and few or no train passages. The SVC trains on train + pool together, so duplicated train rows only reweight them, but they inflate any CV run on this pool.

# CELL 8
## 1 · Scripts
`gauss.py` is the shared-prompt Gaussian group head. Its docstring credits prvsiyan, kirilltym and sangam21 (Apache-2.0).
`infer.py` and `comp_stack.py` are the original scripts, and the only edited lines are marked `# [restructure]`.

# CELL 12
## 2 · Stage A — complementary stack, then a fallback submission
The stack runs in its own process so that all GPU memory is freed before the 14B models load. Its time is charged against the
ATLAS budget. Its blend is then written as `submission.parquet` (Sinkhorn-balanced), so a crash in stage B still leaves a valid
submission from the comp stack alone. Expect a much lower score from that fallback (≈0.6), not the full blend's.

# CELL 14
## 3 · Stage B — main blend (`infer.py`)
Members in order: n-gram SVC → v5 Qwen3-14B folds → v5 Qwen3-8B folds → v5 Llama-3.1-8B folds → Nishant Qwen3-14B → Nishant
Qwen2.5-14B → DeBERTa. After that come the comp-stack mix, ATLAS, Sinkhorn, typography guards, and Hungarian assignment inside dense
clusters. On the 3-row public file, every LoRA is first checked in-sample on 40 training rows (it must score > 0.5).

# CELL 16
## Citation and source audit
I checked each reference in the original code (26 Oct 2026):

| Reference in the code | Status |
|---|---|
| "ATLAS (source-conditioned likelihood, arXiv 2610.19583)" | ✅ Real: Ko, Lee, Song, Kim, Shin, Kim, *Who's Behind the Text? Robust Model Authorship Attribution*, arXiv:2610.19583 (14 Oct 2026). Code: [github.com/komseok0109/ATLAS](https://github.com/komseok0109/ATLAS). The trained bundle used here is josephadamski91's [ntl-atlas-qwen3-8b-v1](https://www.kaggle.com/datasets/josephadamski91/ntl-atlas-qwen3-8b-v1), from the notebook [Teach the Model + 10% ATLAS](https://www.kaggle.com/code/josephadamski91/teach-the-model-10-atlas). The original credits neither. |
| `gauss.py`: "ported, Apache-2.0, from public Kaggle notebooks by prvsiyan / kirilltym / sangam21" | ⚠️ Plausible (Kaggle notebooks default to Apache-2.0) but no links are given. The same code inside `comp_stack.py` says "verbatim core of **our** research tool tools/group_gaussian.py (sha256 8a42427b…)", which is an upstream author's first-person comment copied over. It is not this notebook's tool. |
| `comp_stack.py`: "public Apache-2.0 notebooks by prvsiyan / lucasboesen / vnton37nguyn / kirilltym, as embedded in the public 0.6xx ATLAS notebooks" | ✅ Consistent with the upstream notebooks. It also uses checkpoints by zbigniewkirchner (Claude-head LoRA), lucasboesen (LoRA24 seed 7, token attention), vnton37nguyn (Qwen2.5 in-style LoRA), pjmathematician (UltraBERT) and the desklib detector, none of which are named. The `lora24_public` member is **skipped** in this run (checkpoint not attached), so the comp stack runs with one LoRA fewer than its weights assume. |
| comp stack: "weights of the 0.563 notebook (its L-BFGS fit on prompt-grouped OOF)" | ⚠️ Upstream claim, not verifiable here. Those weights were fit for a different blend. |
| comp stack: "research/gaussian/production_weights_v8g.json, 135 training prompts" | ⚠️ Points to an upstream author's private file. Not verifiable. |
| `infer.py`: "Stamatatos-style text distortion" | ✅ Real method (E. Stamatatos, *Authorship Attribution Using Text Distortion*, EACL 2017). Unused here (`--distort_w 0`). |
| `infer.py` typography guards: "Zero violations on the 38k matched pool for every rule" | ⚠️ Not verifiable without the pool. The guards hard-forbid classes when a punctuation mark appears (e.g. no Claude/GLM if a curly apostrophe is present). |
| Nishant Dahal's checkpoints | ✅ [ntl-handwriting-blend](https://www.kaggle.com/datasets/nishantdahal/ntl-handwriting-blend) (part two, LB 0.620) is used. [ntl-two-stage-lora](https://www.kaggle.com/datasets/nishantdahal/ntl-two-stage-lora) is attached but unused. |
| Base models | Qwen3-8B/14B-Base and Qwen2.5-7B/14B: Apache-2.0. **Llama-3.1-8B: Llama 3.1 Community License**, which requires the notice "Built with Llama" when you redistribute derived work. The original omits it. DeBERTa-v3-large: MIT. |
| Training data of the v5 adapters / `nlm_pool_v5` | ⚠️ **Undocumented** in the notebook. These provide the step from ~0.64 to 0.651. Cell 0d prints basic overlap stats. |

## Credits
* **Original notebook `nlm-v27`, the v5 adapters and pool:** dhruvtibarewal: [nlm-l24-adapters-v5](https://www.kaggle.com/datasets/dhruvtibarewal/nlm-l24-adapters-v5), [nlm-matched-clean](https://www.kaggle.com/datasets/dhruvtibarewal/nlm-matched-clean), [nlm-l24-adapters](https://www.kaggle.com/datasets/dhruvtibarewal/nlm-l24-adapters)
* **Two-stage LoRA, part two:** Nishant Dahal: [notebook](https://www.kaggle.com/code/nishantdahal/two-stage-lora-part-two), [dataset](https://www.kaggle.com/datasets/nishantdahal/ntl-handwriting-blend)
* **ATLAS:** method by Ko et al. (arXiv:2610.19583); Kaggle bundle by [josephadamski91](https://www.kaggle.com/code/josephadamski91/teach-the-model-10-atlas)
* **Complementary stack:** prvsiyan, lucasboesen, vnton37nguyn, kirilltym ([0.639 notebook](https://www.kaggle.com/code/kirilltym/0-639-teach-more-keep-the-signature)), zbigniewkirchner ([Claude-head LoRA](https://www.kaggle.com/datasets/zbigniewkirchner/ntl-qwen3-8b-lora-claude-head)), pjmathematician (UltraBERT-380M), sangam21 (Gaussian head)
* **LoRA24 recipe:** [lightningv08](https://www.kaggle.com/code/lightningv08/ntl-lora24-fivefold-training), [lucasboesen](https://www.kaggle.com/code/lucasboesen/name-that-llm-train-lora24-s7). **Style n-gram views:** yangkuangou. **Centring, twin, Sinkhorn:** [lucasboesen](https://www.kaggle.com/code/lucasboesen/lora-x-adaptive-centering)
* **Models:** Qwen team (Apache-2.0), Meta Llama 3.1 (Llama 3.1 Community License: *Built with Llama*), Microsoft DeBERTa-v3 (MIT)