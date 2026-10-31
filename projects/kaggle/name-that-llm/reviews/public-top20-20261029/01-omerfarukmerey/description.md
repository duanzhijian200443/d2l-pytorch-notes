
# CELL 0
# 0.663 | 0.660 V31 + ATLAS 20%

**Public LB 0.663** · builds on mercurydust's 0.660, which builds on Roman Tamrazov's 0.654 and the 0.651 recipe · T4 ×2 · internet off · every input was public when it scored

This is mercurydust's [Name That LLM-Submission](https://www.kaggle.com/code/mercurydust/name-that-llm-submission) (public **0.660**), version 2, the code that scored. That notebook is Roman Tamrazov's [Name That LLM | Who am I?](https://www.kaggle.com/code/romantamrazov/name-that-llm-who-am-i) (0.654) with a set of changes its author calls V31, and Roman's is Dhruv Tibarewal's [nlm-v27](https://www.kaggle.com/code/dhruvtibarewal/nlm-v27) (0.651) as [avazaku reproduced it](https://www.kaggle.com/code/avazaku/name-that-llm-0-651-verified-reproduction), with Llama-3.1-8B from his hash-checked cache (no longer available, see *Running it*). On top of V31 this version adds **a faster run** (the two 8B LoRA members on both GPUs at once and Llama-3.1-8B read from local disk, which leave the predictions unchanged, and ATLAS scoring up to its deadline) and **ATLAS mixed at 20% instead of 10%**.

The code cells below are byte for byte the version that scored 0.663.

## Leaderboard A/B

| Version | Change | Public LB |
|---|---|---:|
| Roman Tamrazov, [Who am I?](https://www.kaggle.com/code/romantamrazov/name-that-llm-who-am-i), the version V31 starts from | the 0.651 recipe + his typography step | 0.654 |
| mercurydust, [Name That LLM-Submission](https://www.kaggle.com/code/mercurydust/name-that-llm-submission) version 2 (V31) | V31's changes (below) | 0.660 |
| ours: V31 + the faster run | the faster run only | 0.660 |
| **this notebook**: V31 + the faster run + ATLAS 20% | ATLAS-8B mixed at 20% instead of 10% | **0.663** |
| ours: V31 + the faster run + ATLAS 30% | ATLAS-8B mixed at 30% | 0.663 |
| ours: V31 + the faster run + typography margin gate only | no length or marker-count gate | 0.660 |
| ours: V31 + the faster run + ATLAS 20% + margin gate only | both | 0.661 |
| ours: the 0.651 recipe + ATLAS 20% | ATLAS-8B at 20% on nlm-v27 | 0.652 |

## What V31 changed (mercurydust)

Our reading of the code, against Roman's 0.654 notebook:

* **External rows used** — the matched-clean external passages are de-duplicated against the official rows (exact text after Unicode and whitespace normalisation) and used: the 4-view n-gram SVM now trains on official + external (weight 0.5) + pool (weight 0.35) rows, and the TF-IDF (weight 0.25) and text-distortion (0.10) members, both at 0 before, come back.
* **Family gates applied** — the complementary stack fitted within-family gate models (Claude style, Claude LoRA, GPT, Gemini, GLM) whose output never reached its logits. They now rerank the models inside each family where the family's own top two are close (margin below 0.45 within the family), keeping the family's total probability.
* **Softer balancing** — Sinkhorn at 0.45 towards a mix of 0.65 × the predicted and 0.35 × the training model proportions, instead of 1.0 towards equal proportions. On the 0.651 recipe, PeterDreamHan's [Soft Sinkhorn 0.5](https://www.kaggle.com/code/peterdreamhan/name-that-llm-soft-sinkhorn-0-5-t4) had shown that softer balancing pays (0.653).
* **Typography step reworked** — 37 markers (Roman's 34 + `###`, a code fence, `> `), each model's marker rates shrunk towards the overall rate, markers weighted by how much their rates differ between models, evidence clipped to ±0.30, and a gate that multiplies Roman's margin gate (here (0.6 − margin) / 0.6) by a passage-length gate and a marker-count gate.
* `--post 0`, and optional token-attention heads are skipped when their backbone is not attached (the Qwen3.5 one is not, so V31's `NTL_DROP_Q35 = '0'` changes nothing here).

## ATLAS at 20%

ATLAS ([Ko et al.](https://arxiv.org/abs/2610.19583), Joseph Adamski's Qwen3-8B bundle) is a source-conditioned likelihood: the passage is scored under 18 conditioned copies of one frozen language model, one per candidate author. Its author mixed it at 10% into seven public notebooks (+0.002 to +0.007 each) and never published another weight. In the 0.651 recipe it is worth **+0.008** (our ATLAS-off control scored 0.643, the recipe 0.651), on the passages whose top-2 margin is below 0.3.

At 10%, the mix can flip a passage only where the ensemble's top-2 margin is below ~0.11; at 20% the reach grows to ~0.25, still inside the same passages. This version mixes ATLAS at **20%** (`--atlas_w 0.20`). It costs no time: ATLAS scores the same passages either way.

On the leaderboard: the faster run alone scored 0.660, the same as V31; with ATLAS at 20% it scored **0.663**, and at 30% also 0.663. On the 0.651 recipe the same change gave +0.001 (0.651 → 0.652).

## A faster run

Two changes that leave the predictions unchanged reach ATLAS over an hour earlier, and ATLAS uses that time:

* **Llama-3.1-8B is copied to local disk before loading.** Read straight from the notebook-output mount, `from_pretrained` took ~2,050 s (CPU probe) and ~2,140 s (our GPU run); one sequential copy takes 88-135 s and the load from local disk ~60 s. Same bytes, same predictions.
* **The 8B members use both GPUs at once.** `l24_predict` loads every member with `device_map='auto'`, which splits even a 24-layer 8B model (~11 GiB in fp16) across the two T4s, so one GPU waits while the other runs its half of the layers, and it averages the hidden states on the CPU (eight ~100 MB copies per batch). Now every member averages on the GPU, and for the two 24-layer 8B members (Qwen3-8B, Llama-3.1-8B) a full copy sits on each GPU and the recipe's batches, in its order, alternate between them in two threads; if the largest batch would leave under 1 GiB free it falls back to the split copy. Same computation per row, about half the time. Checked on Kaggle's T4 ×2 before use: every member's log-probabilities match upstream's within 7.6e-6.
* **ATLAS runs to its deadline.** V31 stops ATLAS at its deadline and keeps the passages it finished, but also stops it at the first checkpoint where a projection of its finish passes the budget, which can leave much of the time unused. That stop is removed: ATLAS keeps scoring, shortest passages first, until the deadline itself.

## How the pipeline works

1. **Members** — Qwen3-14B-Base (weight 3), Qwen3-8B-Base (2) and Llama-3.1-8B (1) LoRA24 members, each averaged over 5 fold adapters + a full fit; Nishant Dahal's Qwen3-14B (2.5) and Qwen2.5-14B (1) part-two checkpoints; DeBERTa-v3-large (1); the 4-view n-gram SVM (1.5); TF-IDF (0.25) and text-distortion (0.10) members.
2. **Test prompt recovery** — capped average-linkage clustering on content-word TF-IDF; on recovered groups, prompt-centred "twin" heads and the shared-prompt Gaussian head add evidence.
3. **Blend** — weighted sum of log-probabilities, softmax, then the complementary stack (with its family gates) at 15%.
4. **ATLAS** — a source-conditioned likelihood (Joseph Adamski's public bundle) mixed at 20% on passages whose top-2 margin is below 0.3.
5. **Balancing** — V31's soft Sinkhorn (above).
6. **Typography step** — V31's, which writes the submission.

## Things that did not transfer (so you don't have to try them)

| Change on top of a public recipe | Offline | Public LB |
|---|---:|---:|
| Dhruv's ModernBERT-large member (w 1.5) on the 0.651 recipe | +0.008 | 0.645 |
| nlm-v27 without its Llama member | | 0.643 (Llama is worth +0.008) |
| SVM trained with quality-tiered pool weights (three bases) | +0.028 member | +0.001 / -0.001 / -0.003 |
| Our own Ministral-3-8B two-stage LoRA24 member (OOF 0.605 alone) at w 2 / w 3 on the 0.651 recipe | +0.004 / +0.006 | 0.641 / 0.635 |
| A second ATLAS (harshitakumari256's Qwen3-4B bundle) at 10% after ATLAS-8B | +0.007 | 0.650 |
| V31's typography step with its margin gate only (no length or marker-count gate) | +0.0019 | 0.660 alone; 0.661 with ATLAS 20% (0.663 without) |

## Running it

Attach the inputs below, choose **GPU T4 ×2**, Internet **off**, then *Save Version → Save & Run All* and submit that version. On the 3 visible rows the commit takes ~5 min: it resolves every input and writes a placeholder; Kaggle's rerun on the hidden rows takes several hours and runs everything.

**Llama-3.1-8B:** avazaku's Llama asset cache, which this version loaded Llama from, is no longer available (checked 2026-10-29). alexchilton's public model [Llama 3.1 8B base](https://www.kaggle.com/models/alexchilton/llama-3-1-8b-base) holds the same files: on Kaggle, the sha256 of all four weight shards and the hashes of the config, tokenizer and index files match the copy this version loaded (checked 2026-10-29). To rerun with it, attach it (Transformers, `default`, version 1), and in cell 1 replace `BL = mdir('/llama-3.1/', '/8b/')` with `BL = mdir('/llama-3-1-8b-base/')` and remove `'name-that-llm-verified-llama-asset-cache' in BL and` from the assert. Meta's own model (`metaresearch/llama-3.1`, Transformers, `8b`, version 2, after accepting Meta's licence, as Dhruv's nlm-v27 does) needs only the assert change. The shard-size check stays, so the weights must be the same files.

## Credits and licences

* **mercurydust** — the V31 changes and the [0.660 notebook](https://www.kaggle.com/code/mercurydust/name-that-llm-submission) this one forks.
* **Roman Tamrazov** — the [0.654 notebook](https://www.kaggle.com/code/romantamrazov/name-that-llm-who-am-i) V31 starts from, and the typography step it reworks.
* **PeterDreamHan** — [Soft Sinkhorn 0.5](https://www.kaggle.com/code/peterdreamhan/name-that-llm-soft-sinkhorn-0-5-t4), the first public evidence that softer balancing pays.
* **Dhruv Tibarewal** — the 0.651 recipe, its LoRA24 members, pool, matched-clean passages and adapters ([nlm-v27](https://www.kaggle.com/code/dhruvtibarewal/nlm-v27), [Fully Reproducible](https://www.kaggle.com/code/dhruvtibarewal/name-that-llm-fully-reproducible)).
* **avazaku** — the [verified reproduction](https://www.kaggle.com/code/avazaku/name-that-llm-0-651-verified-reproduction) and the [Llama asset cache](https://www.kaggle.com/code/avazaku/name-that-llm-verified-llama-asset-cache) this notebook loaded Llama from (no longer available).
* **Nishant Dahal** — the [two-stage LoRA recipe and part-two checkpoints](https://www.kaggle.com/code/nishantdahal/two-stage-lora-part-two).
* **Lucas Boesen** — the LoRA24 recipe ([trainer](https://www.kaggle.com/code/lucasboesen/name-that-llm-train-lora24-s7)), GLM-4-9B and encoder features and token-attention heads for the stack.
* **prvsiyan** — the [shared-prompt Gaussian and the complementary stack](https://www.kaggle.com/code/prvsiyan/the-prompt-is-not-the-author); **Kirill Tymchyshyn** — the [combined ensemble](https://www.kaggle.com/code/kirilltym/0-600-teach-the-model-remove-the-prompt); **fleongg** — [NTL Part Two Complement](https://www.kaggle.com/code/fleongg/ntl-part-two-complement); **Joseph Adamski** — [ATLAS](https://www.kaggle.com/code/josephadamski91/teach-the-model-10-atlas) (CC BY-SA 4.0; method: [Ko et al.](https://arxiv.org/abs/2610.19583)); **vnton37nguyn**, **Zbigniew Kirchner**, **pjmathematician** — stack members; the text-distortion member follows Stamatatos (EACL 2017); **Nuri Dudak** and every author of the public generation datasets below — the pool and external passages.
* **Built with Llama.** Llama 3.1 is licensed under the [Llama 3.1 Community License](https://www.llama.com/llama3_1/license/) (Copyright © Meta Platforms, Inc.) and its [Acceptable Use Policy](https://www.llama.com/llama3_1/use-policy/).
* **GLM-4-9B-0414** (in the complementary stack, Lucas Boesen's truncated copy) — Zhipu AI / THUDM, MIT. Qwen models — Alibaba Qwen team, under their own licences. Each input keeps its own licence; the notebook's code is Apache 2.0 like its parents.

### Inputs

| Type | Source |
|---|---|
| Dataset | [dhruvtibarewal/nlm-l24-adapters](https://www.kaggle.com/datasets/dhruvtibarewal/nlm-l24-adapters) |
| Dataset | [seyitkaangunes/ntl-extra-api-passages](https://www.kaggle.com/datasets/seyitkaangunes/ntl-extra-api-passages) |
| Dataset | [abdullah1afify/ntl-generations](https://www.kaggle.com/datasets/abdullah1afify/ntl-generations) |
| Dataset | [dophuchung/ntl-proxy-generations](https://www.kaggle.com/datasets/dophuchung/ntl-proxy-generations) |
| Dataset | [zbigniewkirchner/ntl-qwen3-8b-lora-claude-head](https://www.kaggle.com/datasets/zbigniewkirchner/ntl-qwen3-8b-lora-claude-head) |
| Dataset | [dhruvtibarewal/nlm-matched-clean](https://www.kaggle.com/datasets/dhruvtibarewal/nlm-matched-clean) |
| Dataset | [dhruvtibarewal/nlm-offline-wheels](https://www.kaggle.com/datasets/dhruvtibarewal/nlm-offline-wheels) |
| Dataset | [josephadamski91/ntl-atlas-qwen3-8b-v1](https://www.kaggle.com/datasets/josephadamski91/ntl-atlas-qwen3-8b-v1) |
| Dataset | [lucasboesen/ntl-free-proxy-generations](https://www.kaggle.com/datasets/lucasboesen/ntl-free-proxy-generations) |
| Dataset | [nishantdahal/ntl-two-stage-lora](https://www.kaggle.com/datasets/nishantdahal/ntl-two-stage-lora) |
| Dataset | [nishantdahal/ntl-handwriting-blend](https://www.kaggle.com/datasets/nishantdahal/ntl-handwriting-blend) |
| Dataset | [dhruvtibarewal/nlm-l24-adapters-v5](https://www.kaggle.com/datasets/dhruvtibarewal/nlm-l24-adapters-v5) |
| Dataset | [prvsiyan/prvsiyans-extended-name-that-llm-dataset](https://www.kaggle.com/datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset) |
| Dataset | [pjmathematician/ultrabert-380m-ntl](https://www.kaggle.com/datasets/pjmathematician/ultrabert-380m-ntl) |
| Notebook output | [lucasboesen/name-that-llm-glm4-9b-trunc25](https://www.kaggle.com/code/lucasboesen/name-that-llm-glm4-9b-trunc25) |
| Notebook output | [lucasboesen/name-that-llm-extract-encoders2](https://www.kaggle.com/code/lucasboesen/name-that-llm-extract-encoders2) |
| Notebook output | [lucasboesen/name-that-llm-electra-large-weights](https://www.kaggle.com/code/lucasboesen/name-that-llm-electra-large-weights) |
| Notebook output | [lucasboesen/name-that-llm-desklib-weights](https://www.kaggle.com/code/lucasboesen/name-that-llm-desklib-weights) |
| Notebook output | [lucasboesen/name-that-llm-train-lora24-s7](https://www.kaggle.com/code/lucasboesen/name-that-llm-train-lora24-s7) |
| Notebook output | [lucasboesen/name-that-llm-train-tokattn](https://www.kaggle.com/code/lucasboesen/name-that-llm-train-tokattn) |
| Notebook output | [lucasboesen/name-that-llm-train-tokattn2](https://www.kaggle.com/code/lucasboesen/name-that-llm-train-tokattn2) |
| Notebook output | [vnton37nguyn/ntl-train-lora-qwen25-instyle](https://www.kaggle.com/code/vnton37nguyn/ntl-train-lora-qwen25-instyle) |
| Notebook output | [avazaku/name-that-llm-verified-llama-asset-cache](https://www.kaggle.com/code/avazaku/name-that-llm-verified-llama-asset-cache) |
| Model | [qwen-lm/qwen-3/Transformers/8b/1](https://www.kaggle.com/models/qwen-lm/qwen-3/transformers/8b/1) |
| Model | [qwen-lm/qwen-3/Transformers/14b-base/1](https://www.kaggle.com/models/qwen-lm/qwen-3/transformers/14b-base/1) |
| Model | [qwen-lm/qwen-3/Transformers/14b/1](https://www.kaggle.com/models/qwen-lm/qwen-3/transformers/14b/1) |
| Model | [qwen-lm/qwen-3/Transformers/4b-base/1](https://www.kaggle.com/models/qwen-lm/qwen-3/transformers/4b-base/1) |
| Model | [qwen-lm/qwen-3/Transformers/8b-base/1](https://www.kaggle.com/models/qwen-lm/qwen-3/transformers/8b-base/1) |
| Model | [qwen-lm/qwen2.5/Transformers/14b/1](https://www.kaggle.com/models/qwen-lm/qwen2.5/transformers/14b/1) |
| Model | [qwen-lm/qwen2.5/Transformers/7b/1](https://www.kaggle.com/models/qwen-lm/qwen2.5/transformers/7b/1) |
| Competition | [name-that-llm](https://www.kaggle.com/competitions/name-that-llm) |


# CELL 1
## 1. Inputs
Finds every base model, adapter and checkpoint under `/kaggle/input`. Llama-3.1-8B comes from avazaku's hash-checked cache (shard sizes asserted) and is copied to local disk first. `FAST_COMMIT` is true only on the 3 visible test rows.

# CELL 3
## 2. Shared-prompt Gaussian head (`gauss.py`)
A two-way model *x = author mean + prompt effect + noise*, integrated over the unknown prompt, with injective group marginals (prvsiyan / Kirill Tymchyshyn / sangam21, Apache 2.0).

# CELL 5
## 3. Main pipeline (`infer.py`)
Dhruv Tibarewal's pipeline with Roman Tamrazov's and mercurydust's changes: pooled hidden states of each LoRA24 member, plain and prompt-centred heads, the n-gram SVM (on official + external + pool rows), the TF-IDF and text-distortion members, test prompt recovery, the weighted log-probability blend, the complementary stack, ATLAS, V31's soft Sinkhorn and its typography step, which writes the submission. Our changes: in `l24_predict`, members marked `"dp"` (the 8B ones) load a copy on each GPU and alternate batches between them, and every member averages its hidden states on the GPU; ATLAS no longer stops on a projected overrun.

# CELL 7
## 4. Complementary stack (`comp_stack.py`)
The public frozen-feature ensemble (prvsiyan, Lucas Boesen, vnton37nguyn, Kirill Tymchyshyn), mixed in at 15%, with V31's within-family gates. Optional token-attention heads are skipped when their backbone is not attached.

# CELL 9
## 5. Run the complementary stack
In its own process, so all GPU memory is free again afterwards; its wall time is charged to ATLAS's budget.

# CELL 11
## 6. Members and run
The 0.651 members and weights with V31's run flags (`--tfidf_w 0.25 --distort_w 0.10 --sinkhorn 0.45 --sinkhorn_target_mix 0.35 --post 0`). On the 3 visible rows the commit run writes a placeholder; Kaggle's rerun on the ~4,000 hidden rows runs every cell. The run command passes `'--atlas_w', '0.20'` (V31: 0.10).