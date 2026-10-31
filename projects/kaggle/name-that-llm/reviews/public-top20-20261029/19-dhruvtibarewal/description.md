
# CELL 0
# Name That LLM — Fully Reproducible Two-Stage Ensemble

**Public LB 0.651** · T4 ×2 · internet off · about 7 h on the hidden test · every input is a public dataset or model

Eighteen frontier models, one 40–250-word paragraph each, and a test set written for prompts never seen in train. The topic of a passage says almost nothing about its author but dominates every representation computed from it. Most of the work below separates *who wrote it* from *what it is about*, and most of the score comes from **how much matched training data the models see before they ever meet the official rows**.

## What is in the ensemble

| Member | Trained on | Alone (5-fold OOF) |
|---|---|---:|
| Qwen3-14B-Base LoRA24, 27 layers | 120k public pool → official train | 0.634 |
| Qwen3-8B-Base LoRA24, 24 layers | same | 0.614 |
| Llama-3.1-8B LoRA24, 24 layers | same | 0.597 |
| Qwen3-14B LoRA (Nishant Dahal, part two) | 84k pool → official train | 0.611 |
| Qwen2.5-14B LoRA, 32 layers | 120k pool → official train | 0.615 |
| DeBERTa-v3-large (Nishant Dahal) | 61k pool → official train | 0.546 |
| 4-view n-gram SVM | official train + 120k pool | 0.47 |

All LoRA members use the same recipe: truncated decoder, LoRA r16 on every linear layer, masked mean-pool of eight depths with one LayerNorm each, a plain head and a prompt-centred head trained on same-prompt triples. **Stage 1 runs two epochs on the pool, stage 2 two epochs on the 2,699 official rows.** Our own members are fold-averaged: five fold adapters plus the full-data adapter, merged into one loaded base in turn.

## How the pieces are combined

1. **Test prompt recovery.** Capped average-linkage on content-word TF-IDF (never more than 18 passages per cluster).
2. **Centred twins.** On passages that share a recovered prompt, the centred head reads features with the cluster mean removed.
3. **Shared-prompt Gaussian.** A two-way model *x = author mean + prompt effect + noise*, integrated over the unknown prompt, with injective group marginals.
4. **Complementary stack** (public frozen-feature ensemble), mixed at 15%.
5. **ATLAS** source-conditioned likelihood, 10% mix, scored only where the ensemble is undecided (top-2 margin < 0.3).
6. **Sinkhorn** to equal per-model mass, then typography guards and one-model-per-prompt assignment in near-complete clusters.

## What moved the leaderboard

| Change | Public LB |
|---|---:|
| Fold-averaged two-stage LoRAs on a 38k pool + n-gram SVM | 0.592 |
| + shared-prompt Gaussian, guards, dense-group assignment, public checkpoints | 0.600 |
| + 10% ATLAS | 0.605 |
| + Llama-3.1-8B, complementary stack | 0.611 |
| + Nishant's part-two checkpoints | 0.619 |
| + our members retrained on the 120k pool, 2 stage-1 epochs | **0.651** |
| + Qwen2.5-14B retrained on the 120k pool | this notebook |

**The data lesson.** On a 38k pool four different backbones all finished near 0.57. The same recipe on 120k passages gave 0.634 for the 14B and 0.614 for the 8B. Two stage-1 epochs beat one by +1.1 on the 8B; three were worse. Everything else we tried was worth ±0.3 or less.

**Did not help**: a fully fine-tuned ModernBERT-large member (+0.6 OOF, −0.6 LB), family-expert heads, our own two-stage ATLAS, dropping any pool source, OLMo-2-13B, seed averaging, re-tuning the ATLAS weight, stage-1 on three epochs. Single-fold numbers swing by ±2 points; every decision here used all five folds.

# CELL 1
## Setup

# CELL 3
## Shared-prompt Gaussian head

# CELL 5
## Complementary stack

The public frozen-feature ensemble, run as a separate process so all GPU memory is released before the main pipeline. A member whose upstream checkpoint is no longer public is skipped.

# CELL 8
## Main pipeline

# CELL 10
## Members

`w` is the blend weight, `twin` the centred-head weight on recovered prompt groups, `gauss` the shared-prompt Gaussian weight. The ATLAS time budget is what remains after the stack.

# CELL 12
## Run

# CELL 15
## Reproducing the members

- **Pool.** `nlm_pool_v5.parquet` (119,899 passages) = the 122k clean public pool (`nuridudak/ntl-130k-clean-passages`) plus our own free-proxy generations, minus any text in train, minus passages carrying a typographic mark their model never uses in train, minus passages whose content TF-IDF cosine with a train passage is 0.3 or more.
- **LoRA members.** `train_l24.py` in our adapters dataset: stage 1 `--stage 1 --epochs 2` on the pool with class weights, stage 2 `--stage 2 --epochs 2 --lr 5e-5 --head_lr 5e-4` on the official rows, 5 prompt-grouped folds plus a full fit. bf16 on one A100; 10 h for the 14B stage 1.
- **ModernBERT-large.** `train_enc.py`: full fine-tune, mean of the last layer, two epochs on the pool at 2e-5 then three on train at 1e-5.

## Notes

- **Validate per passage.** The hidden test is mostly singletons; folds are `GroupKFold` by `prompt_id`, and the group tricks are applied only where the test itself shows a cluster.
- **Balancing is worth about a point on the LB** although neutral in CV: the hidden class mix is uneven and balanced accuracy rewards recall on the rare models.
- **End tokens.** Checkpoints trained with `<|endoftext|>` appended must get the same id at inference; Kaggle's Qwen3-14B-Base tokenizer reports `<|im_end|>`, which changes 14% of that member's predictions.
- **Run-to-run noise** of the hidden rerun is about ±0.5; identical notebooks scored 0.600 and 0.605 for us.

## Credits

[Nishant Dahal](https://www.kaggle.com/code/nishantdahal/two-stage-lora-part-two) (two-stage recipe, part-two checkpoints and pool) ·
[Nuri Dudak](https://www.kaggle.com/datasets/nuridudak/ntl-130k-clean-passages) (122k clean pool, encoder idea) ·
[prvsiyan](https://www.kaggle.com/code/prvsiyan/the-prompt-is-not-the-author) (shared-prompt Gaussian, complementary stack) ·
[Kirill Tymchyshyn](https://www.kaggle.com/code/kirilltym/0-600-teach-the-model-remove-the-prompt) (combined ensemble) ·
[Joseph Adamski](https://www.kaggle.com/code/josephadamski91/teach-the-model-10-atlas) (ATLAS adapter, CC BY-SA 4.0) ·
Lucas Boesen and vnton37nguyn (stack members) · every author of the public generation datasets.
Llama-3.1 adapters: *Built with Llama* (Llama 3.1 Community License).

If this notebook or the datasets were useful to you, an upvote helps others find them. Thanks for reading.