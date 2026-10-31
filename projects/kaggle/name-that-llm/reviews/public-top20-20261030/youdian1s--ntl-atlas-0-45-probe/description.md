# Name That LLM 0.662: All Public + What Failed

> **All public.** Every model, adapter, checkpoint and dataset this notebook reads is public on Kaggle, so anyone can run it and get the same pipeline.

This version runs omerfarukmerey's public *[0.663 | 0.660 V31 + ATLAS 20%](https://www.kaggle.com/code/omerfarukmerey/0-663-0-660-v31-atlas-20)*, and on our account it scored **0.662** on the public leaderboard (2026-10-29).
It is mercurydust's *[Name That LLM-Submission](https://www.kaggle.com/code/mercurydust/name-that-llm-submission)* (*Who am I? V31*, 0.660, version 7 of this notebook) with ATLAS mixed at 20% instead of 10% and a faster run (the two 8B LoRA members on both GPUs at once).
V31 builds on Roman Tamrazov's *[Name That LLM | Who am I?](https://www.kaggle.com/code/romantamrazov/name-that-llm-who-am-i)* (0.654), which builds on dhruvtibarewal's nlm-v27 (0.651).
Earlier versions of this notebook ran the 0.660 pipeline (0.660), the 0.654 pipeline (0.654 and 0.645 on two runs of identical code) and the 0.651 pipeline (0.651, 0.643, 0.651).

**Llama-3.1-8B is our own public upload.** The 0.663 notebook loaded the Meta base weights from avazaku's hash-checked cache kernel, which is no longer readable (2026-10-29), so the original cannot run as published.
This version loads the same unmodified Meta weights from our own public upload, [alexchilton/llama-3-1-8b-base](https://www.kaggle.com/models/alexchilton/llama-3-1-8b-base) (one line in the setup cell, marked `# ours`); every other input is the 0.663 notebook's own.

**Credits.** omerfarukmerey (ATLAS 20%, the faster run), mercurydust (V31), Roman Tamrazov (*Who am I?*, the typography likelihood), dhruvtibarewal (nlm-v27), avazaku (the verified reproduction), lightningv08, kirilltym (*0.639 Teach More, Keep the Signature* and the spectral style heads), fleongg (*NTL Part Two Complement*), Nishant Dahal (*Two-Stage LoRA Part Two*) and the authors of every public model and dataset attached.
All trained weights remain theirs.

- **Commit-only cheap path:** `FAST_COMMIT` is true only when the competition rerun flag (`KAGGLE_IS_COMPETITION_RERUN`) is absent and the test file is the small public sample, so a save/commit finishes in about 10 GPU minutes; the hidden competition rerun always runs the full pipeline. Submit this notebook to get the real run.
- Our own experiments, what did not work, and how noisy a single public score is are in the next two sections.


## What we tried that did not work
Public leaderboard scores of our own submissions, each against the base it changed.
Each one was a single change to a public notebook run verbatim first, so the difference is the change.

| base (public score) | change | result |
|---|---|---|
| nlm-v27 (0.651) | blend in our own five-backbone model | 0.648 |
| nlm-v27 (0.651) | swap the q8v5 member for a Qwen3.5-27B NF4 member (0.642 on its own) | 0.638 |
| nlm-v27 (0.651) | turn the Qwen3.5-9B token-attention member back on | 0.640 |
| our 5-backbone two-stage LoRA blend (nested CV 0.6865 offline) | run on the board, plain heads | 0.559-0.560 |
| same | prompt clusters + centred heads | 0.567 |
| same | plain + 1.5 x centred on clusters ("twin" rule) | 0.588 |
| same | twin rule + Hungarian assignment per cluster | 0.579 |
| lightningv08 LoRA + Adaptive Centering (0.541) | Hungarian assignment inside each test prompt cluster | 0.524 |
| saddamaditya Apex Stylometric Centering (0.542) | the same Hungarian assignment | 0.525 |
| zbigniewkirchner v6 (0.552) | within-Gemini gate (stylegram LinearSVC, extra Gemini passages) | 0.552 |
| obstacledeveloper adaptive centered stylegram (0.556) | extra seed-11 LoRA24 member | 0.553 |
| obstacledeveloper stylegram (0.559) | 4 intra-family gates (claude / gpt / gemini / glm) | 0.559 |
| nishantdahal two-stage LoRA (0.592) | our two-stage Qwen2.5-7B LoRA as a third member | 0.592 |
| our 0.597 | lightningv08 OLMo2 native24 member, log-prob weight 0.03 | 0.594 |
| our 0.597 | swap lora24 seed 7 for an in-style seed-11 LoRA24 | 0.597 |

What we take from it:
- Our own blend scored 0.6865 in nested CV but 0.56 on the board.
  The offline folds used true 18-passage prompt groups, so centring and Hungarian assignment looked strong offline.
  On the board, the assignment tricks lost every time (-0.009 to -0.017), which points to few passages per prompt in the hidden test.
- Adding more LoRA backbones or members to a strong ensemble moved the score by at most 0.004 in either direction.
- The two changes that did help were small: a 4th LoRA member on v6 (0.552 -> 0.556) and the prvsiyan 0.569 blend as one extra log-prob member on nishantdahal (0.592 -> 0.597).


## How noisy is one public score?

Before reading any table of gains, including ours above: the same notebook, submitted again, does not score the same.
These are our own submissions of code we did not change, against each other and against the score on the author's page.

| pipeline | identical code, our submissions | author's page |
|---|---|---|
| dhruvtibarewal nlm-v27 | 0.651, 0.651, 0.643, 0.651 | 0.651 |
| Roman Tamrazov *Who am I?* | 0.645, 0.654 | 0.654 |
| mercurydust V31 | 0.660, 0.660 | 0.660 |
| omerfarukmerey 0.663 (this notebook) | 0.662 | 0.663 |
| Lucas Boesen LoRA x Adaptive Centering | 0.549 | 0.547 |
| vnton37nguyn lora-instyle | 0.561 | 0.563 |
| josephadamski91 Teach the Model + ATLAS | 0.606 | 0.607 |

The spread is up to **0.009** on identical code (0.645 vs 0.654, 0.643 vs 0.651).
So a single submission that gains less than about 0.01 is not evidence that the change helped; most single-change results in the table above are inside that band.
A related trick from the forum (GoPain, in a comment on discussion 753177): scoring out-of-fold on 2 answers per prompt instead of all 18 tracked the board far better (correlation 0.91 against 0.13 over their 15 submissions).


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