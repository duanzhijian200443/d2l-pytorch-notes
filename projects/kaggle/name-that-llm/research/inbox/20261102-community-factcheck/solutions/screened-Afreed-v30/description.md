# 0.664 | Open Pool in the SVC + ATLAS 20%

I took the best public recipe and changed one thing: the text the 4-view SVC learns from. That was enough for **0.664** on the public leaderboard, with every input public.

## What I changed

- **SVC pool:** I swapped `nlm_pool_v5` (120k passages) for [crtproduction's Name That LLM Open Pool](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool) (about 198k cleaned passages), minus the answers to train prompts. In my prompt-grouped CV the SVC alone went from 0.4805 to 0.5008 (5 folds, pool weight 0.35, measured on the pool's 193k version).
- **`--svc_w 12` instead of 1.5:** the SVC's raw decision values are very flat, so at 1.5 it has almost no say in the blend.

That is the whole diff against the recipe: one block at the end of the setup cell and one number in the run command.

## What I got

| `svc_w` | public LB |
|---|---|
| 1.5 | 0.664 |
| 6 | 0.659 |
| **12 (this notebook)** | **0.664** |
| 20 | 0.664 |

The code behind the recipe scored 0.662-0.663 before the pool swap. Read these numbers with care: one public score moves by up to about 0.009 on identical code, as [alexchilton measured](https://www.kaggle.com/code/alexchilton/name-that-llm-0-662-all-public-what-failed), and my ladder shows no clean trend with `svc_w`. So the higher weight is not a proven gain, and the pool's gain is still small next to that noise.

## Run it yourself

A commit takes about 10 GPU minutes, because it only writes a placeholder on the 3 visible test rows. Submit the notebook to get the real run on the hidden test, which takes about 8 hours.

## Credits

All the models and the pipeline are other people's work, and I only made the change above:
[crtproduction](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool) (Open Pool, CC BY-SA 4.0),
[alex chilton](https://www.kaggle.com/code/alexchilton/name-that-llm-0-662-all-public-what-failed) (the all-public version this is built on, the public [Llama-3.1-8B upload](https://www.kaggle.com/models/alexchilton/llama-3-1-8b-base), the fast commit path, and a good list of what failed),
[omerfarukmerey](https://www.kaggle.com/code/omerfarukmerey/0-663-0-660-v31-atlas-20) (ATLAS 20%, the faster run),
[mercurydust](https://www.kaggle.com/code/mercurydust/name-that-llm-submission) (V31),
[Roman Tamrazov](https://www.kaggle.com/code/romantamrazov/name-that-llm-who-am-i) (*Who am I?*, the typography likelihood),
[Dhruv Tibarewal](https://www.kaggle.com/code/dhruvtibarewal/nlm-v27) (nlm-v27, the LoRA members),
avazaku (the verified reproduction), lightningv08, kirilltym, fleongg, Nishant Dahal, prvsiyan, Lucas Boesen, vnton37nguyn, sangam21, Joseph Adamski (ATLAS), and the authors of every public model and dataset attached. All trained weights remain theirs. Llama-3.1 adapters: *Built with Llama* (Llama 3.1 Community License).


## 1. Inputs
Finds every base model, adapter and checkpoint under `/kaggle/input`. Llama-3.1-8B comes from alexchilton's public upload (shard sizes asserted) and is copied to local disk first. The last block swaps the SVC pool for the Open Pool. `FAST_COMMIT` is true only on the 3 visible test rows outside the competition rerun.


## 2. Shared-prompt Gaussian head (`gauss.py`)
A two-way model *x = author mean + prompt effect + noise*, integrated over the unknown prompt, with injective group marginals (prvsiyan / Kirill Tymchyshyn / sangam21, Apache 2.0).

## 3. Main pipeline (`infer.py`)
Dhruv Tibarewal's pipeline with Roman Tamrazov's and mercurydust's changes: pooled hidden states of each LoRA24 member, plain and prompt-centred heads, the n-gram SVM (on official + external + pool rows), the TF-IDF and text-distortion members, test prompt recovery, the weighted log-probability blend, the complementary stack, ATLAS, V31's soft Sinkhorn and its typography step, which writes the submission. omerfarukmerey's changes: in `l24_predict`, members marked `"dp"` (the 8B ones) load a copy on each GPU and alternate batches between them, and every member averages its hidden states on the GPU; ATLAS no longer stops on a projected overrun.

## 4. Complementary stack (`comp_stack.py`)
The public frozen-feature ensemble (prvsiyan, Lucas Boesen, vnton37nguyn, Kirill Tymchyshyn), mixed in at 15%, with V31's within-family gates. Optional token-attention heads are skipped when their backbone is not attached.

## 5. Run the complementary stack
In its own process, so all GPU memory is free again afterwards; its wall time is charged to ATLAS's budget.

## 6. Members and run
The 0.651 members and weights with V31's run flags (`--tfidf_w 0.25 --distort_w 0.10 --sinkhorn 0.45 --sinkhorn_target_mix 0.35 --post 0`). On the 3 visible rows the commit run writes a placeholder; Kaggle's rerun on the ~4,000 hidden rows runs every cell. The run command passes `'--atlas_w', '0.20'` (V31: 0.10). My one change to the numbers here: `--svc_w 12` (recipe: 1.5).

