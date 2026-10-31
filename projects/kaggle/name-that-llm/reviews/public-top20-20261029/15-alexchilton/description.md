
# CELL 0
# Name That LLM: 0.651 on the public board

This version runs the public 0.651 pipeline unchanged; its submission scored **0.651** on the public leaderboard.
Our earlier five-LoRA-backbone notebook (0.69 offline, 0.56 on the board) is replaced by it.

## What it does
- Fold-averaged two-stage LoRA24 heads on Qwen3-8B-Base, Qwen3-14B-Base, Qwen2.5-7B/14B and Llama-3.1-8B.
- Nishant's two-stage checkpoints, a DeBERTa-v3-large encoder, a 4-view n-gram SVM on train + pool.
- Shared-prompt Gaussian group head, adaptive centred twins, a 10% ATLAS source-conditioned likelihood, Sinkhorn balancing.
- A complementary frozen-feature stack in its own process.

## Only changes
- Llama-3.1-8B is mounted from `alexchilton/llama-3-1-8b-base` (the unmodified model).
- **Commit-only cheap path:** the heavy passes run only in the hidden competition rerun (`KAGGLE_IS_COMPETITION_RERUN`). A save/commit writes a placeholder submission in a few minutes instead of about 2 GPU hours. Submit this notebook to get the real run.

## Credits
All credit for the pipeline goes to its authors:
**dhruvtibarewal** (nlm-v27 notebook, adapters, matched data, wheels), **lucasboesen** (GLM/encoder/LoRA24/token-attention training notebooks, proxy generations),
**nishantdahal** (two-stage LoRA, handwriting blend), **josephadamski91** (ATLAS Qwen3-8B), **prvsiyan**, **kirilltym** and **sangam21** (shared-prompt Gaussian, extended dataset),
**vnton37nguyn** (Qwen2.5 in-style LoRA), **pjmathematician** (UltraBERT), **zbigniewkirchner** (Qwen3-8B LoRA head),
**seyitkaangunes**, **abdullah1afify** and **dophuchung** (extra and proxy generations). If this helps you, upvote their notebooks and datasets too.


# CELL 1
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
