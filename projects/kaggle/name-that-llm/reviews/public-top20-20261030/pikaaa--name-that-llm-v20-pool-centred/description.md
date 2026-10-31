# Name That LLM 0.660: Full Ensemble + What Failed

This version runs mercurydust's public *[Name That LLM-Submission](https://www.kaggle.com/code/mercurydust/name-that-llm-submission)* (*Who am I? V31*, 0.660) unchanged, apart from the commit-only gate below; on our account it scored **0.660** on the public leaderboard (2026-10-28). V31 builds on Roman Tamrazov's *[Name That LLM | Who am I?](https://www.kaggle.com/code/romantamrazov/name-that-llm-who-am-i)* (0.654, version 4 of this notebook): dhruvtibarewal's nlm-v27 0.651 ensemble plus a train-only Bayesian typography likelihood, with V31's changes to auxiliary data usage, family-level gating and adaptive probability balancing (TF-IDF and distortion members on, softer Sinkhorn). Earlier versions ran the 0.654 pipeline (0.654 and 0.645 on two runs of identical code) and the 0.651 pipeline (0.651, 0.643, 0.651).

**One change to the pipeline.** Its Llama-3.1-8B base came from the kernel `avazaku/name-that-llm-verified-llama-asset-cache`, which is no longer readable. This version reads the same unmodified weights from the public model [alexchilton/llama-3-1-8b-base](https://www.kaggle.com/models/alexchilton/llama-3-1-8b-base) instead (one line in the setup cell).

**Credits.** mercurydust (V31, this pipeline), Roman Tamrazov (*Who am I?*, the typography likelihood), lightningv08, dhruvtibarewal (nlm-v27), kirilltym (*0.639 Teach More, Keep the Signature* and the spectral style heads), fleongg (*NTL Part Two Complement*), Nishant Dahal (*Two-Stage LoRA Part Two*) and the authors of every public model and dataset attached. All trained weights remain theirs.

- **Commit-only cheap path:** the heavy passes run only in the hidden competition rerun (`KAGGLE_IS_COMPETITION_RERUN`). A save/commit writes a placeholder submission in a few minutes instead of about 2.7 GPU hours. Submit this notebook to get the real run.
- Our own experiments, and what did not work, are in the next section.


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
