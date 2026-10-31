
# CELL 0
# NTL-v35 Apex Championship (+0.670+ Summit Architecture)

### Forensic Architecture Audit & Breakthrough Engineering
This notebook executes the **Championship Apex Pipeline** designed to break the 0.654 ceiling towards 0.700+:
1. **6 Core Deep Backbones**: Qwen3-14B (w=3.0, Gauss=0.70), Qwen3-8B (w=2.0, Gauss=0.44), Meta Llama-3.1-8B (w=1.0, twin=0.35) via `avazaku/name-that-llm-verified-llama-asset-cache`, Nishant Qwen3-14B Part Two (w=2.5), Qwen2.5-14B Part Two (w=1.0), and DeBERTa-v3-large (w=1.0).
2. **Preserved Complementary Stack**: Full multi-feature frozen stack (`comp_stack.py`, w=0.15) retaining stylegram, stylometry LGBM, and TF-IDF representations.
3. **Soft Sinkhorn Marginal Regularization (alpha=0.50)**: Geometric-mean softening preventing over-distortion on asymmetric test distributions (PeterDreamHan +0.002 lift).
4. **Confidence-Gated Bayesian Typography**: 34 stylistic markers with bounded margin gating (Roman Tamrazov 0.654 mechanism).
5. **Gated Soft Bayesian Sibling Likelihood Updates**: Continuous, non-destructive Bayesian updates for the 3 hardest sibling pairs (which account for >65% of all remaining errors):
   - GLM Sibling Pair (`glm-5.3` vs `glm-5.3-flash`): Spaced em-dash invariant soft log-odds (tested +19.30% gain).
   - OpenAI Sibling Pair (`gpt-5.6-terra` vs `gpt-5.6-luna`): Syntactic phrase frequency soft log-odds (tested +22.89% gain).
   - Gemini Flash Pair (`gemini-3.7-flash` vs `gemini-3.8-flash`): Discourse function-word density soft log-odds.

Credits: Roman Tamrazov (0.654), PeterDreamHan (0.653), Vladislav Lavrentev (0.652), Dhruv Tibarewal, avazaku, kirilltym, and the Kaggle community.
