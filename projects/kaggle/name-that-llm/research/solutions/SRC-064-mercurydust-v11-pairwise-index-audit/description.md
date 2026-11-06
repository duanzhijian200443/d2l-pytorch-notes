# Name That LLM — V34.1: V31 Strong Baseline + Evidence-aware A/B + Pairwise Verification

**Baseline:** the user's strongest V31 submission, Public LB **0.660**. This notebook is built directly from `NTLLM_Kaggle_V31_Improved.ipynb`. It does **not** inherit V32.3's external-source changes or V33's original A/B meta-classifier. The V31 ensemble, model checkpoints/adapters, expert weights, ATLAS/Sinkhorn/post-processing and submission contract remain the starting point.

## V34.1 changes

1. **A evidence-aware prototypes:** use Revision03 exact evidence spans (`dataset_A_cue_spans.csv`) mapped to active surface/semantic feature keys. Evidence snippets, rather than whole-response TF-IDF centroids, form feature prototypes. Unmentioned cues are never treated as negative.
2. **B prompt-context conditioning:** train lightweight task/domain/structure predictors from official answers using prompt-grouped OOF. B is used as context-agreement/nuisance information in pair features, not as direct model-identity labels. Reconstructed Prompt text and prompt IDs are not inference features.
3. **Pairwise identity verifier:** train a small Logistic Regression on balanced positive pairs (same model, different prompts) and hard negatives (different models under the same prompt; same-family different models; cross-family). At inference, compare each test answer with a small set of official reference answers for each of the 18 candidate models.
4. **Confidence-gated fusion:** add a conservative pairwise log-probability correction only when the untouched V31 ensemble has a small top-1/top-2 margin.

## Deliberate exclusions

- No 254k Trust-Worthy Pool or other new external corpus in V34.
- No Qwen/GLM/DeBERTa backbone fine-tuning.
- No V32.3 Sparse Expert stack.
- No V33 original A+B identity Meta Classifier.
- No forced 18-model-per-test-prompt assignment beyond the original V31 behavior.

## Required Kaggle Input

Attach the existing **NTLLM-ChatGPT-A_B** Kaggle Dataset containing Revision03. The notebook checks `data_manifest.json`, `inputs/train.csv`, A surface observations, A semantic candidates, A cue spans, and B prompt structure, and verifies exact official train ID/text/model alignment.

## Outputs

- `/kaggle/working/v34_pairwise_probs.npy`
- `/kaggle/working/v34_pairwise_audit.json`
- `/kaggle/working/submission.parquet`

The pairwise weight (`0.10`) and confidence gate (`0.35`) are conservative initial settings, not validated optima. The local smoke test checks the auxiliary pipeline only; full Kaggle GPU execution and leaderboard performance remain to be tested.


## V34.1 reliability fixes

- Split the `%%writefile v34_pairwise.py` definition and its execution into separate notebook cells.
- Generate and validate `v34_pairwise_probs.npy` before launching main inference.
- Build A evidence prototypes Prompt-Grouped OOF for official training rows (held-out Prompt groups do not contribute their own A evidence prototypes).
- Remove the unavailable-at-inference `same_family` pair feature while retaining same-family hard-negative training pairs.
- Apply Pairwise fusion after the original V31 Bayesian Typography finalizer so it is not overwritten.
