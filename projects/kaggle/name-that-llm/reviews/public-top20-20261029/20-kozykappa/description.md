
# CELL 0
# Name That LLM — enhanced 0.651 recipe

the input checker incorrectly
expected 32 layers for Qwen3-8B-Base. The mounted and official model has 36.
The check now requires 36, with independent architecture fixtures and an actual
preflight CLI regression. Inputs, prediction logic and weights are unchanged.
Official source: https://huggingface.co/Qwen/Qwen3-8B-Base/blob/main/config.json

This edits the supplied **0.651 Verified Reproduction**, rather than the earlier
0.643 notebook. The quoted 0.651 is the source author's reported baseline score;
**this enhancement has no measured leaderboard score yet**.

All **31 input references**, six main members, 18 v5 adapters, Gaussian/twin
heads, complementary stack and 10% ATLAS are retained. Neural forward functions,
precision, pooling, readouts and original ensemble weights remain unchanged.

The prediction experiment softly removes some of the existing lexical SVC's
inferred topic bias in groups of 8–18 responses. It uses the parent's text-only
grouping, confidence gating, and the SVC's original share in the blend. It adds
**zero GPU forward passes**. Sparse groups are unchanged. The original balancing,
typography rules and dense assignment remain; no new class quota is introduced.

## Measured evidence

Fresh train-only four-view SVC, **5 outer prompt folds × 3 inner folds**; vocabulary
and IDF refitted in every fold. Strength selected only on inner folds, averaging
complete, sparse and uneven-label scenarios. No public pool or released
task-trained adapter enters this experiment.

| Scenario | Baseline balanced accuracy | Candidate | Change |
|---|---:|---:|---:|
| full (2699 rows) | 0.38448 | 0.40191 | +0.01742 |
| sparse (373 rows) | 0.41289 | 0.41289 | +0.00000 |
| uneven (1546 rows) | 0.37823 | 0.39154 | +0.01331 |

These are **SVC proxy results before the full ensemble's postprocessing**, not
OOF results for the neural ensemble. Whole-prompt bootstrap intervals, split IDs,
predictions and reproducible validation source are in the accompanying package.
An improvement above public LB 0.651 remains unverified.

## Run on Kaggle

1. Import this notebook into a copy of the supplied reproduction and keep its
   **same 31 inputs**, including the verified base Llama cache.
2. Select **GPU T4 x2**, **Internet off**. Keep the first cell's defaults:
   `RUN_MODE="kaggle"`, `RUN_VALIDATION=False`.
3. **Save Version → Save & Run All**. The first cell prints a run marker;
   `INPUT PREFLIGHT: PASS` precedes any large model. This notebook performs real
   inference even for the three visible rows; there is no hardcoded preview.
4. Inspect `run_report.json`, `complement.log` and `main_inference.log`.
   `full_recipe_completed` must be true for a complete run. Submit the saved
   version that writes `/kaggle/working/submission.parquet`.

The source documentation reports **4 h 4 min** on the three visible rows on T4 x2.
That is not a hidden-test runtime forecast. Allow several hours; this version
enforces an **8.5-hour total deadline** and saves before optional ATLAS, reserving
time for final files. A skipped ATLAS is explicitly marked as a degraded run.
Required main/complement failures stop execution rather than silently dropping
models. Kaggle's outer 9-hour limit remains authoritative.

## What changed and how to review it

* Checked subprocesses retain actual errors and all logs; no `grep` pipeline
  masks a failed worker. Optional ATLAS runs separately with a hard timeout.
* Model discovery verifies base architecture and shard/tokenizer presence,
  follows linked mounts and collects missing inputs in one report. Unused old
  adapter families are no longer unnecessary startup dependencies.
* Zero-weight TF-IDF and unused family-gate fitting are skipped. Repeated
  in-sample checks are opt-in; checkpoint-anchor checks are retained.
* Exact normalized test-text overlaps in both active external pools are excluded and
  counted. This is not an authentication or near-duplicate audit of the public
  pools; upstream provenance remains a limitation.
* Both the selected candidate `submission.parquet` and a same-run baseline
  `submission_baseline_0651.parquet` are saved atomically with checked ID order.
  `main_evidence.npz` and `final_evidence.npz` retain reusable predictions.
  This baseline uses the same overlap filtering and any actual ATLAS completion;
  its filename does not confer a separately verified 0.651 score.

Set `TOPIC_ALPHA=0.0` to disable the sole new score correction. CPU fixtures
verify numerical baseline parity for grouping, final postprocessing and the
actual original-vs-enhanced train-only CLI. The full released neural assets were
not available for local GPU execution.

For an optional rerun of nested SVC validation, set `RUN_VALIDATION=True`; it
does not tune on test data or alter the selected constant automatically. For a
local plumbing check, use `RUN_MODE="cpu_smoke"`; it writes CPU-only predictions
and is not a competition ensemble submission. `preflight` checks inputs only.

Credits: original recipe and v5 adapters by **Dhruv Tibarewal**; supplied credited
reproduction by **avazaku**. Existing public-stack author comments are retained.
The soft SVC topic-correction idea follows the previously inspected credited
Roman Tamrazov public implementation, with contribution scaling and the exact
v27 grouping here. Attached models/data retain their own terms. **Built with Llama.**


# CELL 2
The following cells write the offline runtime. Run all cells in order; no additional package or helper upload is required.

# CELL 11
This final cell starts the selected mode and verifies the output. Kaggle mode retains the original model ensemble. Logs and a completion report are written beside the submission.