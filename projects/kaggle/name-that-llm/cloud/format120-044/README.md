# EXP044 cloud entry

Private notebook: `zhjianduan/name-that-llm-format120-044`. Local control record: `execution-state.json`; inspect the actual remote version before attempting anything new.

## Verified completion

Fixed V2 is COMPLETE (output version ID364606984). All348 archived files,10 fold/arm jobs,20 SVM fits,150 sparse fold caches and both2699x18 OOF matrices passed local and independent verification. Outputs and the original log are in `outputs-v2/`; acceptance is at `experiments/EXP-044-format120-parallel/metrics/verified.json` from the project root.

Raw BA0.4663882675 versus clean0.4656475267: net2 fewer correct, paired150-prompt95% CI crosses0. Do not promote formatting cleanup. Guard elapsed1693.799s (28m13.8s), peak9.844GiB, two workers, one attempt. These are Stylegram development OOF results, not leaderboard scores. See the experiment report for detailed scope and runtime comparison.

The frozen payload compares old raw120 versus its formatting-only clean120 view. It uses the original CPU container and mounted official data/public pool, no internet, two workers, all five prompt-group folds. The notebook runs a small real numerical-equivalence gate before fitting full data. There is no competition submission or GPU allocation.

The controller allows one save-only request followed by one full-run request, both with source/privacy/input/hardware/image readback. An uncertain acknowledgement must be inspected rather than retried. No heartbeat automation is created for this experiment.

The notebook writes `results/` and archives it as `exp044-results.zip` on normal exit or a catchable failure. Recovery must preserve the entire tree, including dot files, budgets, attempts and completion receipts. Only verified completed fold/arm jobs may be reused. Cross-session recovery needs actually published Kaggle outputs; a new runtime does not inherit the old working disk. No automatic rerun is enabled.

After completion, download the fixed-version archive into this directory, extract into a fresh outputs directory, then run the read-only acceptance command from the project root:

```powershell
python -B -X utf8 experiments/EXP-044-format120-parallel/verify_outputs.py --results cloud/format120-044/outputs-v2/results --output experiments/EXP-044-format120-parallel/metrics/verified.json
```

Do not infer success from a process exit code or a notebook title. Require the full artifact, environment, self-test, guard, ten-job and prediction checks. Interpret clean120 minus raw120 using paired prompt uncertainty; keep the formal 0.662 baseline unchanged.
