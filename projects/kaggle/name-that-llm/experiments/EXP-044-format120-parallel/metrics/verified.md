# EXP044 frozen result verification

PASS_FROZEN_RAW_CLEAN_OOF

Run: exp044-20261101T163138Z. Ten sealed jobs,150 sparse fold caches and two2699x18 OOF arrays verified.

Raw BA: 0.4663882675; clean BA: 0.4656475267; clean-raw: -0.0007407407.
Fixed errors: 2; introduced errors: 4; net correct: -2.
150-prompt paired bootstrap95% CI: [-0.0025925925925925925, 0.0011111111111111111] (2000 draws, seed42).

| Sparse m | Raw mean BA | Clean mean BA | Delta |
|---|---:|---:|---:|
| 2 | 0.36255887 | 0.36323856 | +0.00067969 |
| 3 | 0.39465659 | 0.39343170 | -0.00122489 |
| 6 | 0.43297559 | 0.43258080 | -0.00039480 |

EXP042 control is historical and is not a third retrained arm.

Validation is development Stylegram OOF only. No holdout, ATLAS20 or leaderboard improvement is established. Bootstrap excludes retraining uncertainty.
