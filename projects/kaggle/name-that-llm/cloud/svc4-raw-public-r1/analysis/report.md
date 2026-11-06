# SVC4 raw-public development verification

All required downloaded artifacts passed structural, hash and numerical reconciliation.

| Experiment | Public row weight | OOF balanced accuracy | Correct / 2699 |
|---|---:|---:|---:|
| EXP-017 | official only | 0.384484216 | 1038 |
| EXP-018 | 0.35 | 0.402289336 | 1086 |
| EXP-019 | 0.175 | 0.397099180 | 1072 |
| EXP-020 | 0.7 | 0.390800398 | 1055 |

| Candidate minus reference | BA difference | Unadjusted paired prompt bootstrap 95% interval | Fixed / new errors | Fold direction + / - / tie |
|---|---:|---:|---:|---:|
| EXP-018 - EXP-017 | +0.017805121 | [+0.002997557, +0.032222222] | 266 / 218 | 4 / 1 / 0 |
| EXP-019 - EXP-018 | -0.005190157 | [-0.014074074, +0.003703704] | 72 / 86 | 1 / 4 / 0 |
| EXP-020 - EXP-018 | -0.011488939 | [-0.020370370, -0.002965387] | 56 / 87 | 1 / 4 / 0 |

The augmented-versus-official-only contrast measures adding raw public data, including its effect on vocabulary and IDF. The two other predeclared contrasts change only public row weight on shared fold matrices.

Intervals use 2,000 paired resamples of the 150 official prompt groups, seed 42. They are conditional on these fitted development models, without multiplicity adjustment or retraining uncertainty. They do not establish a leaderboard improvement or an automatic winner.

Sparse metrics use the canonical SHA-256 key seed:repeat:id, rank within each prompt, and take the first 2, 3, or 6 IDs, five repeats each. They combine all five folds before computing 18-class balanced accuracy: 60 variant/scenario results. Input order and size enumeration do not change the chosen IDs. Standard deviations describe reused-model sampling variation, not independent training repeats or standard errors.

This is an isolated svc4 experiment, not full ATLAS20 or the original matched+pool recipe. Of 5,400 raw/new texts, 5,376 already occur in the audited baseline projection. Public generation lengths and protocol decisions used official training statistics; this is not an untouched validation source. Supplied provenance logs do not independently authenticate provider weights.

Feature training ID membership and matrix/vocabulary/IDF fingerprint consistency were checked. Matrices, vocabularies, IDF vectors and fitted estimators were not saved, so their contents or the historical training process cannot be independently reconstructed from these receipts alone. Remote run/version binding remains the downloader's evidence responsibility.

Frozen local protocol SHA-256: `9117d147deb4c2ab27a6054655970445095e638bc11a2d506310bf73170f4855`.

See verified.json and the adjacent CSV files for exact comparisons, class recalls, error changes and sparse selections.
