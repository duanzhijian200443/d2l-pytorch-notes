"""Independent CSV-only metric audit; does not import experiment metric helpers."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "analysis-v1" / "predictions.csv"
rows = list(csv.DictReader(SOURCE.open(encoding="utf-8", newline="")))
assert len(rows) == 729 and len({r["id"] for r in rows}) == 729
assert all(all(r[k] for k in ("id", "prompt_id", "model", "prediction_045", "prediction_030")) for r in rows)
labels = sorted({r["model"] for r in rows})
groups = sorted({r["prompt_id"] for r in rows})
assert len(labels) == 17 and len(groups) == 56
support = Counter(r["model"] for r in rows)
correct = {arm: Counter(r["model"] for r in rows if r["model"] == r[f"prediction_{arm}"]) for arm in ("045", "030")}
metrics = {arm: {"correct": sum(correct[arm].values()), "accuracy": sum(correct[arm].values()) / len(rows),
                 "macro_recall": sum(correct[arm][label] / support[label] for label in labels) / len(labels)} for arm in correct}
cells = Counter((r["model"] == r["prediction_045"], r["model"] == r["prediction_030"]) for r in rows)
per_class = [{"label": label, "support": support[label], "correct_045": correct["045"][label], "correct_030": correct["030"][label],
              "recall_045": correct["045"][label] / support[label], "recall_030": correct["030"][label] / support[label],
              "delta": (correct["030"][label] - correct["045"][label]) / support[label]} for label in labels]
# Aggregate class counts per group. Repeated group draws retain every row in that group.
gindex = {g: i for i, g in enumerate(groups)}
lindex = {l: i for i, l in enumerate(labels)}
denominator = np.zeros((len(groups), len(labels)), dtype=np.int64)
numerator_delta = np.zeros_like(denominator)
for row in rows:
    g, l = gindex[row["prompt_id"]], lindex[row["model"]]
    denominator[g, l] += 1
    numerator_delta[g, l] += int(row["prediction_030"] == row["model"]) - int(row["prediction_045"] == row["model"])
draws = np.random.default_rng(42).integers(0, 56, size=(2000, 56))
bootstrap_values = []
invalid = 0
for sample in draws:
    multiplicity = np.bincount(sample, minlength=56)
    sampled_support = multiplicity @ denominator
    if np.any(sampled_support == 0):
        invalid += 1
    else:
        bootstrap_values.append(float(np.mean((multiplicity @ numerator_delta) / sampled_support)))
ci = None if invalid else np.quantile(bootstrap_values, [.025, .975], method="linear").tolist()
absent_label = "gemini-3.7-flash"
assert absent_label not in support
changes = [r for r in rows if r["prediction_045"] != r["prediction_030"]]
absent_enter = [r for r in changes if r["prediction_030"] == absent_label]
absent_exit = [r for r in changes if r["prediction_045"] == absent_label]
summary = json.loads((ROOT / "analysis-v1" / "summary.json").read_text(encoding="utf-8"))
checks = {
    "accuracy_045": abs(metrics["045"]["accuracy"] - summary["baseline_045"]["accuracy"]) < 1e-14,
    "accuracy_030": abs(metrics["030"]["accuracy"] - summary["candidate_030"]["accuracy"]) < 1e-14,
    "macro_recall_045": abs(metrics["045"]["macro_recall"] - summary["baseline_045"]["macro_recall"]) < 1e-14,
    "macro_recall_030": abs(metrics["030"]["macro_recall"] - summary["candidate_030"]["macro_recall"]) < 1e-14,
    "bootstrap_ci": ci is not None and bool(np.allclose(ci, summary["paired_prompt_bootstrap"]["ci95"], atol=1e-14, rtol=0)),
    "four_cells": all(cells[key] == summary[name] for key, name in [((True, True), "both_correct"), ((True, False), "only_045_correct"), ((False, True), "only_030_correct"), ((False, False), "both_wrong")]),
    "prediction_disagreements": len(changes) == summary["prediction_disagreements"],
}
output = {
    "status": "PASS" if all(checks.values()) else "FAIL",
    "scope": "independent_predictions_csv_metric_recalculation",
    "predictions_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "numpy_version": np.__version__,
    "rows": len(rows), "prompt_groups": len(groups), "covered_labels": labels,
    "metrics": metrics, "delta": {k: metrics["030"][k] - metrics["045"][k] for k in metrics["045"]},
    "four_cells": {"both_correct": cells[True, True], "only_045_correct": cells[True, False], "only_030_correct": cells[False, True], "both_wrong": cells[False, False]},
    "per_class": per_class,
    "paired_prompt_bootstrap": {"draws": 2000, "seed": 42, "valid_draws": len(bootstrap_values), "invalid_draws": invalid,
                               "ci95": ci, "ci95_percentage_points": None if ci is None else [x * 100 for x in ci],
                               "draws_delta_nonpositive": sum(x <= 0 for x in bootstrap_values),
                               "group_order": "lexicographically sorted prompt_id", "quantile_method": "linear",
                               "statistic": "mean of 17 class recall differences after aggregating sampled-group correct counts and support"},
    "changed_predictions": changes,
    "missing_true_class": {"label": absent_label, "support": 0,
                           "predicted_045": sum(r["prediction_045"] == absent_label for r in rows),
                           "predicted_030": sum(r["prediction_030"] == absent_label for r in rows),
                           "entered": absent_enter, "exited": absent_exit},
    "matches_published_summary": checks,
    "recommendation": "Retain baseline 0.45; observed gain is exploratory and CI includes zero.",
    "limitations": ["Only supplied predictions and metrics were independently verified, not model execution or source-cache acceptance.",
                    "Absent true Gemini 3.7 has undefined recall; its false positives are counted against other true classes, but potential recovery for actual Gemini 3.7 cannot be assessed.",
                    "Class balancing is panel-composition-dependent; 17-class panel cannot establish full 18-class or leaderboard performance.",
                    "Bootstrap is conditional on existing predictions and selected prompt groups; no retraining, exposure-independence, representativeness or selection-uncertainty claim."]
}
destination = Path(__file__).with_suffix(".json")
destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: output[k] for k in ("status", "metrics", "delta", "four_cells", "paired_prompt_bootstrap", "missing_true_class", "matches_published_summary")}, ensure_ascii=False, indent=2))
