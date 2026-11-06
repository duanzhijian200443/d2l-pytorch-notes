"""Independent arithmetic audit of downloaded EXP-027 V2 scores and rows.

No training or remote calls occur. This script does not import the runner,
margin-blend implementation, or primary verifier. The prompt bootstrap expands
the sampled original rows explicitly instead of using prompt-count products.
Run --self-test before results exist; run --input only after a complete download.
"""
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CLOUD = ROOT / "cloud/fixed-blend-multiclass"
OUTPUT = HERE / "metrics/independent-review.json"
PROTOCOL_SHA = "c3e2b1aab6744a41dc546281f0eb00d61f0d5e10564415444b4bb4e1267ce886"
EVALUATION_SHA = "8ec3217f1f469b66170f68f7abdb3ef1973a64e47d983c59b40a4d1ce2094670"
GATE_SHA = "cead2f4745a3e4af6a8adffe2361efd725e2710971a12123083b927589f573e5"
LABELS = ["claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
          "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
          "glm-5.3", "glm-5.3-flash", "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
          "gpt-6-astra", "kimi-k3", "qwen3.8-max"]
COVERED = [index for index in range(18) if index != 7]
CLASS_COUNTS = dict(zip((LABELS[index] for index in COVERED),
                       (19, 55, 55, 54, 52, 54, 51, 54, 16, 17, 54, 52, 54, 53, 54, 17, 18)))
MEMBERS = ("stylegram", "svc4", "blend")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        check(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def nonfinite_json(value):
    raise ValueError("Nonfinite JSON constant: " + value)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=strict_pairs,
                      parse_constant=nonfinite_json)


def match(actual, expected, path):
    if isinstance(expected, dict):
        check(isinstance(actual, dict) and set(actual) == set(expected), "Different JSON fields: " + path)
        for key, value in expected.items():
            match(actual[key], value, path + "." + key)
    elif isinstance(expected, list):
        check(isinstance(actual, list) and len(actual) == len(expected), "Different JSON list: " + path)
        for index, value in enumerate(expected):
            match(actual[index], value, path + f"[{index}]")
    elif isinstance(expected, float):
        check(type(actual) in (float, int) and math.isfinite(actual)
              and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), "Different numeric value: " + path)
    else:
        check(type(actual) is type(expected) and actual == expected, "Different JSON value: " + path)


def normalized_rows(values):
    """Population z-score, calculated separately for each full 18-value row."""
    result = np.zeros(values.shape, dtype=np.float64)
    for index, original in enumerate(np.asarray(values, dtype=np.float64)):
        scale = float(max(abs(original)))
        if scale == 0:
            continue
        row = original / scale
        centered = row - np.mean(row)
        deviation = np.sqrt(np.dot(centered, centered) / len(row))
        if deviation != 0:
            result[index] = centered / deviation
    return result


def fixed_blend(style, svc, saved):
    check(style.shape == svc.shape == saved.shape and len(style.shape) == 2 and style.shape[1] == 18,
          "Expected matching 18-column matrices")
    check(all(array.dtype.kind == "f" and np.isfinite(array).all() for array in (style, svc, saved)),
          "Nonfinite/nonfloating score matrix")
    recomputed = .9 * normalized_rows(style) + .1 * normalized_rows(svc)
    check(np.allclose(saved, recomputed, rtol=1e-12, atol=1e-12), "Fixed .10 blend arithmetic differs")
    check(np.array_equal(saved.argmax(axis=1), recomputed.argmax(axis=1)),
          "Saved blend argmax differs from independent .10 blend")
    return recomputed, float(np.max(np.abs(saved - recomputed)))


def summarize(truth, prediction):
    rows = []
    for index in COVERED:
        selected = truth == index
        support = int(np.count_nonzero(selected))
        check(support > 0, "Population lacks a frozen true class")
        correct = int(np.count_nonzero(prediction[selected] == index))
        rows.append({"model": LABELS[index], "support": support, "correct": correct, "recall": correct / support})
    return {"metric": "covered_label_macro_recall", "score_columns": 18,
            "covered_labels": [LABELS[index] for index in COVERED], "covered_label_count": 17,
            "macro_recall": sum(row["recall"] for row in rows) / 17,
            "accuracy": int(np.count_nonzero(truth == prediction)) / len(truth),
            "correct": int(np.count_nonzero(truth == prediction)), "rows": len(truth), "per_class": rows}


def bootstrap_rows(truth, reference, prediction, prompts):
    """Retain all 2000 sampled panels, including missing-class panels as null."""
    names = sorted(set(prompts))
    rows_by_prompt = [np.flatnonzero(prompts == name) for name in names]
    check(len(names) > 0 and all(len(rows) > 0 for rows in rows_by_prompt), "Empty bootstrap population")
    random = np.random.default_rng(42)
    deltas = []
    invalid_indices = []
    for draw in range(2000):
        chosen = random.integers(0, len(names), size=len(names))
        selected = np.concatenate([rows_by_prompt[index] for index in chosen])
        actual = truth[selected]
        changes = []
        for index in COVERED:
            positions = selected[actual == index]
            if len(positions) == 0:
                changes = None
                break
            new_correct = int(np.count_nonzero(prediction[positions] == index))
            old_correct = int(np.count_nonzero(reference[positions] == index))
            changes.append((new_correct - old_correct) / len(positions))
        if changes is None:
            invalid_indices.append(draw)
            deltas.append(None)
        else:
            deltas.append(sum(changes) / 17)
    interval = None if invalid_indices else np.quantile(deltas, [.025, .975], method="linear").tolist()
    summary = {"delta_definition": "blend_minus_stylegram_covered_label_macro_recall", "unit": "external_prompt_id",
               "draws": 2000, "seed": 42, "valid_draws": 2000 - len(invalid_indices),
               "invalid_draws": len(invalid_indices), "ci95": interval,
               "missing_class_policy": "null interval if any resample lacks a covered true class",
               "conditional_on_fitted_models": True, "includes_retraining_uncertainty": False}
    return summary, deltas, invalid_indices


def self_test():
    checks = []
    style = np.vstack((np.arange(18, dtype=float), np.ones(18)))
    normalized = normalized_rows(style)
    np.testing.assert_allclose(normalized[0], (style[0] - 8.5) / style[0].std(ddof=0), atol=1e-14)
    np.testing.assert_array_equal(normalized[1], np.zeros(18))
    fixed_blend(style, -style, .8 * normalized)
    checks.append("population_zscore_fixed_point_one_and_constant_rows")
    near = np.full((1, 18), -1.); near[0, 0] = 1.; near[0, 1] = np.nextafter(1., 0.)
    saved = normalized_rows(near); saved[0, 1] += 5e-13
    try:
        fixed_blend(near, near, saved)
    except ValueError as error:
        check("argmax" in str(error), "Expected near-tie argmax rejection")
    else:
        raise AssertionError("Near-tie argmax flip passed")
    checks.append("near_tie_argmax_flip_rejected")
    truth = np.r_[COVERED, COVERED[:-1], COVERED[:1]]
    prompts = np.array(["p0"] * 17 + ["p1"] * 16 + ["p2"])
    reference = truth.copy(); reference[truth == COVERED[0]] = 7
    metrics = summarize(truth, reference)
    check(metrics["macro_recall"] == 16 / 17 and metrics["accuracy"] == 31 / 34, "Macro collapsed to accuracy")
    scores = np.zeros((34, 18)); scores[:, 7] = 1
    check(np.all(scores.argmax(axis=1) == 7), "Excluded true class was masked at prediction time")
    checks.append("imbalanced_macro_and_unmasked_eighteen_way")
    boot, deltas, invalid = bootstrap_rows(truth, reference, truth, prompts)
    expected_choices = np.random.default_rng(42).integers(0, 3, size=(2000, 3))
    expected_invalid = np.flatnonzero(np.all(expected_choices != 0, axis=1)).tolist()
    check(invalid == expected_invalid and boot["ci95"] is None and len(deltas) == 2000
          and sum(value is None for value in deltas) == len(invalid), "Missing bootstrap draws were dropped")
    checks.append("missing_class_bootstrap_null_without_dropped_draws")
    truth = np.tile(COVERED, 10); prompts = np.repeat([f"p{i}" for i in range(10)], 17)
    reference = truth.copy(); reference[:17] = 7
    boot, deltas, invalid = bootstrap_rows(truth, reference, truth, prompts)
    choices = np.random.default_rng(42).integers(0, 10, size=(2000, 10))
    expected = (choices == 0).sum(axis=1) / 10
    np.testing.assert_allclose(deltas, expected, atol=1e-15)
    np.testing.assert_allclose(boot["ci95"], np.quantile(expected, [.025, .975], method="linear"))
    check(not invalid, "Complete panel unexpectedly lost a class")
    checks.append("explicit_row_bootstrap_matches_analytic_complete_panel")
    return {"status": "SELF_TEST_PASS", "checks": checks, "real_result_files_read": False}


def run(directory):
    directory = directory.resolve(strict=True)
    check(directory.is_dir() and not (directory / "failure.json").exists(), "Missing/failed results directory")
    pinned = {CLOUD / "payload/protocol.json": PROTOCOL_SHA, HERE / "evaluation.csv": EVALUATION_SHA,
              HERE / "metrics/data-gate.json": GATE_SHA}
    for path, digest in pinned.items():
        check(sha(path) == digest, "Changed frozen input: " + str(path))
    protocol = load_json(CLOUD / "payload/protocol.json")
    check(protocol["labels"] == LABELS and protocol["evaluation_labels"] == [LABELS[index] for index in COVERED]
          and protocol["evaluation_class_counts"] == CLASS_COUNTS and protocol["alpha"] == .1,
          "Changed frozen label/weight protocol")
    evaluation = pd.read_csv(HERE / "evaluation.csv", dtype=str, keep_default_na=False)
    check(evaluation.columns.tolist() == ["id", "prompt_id", "model", "text"] and len(evaluation) == 729
          and evaluation.id.is_unique and evaluation.prompt_id.nunique() == 56
          and evaluation.model.value_counts().to_dict() == CLASS_COUNTS, "Changed frozen evaluation population")
    names = ["component-and-blend-scores.npz", "stylegram-scores.npz", "predictions.csv", "confusion.csv",
             "metrics.json", "result.json", "artifacts.json", "guard.json", "state.json"]
    check(all((directory / name).is_file() and not (directory / name).is_symlink() for name in names),
          "Missing/symlink audit input")
    read_hashes = {directory / name: sha(directory / name) for name in names}
    manifest = load_json(directory / "artifacts.json")
    entries = manifest["files"]
    check(len(entries) == 18 and len({entry["path"] for entry in entries}) == 18, "Incomplete/duplicate stable manifest")
    for name in names[:6]:
        selected = [entry for entry in entries if entry["path"] == name]
        check(len(selected) == 1 and selected[0]["sha256"] == read_hashes[directory / name]
              and selected[0]["bytes"] == (directory / name).stat().st_size, "Changed relevant manifest artifact: " + name)
    result, guard, state = (load_json(directory / name) for name in ("result.json", "guard.json", "state.json"))
    check(result["status"] == guard["status"] == state["status"] == "COMPLETE"
          and result["protocol_sha256"] == PROTOCOL_SHA and guard["child_exit"] == 0,
          "Not a complete run of the frozen protocol")
    with np.load(directory / "component-and-blend-scores.npz", allow_pickle=False) as saved:
        scores = {key: saved[key].copy() for key in saved.files}
    check(set(scores) == {"ids", "labels", "stylegram_scores", "svc4_scores", "blend_scores", "alpha", "group_ids", "centred_mask"},
          "Different score archive fields")
    check(scores["ids"].tolist() == evaluation.id.tolist() and scores["labels"].tolist() == LABELS
          and scores["alpha"].shape == () and scores["alpha"].dtype.kind == "f" and scores["alpha"].item() == .1,
          "Score IDs/labels/fixed alpha differ")
    check(all(scores[name + "_scores"].shape == (729, 18) for name in MEMBERS), "Wrong score population/width")
    recomputed, maximum_error = fixed_blend(scores["stylegram_scores"], scores["svc4_scores"], scores["blend_scores"])
    with np.load(directory / "stylegram-scores.npz", allow_pickle=False) as style:
        check(style["ids"].tolist() == evaluation.id.tolist() and style["labels"].tolist() == LABELS,
              "Style cache mapping differs")
        for left, right in (("scores", "stylegram_scores"), ("group_ids", "group_ids"), ("centred_mask", "centred_mask")):
            check(np.array_equal(style[left], scores[right]), "Style caches disagree: " + left)
    truth = evaluation.model.map({label: index for index, label in enumerate(LABELS)}).to_numpy(dtype=np.int64)
    predictions = {name: scores[name + "_scores"].argmax(axis=1) for name in MEMBERS}
    check(np.array_equal(predictions["blend"], recomputed.argmax(axis=1)), "Independent blend prediction differs")
    stored = pd.read_csv(directory / "predictions.csv", dtype=str, keep_default_na=False)
    expected_columns = ["id", "prompt_id", "model"]
    for name in MEMBERS:
        expected_columns.extend([name + "_prediction", name + "_correct"])
    expected_columns.extend(["stylegram_group_id", "stylegram_centred"])
    check(stored.columns.tolist() == expected_columns and len(stored) == 729, "Different prediction schema/row count")
    for key in ("id", "prompt_id", "model"):
        check(stored[key].tolist() == evaluation[key].tolist(), "Prediction row mapping differs: " + key)
    for name, prediction in predictions.items():
        check(stored[name + "_prediction"].tolist() == [LABELS[index] for index in prediction]
              and stored[name + "_correct"].tolist() == [str(bool(value)) for value in prediction == truth],
              "Prediction CSV differs from 18-way argmax: " + name)
    check(stored.stylegram_group_id.tolist() == scores["group_ids"].astype(str).tolist()
          and stored.stylegram_centred.tolist() == [str(bool(value)) for value in scores["centred_mask"]],
          "Prediction routing metadata differs")
    metrics = {name: summarize(truth, prediction) for name, prediction in predictions.items()}
    boot, deltas, invalid_indices = bootstrap_rows(truth, predictions["stylegram"], predictions["blend"],
                                                  evaluation.prompt_id.to_numpy())
    old_good, new_good = predictions["stylegram"] == truth, predictions["blend"] == truth
    comparison = {"alpha": .1, "delta_macro_recall": metrics["blend"]["macro_recall"] - metrics["stylegram"]["macro_recall"],
                  "fixed_errors": int(np.count_nonzero(~old_good & new_good)),
                  "new_errors": int(np.count_nonzero(old_good & ~new_good)),
                  "prediction_disagreements": int(np.count_nonzero(predictions["stylegram"] != predictions["blend"])),
                  "paired_prompt_bootstrap": boot}
    receipt = load_json(directory / "metrics.json")
    check(receipt["scope"] == "external 17-label imbalanced panel with 18-way predictions; not 18-class balanced accuracy",
          "Metric scope changed")
    for name, document in (("metrics.json", receipt), ("result.json", result)):
        match(document["metrics"], metrics, name + ".metrics")
        match(document["comparison"], comparison, name + ".comparison")
    confusion = pd.read_csv(directory / "confusion.csv", dtype=str, keep_default_na=False)
    expected_confusion = []
    for member, prediction in predictions.items():
        for actual in COVERED:
            for predicted in range(18):
                count = int(np.count_nonzero((truth == actual) & (prediction == predicted)))
                expected_confusion.append({"member": member, "true_model": LABELS[actual],
                                           "predicted_model": LABELS[predicted], "count": str(count)})
    check(confusion.columns.tolist() == ["member", "true_model", "predicted_model", "count"]
          and confusion.to_dict("records") == expected_confusion, "918-cell confusion matrix differs")
    changes = []
    for index in np.flatnonzero(predictions["stylegram"] != predictions["blend"]):
        changes.append({"id": evaluation.id.iloc[index], "prompt_id": evaluation.prompt_id.iloc[index],
                        "model": LABELS[truth[index]], "stylegram_prediction": LABELS[predictions["stylegram"][index]],
                        "blend_prediction": LABELS[predictions["blend"][index]],
                        "change": "fixed_error" if new_good[index] else "new_error" if old_good[index] else "both_wrong_changed"})
    for path, digest in {**pinned, **read_hashes}.items():
        check(sha(path) == digest, "Input changed during independent audit: " + str(path))
    return {"schema_version": 1, "status": "PASS", "experiment_id": "EXP-027", "declared_notebook_version": 2,
            "scope": "Independent score arithmetic, original row mapping, metrics, changes and prompt bootstrap audit only",
            "protocol_sha256": PROTOCOL_SHA, "evaluation_sha256": EVALUATION_SHA, "data_gate_sha256": GATE_SHA,
            "checker_sha256": sha(Path(__file__)), "input_directory": str(directory),
            "input_hashes": {str(path): digest for path, digest in {**pinned, **read_hashes}.items()},
            "evaluation_rows": 729, "evaluation_groups": 56, "true_classes": 17, "score_columns": 18,
            "class_counts": CLASS_COUNTS, "prediction_mask_applied": False, "alpha": .1,
            "normalization": "per-row population z-score across all 18 scores; constant rows map to zero",
            "maximum_absolute_blend_error": maximum_error, "independent_blend_argmax_exact_match": True,
            "recomputed_metrics": metrics, "recomputed_comparison": comparison, "changed_rows": changes,
            "confusion_cells_verified": len(expected_confusion), "bootstrap_implementation": "explicit sampled original row expansion",
            "bootstrap_invalid_draw_indices": invalid_indices, "bootstrap_all_draw_deltas": deltas,
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
            "training_reconstructed": False, "training_performed": False, "remote_calls_performed": False,
            "limits": ["This independent arithmetic audit does not reconstruct feature matrices, fitted SVMs, or training execution.",
                       "Remote V2 provenance and complete training/guard acceptance remain separately bound by the caller and primary verifier.",
                       "The interval is conditional on these fitted models and this source; it excludes retraining uncertainty.",
                       "These are 17-covered-class external metrics, not full 18-class, ATLAS20, or leaderboard evidence."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        check(args.input is None and not args.check_only, "Self-test accepts no real result arguments")
        print(json.dumps(self_test(), ensure_ascii=False))
        return
    check(args.input is not None, "--input is required for a real arithmetic audit")
    report = run(args.input)
    if not args.check_only:
        timestamp = load_json(OUTPUT)["verified_at_utc"] if OUTPUT.exists() else datetime.now(timezone.utc).isoformat()
        report["verified_at_utc"] = timestamp
        encoded = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
        if OUTPUT.exists():
            check(OUTPUT.is_file() and OUTPUT.read_bytes() == encoded, "Refusing to overwrite a different independent review")
        else:
            OUTPUT.parent.mkdir(parents=True, exist_ok=True)
            with OUTPUT.open("xb") as stream:
                stream.write(encoded)
    print(json.dumps({"status": report["status"], "output_written": not args.check_only,
                      "output": str(OUTPUT) if not args.check_only else None,
                      "comparison": report["recomputed_comparison"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
