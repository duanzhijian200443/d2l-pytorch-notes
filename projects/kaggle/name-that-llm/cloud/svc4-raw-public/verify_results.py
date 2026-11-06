"""Verify downloaded svc4 results, then write independently recomputed metrics.

Usage: python -B verify_results.py --input <downloaded-results-directory>
       [--protocol <retry-cloud-directory>/payload/protocol.json]
Use --check-only to perform every check/calculation without writing analysis.
This module never imports the runner, fits a model, downloads, or submits data.
Missing/incomplete/invalid real results fail before any analysis is published.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import platform
import re
import sys

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAYLOAD = HERE / "payload"
TRAIN_SHA = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
FOLDS_SHA = "e8ba1e3712852168dc57f56960d4f3e00be612d639ade8a7dcb0f0d8335ce551"
SPLIT_MANIFEST_SHA = "2b05e2bde656d5e773e8d512cba607859e9b9bfe7d9d7dfe9deb00907b4d93c8"
VERSIONS = {"numpy": "2.1.3", "pandas": "2.3.3", "scipy": "1.16.3", "sklearn": "1.6.1"}
MODES = ("official-only", "official-plus-raw")
DYNAMIC = {"artifacts.json", "guard.json", "state.json", "training.log", ".run.lock"}
SEMANTICS = "sklearn 1.6.1 balanced weights from unweighted training class counts"
# One microsecond covers floating-point/JSON roundoff only, not execution grace.
BUDGET_TOLERANCE_SECONDS = 1e-6
NATIVE_PARAMS = {"C": 0.5, "class_weight": "balanced", "dual": "auto", "fit_intercept": True,
                 "intercept_scaling": 1, "loss": "squared_hinge", "max_iter": 5000,
                 "multi_class": "ovr", "penalty": "l2", "random_state": 42, "tol": 0.0001, "verbose": 0}


class VerificationError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def invalid_constant(value):
    raise VerificationError(f"Nonfinite JSON constant: {value}")


def read_json(path):
    require(path.is_file(), f"Required file missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=strict_pairs, parse_constant=invalid_constant)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def finite_number(value, name, minimum=None):
    require(not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value), f"Invalid numeric {name}")
    if minimum is not None:
        require(value >= minimum, f"{name} below minimum")
    return value


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f"Invalid integer {name}")
    return value


def close(actual, expected, name, tolerance=1e-12):
    finite_number(actual, name)
    require(math.isclose(actual, float(expected), rel_tol=tolerance, abs_tol=tolerance), f"Recomputed value mismatch: {name}")


def check_array(actual, expected, name, exact=False):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape and actual.dtype.kind in "iuf" and np.isfinite(actual).all(), f"Invalid array: {name}")
    equal = np.array_equal(actual, expected) if exact else np.allclose(actual, expected, rtol=1e-12, atol=1e-10)
    require(equal, f"Recomputed array mismatch: {name}")


def valid_digest(value, name):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None, f"Invalid SHA-256: {name}")


def safe_relative(name):
    require(isinstance(name, str) and name and "\\" not in name and ":" not in name, "Unsafe artifact path")
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts and path.as_posix() == name and name != ".", "Unsafe artifact path")
    return path


def protocol_variants(protocol):
    entries = protocol.get("variants")
    require(isinstance(entries, list) and len(entries) == 4, "Expected four protocol variants")
    require(all(isinstance(row, dict) and isinstance(row.get("id"), str) and re.fullmatch(r"EXP-\d{3,}", row["id"]) for row in entries), "Invalid experiment identifier")
    weights = {row["id"]: row.get("public_row_weight") for row in entries}
    require(len(weights) == 4 and {canonical(value) for value in weights.values()} == {"null", "0.35", "0.175", "0.7"}, "Protocol weight/variant set changed")
    by_weight = {canonical(weight): exp for exp, weight in weights.items()}
    expected_pairs = [[by_weight["0.35"], by_weight["null"]],
                      [by_weight["0.175"], by_weight["0.35"]], [by_weight["0.7"], by_weight["0.35"]]]
    require(protocol.get("comparisons") == expected_pairs, "Predeclared comparison definitions differ")
    return weights


def expected_artifacts(variants):
    paths = {"environment.json", "input_preflight.json", "fold_metrics.csv", "result.json"}
    paths.update(f"feature-fold{fold}-{mode}.json" for fold in range(5) for mode in MODES)
    for exp in variants:
        paths.update({f"{exp}/oof.csv", f"{exp}/oof_decision.npy"})
        for fold in range(5):
            paths.update({f"{exp}/fold{fold}.npz", f"{exp}/metrics-fold{fold}.json", f"{exp}/weights-fold{fold}.json"})
    require(len(paths) == 82, "Internal expected artifact set is wrong")
    return paths


def verify_artifacts(directory, expected):
    """Generic manifest checker; production passes the exact 82-file set."""
    directory = directory.resolve(strict=True)
    manifest = read_json(directory / "artifacts.json")
    require(isinstance(manifest, dict) and set(manifest) == {"files"} and isinstance(manifest["files"], list), "Invalid artifact manifest")
    listed, casefolded, hashes = set(), set(), {}
    for entry in manifest["files"]:
        require(isinstance(entry, dict) and set(entry) == {"path", "bytes", "sha256"}, "Invalid artifact entry")
        name = entry["path"]
        safe_relative(name)
        require(name in expected and name not in listed and name.casefold() not in casefolded, f"Unexpected/duplicate artifact: {name}")
        path = directory / name
        require(not path.is_symlink() and path.resolve(strict=True).is_relative_to(directory) and path.is_file(), f"Unsafe/nonfile artifact: {name}")
        size = integer(entry["bytes"], name + " bytes")
        valid_digest(entry["sha256"], name)
        require(path.stat().st_size == size and digest_file(path) == entry["sha256"], f"Artifact bytes changed: {name}")
        listed.add(name); casefolded.add(name.casefold()); hashes[name] = entry["sha256"]
    require(listed == expected, f"Incomplete stable artifact set: missing={sorted(expected-listed)}")
    actual = set()
    for path in directory.rglob("*"):
        require(not path.is_symlink(), f"Symlink in result tree: {path}")
        if path.is_file():
            name = path.relative_to(directory).as_posix()
            require(name in expected or name in DYNAMIC, f"Unexpected/temporary/failure file in results: {name}")
            actual.add(name)
    require(actual - DYNAMIC == expected, "Actual stable files differ from complete manifest")
    for name in sorted(actual & DYNAMIC):
        hashes[name] = digest_file(directory / name)
    return hashes


def verify_terminal(result, guard, state, protocol):
    variants = protocol_variants(protocol)
    require(result.get("status") == guard.get("status") == state.get("status") == "COMPLETE", "Run/guard/state are not all COMPLETE")
    require(type(result.get("trained_svms")) is int and result["trained_svms"] == 20, "Expected 20 fitted SVMs")
    require(type(result.get("convergence_warnings")) is int and result["convergence_warnings"] == 0, "Convergence warning count must be zero")
    require(type(guard.get("child_exit")) is int and guard["child_exit"] == 0, "Guard child did not exit 0")
    require("termination_reason" in guard and guard["termination_reason"] is None, "Guard has a termination reason")
    require(guard.get("artifacts_verified") is True and not guard.get("traceback"), "Guard did not validate clean output")
    require(result.get("gpu_used") is False and result.get("fitted_weights_saved") is False, "Unexpected execution scope")
    require(result.get("scope") == protocol["scope"], "Result scope differs from protocol")
    close(guard.get("wall_seconds"), protocol["work_budget_seconds"], "guard work budget")
    require(guard.get("rss_limit_bytes") == protocol["rss_limit_bytes"], "Guard RSS budget differs")
    peak = integer(guard.get("peak_rss_bytes"), "guard peak RSS")
    require(peak <= protocol["rss_limit_bytes"], "Recorded guard RSS exceeded budget")
    total = finite_number(result.get("total_seconds"), "result total seconds", 0)
    elapsed = finite_number(guard.get("elapsed_seconds"), "guard elapsed", 0)
    work_budget = finite_number(protocol.get("work_budget_seconds"), "protocol work budget", 1)
    platform_budget = finite_number(protocol.get("platform_wall_seconds"), "protocol platform budget", 1)
    require(work_budget <= platform_budget, "Protocol work budget exceeds platform budget")
    require(total <= work_budget + BUDGET_TOLERANCE_SECONDS, "Runner total exceeded protocol work budget")
    require(elapsed <= platform_budget + BUDGET_TOLERANCE_SECONDS, "Guard elapsed exceeded protocol platform budget")
    require(total <= elapsed + BUDGET_TOLERANCE_SECONDS, "Runner elapsed exceeds guard elapsed")
    complete = state.get("completed")
    require(isinstance(complete, list) and len(complete) == 20, "Incomplete state completion list")
    require(all(isinstance(x, list) and len(x) == 2 and x[0] in variants and type(x[1]) is int for x in complete), "Invalid completed pair")
    require({tuple(x) for x in complete} == {(exp, fold) for exp in variants for fold in range(5)}, "Repeated/missing completion pair")
    timings = result.get("fold_timings")
    require(isinstance(timings, list) and len(timings) == 5 and {x.get("fold") for x in timings} == set(range(5)), "Missing fold timing records")
    for timing in timings:
        finite_number(timing.get("wall_seconds"), "fold wall time", 0)


def local_inputs(protocol_path=None):
    protocol_path = (protocol_path or PAYLOAD / "protocol.json").resolve()
    payload = protocol_path.parent
    protocol = read_json(protocol_path)
    require(protocol.get("required_versions") == VERSIONS, "Local required versions changed")
    weights = protocol_variants(protocol)
    variants = protocol.get("variants")
    require(protocol.get("train_sha256") == TRAIN_SHA and protocol.get("split_id") == "prompt-group-v1", "Official validation target changed")
    require(protocol.get("public_rows") == 5400 and protocol.get("public_prompts") == 300, "Public population changed")
    require(protocol.get("gpu") is False and protocol.get("internet") is False and protocol.get("threads") == 2, "Protocol execution scope changed")
    require(finite_number(protocol.get("work_budget_seconds"), "protocol work seconds", 1) <=
            finite_number(protocol.get("platform_wall_seconds"), "protocol platform seconds", 1), "Work budget exceeds platform budget")
    integer(protocol.get("rss_limit_bytes"), "protocol RSS limit", 1)
    expected_inputs = {"svc4.py", "runner.py", "guard.py", "folds.csv", "public.csv", "input_audit.json", "attribution.md"} | {f"config-{exp}.json" for exp in weights}
    require(set(protocol["input_hashes"]) == expected_inputs, "Protocol input set differs from frozen payload")
    frozen = {protocol_path: digest_file(protocol_path)}
    for name, value in protocol["input_hashes"].items():
        safe_relative(name); valid_digest(value, name)
        require(digest_file(payload / name) == value, f"Local frozen payload changed: {name}")
        frozen[payload / name] = value
    require(protocol["input_hashes"]["folds.csv"] == FOLDS_SHA, "Payload fold hash differs from fixed split")
    for path, expected in ((ROOT / "train.csv", TRAIN_SHA), (ROOT / "validation/prompt-group-v1/folds.csv", FOLDS_SHA)):
        require(digest_file(path) == expected, f"Local official file changed: {path}")
        frozen[path] = expected
    split_path = ROOT / "validation/prompt-group-v1/manifest.json"
    require(digest_file(split_path) == SPLIT_MANIFEST_SHA, "Fixed split manifest changed")
    frozen[split_path] = SPLIT_MANIFEST_SHA
    train, folds, public = read_csv(ROOT / "train.csv"), read_csv(payload / "folds.csv"), read_csv(payload / "public.csv")
    labels = protocol["labels"]
    require(labels == read_json(split_path)["labels"], "Protocol label order differs from fixed split manifest")
    require(len(labels) == len(set(labels)) == 18 and set(train.model) == set(public.model) == set(labels), "Label domain/order invalid")
    require(len(train) == len(folds) == 2699 and len(public) == 5400, "Input row counts invalid")
    require(train.id.is_unique and folds.id.is_unique, "Repeated official ID")
    for column in ("id", "prompt_id", "model"):
        require(train[column].tolist() == folds[column].tolist(), f"Official/fold {column} order mismatch")
    require(train.text.map(lambda x: digest_bytes(x.encode())).tolist() == folds.text_sha256.tolist(), "Official fold text hash mismatch")
    folds["fold"] = folds.fold.astype(int)
    require(set(folds.fold) == set(range(5)) and folds.prompt_id.nunique() == 150 and folds.groupby("prompt_id").fold.nunique().eq(1).all(), "Prompt split invalid")
    require(public.groupby("prompt_id").size().eq(18).all() and public.prompt_id.nunique() == 300 and public.model.value_counts().eq(300).all(), "Public group coverage invalid")
    for column in ("row_id", "id", "call_id"):
        require(public[column].is_unique, "Repeated public " + column)
    require(public.text.map(lambda x: digest_bytes(x.encode())).tolist() == public.text_sha256.tolist(), "Public text SHA mismatch")
    for variant in variants:
        directory = safe_relative(variant["directory"])
        require(len(directory.parts) == 2 and directory.parts[0] == "experiments" and directory.name.startswith(variant["id"] + "-"), "Experiment record directory binding differs")
        name = variant["config_file"]
        require(name == f"config-{variant['id']}.json" and variant["config_sha256"] == protocol["input_hashes"][name], "Config binding mismatch")
        config = read_json(payload / name)
        require(config.get("experiment_id") == variant["id"], "Config experiment identifier mismatch")
        expected_parameters = {"C": .5, "random_state": 42, "max_iter": 5000, "class_weight": "balanced", "dual": "auto", "feature_dtype": "float32", "public_row_weight": variant["public_row_weight"]}
        require(canonical(config["model"]["parameters"]) == canonical(expected_parameters), "Frozen classifier config mismatch")
    return protocol, train, folds, public, frozen


def verify_feature(info, fold, mode, train, folds):
    fit = folds.fold.to_numpy() != fold
    valid = ~fit
    added = mode == "official-plus-raw"
    require(info.get("fold") == fold and info.get("mode") == mode, "Feature identity mismatch")
    require(info.get("training_official_ids") == train.id.loc[fit].tolist(), "Feature official training IDs include wrong/held-out rows")
    require(info.get("validation_ids") == train.id.loc[valid].tolist(), "Feature validation IDs differ")
    require(info.get("public_rows_in_feature_fit") == (5400 if added else 0), "Feature public membership differs")
    require(info.get("training_rows") == int(fit.sum()) + (5400 if added else 0) and info.get("validation_rows") == int(valid.sum()), "Feature row counts differ")
    require(info.get("dtype") == "float32", "Feature dtype is not float32")
    for name in ("train_matrix_sha256", "valid_matrix_sha256"):
        valid_digest(info.get(name), name)
    views = info.get("views")
    require(isinstance(views, list) and len(views) == 4, "Missing feature view receipts")
    total = 0
    for view, (name, scale, cap) in zip(views, (("char", 1., 250000), ("word", .7, 180000), ("shape", .5, 60000), ("function", .5, 50000))):
        require(view.get("view") == name and view.get("scale") == scale, "Feature view configuration mismatch")
        count = integer(view.get("columns"), "view columns", 1)
        require(count <= cap, "Feature column cap exceeded")
        total += count
        valid_digest(view.get("vocabulary_sha256"), "vocabulary"); valid_digest(view.get("idf_sha256"), "IDF")
    require(info.get("features") == total, "Feature column totals disagree")
    nnz = integer(info.get("nnz"), "feature nnz", 1)
    require(nnz <= info["training_rows"] * total, "Impossible feature nnz")
    for name in ("csr_bytes", "validation_csr_bytes", "liblinear_float64_csr_estimate"):
        integer(info.get(name), name, 1)
    finite_number(info.get("feature_seconds"), "feature seconds", 0)


def verify_weights(info, fold, weight, official_counts, labels, public_per_class=300):
    require(info.get("fold") == fold and canonical(info.get("public_row_weight")) == canonical(weight), "Weight identity mismatch")
    require(info.get("labels") == labels and info.get("sample_weight_dtype") == "float32" and info.get("semantics") == SEMANTICS, "Weight metadata mismatch")
    counts = np.asarray(official_counts, dtype=np.int64) + (public_per_class if weight is not None else 0)
    cw = counts.sum() / (len(labels) * counts)
    public_mass = np.full(len(labels), float(np.float32(weight)) * public_per_class if weight is not None else 0.)
    official_mass = np.asarray(official_counts, dtype=float)
    check_array(info.get("class_counts"), counts, "class_counts", exact=True)
    check_array(info.get("class_weights"), cw, "class_weights")
    check_array(info.get("official_sample_mass"), official_mass, "official mass", exact=True)
    check_array(info.get("public_sample_mass"), public_mass, "float32 public mass")
    check_array(info.get("effective_class_mass"), (official_mass + public_mass) * cw, "effective class mass")


def balanced_accuracy(y, pred, n_classes):
    counts = np.bincount(y, minlength=n_classes)
    require(len(counts) == n_classes and (counts > 0).all(), "Balanced accuracy requires all declared classes")
    correct = np.bincount(y[y == pred], minlength=n_classes)
    return float(np.mean(correct / counts))


def verify_fold_npz(path, ids, labels, true, expected_scores, feature):
    required = {"ids", "labels", "true", "scores", "train_matrix_sha256", "valid_matrix_sha256"}
    with np.load(path, allow_pickle=False) as data:
        require(set(data.files) == required and len(data.files) == len(required), "Fold NPZ has missing/extra/repeated arrays")
        require(data["ids"].ndim == 1 and data["ids"].dtype.kind in "US" and data["ids"].tolist() == ids, "Fold NPZ IDs/order mismatch")
        require(data["labels"].ndim == 1 and data["labels"].dtype.kind in "US" and data["labels"].tolist() == labels, "Fold NPZ label order mismatch")
        check_array(data["true"], true, "fold true labels", exact=True)
        scores = data["scores"]
        require(scores.dtype.kind == "f" and scores.shape == (len(ids), len(labels)) and np.isfinite(scores).all(), "Fold scores have invalid shape/dtype/value")
        require(np.array_equal(scores, expected_scores), "Fold scores do not exactly match combined OOF scores")
        for name in ("train_matrix_sha256", "valid_matrix_sha256"):
            require(data[name].shape == () and data[name].dtype.kind in "US" and data[name].item() == feature[name], "Fold matrix fingerprint mismatch")


def verify_scores(directory, protocol, train, folds):
    variants = protocol_variants(protocol)
    labels = protocol["labels"]; label_index = {label: i for i, label in enumerate(labels)}
    y = train.model.map(label_index).to_numpy(dtype=int)
    features = {}
    for fold in range(5):
        for mode in MODES:
            info = read_json(directory / f"feature-fold{fold}-{mode}.json")
            verify_feature(info, fold, mode, train, folds)
            features[fold, mode] = info
    all_scores, metrics, per_class, summary = {}, [], [], []
    for exp, weight in variants.items():
        oof = np.load(directory / exp / "oof_decision.npy", allow_pickle=False)
        require(isinstance(oof, np.ndarray) and oof.dtype.kind == "f" and oof.shape == (2699, 18) and np.isfinite(oof).all(), f"Invalid OOF scores for {exp}")
        frame = read_csv(directory / exp / "oof.csv")
        require(frame.columns.tolist() == ["id", "prompt_id", "model", "fold", "predicted_model", "correct"] and len(frame) == 2699 and frame.id.is_unique, "OOF CSV schema/rows invalid")
        for column in ("id", "prompt_id", "model"):
            require(frame[column].tolist() == train[column].tolist(), "OOF CSV membership/order mismatch: " + column)
        require(frame.fold.tolist() == folds.fold.astype(str).tolist(), "OOF CSV fold mismatch")
        prediction = oof.argmax(axis=1)
        expected_labels = np.asarray(labels)[prediction]
        require(frame.predicted_model.tolist() == expected_labels.tolist(), "OOF argmax/label mismatch")
        require(frame.correct.tolist() == [str(bool(x)) for x in prediction == y], "OOF correct flags mismatch")
        for fold in range(5):
            valid = folds.fold.to_numpy() == fold; fit = ~valid
            feature = features[fold, "official-only" if weight is None else "official-plus-raw"]
            verify_fold_npz(directory / exp / f"fold{fold}.npz", train.id.loc[valid].tolist(), labels, y[valid], oof[valid], feature)
            weights = read_json(directory / exp / f"weights-fold{fold}.json")
            verify_weights(weights, fold, weight, np.bincount(y[fit], minlength=18), labels)
            record = read_json(directory / exp / f"metrics-fold{fold}.json")
            require(record.get("experiment") == exp and record.get("fold") == fold and canonical(record.get("public_row_weight")) == canonical(weight), "Fold metric identity mismatch")
            require(canonical(record.get("native_estimator_parameters")) == canonical(NATIVE_PARAMS), "Native classifier parameters differ")
            for name, value in (("rows", int(valid.sum())), ("training_rows", feature["training_rows"]), ("features", feature["features"]), ("train_matrix_sha256", feature["train_matrix_sha256"]), ("valid_matrix_sha256", feature["valid_matrix_sha256"])):
                require(record.get(name) == value, "Fold metrics/feature mismatch: " + name)
            iterations = integer(record.get("n_iter"), "n_iter", 1)
            require(iterations < 5000, "SVM reached iteration cap")
            for name in ("feature_seconds", "fit_seconds", "prediction_seconds"):
                finite_number(record.get(name), name, 0)
            close(record["feature_seconds"], feature["feature_seconds"], "shared feature timing")
            integer(record.get("rss_bytes"), "fit RSS", 1)
            ba = balanced_accuracy(y[valid], prediction[valid], 18)
            close(record.get("ba"), ba, "fold BA")
            metrics.append({**{k: v for k, v in record.items() if k != "native_estimator_parameters"}, "recomputed_ba": ba})
        ba = balanced_accuracy(y, prediction, 18)
        summary.append({"experiment": exp, "public_row_weight": weight, "balanced_accuracy": ba,
                        "accuracy": float(np.mean(y == prediction)), "correct": int(np.sum(y == prediction)), "rows": 2699})
        for index, label in enumerate(labels):
            mask = y == index; correct = int(np.sum(prediction[mask] == index))
            per_class.append({"experiment": exp, "model": label, "rows": int(mask.sum()), "correct": correct, "recall": correct / int(mask.sum())})
        all_scores[exp] = oof
    require(len(metrics) == 20, "Not all 20 fold metrics reconciled")
    combined = read_csv(directory / "fold_metrics.csv")
    require(len(combined) == 20 and not combined.duplicated(["experiment", "fold"]).any(), "Combined fold_metrics rows duplicated/missing")
    expected_columns = set(metrics[0]) - {"recomputed_ba"}
    require(set(combined.columns) == expected_columns, "Combined fold_metrics columns differ")
    table = combined.set_index(["experiment", "fold"])
    for record in metrics:
        key = (record["experiment"], str(record["fold"]))
        require(key in table.index, "Combined metrics are missing an experiment/fold")
        row = table.loc[key]
        for name in expected_columns - {"experiment", "fold"}:
            value = record[name]
            if value is None:
                require(row[name] == "", "Official-only public weight is not blank")
            elif isinstance(value, str):
                require(row[name] == value, "Combined metric string differs: " + name)
            else:
                close(float(row[name]), value, "combined " + name)
    return y, all_scores, summary, metrics, per_class


def compare_predictions(y, scores, train, folds, labels, pairs, variant_weights, draws=2000, seed=42):
    groups = sorted(train.prompt_id.unique())
    group_index = {value: index for index, value in enumerate(groups)}
    group_of = train.prompt_id.map(group_index).to_numpy()
    denominator = np.zeros((len(groups), len(labels)), dtype=np.int64)
    np.add.at(denominator, (group_of, y), 1)
    rng = np.random.default_rng(seed)
    resample = rng.integers(0, len(groups), size=(draws, len(groups)))
    resampled_counts = denominator[resample].sum(axis=1)
    require((resampled_counts > 0).all(), "Bootstrap draw has an absent class")
    correct_by_prompt, predictions = {}, {}
    for exp, values in scores.items():
        pred = values.argmax(axis=1); predictions[exp] = pred
        correct = np.zeros_like(denominator)
        np.add.at(correct, (group_of[pred == y], y[pred == y]), 1)
        correct_by_prompt[exp] = correct
    comparisons, by_fold, error_changes = [], [], []
    for candidate, reference in pairs:
        cand, ref = predictions[candidate], predictions[reference]
        difference = correct_by_prompt[candidate] - correct_by_prompt[reference]
        bootstrap = (difference[resample].sum(axis=1) / resampled_counts).mean(axis=1)
        lower, upper = np.quantile(bootstrap, [.025, .975], method="linear")
        fold_deltas = []
        for fold in range(5):
            mask = folds.fold.to_numpy() == fold
            delta = balanced_accuracy(y[mask], cand[mask], len(labels)) - balanced_accuracy(y[mask], ref[mask], len(labels))
            fold_deltas.append(delta)
            by_fold.append({"candidate": candidate, "reference": reference, "fold": fold, "delta_ba": delta,
                            "fixed_errors": int(np.sum((ref[mask] != y[mask]) & (cand[mask] == y[mask]))),
                            "new_errors": int(np.sum((ref[mask] == y[mask]) & (cand[mask] != y[mask])))})
        comparisons.append({"candidate": candidate, "reference": reference,
            "contrast": "adding_raw_public_including_feature_fit" if variant_weights[reference] is None else "public_row_weight_only",
            "delta_ba": balanced_accuracy(y, cand, len(labels)) - balanced_accuracy(y, ref, len(labels)),
            "ci95_lower": float(lower), "ci95_upper": float(upper), "bootstrap_draws": draws, "bootstrap_seed": seed,
            "fixed_errors": int(np.sum((ref != y) & (cand == y))), "new_errors": int(np.sum((ref == y) & (cand != y))),
            "prediction_disagreements": int(np.sum(cand != ref)), "both_wrong": int(np.sum((cand != y) & (ref != y))),
            "positive_folds": sum(x > 1e-12 for x in fold_deltas), "negative_folds": sum(x < -1e-12 for x in fold_deltas),
            "tied_folds": sum(abs(x) <= 1e-12 for x in fold_deltas), "multiple_comparison_adjustment": "none"})
        for index in np.flatnonzero(cand != ref):
            error_changes.append({"candidate": candidate, "reference": reference, "id": train.id.iloc[index],
                "prompt_id": train.prompt_id.iloc[index], "fold": int(folds.fold.iloc[index]), "true_model": labels[y[index]],
                "reference_prediction": labels[ref[index]], "candidate_prediction": labels[cand[index]],
                "change": "fixed_error" if cand[index] == y[index] else "new_error" if ref[index] == y[index] else "both_wrong_changed"})
    return comparisons, by_fold, error_changes


def sparse_selections(rows, seed=42, sizes=(2, 3, 6), repeats=5):
    """Match validation.protocol.select_sparse_rows and map IDs back to OOF positions.

    Ranking depends only on seed, repeat and sample ID. Neither the input row
    order nor the order of requested sizes changes a scenario's selected IDs.
    The returned positions follow canonical ascending ID order, not input order.
    """
    require({"id", "prompt_id"}.issubset(rows.columns), "Sparse selection requires ID/prompt metadata")
    require(rows.id.is_unique and rows.id.map(lambda value: isinstance(value, str) and bool(value)).all(), "Sparse IDs must be unique nonempty strings")
    require(rows.prompt_id.map(lambda value: isinstance(value, str) and bool(value)).all(), "Sparse prompt IDs must be nonempty strings")
    require(type(repeats) is int and repeats > 0, "Sparse repeat count must be positive")
    positions = {sample_id: position for position, sample_id in enumerate(rows.id)}
    group_counts = rows.groupby("prompt_id").size()
    output = []
    for size in sizes:
        require(type(size) is int and size > 0 and group_counts.ge(size).all(), "Sparse group is too small or requested size is invalid")
        for repeat in range(repeats):
            ordered = rows[["id", "prompt_id"]].copy()
            ordered["_selection_key"] = ordered.id.map(lambda sample: digest_bytes(f"{seed}:{repeat}:{sample}".encode("utf-8")))
            selected = ordered.sort_values(["prompt_id", "_selection_key"]).groupby("prompt_id", sort=True).head(size)
            selected = selected.sort_values("id")
            chosen = np.asarray([positions[sample_id] for sample_id in selected.id], dtype=np.int64)
            require(len(np.unique(chosen)) == len(chosen), "Repeated row in sparse selection")
            output.append((size, repeat, chosen))
    return output


def sparse_analysis(y, scores, train, folds, pairs):
    metrics, selection_rows, differences = [], [], []
    for size, repeat, chosen in sparse_selections(folds[["id", "prompt_id"]]):
        require(len(chosen) == 150 * size and set(folds.fold.iloc[chosen]) == set(range(5)), "Sparse scenario does not combine all five folds")
        require(train.prompt_id.iloc[chosen].value_counts().eq(size).all(), "Sparse per-prompt count differs")
        ids = train.id.iloc[chosen].tolist()
        selection_sha = digest_bytes(canonical(ids).encode())
        scenario = f"m{size}-repeat{repeat}"
        bas = {}
        for exp, values in scores.items():
            prediction = values[chosen].argmax(axis=1)
            ba = balanced_accuracy(y[chosen], prediction, 18)
            bas[exp] = ba
            metrics.append({"experiment": exp, "scenario": scenario, "m": size, "repeat": repeat, "seed": 42,
                            "rows": len(chosen), "prompts": 150, "folds_combined": 5, "balanced_accuracy": ba,
                            "correct": int(np.sum(y[chosen] == prediction)), "selected_ids_sha256": selection_sha})
        for candidate, reference in pairs:
            differences.append({"candidate": candidate, "reference": reference, "scenario": scenario,
                                "m": size, "repeat": repeat, "delta_ba": bas[candidate] - bas[reference], "selected_ids_sha256": selection_sha})
        for index in chosen:
            selection_rows.append({"scenario": scenario, "m": size, "repeat": repeat, "id": train.id.iloc[index],
                                   "prompt_id": train.prompt_id.iloc[index], "fold": int(folds.fold.iloc[index])})
    require(len(metrics) == 60 and len(differences) == 45, "Sparse result coverage is incomplete")
    summary = []
    for (exp, size), group in pd.DataFrame(metrics).groupby(["experiment", "m"], sort=True):
        require(len(group) == 5, "Sparse repeat count differs")
        summary.append({"experiment": exp, "m": int(size), "scenario_repeats": 5,
                        "mean_ba": float(group.balanced_accuracy.mean()), "scenario_sd_ddof1": float(group.balanced_accuracy.std(ddof=1)),
                        "min_ba": float(group.balanced_accuracy.min()), "max_ba": float(group.balanced_accuracy.max()),
                        "interpretation": "descriptive sampling variation using the same fitted models/prompts; not independent training repeats or standard error"})
    return metrics, summary, selection_rows, differences


def report_text(summary, comparisons, protocol_hash):
    lines = ["# SVC4 raw-public development verification", "", "All required downloaded artifacts passed structural, hash and numerical reconciliation.", "",
             "| Experiment | Public row weight | OOF balanced accuracy | Correct / 2699 |",
             "|---|---:|---:|---:|"]
    for row in summary:
        weight = "official only" if row["public_row_weight"] is None else str(row["public_row_weight"])
        lines.append(f"| {row['experiment']} | {weight} | {row['balanced_accuracy']:.9f} | {row['correct']} |")
    lines += ["", "| Candidate minus reference | BA difference | Unadjusted paired prompt bootstrap 95% interval | Fixed / new errors | Fold direction + / - / tie |",
              "|---|---:|---:|---:|---:|"]
    for row in comparisons:
        lines.append(f"| {row['candidate']} - {row['reference']} | {row['delta_ba']:+.9f} | [{row['ci95_lower']:+.9f}, {row['ci95_upper']:+.9f}] | {row['fixed_errors']} / {row['new_errors']} | {row['positive_folds']} / {row['negative_folds']} / {row['tied_folds']} |")
    lines += ["", "The augmented-versus-official-only contrast measures adding raw public data, including its effect on vocabulary and IDF. The two other predeclared contrasts change only public row weight on shared fold matrices.", "",
              "Intervals use 2,000 paired resamples of the 150 official prompt groups, seed 42. They are conditional on these fitted development models, without multiplicity adjustment or retraining uncertainty. They do not establish a leaderboard improvement or an automatic winner.", "",
              "Sparse metrics use the canonical SHA-256 key seed:repeat:id, rank within each prompt, and take the first 2, 3, or 6 IDs, five repeats each. They combine all five folds before computing 18-class balanced accuracy: 60 variant/scenario results. Input order and size enumeration do not change the chosen IDs. Standard deviations describe reused-model sampling variation, not independent training repeats or standard errors.", "",
              "This is an isolated svc4 experiment, not full ATLAS20 or the original matched+pool recipe. Of 5,400 raw/new texts, 5,376 already occur in the audited baseline projection. Public generation lengths and protocol decisions used official training statistics; this is not an untouched validation source. Supplied provenance logs do not independently authenticate provider weights.", "",
              "Feature training ID membership and matrix/vocabulary/IDF fingerprint consistency were checked. Matrices, vocabularies, IDF vectors and fitted estimators were not saved, so their contents or the historical training process cannot be independently reconstructed from these receipts alone. Remote run/version binding remains the downloader's evidence responsibility.", "",
              f"Frozen local protocol SHA-256: `{protocol_hash}`.", "",
              "See verified.json and the adjacent CSV files for exact comparisons, class recalls, error changes and sparse selections.", ""]
    return "\n".join(lines).encode("utf-8")


def write_analysis(verified, tables, report, analysis_directory):
    timestamp = datetime.now(timezone.utc).isoformat()
    existing = analysis_directory / "verified.json"
    if existing.exists():
        timestamp = read_json(existing).get("verified_at_utc")
        require(isinstance(timestamp, str), "Existing verification timestamp is invalid")
        require(datetime.fromisoformat(timestamp).tzinfo is not None, "Existing verification time lacks timezone")
    outputs = {}
    for name, rows in tables.items():
        frame = pd.DataFrame(rows)
        if name == "error_changes.csv" and not rows:
            frame = pd.DataFrame(columns=["candidate", "reference", "id", "prompt_id", "fold", "true_model",
                                          "reference_prediction", "candidate_prediction", "change"])
        outputs[analysis_directory / name] = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    outputs[analysis_directory / "report.md"] = report
    verified = {**verified, "verified_at_utc": timestamp,
                "analysis_outputs": [{"path": path.name, "bytes": len(data), "sha256": digest_bytes(data)} for path, data in outputs.items()]}
    outputs[existing] = json_bytes(verified)
    for path, data in outputs.items():
        require(not path.exists() or (path.is_file() and path.read_bytes() == data), f"Refusing different existing analysis output: {path}")
    analysis_directory.mkdir(parents=True, exist_ok=True)
    for path, data in outputs.items():
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(data)
        require(path.read_bytes() == data, f"Analysis output changed: {path}")


def verify(directory, write=True, protocol_path=None):
    require(directory.is_dir(), "--input must be the downloaded results directory")
    require(not (directory / "failure.json").exists(), "Runner/guard failure.json is present")
    protocol_path = (protocol_path or PAYLOAD / "protocol.json").resolve()
    require(protocol_path.parent.name == "payload" and protocol_path.is_relative_to((ROOT / "cloud").resolve()), "Protocol must identify a local cloud/payload snapshot")
    analysis_directory = protocol_path.parent.parent / "analysis"
    protocol, train, folds, public, local_hashes = local_inputs(protocol_path)
    variants = protocol_variants(protocol)
    artifact_hashes = verify_artifacts(directory, expected_artifacts(variants))
    result, guard, state = (read_json(directory / name) for name in ("result.json", "guard.json", "state.json"))
    verify_terminal(result, guard, state, protocol)
    environment = read_json(directory / "environment.json")
    require(canonical(environment.get("protocol")) == canonical(protocol), "Result protocol differs from the frozen local protocol")
    require(environment.get("versions") == VERSIONS and environment.get("gpu_requested") is False, "Runtime environment differs")
    require(environment.get("score_type") == "uncalibrated LinearSVC decision_function" and environment.get("class_weight_semantics") == SEMANTICS and environment.get("fitted_weights_saved") is False, "Runtime scoring/weight semantics differ")
    preflight = read_json(directory / "input_preflight.json")
    require(preflight.get("status") == "PASS" and preflight.get("hashes_verified") is True and
            [preflight.get(x) for x in ("official_rows", "public_rows", "public_groups", "folds")] == [2699, 5400, 300, 5], "Runner preflight receipt invalid")
    y, scores, summary, fold_metrics, recalls = verify_scores(directory, protocol, train, folds)
    reported = result.get("summaries")
    require(isinstance(reported, list) and len(reported) == 4 and len({row.get("experiment") for row in reported}) == 4, "Result summary membership invalid")
    reported = {row["experiment"]: row for row in reported}
    require(set(reported) == set(variants), "Result summary variant set differs")
    for row in summary:
        saved = reported[row["experiment"]]
        require(canonical(saved.get("public_row_weight")) == canonical(row["public_row_weight"]) and saved.get("correct") == row["correct"], "Result summary counts/weight mismatch")
        close(saved.get("balanced_accuracy"), row["balanced_accuracy"], "overall BA")
    pairs, pair_folds, changes = compare_predictions(y, scores, train, folds, protocol["labels"], protocol["comparisons"], variants)
    sparse, sparse_summary, selections, sparse_pairs = sparse_analysis(y, scores, train, folds, protocol["comparisons"])
    # Refuse evidence that changed while calculations were being performed.
    for path, expected in local_hashes.items():
        require(digest_file(path) == expected, f"Local frozen input changed during verification: {path}")
    require(verify_artifacts(directory, expected_artifacts(variants)) == artifact_hashes, "Result bytes changed during verification")
    verified = {"schema_version": 1, "status": "PASS", "scope": protocol["scope"],
        "input_directory": str(directory.resolve()), "protocol_sha256": local_hashes[protocol_path],
        "protocol_path": protocol_path.relative_to(ROOT).as_posix(), "experiment_directories": {row["id"]: row["directory"] for row in protocol["variants"]},
        "verifier_sha256": digest_file(Path(__file__)), "official_train_sha256": TRAIN_SHA, "folds_sha256": FOLDS_SHA,
        "runtime_versions": environment["versions"], "stable_artifacts_verified": 82, "input_result_hashes": artifact_hashes,
        "analysis_environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "trained_svms": 20, "convergence_warnings": 0, "oof_rows_per_variant": 2699, "score_columns": 18,
        "budget_reconciliation": {"runner_total_seconds": result["total_seconds"], "work_budget_seconds": protocol["work_budget_seconds"],
                                  "guard_elapsed_seconds": guard["elapsed_seconds"], "platform_wall_seconds": protocol["platform_wall_seconds"],
                                  "comparison_tolerance_seconds": BUDGET_TOLERANCE_SECONDS,
                                  "tolerance_reason": "one microsecond for floating-point/JSON rounding only; no operational grace period"},
        "feature_receipts_checked": 10, "fold_npz_checked": 20, "fold_metrics_checked": 20, "weight_receipts_checked": 20,
        "official_only_feature_fit_is_official_training_fold_only": True, "augmented_variants_share_fold_matrix_fingerprints": True,
        "class_weight_semantics": SEMANTICS, "public_mass_formula": "float(np.float32(public_row_weight)) * 300 per class",
        "summaries": summary, "comparisons": pairs, "bootstrap": {"draws": 2000, "seed": 42, "unit": "official prompt", "paired": True, "percentile": [2.5, 97.5], "multiple_comparison_adjustment": "none", "conditional_on_fitted_development_models": True},
        "sparse": {"sizes": [2, 3, 6], "repeats": 5, "seed": 42, "variant_scenarios": 60,
                   "selection_rule": "validation/protocol.py::select_sparse_rows: SHA-256 of seed:repeat:id, sort by prompt/key, head(m), then sort by ID",
                   "aggregate_folds_before_18_class_ba": True, "repeats_are_independent_training_runs": False},
        "limitations": protocol["limits"] + ["Feature matrices/vocabularies/IDF vectors and fitted estimators were not persisted; receipt consistency and membership are verified, not independent reconstruction of their contents.", "Downloaded run/version identity is not independently queried by this offline verifier."],
        "training_or_remote_actions_performed": False}
    tables = {"summary_metrics.csv": summary, "fold_metrics_recomputed.csv": fold_metrics, "per_class_recall.csv": recalls,
              "comparisons.csv": pairs, "comparison_folds.csv": pair_folds, "error_changes.csv": changes,
              "sparse_metrics.csv": sparse, "sparse_summary.csv": sparse_summary, "sparse_selections.csv": selections,
              "sparse_comparisons.csv": sparse_pairs}
    if write:
        write_analysis(verified, tables, report_text(summary, pairs, verified["protocol_sha256"]), analysis_directory)
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PAYLOAD / "protocol.json", help="Frozen payload protocol; experiment IDs, comparisons, and analysis destination follow this snapshot")
    parser.add_argument("--check-only", action="store_true", help="Run full verification/calculations without writing analysis")
    arguments = parser.parse_args()
    try:
        result = verify(arguments.input, write=not arguments.check_only, protocol_path=arguments.protocol)
    except (VerificationError, OSError, ValueError, KeyError, TypeError, IndexError) as error:
        print("VERIFICATION_FAILED: " + str(error), file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps({"status": result["status"], "stable_artifacts_verified": 82, "trained_svms": 20,
                      "analysis_written": not arguments.check_only, "summaries": result["summaries"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
