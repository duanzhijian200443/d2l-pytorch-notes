"""Offline verification of the fixed EXP-027 result payload.

Usage: python -B verify_outputs.py --input <downloaded-results> [--check-only]
No runner/model module is imported and no training, network or alpha search is
performed. Only fully reconciled COMPLETE results may create analysis outputs.
"""
from collections import Counter
from datetime import datetime, timezone
import argparse
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
EXP = ROOT / "experiments/EXP-027-multiclass-external-validation"
ANALYSIS = HERE / "analysis"
PROTOCOL_SHA = "c3e2b1aab6744a41dc546281f0eb00d61f0d5e10564415444b4bb4e1267ce886"
TRAIN_SHA = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
EVALUATION_SHA = "8ec3217f1f469b66170f68f7abdb3ef1973a64e47d983c59b40a4d1ce2094670"
GATE_SHA = "cead2f4745a3e4af6a8adffe2361efd725e2710971a12123083b927589f573e5"
VERSIONS = {"numpy": "2.1.3", "pandas": "2.3.3", "scipy": "1.16.3", "sklearn": "1.6.1"}
LABELS = [
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "glm-5.3", "glm-5.3-flash", "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
    "gpt-6-astra", "kimi-k3", "qwen3.8-max",
]
COVERED = [label for label in LABELS if label != "gemini-3.7-flash"]
# Frozen from the accepted source gate before any model predictions.
EVALUATION_ROWS = 729
EVALUATION_GROUPS = 56
CLASS_ROWS = {
    "claude-fable-5": 19, "claude-opus-4-6": 55, "claude-opus-4-8": 55, "claude-opus-5": 54,
    "claude-sonnet-5": 52, "gemini-3.1-pro-preview": 54, "gemini-3.6-flash": 51, "gemini-3.8-flash": 54,
    "glm-5.3": 16, "glm-5.3-flash": 17, "gpt-5.5": 54, "gpt-5.6-luna": 52, "gpt-5.6-sol": 54,
    "gpt-5.6-terra": 53, "gpt-6-astra": 54, "kimi-k3": 17, "qwen3.8-max": 18,
}
RESULT_SCOPE = "fixed alpha=.10 full-data refit evaluated once on an accepted external 17-label imbalanced panel"
METRICS_SCOPE = "external 17-label imbalanced panel with 18-way predictions; not 18-class balanced accuracy"
ENVIRONMENT_SCOPE = "full-data refit followed by one external 17-label imbalanced panel evaluation"
FROZEN_RECIPE_INPUTS = {
    "style_extra.csv": "1e9125fd49b19084842332ede84ea65bc0174ee1bb28c8d48acc8ddbd5e36c0e",
    "public.csv": "0a1e31a0f8b0ece7ca44a961075252fe275107f3710804971838d714a4e77aaa",
    "stylegram_core.py": "caa46bd121164aba891c3d78e662540db9294d653db09f53d02833c1b4ab77ef",
    "svc4.py": "80f66aedad27e8017defe31de7f0286c1b54efa5e96ba6e2cb42516a86e73c18",
    "margin_blend.py": "8c4288447de755d9bd160e5646794b2307f56ae00ae11104d062d9697ef5dc83",
    "guard.py": "eaf2c31758045488eb53ba3f9c7f824c848bfe0a33698fab816d96485f9b6d3b",
}
MEMBERS = ("stylegram_plain", "stylegram_centred", "svc4")
DYNAMIC = {"state.json", "guard.json", "training.log", ".run.lock", "artifacts.json"}
STABLE = {
    "environment.json", "training-membership.npz", "input_preflight.json",
    "features-stylegram.json", "weights-stylegram.npz", "weights-stylegram.json",
    "fit-stylegram_plain.json", "fit-stylegram_centred.json", "stylegram-scores.npz",
    "features-svc4.json", "weights-svc4.npz", "weights-svc4.json", "fit-svc4.json",
    "component-and-blend-scores.npz", "metrics.json", "predictions.csv", "confusion.csv", "result.json",
}
PAYLOAD_NAMES = {
    "attribution.md", "CC-BY-4.0.txt", "config.json", "data_gate.json",
    "own-attribution.json", "own-README.md", "evaluation.csv", "guard.py", "margin_blend.py",
    "public.csv", "runner.py", "stylegram_core.py", "style_extra.csv", "svc4.py",
}
# Numeric tolerance only, never an execution-budget grace period.
TIME_TOLERANCE = 1e-6


class VerificationError(ValueError):
    pass


def require(value, message):
    if not value:
        raise VerificationError(message)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pairs_hook(pairs):
    data = {}
    for key, value in pairs:
        require(key not in data, "Duplicate JSON key: " + key)
        data[key] = value
    return data


def invalid_constant(value):
    raise VerificationError("Nonfinite JSON value: " + value)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs_hook, parse_constant=invalid_constant)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def frame(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, "Invalid integer " + name)
    return value


def number(value, name, minimum=None):
    require(not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value), "Invalid number " + name)
    require(minimum is None or value >= minimum, "Number below minimum: " + name)
    return value


def close(actual, expected, name, tolerance=1e-12):
    number(actual, name)
    require(math.isclose(actual, float(expected), rel_tol=tolerance, abs_tol=tolerance), "Recomputed numeric mismatch: " + name)


def compare_json(actual, expected, name):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), "JSON fields differ: " + name)
        for key in expected:
            compare_json(actual[key], expected[key], name + "." + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), "JSON list differs: " + name)
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare_json(left, right, f"{name}[{index}]")
    elif type(expected) is float:
        close(actual, expected, name)
    else:
        require(type(actual) is type(expected) and actual == expected, "JSON value differs: " + name)


def array_equal(actual, expected, name, approximate=False):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape, "Array shape differs: " + name)
    if approximate:
        require(actual.dtype.kind in "iuf" and np.isfinite(actual).all(), "Invalid numeric array: " + name)
        require(np.allclose(actual, expected, rtol=1e-12, atol=1e-10), "Recomputed array differs: " + name)
    else:
        require(np.array_equal(actual, expected), "Array values/order differ: " + name)


def safe_name(name):
    require(isinstance(name, str) and name and "\\" not in name and ":" not in name, "Unsafe artifact path")
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and ".." not in relative.parts and relative.as_posix() == name and name != ".", "Unsafe artifact path")


def hash_valid(value, name):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None, "Invalid SHA-256: " + name)


def verify_manifest(directory, expected=STABLE):
    directory = directory.resolve(strict=True)
    manifest = read_json(directory / "artifacts.json")
    require(isinstance(manifest, dict) and set(manifest) == {"files"} and isinstance(manifest["files"], list), "Invalid manifest")
    hashes, seen_casefold = {}, set()
    for entry in manifest["files"]:
        require(isinstance(entry, dict) and set(entry) == {"path", "bytes", "sha256"}, "Invalid artifact entry")
        name = entry["path"]; safe_name(name)
        require(name in expected and name not in hashes and name.casefold() not in seen_casefold, "Unexpected/duplicate artifact: " + name)
        path = directory / name
        require(not path.is_symlink() and path.resolve(strict=True).is_relative_to(directory) and path.is_file(), "Unsafe artifact target: " + name)
        integer(entry["bytes"], "artifact bytes"); hash_valid(entry["sha256"], name)
        require(path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"], "Artifact content changed: " + name)
        hashes[name] = entry["sha256"]; seen_casefold.add(name.casefold())
    require(set(hashes) == set(expected), "Stable artifact set is incomplete")
    actual = set()
    for path in directory.rglob("*"):
        require(not path.is_symlink(), "Symlink in result directory")
        if path.is_file():
            name = path.relative_to(directory).as_posix()
            require(name in expected or name in DYNAMIC, "Unexpected/failure/temporary result file: " + name)
            actual.add(name)
    require(actual - DYNAMIC == set(expected), "Result files do not match the complete manifest")
    for name in sorted(actual & DYNAMIC):
        hashes[name] = sha(directory / name)
    return hashes


def load_npz(path, keys):
    with np.load(path, allow_pickle=False) as saved:
        require(len(saved.files) == len(keys) and set(saved.files) == set(keys), "NPZ field set differs: " + str(path))
        arrays = {key: saved[key].copy() for key in keys}
    require(all(value.dtype.kind != "O" for value in arrays.values()), "Object NPZ arrays are forbidden")
    return arrays


def check_identity(arrays, ids, labels):
    require(arrays["ids"].dtype.kind in "US" and arrays["labels"].dtype.kind in "US", "NPZ identity arrays must be strings")
    array_equal(arrays["ids"], np.asarray(ids, dtype=str), "prediction IDs")
    array_equal(arrays["labels"], np.asarray(labels, dtype=str), "score label order")


def validate_evaluation(evaluation, rows, groups, class_rows):
    """Check a frozen incomplete panel without imposing balanced class support."""
    require(evaluation.columns.tolist() == ["id", "prompt_id", "model", "text"] and len(evaluation) == rows
            and evaluation.id.is_unique, "Evaluation schema/IDs invalid")
    require(all(evaluation[column].str.strip().ne("").all() for column in evaluation.columns), "Empty evaluation value")
    require(evaluation.prompt_id.nunique() == groups, "Evaluation prompt count changed")
    require(not evaluation.duplicated(["prompt_id", "model"]).any(), "Duplicate evaluation prompt/class pair")
    observed = {label: int(count) for label, count in evaluation.model.value_counts().items()}
    require(set(observed) == set(COVERED) and observed == class_rows, "Frozen evaluation class supports changed")
    require(sum(class_rows.values()) == rows and all(type(value) is int and value > 0 for value in class_rows.values()),
            "Invalid frozen evaluation class supports")
    require(evaluation.groupby("prompt_id").size().between(1, len(COVERED)).all(), "Invalid incomplete evaluation panel")


def local_inputs():
    require(sha(PAYLOAD / "protocol.json") == PROTOCOL_SHA, "Frozen EXP-027 protocol changed")
    protocol = read_json(PAYLOAD / "protocol.json")
    require(protocol.get("experiment_id") == "EXP-027" and protocol.get("evaluation_accepted") is True, "External data gate is not accepted")
    require(protocol.get("evaluation_gate_status") == "PASS_FOR_EXTERNAL_SEVENTEEN_CLASS_CHECK", "Protocol data gate scope changed")
    require(protocol.get("required_versions") == VERSIONS, "Protocol dependency versions changed")
    require(protocol.get("alpha") == .1 and protocol.get("style_share") == .5 and protocol.get("public_row_weight") == .35, "Fixed recipe changed")
    require(protocol.get("bootstrap_draws") == 2000 and protocol.get("bootstrap_seed") == 42, "Fixed bootstrap protocol changed")
    require(protocol.get("official_rows") == 2699 and protocol.get("style_extra_rows") == 10073 and protocol.get("public_rows") == 5400,
            "Fixed training population changed")
    require(protocol.get("work_budget_seconds") == 3300 and protocol.get("platform_wall_seconds") == 3600 and protocol.get("rss_limit_bytes") == 12 * 1024**3, "Frozen budgets changed")
    require(protocol.get("threads") == 2 and protocol.get("gpu") is False and protocol.get("internet") is False, "Execution environment scope changed")
    require(protocol.get("train_sha256") == TRAIN_SHA, "Official training input changed")
    for name, expected in FROZEN_RECIPE_INPUTS.items():
        require(protocol["input_hashes"].get(name) == expected, "Original EXP-026 recipe input changed: " + name)
    require(set(protocol["input_hashes"]) == PAYLOAD_NAMES, "Payload input set changed")
    sources = {PAYLOAD / "protocol.json": PROTOCOL_SHA}
    for name, digest in protocol["input_hashes"].items():
        safe_name(name); hash_valid(digest, name)
        require(sha(PAYLOAD / name) == digest, "Payload input changed: " + name)
        sources[PAYLOAD / name] = digest
    originals = {
        ROOT / "train.csv": TRAIN_SHA,
        ROOT / "experiments/EXP-010-012-stylegram-share/payload/extra.csv": protocol["input_hashes"]["style_extra.csv"],
        ROOT / "cloud/svc4-raw-public-r1/payload/public.csv": protocol["input_hashes"]["public.csv"],
        EXP / "evaluation.csv": EVALUATION_SHA, EXP / "metrics/data-gate.json": GATE_SHA,
    }
    for path, digest in originals.items():
        require(sha(path) == digest, "Original accepted input changed: " + str(path))
        sources[path] = digest
    require(protocol["evaluation_sha256"] == protocol["input_hashes"]["evaluation.csv"] == EVALUATION_SHA, "Evaluation hash binding changed")
    require(protocol["data_gate_sha256"] == protocol["input_hashes"]["data_gate.json"] == GATE_SHA, "Gate hash binding changed")
    gate = read_json(EXP / "metrics/data-gate.json")
    require(gate["status"] == "PASS_FOR_EXTERNAL_SEVENTEEN_CLASS_CHECK" and gate["evaluation"]["sha256"] == EVALUATION_SHA, "Data audit did not pass")
    for key, value in {"rows": EVALUATION_ROWS, "groups": EVALUATION_GROUPS, "labels": COVERED, "class_counts": CLASS_ROWS}.items():
        compare_json(gate["evaluation"].get(key), value, "data_gate.evaluation." + key)
    tables = {"official": frame(ROOT / "train.csv"), "style": frame(PAYLOAD / "style_extra.csv"),
              "svc": frame(PAYLOAD / "public.csv"), "evaluation": frame(EXP / "evaluation.csv")}
    labels = protocol["labels"]
    require(labels == LABELS and protocol["evaluation_labels"] == COVERED, "Training/evaluation label order changed")
    for name, count, identity in (("official", 2699, "id"), ("style", 10073, "row_id"), ("svc", 5400, "id")):
        table = tables[name]
        require(len(table) == count and table[identity].is_unique and set(table.model) == set(labels), "Training membership invalid: " + name)
    validate_evaluation(tables["evaluation"], EVALUATION_ROWS, EVALUATION_GROUPS, CLASS_ROWS)
    require(protocol["evaluation_rows"] == EVALUATION_ROWS and protocol["evaluation_groups"] == EVALUATION_GROUPS,
            "Evaluation protocol counts changed")
    compare_json(protocol.get("evaluation_class_counts"), CLASS_ROWS, "protocol.evaluation_class_counts")
    return protocol, tables, sources


def verify_terminal(result, guard, state, protocol):
    require(result.get("status") == guard.get("status") == state.get("status") == "COMPLETE", "Run/guard/state are not all COMPLETE")
    require(type(result.get("trained_svms")) is int and result["trained_svms"] == 3, "Expected three fitted SVMs")
    require(type(result.get("convergence_warnings")) is int and result["convergence_warnings"] == 0, "Convergence warnings present")
    for obj in (result, state):
        completed = obj.get("completed_fits")
        require(isinstance(completed, list) and len(completed) == 3 and set(completed) == set(MEMBERS), "Missing/duplicate completed fit")
    require(type(guard.get("child_exit")) is int and guard["child_exit"] == 0, "Nonzero child exit")
    require("termination_reason" in guard and guard["termination_reason"] is None and guard.get("artifacts_verified") is True and not guard.get("traceback"), "Guard did not complete cleanly")
    require(guard.get("wall_seconds") == protocol["work_budget_seconds"] and guard.get("rss_limit_bytes") == protocol["rss_limit_bytes"], "Guard budgets differ")
    elapsed = number(guard.get("elapsed_seconds"), "guard elapsed", 0)
    total = number(result.get("total_seconds"), "runner total", 0)
    require(total <= protocol["work_budget_seconds"] + TIME_TOLERANCE, "Runner exceeded work budget")
    require(elapsed <= protocol["platform_wall_seconds"] + TIME_TOLERANCE, "Guard exceeded platform budget")
    require(total <= elapsed + TIME_TOLERANCE, "Runner elapsed exceeds guard elapsed")
    require(integer(guard.get("peak_rss_bytes"), "peak RSS") <= protocol["rss_limit_bytes"], "Observed guard RSS exceeded limit")
    require(result.get("protocol_sha256") == PROTOCOL_SHA, "Result belongs to a different protocol")
    require(result.get("gpu_used") is False and result.get("models_saved") is False and result.get("calibrated_probabilities") is False and result.get("sparse_scenarios_run") is False, "Execution scope differs")
    require(result.get("alpha") == .1 and result.get("score_columns") == 18 and result.get("metric") == "covered_label_macro_recall", "Wrong prediction/metric scope")
    require(result.get("scope") == RESULT_SCOPE, "Result scope is not the fixed external 17-label check")
    require(result.get("evaluation_rows") == EVALUATION_ROWS and result.get("evaluation_groups") == EVALUATION_GROUPS
            and result.get("training_labels") == LABELS == protocol["labels"] and result.get("evaluation_labels") == COVERED, "Result population differs")
    compare_json(result.get("evaluation_class_counts"), CLASS_ROWS, "result.evaluation_class_counts")


def validate_membership(directory, tables, labels):
    keys = {"labels", "official_ids", "official_prompt_ids", "official_true", "style_public_ids",
            "style_public_prompt_ids", "style_public_true", "svc_public_ids", "svc_public_prompt_ids", "svc_public_true"}
    saved = load_npz(directory / "training-membership.npz", keys)
    array_equal(saved["labels"], labels, "membership label order")
    index = {label: number for number, label in enumerate(labels)}
    targets = {}
    for name, prefix, identity in (("official", "official", "id"), ("style", "style_public", "row_id"), ("svc", "svc_public", "id")):
        table = tables[name]
        target = table.model.map(index).to_numpy(dtype=np.int64)
        array_equal(saved[prefix + "_ids"], table[identity].to_numpy(dtype=str), prefix + " IDs")
        array_equal(saved[prefix + "_prompt_ids"], table.prompt_id.to_numpy(dtype=str), prefix + " prompt IDs")
        require(saved[prefix + "_true"].dtype.kind in "iu", "Training targets must be integers")
        array_equal(saved[prefix + "_true"], target, prefix + " labels")
        targets[name] = target
    return targets


def verify_weights(directory, kind, official_y, public_y, labels):
    saved = load_npz(directory / f"weights-{kind}.npz", {"labels", "target", "weights"})
    metadata = read_json(directory / f"weights-{kind}.json")
    array_equal(saved["labels"], labels, kind + " weight labels")
    target = np.r_[official_y, public_y]
    require(saved["target"].dtype.kind in "iu", "Weight targets must be integer class indices")
    array_equal(saved["target"], target, kind + " target order")
    counts = np.bincount(target, minlength=len(labels))
    official_counts = np.bincount(official_y, minlength=len(labels))
    public_counts = np.bincount(public_y, minlength=len(labels))
    require((counts > 0).all() and (public_counts > 0).all(), "Weights require every training class")
    if kind == "stylegram":
        expected = np.r_[np.full(len(official_y), .5), .5 * official_counts[public_y] / public_counts[public_y]]
        class_weights = np.ones(len(labels))
        require(metadata.get("share") == .5 and metadata.get("semantics") == "official .5 per row; public .5 * official class count / public class count; no class_weight", "Wrong Stylegram weight semantics")
        expected_dtype = np.dtype("float64")
    else:
        require(kind == "svc4", "Unknown weight recipe")
        expected = np.r_[np.ones(len(official_y), dtype=np.float32), np.full(len(public_y), .35, dtype=np.float32)]
        class_weights = len(target) / (len(labels) * counts)
        require(metadata.get("public_row_weight") == .35 and metadata.get("semantics") == "sklearn 1.6.1 balanced from unweighted class counts, multiplied by float32 per-row weights", "Wrong svc4 class-weight semantics")
        expected_dtype = np.dtype("float32")
    require(saved["weights"].dtype == expected_dtype and metadata.get("sample_weight_dtype") == str(expected_dtype), "Wrong sample-weight dtype")
    array_equal(saved["weights"], expected, kind + " exact sample weights")
    official_mass = np.bincount(official_y, weights=expected[:len(official_y)], minlength=len(labels))
    public_mass = np.bincount(public_y, weights=expected[len(official_y):], minlength=len(labels))
    require(metadata.get("labels") == labels, "Weight receipt label order differs")
    for key, value in (("class_counts", counts), ("class_weights", class_weights), ("official_sample_mass", official_mass),
                       ("public_sample_mass", public_mass), ("effective_class_mass", (official_mass + public_mass) * class_weights)):
        array_equal(metadata.get(key), value, kind + "." + key, approximate=True)
    if kind == "stylegram":
        array_equal(official_mass + public_mass, official_counts, "Stylegram preserved per-class mass", approximate=True)
    else:
        array_equal(public_mass, float(np.float32(.35)) * public_counts, "svc4 public float32 mass", approximate=True)
    return {"kind": kind, "rows": len(target), "sample_weight_dtype": str(expected_dtype), "verified": True}


def native_parameters(member):
    result = {"C": .5, "class_weight": None, "dual": True, "fit_intercept": True, "intercept_scaling": 1,
              "loss": "squared_hinge", "max_iter": 4000, "multi_class": "ovr", "penalty": "l2",
              "random_state": 42, "tol": .0001, "verbose": 0}
    if member == "svc4":
        result.update(class_weight="balanced", dual="auto", max_iter=5000)
    return result


def check_feature_receipt(info, kind, rows):
    dtype = "float64" if kind == "stylegram" else "float32"
    matrices = ("plain", "centred") if kind == "stylegram" else ("plain",)
    columns = None
    for name in matrices:
        receipt = info[name]
        shape = receipt.get("shape")
        require(isinstance(shape, list) and len(shape) == 2 and shape[0] == rows, "Feature training row count differs")
        integer(shape[1], "feature columns", 1)
        columns = shape[1] if columns is None else columns
        require(shape[1] == columns and receipt.get("dtype") == dtype, "Feature dtype/column shape differs")
        require(integer(receipt.get("nnz"), "feature nnz", 1) <= rows * columns, "Impossible feature nnz")
        integer(receipt.get("csr_bytes"), "feature bytes", 1); hash_valid(receipt.get("sha256"), "matrix")
    require(info.get("evaluation_rows_in_fit") == 0, "Feature receipt includes evaluation rows")
    number(info.get("feature_seconds"), "feature seconds", 0)
    views = info.get("views")
    require(isinstance(views, list) and len(views) == 4, "Missing four-view receipt")
    total = 0
    for view, (name, scale, cap) in zip(views, (("char", 1., 250000), ("word", .7, 180000), ("shape", .5, 60000), ("function", .5, 50000))):
        require(view.get("view") == name and view.get("scale") == scale, "Feature view configuration differs")
        count = integer(view.get("columns"), "view columns", 1)
        require(count <= cap, "View feature cap exceeded")
        total += count
        hash_valid(view.get("vocabulary_sha256"), "vocabulary"); hash_valid(view.get("idf_sha256"), "IDF")
    require(total == columns, "Feature view sizes do not sum to matrix width")
    return columns


def verify_fits(directory, protocol, result):
    style = read_json(directory / "features-stylegram.json")
    svc = read_json(directory / "features-svc4.json")
    widths = {"stylegram": check_feature_receipt(style, "stylegram", 12772), "svc4": check_feature_receipt(svc, "svc4", 8099)}
    records = []
    for member in MEMBERS:
        record = read_json(directory / f"fit-{member}.json")
        kind = "svc4" if member == "svc4" else "stylegram"
        require(record.get("member") == member and record.get("rows") == (8099 if kind == "svc4" else 12772) and record.get("features") == widths[kind], "Fit identity/shape differs")
        require(record.get("labels") == protocol["labels"] and record.get("classes") == list(range(18)), "Fit label order differs")
        require(canonical(record.get("parameters")) == canonical(native_parameters(member)), "Native fit parameters differ")
        require(0 < integer(record.get("n_iter"), "fit iterations") < native_parameters(member)["max_iter"], "Fit reached iteration cap")
        require(type(record.get("convergence_warnings")) is int and record["convergence_warnings"] == 0 and record.get("models_saved") is False, "Fit convergence/saving scope differs")
        number(record.get("fit_seconds"), "fit seconds", 0)
        require(integer(record.get("rss_bytes_after_fit"), "fit RSS", 1) <= protocol["rss_limit_bytes"], "Recorded fit RSS exceeded limit")
        records.append(record)
    compare_json(result.get("fit_records"), records, "result.fit_records")
    close(result["timings"]["stylegram_feature_seconds"], style["feature_seconds"], "Stylegram feature timing")
    close(result["timings"]["svc4_feature_seconds"], svc["feature_seconds"], "svc4 feature timing")
    for key, value in result["timings"].items():
        number(value, key, 0)
    return records


def independent_blend(style, svc):
    def normalize(matrix):
        values = np.asarray(matrix, dtype=np.float64)
        maximum = np.max(np.abs(values), axis=1, keepdims=True)
        scaled = np.divide(values, maximum, out=np.zeros_like(values), where=maximum != 0)
        centered = scaled - scaled.mean(axis=1, keepdims=True)
        population_std = np.sqrt(np.mean(centered * centered, axis=1, keepdims=True))
        return np.divide(centered, population_std, out=np.zeros_like(centered), where=population_std != 0)
    return .9 * normalize(style) + .1 * normalize(svc)


def valid_scores(scores, rows, name):
    require(scores.dtype.kind == "f" and scores.shape == (rows, 18) and np.isfinite(scores).all(), "Invalid finite 18-column score matrix: " + name)


def verify_score_files(directory, evaluation, labels):
    style = load_npz(directory / "stylegram-scores.npz", {"ids", "labels", "scores", "plain_scores", "group_ids", "centred_mask"})
    mixed = load_npz(directory / "component-and-blend-scores.npz", {"ids", "labels", "stylegram_scores", "svc4_scores", "blend_scores", "alpha", "group_ids", "centred_mask"})
    ids = evaluation.id.tolist()
    check_identity(style, ids, labels); check_identity(mixed, ids, labels)
    for name in ("scores", "plain_scores"):
        valid_scores(style[name], len(evaluation), name)
    for name in ("stylegram_scores", "svc4_scores", "blend_scores"):
        valid_scores(mixed[name], len(evaluation), name)
    array_equal(style["scores"], mixed["stylegram_scores"], "Stylegram scores across caches")
    for name in ("group_ids", "centred_mask"):
        array_equal(style[name], mixed[name], "Stylegram route across caches: " + name)
    groups, mask = style["group_ids"], style["centred_mask"]
    require(groups.shape == mask.shape == (len(evaluation),) and groups.dtype.kind in "iu" and (groups >= 1).all() and mask.dtype.kind == "b", "Invalid inferred group/mask arrays")
    counts = Counter(groups.tolist())
    expected_mask = np.asarray([2 <= counts[group] <= 18 for group in groups], dtype=bool)
    array_equal(mask, expected_mask, "inferred group size 2..18 route")
    array_equal(style["scores"][~mask], style["plain_scores"][~mask], "Stylegram plain fallback")
    require(mixed["alpha"].shape == () and mixed["alpha"].dtype.kind == "f" and mixed["alpha"].item() == .1, "Blend alpha is not fixed .10")
    expected_blend = independent_blend(mixed["stylegram_scores"], mixed["svc4_scores"])
    require(np.allclose(mixed["blend_scores"], expected_blend, rtol=1e-12, atol=1e-12), "Blend scores differ from independent fixed .10 z-score calculation")
    require(np.array_equal(mixed["blend_scores"].argmax(axis=1), expected_blend.argmax(axis=1)),
            "Blend argmax differs from independent fixed .10 z-score calculation")
    scores = {"stylegram": mixed["stylegram_scores"], "svc4": mixed["svc4_scores"], "blend": mixed["blend_scores"]}
    return scores, groups, mask, float(np.max(np.abs(expected_blend - mixed["blend_scores"])))


def prediction_indices(scores):
    # Deliberately use every score column; covered labels are never a mask.
    return {name: values.argmax(axis=1) for name, values in scores.items()}


def validate_prediction_frame(saved, evaluation, labels, predictions, groups, mask):
    columns = ["id", "prompt_id", "model"]
    for member in ("stylegram", "svc4", "blend"):
        columns.extend([member + "_prediction", member + "_correct"])
    columns.extend(["stylegram_group_id", "stylegram_centred"])
    require(saved.columns.tolist() == columns and len(saved) == len(evaluation) and saved.id.is_unique, "Prediction CSV schema/rows differ")
    index = {label: i for i, label in enumerate(labels)}
    truth = evaluation.model.map(index).to_numpy(dtype=np.int64)
    for column in ("id", "prompt_id", "model"):
        require(saved[column].tolist() == evaluation[column].tolist(), "Prediction CSV mapping differs: " + column)
    for member, prediction in predictions.items():
        require(saved[member + "_prediction"].tolist() == np.asarray(labels)[prediction].tolist(), "18-way argmax prediction mismatch: " + member)
        require(saved[member + "_correct"].tolist() == [str(bool(value)) for value in prediction == truth], "Stored correctness differs: " + member)
    require(saved.stylegram_group_id.tolist() == groups.astype(str).tolist() and saved.stylegram_centred.tolist() == [str(bool(value)) for value in mask], "Prediction CSV route differs")
    return truth


def covered_metrics(truth, prediction, labels):
    covered = np.flatnonzero(np.bincount(truth, minlength=len(labels)) > 0)
    support = np.bincount(truth, minlength=len(labels))
    correct = np.bincount(truth[truth == prediction], minlength=len(labels))
    return {"metric": "covered_label_macro_recall", "score_columns": len(labels),
            "covered_labels": [labels[index] for index in covered], "covered_label_count": int(len(covered)),
            "macro_recall": float(np.mean(correct[covered] / support[covered])), "accuracy": float(np.mean(truth == prediction)),
            "correct": int(np.sum(truth == prediction)), "rows": len(truth),
            "per_class": [{"model": labels[index], "support": int(support[index]), "correct": int(correct[index]),
                           "recall": float(correct[index] / support[index])} for index in covered]}


def paired_bootstrap(truth, reference, prediction, prompts, labels, draws=2000, seed=42):
    names = sorted(set(prompts)); code = {name: i for i, name in enumerate(names)}
    group = np.asarray([code[name] for name in prompts], dtype=int)
    covered = np.unique(truth); class_code = {value: i for i, value in enumerate(covered)}
    column = np.asarray([class_code[value] for value in truth], dtype=int)
    support = np.zeros((len(names), len(covered)), dtype=np.int64)
    delta = np.zeros_like(support)
    np.add.at(support, (group, column), 1)
    np.add.at(delta, (group, column), (prediction == truth).astype(int) - (reference == truth).astype(int))
    choices = np.random.default_rng(seed).integers(0, len(names), size=(draws, len(names)))
    multiplicity = np.zeros((draws, len(names)), dtype=np.int64)
    np.add.at(multiplicity, (np.arange(draws)[:, None], choices), 1)
    denominators = multiplicity @ support
    invalid = int((denominators == 0).any(axis=1).sum())
    interval = None
    if invalid == 0:
        changes = (multiplicity @ delta / denominators).mean(axis=1)
        interval = np.quantile(changes, [.025, .975], method="linear").tolist()
    return {"delta_definition": "blend_minus_stylegram_covered_label_macro_recall", "unit": "external_prompt_id",
            "draws": draws, "seed": seed, "valid_draws": draws - invalid, "invalid_draws": invalid, "ci95": interval,
            "missing_class_policy": "null interval if any resample lacks a covered true class",
            "conditional_on_fitted_models": True, "includes_retraining_uncertainty": False}


def verify_metrics(directory, result, evaluation, labels, scores, groups, mask):
    predictions = prediction_indices(scores)
    truth = validate_prediction_frame(frame(directory / "predictions.csv"), evaluation, labels, predictions, groups, mask)
    metrics = {name: covered_metrics(truth, prediction, labels) for name, prediction in predictions.items()}
    require(all(value["covered_label_count"] == len(COVERED) and value["covered_labels"] == COVERED for value in metrics.values()),
            "Metrics do not cover exactly the frozen 17 labels")
    style_good, blend_good = predictions["stylegram"] == truth, predictions["blend"] == truth
    comparison = {"alpha": .1, "delta_macro_recall": metrics["blend"]["macro_recall"] - metrics["stylegram"]["macro_recall"],
                  "fixed_errors": int(np.sum(~style_good & blend_good)), "new_errors": int(np.sum(style_good & ~blend_good)),
                  "prediction_disagreements": int(np.sum(predictions["stylegram"] != predictions["blend"])),
                  "paired_prompt_bootstrap": paired_bootstrap(truth, predictions["stylegram"], predictions["blend"], evaluation.prompt_id.to_numpy(), labels)}
    receipt = read_json(directory / "metrics.json")
    require(receipt.get("scope") == METRICS_SCOPE, "Metrics scope is not explicitly 17-covered-class")
    compare_json(receipt.get("metrics"), metrics, "metrics.json.metrics")
    compare_json(receipt.get("comparison"), comparison, "metrics.json.comparison")
    compare_json(result.get("metrics"), metrics, "result.metrics")
    compare_json(result.get("comparison"), comparison, "result.comparison")
    confusion = frame(directory / "confusion.csv")
    require(confusion.columns.tolist() == ["member", "true_model", "predicted_model", "count"]
            and len(confusion) == 3 * len(COVERED) * 18
            and not confusion.duplicated(["member", "true_model", "predicted_model"]).any(), "Confusion table must have all 918 unique cells")
    expected_confusion = []
    for member, prediction in predictions.items():
        for true in np.unique(truth):
            counts = np.bincount(prediction[truth == true], minlength=18)
            for predicted in range(18):
                expected_confusion.append({"member": member, "true_model": labels[true], "predicted_model": labels[predicted], "count": str(int(counts[predicted]))})
    require(confusion.to_dict("records") == expected_confusion, "Confusion counts/order differ from full 18-way predictions")
    per_class = [{"member": member, **row} for member, value in metrics.items() for row in value["per_class"]]
    errors = []
    for index in np.flatnonzero(predictions["stylegram"] != predictions["blend"]):
        errors.append({"id": evaluation.id.iloc[index], "prompt_id": evaluation.prompt_id.iloc[index], "model": labels[truth[index]],
                       "stylegram_prediction": labels[predictions["stylegram"][index]], "svc4_prediction": labels[predictions["svc4"][index]],
                       "blend_prediction": labels[predictions["blend"][index]], "stylegram_centred": bool(mask[index]),
                       "change": "fixed_error" if blend_good[index] else "new_error" if style_good[index] else "both_wrong_changed"})
    return metrics, comparison, per_class, errors


def persist(verified, per_class, errors, comparison):
    verified_path = ANALYSIS / "verified.json"
    observed = read_json(verified_path).get("verified_at_utc") if verified_path.exists() else datetime.now(timezone.utc).isoformat()
    require(isinstance(observed, str) and datetime.fromisoformat(observed).tzinfo is not None, "Invalid existing verification timestamp")
    boot = comparison["paired_prompt_bootstrap"]
    comparison_row = {key: value for key, value in comparison.items() if key != "paired_prompt_bootstrap"}
    comparison_row.update(ci95_lower=boot["ci95"][0] if boot["ci95"] else None,
                          ci95_upper=boot["ci95"][1] if boot["ci95"] else None,
                          metric="covered_label_macro_recall", score_columns=18, covered_classes=len(COVERED),
                          bootstrap_draws=2000, bootstrap_seed=42, conditional_on_refitted_models=True)
    error_columns = ["id", "prompt_id", "model", "stylegram_prediction", "svc4_prediction", "blend_prediction", "stylegram_centred", "change"]
    tables = {"per_class.csv": pd.DataFrame(per_class), "errors.csv": pd.DataFrame(errors, columns=error_columns),
              "comparison.csv": pd.DataFrame([comparison_row])}
    outputs = {ANALYSIS / name: value.to_csv(index=False, lineterminator="\n").encode("utf-8") for name, value in tables.items()}
    verified.update(verified_at_utc=observed, analysis_outputs=[{"path": path.name, "sha256": sha_bytes(content), "bytes": len(content)} for path, content in outputs.items()])
    outputs[verified_path] = json_bytes(verified)
    for path, content in outputs.items():
        require(not path.exists() or (path.is_file() and path.read_bytes() == content), "Refusing different existing analysis output: " + str(path))
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    for path, content in outputs.items():
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(content)
        require(path.read_bytes() == content, "Analysis output changed: " + str(path))


def verify(directory, write=True):
    require(directory.is_dir(), "--input must name the downloaded fixed-EXP-027 results directory")
    require(not (directory / "failure.json").exists(), "Run left failure.json")
    protocol, tables, input_hashes = local_inputs()
    artifact_hashes = verify_manifest(directory)
    result, guard, state = [read_json(directory / name) for name in ("result.json", "guard.json", "state.json")]
    verify_terminal(result, guard, state, protocol)
    environment = read_json(directory / "environment.json")
    require(canonical(environment.get("protocol")) == canonical(protocol) and environment.get("versions") == VERSIONS, "Runtime protocol/dependencies differ")
    require(environment.get("gpu_requested") is False and environment.get("models_saved") is False and environment.get("score_type") == "uncalibrated 18-class LinearSVC decision margins", "Runtime model/scoring scope differs")
    require(environment.get("scope") == ENVIRONMENT_SCOPE, "Runtime evaluation scope differs")
    preflight = read_json(directory / "input_preflight.json")
    for key, value in {"status": "PASS", "official_rows": 2699, "style_public_rows": 10073, "svc_public_rows": 5400,
                       "evaluation_rows": EVALUATION_ROWS, "evaluation_groups": EVALUATION_GROUPS, "evaluation_labels": COVERED,
                       "evaluation_class_counts": CLASS_ROWS,
                       "training_labels": protocol["labels"], "hashes_verified": True, "evaluation_accepted": True,
                       "evaluation_texts_used_for_classifier_feature_fit": False,
                       "inference_grouping": "unlabelled full external text batch only; no true prompt IDs or labels",
                       "content_grouping_float64_square_bytes": 8 * EVALUATION_ROWS**2}.items():
        compare_json(preflight.get(key), value, "input_preflight." + key)
    targets = validate_membership(directory, tables, protocol["labels"])
    weights = [verify_weights(directory, "stylegram", targets["official"], targets["style"], protocol["labels"]),
               verify_weights(directory, "svc4", targets["official"], targets["svc"], protocol["labels"])]
    fit_records = verify_fits(directory, protocol, result)
    scores, groups, mask, maximum_blend_error = verify_score_files(directory, tables["evaluation"], protocol["labels"])
    metrics, comparison, per_class, errors = verify_metrics(directory, result, tables["evaluation"], protocol["labels"], scores, groups, mask)
    for path, digest in input_hashes.items():
        require(sha(path) == digest, "Local input changed during verification: " + str(path))
    require(verify_manifest(directory) == artifact_hashes, "Result bytes changed during verification")
    verified = {
        "schema_version": 1, "status": "PASS", "experiment_id": "EXP-027", "declared_notebook_version": 2,
        "scope": "external 17-covered-class macro recall on an imbalanced incomplete panel with 18-way argmax; not 18-class BA, full ATLAS20, or a leaderboard result",
        "input_directory": str(directory.resolve()), "protocol_sha256": PROTOCOL_SHA, "verifier_sha256": sha(Path(__file__)),
        "evaluation_sha256": EVALUATION_SHA, "data_gate_sha256": GATE_SHA,
        "local_inputs": {path.relative_to(ROOT).as_posix(): digest for path, digest in sorted(input_hashes.items(), key=lambda item: str(item[0]))},
        "result_files": artifact_hashes, "stable_artifacts_verified": 18, "trained_svms": 3, "convergence_warnings": 0,
        "evaluation_rows": EVALUATION_ROWS, "evaluation_groups": EVALUATION_GROUPS, "evaluation_labels": COVERED, "class_rows": CLASS_ROWS,
        "score_columns": 18, "prediction_class_mask_applied": False, "alpha": .1,
        "weight_checks": weights, "fit_records": fit_records, "runtime_versions": environment["versions"],
        "analysis_versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "budget": {"runner_seconds": result["total_seconds"], "work_limit_seconds": 3300, "guard_seconds": guard["elapsed_seconds"],
                   "platform_limit_seconds": 3600, "peak_rss_bytes": guard["peak_rss_bytes"], "rss_limit_bytes": 12 * 1024**3,
                   "comparison_tolerance_seconds": TIME_TOLERANCE, "tolerance_scope": "floating-point rounding only, no execution grace"},
        "stylegram_route": {"inferred_groups": len(set(groups.tolist())), "centred_rows": int(mask.sum()), "plain_rows": int((~mask).sum()),
                            "eligible_group_size_range": [2, 18], "plain_fallback_exact_match": True},
        "maximum_independent_blend_absolute_error": maximum_blend_error, "recomputed_metrics": metrics, "recomputed_comparison": comparison,
        "independent_blend_argmax_exact_match": True,
        "limits": protocol["limits"] + ["Matrix/vocabulary/IDF receipts and memberships are checked, but fitted estimators and matrix contents were not persisted for independent reconstruction.",
                                         "The bootstrap interval is conditional on these refitted models and the selected 17-class source; it includes no retraining uncertainty. Any missing-class draw makes the interval null without dropping draws.",
                                         "The caller must bind the local download to Kaggle V2; this offline verifier authenticates the frozen protocol/input/result bytes without querying the remote service."],
        "real_training_performed_by_verifier": False, "remote_calls_performed": False,
    }
    if write:
        persist(verified, per_class, errors, comparison)
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        result = verify(args.input, write=not args.check_only)
    except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
        print("VERIFICATION_FAILED: " + str(error), file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps({"status": result["status"], "stable_artifacts_verified": 18, "trained_svms": 3,
                      "scope": result["scope"], "analysis_written": not args.check_only,
                      "comparison": result["recomputed_comparison"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
