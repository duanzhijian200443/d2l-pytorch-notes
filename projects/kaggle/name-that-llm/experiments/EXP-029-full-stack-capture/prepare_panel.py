"""Prepare the frozen EXP027 texts for exploratory full-stack inference.

No selection, text edits, model calls, remote operations, or scoring. Only
panel/test.csv may enter this panel's cloud input. The label mapping remains
outside panel/ and is explicitly local-only. Different existing output bytes
are refused; repeating the unchanged preparation is idempotent.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "experiments/EXP-027-multiclass-external-validation/evaluation.csv"
SOURCE_GATE = SOURCE.parent / "metrics/data-gate.json"
OFFICIAL = ROOT / "train.csv"
MANIFEST = HERE / "panel-manifest.json"
LABELS = [
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "glm-5.3", "glm-5.3-flash", "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
    "gpt-6-astra", "kimi-k3", "qwen3.8-max",
]
PINS = {
    SOURCE: "8ec3217f1f469b66170f68f7abdb3ef1973a64e47d983c59b40a4d1ce2094670",
    SOURCE_GATE: "cead2f4745a3e4af6a8adffe2361efd725e2710971a12123083b927589f573e5",
    OFFICIAL: "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707",
}
EXPECTED_TEST_SHA256 = "8eeec7f64e924f0fd79dca57d324d42d269f400bdfbfff76d60b4c168d9ae6f2"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(content):
    return hashlib.sha256(content).hexdigest()


def receipt(path):
    content = path.read_bytes()
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": len(content), "sha256": sha(content)}


def csv_bytes(frame):
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def persist(outputs):
    for path, content in outputs.items():
        require(not path.is_symlink(), "Refusing output symlink: " + str(path))
        require(not path.exists() or (path.is_file() and path.read_bytes() == content),
                "Refusing different existing output: " + str(path))
    for path, content in outputs.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(content)
        require(path.read_bytes() == content, "Written output bytes differ: " + str(path))


def main():
    inputs = [receipt(path) for path in PINS]
    for path, expected in PINS.items():
        require(sha(path.read_bytes()) == expected, "Pinned input differs: " + str(path))
    gate = json.loads(SOURCE_GATE.read_text(encoding="utf-8"))
    require(gate["status"] == "PASS_FOR_EXTERNAL_SEVENTEEN_CLASS_CHECK", "Source gate is not accepted")
    require(gate["evaluation"]["sha256"] == PINS[SOURCE], "Source gate does not bind this evaluation")
    source = pd.read_csv(SOURCE, dtype=str, keep_default_na=False)
    official = pd.read_csv(OFFICIAL, dtype=str, keep_default_na=False)
    covered = [label for label in LABELS if label != "gemini-3.7-flash"]
    require(list(source.columns) == ["id", "prompt_id", "model", "text"], "Source schema differs")
    require(len(source) == 729 and source.id.is_unique and source.prompt_id.nunique() == 56,
            "Source row, ID or prompt count differs")
    require(source.text.str.strip().ne("").all() and not source.duplicated(["prompt_id", "model"]).any(),
            "Blank text or duplicate source prompt/model")
    require(set(source.model) == set(covered), "Source covered classes differ")
    require(source.model.value_counts().to_dict() == gate["evaluation"]["class_counts"], "Source class counts differ")
    require(len(official) == 2699 and official.id.is_unique and official.prompt_id.nunique() == 150
            and set(official.model) == set(LABELS), "Official training identity differs")
    source["text_sha256"] = source.text.map(lambda text: sha(text.encode("utf-8")))
    source["opaque_id"] = "x27-" + source.text_sha256.str[:24]
    require(source.opaque_id.is_unique and source.text_sha256.is_unique, "Opaque ID or text hash collision")
    ordered = source.sort_values("opaque_id", kind="stable").reset_index(drop=True)
    panel = ordered[["opaque_id", "text"]].rename(columns={"opaque_id": "id"})
    sidecar = ordered[["opaque_id", "id", "prompt_id", "model", "text_sha256"]].rename(
        columns={"opaque_id": "id", "id": "source_id"})
    require(list(panel.columns) == ["id", "text"] and panel.id.str.fullmatch(r"x27-[0-9a-f]{24}").all(),
            "Inference panel contains unexpected metadata")
    require(panel.text.map(lambda text: sha(text.encode("utf-8"))).tolist() == sidecar.text_sha256.tolist(),
            "Panel text changed during preparation")
    require(set(sidecar.source_id) == set(source.id) and len(panel) == len(source), "Selection changed")
    test_content, label_content = csv_bytes(panel), csv_bytes(sidecar)
    require(sha(test_content) == EXPECTED_TEST_SHA256, "Canonical blind test projection changed")
    observed = (json.loads(MANIFEST.read_text(encoding="utf-8"))["observed_at_utc"]
                if MANIFEST.exists() else datetime.now(timezone.utc).isoformat())
    manifest = {
        "schema_version": 1, "experiment_id": "EXP-029", "observed_at_utc": observed,
        "status": "PANEL_PREPARED_NOT_EXECUTED", "scope": "exploratory_external",
        "inputs": inputs, "preparation_code": receipt(Path(__file__).resolve()),
        "rows": 729, "source_prompt_groups": 56, "covered_class_count": 17,
        "output_class_count": 18, "output_labels": LABELS, "covered_labels": covered,
        "class_counts_local_audit_only": {label: int(source.model.eq(label).sum()) for label in covered},
        "source_id_contains_truth_label_rows": int(sum(row.model in row.id for row in source.itertuples())),
        "id_rule": "x27- plus first 24 lowercase hex characters of SHA256(unchanged UTF-8 response text)",
        "row_order": "ascending opaque id, stable sort; independent of source model/prompt metadata",
        "selection": "All 729 already accepted EXP027 rows; no new exclusions, truncation, filtering or sampling",
        "test": {"path": "panel/test.csv", "columns": ["id", "text"], "rows": len(panel),
                 "bytes": len(test_content), "sha256": sha(test_content), "ids_unique": True,
                 "text_bytes_unchanged": True, "cloud_input_allowed": True},
        "labels_local_only": {"path": "labels-local-only.csv", "columns": list(sidecar.columns),
                              "rows": len(sidecar), "bytes": len(label_content), "sha256": sha(label_content),
                              "cloud_upload_allowed": False, "model_process_access_allowed": False},
        "cloud_panel_file_allowlist": ["panel/test.csv"],
        "training": {"path": "train.csv", "rows": 2699, "prompt_groups": 150, "classes": 18,
                     "sha256": PINS[OFFICIAL], "instruction": "Copy official train.csv bytes unchanged into the same runtime input directory; this script does not replace training data."},
        "runtime_requirements": [
            "Main --data and comp DATA_DIR must resolve to the same panel directory containing this test.csv and unchanged official train.csv.",
            "Keep source evaluation.csv and labels-local-only.csv outside the uploaded inference payload and model process inputs.",
            "Use all 18 output classes and original predicted/train-based priors. The 17-class coverage and local class counts must not mask outputs or set Sinkhorn targets.",
            "No --sim/--fake_test, NTL_FAKE_TEST or NTL_SMOKE path. Infer groups from texts; never pass source prompt_id as model group assignments.",
        ],
        "limits": [
            "The EXP027 gate covers only the specified two lightweight training recipes and known/exact ancestry; independence from every full-stack checkpoint is not certified.",
            "This panel has already been inspected and its publisher filtered using official 8grams. It is an exploratory external diagnostic, not fresh confirmatory validation or clean official OOF.",
            "Gemini 3.7 is absent, class supports differ, and most of the 56 true task groups are incomplete. Report covered-17-class metrics only after inference, using the local sidecar.",
        ],
        "training_performed": False, "model_predictions_read": False, "remote_operations_performed": False,
    }
    manifest_content = (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    for path, expected in PINS.items():
        require(sha(path.read_bytes()) == expected, "Input changed during preparation: " + str(path))
    persist({HERE / "panel/test.csv": test_content, HERE / "labels-local-only.csv": label_content,
             MANIFEST: manifest_content})
    reread = pd.read_csv(HERE / "panel/test.csv", dtype=str, keep_default_na=False)
    require(reread.equals(panel), "Panel CSV roundtrip changed values")
    print(json.dumps({"status": manifest["status"], "test": manifest["test"],
                      "local_sidecar_sha256": sha(label_content), "manifest_sha256": sha(manifest_content)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
