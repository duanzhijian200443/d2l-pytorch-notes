"""Prepare the frozen raw/new public data for EXP-013--016 without training.

Run with the already installed EXP-008 pandas/pyarrow interpreter, for example:
  <uv-cache>/environments-v2/audit-e5542b9a7a231d97/Scripts/python.exe -B prepare_inputs.py

No dependencies are installed. Every gate is checked before output creation.
Existing outputs must be byte-identical; an existing audit timestamp is reused
only to reproduce and compare the complete expected output. Inputs are read-only.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import platform
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAYLOAD = HERE / "payload"
RAW_DIR = "data/public-pools/prvsiyan/prvsiyans-extended-name-that-llm-dataset/v6"
DOWNLOAD_MANIFEST = "experiments/EXP-008-public-pool-audit/download-manifest.json"
SPLIT_MANIFEST = "validation/prompt-group-v1/manifest.json"
FOLDS = "validation/prompt-group-v1/folds.csv"
ROW_FLAGS = "experiments/EXP-008-public-pool-audit/outputs/row_flags.parquet"
MAIN_ROWS = "experiments/EXP-009-baseline-data-lineage/outputs/main_svc_rows.parquet"
GATE_PATH = ROOT / "experiments/EXP-014-svc4-raw-w035/metrics/data-gate.json"
AUDIT_PATH = PAYLOAD / "input_audit.json"
PINNED_SHA256 = {
    "train.csv": "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707",
    "test.csv": "4545aa3a3459f4c0724f97ad5bd06c4caa3506bdfea5fab905e8203184fc0c57",
    SPLIT_MANIFEST: "2b05e2bde656d5e773e8d512cba607859e9b9bfe7d9d7dfe9deb00907b4d93c8",
    FOLDS: "e8ba1e3712852168dc57f56960d4f3e00be612d639ade8a7dcb0f0d8335ce551",
    DOWNLOAD_MANIFEST: "5f2da21401efada94e636fd2fb843013c7eff6ae1ed2d07a5a2771c3ed071cfa",
    ROW_FLAGS: "dd0ef1c8b006f90fb5f6c493b0b5d4322563af61105d590eac10dcadaf26e6f8",
    MAIN_ROWS: "0a9eb07418b131a4da0da1d8898fa253e16090aac8c689a4f128a0626fa29643",
    "experiments/EXP-008-public-pool-audit/audit.py": "d4e88a90995b74ad0d518d8bdb476b6db62a7b6d963ac6664fd5ee4bc58c291e",
}
RISK_COLUMNS = (
    "blank_text", "unknown_label", "missing_prompt_key", "identity_mismatch",
    "label_conflict", "provenance_issue", "official_exact_match",
    "official_high_similarity", "official_shingle_review", "linked_official_prompt",
    "reported_refusal", "reported_truncated", "error_or_refusal_screen", "hold_for_review",
)
LIMITATIONS = [
    "The selected raw/new passages avoid the aggregate pool's label-conditioned punctuation cleaning, but their generation protocol was informed by official train data.",
    "The pinned prvs README lines 38-39 say target lengths were sampled from the official train word-count distribution and the protocol was adjusted using label/length comparisons with official train. The generator source is not included, so further upstream selection cannot be ruled out independently.",
    "Request/response logs establish internal consistency, not independent authentication of the underlying provider weights; Kaggle proxy rows echo routing slugs.",
    "Zero known prompt links, exact response overlap and reconstructed-prompt exact overlap do not prove the absence of semantic or undocumented prompt overlap. The host's original prompts are not available here.",
    "This is fixed-external-data development OOF preparation, not a strictly untouched holdout, full ATLAS20 OOF, or a leaderboard result.",
    "5376 of the 5400 selected texts already occur in the EXP-009 current-input main-svc4 projection. This is a source/weight diagnostic, not 5400 novel additions to ATLAS20, and that projection is not historical hidden-run membership evidence.",
]


class GateError(RuntimeError):
    """An input or output violates the frozen preparation contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha(text: str) -> str:
    return sha_bytes(text.encode("utf-8"))


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def audit_hash(text: str) -> str:
    return text_sha(normalized(text))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, dtype=str)


def source_entry(relative: str, expected: str, expected_bytes: int | None = None) -> dict:
    path = ROOT / relative
    require(path.is_file(), f"Missing pinned input: {relative}")
    require(path.resolve().is_relative_to(ROOT.resolve()), f"Input escaped project root: {relative}")
    observed_bytes = path.stat().st_size
    observed_sha = sha_file(path)
    require(observed_sha == expected, f"Pinned SHA-256 mismatch: {relative}")
    if expected_bytes is not None:
        require(observed_bytes == expected_bytes, f"Pinned size mismatch: {relative}")
    return {"path": relative, "bytes": observed_bytes, "sha256": observed_sha}


def verify_sources() -> tuple[list[dict], dict]:
    entries = [source_entry(path, digest) for path, digest in PINNED_SHA256.items()]
    manifest = read_json(ROOT / DOWNLOAD_MANIFEST)
    downloads = []
    for dataset in manifest["datasets"]:
        for file in dataset["files"]:
            entry = source_entry(file["local_path"], file["sha256"], file["bytes"])
            entry.update({"dataset_ref": dataset["ref"], "dataset_version": dataset["version"]})
            downloads.append(entry)
    require(len(downloads) == 27, "Expected exactly 27 fixed EXP-008 downloaded inputs")
    require(len({x["path"] for x in downloads}) == 27, "Duplicate download paths")
    entries.extend(downloads)
    entries.append({"path": Path(__file__).relative_to(ROOT).as_posix(), "bytes": Path(__file__).stat().st_size,
                    "sha256": sha_file(Path(__file__)), "role": "preparation code"})
    return entries, read_json(ROOT / SPLIT_MANIFEST)


def verify_official(split: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train, test, folds = csv(ROOT / "train.csv"), csv(ROOT / "test.csv"), csv(ROOT / FOLDS)
    labels = split["labels"]
    require(len(labels) == 18 and len(set(labels)) == 18, "Official label order is invalid")
    require(split["train_sha256"] == PINNED_SHA256["train.csv"], "Split train hash changed")
    require(split["folds_sha256"] == PINNED_SHA256[FOLDS], "Split fold hash changed")
    require(len(train) == len(folds) == 2699 and len(test) == 3, "Official row counts changed")
    require(train.id.is_unique and folds.id.is_unique and test.id.is_unique, "Official IDs are not unique")
    require(set(train.model) == set(labels) and train.prompt_id.nunique() == 150, "Official groups or labels changed")
    for column in ("id", "prompt_id", "model"):
        require(train[column].tolist() == folds[column].tolist(), f"Fold {column} membership/order mismatch")
    require(train.text.map(text_sha).tolist() == folds.text_sha256.tolist(), "Fold text SHA mismatch")
    require(set(folds.fold) == {"0", "1", "2", "3", "4"}, "Unexpected fold IDs")
    require(bool(folds.groupby("prompt_id").fold.nunique().eq(1).all()), "Official prompt crosses folds")
    require(bool(folds.groupby("fold").prompt_id.nunique().eq(30).all()), "Expected 30 prompts in each fold")
    require(all(set(part.model) == set(labels) for _, part in folds.groupby("fold")), "A fold lacks a label")
    return train, test, {
        "rows": len(train), "prompts": 150, "labels": labels, "test_example_rows": len(test),
        "split_id": split["split_id"], "fold_rows": {str(k): int(v) for k, v in folds.fold.value_counts().sort_index().items()},
        "fold_prompts": {str(k): int(v) for k, v in folds.groupby("fold").prompt_id.nunique().items()},
        "ordered_id_prompt_model_text_sha_alignment": True, "prompt_disjoint_folds": True,
    }


def select_raw(labels: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw = csv(ROOT / RAW_DIR / "passages.csv")
    prompts = csv(ROOT / RAW_DIR / "prompts.csv")
    clean = csv(ROOT / RAW_DIR / "passages_clean.csv")
    require(len(raw) == 9290 and len(prompts) == 510, "Pinned raw or prompt row count changed")
    selected = raw.loc[raw.split.eq("new")].copy()
    selected.insert(0, "raw_row_number", selected.index.astype(int))
    selected.insert(0, "row_id", [f"prvs-raw-v6:{i}" for i in selected.index])
    require(len(selected) == 5400, "Expected every one of the 5400 raw/new rows")
    for column in ("row_id", "id", "call_id"):
        require(selected[column].is_unique and selected[column].astype(str).str.strip().ne("").all(), f"Invalid unique key: {column}")
    require(not selected.duplicated(["prompt_id", "model"]).any(), "Repeated prompt/model cell")
    require(set(selected.model) == set(labels), "Public label domain differs from official labels")
    require(selected.prompt_id.nunique() == 300, "Expected 300 public prompt groups")
    require(bool(selected.groupby("prompt_id").size().eq(18).all()), "A public prompt does not have 18 rows")
    require(bool(selected.groupby("prompt_id").model.nunique().eq(18).all()), "A public prompt lacks 18 distinct labels")
    require(bool(selected.model.value_counts().eq(300).all()), "Expected 300 rows per public label")
    require(selected.source_prompt_id.eq("").all(), "raw/new has an unexpected source_prompt_id")
    require(selected.text.str.strip().ne("").all(), "Empty raw text")
    require(selected.finish_reason_norm.eq("stop").all(), "A public response did not finish normally")
    for column in ("is_refusal", "is_truncated", "is_empty"):
        require(selected[column].str.lower().eq("false").all(), f"Invalid published response status: {column}")
    selected["text_sha256"] = selected.text.map(text_sha)
    selected["audit_text_hash"] = selected.text.map(audit_hash)
    require(selected.text_sha256.is_unique and selected.audit_text_hash.is_unique, "Duplicate public response text")
    bank = prompts.loc[prompts.split.eq("new")].copy()
    require(len(bank) == 300 and bank.prompt_id.is_unique, "Invalid new prompt bank")
    require(set(bank.prompt_id) == set(selected.prompt_id), "Public response/prompt bank coverage mismatch")
    require(bank.prompt.str.strip().ne("").all() and bank.prompt_body.str.strip().ne("").all(), "Empty public prompt template")
    require(bank.prompt.map(normalized).nunique() == 300, "Duplicate public prompt template")
    joined = selected.merge(bank[["prompt_id", "prompt", "prompt_body", "length_mode"]], on="prompt_id", how="left", validate="many_to_one")
    require(joined.length_mode.eq("target").all(), "Unexpected public prompt protocol")
    for row in joined.itertuples():
        target = float(row.target_words)
        require(math.isfinite(target) and target.is_integer(), f"Invalid target_words for {row.row_id}")
        require(row.prompt.replace("{N}", str(int(target))) == row.prompt_sent, f"Prompt template mismatch for {row.row_id}")
    regen_bodies = set(prompts.loc[prompts.split.eq("train_regen"), "prompt_body"].map(normalized))
    require(not set(bank.prompt_body.map(normalized)) & regen_bodies, "New prompt body exactly overlaps a reconstructed official prompt")
    new_clean = clean.loc[clean.split.eq("new")]
    require(len(new_clean) == 4158 and new_clean.id.is_unique, "Unexpected clean/new diagnostic membership")
    raw_text = selected.set_index("id").text.to_dict()
    require(all(raw_text.get(row.id) == row.text for row in new_clean.itertuples()), "Clean diagnostic text differs from raw")
    words = selected.text.str.split().str.len()
    funnel_pass = selected.passes_funnel.str.lower().eq("true")
    require(int(funnel_pass.sum()) == 5326 and int((~words.between(40, 250)).sum()) == 70, "Raw diagnostic counts changed")
    summary = {
        "selection": "passages.csv split == new; preserve original row order and exact text; no row filtering beyond split selection",
        "raw_source_rows": len(raw), "selected_rows": len(selected), "prompts": 300, "classes": 18,
        "rows_per_prompt": 18, "rows_per_class": 300, "class_counts": {label: int(selected.model.eq(label).sum()) for label in labels},
        "unique_row_id_id_call_id_prompt_model_and_text": True, "prompt_template_and_sent_text_match_rows": 5400,
        "source_prompt_id_nonempty_rows": 0, "new_vs_reconstructed_official_prompt_exact_overlap": 0,
        "raw_response_words_min": int(words.min()), "raw_response_words_max": int(words.max()),
        "outside_authors_40_250_word_range_retained": 70,
        "author_funnel_pass_retained": 5326, "author_funnel_fail_retained": 74,
        "author_funnel_fail_step_counts": {str(k): int(v) for k, v in selected.loc[~funnel_pass, "funnel_fail_step"].value_counts().items()},
        "clean_new_rows_for_diagnostic_only": 4158, "clean_or_funnel_used_for_selection": False,
        "punctuation_length_markdown_filter_used": False,
        "provider_route_counts": {str(k): int(v) for k, v in selected.provider_route.value_counts().sort_index().items()},
    }
    return selected, prompts, summary


def canonical_model(value: str) -> str:
    value = value.removesuffix("@default")
    if "/" in value:
        provider, value = value.split("/", 1)
        require(provider in {"openai", "google", "anthropic", "moonshotai", "qwen", "z-ai"}, "Unrecognized provider wrapper")
    return {"gpt-5.5-2026-04-23": "gpt-5.5", "qwen3.8-max-0902": "qwen3.8-max"}.get(value, value)


def verify_logs(selected: pd.DataFrame) -> dict:
    wanted = selected.set_index("call_id")
    all_ids, matched = set(), set()
    source_split_counts = Counter()
    for line in (ROOT / RAW_DIR / "provenance.jsonl").open(encoding="utf-8"):
        log = json.loads(line)
        call_id = log.get("call_id")
        require(isinstance(call_id, str) and call_id and call_id not in all_ids, "Missing or duplicate provenance call_id")
        all_ids.add(call_id)
        if log.get("split") == "new":
            source_split_counts[str(log.get("published_split"))] += 1
        if call_id not in wanted.index:
            continue
        matched.add(call_id)
        row = wanted.loc[call_id]
        require(log.get("status") == "ok" and log.get("published_split") == "new", f"Wrong publication/status for {row.row_id}")
        require(log.get("label") == row.model and log.get("prompt_id") == row.prompt_id, f"Log label/prompt mismatch for {row.row_id}")
        require(log.get("identity", {}).get("ok") is True, f"Log identity gate failed for {row.row_id}")
        body = log["request"]["body"]
        response = log["response"]["json"]
        route = log["route"]
        require(route == row.provider_route, f"Route mismatch for {row.row_id}")
        require(body.get("model") == row.requested_model_id == log.get("requested_model_id"), f"Request model mismatch for {row.row_id}")
        require(log.get("served_model_id") == row.served_model_id, f"Served model mismatch for {row.row_id}")
        if route == "google":
            request_text = body["contents"]
            require(isinstance(request_text, str), f"Unexpected Google request content for {row.row_id}")
            response_text = "".join(part.get("text", "") for part in response["candidates"][0]["content"]["parts"] if not part.get("thought", False))
        else:
            messages = body["messages"]
            require(len(messages) == 1 and messages[0].get("role") == "user", f"Unexpected message history for {row.row_id}")
            request_text = messages[0]["content"]
            if route == "anthropic":
                response_text = "".join(part.get("text", "") for part in response["content"] if part.get("type") == "text")
            else:
                require(route in {"openai", "openrouter", "kaggle_proxy"}, f"Unrecognized response route for {row.row_id}")
                response_text = response["choices"][0]["message"]["content"]
        require(request_text == row.prompt_sent, f"Request prompt mismatch for {row.row_id}")
        require(response_text == row.text == log.get("text"), f"Raw response text mismatch for {row.row_id}")
        raw_model = response.get("model") or response.get("model_version") or response.get("modelVersion")
        require(isinstance(raw_model, str) and canonical_model(raw_model) == row.model, f"Response model identity mismatch for {row.row_id}")
    require(len(all_ids) == 9291, "Unexpected provenance call count")
    require(matched == set(wanted.index), "Public response is missing its provenance call")
    return {
        "unique_log_calls": len(all_ids), "selected_calls_checked": len(matched), "checks_failed": 0,
        "request_prompt_exact_match_rows": len(matched), "response_and_envelope_text_exact_match_rows": len(matched),
        "request_response_model_identity_match_rows": len(matched),
        "join_rule": "CSV split=new selection joined to logs by call_id, not log.split filtering",
        "log_split_new_published_split_counts": dict(sorted(source_split_counts.items())),
        "limit": "Supplied-record internal consistency; no independent provider-weight authentication",
    }


def verify_risks_and_overlap(selected: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> tuple[dict, dict, dict]:
    flags = pd.read_parquet(ROOT / ROW_FLAGS)
    require(flags.row_id.is_unique, "EXP-008 row flags are not unique")
    selected_flags = flags.set_index("row_id").reindex(selected.row_id)
    require(not selected_flags["table"].isna().any(), "Missing EXP-008 public row flags")
    for field, expected in (("table", ["prvs-raw-v6"] * 5400), ("split", ["new"] * 5400),
                            ("original_id", selected.id.tolist()), ("model", selected.model.tolist()),
                            ("call_id", selected.call_id.tolist()), ("prompt_key", selected.prompt_id.tolist()),
                            ("text_hash", selected.audit_text_hash.tolist()), ("row_number", selected.raw_row_number.tolist())):
        require(selected_flags[field].tolist() == expected, f"EXP-008 row identity mismatch: {field}")
    counts = {}
    for column in RISK_COLUMNS:
        values = selected_flags[column]
        require(not values.isna().any() and values.isin([True, False]).all(), f"Missing/nonboolean risk flag: {column}")
        counts[column] = int(values.sum())
        require(counts[column] == 0, f"Known public data risk: {column}")
    for column in ("related_official_prompts", "exclude_validation_folds"):
        require(selected_flags[column].eq("").all(), f"Known official relation in {column}")
    for name, official in (("train", train), ("test_example", test)):
        count = int(selected.audit_text_hash.isin(official.text.map(audit_hash)).sum())
        counts[f"recomputed_exact_official_{name}"] = count
        require(count == 0, f"Public response exactly overlaps official {name}")
    main = pd.read_parquet(ROOT / MAIN_ROWS)
    require(main.row_id.is_unique and main.audit_text_hash.is_unique, "EXP-009 main rows are not unique")
    expected_origins = {"matched-v3": 24443, "official": 2699, "pool-v5": 96686}
    require(main.origin.value_counts().to_dict() == expected_origins, "EXP-009 projection row counts changed")
    joined = selected.merge(main[["audit_text_hash", "origin"]], on="audit_text_hash", how="left", validate="one_to_one")
    overlap_counts = joined.origin.fillna("absent").value_counts().to_dict()
    require(overlap_counts == {"matched-v3": 5119, "pool-v5": 257, "absent": 24}, "Raw/new baseline overlap changed")
    baseline_overlap = {
        "normalization": "NFKC + casefold + collapsed whitespace, the EXP-008 audit key",
        "matched_rows": 5119, "pool_rows": 257, "already_in_main_projection": 5376, "not_in_main_projection": 24,
        "scope": "EXP-009 frozen-baseline rules on EXP-008 pinned current files, not historical hidden-run membership",
        "new_data_claim_allowed": False,
    }
    public_main = main.loc[main.origin.ne("official")].merge(flags[["row_id", "linked_official_prompt", "missing_prompt_key"]], on="row_id", how="left", validate="one_to_one")
    require(not public_main.linked_official_prompt.isna().any(), "Original main public rows lack flags")
    known_linked = int(public_main.linked_official_prompt.sum())
    require(known_linked == 121, "Original main known official-prompt link count changed")
    original_gate = {
        "status": "NOT_MET", "original_recipe_executed": False,
        "recipe": "official=1.0 + matched=0.5 + pool=0.35, compare pool 0.175/0.35/0.70",
        "projection_rows": 123828, "public_projection_rows": len(public_main), "known_official_prompt_linked_rows_lower_bound": known_linked,
        "reasons": [
            "matched lacks source/prompt fields and the aggregate pool's complete raw lineage has not been recovered",
            "removing known prompt-linked rows does not undo pool-v5 cleaning based on full official-label punctuation statistics",
            "the replacement raw/new experiment changes sources and must not inherit the original-recipe or full-ensemble claim",
        ],
        "source_evidence": ["experiments/EXP-008-public-pool-audit/report.md", "experiments/EXP-009-baseline-data-lineage/report.md"],
    }
    return {"status": "PASS_KNOWN_CHECKS", "risk_counts": counts, "row_flags_rows_checked": 5400,
            "all_selected_rows_preserved": True, "strict_semantic_independence_proven": False}, baseline_overlap, original_gate


def audit_time() -> str:
    previous_times = []
    for path in (AUDIT_PATH, GATE_PATH):
        if path.exists():
            old = read_json(path)
            require(isinstance(old.get("observed_at_utc"), str), f"Existing output has no valid audit time: {path}")
            parsed = datetime.fromisoformat(old["observed_at_utc"])
            require(parsed.tzinfo is not None, f"Existing audit time is not timezone-aware: {path}")
            previous_times.append(old["observed_at_utc"])
    require(len(set(previous_times)) <= 1, "Existing outputs disagree on their audit time")
    return previous_times[0] if previous_times else datetime.now(timezone.utc).isoformat()


def persist_identical_or_new(outputs: dict[Path, bytes]) -> dict[str, str]:
    # Preflight the entire allowlist before creating any payload or directory.
    for path, expected in outputs.items():
        require(not path.exists() or (path.is_file() and path.read_bytes() == expected),
                f"Refusing to overwrite different existing output: {path.relative_to(ROOT)}")
    result = {}
    for path, expected in outputs.items():
        if path.exists():
            require(path.read_bytes() == expected, f"Output changed during preflight: {path}")
            result[path.relative_to(ROOT).as_posix()] = "REUSED_IDENTICAL"
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as stream:
                stream.write(expected)
        except FileExistsError:
            require(path.is_file() and path.read_bytes() == expected, f"Output appeared with different contents: {path}")
        require(path.read_bytes() == expected, f"Output write verification failed: {path}")
        result[path.relative_to(ROOT).as_posix()] = "CREATED_OR_IDENTICAL_RACE"
    return result


def main() -> None:
    sources, split = verify_sources()
    train, test, official = verify_official(split)
    selected, _, public_summary = select_raw(split["labels"])
    provenance = verify_logs(selected)
    risks, overlap, original_gate = verify_risks_and_overlap(selected, train, test)
    public = selected[["row_id", "raw_row_number", "id", "prompt_id", "model", "text", "call_id", "text_sha256"]]
    public_bytes = public.to_csv(index=False, lineterminator="\n").encode("utf-8")
    roundtrip = pd.read_csv(io.BytesIO(public_bytes), keep_default_na=False, dtype=str)
    require(roundtrip.columns.tolist() == public.columns.tolist(), "Public CSV column roundtrip failed")
    for column in public.columns:
        require(roundtrip[column].tolist() == public[column].astype(str).tolist(), f"Public CSV roundtrip failed: {column}")
    observed_at = audit_time()
    attribution = (
        "# Raw public data attribution\n\n"
        "prvsiyan (2026), prvsiyan's Extended Name That LLM Dataset, fixed Kaggle dataset version 6.\n\n"
        "Source: https://www.kaggle.com/datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset/versions/6\n\n"
        "The pinned source README states CC BY 4.0 for the generated passages and the author-written new prompts. "
        "This payload contains every one of the 5,400 passages with CSV split=new, preserving their original text and order. "
        "It includes no train_regen, proxy_v1, pilot_range, or new_kimi_nothink rows and no reconstructed competition prompt texts.\n\n"
        "The CSV row_id is prvs-raw-v6:<zero-based original passages.csv row number>; id and call_id retain the source identifiers. "
        "text_sha256 is SHA-256 of the exact UTF-8 text, without normalization. The source's clean/funnel, punctuation, word-range, "
        "and markdown filters are not used for selection.\n\n"
        "Intended evidence scope: isolated svc4 fixed-external-data development OOF. The generation protocol used official train length "
        "statistics, so this is not a strictly untouched validation source. Supplied API logs verify internal consistency only. "
        "The 5,376 texts already present in the EXP-009 main-svc4 projection must not be called new additions to ATLAS20.\n\n"
        "The source README and input_audit.json retain provenance and limitations. No training, competition submission, or publication "
        "was performed by this preparation script.\n"
    ).encode("utf-8")
    audit = {
        "schema_version": 1, "observed_at_utc": observed_at, "status": "PASS_FOR_SCOPED_DEVELOPMENT",
        "scope": "Local read-only-input preparation for EXP-013/014/015/016; no model fit or remote action",
        "source_ref": "prvsiyan/prvsiyans-extended-name-that-llm-dataset", "source_version": 6,
        "source_selection_file": RAW_DIR + "/passages.csv", "sources": sources,
        "official": official, "public": public_summary, "provenance": provenance, "risks": risks,
        "baseline_overlap": overlap, "original_three_source_recipe_gate": original_gate,
        "limitations": LIMITATIONS,
        "environment": {"python": platform.python_version(), "pandas": pd.__version__, "pyarrow": pyarrow.__version__},
        "outputs": {
            "public_csv": {"path": (PAYLOAD / "public.csv").relative_to(ROOT).as_posix(), "rows": 5400,
                           "bytes": len(public_bytes), "sha256": sha_bytes(public_bytes), "columns": public.columns.tolist()},
            "attribution": {"path": (PAYLOAD / "attribution.md").relative_to(ROOT).as_posix(), "sha256": sha_bytes(attribution)},
        },
        "output_policy": "Existing different bytes are rejected; same-byte reruns reuse the initial observed_at_utc after full revalidation",
    }
    audit_bytes = json_bytes(audit)
    gate = {
        "schema_version": 1, "observed_at_utc": observed_at, "experiment_id": "EXP-014",
        "applies_to": ["EXP-013", "EXP-014", "EXP-015", "EXP-016"],
        "status": "PASS_FOR_SCOPED_DEVELOPMENT", "training_executed_by_preparation": False,
        "input_audit": {"path": AUDIT_PATH.relative_to(ROOT).as_posix(), "sha256": sha_bytes(audit_bytes)},
        "public_csv": audit["outputs"]["public_csv"], "official": official,
        "raw_public": public_summary, "provenance": provenance, "risks": risks,
        "baseline_overlap": overlap, "original_three_source_recipe_gate": original_gate,
        "limitations": LIMITATIONS,
        "conclusion": "All 5400 original raw/new rows are retained. Proceed only with the separately named isolated svc4 raw-public comparison; the original matched+pool weight draft remains below its data gate.",
    }
    writes = persist_identical_or_new({PAYLOAD / "public.csv": public_bytes, AUDIT_PATH: audit_bytes,
                                      PAYLOAD / "attribution.md": attribution, GATE_PATH: json_bytes(gate)})
    print(json.dumps({"status": audit["status"], "rows": 5400, "prompts": 300, "classes": 18,
                      "public_csv_sha256": sha_bytes(public_bytes), "input_audit_sha256": sha_bytes(audit_bytes),
                      "original_recipe_gate": "NOT_MET", "outputs": writes}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (GateError, KeyError, ValueError, TypeError, OSError) as error:
        print(f"DATA_GATE_FAILED: {error}", file=sys.stderr)
        raise SystemExit(1) from error
