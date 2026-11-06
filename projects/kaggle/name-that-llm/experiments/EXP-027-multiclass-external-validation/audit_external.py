"""Freeze Earthmaker V1's admissible external diagnostic rows without scoring.

Rows retain original text. Task-body identity, rather than a batch prefix,
defines a prompt group. Any training/prior-evaluation intersection excludes
the whole body group; the unverified Gemini 3.7 route is excluded by label.
No predictor, training code, remote service, or EXP-025 artifact is accessed.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_REF = "earthmaker/ntl-own-prompt-generations"
SOURCE = ROOT / f"data/public-pools/{SOURCE_REF}/v1"
PREVIOUS = ROOT / "experiments/EXP-026-independent-blend-validation"
STYLE = ROOT / "experiments/EXP-010-012-stylegram-share/payload"
SVC = ROOT / "cloud/svc4-raw-public-r1/payload"
GATE = HERE / "metrics/data-gate.json"
EXCLUDED_MODEL = "gemini-3.7-flash"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def csv(path):
    return pd.read_csv(path, keep_default_na=False)


def track(path, inputs, expected=None, size=None):
    require(path.is_file(), "Missing input: " + str(path))
    digest = sha(path)
    require(expected is None or digest == expected, "Input hash mismatch: " + str(path))
    require(size is None or path.stat().st_size == size, "Input byte size mismatch: " + str(path))
    inputs[path.relative_to(ROOT).as_posix()] = {"path": path.relative_to(ROOT).as_posix(), "sha256": digest, "bytes": path.stat().st_size}


def canonical_request_model(slug):
    parts = slug.split("/", 1)
    if len(parts) == 2:
        require(parts[0] in {"anthropic", "z-ai", "moonshotai", "qwen"}, "Unknown OpenRouter request wrapper")
        slug = parts[1]
    return {"qwen3.8-max-0902": "qwen3.8-max"}.get(slug, slug)


def load_inputs():
    inputs = {}
    receipt = HERE / "download-manifest.json"
    track(receipt, inputs)
    download = read_json(receipt)["dataset"]
    require(download["ref"] == SOURCE_REF and download["version"] == 1, "Wrong external dataset/version")
    expected_names = {"README.md", "passages.csv", "prompts_pilot.json", "prompts_v22.json", "templates.json"}
    require({entry["name"] for entry in download["files"]} == expected_names and len(download["files"]) == 5, "Wrong external file inventory")
    for entry in download["files"]:
        path = ROOT / entry["local_path"]
        require(path.resolve().parent == SOURCE.resolve(), "Source path escaped the fixed version directory")
        track(path, inputs, entry["sha256"], entry["bytes"])
    previous_gate = PREVIOUS / "metrics/data-gate.json"
    track(previous_gate, inputs, "8b5d203368ccbefc6c0165e3556187514eb117e7aca6e768c05154768091509a")
    gate = read_json(previous_gate)
    require(gate["status"] == "PASS_FOR_EXTERNAL_FOUR_CLASS_CHECK", "Prior accepted evaluation gate changed")
    previous_inputs = {entry["path"]: entry for entry in gate["inputs"]}
    required = [
        "train.csv", "experiments/EXP-010-012-stylegram-share/payload/extra.csv",
        "experiments/EXP-010-012-stylegram-share/payload/protocol.json",
        "cloud/svc4-raw-public-r1/payload/public.csv", "cloud/svc4-raw-public-r1/payload/protocol.json",
        "data/public-pools/seyitkaangunes/ntl-extra-api-passages/v2/prompts.csv",
        "data/public-pools/prvsiyan/prvsiyans-extended-name-that-llm-dataset/v6/prompts.csv",
        "data/public-pools/enaimaan/ntl-openai-generated-responses/v3/prompts.jsonl",
        "data/public-pools/enaimaan/ntl-openai-generated-responses/v3/train.csv",
        "experiments/EXP-026-independent-blend-validation/audit_external.py",
    ]
    for relative in required:
        entry = previous_inputs[relative]
        track(ROOT / relative, inputs, entry["sha256"], entry["bytes"])
    track(PREVIOUS / "evaluation.csv", inputs, gate["evaluation"]["sha256"], gate["evaluation"]["bytes"])
    training_labels = read_json(STYLE / "protocol.json")["labels"]
    require(training_labels == read_json(SVC / "protocol.json")["labels"] and len(training_labels) == 18, "Training labels differ")
    # Reuse only the reviewed, hash-bound read-only training-exposure helper.
    spec = importlib.util.spec_from_file_location("accepted_external_training_exposure", PREVIOUS / "audit_external.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    text_sets, body_sets, training_pairs, training_keys = helper.training_exposure()
    prior_evaluation = csv(PREVIOUS / "evaluation.csv")
    enai = csv(ROOT / "data/public-pools/enaimaan/ntl-openai-generated-responses/v3/train.csv").set_index("id")
    prior_ids = set(prior_evaluation.id)
    prior_bodies, prior_pairs, prior_seen = set(), set(), set()
    prior_original_groups, prior_publisher_hashes = set(), set()
    prior_normalized_groups = defaultdict(set)
    jsonl = ROOT / "data/public-pools/enaimaan/ntl-openai-generated-responses/v3/prompts.jsonl"
    for line in jsonl.open(encoding="utf-8"):
        prompt = json.loads(line)
        if prompt["response_id"] not in prior_ids:
            continue
        require(prompt["response_id"] not in prior_seen, "Duplicate previous response metadata")
        prior_seen.add(prompt["response_id"])
        require(enai.at[prompt["response_id"], "kind"] == "new", "Prior evaluation membership is not the accepted new panel")
        suffix = f"\n\nWrite about {prompt['target_words']} words."
        user = prompt["messages"][-1]["content"]
        require(user.endswith(suffix), "Cannot recover previous source body")
        body = user[:-len(suffix)]
        require(sha_bytes(helper.whitespace(body).encode()) == enai.at[prompt["response_id"], "prompt_hash"], "Prior source-body hash differs")
        body_hash = helper.norm_hash(body)
        original_group = enai.at[prompt["response_id"], "cv_group"]
        prior_bodies.add(body_hash)
        prior_original_groups.add(original_group)
        prior_publisher_hashes.add(enai.at[prompt["response_id"], "prompt_hash"])
        prior_normalized_groups[body_hash].add(original_group)
        prior_pairs.update((source["source"], source["prompt_id"]) for source in prompt["sources"])
    require(prior_seen == prior_ids and len(prior_ids) == 5200
            and len(prior_original_groups) == 1300 and len(prior_publisher_hashes) == 1300,
            "Incomplete previous source linkage")
    prior_linkage = {"rows_checked": len(prior_ids), "original_cv_groups": len(prior_original_groups),
                     "publisher_body_hashes": len(prior_publisher_hashes),
                     "normalized_body_hashes_checked": len(prior_bodies),
                     "normalized_body_groups_merging_original_cv_groups": [
                         {"normalized_body_sha256": digest, "original_cv_groups": sorted(groups)}
                         for digest, groups in sorted(prior_normalized_groups.items()) if len(groups) > 1],
                     "previous_files_modified": False}
    text_sets["exp026_evaluation"] = set(prior_evaluation.text.map(helper.norm_hash))
    body_sets["exp026_evaluation"] = prior_bodies
    track(Path(__file__), inputs)
    return inputs, helper.norm_hash, text_sets, body_sets, training_pairs, training_keys, prior_pairs, training_labels, prior_linkage


def persist(outputs, previous_failed_gate=None):
    for path, content in outputs.items():
        permitted_recovery = (path == GATE and previous_failed_gate is not None
                              and path.is_file() and path.read_bytes() == previous_failed_gate)
        require(not path.exists() or (path.is_file() and path.read_bytes() == content)
                or permitted_recovery, "Refusing different existing output: " + str(path))
    for path, content in outputs.items():
        if path == GATE and previous_failed_gate is not None:
            require(path.read_bytes() == previous_failed_gate, "Failure record changed before explicit recovery")
            path.write_bytes(content)
        elif not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(content)
        require(path.read_bytes() == content, "Written output changed: " + str(path))


def run():
    inputs, norm_hash, text_sets, body_sets, training_pairs, training_keys, prior_pairs, training_labels, prior_linkage = load_inputs()
    labels = [label for label in training_labels if label != EXCLUDED_MODEL]
    require(len(labels) == 17, "Expected exactly 17 admitted evaluation labels")
    passages = csv(SOURCE / "passages.csv")
    required_columns = {"id", "batch", "prompt_id", "model", "text", "target_words", "template_id", "route", "request_slug", "generated_on"}
    require(set(passages.columns) == required_columns and len(passages) == 749 and passages.id.is_unique, "Source response schema/count/IDs differ")
    require(set(passages.model) == set(training_labels) and passages.text.str.strip().ne("").all(), "Missing source labels or blank responses")
    templates_raw = read_json(SOURCE / "templates.json")
    templates = {entry["template_id"]: entry["text"] for entry in templates_raw}
    require(set(templates) == {0, 1, 2} and len(templates_raw) == 3, "Request templates are not uniquely identified")
    bank, bank_rows = {}, []
    for batch, filename, expected in (("pilot_20261015", "prompts_pilot.json", 20), ("v22_20261022", "prompts_v22.json", 36)):
        prompts = read_json(SOURCE / filename)
        require(len(prompts) == expected and len({row["pid"] for row in prompts}) == expected, "Prompt bank IDs/counts differ")
        for prompt in prompts:
            require(isinstance(prompt["task"], str) and prompt["task"].strip(), "Missing task body")
            require(type(prompt["words"]) is int and prompt["words"] > 0 and prompt.get("tmpl", 0) in templates, "Invalid request template/target metadata")
            bank[batch, prompt["pid"]] = prompt
            bank_rows.append({"batch": batch, "prompt_id": prompt["pid"], "task_sha256": norm_hash(prompt["task"])})
    require(set(zip(passages.batch, passages.prompt_id)) == set(bank), "Published responses do not map onto the two exact batch/prompt banks")
    records = []
    group_hits = defaultdict(set)
    for number, row in enumerate(passages.itertuples()):
        prompt = bank[row.batch, row.prompt_id]
        template_id = prompt.get("tmpl", 0)
        require(int(row.target_words) == prompt["words"] and int(row.template_id) == template_id, "Published request target/template differs from its batch/prompt task")
        require(row.generated_on == ("2026-10-15" if row.batch == "pilot_20261015" else "2026-10-22"), "Published generation date differs from batch")
        require(row.route in {"kaggle_benchmarks_model_proxy", "openrouter"} and canonical_request_model(row.request_slug) == row.model, "Requested model/label mapping differs")
        task = prompt["task"]
        request = templates[template_id].format(words=prompt["words"], task=task)
        task_hash, request_hash, text_hash = norm_hash(task), norm_hash(request), norm_hash(row.text)
        group = "earthmaker-task:" + task_hash
        reasons = []
        for name, exposed in text_sets.items():
            if text_hash in exposed:
                reasons.append("response_overlap_" + name)
        for name, exposed in body_sets.items():
            if task_hash in exposed:
                reasons.append("task_body_overlap_" + name)
            if request_hash in exposed:
                reasons.append("rendered_request_overlap_" + name)
        if (SOURCE_REF, row.prompt_id) in training_pairs:
            reasons.append("known_training_source_prompt_ancestry")
        if (SOURCE_REF, row.prompt_id) in prior_pairs:
            reasons.append("known_exp026_source_prompt_ancestry")
        group_hits[group].update(reasons)
        records.append({"id": row.id, "source_row_number": number, "batch": row.batch,
                        "source_prompt_id": row.prompt_id, "canonical_prompt_id": group,
                        "model": row.model, "route": row.route, "request_slug": row.request_slug,
                        "template_id": template_id, "target_words": int(row.target_words),
                        "task_sha256": task_hash, "rendered_request_sha256": request_hash,
                        "text_sha256": sha_bytes(row.text.encode("utf-8")), "normalized_text_sha256": text_hash,
                        "direct_overlap_reasons": "|".join(sorted(reasons))})
    rows_by_group = defaultdict(list)
    for record in records:
        reasons = set(group_hits[record["canonical_prompt_id"]])
        if record["model"] == EXCLUDED_MODEL:
            reasons.add("excluded_unverified_gemini_3_7_route")
        record["selected"] = not reasons
        record["exclusion_reasons"] = "|".join(sorted(reasons))
        rows_by_group[record["canonical_prompt_id"]].append(record)
    group_records = []
    for group, rows in sorted(rows_by_group.items()):
        chosen = [row for row in rows if row["selected"]]
        group_records.append({"canonical_prompt_id": group, "task_sha256": rows[0]["task_sha256"],
                              "batch_prompt_members": json.dumps(sorted({(row["batch"], row["source_prompt_id"]) for row in rows})),
                              "batches": "|".join(sorted({row["batch"] for row in rows})),
                              "source_rows": len(rows), "selected_rows": len(chosen),
                              "selected_classes": len({row["model"] for row in chosen}),
                              "group_exclusion_reasons": "|".join(sorted(group_hits[group])),
                              "identity_excluded_rows": sum(row["model"] == EXCLUDED_MODEL for row in rows)})
    row_audit = pd.DataFrame(records)
    selected_audit = row_audit.loc[row_audit.selected].copy()
    selected = passages.iloc[selected_audit.source_row_number].copy()
    selected["prompt_id"] = selected_audit.canonical_prompt_id.to_numpy()
    evaluation = selected[["id", "prompt_id", "model", "text"]].sort_values(["prompt_id", "model", "id"]).reset_index(drop=True)
    require(len(evaluation) > 0 and set(evaluation.model) == set(labels), "Eligible data do not cover all 17 admitted labels; evaluation is blocked")
    require(evaluation.id.is_unique and not evaluation.duplicated(["prompt_id", "model"]).any(), "Repeated model/response in a canonical task group requires an explicit evaluation policy")
    require(evaluation.text.map(norm_hash).is_unique, "Duplicate selected response texts require an explicit evaluation policy")
    require(not evaluation.model.eq(EXCLUDED_MODEL).any(), "Unverified Gemini 3.7 survived selection")
    require(all(not group_hits[group] for group in evaluation.prompt_id.unique()), "A selected group has a known exposure")
    require(len(evaluation) == 729 and evaluation.prompt_id.nunique() == 56,
            "The predeclared 729-row/56-task population changed or has an overlap; evaluation is blocked")
    evaluation_bytes = evaluation.to_csv(index=False, lineterminator="\n").encode("utf-8")
    row_bytes = row_audit.to_csv(index=False, lineterminator="\n").encode("utf-8")
    group_bytes = pd.DataFrame(group_records).to_csv(index=False, lineterminator="\n").encode("utf-8")
    observed = datetime.now(timezone.utc).isoformat()
    previous_failed_gate, previous_attempts = None, []
    if GATE.exists():
        existing_bytes = GATE.read_bytes()
        existing = json.loads(existing_bytes)
        if existing.get("status") == "FAILED_DATA_GATE":
            require(existing.get("reason") == "Incomplete previous source linkage"
                    and existing.get("evaluation_written") is False,
                    "Existing failed gate needs explicit review before recovery")
            previous_failed_gate = existing_bytes
            previous_attempts = [{"sha256": sha_bytes(existing_bytes), "record": existing,
                                  "resolution": "Original 1300 publisher groups were verified separately from 1299 normalized bodies; one case-only cross-group pair is preserved in exp026_crosscheck and its merged body is included in all overlap checks."}]
        else:
            require(existing.get("status") == "PASS_FOR_EXTERNAL_SEVENTEEN_CLASS_CHECK", "Unknown existing gate state")
            observed = existing["observed_at_utc"]
            previous_attempts = existing.get("previous_attempts", [])
    class_counts = {label: int(evaluation.model.eq(label).sum()) for label in labels}
    group_classes = evaluation.groupby("prompt_id").model.nunique()
    excluded_counts = Counter(reason for row in records for reason in row["exclusion_reasons"].split("|") if reason)
    limits = [
        "Coverage is 17 admitted labels with unequal counts and incomplete prompt groups; Gemini 3.7 is excluded because the publisher could not verify the proxy serving version. Predictions must still use all 18 original training classes.",
        "The publisher filtered 31 of 360 pilot responses by official 8-word overlap and 12 of 432 v22 responses by truncation/format/content-overlap checks. The released population is upstream-selected using official text and is not a strictly untouched holdout.",
        "Isolation here covers actual training response bytes after documented normalization, known source/prompt ancestry, and exact normalized task/request bodies including documented official reconstructions. True organizer prompts are unknown; semantic topic overlap and paraphrases were not exhaustively excluded.",
        "Request slugs match published labels after the explicit Qwen dated alias; no raw serving-provider response logs are included. This establishes published-request consistency, not independent authentication of weights or settings.",
        "Pilot's five OpenRouter-only labels have small support. Batch, source route, template and topic distributions differ from official data; no length, style, score or prediction-based selection was performed by this audit.",
        "No overlap with the accepted EXP-026 rows, declared source ancestors or recovered source bodies was found under these checks. This does not make the two sources identically distributed.",
        "The full ATLAS20 stack contains other runtime and pretrained members with incomplete training-corpus lineage. Passing this two-member gate does not establish independence from every full-stack member or full-ensemble improvement.",
    ]
    gate = {
        "schema_version": 1, "experiment_id": "EXP-027", "observed_at_utc": observed,
        "status": "PASS_FOR_EXTERNAL_SEVENTEEN_CLASS_CHECK", "source": SOURCE_REF, "source_version": 1,
        "selection_rule": "Keep all published rows except gemini-3.7-flash; resolve each response by exact batch+prompt bank and rendered template; group identical NFKC/casefold/whitespace-normalized task bodies across batches; exclude every row of any group with a training/known-official/EXP026 ancestry, response or body intersection; no score/style/length filtering",
        "inputs": [inputs[path] for path in sorted(inputs)],
        "rows": len(evaluation), "groups": int(evaluation.prompt_id.nunique()), "labels": labels, "class_counts": class_counts,
        "evaluation": {"path": (HERE / "evaluation.csv").relative_to(ROOT).as_posix(), "rows": len(evaluation),
                       "groups": int(evaluation.prompt_id.nunique()), "labels": labels, "class_counts": class_counts,
                       "bytes": len(evaluation_bytes), "sha256": sha_bytes(evaluation_bytes)},
        "row_audit": {"path": (HERE / "metrics/row-audit.csv").relative_to(ROOT).as_posix(), "rows": len(records), "sha256": sha_bytes(row_bytes)},
        "group_audit": {"path": (HERE / "metrics/group-audit.csv").relative_to(ROOT).as_posix(), "groups": len(group_records), "sha256": sha_bytes(group_bytes)},
        "source_population": {"rows": 749, "batch_prompt_pairs": len(bank), "normalized_task_bodies": len(rows_by_group),
                              "cross_batch_same_task_groups": sum("|" in row["batches"] for row in group_records),
                              "batch_counts": passages.batch.value_counts().to_dict(), "source_labels": training_labels},
        "selection": {"excluded_rows": int((~row_audit.selected).sum()), "excluded_reason_counts_nonexclusive": dict(excluded_counts),
                      "groups_excluded_for_overlap": sum(bool(reasons) for reasons in group_hits.values()),
                      "selected_groups_by_class_coverage": {str(key): int(value) for key, value in group_classes.value_counts().sort_index().items()},
                      "complete_seventeen_class_groups": int(group_classes.eq(17).sum()),
                      "selected_batch_counts": selected.batch.value_counts().to_dict(),
                      "selected_route_counts": selected.route.value_counts().to_dict()},
        "training_exposure": {"official_rows": 2699, "stylegram_extra_rows": 10073, "svc4_public_rows": 5400,
                              "known_public_prompt_ancestors": len(training_keys), "known_official_reconstructed_bodies": 150},
        "exp026_crosscheck": {**prior_linkage,
                              "source_ancestor_matches": sum("known_exp026_source_prompt_ancestry" in row["direct_overlap_reasons"] for row in records),
                              "response_matches": sum("response_overlap_exp026_evaluation" in row["direct_overlap_reasons"] for row in records),
                              "task_body_matches": sum("task_body_overlap_exp026_evaluation" in row["direct_overlap_reasons"] for row in records)},
        "hash_semantics": {"task_sha256": "NFKC + casefold + collapsed whitespace of the JSON task field; excludes template and target words",
                           "rendered_request_sha256": "same normalization of the exactly reconstructed template.format(words,task); not an archived raw request or serving response",
                           "text_sha256": "SHA-256 of unchanged UTF-8 response text",
                           "canonical_prompt_id": "earthmaker-task: plus the complete task_sha256; no batch-prefix independence assumption"},
        "upstream_official_eightgram_filter": {"publisher_declared": True, "recomputed_or_reapplied_here": False},
        "identity_checks": {"requested_slug_label_consistent_rows": 749, "excluded_unverified_model": EXCLUDED_MODEL,
                            "excluded_identity_rows": int(passages.model.eq(EXCLUDED_MODEL).sum()), "raw_serving_response_logs_available": False},
        "previous_attempts": previous_attempts,
        "limits": limits, "predictions_read": False, "training_performed": False, "scoring_performed": False,
    }
    for entry in inputs.values():
        require(sha(ROOT / entry["path"]) == entry["sha256"], "Input changed during audit: " + entry["path"])
    persist({HERE / "evaluation.csv": evaluation_bytes, HERE / "metrics/row-audit.csv": row_bytes,
             HERE / "metrics/group-audit.csv": group_bytes, GATE: json_bytes(gate)}, previous_failed_gate)
    print(json.dumps({"status": gate["status"], "evaluation": gate["evaluation"], "gate_sha256": sha(GATE),
                      "exclusions": gate["selection"], "exp026_crosscheck": gate["exp026_crosscheck"]}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        run()
    except (ValueError, KeyError, TypeError, OSError, AssertionError) as error:
        if not GATE.exists() and not (HERE / "evaluation.csv").exists():
            GATE.parent.mkdir(parents=True, exist_ok=True)
            GATE.write_bytes(json_bytes({"schema_version": 1, "experiment_id": "EXP-027", "observed_at_utc": datetime.now(timezone.utc).isoformat(),
                                         "status": "FAILED_DATA_GATE", "reason": str(error), "evaluation_written": False,
                                         "training_performed": False, "scoring_performed": False}))
        print("EXTERNAL_DATA_GATE_FAILED: " + str(error), file=sys.stderr)
        raise SystemExit(1) from error
