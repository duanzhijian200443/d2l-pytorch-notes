"""Audit fixed SRC-062 Dolly-only panels against the actual two training pools.

Only data/provenance files are read. No predictions, estimators, training,
network calls, or evaluation scores are loaded or produced. The selected raw
responses are copied unchanged; existing different outputs are never replaced.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "data/public-pools/enaimaan/ntl-openai-generated-responses/v3"
DOLLY = "databricks/databricks-dolly-15k"
PRVS = "prvsiyan/prvsiyans-extended-name-that-llm-dataset"
SEYIT = "seyitkaangunes/ntl-extra-api-passages"
LABELS = ["gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra"]
STYLE = ROOT / "experiments/EXP-010-012-stylegram-share/payload"
SVC = ROOT / "cloud/svc4-raw-public-r1/payload"
GATE = HERE / "metrics/data-gate.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def compact(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def csv(path):
    return pd.read_csv(path, keep_default_na=False)


def whitespace(text):
    return " ".join(str(text).split())


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def norm_hash(text):
    return sha_bytes(normalized(text).encode("utf-8"))


def track(path, receipt, expected=None, size=None):
    require(path.is_file(), "Missing input: " + str(path))
    digest = sha(path)
    require(expected is None or digest == expected, "Input hash mismatch: " + str(path))
    require(size is None or path.stat().st_size == size, "Input size mismatch: " + str(path))
    receipt[path.relative_to(ROOT).as_posix()] = {"sha256": digest, "bytes": path.stat().st_size}


def pinned_inputs():
    inputs = {}
    manifest_path = HERE / "download-manifest.json"
    track(manifest_path, inputs)
    manifest = read_json(manifest_path)
    require(manifest["dataset"]["ref"] == "enaimaan/ntl-openai-generated-responses" and manifest["dataset"]["version"] == 3, "Wrong external source/version")
    require(len(manifest["dataset"]["files"]) == 10, "Expected ten source files")
    names = set()
    for file in manifest["dataset"]["files"]:
        require(file["name"] not in names, "Duplicate source filename")
        names.add(file["name"])
        path = ROOT / file["local_path"]
        require(path.resolve().parent == SOURCE.resolve(), "Source manifest path escaped its version directory")
        track(path, inputs, file["sha256"], file["bytes"])
    sums = {}
    for line in (SOURCE / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})\s+\*?([^/\\]+)", line)
        require(match is not None and match[2] not in sums, "Invalid publisher checksum line")
        sums[match[2]] = match[1]
    require(set(sums) == names - {"SHA256SUMS"}, "Publisher checksums do not cover all nine other files")
    for name, digest in sums.items():
        require(sha(SOURCE / name) == digest, "Publisher checksum mismatch: " + name)
    for directory, filename in ((STYLE, "extra.csv"), (SVC, "public.csv")):
        track(directory / "protocol.json", inputs)
        protocol = read_json(directory / "protocol.json")
        track(directory / filename, inputs, protocol["input_hashes"][filename])
        track(ROOT / "train.csv", inputs, protocol["train_sha256"])
    old_manifest_path = ROOT / "experiments/EXP-008-public-pool-audit/download-manifest.json"
    track(old_manifest_path, inputs, "5f2da21401efada94e636fd2fb843013c7eff6ae1ed2d07a5a2771c3ed071cfa")
    old_files = {file["local_path"]: file for dataset in read_json(old_manifest_path)["datasets"] for file in dataset["files"]}
    for relative in (f"data/public-pools/{SEYIT}/v2/prompts.csv", f"data/public-pools/{PRVS}/v6/prompts.csv"):
        track(ROOT / relative, inputs, old_files[relative]["sha256"], old_files[relative]["bytes"])
    track(Path(__file__), inputs)
    return inputs


def training_exposure():
    extra, svc, official = csv(STYLE / "extra.csv"), csv(SVC / "public.csv"), csv(ROOT / "train.csv")
    require(len(extra) == 10073 and len(svc) == 5400 and len(official) == 2699, "Training row counts changed")
    require(extra.row_id.is_unique and svc.row_id.is_unique and official.id.is_unique, "Duplicate training row identifiers")
    seyit_bank = csv(ROOT / f"data/public-pools/{SEYIT}/v2/prompts.csv").set_index("prompt_id")
    prvs_bank = csv(ROOT / f"data/public-pools/{PRVS}/v6/prompts.csv").set_index("prompt_id")
    source_pairs = set()
    body_sets = {"stylegram": set(), "svc4": set(), "official_reconstructed": set()}
    training_keys = set(extra.canonical_prompt) | {"prvs:" + prompt for prompt in svc.prompt_id}
    require("" not in training_keys and len(training_keys) == 763, "Unexpected training prompt ancestry")
    for key in sorted(training_keys):
        if key.startswith("seyit:"):
            prompt = key.removeprefix("seyit:")
            body = seyit_bank.at[prompt, "prompt"]
            source_pairs.add((SEYIT, prompt))
            # These aliases are explicitly declared in SRC-062 attribution.
            for alias in ("illiadolenko/ntl-prompt-bank-v1", "willrice/name-that-llm-openrouter-harvest"):
                source_pairs.add((alias, prompt))
            source_pairs.add(("trnhip/ntl-api-gen", "ill:" + prompt))
        elif key.startswith("prvs:"):
            prompt = key.removeprefix("prvs:")
            body = prvs_bank.at[prompt, "prompt_body"]
            source_pairs.add((PRVS, prompt))
        else:
            raise ValueError("Unresolved actual training canonical prompt: " + key)
        if key in set(extra.canonical_prompt):
            body_sets["stylegram"].add(norm_hash(body))
        if key in {"prvs:" + p for p in svc.prompt_id}:
            body_sets["svc4"].add(norm_hash(body))
    regen = prvs_bank.loc[prvs_bank.split.eq("train_regen")]
    require(len(regen) == 150, "Expected 150 documented reconstructed official prompts")
    for prompt, row in regen.iterrows():
        source_pairs.add((PRVS, prompt))
        body_sets["official_reconstructed"].add(norm_hash(row.prompt_body))
    # Exact source-row aliases from actual training records are also preserved.
    origin_source = {"seyit-v2": SEYIT, "lucas-v7": "lucasboesen/ntl-free-proxy-generations",
                     "dophuchung-v1": "dophuchung/ntl-proxy-generations", "prvs-clean-v6": PRVS}
    for row in extra.itertuples():
        require(row.origin in origin_source, "Unknown Stylegram training source")
        prompt = row.prompt_id.removeprefix("pv_") if row.origin == "prvs-clean-v6" else row.prompt_id
        source_pairs.add((origin_source[row.origin], prompt))
    texts = {"stylegram": set(extra.text.map(norm_hash)), "svc4": set(svc.text.map(norm_hash)),
             "official": set(official.text.map(norm_hash))}
    return texts, body_sets, source_pairs, training_keys


def resolve_alias(value, aliases):
    seen = set()
    while value in aliases and aliases[value] != value:
        require(value not in seen, "Cycle in lineage aliases")
        seen.add(value)
        value = aliases[value]
    return value


class Components:
    def __init__(self, keys):
        self.parent = {key: key for key in keys}

    def root(self, key):
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def join(self, left, right):
        left, right = self.root(left), self.root(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def run():
    inputs = pinned_inputs()
    text_sets, body_sets, training_pairs, training_keys = training_exposure()
    data = csv(SOURCE / "train.csv")
    parquet = pd.read_parquet(SOURCE / "train.parquet")
    pd.testing.assert_frame_equal(data, parquet, check_dtype=False)
    require(len(data) == 9719 and data.id.is_unique and set(data.model) == set(LABELS), "External response population mismatch")
    require(data.text.str.strip().ne("").all(), "Empty response text")
    require(data.text.str.split().str.len().tolist() == data.n_words.tolist(), "Response word-count metadata mismatch")
    require(data.kind.value_counts().to_dict() == {"new": 5408, "gap": 4311}, "External kind counts changed")
    prompts = [json.loads(line) for line in (SOURCE / "prompts.jsonl").open(encoding="utf-8")]
    require(len(prompts) == 9719 and len({row["response_id"] for row in prompts}) == 9719, "Prompt join keys are not one-to-one")
    prompt_of = {row["response_id"]: row for row in prompts}
    require(set(prompt_of) == set(data.id), "Response/prompt population differs")
    aliases = read_json(SOURCE / "lineage-aliases.json")
    require(isinstance(aliases, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in aliases.items()), "Invalid alias map")
    for alias in aliases:
        resolve_alias(alias, aliases)
    data["canonical_group"] = data.cv_group.map(lambda key: resolve_alias(key, aliases))
    components = Components(set(data.canonical_group))
    ancestor_owner, body_owner = {}, {}
    row_records = []
    message_hashes, request_hashes = set(), set()
    for row in data.itertuples():
        prompt = prompt_of[row.id]
        require(prompt["provider_model"] == row.model and prompt["request_status"] == row.request_status and prompt["prompt_group"] == row.prompt_group, "Response/request metadata mismatch")
        require(prompt["reasoning"] == ("low" if row.model == "gpt-6-astra" else "none"), "Unexpected requested reasoning setting")
        sources = prompt["sources"]
        require(isinstance(sources, list) and sources and compact(sources) == compact(json.loads(row.prompt_sources)), "Prompt source records differ")
        pairs = set()
        for source in sources:
            require(all(isinstance(source.get(key), str) for key in ("source", "prompt_id", "url", "license")), "Malformed source provenance")
            pair = (source["source"], source["prompt_id"])
            pairs.add(pair)
            if pair in ancestor_owner:
                components.join(row.canonical_group, ancestor_owner[pair])
            ancestor_owner[pair] = row.canonical_group
        if row.prompt_hash in body_owner:
            components.join(row.canonical_group, body_owner[row.prompt_hash])
        body_owner[row.prompt_hash] = row.canonical_group
        messages = prompt["messages"]
        require(isinstance(messages, list) and messages and all(isinstance(message.get("content"), str) and isinstance(message.get("role"), str) for message in messages), "Malformed exact messages")
        message_hash = sha_bytes(compact(messages).encode("utf-8"))
        request_hash = sha_bytes(compact({"messages": messages, "provider_model": prompt["provider_model"], "reasoning": prompt["reasoning"]}).encode("utf-8"))
        message_hashes.add(message_hash); request_hashes.add(request_hash)
        body_hash = None
        if row.kind == "new":
            require([message["role"] for message in messages] == ["system", "user"], "New request role structure differs")
            target = prompt["target_words"]
            require(type(target) is int and target > 0, "Invalid requested word target")
            suffix = f"\n\nWrite about {target} words."
            require(messages[-1]["content"].endswith(suffix), "Cannot reconstruct new source body exactly")
            body = messages[-1]["content"][:-len(suffix)]
            require(body.strip(), "Empty new source body")
            require(sha_bytes(whitespace(body).encode("utf-8")) == row.prompt_hash, "Source-body hash mismatch after exact wrapper removal")
            body_hash = norm_hash(body)
        text_hash = norm_hash(row.text)
        reasons = []
        if row.kind != "new":
            reasons.append("kind_not_new")
        if not all(source["source"] == DOLLY for source in sources):
            reasons.append("not_dolly_only")
        if pairs & training_pairs or row.canonical_group in training_keys:
            reasons.append("known_training_prompt_ancestry")
        for label, hashes in text_sets.items():
            if text_hash in hashes:
                reasons.append("response_text_overlap_" + label)
        for label, hashes in body_sets.items():
            if body_hash is not None and body_hash in hashes:
                reasons.append("source_body_overlap_" + label)
        row_records.append({"id": row.id, "canonical_group": row.canonical_group, "kind": row.kind, "model": row.model,
                            "sources": sorted({source["source"] for source in sources}), "source_body_hash": row.prompt_hash,
                            "normalized_source_body_hash": body_hash, "normalized_text_hash": text_hash,
                            "messages_sha256": message_hash, "application_request_sha256": request_hash,
                            "reasons": reasons})
    component_members = defaultdict(set)
    component_training_risks = defaultdict(set)
    for record in row_records:
        component = components.root(record["canonical_group"])
        component_members[component].add(record["canonical_group"])
        component_training_risks[component].update(reason for reason in record["reasons"] if "overlap" in reason or reason == "known_training_prompt_ancestry")
    records_by_group = defaultdict(list)
    for record in row_records:
        records_by_group[record["canonical_group"]].append(record)
    excluded, selected_groups, group_reasons = [], set(), {}
    for group, records in sorted(records_by_group.items()):
        component = components.root(group)
        reasons = set().union(*(set(record["reasons"]) for record in records))
        reasons.update(component_training_risks[component])
        if len(component_members[component]) != 1:
            reasons.add("shared_source_ancestor_across_multiple_cv_groups")
        if len(records) != 4 or Counter(record["model"] for record in records) != Counter(LABELS):
            reasons.add("not_exactly_one_response_for_each_of_four_labels")
        group_reasons[group] = sorted(reasons)
        if not reasons:
            selected_groups.add(group)
        else:
            excluded.append({"canonical_group": group, "rows": len(records),
                             "kinds": "|".join(sorted({record["kind"] for record in records})),
                             "classes": len({record["model"] for record in records}),
                             "sources": "|".join(sorted(set().union(*(set(record["sources"]) for record in records)))),
                             "reasons": "|".join(sorted(reasons))})
    selected = data.loc[data.canonical_group.isin(selected_groups)].copy()
    selected = selected.sort_values(["canonical_group", "model", "id"])
    require(len(selected) > 0 and len(selected) == 4 * len(selected_groups), "No valid complete external four-class panels")
    require(set(selected.model) == set(LABELS) and selected.groupby("canonical_group").model.nunique().eq(4).all(), "Selected class coverage invalid")
    require(selected.text.map(norm_hash).is_unique, "Selected external texts are duplicated")
    selection_records = [record for record in row_records if record["canonical_group"] in selected_groups]
    require(all(not record["reasons"] for record in selection_records), "Selected response has a gate exclusion")
    evaluation = selected.rename(columns={"canonical_group": "prompt_id"})[["id", "prompt_id", "model", "text"]]
    evaluation_bytes = evaluation.to_csv(index=False, lineterminator="\n").encode("utf-8")
    exclusions_bytes = pd.DataFrame(excluded).to_csv(index=False, lineterminator="\n").encode("utf-8")
    observed = read_json(GATE)["observed_at_utc"] if GATE.exists() else datetime.now(timezone.utc).isoformat()
    limits = [
        "Independent means no exact normalized response/source-body match or known shared prompt ancestry with the two specified fitting pools and documented official reconstructions; semantic overlap was not exhaustively tested and the host's true prompts are unavailable.",
        "Only four GPT labels and Dolly-derived prompts are covered. This is not an 18-class confirmation or a representative official-distribution estimate.",
        "Application-supplied messages and publisher provider_model labels are internally consistent; no raw serving-provider response logs independently authenticate the underlying weights.",
        "The new kind is a collection panel label, not proof of new ancestry. Community-derived new panels and gap requests are excluded from this primary selection.",
        "Generated-response rights are recorded by the publisher as not independently established; compilation/wrapper CC BY 4.0 and Dolly prompt CC BY-SA 3.0 attribution are preserved. This is a provenance statement, not an approval workflow.",
        "Response generation used a candidate system instruction and reasoning none for Luna/Sol/Terra, low for Astra; settings are not established as matching competition/provider-community generation.",
        "This selection uses source/provenance and fixed panel completeness only, with no model predictions, scores, length/style filtering, alpha search, or model fitting. Prior source diagnostics are not a guarantee of a pristine never-inspected population.",
    ]
    reason_counts = Counter(reason for reasons in group_reasons.values() for reason in reasons)
    selected_record_ids = {record["id"] for record in selection_records}
    require(selected_record_ids == set(evaluation.id), "Evaluation membership differs from selected audit rows")
    intersection_counts = {reason: sum(reason in record["reasons"] for record in selection_records)
                           for reason in ["known_training_prompt_ancestry"] + ["response_text_overlap_" + key for key in text_sets] + ["source_body_overlap_" + key for key in body_sets]}
    gate = {
        "schema_version": 1, "observed_at_utc": observed, "status": "PASS_FOR_EXTERNAL_FOUR_CLASS_CHECK",
        "experiment_id": "EXP-026", "source": "enaimaan/ntl-openai-generated-responses", "source_version": 3,
        "selection_rule": "kind=new; every recorded prompt source is databricks/databricks-dolly-15k; exact complete four-label panel per canonical cv_group; no known training ancestry/body/text intersection; shared ancestor components kept intact",
        "inputs": [{"path": path, **receipt} for path, receipt in sorted(inputs.items())],
        "source_rows": 9719, "source_groups": int(data.canonical_group.nunique()),
        "rows": len(evaluation), "groups": len(selected_groups), "labels": LABELS,
        "class_counts": {label: int(evaluation.model.eq(label).sum()) for label in LABELS},
        "evaluation": {"path": (HERE / "evaluation.csv").relative_to(ROOT).as_posix(), "bytes": len(evaluation_bytes), "sha256": sha_bytes(evaluation_bytes)},
        "excluded_groups": {"path": (HERE / "metrics/excluded-groups.csv").relative_to(ROOT).as_posix(), "groups": len(excluded), "sha256": sha_bytes(exclusions_bytes), "reason_counts_nonexclusive": dict(sorted(reason_counts.items()))},
        "training_exposure": {"stylegram_extra_rows": 10073, "svc4_public_rows": 5400, "official_rows": 2699,
                              "known_public_prompt_ancestors": 763, "official_reconstructed_prompt_bodies": 150},
        "selected_intersections": intersection_counts,
        "hash_semantics": {"source_body_sha256": "publisher prompt_hash = SHA256(whitespace-collapsed source body), case preserved; exact new request suffix removed before verification",
                           "comparison_normalization": "NFKC + casefold + collapsed whitespace for both response and source-body exact matching",
                           "messages_sha256": "canonical JSON of ordered role/content messages; contains system wrapper and per-response word target",
                           "application_request_sha256": "canonical JSON of messages + provider_model + reasoning; not a full raw HTTP request or underlying response authentication"},
        "request_checks": {"one_to_one_response_prompt_rows": 9719, "provider_model_label_matches": 9719,
                           "csv_parquet_equal": True, "new_source_body_hash_matches": 5408,
                           "unique_messages_all_source": len(message_hashes), "unique_application_requests_all_source": len(request_hashes)},
        "alias_checks": {"entries": len(aliases), "nonidentity_entries": sum(key != value for key, value in aliases.items()),
                         "cycles": 0, "lineage_components": len(component_members),
                         "multi_cv_group_components": sum(len(members) > 1 for members in component_members.values()),
                         "note": "Identity-only alias entries do not prove absence of shared ancestors; source-ID and body connections are checked separately."},
        "publisher_response_license_counts": data.response_license.value_counts().to_dict(),
        "limits": limits, "scoring_performed": False, "training_performed": False, "model_predictions_read": False,
    }
    outputs = {HERE / "evaluation.csv": evaluation_bytes, HERE / "metrics/excluded-groups.csv": exclusions_bytes, GATE: json_bytes(gate)}
    for path, content in outputs.items():
        require(not path.exists() or path.read_bytes() == content, "Refusing different existing output: " + str(path))
    for path, receipt in inputs.items():
        require(sha(ROOT / path) == receipt["sha256"], "Input changed during audit: " + path)
    for path, content in outputs.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(content)
        require(path.read_bytes() == content, "Output verification failed: " + str(path))
    print(json.dumps({"status": gate["status"], "rows": gate["rows"], "groups": gate["groups"], "labels": gate["labels"],
                      "evaluation_sha256": gate["evaluation"]["sha256"], "gate_sha256": sha(GATE),
                      "excluded_groups": len(excluded), "selected_intersections": intersection_counts,
                      "scoring_performed": False}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        run()
    except (ValueError, KeyError, TypeError, OSError, AssertionError) as error:
        print("EXTERNAL_DATA_GATE_FAILED: " + str(error), file=sys.stderr)
        raise SystemExit(1) from error
