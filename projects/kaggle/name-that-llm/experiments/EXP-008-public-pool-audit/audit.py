# /// script
# requires-python = ">=3.13"
# dependencies = ["pandas==3.0.6", "pyarrow==23.0.1"]
# ///
"""Read-only audit of fixed-version public pools. Run: uv run audit.py.

Original files are never changed. Outputs are row flags and candidate indices,
not a trained model or a claim that community labels are ground truth.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = ROOT / "data/public-pools"
OUT = HERE / "outputs"
SPECS = [
    ("seyit-v2", "seyitkaangunes/ntl-extra-api-passages", 2, "ntl_extra_passages_claude.csv"),
    ("abdullah-v2", "abdullah1afify/ntl-generations", 2, "ntl_generations.csv"),
    ("dophuchung-v1", "dophuchung/ntl-proxy-generations", 1, "ntl_proxy_generations.csv"),
    ("lucas-v7", "lucasboesen/ntl-free-proxy-generations", 7, "gen_passages.csv"),
    ("prvs-raw-v6", "prvsiyan/prvsiyans-extended-name-that-llm-dataset", 6, "passages.csv"),
    ("prvs-clean-v6", "prvsiyan/prvsiyans-extended-name-that-llm-dataset", 6, "passages_clean.csv"),
    ("matched-v3", "dhruvtibarewal/nlm-matched-clean", 3, "nlm_matched_clean.parquet"),
    ("pool-v2", "dhruvtibarewal/nlm-matched-clean", 3, "nlm_pool_v2.parquet"),
    ("pool-v5", "dhruvtibarewal/nlm-matched-clean", 3, "nlm_pool_v5.parquet"),
]
PILOT_TABLES = {"seyit-v2", "abdullah-v2", "dophuchung-v1", "lucas-v7", "prvs-clean-v6"}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def fingerprint(text):
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def column(df, name, default=""):
    return df[name].fillna(default).astype(str) if name in df else pd.Series(default, index=df.index)


def bool_column(df, name):
    return column(df, name).str.lower().eq("true")


def canonical_model(value):
    # Only explicitly observed provider wrappers and documented dated IDs.
    value = str(value).strip()
    if "/" in value:
        provider, value = value.split("/", 1)
        if provider not in {"openai", "google", "anthropic", "moonshotai", "qwen", "z-ai"}:
            return "UNKNOWN_PROVIDER:" + provider
    if value.endswith("@default"):
        value = value[:-8]
    return {"gpt-5.5-2026-04-23": "gpt-5.5", "qwen3.8-max-0902": "qwen3.8-max"}.get(value, value)


def prompt_identity(table, row):
    explicit = row["source_prompt_id"]
    if row["split"] == "train_regen" and re.fullmatch(r"p\d{3}", explicit):
        return "official:" + explicit
    key = row["prompt_key"]
    match = re.search(r"(?:^|:)tr_(p\d{3})(?:$|:)", key)
    if match:
        return "official:" + match.group(1)
    if table in {"seyit-v2", "abdullah-v2", "dophuchung-v1", "lucas-v7"}:
        if re.fullmatch(r"g\d{4}", key):
            return "seyit:" + key
        if table == "lucas-v7" and key.startswith("g:"):
            return "seyit:" + key[2:]
    if table.startswith("prvs-") and key:
        # n### is reused across pilot/default/no-thinking variants, intentionally.
        return "prvs:" + key
    if key.startswith("seyit:"):
        return key
    return ""


def verify_inputs():
    manifest = json.loads((HERE / "download-manifest.json").read_text(encoding="utf-8"))
    for dataset in manifest["datasets"]:
        for file in dataset["files"]:
            path = ROOT / file["local_path"]
            if path.stat().st_size != file["bytes"] or sha(path) != file["sha256"]:
                raise ValueError(f"Downloaded input changed: {file['local_path']}")
    split = json.loads((ROOT / "validation/prompt-group-v1/manifest.json").read_text(encoding="utf-8"))
    if sha(ROOT / "train.csv") != split["train_sha256"]:
        raise ValueError("Official train changed")
    # Preserve provenance of the historical expectations, not only newly observed hashes.
    old = json.loads((ROOT / "cloud/atlas20-drive/drive-inputs.json").read_text(encoding="utf-8"))
    checks = []
    for entry in old.get("small_file_hashes", []):
        for dataset in manifest["datasets"]:
            prefix = "datasets/" + dataset["ref"] + "/"
            if entry["path"].startswith(prefix):
                name = entry["path"][len(prefix):]
                file = next((f for f in dataset["files"] if f["name"] == name), None)
                if file:
                    passed = file["sha256"] == entry["sha256"]
                    checks.append({"path": file["local_path"], "historical_sha256": entry["sha256"], "match": passed})
                    if not passed:
                        raise ValueError("Historical expected hash mismatch: " + name)
    dump(OUT / "input-integrity.json", {"status": "PASS", "files": sum(len(d["files"]) for d in manifest["datasets"]),
                                        "historical_hash_checks": checks, "train_sha256": split["train_sha256"]})
    return split["labels"]


def load_tables(labels):
    tables, profiles = [], []
    for name, ref, version, filename in SPECS:
        path = RAW / ref / f"v{version}" / filename
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)
        label_field = "model" if "model" in df else "label"
        text = column(df, "text")
        pkey = column(df, "prompt_key") if "prompt_key" in df else column(df, "prompt_id")
        rows = pd.DataFrame({
            "table": name, "row_number": range(len(df)), "row_id": [f"{name}:{i}" for i in range(len(df))],
            "original_id": column(df, "id"), "model": column(df, label_field), "text": text,
            "text_hash": text.map(fingerprint), "prompt_key": pkey,
            "source_prompt_id": column(df, "source_prompt_id"), "split": column(df, "split"),
            "src": column(df, "src", ref), "route": column(df, "provider_route"),
            "setting": column(df, "setting") if "setting" in df else column(df, "reasoning_setting"),
            "call_id": column(df, "call_id"), "requested_model_id": column(df, "requested_model_id"),
            "served_model_id": column(df, "served_model_id") if "served_model_id" in df else column(df, "model_seen"),
            "requested_slug": column(df, "model_slug"), "finish_reason": column(df, "finish_reason"),
            "reported_refusal": bool_column(df, "is_refusal"), "reported_truncated": bool_column(df, "is_truncated"),
            "author_passes_funnel": bool_column(df, "passes_funnel"),
        })
        rows["canonical_prompt"] = rows.apply(lambda r: prompt_identity(name, r), axis=1)
        rows["blank_text"] = text.str.strip().eq("")
        rows["unknown_label"] = ~rows.model.isin(labels)
        rows["words"] = text.str.count(r"\S+")
        rows["em_dash"] = text.str.contains("—", regex=False)
        rows["curly_quote"] = text.str.contains("[‘’“”]", regex=True)
        rows["markdown_flag"] = text.str.contains(r"(?m)^\s*(?:#{1,6}\s|[-*]\s|\d+\.\s)|```", regex=True)
        rows["error_or_refusal_screen"] = text.str.match(
            r"(?is)^\s*(?:error\s*[:{]|\{\s*\"error\"|rate limit|quota exceeded|as an ai(?: language model)?[, ]|i(?:'m| am) sorry[, ]|i cannot (?:assist|help|comply)|i can't (?:assist|help|comply))")
        rows["missing_prompt_key"] = pkey.str.strip().eq("") | pkey.str.endswith(":")
        reported_id = rows.served_model_id.where(rows.served_model_id.ne(""), rows.requested_slug)
        rows["identity_mismatch"] = reported_id.ne("") & reported_id.map(canonical_model).ne(rows.model)
        profile = {
            "table": name, "ref": ref, "version": version, "file": filename, "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha(path), "rows": len(df), "columns": df.columns.tolist(),
            "null_counts": {str(k): int(v) for k, v in df.isna().sum().items()},
            "class_counts": {str(k): int(v) for k, v in rows.model.value_counts().items()},
            "classes": int(rows.model.nunique()), "missing_labels": sorted(set(labels) - set(rows.model)),
            "split_counts": {str(k): int(v) for k, v in rows.split.value_counts().items()},
            "missing_prompt_key_rows": int(rows.missing_prompt_key.sum()),
            "blank_text_rows": int(rows.blank_text.sum()), "unknown_label_rows": int(rows.unknown_label.sum()),
            "exact_duplicate_extra_rows": int(text.duplicated().sum()),
            "normalized_duplicate_extra_rows": int(rows.text_hash.duplicated().sum()),
            "identity_id_mismatch_rows": int(rows.identity_mismatch.sum()),
            "id_column_duplicate_extra_rows": int(df.id.duplicated().sum()) if "id" in df else None,
            "prompt_label_repeat_extra_rows": int(rows.loc[~rows.missing_prompt_key].duplicated(["prompt_key", "model"]).sum()),
            "word_quantiles": {str(k): float(v) for k, v in rows.words.quantile([0,.05,.5,.95,1]).items()},
        }
        profiles.append(profile)
        tables.append(rows)
        print(f"Profiled {name}: {len(df):,} rows / {rows.model.nunique()} labels", flush=True)
    return pd.concat(tables, ignore_index=True), profiles


def audit_provenance(allrows):
    path = RAW / "prvsiyan/prvsiyans-extended-name-that-llm-dataset/v6/provenance.jsonl"
    calls = {}
    duplicate_ids = []
    statuses = Counter()
    routes = Counter()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            x = json.loads(line)
            call_id = x.get("call_id")
            if call_id in calls:
                duplicate_ids.append(call_id)
            body = (x.get("response") or {}).get("json") or {}
            raw_model = body.get("model") or body.get("model_version") or body.get("modelVersion") if isinstance(body, dict) else None
            calls[call_id] = {
                "label": x.get("label"), "text_hash": fingerprint(x.get("text") or ""),
                "requested_model_id": x.get("requested_model_id"), "served_model_id": x.get("served_model_id"),
                "raw_response_model_id": raw_model, "identity_ok": (x.get("identity") or {}).get("ok"),
                "route": x.get("route"), "status": x.get("status"),
            }
            statuses[str(x.get("status"))] += 1
            routes[str(x.get("route"))] += 1
    checks = []
    for row in allrows.loc[allrows.table.eq("prvs-raw-v6")].itertuples():
        source = calls.get(row.call_id)
        errors = []
        if not source:
            errors.append("missing_call")
        else:
            for field in ("label", "text_hash", "requested_model_id", "served_model_id"):
                actual = row.model if field == "label" else getattr(row, field)
                if str(source[field]) != str(actual):
                    errors.append("csv_vs_log_" + field)
            if source["identity_ok"] is not True:
                errors.append("identity_not_ok")
            if canonical_model(source["raw_response_model_id"]) != row.model:
                errors.append("raw_response_model_mismatch_or_missing")
        checks.append({"row_id": row.row_id, "call_id": row.call_id, "model": row.model,
                       "route": row.route, "text_hash": row.text_hash, "checks_pass": not errors,
                       "issues": "|".join(errors)})
    check_df = pd.DataFrame(checks)
    check_df.to_csv(OUT / "prvs_provenance_checks.csv", index=False)
    result = {"log_records": sum(statuses.values()), "unique_call_ids": len(calls), "duplicate_call_ids": duplicate_ids,
              "status_counts": dict(statuses), "route_counts": dict(routes), "published_rows": len(check_df),
              "published_rows_matching_raw_logs": int(check_df.checks_pass.sum()),
              "issue_counts": check_df.loc[~check_df.checks_pass, "issues"].value_counts().to_dict(),
              "limit": "Checks supplied logs for internal consistency. A proxy echoes its routing slug; even a matching response does not independently authenticate the underlying weights."}
    dump(OUT / "provenance-audit.json", result)
    return set(check_df.loc[~check_df.checks_pass, "text_hash"])


def shingles(text):
    tokens = re.findall(r"\w+", normalize(text))
    return {tuple(tokens[i:i+8]) for i in range(len(tokens)-7)}


def char_counts(text):
    text = normalize(text)
    return Counter(text[i:i+5] for i in range(max(0, len(text)-4)))


def cosine(left, right):
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    return dot / math.sqrt(sum(v*v for v in left.values()) * sum(v*v for v in right.values()))


def screen_official(allrows, official):
    """Exhaustive shared 8-word shingle screen; char cosine only on candidates.

    This is not an exhaustive char-cosine or semantic near-duplicate search.
    """
    index = defaultdict(set)
    official_chars = {}
    for i, row in official.iterrows():
        for shingle in shingles(row.text):
            index[shingle].add(i)
    unique = allrows.drop_duplicates("text_hash")[["text_hash", "text"]]
    candidates = []
    for n, row in enumerate(unique.itertuples(index=False)):
        matches = Counter()
        for shingle in shingles(row.text):
            matches.update(index.get(shingle, ()))
        selected = [(i, count) for i, count in matches.items() if count >= 3]
        if selected:
            chars = char_counts(row.text)
            for i, count in selected:
                if i not in official_chars:
                    official_chars[i] = char_counts(official.loc[i, "text"])
                value = cosine(chars, official_chars[i])
                candidates.append({"text_hash": row.text_hash, "official_id": official.loc[i, "id"],
                    "official_prompt_id": official.loc[i, "prompt_id"], "official_model": official.loc[i, "model"],
                    "official_dataset": official.loc[i, "official_dataset"], "shared_unique_8grams": count,
                    "char5_tf_cosine": value, "high_similarity": value >= .90})
        if n and n % 20000 == 0:
            print(f"Official-text overlap screen: {n:,}/{len(unique):,} unique external texts", flush=True)
    columns = ["text_hash", "official_id", "official_prompt_id", "official_model", "official_dataset",
               "shared_unique_8grams", "char5_tf_cosine", "high_similarity"]
    found = pd.DataFrame(candidates, columns=columns)
    found.sort_values("char5_tf_cosine", ascending=False).to_csv(OUT / "official_overlap_candidates.csv", index=False)
    return found


def build_flags(allrows, official, provenance_issues):
    hashes = allrows.groupby("text_hash").agg(rows=("row_id", "size"), labels=("model", "nunique"), tables=("table", "nunique"))
    conflicts = set(hashes.index[hashes.labels.gt(1)])
    allrows["label_conflict"] = allrows.text_hash.isin(conflicts)
    allrows["provenance_issue"] = allrows.text_hash.isin(provenance_issues)
    allrows["cluster_rows"] = allrows.text_hash.map(hashes.rows)
    allrows["cluster_tables"] = allrows.text_hash.map(hashes.tables)
    allrows["official_exact_match"] = allrows.text_hash.isin(set(official.text.map(fingerprint)))
    seeds = defaultdict(set)
    for row in allrows.loc[allrows.canonical_prompt.ne("")].itertuples():
        seeds[row.text_hash].add(row.canonical_prompt)
    # Matching text propagates explicit official-prompt lineage even when a pool lost its metadata.
    for row in official.itertuples():
        if row.prompt_id:
            seeds[fingerprint(row.text)].add("official:" + row.prompt_id)
    overlap_candidates = screen_official(allrows, official)
    near = overlap_candidates.loc[overlap_candidates.high_similarity]
    for row in near.itertuples():
        if row.official_prompt_id:
            seeds[row.text_hash].add("official:" + row.official_prompt_id)
    allrows["official_high_similarity"] = allrows.text_hash.isin(set(near.text_hash))
    allrows["official_shingle_review"] = allrows.text_hash.isin(set(overlap_candidates.text_hash))
    allrows["prompt_links"] = allrows.text_hash.map(lambda h: "|".join(sorted(seeds.get(h, ()))))
    folds = pd.read_csv(ROOT / "validation/prompt-group-v1/folds.csv")
    fold_of = folds.groupby("prompt_id").fold.first().to_dict()
    official_ids = {h: {s.split(":", 1)[1] for s in links if s.startswith("official:")}
                    for h, links in seeds.items()}
    allrows["related_official_prompts"] = allrows.text_hash.map(lambda h: "|".join(sorted(official_ids.get(h, ()))))
    allrows["exclude_validation_folds"] = allrows.text_hash.map(
        lambda h: "|".join(map(str, sorted({int(fold_of[p]) for p in official_ids.get(h, ()) if p in fold_of}))))
    allrows["linked_official_prompt"] = allrows.related_official_prompts.ne("")
    for name in ["blank_text", "unknown_label", "label_conflict", "identity_mismatch", "provenance_issue",
                 "official_exact_match", "official_high_similarity", "linked_official_prompt", "reported_refusal",
                 "reported_truncated", "error_or_refusal_screen"]:
        if name not in allrows:
            raise ValueError("Missing screening field: " + name)
    reasons = ["blank_text", "unknown_label", "label_conflict", "identity_mismatch", "provenance_issue",
               "official_exact_match", "official_high_similarity", "linked_official_prompt", "reported_refusal",
               "reported_truncated", "error_or_refusal_screen"]
    allrows["hold_for_review"] = allrows[reasons].any(axis=1)
    # Initial pilot prioritizes traceable direct sources; aggregate versions remain separate controls.
    eligible = allrows.table.isin(PILOT_TABLES) & ~allrows.hold_for_review & ~allrows.official_shingle_review
    candidates = allrows.loc[eligible].copy()
    preference = {"prvs-clean-v6": 0, "seyit-v2": 1, "abdullah-v2": 2, "dophuchung-v1": 3, "lucas-v7": 4}
    candidates["_priority"] = candidates.table.map(preference)
    candidates = candidates.sort_values(["_priority", "row_number"]).drop_duplicates("text_hash")
    allrows["initial_pilot_selected"] = allrows.row_id.isin(set(candidates.row_id))
    candidates.drop(columns=["text", "_priority"]).to_csv(OUT / "initial_candidate_index.csv", index=False)
    allrows.drop(columns="text").to_parquet(OUT / "row_flags.parquet", index=False)
    allrows.loc[allrows.label_conflict, ["row_id", "table", "model", "text_hash", "text"]].to_csv(OUT / "label_conflict_rows.csv", index=False)
    allrows.loc[allrows.linked_official_prompt, ["row_id", "table", "model", "prompt_key", "text_hash", "related_official_prompts", "exclude_validation_folds"]].to_csv(OUT / "official_prompt_linked_rows.csv", index=False)
    allrows.loc[allrows.error_or_refusal_screen | allrows.reported_refusal, ["row_id", "model", "text"]].to_csv(OUT / "refusal_screen_examples.csv", index=False)
    return allrows, {"physical_rows_across_nine_views": len(allrows), "unique_normalized_texts": len(hashes),
                     "extra_occurrences_across_views": len(allrows) - len(hashes),
                     "cross_table_text_clusters": int(hashes.tables.gt(1).sum()), "conflicting_label_clusters": len(conflicts),
                     "initial_candidate_unique_rows": len(candidates),
                     "candidate_class_counts": {k: int(v) for k, v in candidates.model.value_counts().items()},
                     "near_screen": {"method": "All external unique texts vs official train+public test; at least 3 distinct shared 8-word shingles, then char5 TF cosine >=0.90",
                                     "shared_shingle_candidate_pairs": int(len(overlap_candidates)),
                                     "shared_shingle_unique_external_texts": int(overlap_candidates.text_hash.nunique()),
                                     "high_similarity_pairs": int(len(near)),
                                     "scope": "Not exhaustive cosine or semantic search; unflagged does not prove no overlap"}}


def aggregate(allrows, profiles, official):
    lookup = {p["table"]: p for p in profiles}
    flags = ["official_exact_match", "official_high_similarity", "official_shingle_review", "linked_official_prompt",
             "label_conflict", "provenance_issue", "reported_refusal", "reported_truncated", "error_or_refusal_screen",
             "hold_for_review", "initial_pilot_selected"]
    for name, group in allrows.groupby("table", sort=False):
        lookup[name].update({field + "_rows": int(group[field].sum()) for field in flags})
        lookup[name]["known_canonical_prompt_rows"] = int(group.prompt_links.ne("").sum())
        lookup[name]["within_table_unique_texts"] = int(group.text_hash.nunique())
    pd.DataFrame([{k: v for k, v in p.items() if not isinstance(v, (dict, list))} for p in profiles]).to_csv(OUT / "table_summary.csv", index=False)
    dump(OUT / "profiles.json", profiles)
    sets = {name: set(group.text_hash) for name, group in allrows.groupby("table", sort=False)}
    overlap = pd.DataFrame({a: {b: len(sets[a] & sets[b]) for b in sets} for a in sets})
    overlap.to_csv(OUT / "cross_table_exact_overlap.csv")
    allrows.groupby(["table", "model"]).size().unstack(fill_value=0).to_csv(OUT / "class_counts.csv")
    allrows.groupby(["table", "src", "model"]).size().rename("rows").to_csv(OUT / "source_class_counts.csv")
    allrows.groupby(["table", "setting", "model"]).size().rename("rows").to_csv(OUT / "settings_class_counts.csv")
    group_counts = allrows.loc[~allrows.missing_prompt_key].groupby(["table", "prompt_key"]).agg(rows=("row_id", "size"), classes=("model", "nunique"))
    group_counts.to_csv(OUT / "prompt_group_coverage.csv")
    styles = allrows[["table", "model", "words", "em_dash", "curly_quote", "markdown_flag"]].copy()
    train = official.loc[official.official_dataset.eq("train")].copy()
    ref = pd.DataFrame({"table": "official-train", "model": train.model,
                        "words": train.text.str.count(r"\S+"), "em_dash": train.text.str.contains("—", regex=False),
                        "curly_quote": train.text.str.contains("[‘’“”]", regex=True),
                        "markdown_flag": train.text.str.contains(r"(?m)^\s*(?:#{1,6}\s|[-*]\s|\d+\.\s)|```", regex=True)})
    styles = pd.concat([ref, styles], ignore_index=True).groupby(["table", "model"]).agg(
        rows=("words", "size"), median_words=("words", "median"), em_dash_rate=("em_dash", "mean"),
        curly_quote_rate=("curly_quote", "mean"), markdown_screen_rate=("markdown_flag", "mean"))
    styles.to_csv(OUT / "style_by_class.csv")
    # Coverage claims in a README can lag actual rows; record observed counts.
    group_counts.groupby("table").agg(prompts=("classes", "size"), full18=("classes", lambda s: int(s.eq(18).sum())),
                                      min_classes=("classes", "min"), median_classes=("classes", "median"), max_classes=("classes", "max")).to_csv(OUT / "prompt_completeness.csv")


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    labels = verify_inputs()
    allrows, profiles = load_tables(labels)
    provenance_issues = audit_provenance(allrows)
    official = pd.read_csv(ROOT / "train.csv").assign(official_dataset="train")
    test = pd.read_csv(ROOT / "test.csv").assign(prompt_id="", model="", official_dataset="public_test_example")
    official = pd.concat([official, test], ignore_index=True)
    allrows, summary = build_flags(allrows, official, provenance_issues)
    aggregate(allrows, profiles, official)
    summary.update({"experiment": "EXP-008", "observed_at": datetime.now(timezone.utc).isoformat(),
                    "status": "COMPLETE", "tables": len(SPECS), "datasets": 6,
                    "run_seconds": round(time.monotonic() - started, 3),
                    "analysis_sha256": sha(Path(__file__)), "download_manifest_sha256": sha(HERE / "download-manifest.json"),
                    "versions": {"python": platform.python_version(), "pandas": pd.__version__, "pyarrow": pyarrow.__version__},
                    "cost": {"gpu_seconds": 0, "api_usd": 0, "colab_cu": 0},
                    "scope": "Data audit and conservative candidate index only; no training, scoring, redistribution or model identity authentication",
                    "limitations": ["Only six pinned datasets; aggregate pool contains other upstream sources not independently downloaded",
                        "Normalized exact duplicates are exhaustive across these tables; semantic duplicates and undocumented prompt reuse can remain",
                        "Proxy routing identifiers cannot prove backend model version",
                        "Aggregated pools may have been selected with full official-label statistics; do not call their downstream CV strictly untouched",
                        "An empty prompt key is unknown, not a single real shared prompt",
                        "Candidate index is for a controlled ablation, not a claim of accuracy improvement"]})
    dump(OUT / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
