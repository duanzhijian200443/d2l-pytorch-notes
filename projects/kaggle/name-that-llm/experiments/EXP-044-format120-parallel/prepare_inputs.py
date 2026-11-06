"""Prepare only frozen same-60-prompt old raw120 and derived clean120.
No estimators, remote calls or source mutation. Fail closed on any mismatch.
"""
from __future__ import annotations
import csv
import hashlib
import importlib.util
import io
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VIEW = ROOT / "experiments/EXP-043-codex-format-view"
PAYLOAD = ROOT / "cloud/format120-044/payload"
METRICS = HERE / "metrics"
OLD_PATH = "experiments/EXP-023-codex-subscription-pilot-r1/collection/records.jsonl"
NEW_PATH = "experiments/EXP-040-codex300-concurrency2/collection/combined-records.jsonl"
EXPECTED = {
 OLD_PATH: "cb903279b3727b31ce2125b89c5ce8f530d769a984f49c2cb86e4fbefb1e9722",
 NEW_PATH: "2d6c691016a6ae6f789d5c6a625a207a3ec89e5093bf804c5052b11c90a6125b",
 "experiments/EXP-043-codex-format-view/manifest.json": "98dbb285863a5e9d806b1399b96a76e09554249b549455c1fa75777fd0565deb",
 "experiments/EXP-043-codex-format-view/metrics/independent-review.json": "c2a93d6b27d7a5e7e0217049c8f2387e90ed0ee84f9a2a95f6fb1fc1b740c793",
 "cloud/luna-terra120-042/payload/new120.csv": "cbb0f2227f7748d9172f04d23c1653403103a192dbb7503d897153d24bb00acd",
 "cloud/luna-terra120-042/payload/protocol.json": "35e546b80c1e0ef0a3f489575260bbb5b64eca4e4b6ef553e359ead9f6135a1b",
 "train.csv": "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707",
 "experiments/EXP-010-012-stylegram-share/payload/extra.csv": "1e9125fd49b19084842332ede84ea65bc0174ee1bb28c8d48acc8ddbd5e36c0e",
 "validation/prompt-group-v1/folds.csv": "e8ba1e3712852168dc57f56960d4f3e00be612d639ade8a7dcb0f0d8335ce551",
 "data/public-pools/seyitkaangunes/ntl-extra-api-passages/v2/prompts.csv": "25df24d4e4ca0f8242bd365cc9ea8039f0056f3eae78fa3dd186bbae5a608395",
 "data/public-pools/prvsiyan/prvsiyans-extended-name-that-llm-dataset/v6/prompts.csv": "58dabf5d31bb3fd76ff49dda3c4603970aece7553a5afe78d2e2dc4463b98acd",
}
BASE_FIELDS = ["row_id", "origin", "prompt_id", "model", "text", "base_prompt",
               "text_sha256", "job_key", "split", "source_protocol"]
FIELDS = BASE_FIELDS + ["raw_text_sha256"]
LABELS = {"gpt-5.6-luna", "gpt-5.6-terra"}
SOURCE = "codex_subscription_original_format_pair_v1"

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()

def sha(path):
    return sha_bytes(path.read_bytes())

def text_sha(text):
    return sha_bytes(text.encode("utf-8"))

def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())

def nh(text):
    return text_sha(normalize(text))

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded

def csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")

def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode("utf-8")

def checked_write(path, content):
    require(not path.exists() or path.read_bytes() == content,
            "Refusing to overwrite different output: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("xb") as stream:
            stream.write(content)
    require(path.read_bytes() == content, "Output readback differs")

def run():
    inputs = {}
    def track(path, expected=None, size=None):
        rel = path.relative_to(ROOT).as_posix()
        digest = sha(path)
        require(expected is None or digest == expected, "Input hash mismatch: " + rel)
        require(size is None or path.stat().st_size == size, "Input size mismatch: " + rel)
        inputs[rel] = {"sha256": digest, "bytes": path.stat().st_size}
    for name, expected in EXPECTED.items():
        track(ROOT/name, expected)
    manifest = read_json(VIEW/"manifest.json")
    review = read_json(VIEW/"metrics/independent-review.json")
    require(manifest["freeze_status"] == "DERIVED_VIEW_COMPLETE_NOT_TRAINED", "EXP043 freeze changed")
    require(review["status"] == "PASS_FROZEN_LOCAL_DERIVED_VIEW" and not review["findings"], "EXP043 review not accepted")
    require(review["manifest_sha256"] == sha(VIEW/"manifest.json"), "EXP043 reviewed manifest mismatch")
    entries = [manifest["config"], manifest["metrics"]] + manifest["inputs"] + manifest["code"] + manifest["outputs"]
    for entry in entries:
        track(ROOT/entry["path"], entry["sha256"], entry["bytes"])
    track(Path(__file__))
    old = read_jsonl(ROOT/OLD_PATH)
    aligned = read_jsonl(ROOT/NEW_PATH)
    derived = read_jsonl(VIEW/"derived/train.jsonl")
    derived_held = read_jsonl(VIEW/"derived/holdout.jsonl")
    mapping = read_json(VIEW/"derived/selections.json")["matched_with_aligned120"]
    new120 = read_csv(ROOT/"cloud/luna-terra120-042/payload/new120.csv")
    require(list(new120[0]) == BASE_FIELDS, "EXP042 schema changed")
    require(len(old) == 1000 and len(derived) == 800 and len(derived_held) == 200, "Frozen populations changed")
    require(len(mapping) == 120 and len(new120) == 120, "Selection population changed")
    old_by = {r["job_key"]: r for r in old}
    new_by = {r["job_key"]: r for r in aligned}
    derived_by = {r["job_key"]: r for r in derived}
    require(len(old_by) == 1000 and len(new_by) == 300 and len(derived_by) == 800, "Duplicate source jobs")
    require({r["aligned_job_key"] for r in mapping} == {r["job_key"] for r in new120}, "EXP042 population differs")
    require(len({r["job_key"] for r in mapping}) == 120, "Repeated selected old job")
    transform = module("frozen_exp043_transform", VIEW/"transform.py").transform
    raw, clean = [], []
    changed = 0
    for item in mapping:
        r, d, n = old_by[item["job_key"]], derived_by[item["job_key"]], new_by[item["aligned_job_key"]]
        require(r["split"] == d["split"] == n["split"] == "train", "Holdout row selected")
        require(r["model_requested"] == d["model_requested"] == n["model_requested"] == item["model_requested"] and r["model_requested"] in LABELS, "Label mapping mismatch")
        require(r["prompt_group"] == d["prompt_group"] == n["prompt_group"] == item["prompt_group"], "Prompt mapping mismatch")
        require(r["prompt"] == d["prompt"] == n["base_prompt"], "Base body mismatch")
        require(r["job_key"] != n["job_key"], "Aligned job masquerading as old source")
        require(text_sha(r["text"]) == r["text_sha256"] == r["raw_final_sha256"] == d["raw_text_sha256"], "Old source text hash mismatch")
        result = transform(r["text"])
        require(not result["ambiguous_double_star_runs"], "Ambiguous transform")
        require(result["text"] == d["text"] and text_sha(d["text"]) == d["text_sha256"], "Frozen derived transform mismatch")
        row = dict(zip(FIELDS, ["codex:"+r["job_key"], SOURCE, "codex:"+r["prompt_group"],
                   r["model_requested"], r["text"], r["prompt"], r["text_sha256"],
                   r["job_key"], "train", SOURCE, r["text_sha256"]]))
        raw.append(row)
        clean.append({**row, "text": d["text"], "text_sha256": d["text_sha256"]})
        changed += r["text"] != d["text"]
    require(Counter(r["model"] for r in raw) == {m:60 for m in LABELS}, "Selected classes unbalanced")
    require(len({r["prompt_id"] for r in raw}) == len({nh(r["base_prompt"]) for r in raw}) == 60, "Expected 60 original body groups")
    require(all(n == 2 for n in Counter(r["prompt_id"] for r in raw).values()), "Each original prompt must have two rows")
    require(len({(r["prompt_id"], r["model"]) for r in raw}) == 120, "Repeated prompt/model cell")
    for a,b in zip(raw,clean):
        require({k:v for k,v in a.items() if k not in ("text","text_sha256")} == {k:v for k,v in b.items() if k not in ("text","text_sha256")}, "Paired metadata differs")
    held = [r for r in old if r["split"] == "holdout"]
    holdout_hashes = sorted({nh(r["prompt"]) for r in held})
    require(len(held) == 200 and len(holdout_hashes) == 50, "Expected 50 reserved old groups")
    require({r["job_key"] for r in derived_held} == {r["job_key"] for r in held}, "Derived holdout identity changed")
    require({nh(r["base_prompt"]) for r in raw}.isdisjoint(holdout_hashes), "Holdout task crossed into train")
    for r in aligned:
        o = next(x for x in old if x["prompt_group"] == r["prompt_group"] and x["model_requested"] == r["model_requested"])
        require(o["prompt"] == r["base_prompt"] and o["split"] == r["split"], "Cross-protocol group split mismatch")
    train = read_csv(ROOT/"train.csv")
    extra = read_csv(ROOT/"experiments/EXP-010-012-stylegram-share/payload/extra.csv")
    require(len(train) == 2699 and len(extra) == 10073, "Baseline row counts changed")
    seyit = {r["prompt_id"]: r["prompt"] for r in read_csv(ROOT/"data/public-pools/seyitkaangunes/ntl-extra-api-passages/v2/prompts.csv")}
    prvs = {r["prompt_id"]: r for r in read_csv(ROOT/"data/public-pools/prvsiyan/prvsiyans-extended-name-that-llm-dataset/v6/prompts.csv")}
    public_bodies = []
    for key in sorted({r["canonical_prompt"] for r in extra}):
        require(key.startswith(("seyit:","prvs:")), "Unresolved public ancestry")
        public_bodies.append(seyit[key[6:]] if key.startswith("seyit:") else prvs[key[5:]]["prompt_body"])
    official_bodies = [r["prompt_body"] for r in prvs.values() if r["split"] == "train_regen"]
    require(len(public_bodies) == 694 and len(official_bodies) == 150, "Baseline prompt exposure counts changed")
    refs = {"official":train, "stylegram_public":extra, "old_holdout":held, "derived_holdout":derived_held}
    checks = {}
    for view, rows in (("raw120",raw),("clean120",clean)):
        checks[view] = {}
        for mode, digest in (("exact",text_sha),("normalized",nh)):
            row_hashes = [digest(r["text"]) for r in rows]
            require(len(set(row_hashes)) == 120, view+" internal "+mode+" duplicates")
            checks[view]["internal_"+mode+"_duplicates"] = 0
            for name, refs_rows in refs.items():
                count = len(set(row_hashes) & {digest(r["text"]) for r in refs_rows})
                checks[view][name+"_"+mode+"_response_matches"] = count
                require(count == 0, view+" overlaps "+name+" by "+mode+" response")
        for name,bodies in (("public",public_bodies),("official_reconstructed",official_bodies)):
            for mode,digest in (("exact",text_sha),("normalized",nh)):
                count = len({digest(r["base_prompt"]) for r in rows} & {digest(t) for t in bodies})
                checks[view][name+"_"+mode+"_prompt_matches"] = count
                require(count == 0, "Known task overlap: "+name)
    cross = 0
    for a in raw:
        for b in clean:
            if a["job_key"] != b["job_key"] and nh(a["text"]) == nh(b["text"]):
                cross += 1
    require(cross == 0, "Cross-view different-job normalized collision")
    # Verify frozen original inputs again before publishing any permitted output.
    for relative, entry in inputs.items():
        require(sha(ROOT/relative) == entry["sha256"], "Input changed during preparation")
    raw_bytes, clean_bytes = csv_bytes(raw), csv_bytes(clean)
    report = {
        "status":"PASS_KNOWN_EXACT_FIXED_RAW_CLEAN_120",
        "inputs":inputs,
        "counts":{"rows_per_arm":120,"models":{"gpt-5.6-luna":60,"gpt-5.6-terra":60},
                  "original_prompt_groups":60,"text_changed_rows":changed,
                  "reserved_old_rows":200,"reserved_prompt_groups":50},
        "selection":"EXP043 selections.matched_with_aligned120 in original list order; old job keys only",
        "schema":FIELDS,
        "metadata_identical_except":["text","text_sha256"],
        "origin_and_source_protocol":SOURCE,
        "views":{"raw120":"Actual unchanged EXP023 old text; not EXP042 aligned generation",
                 "clean120":"Exact EXP043 derived text; only accepted paired strong markers and line separators transformed"},
        "holdout_prompt_group_hashes":holdout_hashes,
        "holdout_hash_normalization":"SHA256 UTF8(NFKC + casefold + collapse whitespace) of original base prompt",
        "checks":checks,
        "different_job_cross_view_normalized_collisions":cross,
        "weights":"No weights encoded or recalculated here. Both arms require identical EXP042 fixed_weights using original extra10073 counts; identical row order/labels ensure identical per-fold row weights. No post-append normalization.",
        "outputs":{
            "raw120.csv":{"path":(PAYLOAD/"raw120.csv").relative_to(ROOT).as_posix(),"sha256":sha_bytes(raw_bytes),"bytes":len(raw_bytes)},
            "clean120.csv":{"path":(PAYLOAD/"clean120.csv").relative_to(ROOT).as_posix(),"sha256":sha_bytes(clean_bytes),"bytes":len(clean_bytes)}},
        "limits":["Known/exact prompt and response checks only; true official prompts unknown and reconstructed prompts approximate.",
                  "Inherited EXP042 lexical review found cp0049/version-control and cp0029/tides related to EXP027 tasks; EXP027 is not independent evaluation here.",
                  "Old requested model identity is not independent provider-snapshot authentication.",
                  "Clean view retains wording and other formatting; not a new-protocol generation or semantic quality certification.",
                  "Only120 permitted train rows copied into each CSV. Source800, all holdout text and aligned-new120 text are not copied into these payloads."],
        "training":False,"remote_operations":False}
    markdown = """# EXP044 data preparation gate

PASS_KNOWN_EXACT_FIXED_RAW_CLEAN_120.

Both inputs contain the frozen EXP043 same60-prompt old120 selection, in the original selection-list order. Raw uses actual EXP023 text and old job keys. Clean equals the accepted EXP043 transform exactly. All metadata, IDs, labels, prompts and raw hashes are identical across arms; only text and text_sha256 may differ.

There are60 prompts and60 rows per model (Luna/Terra); %s rows change text. No source rows were dropped or remapped. Both arms inherit identical EXP042 per-fold weights, with the old10073 public counts frozen.

The50 old reserved prompt hashes are stored in data-review.json for runner configuration. Exact and normalized response overlap with official2699/public10073/old and derived holdouts is zero in both views; exact and normalized known public/150 reconstructed official prompt intersections are zero. Internal duplicates and different-job cross-view normalized collisions are zero.

All frozen EXP043 manifest entries and independent acceptance, old source hash and EXP042 reference schema/population were checked before output. Only the two120-row CSVs are exported. No800-row pool or holdout prose is copied. This gate does not establish exhaustive semantic isolation, provider identity, or classifier utility. EXP027 related topic groups remain unsuitable for an independent-validation claim.

Preparation does not train or touch remote services. Full hashes and checks accompany this report.
""" % changed
    checked_write(PAYLOAD/"raw120.csv",raw_bytes)
    checked_write(PAYLOAD/"clean120.csv",clean_bytes)
    checked_write(METRICS/"data-review.json",json_bytes(report))
    checked_write(METRICS/"data-review.md",markdown.encode("utf-8"))
    require(read_csv(PAYLOAD/"raw120.csv") == raw and read_csv(PAYLOAD/"clean120.csv") == clean, "CSV roundtrip mismatch")
    print(json.dumps({"status":report["status"],"counts":report["counts"],"outputs":report["outputs"]},ensure_ascii=False))

if __name__ == "__main__":
    run()
