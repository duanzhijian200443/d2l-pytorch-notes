# /// script
# requires-python = ">=3.13"
# dependencies = ["pandas==3.0.6", "pyarrow==23.0.1"]
# ///
"""Project fixed baseline data filters onto EXP-008 inputs; never fit a model.

Only the reviewed loader / weighting function definitions are executed in an
isolated namespace. Never import or run infer.py or comp_stack.py as modules.
"""
import ast
import hashlib
import json
import re
import types
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "outputs"
RAW = ROOT / "data/public-pools"
SOURCES = [
    ("seyit-v2", "seyitkaangunes/ntl-extra-api-passages/v2/ntl_extra_passages_claude.csv", "model", None),
    ("lucas-v7", "lucasboesen/ntl-free-proxy-generations/v7/gen_passages.csv", "model", None),
    ("abdullah-v2", "abdullah1afify/ntl-generations/v2/ntl_generations.csv", "label", "finish_reason"),
    ("dophuchung-v1", "dophuchung/ntl-proxy-generations/v1/ntl_proxy_generations.csv", "label", None),
    ("prvs-clean-v6", "prvsiyan/prvsiyans-extended-name-that-llm-dataset/v6/passages_clean.csv", "model", None),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main_key(value):
    # infer.py _norm_text_key: case PRESERVED.
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value or ""))).strip()


def comp_text(value):
    return re.sub(r"\s+", " ", re.sub(r"^\*\*(.*)\*\*$", r"\1", str(value).strip())).strip()


def audit_hash(text):
    key = " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def extract_function(file, name, namespace):
    tree = ast.parse(file.read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(functions) == 1
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(file), "exec"), namespace)
    return namespace[name]


def run():
    OUT.mkdir(exist_ok=True)
    baseline = json.loads((ROOT / "tracking/records/experiments/EXP-004.json").read_text(encoding="utf-8"))
    notebook = ROOT / baseline["source"]["local_path"]
    assert sha(notebook) == baseline["source"]["sha256"]
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    for cell, name in [(6, "infer.py"), (8, "comp_stack.py")]:
        assert (HERE / name).read_text(encoding="utf-8") == "".join(nb["cells"][cell]["source"]).split("\n", 1)[1]
    spec = json.loads((ROOT / "experiments/EXP-008-public-pool-audit/download-manifest.json").read_text(encoding="utf-8"))
    files = {f["local_path"]: f for dataset in spec["datasets"] for f in dataset["files"]}
    selected = [RAW / entry[1] for entry in SOURCES] + [RAW / "dhruvtibarewal/nlm-matched-clean/v3" / name for name in ["nlm_matched_clean.parquet", "nlm_pool_v5.parquet"]]
    for file in selected:
        assert sha(file) == files[file.relative_to(ROOT).as_posix()]["sha256"]
    train = pd.read_csv(ROOT / "train.csv")
    split = json.loads((ROOT / "validation/prompt-group-v1/manifest.json").read_text(encoding="utf-8"))
    assert sha(ROOT / "train.csv") == split["train_sha256"]
    labels = split["labels"]
    official = train[["text", "model"]].assign(origin="official", row_id=[f"official:{i}" for i in range(len(train))], source_weight=1.0)
    matched = pd.read_parquet(selected[-2]).assign(origin="matched-v3")
    matched["row_id"] = [f"matched-v3:{i}" for i in range(len(matched))]
    pool = pd.read_parquet(selected[-1]).assign(origin="pool-v5")
    pool["row_id"] = [f"pool-v5:{i}" for i in range(len(pool))]
    actual_key = extract_function(HERE / "infer.py", "_norm_text_key", {"re": re})
    assert [actual_key(t) for t in matched.text] == [main_key(t) for t in matched.text]
    seen = set(official.text.map(main_key))
    main_parts, stages = [official], []
    for data, weight in [(matched, .5), (pool, .35)]:
        data = data.loc[data.model.isin(labels)].copy()
        data["text"] = data.text.fillna("").astype(str)
        keep = []
        for text in data.text:
            key = main_key(text)
            ok = bool(key) and key not in seen
            keep.append(ok)
            if ok:
                seen.add(key)
        retained = data.loc[keep].copy()
        retained["source_weight"] = weight
        main_parts.append(retained[["text", "model", "origin", "row_id", "source_weight"]])
        stages.append({"origin": str(data.origin.iloc[0]), "input_rows": len(data), "retained_rows": len(retained), "removed_prior_or_within_duplicate_rows": int(len(data)-len(retained)), "source_weight": weight})
    main_fit = pd.concat(main_parts, ignore_index=True)
    fit_lr = pd.concat(main_parts[:2], ignore_index=True)
    main_fit["audit_text_hash"] = main_fit.text.map(audit_hash)
    temporary = OUT / "main_svc_rows.pending.parquet"
    main_fit.drop(columns="text").to_parquet(temporary, index=False)
    temporary.replace(OUT / "main_svc_rows.parquet")

    # Annotated mirror of the exact comp loader; compare its final three columns
    # against the unmodified loader extracted from the frozen Notebook below.
    parts, comp_stages = [], []
    for table, path, label_col, finish_col in SOURCES:
        d = pd.read_csv(RAW / path)
        d["row_id"] = [f"{table}:{i}" for i in range(len(d))]
        d["origin"] = table
        comp_stages.append({"origin": table, "input_rows": len(d)})
        if finish_col and finish_col in d:
            d = d.loc[d[finish_col].eq("stop")]
        comp_stages[-1]["after_finish_filter"] = len(d)
        d = d.rename(columns={label_col: "model"})[["prompt_id", "model", "text", "row_id", "origin"]].copy()
        d["prompt_id"] = ("pv_" if path.endswith("passages_clean.csv") else "") + d.prompt_id.astype(str)
        parts.append(d)
    comp = pd.concat(parts, ignore_index=True)
    def count_stage(name):
        counts = comp.origin.value_counts()
        for row in comp_stages:
            row[name] = int(counts.get(row["origin"], 0))
    comp = comp.loc[comp.model.isin(labels)].copy()
    comp["text"] = comp.text.map(comp_text)
    comp = comp.loc[comp.text.str.split().str.len().between(30, 300)]
    count_stage("after_label_and_length")
    comp = comp.drop_duplicates(["model", "text"])
    count_stage("after_model_text_dedup")
    comp = comp.drop_duplicates(["prompt_id", "model"])
    count_stage("after_prompt_model_dedup")
    comp = comp.loc[comp.groupby("prompt_id").model.transform("nunique").ge(12)].reset_index(drop=True)
    count_stage("after_min_12_labels")
    def fake_glob(pattern, recursive=True):
        suffix = pattern.split("/**/", 1)[1]
        import fnmatch
        return sorted(str(RAW / path) for _, path, _, _ in SOURCES if fnmatch.fnmatch(Path(path).name, suffix))
    ns = {"glob": types.SimpleNamespace(glob=fake_glob), "pd": pd, "re": re, "LABELS": labels, "EXTRA_MIN_LABELS": 12}
    original = extract_function(HERE / "comp_stack.py", "load_extra_passages", ns)()
    pd.testing.assert_frame_equal(comp[["prompt_id", "model", "text"]], original)
    index = {label: i for i, label in enumerate(labels)}
    y = train.model.map(index).to_numpy()
    ey = comp.model.map(index).to_numpy()
    w = extract_function(HERE / "comp_stack.py", "extra_sample_weights", {"np": np, "y": y, "EX_Y": ey, "K": 18, "EXTRA_SHARE": .5})()
    comp["source_weight"] = w[len(train):]
    comp["audit_text_hash"] = comp.text.map(audit_hash)
    comp.drop(columns="text").to_csv(OUT / "complement_extra_rows.csv", index=False)
    mass = pd.concat([pd.DataFrame({"model":train.model,"origin":"official","weight":w[:len(train)]}),pd.DataFrame({"model":comp.model,"origin":"public","weight":comp.source_weight})]).groupby(["model","origin"]).weight.sum().unstack(fill_value=0)
    mass.to_csv(OUT / "stylegram_class_weight_mass.csv")
    assert np.allclose(mass.sum(axis=1).reindex(labels), train.model.value_counts().reindex(labels))
    pd.DataFrame(comp_stages).to_csv(OUT / "complement_filter_stages.csv", index=False)
    pd.DataFrame(stages).to_csv(OUT / "main_filter_stages.csv", index=False)
    flags = pd.read_parquet(ROOT / "experiments/EXP-008-public-pool-audit/outputs/row_flags.parquet")
    known_regen = set(flags.loc[flags.linked_official_prompt, "row_id"])
    members = {"main_tfidf_distort":fit_lr, "main_svc4":main_fit, "comp_stylegram":comp}
    # The Claude gate bypasses the shared comp filter completely.
    gate_details = []
    for family in ["claude", "gpt", "gemini", "glm"]:
        gate_extra = parts[0] if family == "claude" else comp.loc[comp.model.str.startswith(family)]
        gate_details.append({"family":family,"official_rows":int(train.model.str.startswith(family).sum()),"public_rows":len(gate_extra),"public_known_regen_rows":int(gate_extra.row_id.isin(known_regen).sum()),"sample_weight":"1 for both sources; no class_weight" if family=="claude" else "official=1/public=0.25 plus class_weight=balanced"})
        members["comp_gate_"+family] = gate_extra
    exposure=[]
    for name, data in members.items():
        total = len(data) + len(train) if name == "comp_stylegram" else len(data)
        if name.startswith("comp_gate"):
            total += int(train.model.str.startswith(name.removeprefix("comp_gate_")).sum())
        exposure.append({"branch":name,"total_fit_rows":total,"public_rows":int(data.origin.ne("official").sum()),"known_official_prompt_regen_rows":int(data.row_id.isin(known_regen).sum())})
    candidates=pd.read_csv(ROOT / "experiments/EXP-008-public-pool-audit/outputs/initial_candidate_index.csv",keep_default_na=False)
    union=set(main_fit.audit_text_hash)|set(comp.audit_text_hash)
    comparison=candidates[["row_id","table","split","text_hash"]].copy()
    comparison["already_in_projected_main_svc4"]=comparison.text_hash.isin(set(main_fit.audit_text_hash))
    comparison["already_in_projected_comp_stylegram"]=comparison.text_hash.isin(set(comp.audit_text_hash))
    comparison["already_in_either"]=comparison.text_hash.isin(union)
    runtime_union = union | set(parts[0].text.map(audit_hash))
    comparison["already_in_any_audited_runtime_branch"] = comparison.text_hash.isin(runtime_union)
    comparison.to_csv(OUT / "candidate_overlap.csv",index=False)
    counts = comparison.groupby("table").agg(candidate_rows=("row_id","size"),in_main=("already_in_projected_main_svc4","sum"),in_comp=("already_in_projected_comp_stylegram","sum"),in_either=("already_in_either","sum"),in_any_audited_runtime_branch=("already_in_any_audited_runtime_branch","sum"))
    counts.to_csv(OUT / "candidate_overlap_by_source.csv")
    new = comparison.loc[comparison.table.eq("prvs-clean-v6") & comparison.split.eq("new")]
    result={"observed_at":datetime.now(timezone.utc).isoformat(),"scope":"baseline rules on EXP-008 pinned files; historical hidden-run counts NOT verified", "source_notebook_sha256":sha(notebook),"audit_sha256":sha(Path(__file__)),"main_stages":stages,"main_lr_rows":len(fit_lr),"main_svc_rows":len(main_fit),"complement_rows":len(comp),"complement_prompt_groups":int(comp.prompt_id.nunique()),"complement_stages":comp_stages,"gates":gate_details,"known_regen_exposure":exposure,"candidate_rows":len(candidates),"candidate_already_in_either":int(comparison.already_in_either.sum()),"prvs_new_candidate_rows":len(new),"prvs_new_already_in_either":int(new.already_in_either.sum()),"checks":{"exact_comp_loader_equivalence":True,"exact_main_key_equivalence":True,"stylegram_class_mass_preserved":True},"unobserved_inputs":["ntlgenerated wildcard files outside six audited pools","historical snapshot bytes and original GPU fit counts","all checkpoint original training row membership"]}
    result["candidate_already_in_any_audited_runtime_branch"] = int(comparison.already_in_any_audited_runtime_branch.sum())
    result["candidate_not_in_audited_runtime_branches"] = int((~comparison.already_in_any_audited_runtime_branch).sum())
    result["novelty_limit"] = "Not present in audited runtime fitting tables does not mean unseen by checkpoint training"
    comparison.loc[~comparison.already_in_any_audited_runtime_branch].to_csv(OUT / "potential_increment_rows.csv", index=False)
    dump(OUT / "projection.json",result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__":
    run()
