"""Read-only, fail-closed EXP044 real-output verification and paired metrics.

Usage: python -B verify_outputs.py --results DIR [--output report.json]
Without --output, prints the JSON summary and writes nothing. --output writes
that JSON and a same-stem Markdown sibling, outside the evidence directory.
Never trains, launches/resumes a run, edits evidence, or imports remote code.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import platform
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import scipy
import sklearn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAYLOAD = ROOT / "cloud/format120-044/payload"
ARMS = ("raw120", "clean120")
EXCLUDED = {"artifacts.json", ".guard.lock", ".claim.lock", "guard.json", "state.json"}
ARRAY_FIELDS = ("ids", "scores", "plain_scores", "true", "group_ids", "centred_mask", "route", "fold")
HARD = 6600
RSS = 20 * 1024**3

def need(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)

def close(a, b, message, atol=1e-12):
    need(finite(a) and finite(b) and math.isclose(a, b, rel_tol=0, abs_tol=atol), message)

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded

def safe_path(root, relative):
    need(isinstance(relative, str) and relative and "\\" not in relative and ":" not in relative, "Unsafe path")
    part = PurePosixPath(relative)
    need(not part.is_absolute() and ".." not in part.parts and part.as_posix() == relative, "Unsafe relative path")
    candidate = root / relative
    need(candidate.resolve().is_relative_to(root.resolve()), "Path escapes evidence root")
    for parent in [candidate] + list(candidate.parents):
        if parent == root.parent:
            break
        need(not parent.is_symlink(), "Symlink in evidence")
    return candidate

def check_inventory(root, entries, excluded=()):
    need(isinstance(entries, list) and bool(entries), "Empty inventory")
    names = set()
    for item in entries:
        p = safe_path(root, item["path"])
        need(item["path"] not in names and item["path"] not in excluded, "Duplicate/excluded inventory entry")
        need(p.is_file() and type(item["bytes"]) is int and p.stat().st_size == item["bytes"], "Inventory size mismatch")
        need(sha(p) == item["sha256"], "Inventory SHA256 mismatch: " + item["path"])
        names.add(item["path"])
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.relative_to(root).as_posix() not in excluded}
    need(actual == names, "Inventory is not closed")
    return {item["path"]: item["sha256"] for item in entries}

def scores_metric(true, prediction, classes=18):
    true, prediction = np.asarray(true), np.asarray(prediction)
    need(true.ndim == prediction.ndim == 1 and true.shape == prediction.shape and len(true) > 0, "Metric population")
    need(true.dtype.kind in "iu" and prediction.dtype.kind in "iu", "Metric labels must be integer")
    need(((true >= 0) & (true < classes)).all() and ((prediction >= 0) & (prediction < classes)).all(), "Metric label range")
    support = np.bincount(true, minlength=classes)
    need((support > 0).all(), "Metric missing true class")
    correct = np.bincount(true[true == prediction], minlength=classes)
    recalls = correct / support
    return {"balanced_accuracy": float(recalls.mean()), "accuracy": float((true == prediction).mean()),
            "correct": int(correct.sum()), "rows": len(true), "support": support.tolist(),
            "correct_by_class": correct.tolist(), "recall": recalls.tolist()}

def paired_metrics(true, raw, clean, labels):
    a, b = scores_metric(true, raw, len(labels)), scores_metric(true, clean, len(labels))
    fixed = int(((raw != true) & (clean == true)).sum())
    broken = int(((raw == true) & (clean != true)).sum())
    need(b["correct"]-a["correct"] == fixed-broken, "Paired net-correct identity")
    return {"raw120": {k:v for k,v in a.items() if k not in ("support", "correct_by_class", "recall")},
            "clean120": {k:v for k,v in b.items() if k not in ("support", "correct_by_class", "recall")},
            "clean_minus_raw_ba": b["balanced_accuracy"]-a["balanced_accuracy"],
            "fixed_errors": fixed, "introduced_errors": broken, "net_correct": fixed-broken,
            "prediction_disagreements": int((raw != clean).sum()),
            "per_class": [{"label":label, "support":a["support"][i],
                           "raw_correct":a["correct_by_class"][i], "clean_correct":b["correct_by_class"][i],
                           "raw_recall":a["recall"][i], "clean_recall":b["recall"][i],
                           "delta_recall":b["recall"][i]-a["recall"][i]} for i,label in enumerate(labels)]}

def prompt_bootstrap(true, raw, clean, prompts, classes=18, draws=2000, seed=42):
    prompts = np.asarray(prompts)
    unique, inverse = np.unique(prompts, return_inverse=True)
    require_n = len(unique)
    need(len(true) == len(prompts) and require_n > 0, "Bootstrap population")
    support = np.zeros((require_n,classes), dtype=np.int64)
    delta = np.zeros_like(support)
    np.add.at(support, (inverse,true), 1)
    np.add.at(delta, (inverse,true), (clean == true).astype(np.int64)-(raw == true).astype(np.int64))
    rng = np.random.default_rng(seed)
    values = []
    invalid = 0
    for _ in range(draws):
        selected = rng.integers(0, require_n, size=require_n)
        den = support[selected].sum(axis=0)
        if not (den > 0).all():
            invalid += 1
            continue
        values.append(float((delta[selected].sum(axis=0)/den).mean()))
    return {"unit":"original official prompt_id", "groups":require_n, "draws":draws, "seed":seed,
            "valid_draws":len(values), "invalid_draws":invalid,
            "delta_definition":"clean120 minus raw120 pooled 18-class macro recall",
            "ci95":np.quantile(values,[.025,.975]).tolist() if values and invalid == 0 else None,
            "conditional_on_fitted_models":True, "includes_retraining_uncertainty":False,
            "missing_class_policy":"null interval if any resample lacks a true class"}

def sparse_indices(valid, m, repeat):
    # Independent transcription of the frozen label-blind selection rule.
    keys = valid.id.map(lambda s: hashlib.sha256(f"42:{repeat}:{s}".encode()).hexdigest())
    order = pd.DataFrame({"prompt":valid.prompt_id.to_numpy(),"key":keys.to_numpy(),"position":np.arange(len(valid))})
    return np.sort(order.sort_values(["prompt","key"]).groupby("prompt",sort=True).head(m).position.to_numpy())

def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key:archive[key] for key in archive.files}

def validate_array(data, identity, runner, valid, true, fold, arm, group, mask, m=None, repeat=None):
    need(set(ARRAY_FIELDS).union({"labels","arm"}).issubset(data), "Missing score-cache arrays")
    for key,value in runner.flat_identity(identity).items():
        need(data[key].shape == () and data[key].item() == value, "NPZ frozen identity: " + key)
    need(data["labels"].tolist() == identity["labels"], "NPZ label order")
    need(data["arm"].shape == () and data["arm"].item() == arm, "NPZ arm")
    n = len(valid)
    need(data["ids"].shape == (n,) and data["ids"].tolist() == valid.id.tolist(), "NPZ sample order")
    need(len(set(data["ids"])) == n, "Duplicate NPZ IDs")
    need(data["true"].dtype.kind in "iu" and data["true"].shape == (n,) and np.array_equal(data["true"],true), "NPZ ground truth")
    need(data["fold"].dtype.kind in "iu" and data["fold"].shape == (n,) and np.all(data["fold"] == fold), "NPZ fold")
    for key in ("scores","plain_scores"):
        need(data[key].dtype == np.float64 and data[key].shape == (n,18) and np.isfinite(data[key]).all(), "Nonfinite/non-float64 scores")
    need(data["group_ids"].dtype.kind in "iu" and data["group_ids"].shape == (n,) and np.array_equal(data["group_ids"],group), "Text-only grouping mismatch")
    need(data["centred_mask"].dtype == np.bool_ and data["centred_mask"].shape == (n,) and np.array_equal(data["centred_mask"],mask), "Group eligible mask mismatch")
    need(data["route"].shape == (n,) and np.array_equal(data["route"],np.where(mask,"centred","plain")), "Route differs from eligibility")
    need(np.array_equal(data["scores"][~mask],data["plain_scores"][~mask]), "Plain route score was modified")
    if m is not None:
        need(data["m"].shape == data["repeat"].shape == () and data["m"].item() == m and data["repeat"].item() == repeat, "Sparse scenario metadata")
    else:
        need("m" not in data and "repeat" not in data, "Dense cache has sparse markers")

def local_bindings():
    manifest = read(HERE/"manifest.json")
    need(manifest.get("experiment_id") == "EXP-044" and manifest.get("freeze_status") == "FROZEN_BEFORE_TRAINING", "Local final manifest not frozen")
    entries = [manifest["config"],manifest["protocol"]] + manifest["inputs"] + manifest["code"]
    bound = {}
    for item in entries:
        p = safe_path(ROOT,item["path"])
        need(p.is_file() and sha(p) == item["sha256"], "Local frozen manifest drift: " + item["path"])
        if "bytes" in item:
            need(p.stat().st_size == item["bytes"], "Frozen local file size changed")
        bound[item["path"]] = item["sha256"]
    config = read(PAYLOAD/"config.json")
    need(config == read(HERE/"config.json") and manifest["run_id"] == config["run_id"], "Local payload/config mismatch")
    protocol = read(PAYLOAD/"protocol.json")
    need(protocol["run_id"] == config["run_id"] and protocol["experiment_id"] == "EXP-044", "Local protocol run mismatch")
    need(protocol["work_budget_seconds"] == HARD and protocol["rss_limit_bytes"] == RSS, "Protocol resource limits")
    for name,digest in protocol["input_hashes"].items():
        need(sha(safe_path(PAYLOAD,name)) == digest, "Frozen payload mismatch: " + name)
    for name,digest in config["input_hashes"].items():
        need(protocol["input_hashes"][name] == digest, "Protocol/config input mismatch")
    need(protocol["input_hashes"]["config.json"] == sha(PAYLOAD/"config.json"), "Protocol config hash")
    for name in ("runner.py","guard.py"):
        need(sha(HERE/name) == sha(PAYLOAD/name) == config["input_hashes"][name], "Frozen source/payload mismatch")
    runner = module(HERE/"runner.py","exp044_verified_runner")
    legacy, core, _, _ = runner.modules(PAYLOAD)
    legacy.validate_config(config)
    need(config["arms"] == list(ARMS) and config["parallel_workers"] == 2 and config["share"] == .5, "Experiment recipe changed")
    need(config["environment"]["required_versions"] == legacy.VERSIONS, "Required environment differs from frozen legacy")
    train = pd.read_csv(PAYLOAD/"train.csv",dtype=str,keep_default_na=False)
    folds = pd.read_csv(PAYLOAD/"folds.csv",dtype=str,keep_default_na=False)
    folds["fold"] = pd.to_numeric(folds.fold,errors="raise")
    need(len(train) == len(folds) == 2699 and train.prompt_id.nunique() == 150, "Official population")
    for key in ("id","prompt_id","model"):
        need(train[key].tolist() == folds[key].tolist(), "Official folded population differs")
    need(train.id.is_unique and folds.groupby("prompt_id").fold.nunique().eq(1).all(), "Official split group/ID violation")
    need(train.text.map(legacy.text_hash).tolist() == folds.text_sha256.tolist(), "Official row text changed")
    return manifest,config,protocol,runner,legacy,core,train,folds,bound

def verify(results):
    results = results.resolve()
    manifest,config,protocol,runner,legacy,core,train,folds,bound = local_bindings()
    initial_manifest_hash = sha(HERE/"manifest.json")
    initial_artifact_hash = sha(results/"artifacts.json")
    artifacts = read(results/"artifacts.json")
    need(artifacts["status"] == "COMPLETE" and artifacts["run_id"] == config["run_id"], "Artifact run incomplete/mismatch")
    check_inventory(results,artifacts["files"],EXCLUDED)
    control_hashes = {n:sha(results/n) for n in EXCLUDED if (results/n).is_file()}
    run = read(results/"run.json")
    need(run["config"] == config, "Output run embeds another config")
    identity = run["identity"]
    environment = identity["environment"]
    need(set(environment) == {"numpy","pandas","scipy","sklearn","python"}, "Environment schema")
    need(all(environment[k] == v for k,v in config["environment"]["required_versions"].items()), "Runtime version mismatch")
    need(isinstance(environment["python"],str) and bool(environment["python"]), "Python version missing")
    expected = {"schema_version":1,"run_id":config["run_id"],"config_sha256":sha(PAYLOAD/"config.json"),
                "runner_sha256":sha(PAYLOAD/"runner.py"),"guard_sha256":sha(PAYLOAD/"guard.py"),
                "legacy_runner_sha256":sha(PAYLOAD/"legacy_runner.py"),"core_sha256":sha(PAYLOAD/"stylegram_core.py"),
                "input_hashes":config["input_hashes"],"environment":environment,"labels":config["labels"],
                "arms":list(ARMS),"validation_scheme":"prompt-group-v1","folds":5,"strict_environment":True}
    need(identity == expected, "Output identity differs from frozen local manifest")
    guard = read(results/"guard.json")
    need(guard["status"] == "COMPLETE" and guard["child_exit"] == 0 and guard["termination_reason"] is None
         and guard["tree_stopped"] is True and not guard.get("cause") and not guard.get("traceback"), "Guard did not finish cleanly")
    need(guard["hard_seconds"] == HARD and guard["rss_limit_bytes"] == RSS, "Guard caps changed")
    need(finite(guard["cumulative_elapsed_seconds"]) and 0 < guard["cumulative_elapsed_seconds"] <= HARD, "Cumulative time cap")
    need(finite(guard["peak_rss_bytes"]) and 0 < guard["peak_rss_bytes"] <= RSS, "RSS cap")
    final_attempt = safe_path(results,"attempts/"+guard["attempt"])
    need(read(final_attempt/"guard.json") == guard, "Unsealed root guard differs from sealed terminal receipt")
    ticket = read(results/".guard.lock")
    need(ticket["nonce"] == guard["nonce"] and ticket["attempt"] == guard["attempt"], "Guard nonce/attempt binding")
    budget = read(results/"budget.json")
    need(budget["run_id"] == config["run_id"] and budget["config_sha256"] == expected["config_sha256"]
         and budget["protocol_sha256"] == sha(PAYLOAD/"protocol.json") and budget["hard_seconds"] == HARD, "Budget provenance")
    need(budget["attempts"] and len({a["attempt"] for a in budget["attempts"]}) == len(budget["attempts"]), "Budget attempt duplication")
    charged = 0.
    peaks = []
    for attempt in budget["attempts"]:
        need(finite(attempt["charged_seconds"]) and attempt["charged_seconds"] >= 0, "Unfinished/invalid budget charge")
        close(attempt["used_before_seconds"],charged,"Prior cumulative budget mismatch",1e-6)
        charged += attempt["charged_seconds"]
        terminal = results/"attempts"/attempt["attempt"]/"guard.json"
        if terminal.exists():
            g = read(terminal)
            need(g["attempt"] == attempt["attempt"] and g["nonce"] == attempt["nonce"] and g["tree_stopped"] is True, "Historical terminal identity/tree proof")
            need(g["peak_rss_bytes"] <= RSS, "Historical RSS exceeded cap")
            peaks.append(g["peak_rss_bytes"])
    close(charged,guard["cumulative_elapsed_seconds"],"Budget cumulative charge mismatch",1e-6)
    need(charged <= HARD and budget["attempts"][-1]["attempt"] == guard["attempt"]
         and budget["attempts"][-1]["status"] == "COMPLETE", "Budget final attempt mismatch")
    preflight = read(final_attempt/"preflight.json")
    need(preflight == {"status":"PASS","identity":identity}, "Final preflight identity")
    scheduler = read(final_attempt/"scheduler.json")
    need(scheduler["status"] == "COMPLETE" and type(scheduler["max_active_workers"]) is int and 0 <= scheduler["max_active_workers"] <= 2, "Scheduler bounds")
    gate_summaries = []
    for directory in sorted((results/"selftests").glob("attempt-*")):
        receipt_path = directory/"selftest-receipt.json"
        if not receipt_path.exists():
            continue
        receipt = read(receipt_path)
        check_inventory(directory,receipt["files"],{"selftest-receipt.json"})
        gate = receipt["result"]
        need(read(directory/"selftest.json") == gate and gate["status"] == "PASS", "Self-test gate receipt mismatch")
        for key in ("runner_sha256","legacy_runner_sha256","core_sha256","environment"):
            need(gate[key] == identity[key], "Self-test identity differs")
        need(gate["classes"] == 18 and gate["parallel_workers_observed"] == 2
             and gate["completed_resume_reused"] is True and gate["corruption_rejected"] is True, "Self-test cases missing")
        need(finite(gate["max_abs_score_difference"]) and 0 <= gate["max_abs_score_difference"] <= 1e-12, "Self-test serial/parallel discrepancy")
        gate_summaries.append({"path":directory.relative_to(results).as_posix(),"max_abs_score_difference":gate["max_abs_score_difference"]})
    need(gate_summaries,"No accepted sealed self-test gate")
    result = read(results/"result.json")
    need(result["status"] == "COMPLETE" and result["identity"] == identity and result["trained_svms"] == 20 and result["gpu_used"] is False, "Incomplete final result")
    for key,value in runner.flat_identity(identity).items():
        need(result[key] == value, "Final result flat identity")
    state = read(results/"state.json")
    need(state["status"] == "COMPLETE" and state["identity_sha256"] == runner.canonical_hash(identity)
         and state["completed_jobs"] == result["completed_jobs"] and not state.get("cause"), "Final state mismatch")
    plan = [(fold,arm) for fold in range(5) for arm in ARMS]
    jobs = runner.find_completed(results,identity,plan)
    need(set(jobs) == set(plan), "Not exactly ten completed fold/arm jobs")
    need(len(result["completed_jobs"]) == 10,"Final job count")
    declared = {}
    for item in result["completed_jobs"]:
        key = (item["fold"],item["arm"])
        need(key not in declared and key in jobs,"Duplicate/unknown final job")
        path = safe_path(results,item["path"])
        need(path == jobs[key] and sha(path/"completion.json") == item["completion_sha256"],"Final completion receipt binding")
        runner.validate_job(path,identity,*key)
        declared[key] = item
    aggregate = safe_path(results,result["aggregate_path"])
    need(aggregate == results/"aggregates"/guard["attempt"],"Final aggregate attempt mismatch")
    need(read(aggregate/"result.json") == result,"Aggregate/root result mismatch")
    labels = config["labels"]
    lookup = {label:i for i,label in enumerate(labels)}
    true = train.model.map(lookup).to_numpy(dtype=int)
    official_folds = folds.fold.to_numpy(dtype=int)
    dense = {arm:{} for arm in ARMS}
    sparse = {(arm,m,repeat):[] for arm in ARMS for m in (2,3,6) for repeat in range(5)}
    extra = pd.read_csv(PAYLOAD/"extra.csv",dtype=str,keep_default_na=False)
    added = pd.read_csv(PAYLOAD/"raw120.csv",dtype=str,keep_default_na=False)
    ey,ny = extra.model.map(lookup).to_numpy(dtype=int),added.model.map(lookup).to_numpy(dtype=int)
    for fold in range(5):
        vi = np.flatnonzero(official_folds == fold)
        ti = np.flatnonzero(official_folds != fold)
        valid = train.iloc[vi].reset_index(drop=True)
        group = core.normalised_content_groups(valid.text.tolist())
        mask = core.eligible(group)
        _,_,masses = legacy.fixed_weights(true[ti],ey,ny)
        for arm in ARMS:
            directory = jobs[(fold,arm)]
            cache = load_npz(directory/"dense.npz")
            validate_array(cache,identity,runner,valid,true[vi],fold,arm,group,mask)
            dense[arm][fold] = cache
            job_result = read(directory/"result.json")
            close(job_result["balanced_accuracy"],scores_metric(true[vi],cache["scores"].argmax(1))["balanced_accuracy"],"Stored fold BA mismatch")
            need(job_result["rows"] == len(vi),"Stored fold row count")
            weights = read(directory/"weights.json")
            need(weights["old_row_weights_identical"] is True and weights["normalization_denominator"] == "base_extra_only","Weight policy mismatch")
            need(weights["classes"] == [dict(row,label=labels[row["class_index"]]) for row in masses],"Per-fold weights changed")
            for m in (2,3,6):
                for repeat in range(5):
                    selected = sparse_indices(valid,m,repeat)
                    sub = valid.iloc[selected].reset_index(drop=True)
                    sg = core.normalised_content_groups(sub.text.tolist())
                    sm = core.eligible(sg)
                    sp = load_npz(directory/f"sparse/m{m}-r{repeat}.npz")
                    validate_array(sp,identity,runner,sub,true[vi[selected]],fold,arm,sg,sm,m,repeat)
                    need(np.array_equal(sp["plain_scores"],cache["plain_scores"][selected]),"Sparse plain scores differ from dense")
                    sparse[(arm,m,repeat)].append(sp)
    oof = {}
    summaries = {s["arm"]:s for s in result["summaries"]}
    need(len(result["summaries"]) == 2 and set(summaries) == set(ARMS),"Summary arm population")
    for arm in ARMS:
        cache = load_npz(aggregate/f"oof-{arm}.npz")
        expected_arrays = {}
        for key in ARRAY_FIELDS:
            positions = np.concatenate([np.flatnonzero(official_folds == fold) for fold in range(5)])
            arr = np.concatenate([dense[arm][fold][key] for fold in range(5)])[np.argsort(positions)]
            expected_arrays[key] = arr
            need(np.array_equal(cache[key],arr),"Aggregate does not exactly reconstruct sealed dense jobs: "+key)
        # Full per-fold validation above plus reconstruction covers all routes/groups.
        need(cache["scores"].shape == (2699,18) and cache["scores"].dtype == np.float64 and np.isfinite(cache["scores"]).all(),"Aggregate score shape")
        need(cache["labels"].tolist() == labels and cache["ids"].tolist() == train.id.tolist(),"Aggregate labels/IDs")
        need(cache["arm"].item() == arm,"Aggregate arm")
        for key,value in runner.flat_identity(identity).items():
            need(cache[key].item() == value,"Aggregate identity")
        pred = cache["scores"].argmax(1)
        metrics = scores_metric(true,pred)
        for key in ("balanced_accuracy","accuracy"):
            close(summaries[arm][key],metrics[key],"Stored aggregate metric mismatch")
        need(summaries[arm]["correct"] == metrics["correct"],"Stored correct count mismatch")
        table = pd.read_csv(aggregate/f"oof-{arm}.csv",dtype=str,keep_default_na=False)
        expected_cols = ["id","prompt_id","model","fold","predicted_model","correct"]
        need(table.columns.tolist() == expected_cols and len(table) == 2699,"OOF CSV schema")
        for col in ("id","prompt_id","model"):
            need(table[col].tolist() == train[col].tolist(),"OOF CSV population")
        need(table.fold.astype(int).tolist() == official_folds.tolist(),"OOF CSV fold")
        need(table.predicted_model.tolist() == np.asarray(labels)[pred].tolist(),"OOF CSV prediction")
        need(table.correct.tolist() == ["True" if p == t else "False" for p,t in zip(pred,true)],"OOF CSV correctness")
        oof[arm] = pred
    comparison = paired_metrics(true,oof["raw120"],oof["clean120"],labels)
    fold_comparisons = []
    for fold in range(5):
        selected = official_folds == fold
        row = paired_metrics(true[selected],oof["raw120"][selected],oof["clean120"][selected],labels)
        row.pop("per_class")
        fold_comparisons.append({"fold":fold,**row})
    bootstrap = prompt_bootstrap(true,oof["raw120"],oof["clean120"],train.prompt_id.to_numpy())
    need(bootstrap["groups"] == 150 and bootstrap["valid_draws"] == 2000,"Official cluster bootstrap incomplete")
    scenarios = []
    for m in (2,3,6):
        for repeat in range(5):
            a,b = sparse[("raw120",m,repeat)],sparse[("clean120",m,repeat)]
            for ca,cb in zip(a,b):
                for field in ("ids","true","fold","group_ids","centred_mask","route"):
                    need(np.array_equal(ca[field],cb[field]),"Paired sparse population/routing differs")
            sy = np.concatenate([c["true"] for c in a])
            ap = np.concatenate([c["scores"].argmax(1) for c in a])
            bp = np.concatenate([c["scores"].argmax(1) for c in b])
            metric = paired_metrics(sy,ap,bp,labels)
            metric.pop("per_class")
            scenarios.append({"m":m,"repeat":repeat,**metric})
    sparse_means = []
    for m in (2,3,6):
        rows = [s for s in scenarios if s["m"] == m]
        sparse_means.append({"m":m,"repeats":5,
            "raw120_mean_ba":float(np.mean([s["raw120"]["balanced_accuracy"] for s in rows])),
            "clean120_mean_ba":float(np.mean([s["clean120"]["balanced_accuracy"] for s in rows])),
            "mean_delta_ba":float(np.mean([s["clean_minus_raw_ba"] for s in rows]))})
    # Closing frozen-evidence check; verifier does not repair or annotate results.
    need(sha(HERE/"manifest.json") == initial_manifest_hash and sha(results/"artifacts.json") == initial_artifact_hash,"Manifest changed during verification")
    for relative,digest in bound.items():
        need(sha(ROOT/relative) == digest,"Local frozen file changed during verification")
    for name,digest in protocol["input_hashes"].items():
        need(sha(safe_path(PAYLOAD,name)) == digest,"Payload changed during verification")
    check_inventory(results,artifacts["files"],EXCLUDED)
    need(all(sha(results/name) == digest for name,digest in control_hashes.items()),"Live-control file changed during verification")
    return {"status":"PASS_FROZEN_RAW_CLEAN_OOF","experiment_id":"EXP-044","run_id":config["run_id"],
            "local_manifest_sha256":initial_manifest_hash,"artifact_manifest_sha256":initial_artifact_hash,
            "verifier_sha256":sha(__file__),"identity":identity,
            "verified":{"jobs":10,"dense_fold_caches":10,"sparse_fold_caches":150,"aggregate_oof":[2699,18],
                        "official_prompt_groups":150,"closed_inventory":True,"no_training_performed":True},
            "guard":{"attempts":len(budget["attempts"]),"cumulative_seconds":charged,"peak_rss_bytes":max(peaks),
                     "final_child_exit":0,"tree_stopped":True},
            "selftests":gate_summaries,"primary":comparison,"fold_comparisons":fold_comparisons,"paired_prompt_bootstrap":bootstrap,
            "sparse_scenarios":scenarios,"sparse_means":sparse_means,
            "sparse_overall_mean_delta_ba":float(np.mean([s["clean_minus_raw_ba"] for s in scenarios])),
            "historical_control":{"included":False,"note":"EXP042 no-Codex control is historical; it is not a third retrained arm of EXP044."},
            "verification_environment":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__},
            "limitations":["Previously inspected development OOF; not untouched holdout, full ATLAS20, or leaderboard.",
                           "Cluster bootstrap is conditional on these fitted models and excludes retraining uncertainty.",
                           "Sparse15 settings are correlated pressure tests, not15 independent experiments.",
                           "Verifier recomputes text-only grouping locally and fails if it differs from sealed routes; classifier fitting is never repeated."]}

def markdown(report):
    p = report["primary"]
    lines = ["# EXP044 frozen result verification","",report["status"],"",
             f"Run: {report['run_id']}. Ten sealed jobs,150 sparse fold caches and two2699x18 OOF arrays verified.",
             "",f"Raw BA: {p['raw120']['balanced_accuracy']:.10f}; clean BA: {p['clean120']['balanced_accuracy']:.10f}; clean-raw: {p['clean_minus_raw_ba']:+.10f}.",
             f"Fixed errors: {p['fixed_errors']}; introduced errors: {p['introduced_errors']}; net correct: {p['net_correct']:+d}.",
             f"150-prompt paired bootstrap95% CI: {report['paired_prompt_bootstrap']['ci95']} (2000 draws, seed42).",
             "", "| Sparse m | Raw mean BA | Clean mean BA | Delta |","|---|---:|---:|---:|"]
    for row in report["sparse_means"]:
        lines.append(f"| {row['m']} | {row['raw120_mean_ba']:.8f} | {row['clean120_mean_ba']:.8f} | {row['mean_delta_ba']:+.8f} |")
    lines += ["","EXP042 control is historical and is not a third retrained arm.","",
              "Validation is development Stylegram OOF only. No holdout, ATLAS20 or leaderboard improvement is established. Bootstrap excludes retraining uncertainty."]
    return "\n".join(lines)+"\n"

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,required=True)
    parser.add_argument("--output",type=Path,help="Optional JSON file; Markdown written beside it, never inside results")
    args = parser.parse_args(argv)
    if args.output:
        output = args.output.resolve()
        need(output.suffix == ".json","--output must be a JSON file")
        need(not output.is_relative_to(args.results.resolve()) and not output.with_suffix(".md").is_relative_to(args.results.resolve()),"Cannot write into evidence directory")
    report = verify(args.results)
    content = json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    if args.output:
        output.parent.mkdir(parents=True,exist_ok=True)
        for path,data in ((output,content),(output.with_suffix(".md"),markdown(report))):
            need(not path.exists() or path.read_text(encoding="utf-8") == data,"Refusing to overwrite a different verification report")
            if not path.exists():
                path.write_text(data,encoding="utf-8")
    print(content)

if __name__ == "__main__":
    main()
