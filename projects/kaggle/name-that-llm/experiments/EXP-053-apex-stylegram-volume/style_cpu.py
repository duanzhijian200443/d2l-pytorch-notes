"""EXP053. Source-bound Stylegram CPU fits for three fixed volume arms."""
import os
import time

START = time.monotonic()
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"

import argparse
import ast
from collections import deque
import hashlib
import json
from pathlib import Path
import re
import signal
import subprocess
import sys
import traceback
import warnings

import numpy as np
import pandas as pd
import psutil
import scipy
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import csr_matrix, diags, hstack, vstack
from scipy.spatial.distance import squareform
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

ARMS = ("zero", "four120", "four240")
MAX_SECONDS = 6600
MAX_RSS = 20 * 1024**3
REQUIRED = {"numpy": "2.1.3", "pandas": "2.3.3", "scipy": "1.16.3", "sklearn": "1.6.1"}
NEW_MODELS = {"gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol", "gpt-6-astra"}
COMP_SHA = "4a4b800a5380094d78da66d140d224f8a3974d1e3012086783d5bf81b1264514"
PINNED = {
    "source.ipynb": "ca4c9e18ed32dcd8168b8118b192d39d657803d30dcd89a564b1c2d1d9c846dd",
    "train.csv": "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707",
    "folds.csv": "e8ba1e3712852168dc57f56960d4f3e00be612d639ade8a7dcb0f0d8335ce551",
    "panel.csv": "8eeec7f64e924f0fd79dca57d324d42d269f400bdfbfff76d60b4c168d9ae6f2",
    "four120.csv": "a4e86314a77f7ed699dbfb107b32a8263a1ff09fc7c3ec92b98d4028d487005d",
    "four240.csv": "694d91e2536db2d77d81f8d8b32a802763a750a2e9643fdb0ba6a10b76755a01",
    "shared.csv": "b98c246578afbc38b9bf3c0be1e666a41cc15d051c64d03e875ab0c85da1bd3b",
    "regen-fold-map.csv": "2d7680e2001d711d5a7ec92d9cc6c8bd8ef0b197b81a79ee795c298c1a0b56c5",
}
MATRIX_KEYS = ("ids", "scores", "plain_scores", "probabilities", "group_ids", "centred_mask", "fold")


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def table(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def author_module(path):
    need(sha(path) == PINNED["source.ipynb"], "SRC067 V1 notebook hash mismatch")
    cells = ["".join(c["source"]) for c in read(path)["cells"] if c.get("cell_type") == "code"]
    body, = [s.split("\n", 1)[1] for s in cells if s.startswith("%%writefile comp_stack.py\n")]
    need(hashlib.sha256(body.encode()).hexdigest() == COMP_SHA, "Author comp_stack.py hash mismatch")
    return body


def author_namespace(body):
    names = {
        "softmax_rows", "full_prob", "average_linkage_groups", "content_cosine_distance",
        "normalised_content_groups", "eligible", "describe_groups", "shape_view", "function_view",
        "StyleGrams", "sparse_group_centre", "svc_probs", "extra_sample_weights", "fit_svc",
        "add_member", "self_test_rows", "family_gate",
    }
    assigned = {"LABELS", "K", "LAB2I", "GROUP_THRESH", "GROUP_LAYER", "GROUP_MIN", "GROUP_MAX",
                "STYLE_GROUP_THRESH", "W", "EXTRA_MIN_LABELS", "EXTRA_SHARE", "FUNCTION_WORDS", "TOKEN_RE", "FAMILY_IDX"}
    nodes, seen = [], set()
    for node in ast.parse(body).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            nodes.append(node)
            seen.add(node.name)
        elif isinstance(node, ast.Assign):
            targets = {n.id for target in node.targets for n in ast.walk(target) if isinstance(n, ast.Name)}
            if targets and targets <= assigned:
                nodes.append(node)
                seen.update(targets)
    need(seen == names | assigned, "Unexpected source definitions")
    env = {key: globals()[key] for key in ("np", "pd", "re", "time", "fcluster", "linkage", "csr_matrix",
                                            "diags", "hstack", "vstack", "squareform", "TfidfVectorizer", "LinearSVC")}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<SRC067 Stylegram definitions>", "exec"), env)
    return env


def init_inputs(env, train, shared, test):
    env.update(TR=train.text.tolist(), TE=test.text.tolist(), EX_T=shared.text.tolist(),
               y=train.model.map(env["LAB2I"]).to_numpy(dtype=int),
               EX_Y=shared.model.map(env["LAB2I"]).to_numpy(dtype=int),
               EX_G=shared.prompt_id.to_numpy(dtype=object), PROMPT=train.prompt_id.to_numpy(dtype=object),
               TINY=len(test) < 50, T0=time.time(), LOGIT=np.zeros((len(test), env["K"])))


def added_fit(env, added):
    """Use the author's definitions; the new texts affect Stylegram only."""
    new_texts = added.text.tolist()
    grams = env["StyleGrams"]().fit(env["TR"] + env["EX_T"] + new_texts)
    official, public, test = (grams.transform(env[name]) for name in ("TR", "EX_T", "TE"))
    targets = [env["y"], env["EX_Y"]]
    plain = [official, public]
    centre = env["sparse_group_centre"]
    centred = [centre(official, env["PROMPT"]), centre(public, env["EX_G"], env["GROUP_MIN"], 10**6)]
    old_weights = env["extra_sample_weights"]()
    weights = [old_weights]
    if new_texts:
        new = grams.transform(new_texts)
        new_y = added.model.map(env["LAB2I"]).to_numpy(dtype=int)
        targets.append(new_y)
        plain.append(new)
        centred.append(centre(new, added.prompt_id.to_numpy(dtype=object), env["GROUP_MIN"], 10**6))
        counts = np.bincount(env["y"], minlength=env["K"])
        public_counts = np.bincount(env["EX_Y"], minlength=env["K"])
        need((public_counts[new_y] > 0).all(), "Added class lacks an original public per-row weight")
        weights.append(env["EXTRA_SHARE"] * counts[new_y] / public_counts[new_y])
    target = np.concatenate(targets)
    weight = np.concatenate(weights)
    need(np.array_equal(weight[:len(old_weights)], old_weights), "Original row weights changed")
    env["svc_plain"] = env["fit_svc"](vstack(plain).tocsr(), target, sample_weight=weight)
    env["svc_centred"] = env["fit_svc"](vstack(centred).tocsr(), target, sample_weight=weight)
    prob = env["svc_probs"](env["svc_plain"], test)
    groups = env["normalised_content_groups"](env["TE"])
    mask = env["eligible"](groups)
    env.update(G_style=groups, rows_style=mask)
    if mask.any():
        prob[mask] = env["svc_probs"](env["svc_centred"], centre(test[mask], groups[mask]))
    env["add_member"]("stylegram", prob)


def fit(body, train, shared, added, test, direct_author):
    env = author_namespace(body)
    init_inputs(env, train, shared, test)
    original_softmax, original_add = env["softmax_rows"], env["add_member"]
    observed_margins, observed = [], {}
    def observe_softmax(values):
        observed_margins.append(values.copy())
        return original_softmax(values)
    def observe_member(key, probabilities):
        need(key == "stylegram", "Unexpected member")
        original_add(key, probabilities)
        observed["probabilities"] = probabilities.copy()
        observed["fit_details"] = {
            kind: {"settings": env[kind].get_params(), "n_iter": int(env[kind].n_iter_),
                   "class_indices": env[kind].classes_.tolist()}
            for kind in ("svc_plain", "svc_centred")
        }
    env.update(softmax_rows=observe_softmax, add_member=observe_member)
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=2):
        warnings.simplefilter("always")
        if direct_author:
            need(added.empty, "Author baseline accepts no synthetic rows")
            begin = body.index("grams = StyleGrams().fit(TR + EX_T)")
            end = body.index("# ---- gates: style SVCs", begin)
            exec(compile(body[begin:end], "<SRC067 unmodified Stylegram fit block>", "exec"), env)
        else:
            added_fit(env, added)
    mask = env["rows_style"]
    need(len(observed_margins) == 1 + int(mask.any()), "Unexpected author margin capture")
    plain_scores = observed_margins[0]
    scores = plain_scores.copy()
    if mask.any():
        scores[mask] = observed_margins[1]
    prob = observed["probabilities"]
    need(prob.shape == scores.shape == (len(test), 18) and np.isfinite(scores).all(), "Invalid score matrix")
    need(np.array_equal(original_softmax(scores), prob) and np.allclose(prob.sum(1), 1.), "Margin/probability mismatch")
    need(all(f["class_indices"] == list(range(18)) for f in observed["fit_details"].values()), "Model class order")
    matrix = {"scores": scores, "plain_scores": plain_scores, "probabilities": prob,
              "group_ids": env["G_style"], "centred_mask": mask}
    details = {"models": observed["fit_details"], "warnings": [f"{w.category.__name__}: {w.message}" for w in caught],
               "direct_author_block": direct_author, "shared_class_counts": shared.model.value_counts().to_dict(),
               "added_class_counts": added.model.value_counts().to_dict(), "train_class_counts": train.model.value_counts().to_dict()}
    return matrix, details


def inputs(directory):
    cfg = read(directory / "experiment-config.json")
    need(cfg["experiment_id"] == "EXP-053" and cfg["required_versions"] == REQUIRED, "Unexpected experiment or environment")
    versions = {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__}
    need(versions == REQUIRED and not Path("/dev/nvidia0").exists(), "Kaggle pinned CPU environment required")
    manifest = cfg["input_hashes"]
    need(set(manifest) == set(PINNED) | {"style_cpu.py"}, "Incomplete input hash set")
    for key, expected in manifest.items():
        path = Path(__file__) if key == "style_cpu.py" else directory / key
        need(sha(path) == expected and PINNED.get(key, expected) == expected, "Input mismatch: " + key)
    body = author_module(directory / "source.ipynb")
    train, folds, shared, small, large, panel = (
        table(directory / name) for name in ("train.csv", "folds.csv", "shared.csv", "four120.csv", "four240.csv", "panel.csv"))
    need(set(panel.columns) == {"id", "text"} and len(panel) == 729 and panel.id.is_unique, "Blind panel contract")
    need(len(train) == len(folds) == 2699 and train.id.is_unique, "Official row count")
    for col in ("id", "prompt_id", "model"):
        need(train[col].tolist() == folds[col].tolist(), "Fold alignment: " + col)
    folds["fold"] = folds.fold.astype(int)
    need(set(folds.fold) == set(range(5)) and folds.groupby("prompt_id").fold.nunique().eq(1).all(), "Prompt split")
    labels = author_namespace(body)["LABELS"]
    need(set(train.model) == set(labels) and set(shared.model) <= set(labels), "Unexpected class")
    need(set(shared.columns) == {"prompt_id", "model", "text"} and not shared.empty
         and not shared.duplicated(["prompt_id", "model"]).any()
         and shared.groupby("prompt_id").model.nunique().ge(12).all(), "Author shared loader output mismatch")
    regen = table(directory / "regen-fold-map.csv")
    regen_groups = validate_regen_mapping(regen, folds, shared)
    need(set(large.model) == NEW_MODELS and large.groupby("prompt_id").size().eq(4).all()
         and small.groupby("prompt_id").size().eq(4).all(), "Added groups must be complete")
    need(large.set_index("row_id").reindex(small.row_id).equals(small.set_index("row_id")), "Nested arm mismatch")
    identity = {"experiment_id": "EXP-053", "config_sha256": sha(directory / "experiment-config.json"),
                "input_hashes": manifest, "source_sha256": PINNED["source.ipynb"], "comp_stack_sha256": COMP_SHA,
                "shared_sha256": manifest["shared.csv"], "labels": labels, "runtime": versions,
                "shared_full_rows": len(shared), "heldout_regen_groups_by_fold": regen_groups}
    return body, train, folds, shared, {"zero": small.iloc[:0], "four120": small, "four240": large}, panel, identity


def validate_regen_mapping(regen, folds, shared):
    need(set(regen.columns) == {"prompt_id", "official_prompt_id", "official_fold"}
         and len(regen) == 14 and regen.prompt_id.is_unique and regen.official_prompt_id.is_unique, "Regeneration map shape")
    official_folds = folds[["prompt_id", "fold"]].drop_duplicates().set_index("prompt_id")["fold"]
    actual = regen.official_prompt_id.map(official_folds)
    need(not actual.isna().any() and np.array_equal(actual.astype(int), regen.official_fold.astype(int)),
         "Regeneration group binding to official fold differs")
    selected = shared[shared.prompt_id.isin(regen.prompt_id)]
    need(len(selected) == 252 and selected.prompt_id.nunique() == 14
         and selected.groupby("prompt_id").size().eq(18).all(), "Regeneration rows do not match full groups")
    return {str(fold): sorted(regen.loc[actual.eq(fold), "prompt_id"].tolist()) for fold in range(5)}


def training_shared(shared, regen_groups, fold):
    removed = regen_groups[str(fold)] if fold != -1 else []
    retained = shared.loc[~shared.prompt_id.isin(removed)].reset_index(drop=True)
    need(set(retained.prompt_id).isdisjoint(removed) and len(retained) == len(shared) - 18 * len(removed),
         "Incomplete removal of held-out regeneration group")
    return retained, removed


def fit_family_gates(body, train, shared, test):
    """Execute the author's GPT/Gemini/GLM loop with its original shared data."""
    env = author_namespace(body)
    init_inputs(env, train, shared, test)
    env["GATES"] = {}
    begin = body.index("for fam, C in [('gpt', 1.0), ('gemini', 1.0), ('glm', 0.5)]:")
    end = body.index("del grams, X_tr, X_ex, X_te", begin)
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=2):
        warnings.simplefilter("always")
        exec(compile(body[begin:end], "<SRC067 original family gate loop>", "exec"), env)
    matrices, sources = {}, {}
    for family in ("gpt", "gemini", "glm"):
        indices = env["FAMILY_IDX"][family]
        scores = env["GATES"][family]
        need(scores.shape == (len(test), len(indices)) and np.isfinite(scores).all(), "Invalid family gate scores")
        matrices[family], matrices[family + "_indices"] = scores, indices
        sources[family] = {"official_rows": int(np.isin(env["y"], indices).sum()),
                           "public_rows": int(np.isin(env["EX_Y"], indices).sum()), "added_rows": 0}
    return matrices, {"sources": sources, "svm_fits": 3, "warnings": [f"{w.category.__name__}: {w.message}" for w in caught]}


def run_job(args):
    body, train, folds, shared, additions, panel, identity = inputs(args.input)
    need(read(args.output / "identity.json") == identity, "Input identity changed during training")
    fold, arm = args.fold, args.arm
    train_rows = train if fold == -1 else train.loc[folds.fold.ne(fold)]
    test = panel if fold == -1 else train.loc[folds.fold.eq(fold), ["id", "text"]]
    fit_shared, removed = training_shared(shared, identity["heldout_regen_groups_by_fold"], fold)
    matrix, details = fit(body, train_rows, fit_shared, additions[arm], test, direct_author=(arm == "zero"))
    directory = args.output / "jobs" / job_name(arm, fold)
    np.savez_compressed(directory / "predictions.npz", **matrix, ids=test.id.to_numpy(dtype=str),
                        fold=np.full(len(test), fold), labels=np.array(identity["labels"]), arm=arm)
    if fold == -1 and arm == "zero":
        gates, gate_details = fit_family_gates(body, train_rows, fit_shared, test)
        path = args.output / "panel/family_gates.npz"
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, **gates, ids=test.id.to_numpy(dtype=str), labels=np.array(identity["labels"]),
                            config_sha256=identity["config_sha256"], shared_sha256=identity["shared_sha256"])
        save(path.with_suffix(".json"), {"status": "COMPLETE", "sha256": sha(path),
             "shared_sha256": identity["shared_sha256"], **gate_details})
    save(directory / "receipt.json", {"status": "COMPLETE", "arm": arm, "fold": fold, "svm_fits": 2,
         "train_rows": len(train_rows), "shared_rows": len(fit_shared), "full_shared_rows": len(shared),
         "excluded_regen_groups": removed, "excluded_regen_rows": len(shared)-len(fit_shared),
         "added_rows": len(additions[arm]), "valid_rows": len(test),
         "elapsed_seconds": time.monotonic()-START, "config_sha256": identity["config_sha256"],
         "matrix_sha256": sha(directory / "predictions.npz"), **details})


def job_name(arm, fold):
    return f"{arm}-" + ("panel" if fold == -1 else f"f{fold}")


def aggregate(output, train, folds, panel, identity):
    files, receipts = {"oof": {}, "panel": {}}, []
    for arm in ARMS:
        accum = {key: [] for key in MATRIX_KEYS}
        positions = []
        for fold in [0, 1, 2, 3, 4, -1]:
            directory = output / "jobs" / job_name(arm, fold)
            receipt = read(directory / "receipt.json")
            need(receipt["status"] == "COMPLETE" and receipt["fold"] == fold and receipt["arm"] == arm
                 and receipt["svm_fits"] == 2 and receipt["config_sha256"] == identity["config_sha256"]
                 and sha(directory / "predictions.npz") == receipt["matrix_sha256"], "Unsealed predictions")
            receipts.append({"arm": arm, "fold": fold, "path": directory.relative_to(output).as_posix(),
                             "receipt_sha256": sha(directory / "receipt.json")})
            with np.load(directory / "predictions.npz", allow_pickle=False) as mat:
                need(mat["labels"].tolist() == identity["labels"] and mat["arm"].item() == arm
                     and np.all(mat["fold"] == fold), "Score identity mismatch")
                if fold == -1:
                    need(mat["ids"].tolist() == panel.id.tolist(), "Panel IDs mismatch")
                    arrays = {key: mat[key] for key in MATRIX_KEYS}
                else:
                    idx = np.flatnonzero(folds.fold.to_numpy() == fold)
                    need(mat["ids"].tolist() == train.id.iloc[idx].tolist(), "OOF IDs mismatch")
                    positions.extend(idx.tolist())
                    for key in MATRIX_KEYS:
                        accum[key].append(mat[key])
        need(sorted(positions) == list(range(len(train))), "OOF coverage mismatch")
        order = np.argsort(positions)
        oof = {key: np.concatenate(parts)[order] for key, parts in accum.items()}
        for kind, matrices in (("oof", oof), ("panel", arrays)):
            path = output / kind / (arm + ".npz")
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path, **matrices, labels=np.array(identity["labels"]), arm=arm,
                                config_sha256=identity["config_sha256"])
            files[kind][arm] = {"path": path.relative_to(output).as_posix(), "sha256": sha(path), "bytes": path.stat().st_size}
    return files, receipts


def total_rss():
    parent = psutil.Process()
    rss = parent.memory_info().rss
    for child in parent.children(recursive=True):
        try:
            rss += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return rss


def supervise(args):
    args.output.mkdir(parents=True, exist_ok=True)
    need(not any(args.output.iterdir()), "Output contains previous results")
    with (args.output / ".run.lock").open("x", encoding="ascii") as lock:
        lock.write(str(os.getpid()))
    active, completed = {}, []
    peak = 0
    try:
        _, train, folds, _, _, panel, identity = inputs(args.input)
        save(args.output / "identity.json", identity)
        queue = deque((arm, f) for f in [0, 1, 2, 3, 4, -1] for arm in ARMS)
        last = -60.
        while queue or active:
            elapsed = time.monotonic()-START
            peak = max(peak, total_rss())
            need(elapsed < MAX_SECONDS and peak <= MAX_RSS, "CPU time or RSS budget exceeded")
            for key, child in list(active.items()):
                if child.poll() is not None:
                    need(child.returncode == 0, "Fit failed: " + key)
                    need((args.output / "jobs" / key / "receipt.json").is_file(), "Job missing fit receipt")
                    completed.append(key)
                    del active[key]
            while queue and len(active) < 2:
                arm, fold = queue.popleft()
                key = job_name(arm, fold)
                directory = args.output / "jobs" / key
                directory.mkdir(parents=True, exist_ok=False)
                command = [sys.executable, "-u", str(Path(__file__).resolve()), "--input", str(args.input),
                           "--output", str(args.output), "--arm", arm, "--fold", str(fold)]
                with (directory / "training.log").open("x", encoding="utf-8") as log:
                    active[key] = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                                   env=dict(os.environ, EXP053_PARENT_PID=str(os.getpid())),
                                                   start_new_session=True)
            if elapsed - last >= 30 or not active:
                state = {"status": "RUNNING", "active": list(active), "completed": list(completed),
                         "elapsed_seconds": elapsed, "peak_observed_rss_bytes": peak}
                save(args.output / "state.json", state)
                print(json.dumps(state), flush=True)
                last = elapsed
            if active:
                time.sleep(1)
        files, receipts = aggregate(args.output, train, folds, panel, identity)
        need(inputs(args.input)[-1] == identity and len(receipts) == 18, "Final state or input mismatch")
        elapsed = time.monotonic() - START
        need(elapsed < MAX_SECONDS, "CPU wall budget exceeded")
        gate_path = args.output / "panel/family_gates.npz"
        gate = read(gate_path.with_suffix(".json"))
        need(gate["sha256"] == sha(gate_path) and gate["svm_fits"] == 3, "Family gate output mismatch")
        summary = {"status": "COMPLETE", **identity, "trained_svms": 36, "family_gate_svm_fits": 3,
                   "total_svm_fits": 39, "family_gates": {**gate, "path": "panel/family_gates.npz", "bytes": gate_path.stat().st_size},
                   "outputs": files, "jobs": receipts,
                   "elapsed_seconds": elapsed, "peak_observed_rss_bytes": peak, "gpu_used": False}
        save(args.output / "summary.json", summary)
        save(args.output / "state.json", {"status": "COMPLETE", "summary_sha256": sha(args.output / "summary.json")})
    except BaseException:
        save(args.output / "state.json", {"status": "FAILED", "completed": completed, "active": list(active),
             "elapsed_seconds": time.monotonic()-START, "peak_observed_rss_bytes": peak, "failure": traceback.format_exc()})
        raise
    finally:
        for child in active.values():
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait(timeout=20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--fold", choices=range(-1, 5), type=int)
    args = parser.parse_args()
    args.input, args.output = args.input.resolve(), args.output.resolve()
    if args.arm is None and args.fold is None:
        supervise(args)
    else:
        parent = int(os.environ["EXP053_PARENT_PID"])
        need(args.arm in ARMS and args.fold in range(-1, 5) and parent == os.getppid()
             and (args.output / ".run.lock").read_text() == str(parent), "No live supervisor for fit job")
        run_job(args)


if __name__ == "__main__":
    main()
