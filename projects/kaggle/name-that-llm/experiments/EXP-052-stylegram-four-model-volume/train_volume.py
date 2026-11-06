"""Train the two frozen EXP052 Stylegram volume arms on Kaggle CPU."""
import os
import time

START = time.monotonic()
for env_name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[env_name] = "2"

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
import traceback

import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

ARMS = ("four120", "four240")
NEW_LABELS = ("gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol", "gpt-6-astra")
RUN_ID = "exp052-stylegram-four-volume-v1"
FORMAT_HASH = "2bb7e8376b44b3450cc5fc2b71f315a8cd1acea29b26d05e207bc3d666fbce8d"
HARD_SECONDS = 6600
RSS_LIMIT = 20 * 1024**3


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_format(payload):
    path = payload / "format_runner.py"
    require(digest(path) == FORMAT_HASH, "Frozen EXP044 runner changed")
    spec = importlib.util.spec_from_file_location("exp052_format", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_added(frame, size, legacy, baseline_text, holdout):
    fields = ["row_id", "origin", "prompt_id", "model", "text", "base_prompt", "text_sha256", "job_key", "split", "source_protocol"]
    require(set(fields).issubset(frame.columns) and frame[fields].ne("").all().all(), "Missing added row data")
    groups = size // 4
    require(len(frame) == size and frame.row_id.is_unique and frame.job_key.is_unique, "Added IDs or count")
    require(frame.split.eq("train").all(), "Holdout row in training input")
    require(frame.model.value_counts().to_dict() == dict.fromkeys(NEW_LABELS, groups), "Four-label balance changed")
    by_group = frame.groupby("prompt_id")
    require(len(by_group) == groups and by_group.size().eq(4).all() and by_group.model.nunique().eq(4).all(),
            "Each added prompt must contain exactly one answer from each model")
    raw_hashes = frame.text.map(legacy.text_hash)
    normalized = frame.text.map(legacy.normalize)
    require(raw_hashes.tolist() == frame.text_sha256.tolist() and normalized.is_unique, "Added text hash or duplication")
    require(set(normalized).isdisjoint(baseline_text), "Added text overlaps official/public inputs")
    source_groups = frame.base_prompt.map(lambda text: legacy.text_hash(legacy.normalize(text)))
    mapping = pd.DataFrame({"prompt": frame.prompt_id, "hash": source_groups}).drop_duplicates()
    require(len(mapping) == groups and mapping["prompt"].is_unique and mapping["hash"].is_unique, "Group identity collision")
    require(set(source_groups).isdisjoint(holdout), "Held-out prompt leaks into training input")


def validate_volume(frames, legacy, gate, config):
    train, folds, extra, small, large = frames
    require(len(train) == len(folds) == 2699 and train.id.is_unique, "Official IDs or count")
    for col in ("id", "model", "prompt_id"):
        require(train[col].tolist() == folds[col].tolist(), "Official fold alignment: " + col)
    require(train.text.map(legacy.text_hash).tolist() == folds.text_sha256.tolist(), "Official text mismatch")
    folds["fold"] = pd.to_numeric(folds.fold, errors="raise")
    require(set(folds.fold) == set(range(5)) and folds.groupby("prompt_id").fold.nunique().eq(1).all(), "Prompt fold leakage")
    require(set(train.model) == set(extra.model) == set(legacy.LABELS), "Official/public classes changed")
    require(len(extra) == 10073 and extra.row_id.is_unique and extra.prompt_id.nunique() == 694, "Public inventory changed")
    require(extra.groupby("prompt_id").canonical_prompt.nunique().eq(1).all()
            and not extra.canonical_prompt.str.startswith("official:").any(), "Public group identity changed")
    official_text = set(train.text.map(legacy.normalize))
    public_text = set(extra.text.map(legacy.normalize))
    require(official_text.isdisjoint(public_text), "Official/public answer overlap")
    held = config["holdout_prompt_group_hashes"]
    held_gate = gate["holdout_prompt_group_hashes"]
    require(len(held) == len(set(held)) and set(held) == set(held_gate) and held, "Holdout list mismatch")
    for arm, frame, size in zip(ARMS, (small, large), (120, 240)):
        receipt = gate["outputs"][arm + ".csv"]
        require(receipt["sha256"] == config["input_hashes"][arm + ".csv"]
                and receipt["rows"] == size and receipt["groups"] == size // 4, "Volume gate mismatch: " + arm)
        validate_added(frame, size, legacy, official_text | public_text, held)
    require(small.columns.tolist() == large.columns.tolist(), "Volume arm schemas differ")
    observed = large.set_index("row_id").reindex(small.row_id)
    require(observed.equals(small.set_index("row_id")), "four120 is not the exact nested subset of four240")


def preflight(payload):
    fmt = load_format(payload)
    legacy, core, legacy_path, core_path = fmt.modules(payload)
    config = read(payload / "config.json")
    frozen = {"experiment_id": "EXP-052", "run_id": RUN_ID, "arms": list(ARMS), "labels": legacy.LABELS,
              "new_labels": list(NEW_LABELS), "validation_scheme": "prompt-group-v1", "C": .5,
              "max_iter": 4000, "dual": True, "seed": 42, "share": .5, "sparse_per_prompt": [2, 3, 6],
              "sparse_repeats": 5, "parallel_workers": 2, "threads": 2}
    for key, expected in frozen.items():
        require(config.get(key) == expected, "Fixed configuration changed: " + key)
    budget = config["budget"]
    require(budget["hard_seconds"] == HARD_SECONDS and budget["rss_limit_bytes"] == RSS_LIMIT
            and budget["platform_wall_seconds"] == 7200, "Resource budget mismatch")
    hashes = config["input_hashes"]
    required = {*legacy.FROZEN_HASHES, "four120.csv", "four240.csv", "train_volume.py", "legacy_runner.py", "format_runner.py", "volume_gate.json"}
    require(required.issubset(hashes), "Missing input/code hashes")
    for name, value in hashes.items():
        require(Path(name).name == name and ":" not in name and "\\" not in name and digest(payload / name) == value,
                "Input hash mismatch: " + name)
    require(hashes["train_volume.py"] == digest(__file__) and all(hashes[k] == v for k, v in legacy.FROZEN_HASHES.items()),
            "Frozen original input or executing code changed")
    environment = config["environment"]
    require(environment["required_versions"] == legacy.VERSIONS and environment["internet"] is False
            and environment["accelerator"] is None, "CPU environment policy changed")
    versions = fmt.runtime()
    require(all(versions[k] == v for k, v in legacy.VERSIONS.items()), "Required Python package versions differ")
    require(sys.platform.startswith("linux") and Path("/kaggle/working").is_dir() and not Path("/dev/nvidia0").exists(),
            "Kaggle CPU runtime required")
    gate = read(payload / "volume_gate.json")
    require(gate["status"] == "PASS_FROZEN_BEFORE_EXP052_OOF", "Training volume gate is not frozen")
    frames = [pd.read_csv(payload / name, dtype=str, keep_default_na=False)
              for name in ("train.csv", "folds.csv", "extra.csv", "four120.csv", "four240.csv")]
    validate_volume(frames, legacy, gate, config)
    identity = {"run_id": RUN_ID, "config_sha256": digest(payload / "config.json"),
                "runner_sha256": digest(__file__), "legacy_runner_sha256": digest(legacy_path),
                "core_sha256": digest(core_path), "format_runner_sha256": FORMAT_HASH, "input_hashes": hashes,
                "environment": versions, "labels": config["labels"], "folds": 5, "arms": list(ARMS)}
    config["_config_path"] = str(payload / "config.json")
    return fmt, legacy, core, config, frames, identity


def worker(args):
    fmt, legacy, core, config, frames, identity = preflight(args.payload)
    require(identity == read(args.output / "identity.json"), "Training job identity differs from supervisor")
    arm, fold = args.arm, args.fold
    directory = args.output / "jobs" / f"f{fold}-{arm}"
    fmt.save(directory / "identity.json", {"identity": identity, "fold": fold, "arm": arm})
    fmt.save(directory / "process.json", fmt.process_record())
    selected = frames[3 + ARMS.index(arm)]
    # EXP044 chooses raw only when arm=='raw120', else clean. Supplying the
    # explicitly selected table in BOTH positions preserves the actual arm ID.
    inputs = [*frames[:3], selected, selected]
    with threadpool_limits(limits=2):
        fmt.compute_job(config, inputs, legacy, core, identity, fold, arm, directory)
    # The parent seals only after the child exits and training.log is closed.


def aggregate(directory, frames, fmt, identity, jobs):
    train, folds = frames[:2]
    oof = {}
    for arm in ARMS:
        blocks = {key: [] for key in ("ids", "scores", "plain_scores", "true", "group_ids", "centred_mask", "route", "fold")}
        positions = []
        for job in (j for j in jobs if j["arm"] == arm):
            fold = job["fold"]
            path = directory / job["path"]
            fmt.validate_job(path, identity, fold, arm)
            idx = np.flatnonzero(folds.fold.to_numpy() == fold)
            with np.load(path / "dense.npz", allow_pickle=False) as matrix:
                require(matrix["ids"].tolist() == train.id.iloc[idx].tolist()
                        and np.array(identity["labels"])[matrix["true"]].tolist() == train.model.iloc[idx].tolist(),
                        "Fold outputs do not match official rows")
                for key in blocks:
                    blocks[key].append(matrix[key])
            positions.extend(idx.tolist())
        require(sorted(positions) == list(range(2699)), "Incomplete or repeated official fold coverage")
        order = np.argsort(positions)
        arrays = {k: np.concatenate(v)[order] for k, v in blocks.items()}
        path = directory / "oof" / (arm + ".npz")
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, **arrays, labels=np.array(identity["labels"]), arm=arm, **fmt.flat_identity(identity))
        oof[arm] = {"path": path.relative_to(directory).as_posix(), "sha256": digest(path)}
    return oof


def command(args, role, *extra):
    return [sys.executable, "-u", str(Path(__file__).resolve()), "--payload", str(args.payload),
            "--output", str(args.output), "--role", role, *extra]


def coordinate(args):
    fmt, _, _, _, frames, identity = preflight(args.payload)
    fmt.save(args.output / "identity.json", identity)
    plan = [(fold, arm) for fold in range(5) for arm in ARMS]
    active, completed = {}, []
    try:
        while plan or active:
            for (fold, arm), proc in list(active.items()):
                status = proc.poll()
                if status is None:
                    continue
                require(status == 0, f"Worker f{fold}-{arm} failed with code {status}")
                path = args.output / "jobs" / f"f{fold}-{arm}"
                fmt.seal_job(path, identity, fold, arm)
                completed.append({"fold": fold, "arm": arm, "path": path.relative_to(args.output).as_posix(),
                                  "completion_sha256": digest(path / "completion.json")})
                del active[(fold, arm)]
            while plan and len(active) < 2:
                fold, arm = plan.pop(0)
                directory = args.output / "jobs" / f"f{fold}-{arm}"
                directory.mkdir(parents=True, exist_ok=False)
                with (directory / "training.log").open("x", encoding="utf-8") as log:
                    active[(fold, arm)] = subprocess.Popen(command(args, "worker", "--fold", str(fold), "--arm", arm),
                                                           stdout=log, stderr=subprocess.STDOUT)
            fmt.save(args.output / "progress.json", {"completed_jobs": completed, "active": list(active), "pending": plan})
            if active:
                time.sleep(2)
    finally:
        for child in active.values():
            if child.poll() is None:
                fmt.stop_tree(child.pid)
    require(len(completed) == 10, "Incomplete fold/arm schedule")
    completed.sort(key=lambda row: (row["fold"], ARMS.index(row["arm"])))
    oof = aggregate(args.output, frames, fmt, identity, completed)
    require(preflight(args.payload)[-1] == identity, "Inputs changed during training")
    fmt.save(args.output / "result.json", {"status": "COMPLETE", **fmt.flat_identity(identity),
             "format_runner_sha256": FORMAT_HASH, "identity": identity, "completed_jobs": completed,
             "oof": oof, "trained_svms": 20, "control_fits": 0, "gpu_used": False})


def enforce_budget(elapsed, rss):
    require(elapsed < HARD_SECONDS, "WALL_BUDGET_EXCEEDED")
    require(rss <= RSS_LIMIT, "RSS_BUDGET_EXCEEDED")


def tree_rss():
    root = psutil.Process()
    total = root.memory_info().rss
    for child in root.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total


def supervise(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    require(not any(output.iterdir()), "Output already exists; no retry or overwrite")
    with (output / ".run.lock").open("x") as lock:
        lock.write(str(os.getpid()))
    fmt = load_format(args.payload)
    save(output / "state.json", {"status": "RUNNING", "started_at": time.time(), "run_id": RUN_ID})
    env = dict(os.environ, EXP052_SUPERVISOR_PID=str(os.getpid()))
    child = None
    peak = 0
    last_report = -60.0
    try:
        with (output / "coordinator.log").open("x", encoding="utf-8") as log:
            child = subprocess.Popen(command(args, "coordinator"), stdout=log, stderr=subprocess.STDOUT,
                                     env=env, start_new_session=True)
            while True:
                elapsed = time.monotonic() - START
                rss = tree_rss()
                peak = max(peak, rss)
                enforce_budget(elapsed, rss)
                if elapsed - last_report >= 30:
                    report = {"status": "RUNNING", "run_id": RUN_ID, "elapsed_seconds": elapsed, "rss_bytes": rss,
                              "peak_observed_rss_bytes": peak}
                    if (output / "progress.json").exists():
                        progress = read(output / "progress.json")
                        report.update(active=progress["active"], completed_jobs=len(progress["completed_jobs"]))
                    save(output / "guard.json", report)
                    print(json.dumps(report), flush=True)
                    last_report = elapsed
                if child.poll() is not None:
                    require(child.returncode == 0, f"Coordinator failed ({child.returncode}); see coordinator.log")
                    break
                time.sleep(1)
        result = read(output / "result.json")
        require(result["status"] == "COMPLETE" and result["trained_svms"] == 20, "Missing final result")
        final = {"status": "COMPLETE", "run_id": RUN_ID, "elapsed_seconds": time.monotonic() - START,
                 "peak_observed_rss_bytes": peak, "result_sha256": digest(output / "result.json")}
        save(output / "guard.json", final)
        save(output / "state.json", final)
        print(json.dumps(final), flush=True)
    except BaseException:
        if child is not None:
            # Workers inherit this private process group. This also stops them
            # if a coordinator is killed before it can run its finally block.
            if sys.platform.startswith("linux"):
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            elif child.poll() is None:
                fmt.stop_tree(child.pid)
            child.wait(timeout=20)
        failure = {"status": "FAILED", "run_id": RUN_ID, "elapsed_seconds": time.monotonic() - START,
                   "peak_observed_rss_bytes": peak, "traceback": traceback.format_exc()}
        save(output / "guard.json", failure)
        save(output / "state.json", failure)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/results"))
    parser.add_argument("--role", choices=("supervisor", "coordinator", "worker"), default="supervisor")
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    args.payload = args.payload.resolve()
    args.output = args.output.resolve()
    if args.role == "supervisor":
        supervise(args)
        return
    pid = int(os.environ["EXP052_SUPERVISOR_PID"])
    require(pid in {p.pid for p in psutil.Process().parents()} and (args.output / ".run.lock").read_text() == str(pid),
            "No live supervisor for child")
    if args.role == "coordinator":
        coordinate(args)
    else:
        require(args.fold in range(5) and args.arm in ARMS, "Invalid worker job")
        worker(args)


if __name__ == "__main__":
    main()
