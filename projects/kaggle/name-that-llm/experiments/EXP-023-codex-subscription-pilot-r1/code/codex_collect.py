"""Bounded, resumable Codex collection. Import and default CLI mode never infer.

Configuration (paths are relative to the experiment directory):
  models: [{"slug": "...", "reasoning_effort": "low"}]
  instructions_file: "instructions.txt"
  workspace: "workspace"
  cli_overrides: ["features.apps=false", ...]
  prompts_file: "prompts.jsonl"  # optional, can reuse another run's frozen prompts
  request_timeout_seconds: 180  # optional, capped at 180

This controller does not promise API equivalence or account-policy approval.
It never retries a request. Any failed/uncertain attempt requires inspection.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import uuid


SOURCE = "codex_subscription"
FAILURE_WORDS = re.compile(
    r"\b429\b|rate.?limit|usage.?limit|quota.?exceed|unauthori[sz]ed|forbidden|"
    r"account.{0,35}(restrict|suspend)|authentication.{0,25}(fail|error)", re.I
)
TOOL_TYPES = {
    "command_execution", "mcp_tool_call", "web_search", "file_change",
    "collab_tool_call", "tool_call", "function_call", "computer_call",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def write_json(path, value, exclusive=False):
    data = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if exclusive:
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    else:
        temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        with temp.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_bytes().decode("utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_jsonl(path, value):
    with path.open("ab") as stream:
        stream.write((json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


class InspectionRequired(RuntimeError):
    """A previous request must not be replayed automatically."""


class RunLock:
    def __init__(self, path):
        self.path = path
        self.token = uuid.uuid4().hex

    def __enter__(self):
        try:
            write_json(self.path, {"pid": os.getpid(), "token": self.token,
                                   "created_at": utc_now()}, exclusive=True)
        except FileExistsError as exc:
            raise InspectionRequired(
                "Collection lock exists; inspect the recorded PID and prior run before removing it."
            ) from exc
        return self

    def __exit__(self, *unused):
        # Never remove another controller's lock.
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("token") == self.token and data.get("pid") == os.getpid():
                self.path.unlink()
        except (FileNotFoundError, ValueError):
            pass


def event_failure(event):
    kind = event.get("type", "")
    if kind in {"error", "turn.failed"} or event.get("error"):
        return "MODEL_ERROR"
    item = event.get("item") or {}
    item_type = item.get("type", "") if isinstance(item, dict) else ""
    if item_type in TOOL_TYPES or item_type.endswith("_tool_call"):
        return "TOOL_USE"
    return None


class SubprocessRunner:
    """Only owns processes it starts; no shell command strings or global kills."""

    def __init__(self):
        self._processes = set()
        self._mutex = threading.Lock()

    def terminate_all(self):
        with self._mutex:
            for process in list(self._processes):
                if process.poll() is None:
                    process.terminate()

    def __call__(self, command, prompt, attempt, timeout, stop_event, cancel_event, cwd):
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        environment = dict(os.environ)
        for key in ("CODEX_API_KEY", "OPENAI_API_KEY"):
            environment.pop(key, None)
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, cwd=cwd, shell=False, creationflags=flags,
            env=environment,
        )
        with self._mutex:
            self._processes.add(process)
        write_json(attempt / "process.json", {"pid": process.pid, "started_at": utc_now()}, True)

        def drain(pipe, destination, is_stdout):
            with destination.open("xb") as stream:
                for line in iter(pipe.readline, b""):
                    stream.write(line)
                    stream.flush()
                    text = line.decode("utf-8", errors="replace")
                    if is_stdout and text.strip():
                        try:
                            if event_failure(json.loads(text)):
                                stop_event.set()
                        except (ValueError, AttributeError):
                            stop_event.set()
                    elif FAILURE_WORDS.search(text):
                        stop_event.set()
            pipe.close()

        threads = [
            threading.Thread(target=drain, args=(process.stdout, attempt / "raw_stdout.jsonl", True)),
            threading.Thread(target=drain, args=(process.stderr, attempt / "stderr.txt", False)),
        ]
        for thread in threads:
            thread.start()
        expired = False
        try:
            try:
                process.stdin.write(prompt.encode("utf-8"))
                process.stdin.close()
            except BrokenPipeError:
                pass
            started = time.monotonic()
            while process.poll() is None:
                if cancel_event.is_set() or time.monotonic() - started >= timeout:
                    expired = True
                    stop_event.set()
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    break
                time.sleep(0.1)
            code = process.wait()
        finally:
            for thread in threads:
                thread.join()
            with self._mutex:
                self._processes.discard(process)
        return {"returncode": code, "interrupted": expired}


def inspect_response(attempt, process_result, requested_model):
    reasons = []
    if process_result.get("interrupted"):
        reasons.append("INTERRUPTED_OR_TIMEOUT")
    if process_result.get("returncode") != 0:
        reasons.append("PROCESS_FAILED")
    try:
        events = read_jsonl(attempt / "raw_stdout.jsonl")
    except (ValueError, UnicodeError):
        events = []
        reasons.append("INVALID_EVENT_STREAM")
    completed = [event for event in events if event.get("type") == "turn.completed"]
    if not completed:
        reasons.append("NO_TURN_COMPLETED")
    for event in events:
        failure = event_failure(event)
        if failure:
            reasons.append(failure)
    stderr_path = attempt / "stderr.txt"
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
    if FAILURE_WORDS.search(stderr):
        reasons.append("ACCESS_OR_RATE_LIMIT")
    final = attempt / "final.txt"
    text = ""
    raw_hash = None
    if final.exists():
        raw = final.read_bytes()
        raw_hash = sha_bytes(raw)
        try:
            text = raw.decode("utf-8")  # Do not normalize newlines or strip the answer.
        except UnicodeError:
            reasons.append("INVALID_FINAL_ENCODING")
    if not text.strip():
        reasons.append("EMPTY_FINAL")
    observed_models = sorted({str(event[key]) for event in events
                              for key in ("model", "model_slug")
                              if isinstance(event.get(key), str)})
    if any(model != requested_model for model in observed_models):
        reasons.append("MODEL_IDENTIFIER_MISMATCH")
    words = len(re.findall(r"\S+", text))
    flags = []
    if not 40 <= words <= 250:
        flags.append("outside_40_250_words")
    return {
        "success": not reasons, "reasons": sorted(set(reasons)), "text": text,
        "text_sha256": sha_bytes(text.encode("utf-8")), "raw_final_sha256": raw_hash,
        "word_count": words, "flags": flags, "observed_models": observed_models,
        "usage": [event.get("usage") for event in completed],
    }


def resolve_path(experiment, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (experiment / path).resolve()


def prepare(experiment, codex):
    experiment = Path(experiment).resolve()
    config_raw = (experiment / "config.json").read_bytes()
    config = json.loads(config_raw.decode("utf-8-sig"))
    prompts_path = resolve_path(experiment, config.get("prompts_file", "prompts.jsonl"))
    prompts_raw = prompts_path.read_bytes()
    prompts = read_jsonl(prompts_path)
    models = config["models"]
    if not prompts or not models:
        raise ValueError("Nonempty prompts and models are required")
    ids = [p["prompt_id"] for p in prompts]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate prompt_id")
    if len({m["slug"] for m in models}) != len(models):
        raise ValueError("Duplicate model slug")
    for prompt in prompts:
        if not isinstance(prompt.get("prompt"), str) or not prompt["prompt"].strip():
            raise ValueError("Every prompt must contain nonempty text")
        if not prompt.get("prompt_group") or prompt.get("split") not in {"train", "holdout"}:
            raise ValueError("Every prompt needs prompt_group and train/holdout split")
    instructions = resolve_path(experiment, config["instructions_file"])
    workspace = resolve_path(experiment, config["workspace"])
    if not workspace.is_dir() or not instructions.is_file():
        raise ValueError("Configured workspace/instructions must already exist")
    overrides = config.get("cli_overrides", [])
    if not isinstance(overrides, list) or not all(isinstance(v, str) and "=" in v for v in overrides):
        raise ValueError("cli_overrides must be TOML key=value strings")
    # Later -c arguments must never silently replace a job's model or instructions.
    protected = {"model", "model_reasoning_effort", "model_instructions_file", "approval_policy", "sandbox_mode"}
    if any(v.split("=", 1)[0].strip() in protected for v in overrides):
        raise ValueError("cli_overrides cannot replace per-job model/instructions/security settings")
    binding = {"config_sha256": sha_bytes(config_raw), "prompts_sha256": sha_bytes(prompts_raw),
               "instructions_sha256": sha_bytes(instructions.read_bytes()),
               "codex_path": str(Path(codex).resolve())}
    fingerprint = sha_bytes(json.dumps(binding, sort_keys=True).encode("utf-8"))
    jobs = []
    for prompt in prompts:
        for model in models:
            key = sha_bytes((fingerprint + "\0" + prompt["prompt_id"] + "\0" + model["slug"]).encode("utf-8"))
            jobs.append({"key": key, "prompt": prompt, "model": model})
    return experiment, config, instructions, workspace, binding, jobs


def load_previous(collection, jobs, binding):
    binding_path = collection / "binding.json"
    if binding_path.exists() and json.loads(binding_path.read_text(encoding="utf-8")) != binding:
        raise InspectionRequired("Inputs changed since collection began; create a new experiment")
    records = read_jsonl(collection / "records.jsonl")
    job_keys = {job["key"] for job in jobs}
    seen = set()
    for record in records:
        key = record["job_key"]
        if key in seen or key not in job_keys:
            raise InspectionRequired("Duplicate or unknown saved record")
        if sha_bytes(record["text"].encode("utf-8")) != record["text_sha256"]:
            raise InspectionRequired("Saved text hash mismatch")
        seen.add(key)
    attempts = collection / "attempts"
    if attempts.exists():
        for path in attempts.iterdir():
            if not path.is_dir():
                raise InspectionRequired("Unexpected file in attempts directory")
            result_file = path / "result.json"
            if not result_file.exists():
                raise InspectionRequired("Unfinished attempt exists; do not replay automatically")
            result = json.loads(result_file.read_text(encoding="utf-8"))
            if result.get("status") != "SUCCEEDED" or result.get("job_key") not in seen:
                raise InspectionRequired("Failed or uncommitted attempt exists; inspect before continuing")
    for key in seen:
        result_file = attempts / key / "result.json"
        if not result_file.exists():
            raise InspectionRequired("Saved record lacks its original successful attempt")
    return records, seen


def run_collection(experiment, codex, *, limit=1000, workers=2, run=False,
                   max_seconds=21600, smoke_per_model=None, runner=None):
    if not 1 <= workers <= 2 or not 1 <= limit <= 1000:
        raise ValueError("workers must be 1..2 and limit must be 1..1000")
    if max_seconds <= 0 or (smoke_per_model is not None and smoke_per_model < 1):
        raise ValueError("Time and smoke limits must be positive")
    experiment, config, instructions, workspace, binding, jobs = prepare(experiment, codex)
    collection = experiment / "collection"
    if runner is None and not Path(codex).is_file():
        raise ValueError("Codex executable does not exist")
    runner = runner or SubprocessRunner()
    records, seen = load_previous(collection, jobs, binding)
    selected = jobs
    if smoke_per_model:
        counts = {}
        selected = []
        for job in jobs:
            slug = job["model"]["slug"]
            counts[slug] = counts.get(slug, 0) + 1
            if counts[slug] <= smoke_per_model:
                selected.append(job)
    target_keys = {job["key"] for job in selected[:limit]}
    pending = [job for job in selected[:limit] if job["key"] not in seen]
    plan = {"status": "DRY_RUN", "planned_total": len(jobs), "target_total": len(target_keys),
            "already_saved": len(records), "remaining_in_this_run": len(pending),
            "workers": workers, "controller_automatic_retries": 0,
            "cli_internal_retry_policy": "see frozen cli_overrides; not inferred",
            "observed_at": utc_now()}
    if not run:
        return plan
    collection.mkdir(parents=True, exist_ok=True)
    with RunLock(collection / "run.lock"):
        # Recheck after acquiring the lock to avoid a stale pre-lock resume view.
        records, seen = load_previous(collection, jobs, binding)
        pending = [job for job in selected[:limit] if job["key"] not in seen]
        if not (collection / "binding.json").exists():
            write_json(collection / "binding.json", binding, True)
        (collection / "attempts").mkdir(exist_ok=True)
        stop_event, cancel_event = threading.Event(), threading.Event()
        started = utc_now()
        monotonic_start = time.monotonic()
        failures = []
        status = "RUNNING"
        iterator = iter(pending)
        active = {}
        launched = 0
        run_id = uuid.uuid4().hex

        def snapshot():
            value = {**plan, "status": status, "run_id": run_id, "controller_pid": os.getpid(),
                     "started_at": started, "observed_at": utc_now(), "saved_total": len(records),
                     "saved_in_target": len(seen & target_keys), "launched_this_run": launched,
                     "failed_this_run": len(failures), "failures": failures,
                     "active": [job["key"] for job in active.values()],
                     "elapsed_seconds": round(time.monotonic() - monotonic_start, 3)}
            write_json(collection / "state.json", value)
            return value

        def execute(job):
            attempt = collection / "attempts" / job["key"]
            attempt.mkdir()  # A previous request can never be silently overwritten/replayed.
            model = job["model"]
            begin = utc_now()
            command = [str(codex), "exec", "--ignore-user-config", "--ephemeral", "--json",
                       "--skip-git-repo-check", "-s", "read-only", "-m", model["slug"],
                       "-c", "approval_policy=\"never\"",
                       "-c", "model_reasoning_effort=" + json.dumps(model["reasoning_effort"]),
                       "-c", "model_instructions_file=" + json.dumps(str(instructions)),
                       "-o", str(attempt / "final.txt")]
            for override in config.get("cli_overrides", []):
                command += ["-c", override]
            command += ["-"]
            write_json(attempt / "request.json", {"job_key": job["key"], "model": model,
                "prompt": job["prompt"], "command": command, "started_at": begin,
                "controller_pid": os.getpid(), "run_id": run_id}, True)
            try:
                outcome = runner(command, job["prompt"]["prompt"], attempt,
                                 min(180.0, float(config.get("request_timeout_seconds",
                                                            config.get("process_timeout_seconds", 180)))),
                                 stop_event, cancel_event, str(workspace))
                response = inspect_response(attempt, outcome, model["slug"])
            except Exception as exc:
                # Exception text can contain service responses. Store only the class in summaries.
                outcome = {"returncode": None}
                response = {"success": False, "reasons": ["CONTROLLER_" + type(exc).__name__]}
            end = utc_now()
            if not response["success"]:
                stop_event.set()
            result = {"job_key": job["key"], "status": "SUCCEEDED" if response["success"] else "FAILED",
                      "started_at": begin, "ended_at": end, "returncode": outcome.get("returncode"),
                      "reasons": response["reasons"]}
            if response["success"]:
                prompt = job["prompt"]
                record = {**prompt, "job_key": job["key"], "source": SOURCE,
                          "model_requested": model["slug"], "model_observed": response["observed_models"],
                          "model_observation_scope": "cli_event" if response["observed_models"] else "requested_only",
                          "reasoning_effort": model["reasoning_effort"], "started_at": begin, "ended_at": end,
                          "prompt_sha256": sha_bytes(prompt["prompt"].encode("utf-8")),
                          "text": response["text"], "text_sha256": response["text_sha256"],
                          "raw_final_sha256": response["raw_final_sha256"], "word_count": response["word_count"],
                          "flags": response["flags"], "usage": response["usage"],
                          "attempt_path": str(attempt.relative_to(experiment)), "binding": binding}
                write_json(attempt / "record.json", record, True)
                result["text_sha256"] = record["text_sha256"]
            else:
                record = None
            write_json(attempt / "result.json", result, True)
            return result, record

        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            exhausted = False
            while True:
                if time.monotonic() - monotonic_start >= max_seconds:
                    status = "STOPPED_TIME_BUDGET"
                    stop_event.set()
                    cancel_event.set()
                while not exhausted and not stop_event.is_set() and len(active) < workers:
                    job = next(iterator, None)
                    if job is None:
                        exhausted = True
                        break
                    active[pool.submit(execute, job)] = job
                    launched += 1
                snapshot()
                if not active:
                    break
                done, _ = wait(active, timeout=1.0, return_when=FIRST_COMPLETED)
                for future in done:
                    job = active.pop(future)
                    try:
                        result, record = future.result()
                    except Exception as exc:
                        stop_event.set()
                        result = {"job_key": job["key"], "status": "UNCERTAIN",
                                  "reasons": ["CONTROLLER_" + type(exc).__name__]}
                        record = None
                    if record is not None:
                        append_jsonl(collection / "records.jsonl", record)
                        records.append(record)
                        seen.add(job["key"])
                    else:
                        failures.append(result)
                if failures:
                    stop_event.set()
                    status = "STOPPED_FAILURE"
            if status == "RUNNING":
                status = "COMPLETE" if target_keys <= seen else "STOPPED_SIGNAL"
        except KeyboardInterrupt:
            status = "INTERRUPTED"
            stop_event.set()
            cancel_event.set()
            if hasattr(runner, "terminate_all"):
                runner.terminate_all()
            # Preserve completed in-flight answers, but never submit another request.
            for future, job in list(active.items()):
                try:
                    result, record = future.result()
                    if record is not None:
                        append_jsonl(collection / "records.jsonl", record)
                        records.append(record)
                        seen.add(job["key"])
                    else:
                        failures.append(result)
                except Exception as exc:
                    failures.append({"job_key": job["key"], "status": "UNCERTAIN",
                                     "reasons": ["CONTROLLER_" + type(exc).__name__]})
            active.clear()
        finally:
            pool.shutdown(wait=True)
        summary = snapshot()
        write_json(collection / "summary.json", summary)
        write_json(collection / ("run-" + run_id + ".json"), summary, True)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--codex", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-seconds", type=float, default=21600)
    parser.add_argument("--smoke-per-model", type=int)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--run", action="store_true", help="Actually call Codex; default is dry-run")
    modes.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_collection(args.experiment, args.codex, limit=args.limit, workers=args.workers,
                                max_seconds=args.max_seconds, smoke_per_model=args.smoke_per_model, run=args.run)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] in {"DRY_RUN", "COMPLETE"} else 2
    except (InspectionRequired, ValueError, OSError, KeyError) as exc:
        # Expected local validation messages never include model/service response bodies.
        print(json.dumps({"status": "NOT_STARTED", "error_type": type(exc).__name__,
                          "message": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
