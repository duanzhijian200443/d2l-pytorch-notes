"""Run the authorized bounded collection, then save integrity metrics and ledger state."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


EXP = Path(__file__).resolve().parent
ROOT = EXP.parent.parent
DATA_EXP = ROOT / "experiments/EXP-023-codex-subscription-pilot-r1"
CODEX = Path(r"C:\Users\34331\AppData\Local\OpenAI\Codex\bin\22cfb1d302613eae\codex.exe")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for delay in (0.05, 0.10, 0.20, 0.40, 0.80, None):
        try:
            tmp.replace(path)
            break
        except PermissionError:
            if delay is None:
                raise
            time.sleep(delay)


def terminal_matches(summary, state, returncode):
    return (returncode == 0 and summary.get("status") == state.get("status") == "COMPLETE"
            and bool(summary.get("run_id")) and summary.get("run_id") == state.get("run_id")
            and summary.get("target_total") == state.get("target_total") == 1000
            and summary.get("saved_total") == state.get("saved_total") == 1000
            and not state.get("active") and not summary.get("active"))


def finalize(collector_returncode):
    now = datetime.now(timezone.utc).isoformat()
    collection = DATA_EXP / "collection"
    if (collection / "run.lock").exists():
        raise RuntimeError("Collector is active or unsettled; do not finalize another process")
    summary = load(collection / "summary.json")
    state = load(collection / "state.json")
    rows = [json.loads(line) for line in (collection / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()] if (collection / "records.jsonl").exists() else []
    by_model = Counter(row["model_requested"] for row in rows)
    by_split = Counter(row["split"] for row in rows)
    flags = Counter(flag for row in rows for flag in row["flags"])
    groups = defaultdict(set)
    hash_errors = []
    events = []
    for row in rows:
        groups[row["prompt_group"]].add(row["split"])
        if hashlib.sha256(row["text"].encode("utf-8")).hexdigest() != row["text_sha256"]:
            hash_errors.append(row["job_key"])
    attempts = list((collection / "attempts").glob("*/result.json"))
    attempts_by_status = Counter()
    for path in attempts:
        result = load(path)
        attempts_by_status[result["status"]] += 1
        events.extend([(result["started_at"], 1), (result["ended_at"], -1)])
    active = peak = 0
    for _, delta in sorted(events):
        active += delta
        peak = max(peak, active)
    duplicates = len(rows) - len({row["job_key"] for row in rows})
    split_leaks = [key for key, value in groups.items() if len(value) != 1]
    complete = (terminal_matches(summary, state, collector_returncode) and len(rows) == 1000
                and len(by_model) == 4 and set(by_model.values()) == {250}
                and by_split == {"train": 800, "holdout": 200}
                and not hash_errors and not duplicates and not split_leaks and peak <= 2
                and len(attempts) == 1000 and attempts_by_status == {"SUCCEEDED": 1000}
                and len(list((collection / "attempts").iterdir())) == len(attempts))
    status = "COLLECTED_UNVALIDATED" if complete else "STOPPED_INCOMPLETE"
    audit = {"observed_at": now, "status": status, "responses": len(rows),
             "by_model": dict(by_model), "by_split": dict(by_split), "flags": dict(flags),
             "prompt_groups": len(groups), "duplicate_job_keys": duplicates,
             "duplicate_text_rows": len(rows) - len({r["text_sha256"] for r in rows}),
             "text_hash_errors": hash_errors, "split_leaks": split_leaks,
             "peak_attempt_concurrency": peak, "attempts_by_status": dict(attempts_by_status),
             "controller_status": state["status"], "collector_returncode": collector_returncode,
             "summary_run_id": summary.get("run_id"), "state_run_id": state.get("run_id"), "training_acceptance": "NOT_EVALUATED",
             "source": "codex_subscription", "model_snapshot_verification": "NOT_ESTABLISHED"}
    save(EXP / "metrics" / "collection-audit.json", audit)
    record_path = ROOT / "tracking" / "records" / "experiments" / "EXP-025.json"
    item = load(record_path)
    item["observed_at"] = now
    item["execution"].update(status=status, observed_at=now, responses=len(rows), controller_status=summary["status"])
    item["conclusion"] = (f"Saved {len(rows)} raw Codex-source responses; integrity complete={complete}. Training utility and API equivalence not evaluated.")
    item["next_action"] = "Compare against source-controlled API data on prompt-isolated validation." if complete else "Inspect original failed/uncertain attempts before any new request; do not replay automatically."
    for filename in ["collection/records.jsonl", "collection/summary.json", "metrics/collection-audit.json", "run_and_finalize.py"]:
        origin = DATA_EXP if filename.startswith("collection/") else EXP
        relative = (origin / filename).relative_to(ROOT).as_posix()
        if (origin / filename).exists() and relative not in item["evidence"]:
            item["evidence"].append(relative)
    save(record_path, item)
    report = EXP / "report.md"
    body = report.read_text(encoding="utf-8")
    body = body.split("\n## 自动收尾结果\n", 1)[0]
    body += f"\n## 自动收尾结果\n\n观察时间：{now}。状态：**{status}**。保存 **{len(rows)}/1000** 条。\n\n"
    body += "| 请求模型 | 原始回答数 |\n|---|---:|\n"
    for model, count in sorted(by_model.items()):
        body += f"| {model} | {count} |\n"
    body += f"\ntrain/holdout：{dict(by_split)}；观测最大请求并发：{peak}；字数等标记：{dict(flags)}。\n"
    body += "\n[完整性审计](metrics/collection-audit.json) · [原始数据](../EXP-023-codex-subscription-pilot-r1/collection/records.jsonl) · [控制器结果](../EXP-023-codex-subscription-pilot-r1/collection/summary.json)\n\n这些是 Codex 订阅管线原文；没有进行训练、榜单提交、公开发布或 API 等价性验收。\n"
    report.write_text(body, encoding="utf-8")
    checks = {}
    for command in ["check", "render"]:
        done = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "tracking/ledger.py"), command], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        checks[command] = {"returncode": done.returncode, "stdout": done.stdout, "stderr": done.stderr}
    save(EXP / "metrics" / "ledger-finalization.json", checks)
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--collector-returncode", type=int)
    args = parser.parse_args()
    returncode = args.collector_returncode
    if not args.finalize_only:
        command = [sys.executable, "-X", "utf8", str(EXP / "code/codex_collect.py"),
                   "--experiment", str(DATA_EXP), "--codex", str(CODEX),
                   "--workers", "2", "--limit", "1000", "--max-seconds", "21600", "--run"]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        with (EXP / "logs" / f"controller-{stamp}.log").open("xb") as stream:
            done = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            returncode = done.returncode
    result = finalize(returncode)
    print(json.dumps({"status": result["status"], "responses": result["responses"]}))


if __name__ == "__main__":
    main()
