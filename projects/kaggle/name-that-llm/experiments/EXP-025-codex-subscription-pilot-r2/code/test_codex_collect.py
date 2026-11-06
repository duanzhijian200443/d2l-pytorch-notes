"""Offline behavioral checks: fake runner only, never real Codex inference."""

import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import codex_collect as collector


class FakeRunner:
    def __init__(self, mode="success", text="First paragraph.\r\n\r\nSecond paragraph.\n"):
        self.mode, self.text = mode, text
        self.calls = []
        self.active = self.maximum = 0
        self.lock = threading.Lock()

    def __call__(self, command, prompt, attempt, timeout, stop_event, cancel_event, cwd):
        with self.lock:
            self.calls.append((command, prompt))
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            number = len(self.calls)
        try:
            if self.mode == "fail_first" and number == 1:
                stop_event.set()
                events = [{"type": "turn.failed", "error": {"message": "test failure"}}]
                code = 1
            else:
                time.sleep(0.025)
                events = [{"type": "thread.started", "thread_id": "test"}]
                if self.mode != "no_completed":
                    events.append({"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 2}})
                if self.mode == "tool":
                    events.insert(1, {"type": "item.completed", "item": {"type": "command_execution"}})
                if self.mode == "model_mismatch":
                    events[0]["model"] = "different-model"
                code = 0
            (attempt / "raw_stdout.jsonl").write_text(
                "".join(json.dumps(e) + "\n" for e in events), encoding="utf-8"
            )
            (attempt / "stderr.txt").write_text("", encoding="utf-8")
            (attempt / "final.txt").write_bytes(("" if self.mode == "empty" else self.text).encode("utf-8"))
            return {"returncode": code}
        finally:
            with self.lock:
                self.active -= 1


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.exp = Path(self.temp.name)
        (self.exp / "workspace").mkdir()
        (self.exp / "instructions.txt").write_text("Answer the request.", encoding="utf-8")
        self.config = {
            "models": [{"slug": "test-a", "reasoning_effort": "low"},
                       {"slug": "test-b", "reasoning_effort": "low"}],
            "instructions_file": "instructions.txt", "workspace": "workspace",
            "cli_overrides": ["features.apps=false"],
        }
        (self.exp / "config.json").write_text(json.dumps(self.config), encoding="utf-8")
        prompts = [{"prompt_id": f"p{i}", "prompt_group": f"p{i}", "split": "train",
                    "prompt": f"Test prompt {i}"} for i in range(5)]
        (self.exp / "prompts.jsonl").write_text(
            "".join(json.dumps(p) + "\n" for p in prompts), encoding="utf-8"
        )

    def tearDown(self):
        self.temp.cleanup()

    def run_fake(self, runner, **kwargs):
        return collector.run_collection(self.exp, "fake-codex.exe", runner=runner, run=True, **kwargs)

    def test_default_dry_run_makes_no_request_or_artifact(self):
        runner = FakeRunner()
        result = collector.run_collection(self.exp, "fake-codex.exe", runner=runner)
        self.assertEqual(result["status"], "DRY_RUN")
        self.assertEqual(result["planned_total"], 10)
        self.assertFalse(runner.calls)
        self.assertFalse((self.exp / "collection").exists())

    def test_reuse_frozen_prompt_file_without_copy(self):
        source = self.exp / "frozen-prompts.jsonl"
        (self.exp / "prompts.jsonl").rename(source)
        self.config["prompts_file"] = source.name
        (self.exp / "config.json").write_text(json.dumps(self.config), encoding="utf-8")
        result = collector.run_collection(self.exp, "fake-codex.exe", runner=FakeRunner())
        self.assertEqual(result["planned_total"], 10)
        self.assertFalse((self.exp / "collection").exists())

    def test_concurrency_limit_and_literal_newlines(self):
        runner = FakeRunner()
        result = self.run_fake(runner, limit=8, workers=2)
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(len(runner.calls), 8)
        self.assertEqual(runner.maximum, 2)
        records = collector.read_jsonl(self.exp / "collection" / "records.jsonl")
        self.assertTrue(all(r["text"] == runner.text for r in records))
        self.assertTrue(all(r["source"] == "codex_subscription" for r in records))
        self.assertFalse((self.exp / "collection" / "run.lock").exists())

    def test_failed_request_stops_new_dispatch_and_blocks_resume(self):
        runner = FakeRunner(mode="fail_first")
        result = self.run_fake(runner, workers=2)
        self.assertEqual(result["status"], "STOPPED_FAILURE")
        self.assertLessEqual(len(runner.calls), 2)
        second = FakeRunner()
        with self.assertRaises(collector.InspectionRequired):
            self.run_fake(second)
        self.assertFalse(second.calls)

    def test_resume_skips_committed_requests(self):
        runner = FakeRunner()
        first = self.run_fake(runner, limit=3)
        self.assertEqual(first["saved_total"], 3)
        second = FakeRunner()
        result = self.run_fake(second, limit=7)
        self.assertEqual(len(second.calls), 4)
        self.assertEqual(result["saved_total"], 7)
        again = FakeRunner()
        self.run_fake(again, limit=7)
        self.assertFalse(again.calls)

    def test_empty_response_and_missing_completed_are_not_success(self):
        for mode in ("empty", "no_completed", "tool", "model_mismatch"):
            with self.subTest(mode=mode):
                # Independent sub-experiments leave failure evidence intact.
                sub = self.exp / mode
                sub.mkdir()
                (sub / "workspace").mkdir()
                for name in ("config.json", "prompts.jsonl", "instructions.txt"):
                    (sub / name).write_bytes((self.exp / name).read_bytes())
                result = collector.run_collection(sub, "fake", run=True, workers=1, runner=FakeRunner(mode))
                self.assertEqual(result["status"], "STOPPED_FAILURE")
                self.assertEqual(result["saved_total"], 0)

    def test_orphan_attempt_cannot_be_replayed(self):
        path = self.exp / "collection" / "attempts" / "unfinished"
        path.mkdir(parents=True)
        runner = FakeRunner()
        with self.assertRaises(collector.InspectionRequired):
            self.run_fake(runner)
        self.assertFalse(runner.calls)

    def test_lock_is_not_stolen_even_if_pid_looks_stale(self):
        collection = self.exp / "collection"
        collection.mkdir()
        original = {"pid": 99999999, "token": "another-controller"}
        collector.write_json(collection / "run.lock", original)
        runner = FakeRunner()
        with self.assertRaises(collector.InspectionRequired):
            self.run_fake(runner)
        self.assertFalse(runner.calls)
        self.assertEqual(json.loads((collection / "run.lock").read_text()), original)

    def test_smoke_then_full_resumes_same_jobs(self):
        runner = FakeRunner()
        self.run_fake(runner, smoke_per_model=1)
        self.assertEqual(len(runner.calls), 2)
        second = FakeRunner()
        result = self.run_fake(second)
        self.assertEqual(len(second.calls), 8)
        self.assertEqual(result["saved_total"], 10)

    def test_modified_input_fails_closed(self):
        self.run_fake(FakeRunner(), limit=1)
        with (self.exp / "instructions.txt").open("a") as stream:
            stream.write(" Changed.")
        runner = FakeRunner()
        with self.assertRaises(collector.InspectionRequired):
            self.run_fake(runner)
        self.assertFalse(runner.calls)

    def test_no_worker_or_total_budget_escalation(self):
        for kwargs in ({"workers": 3}, {"limit": 1001}, {"max_seconds": 0}):
            with self.assertRaises(ValueError):
                self.run_fake(FakeRunner(), **kwargs)

    def test_atomic_replace_recovers_from_transient_permission_error(self):
        destination = self.exp / "replace-test.json"
        real_replace = collector.os.replace
        calls = []

        def transient(source, target):
            calls.append((source, target))
            if len(calls) == 1:
                raise PermissionError(5, "simulated Windows reader lock")
            return real_replace(source, target)

        with patch.object(collector.os, "replace", side_effect=transient), \
                patch.object(collector.time, "sleep") as delay:
            collector.write_json(destination, {"complete": True})
        self.assertEqual(json.loads(destination.read_text()), {"complete": True})
        self.assertEqual(len(calls), 2)
        delay.assert_called_once_with(0.05)

    def test_persistent_state_replace_error_drains_inflight_without_new_requests(self):
        runner = FakeRunner()
        real_replace = collector.os.replace
        blocked = []

        def deny_state(source, target):
            if Path(target).name == "state.json":
                blocked.append(target)
                raise PermissionError(5, "simulated persistent Windows reader lock")
            return real_replace(source, target)

        # Do not mock sleep globally: it also controls FakeRunner concurrency.
        with patch.object(collector.os, "replace", side_effect=deny_state), \
                patch.object(collector, "REPLACE_RETRY_DELAYS", (0.001, 0.002)):
            result = self.run_fake(runner, workers=2)
        self.assertEqual(result["status"], "LOCAL_ERROR")
        self.assertEqual(len(runner.calls), 2)
        self.assertEqual(result["saved_total"], 2)
        self.assertTrue(result["local_errors"])
        records = collector.read_jsonl(self.exp / "collection" / "records.jsonl")
        self.assertEqual(len(records), 2)
        for row in records:
            attempt = self.exp / row["attempt_path"]
            self.assertEqual(json.loads((attempt / "result.json").read_text())["status"], "SUCCEEDED")
            self.assertEqual((attempt / "final.txt").read_bytes().decode("utf-8"), runner.text)
        self.assertEqual(json.loads((self.exp / "collection" / "summary.json").read_text())["status"], "LOCAL_ERROR")
        self.assertFalse((self.exp / "collection" / "run.lock").exists())
        # After an explicit restart with the local disk issue gone, both known
        # successful answers are reused; only the third request is sent.
        next_runner = FakeRunner()
        resumed = self.run_fake(next_runner, workers=1, limit=3)
        self.assertEqual(len(next_runner.calls), 1)
        self.assertEqual(resumed["saved_total"], 3)


if __name__ == "__main__":
    unittest.main()
