"""Offline recovery evidence checks; only temporary fixtures are read or written."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch


WRAPPER_PATH = Path(__file__).resolve().parents[1] / "run_and_finalize.py"
# Compile without importing a cached module or writing bytecode beside the wrapper.
wrapper = types.ModuleType("resume_wrapper_under_test")
wrapper.__file__ = str(WRAPPER_PATH)
exec(compile(WRAPPER_PATH.read_text(encoding="utf-8"), str(WRAPPER_PATH), "exec"),
     wrapper.__dict__)


def sha(data):
    return hashlib.sha256(data).hexdigest()


class ResumeEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "project"
        self.exp = self.root / "experiments" / "new-resume"
        self.data_exp = self.root / "experiments" / "shared-collection"
        (self.exp / "metrics").mkdir(parents=True)
        (self.data_exp / "collection").mkdir(parents=True)
        self.records = self.data_exp / "collection" / "records.jsonl"
        self.prefix = b"".join(
            (json.dumps({"job_key": f"saved-{i}", "text": f"Response {i}"}) + "\n").encode()
            for i in range(558)
        )
        self.records.write_bytes(self.prefix)
        self.receipt = {
            "initial_records": {"bytes": len(self.prefix), "sha256": sha(self.prefix), "count": 558},
            "archived_attempts": [],
        }
        self.archives = []
        for index in range(2):
            archive = self.exp / "recovery" / "failed-attempts" / f"failed-{index}"
            archive.mkdir(parents=True)
            result = {
                "job_key": f"failed-{index}", "status": "FAILED", "returncode": 1,
                "started_at": "2026-11-01T10:00:00+00:00",
                "ended_at": "2026-11-01T10:03:00+00:00",
                "reasons": ["EMPTY_FINAL", "NO_TURN_COMPLETED"],
            }
            (archive / "result.json").write_text(json.dumps(result), encoding="utf-8")
            (archive / "raw_stdout.jsonl").write_text('{"type":"turn.started"}\n', encoding="utf-8")
            (archive / "stderr.txt").write_text("synthetic network timeout\n", encoding="utf-8")
            self.receipt["archived_attempts"].append({
                "archive_path": archive.relative_to(self.root).as_posix(),
                "files": {p.name: sha(p.read_bytes()) for p in archive.iterdir()},
                "result": result,
            })
            self.archives.append(archive)
        self.save_receipt()
        globals_patch = patch.multiple(wrapper, ROOT=self.root, EXP=self.exp, DATA_EXP=self.data_exp)
        globals_patch.start()
        self.addCleanup(globals_patch.stop)
        process_guard = patch.object(wrapper.subprocess, "run", side_effect=AssertionError(
            "Offline evidence tests must not launch any process"))
        process_guard.start()
        self.addCleanup(process_guard.stop)

    def save_receipt(self):
        (self.exp / "metrics" / "resume-preflight.json").write_text(
            json.dumps(self.receipt), encoding="utf-8")

    def test_preserves_all_558_original_records_and_failed_evidence(self):
        count, results = wrapper.verify_resume_evidence()
        self.assertEqual(count, 558)
        self.assertEqual(results, [entry["result"] for entry in self.receipt["archived_attempts"]])
        self.assertEqual(self.records.read_bytes(), self.prefix)

    def test_appending_remaining_442_records_is_allowed(self):
        suffix = b"".join((json.dumps({"job_key": f"new-{i}"}) + "\n").encode()
                          for i in range(442))
        self.records.write_bytes(self.prefix + suffix)
        count, failures = wrapper.verify_resume_evidence()
        self.assertEqual(count, 558)
        self.assertEqual(len(failures), 2)
        self.assertEqual(len(self.records.read_bytes().splitlines()), 1000)
        self.assertEqual(self.records.read_bytes(), self.prefix + suffix)

    def test_prefix_byte_change_is_rejected(self):
        changed = self.prefix.replace(b"Response 0", b"Response X", 1)
        self.assertEqual(len(changed), len(self.prefix))
        self.records.write_bytes(changed)
        with self.assertRaisesRegex(RuntimeError, "prefix changed"):
            wrapper.verify_resume_evidence()

    def test_truncated_prefix_is_rejected(self):
        self.records.write_bytes(self.prefix[:-1])
        with self.assertRaisesRegex(RuntimeError, "prefix changed"):
            wrapper.verify_resume_evidence()

    def test_wrong_original_row_count_is_rejected(self):
        self.receipt["initial_records"]["count"] = 557
        self.save_receipt()
        with self.assertRaisesRegex(RuntimeError, "prefix changed"):
            wrapper.verify_resume_evidence()

    def test_tampered_archive_file_is_rejected(self):
        (self.archives[0] / "stderr.txt").write_text("changed evidence", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "evidence changed"):
            wrapper.verify_resume_evidence()

    def test_missing_archive_file_is_rejected(self):
        (self.archives[1] / "raw_stdout.jsonl").unlink()
        with self.assertRaisesRegex(RuntimeError, "evidence changed"):
            wrapper.verify_resume_evidence()

    def test_unrecorded_archive_file_is_rejected(self):
        (self.archives[0] / "unexpected.txt").write_text("extra", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "evidence changed"):
            wrapper.verify_resume_evidence()

    def test_external_archive_paths_are_rejected(self):
        paths = [
            "outside/failed-0",
            "experiments/new-resume/recovery/failed-attempts/../../outside",
            str(self.root.parent / "outside-project"),
        ]
        for path in paths:
            with self.subTest(path=path):
                self.receipt["archived_attempts"][0]["archive_path"] = path
                self.save_receipt()
                with self.assertRaisesRegex(RuntimeError, "outside this experiment"):
                    wrapper.verify_resume_evidence()

    def test_failure_result_must_match_inspected_receipt(self):
        self.receipt["archived_attempts"][0]["result"]["returncode"] = 9
        self.save_receipt()
        with self.assertRaisesRegex(RuntimeError, "not the inspected failure"):
            wrapper.verify_resume_evidence()

    def test_successful_attempt_cannot_be_classified_as_historical_failure(self):
        entry = self.receipt["archived_attempts"][0]
        entry["result"]["status"] = "SUCCEEDED"
        result_path = self.archives[0] / "result.json"
        result_path.write_text(json.dumps(entry["result"]), encoding="utf-8")
        entry["files"]["result.json"] = sha(result_path.read_bytes())
        self.save_receipt()
        with self.assertRaisesRegex(RuntimeError, "not the inspected failure"):
            wrapper.verify_resume_evidence()


class TerminalEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.terminal = {"status": "COMPLETE", "run_id": "current-run",
                         "target_total": 1000, "saved_total": 1000, "active": []}

    def test_matching_full_run_with_successful_exit_is_accepted(self):
        self.assertTrue(wrapper.terminal_matches(self.terminal, copy.deepcopy(self.terminal), 0))

    def test_old_run_summary_cannot_complete_current_run(self):
        summary = {**self.terminal, "run_id": "old-smoke-run"}
        self.assertFalse(wrapper.terminal_matches(summary, self.terminal, 0))

    def test_smoke_or_incomplete_counts_are_rejected(self):
        for side in ("summary", "state"):
            for key, value in (("target_total", 4), ("saved_total", 4), ("saved_total", 558)):
                with self.subTest(side=side, key=key, value=value):
                    summary, state = copy.deepcopy(self.terminal), copy.deepcopy(self.terminal)
                    (summary if side == "summary" else state)[key] = value
                    self.assertFalse(wrapper.terminal_matches(summary, state, 0))

    def test_failed_exit_active_requests_or_missing_run_id_are_rejected(self):
        self.assertFalse(wrapper.terminal_matches(self.terminal, self.terminal, 2))
        for side in ("summary", "state"):
            for key, value in (("active", ["pending-job"]), ("run_id", ""),
                               ("status", "STOPPED_FAILURE")):
                with self.subTest(side=side, key=key):
                    summary, state = copy.deepcopy(self.terminal), copy.deepcopy(self.terminal)
                    (summary if side == "summary" else state)[key] = value
                    self.assertFalse(wrapper.terminal_matches(summary, state, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
