"""Small regression checks for the mistakes this ledger is intended to prevent."""
import copy
import contextlib
import hashlib
import io
import json
import pathlib
import tempfile
import unittest
from unittest import mock
import ledger


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.exps = ledger.records("experiments")
        self.sources = ledger.records("sources")
        self.ideas = ledger.read(ledger.HERE / "ideas.json")
        self.settings = ledger.read(ledger.HERE / "settings.json")

    def check(self, exps=None, sources=None):
        return ledger.validate(exps or self.exps, sources or self.sources, self.ideas, self.settings)

    def test_existing_records(self):
        self.assertEqual(self.check(), [])

    def test_pending_submission_cannot_have_public_score(self):
        exps = copy.deepcopy(self.exps)
        e = next(x for x in exps if x["id"] == "EXP-005")
        e["submission"].update(status="PENDING", public_score=0.664)
        self.assertTrue(any("score without complete" in x for x in self.check(exps=exps)))

    def test_source_and_scored_version_must_match(self):
        sources = copy.deepcopy(self.sources)
        e = next(x for x in sources if x["id"] == "SRC-021")
        e["source_version"]["script_version_id"] = 1
        self.assertTrue(any("no matching version" in x for x in self.check(sources=sources)))

    def test_commit_complete_is_not_competition_complete(self):
        e = {"execution": {"scope": "commit", "status": "COMPLETE"}, "submission": None}
        self.assertEqual(ledger.state(e), "保存完成；未关联正式提交")
        self.assertTrue(ledger.needs_attention(e))

    def test_completed_commit_does_not_hide_pending_submission(self):
        e = {"execution": {"scope": "commit", "status": "COMPLETE"}, "submission": {"id": 1, "status": "PENDING"}}
        self.assertTrue(ledger.needs_attention(e))

    def test_known_result_does_not_need_runtime_polling(self):
        e = {"execution": {"scope": "commit", "status": "COMPLETE"}, "submission": {"id": 1, "status": "COMPLETE"}}
        self.assertFalse(ledger.needs_attention(e))

    def test_score_outside_metric_range_is_rejected(self):
        sources = copy.deepcopy(self.sources)
        sources[0]["public_score"]["value"] = 1645.5
        self.assertTrue(any("outside [0,1]" in x for x in self.check(sources=sources)))


class IdeaPresentationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ntl-ideas-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = pathlib.Path(self.temporary.name).resolve() / "project"
        self.here = self.root / "tracking"
        (self.here / "records" / "experiments").mkdir(parents=True)
        (self.root / "research").mkdir()
        (self.root / "research" / "notes.md").write_text("Evidence.\n", encoding="utf-8")
        self.experiments = [{"id": "EXP-001", "title": "Baseline",
                             "execution": {"scope": "grouped_oof", "status": "COMPLETE"}}]
        (self.here / "records" / "experiments" / "EXP-001.json").write_text("{}", encoding="utf-8")
        self.settings = {"baseline_experiment_id": "EXP-001"}
        self.idea = {"id": "IDEA-001", "priority": 1, "title": "Candidate", "status": "planned",
                     "sources": [], "next_action": "Read evidence", "acceptance": "Fixed comparison",
                     "budget": "CPU only"}
        patcher = mock.patch.multiple(ledger, ROOT=self.root, HERE=self.here)
        patcher.start()
        self.addCleanup(patcher.stop)

    def render_ideas(self, ideas):
        ledger.render(self.experiments, [], ideas, self.settings)
        return (self.here / "待验证想法.md").read_text(encoding="utf-8")

    def test_optional_details_and_references_are_visible(self):
        self.idea.update(
            baseline_delta="Only the balancing strength changes",
            evidence_assessment="Version-bound community score; no causal proof",
            expected_benefit="Small and uncertain",
            prerequisites="Fixed predictions and label order",
            stop_rule="Stop if output parity fails",
            related_experiments=["EXP-001"], evidence=["research/notes.md"])
        self.assertEqual(ledger.validate(self.experiments, [], [self.idea], self.settings), [])
        view = self.render_ideas([self.idea])
        for text in (
            "- 相对正式基线的增量：Only the balancing strength changes",
            "- 证据与限制：Version-bound community score; no causal proof",
            "- 预期收益：Small and uncertain",
            "- 前置条件：Fixed predictions and label order",
            "- 停止/降级条件：Stop if output parity fails",
            "- 关联实验：[EXP-001](实验台账.md#exp-001)",
            "- 证据：[notes.md](../research/notes.md)",
        ):
            self.assertIn(text, view)

    def test_legacy_ideas_keep_existing_details_and_sort_order(self):
        second = dict(self.idea, id="IDEA-002", priority=0, status="in_progress")
        self.assertEqual(ledger.validate(self.experiments, [], [self.idea, second], self.settings), [])
        view = self.render_ideas([self.idea, second])
        self.assertIn("### IDEA-001 Candidate\n\n- 下一步：Read evidence\n"
                      "- 判断标准：Fixed comparison\n- 成本边界：CPU only\n", view)
        self.assertLess(view.index("### IDEA-002"), view.index("### IDEA-001"))
        self.assertIn("| IDEA-002 | P0 | Candidate | in_progress |", view)
        for label in ("相对正式基线的增量", "证据与限制", "预期收益", "前置条件", "停止/降级条件", "关联实验", "- 证据："):
            self.assertNotIn(label, view)

    def test_unknown_experiment_is_rejected(self):
        self.idea["related_experiments"] = ["EXP-999"]
        errors = ledger.validate(self.experiments, [], [self.idea], self.settings)
        self.assertIn("IDEA-001: unknown experiment EXP-999", errors)

    def test_missing_outside_and_nonfile_evidence_is_rejected(self):
        outside = self.root.parent / "outside.md"
        outside.write_text("Outside project.\n", encoding="utf-8")
        for path in ("research/missing.md", "../outside.md", str(outside),
                     str(self.root / "research" / "notes.md"), "research"):
            with self.subTest(path=path):
                self.idea["evidence"] = [path]
                errors = ledger.validate(self.experiments, [], [self.idea], self.settings)
                self.assertIn(f"IDEA-001: missing/outside evidence {path}", errors)


class NewExperimentTests(unittest.TestCase):
    def setUp(self):
        record_template = ledger.read(ledger.HERE / "templates" / "experiment.json")
        self.temporary = tempfile.TemporaryDirectory(prefix="ntl-ledger-test-")
        self.temporary_root = pathlib.Path(self.temporary.name).resolve()
        self.assertEqual(self.temporary_root.parent, pathlib.Path(tempfile.gettempdir()).resolve())
        self.addCleanup(self.temporary.cleanup)
        self.root = self.temporary_root / "project"
        self.here = self.root / "tracking"
        for directory in (self.here / "templates", self.here / "records" / "experiments",
                          self.here / "records" / "sources", self.root / "configs", self.root / "experiments"):
            directory.mkdir(parents=True, exist_ok=True)
        self.config = {
            "schema_version": 1,
            "model": {"name": None, "parameters": {}},
            "validation": {"split_id": "prompt-group-v1", "seed": 42},
            "environment": {"platform": None, "accelerator": None},
            "budget": {"wall_seconds": None, "gpu_seconds": None, "api_usd": None, "colab_cu": None},
        }
        self.template_path = self.root / "configs" / "experiment-defaults.json"
        self.put_json(self.template_path, self.config)
        self.put_json(self.here / "templates" / "experiment.json", record_template)
        baseline = copy.deepcopy(record_template)
        baseline.update(id="EXP-001", title="Existing baseline")
        self.experiments = [baseline]
        self.settings = {"baseline_experiment_id": "EXP-001"}
        self.put_json(self.here / "records" / "experiments" / "EXP-001.json", baseline)
        self.put_json(self.here / "settings.json", self.settings)
        self.put_json(self.here / "ideas.json", [])
        (self.root / "experiments" / "README.md").write_text("Preserve existing files.\n", encoding="utf-8")
        patcher = mock.patch.multiple(ledger, ROOT=self.root, HERE=self.here)
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def put_json(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def snapshot(self):
        return {path.relative_to(self.root).as_posix(): path.read_bytes() if path.is_file() else None
                for path in self.root.rglob("*")}

    def create(self, **kwargs):
        return ledger.create_experiment(self.experiments, self.settings, "计划实验", **kwargs)

    def test_default_config_creates_complete_planned_workspace(self):
        original_record = (self.here / "records" / "experiments" / "EXP-001.json").read_bytes()
        original_template = self.template_path.read_bytes()
        item = self.create()
        workspace = self.root / "experiments" / "EXP-002"
        self.assertEqual({p.name for p in workspace.iterdir()},
                         {"config.json", "manifest.json", "report.md", "oof", "metrics", "logs"})
        for name in ("oof", "metrics", "logs"):
            self.assertTrue((workspace / name).is_dir())
            self.assertEqual(list((workspace / name).iterdir()), [])
        self.assertEqual((workspace / "config.json").read_bytes(), original_template)
        self.assertEqual(item["config_path"], "experiments/EXP-002/config.json")
        self.assertIn("config_path", item["config"]["note"])
        self.assertEqual(item["execution"]["status"], "PLANNED")
        self.assertFalse(item["inputs"]["content_hashes_verified"])
        self.assertIsNone(item["validation"]["balanced_accuracy"])
        self.assertIsNone(item["submission"])
        self.assertTrue(all(value is None for value in item["cost"].values()))
        manifest = ledger.read(workspace / "manifest.json")
        self.assertEqual(manifest["freeze_status"], "DRAFT")
        self.assertEqual(manifest["inputs"], [])
        self.assertEqual(manifest["code"], [])
        for name in ("config", "config_template"):
            file = self.root / manifest[name]["path"]
            self.assertEqual(manifest[name]["sha256"], hashlib.sha256(file.read_bytes()).hexdigest())
        report = (workspace / "report.md").read_text(encoding="utf-8")
        for text in ("PLANNED", "未运行", "DRAFT", "待填写", "../../tracking/records/experiments/EXP-002.json"):
            self.assertIn(text, report)
        self.assertEqual(ledger.validate(self.experiments + [item], [], [], self.settings), [])
        self.assertEqual(ledger.read(self.here / "records" / "experiments" / "EXP-002.json"), item)
        self.assertEqual((self.here / "records" / "experiments" / "EXP-001.json").read_bytes(), original_record)
        self.assertEqual(self.template_path.read_bytes(), original_template)

    def test_slug_and_custom_config_create_independent_snapshot(self):
        custom = copy.deepcopy(self.config)
        custom["model"] = {"name": "svm", "parameters": {"C": 0.5}}
        custom_path = self.root / "configs" / "custom.json"
        self.put_json(custom_path, custom)
        item = self.create(slug="svm-05", config_path="configs/custom.json")
        self.assertEqual(item["config_path"], "experiments/EXP-002-svm-05/config.json")
        snapshot_path = self.root / item["config_path"]
        self.assertEqual(ledger.read(snapshot_path), custom)
        custom["model"]["parameters"]["C"] = 3
        self.put_json(custom_path, custom)
        self.assertEqual(ledger.read(snapshot_path)["model"]["parameters"]["C"], 0.5)

    def test_duplicate_record_is_not_overwritten(self):
        self.create(slug="first")
        before = self.snapshot()
        with self.assertRaises(FileExistsError):
            self.create(slug="second")
        self.assertEqual(self.snapshot(), before)

    def test_existing_workspace_id_with_any_slug_is_rejected(self):
        for dirname in ("EXP-002", "EXP-002-planned", "EXP-002-another", "exp-002-existing"):
            with self.subTest(dirname=dirname):
                directory = self.root / "experiments" / dirname
                directory.mkdir()
                marker = directory / "keep.txt"
                marker.write_text("existing", encoding="utf-8")
                before = self.snapshot()
                with self.assertRaises(FileExistsError):
                    self.create(slug="planned")
                self.assertEqual(self.snapshot(), before)
                marker.unlink()
                directory.rmdir()

    def test_unsafe_slugs_leave_no_output(self):
        before = self.snapshot()
        for slug in ("", "..", "../outside", "part/child", "part\\child", "/absolute",
                     "C:\\absolute", "Uppercase", "has space", "has_underscore", "bad..name", "-start", "end-"):
            with self.subTest(slug=slug):
                with self.assertRaises(ValueError):
                    self.create(slug=slug)
                self.assertEqual(self.snapshot(), before)

    def test_absolute_and_traversing_config_paths_leave_no_output(self):
        outside = self.temporary_root / "outside.json"
        self.put_json(outside, self.config)
        before = self.snapshot()
        for config_path in (str(outside), "../outside.json", "..\\outside.json",
                            "configs/../../outside.json", "C:\\outside.json", "C:outside.json", "/outside.json"):
            with self.subTest(config_path=config_path):
                with self.assertRaises(ValueError):
                    self.create(config_path=config_path)
                self.assertEqual(self.snapshot(), before)

    def test_config_symlink_cannot_escape_project(self):
        outside = self.temporary_root / "outside.json"
        self.put_json(outside, self.config)
        link = self.root / "configs" / "linked.json"
        try:
            link.symlink_to(outside)
        except OSError as error:
            self.skipTest(f"symlinks unavailable: {error}")
        with self.assertRaises(ValueError):
            self.create(config_path="configs/linked.json")
        self.assertFalse((self.root / "experiments" / "EXP-002").exists())
        self.assertEqual(ledger.read(outside), self.config)

    def test_invalid_config_leaves_no_output(self):
        for contents in ('{"model":', "[]", "null", '{"value": NaN}', '{"value": Infinity}'):
            with self.subTest(contents=contents):
                self.template_path.write_text(contents, encoding="utf-8")
                before = self.snapshot()
                with self.assertRaises(ValueError):
                    self.create()
                self.assertEqual(self.snapshot(), before)

    def test_missing_config_leaves_no_output(self):
        before = self.snapshot()
        with self.assertRaises(FileNotFoundError):
            self.create(config_path="configs/missing.json")
        self.assertEqual(self.snapshot(), before)

    def test_write_failure_rolls_back_only_new_outputs(self):
        before = self.snapshot()
        record_path = self.here / "records" / "experiments" / "EXP-002.json"
        original_open = pathlib.Path.open

        def fail_record_open(path, *args, **kwargs):
            if path == record_path and args and args[0] == "xb":
                raise OSError("simulated record write failure")
            return original_open(path, *args, **kwargs)

        with mock.patch.object(pathlib.Path, "open", fail_record_open):
            with self.assertRaisesRegex(OSError, "simulated record write failure"):
                self.create()
        self.assertEqual(self.snapshot(), before)

    def test_record_collision_during_creation_preserves_other_writer(self):
        record_path = self.here / "records" / "experiments" / "EXP-002.json"
        original_open = pathlib.Path.open

        def competing_record_open(path, *args, **kwargs):
            if path == record_path and args and args[0] == "xb":
                with original_open(path, "wb") as stream:
                    stream.write(b"other writer")
            return original_open(path, *args, **kwargs)

        with mock.patch.object(pathlib.Path, "open", competing_record_open):
            with self.assertRaises(FileExistsError):
                self.create()
        self.assertEqual(record_path.read_bytes(), b"other writer")
        self.assertFalse((self.root / "experiments" / "EXP-002").exists())
        self.assertTrue((self.root / "experiments" / "README.md").exists())

    def test_cli_accepts_slug_and_config_and_renders_links(self):
        custom_path = self.root / "configs" / "custom.json"
        self.put_json(custom_path, self.config)
        with contextlib.redirect_stdout(io.StringIO()):
            result = ledger.main(["new-experiment", "--title", "计划实验", "--slug", "cli-plan",
                                  "--config", "configs/custom.json"])
        self.assertEqual(result, 0)
        view = (self.here / "实验台账.md").read_text(encoding="utf-8")
        self.assertIn("../experiments/EXP-002-cli-plan/config.json", view)
        self.assertIn("../experiments/EXP-002-cli-plan/report.md", view)

    def test_cli_rejects_invalid_existing_ledger_before_creating(self):
        self.settings["baseline_experiment_id"] = "EXP-999"
        self.put_json(self.here / "settings.json", self.settings)
        before = self.snapshot()
        with contextlib.redirect_stdout(io.StringIO()):
            result = ledger.main(["new-experiment", "--title", "计划实验"])
        self.assertNotEqual(result, 0)
        self.assertEqual(self.snapshot(), before)

    def test_cli_invalid_config_returns_nonzero_without_outputs(self):
        self.template_path.write_text("{bad json", encoding="utf-8")
        before = self.snapshot()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                ledger.main(["new-experiment", "--title", "计划实验"])
        self.assertNotEqual(error.exception.code, 0)
        self.assertEqual(self.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
