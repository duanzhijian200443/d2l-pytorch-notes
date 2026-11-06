"""Small synthetic acceptance tests; never load a real prediction cache."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('exp032_compare_under_test', HERE / 'compare_cached.py')
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


class CaptureGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ntl-compare-gate-')
        self.root = Path(self.temp.name).resolve()
        self.assertEqual(self.root.parent, Path(tempfile.gettempdir()).resolve())
        self.assertTrue(self.root.name.startswith('ntl-compare-gate-'))
        self.addCleanup(self.temp.cleanup)
        self.source = {'experiment_id': C.SOURCE_EXPERIMENT, 'run_id': C.SOURCE_RUN,
                       'plan_sha256': C.SOURCE_PLAN_SHA}
        self.result = {'scope': 'exploratory_external', 'experiment_id': C.SOURCE_EXPERIMENT,
                       'run_id': C.SOURCE_RUN, 'status': 'COMPLETE', 'cause': None, 'child_exit': 0}
        self.worker = {'status': 'COMPLETE', 'scope': 'exploratory_external', 'run_id': C.SOURCE_RUN,
                       'plan_sha256': C.SOURCE_PLAN_SHA, 'rows': 729, 'score_columns': 18,
                       'formal_submission_created': False, 'metrics_computed_from_truth': False,
                       'full_training_independence_claimed': False}

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text(json.dumps(value), encoding='utf-8')
        return path

    def header(self):
        self.write('result.json', self.result)
        self.write('worker-result.json', self.worker)

    def inventory(self):
        files = [{'path': path.relative_to(self.root).as_posix(), 'bytes': path.stat().st_size,
                  'sha256': C.sha(path)} for path in self.root.rglob('*') if path.is_file()]
        path = self.write('artifacts.json', {'status': 'COMPLETE_INVENTORY', 'files': files})
        return C.sha(path)

    def test_missing_and_empty_result_directories_are_rejected(self):
        for path in (self.root / 'absent', self.root):
            with self.subTest(path=path), self.assertRaises((C.AcceptanceError, FileNotFoundError)):
                C.verify_real_artifacts(path, {'source': self.source}, 'a' * 64)

    def test_synthetic_cache_cannot_use_a_real_run_id(self):
        self.header()
        self.result['scope'] = 'synthetic_math'; self.write('result.json', self.result)
        with self.assertRaisesRegex(C.AcceptanceError, 'Synthetic'):
            C.verify_real_artifacts(self.root, {'source': self.source}, 'a' * 64)

    def test_failed_or_incomplete_run_is_rejected_before_inventory(self):
        for change in ({'status': 'ERROR'}, {'cause': 'WORK_BUDGET_EXCEEDED'}, {'child_exit': 1},
                       {'traceback': 'capture failed'}):
            result = {**self.result, **change}
            self.write('result.json', result); self.write('worker-result.json', self.worker)
            with self.subTest(change=change), self.assertRaisesRegex(C.AcceptanceError, 'did not complete'):
                C.verify_real_artifacts(self.root, {'source': self.source}, 'a' * 64)

    def test_failure_file_and_incomplete_worker_are_rejected(self):
        self.header(); self.write('failure.json', {'cause': 'missing member'})
        with self.assertRaisesRegex(C.AcceptanceError, 'failure record'):
            C._complete_header(self.root, self.source)
        (self.root / 'failure.json').unlink()
        self.worker['status'] = 'RUNNING'; self.write('worker-result.json', self.worker)
        with self.assertRaisesRegex(C.AcceptanceError, 'Worker is incomplete'):
            C._complete_header(self.root, self.source)

    def test_wrong_run_or_saved_version_cannot_pass_source_gate(self):
        self.header()
        for key, value in (('run_id', 'synthetic'), ('experiment_id', 'EXP-028'), ('plan_sha256', 'a' * 64)):
            source = {**self.source, key: value}
            with self.subTest(key=key), self.assertRaisesRegex(C.AcceptanceError, 'fixed EXP-031'):
                C._complete_header(self.root, source)

    def test_inventory_needs_external_digest_and_checks_exact_files(self):
        self.header(); digest = self.inventory()
        files = C.verify_artifact_inventory(self.root, digest)
        self.assertEqual(set(files), {'result.json', 'worker-result.json'})
        with self.assertRaises(C.AcceptanceError):
            C.verify_artifact_inventory(self.root, None)
        with self.assertRaises(C.AcceptanceError):
            C.verify_artifact_inventory(self.root, 'a' * 64)
        self.write('unlisted.json', {})
        with self.assertRaisesRegex(C.AcceptanceError, 'Unlisted'):
            C.verify_artifact_inventory(self.root, digest)

    def test_changed_stable_file_is_rejected(self):
        self.header(); digest = self.inventory()
        self.write('worker-result.json', {'status': 'COMPLETE'})
        with self.assertRaisesRegex(C.AcceptanceError, 'bytes changed'):
            C.verify_artifact_inventory(self.root, digest)

    def test_file_only_complete_header_cannot_pass_deep_stage_gate(self):
        self.header(); digest = self.inventory()
        with self.assertRaisesRegex(C.AcceptanceError, 'stage/artifact'):
            C.verify_real_artifacts(self.root, {'source': self.source}, digest)

    def test_partial_atlas_is_rejected_even_when_replay_says_pass(self):
        plan = {'expected_main_members': ['main'], 'expected_comp_members': ['comp'], 'expected_family_gates': ['gate']}
        replay = {'accepted': True, 'status': 'PASS_CACHED_BASELINE_MATH', 'input_scope': 'exploratory_external',
                  'rows': 729, 'all_checked_argmax_equal': True, 'main_members': ['main'], 'comp_members': ['comp'],
                  'family_gates': ['gate'], 'atlas_coverage': [{'requested_rows': 2, 'completed_rows': 1,
                                                             'missing_row_indices': [1], 'missing_ids': ['b']}]}
        with self.assertRaisesRegex(C.AcceptanceError, 'ATLAS requested rows are incomplete'):
            C._accept_replay(replay, plan)

    def test_baseline_uses_source_dtype_tolerance_and_exact_near_tie_argmax(self):
        original = np.zeros((1, 18), dtype=np.float64)
        original[0, 0] = 1.0
        original[0, 1] = 1.0 - 5e-13
        changed = original.copy(); changed[0, 1] = 1.0 + 5e-13
        with self.assertRaisesRegex(C.AcceptanceError, 'argmax differs'):
            C._same_scores(changed, original, 'near tie')
        with self.assertRaisesRegex(C.AcceptanceError, 'dtype differs'):
            C._same_scores(original.astype(np.float32), original, 'dtype')
        changed = original.copy(); changed[0, 0] += 1e-8
        with self.assertRaisesRegex(C.AcceptanceError, 'Recomputed .45 differs'):
            C._same_scores(changed, original, 'strict float64')


if __name__ == '__main__':
    unittest.main()
