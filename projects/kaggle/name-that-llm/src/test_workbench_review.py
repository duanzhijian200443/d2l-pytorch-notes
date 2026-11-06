"""Independent follow-up checks: temporary files only, no remote operations."""
import copy
import io
import json
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import workbench_core as core
import experiment_workbench as wb
from workbench_parquet import verify_extra_receipt


class FollowupReviewTests(unittest.TestCase):
    def test_remote_delta_requires_confirmed_score_and_is_only_numeric_difference(self):
        p = {'id': 'EXP-048', 'remote': {'submission_id': 123}, 'report': 'detail.md'}
        record = {'title': 'test', 'baseline_id': 'EXP-046'}
        baseline = {'submission': {'status': 'COMPLETE', 'public_score': .663}}
        with tempfile.TemporaryDirectory() as tmp, patch.object(wb, 'WORK', Path(tmp)), \
             patch.object(wb, 'inside', lambda name: Path(tmp) / name), \
             patch.object(wb, 'read', return_value=baseline), \
             patch.object(wb, 'latest', return_value=None):
            with patch.object(wb, 'confirmed_score', return_value=None):
                text = wb.make_report(p, record).read_text(encoding='utf-8')
                self.assertNotIn('分数差', text)
            with patch.object(wb, 'confirmed_score', return_value={'value': .664, 'observed_at': 'T1'}):
                text = wb.make_report(p, record).read_text(encoding='utf-8')
                self.assertIn('+0.001000', text)
                self.assertIn('+0.1000 个百分点', text)
                self.assertIn('不能单独证明改动稳定有效', text)

    def _local_numeric_report(self, ident, summary, expected):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'summary.json'
            path.write_text(json.dumps(summary), encoding='utf-8')
            p = {'id': ident, 'local_summary': 'summary.json', 'report': 'detail.md'}
            checked = {'status': 'PASS_FIXED_LOCAL_JOB', 'verified_summary': {
                'path': 'summary.json', 'sha256': core.sha(path)}}
            with patch.object(wb, 'WORK', root), \
                 patch.object(wb, 'inside', lambda name: root / name), \
                 patch.object(wb, 'latest', side_effect=lambda key, op: checked if op == 'verify' else None):
                text = wb.make_report(p, {'title': 'test'}).read_text(encoding='utf-8')
                for value in expected:
                    self.assertIn(value, text)
                path.write_text('{}', encoding='utf-8')
                text = wb.make_report(p, {'title': 'test'}).read_text(encoding='utf-8')
                self.assertIn('暂不展示数值', text)
                for value in expected[:1]:
                    self.assertNotIn(value, text)

    def test_local_macro_display_remains_hash_bound_and_not_oof(self):
        self._local_numeric_report('EXP-049', {
            'status': 'COMPLETE_LOCAL_DIAGNOSTIC',
            'baseline_045': {'correct': 488, 'macro_recall': .6010754296632905},
            'candidate_030': {'correct': 490, 'macro_recall': .6032992310481546}},
            ['+0.22238 个百分点', '不是干净OOF', '差值区间跨0'])

    def test_pool_overlap_display_remains_hash_bound_and_not_training_approval(self):
        self._local_numeric_report('EXP-050', {
            'status': 'AUDIT_COMPLETE_NOT_TRAINING_APPROVED', 'rows': 190588,
            'risk_counts': {'in_a1_runtime_union': 97870, 'novel_text_to_audited_runtime': 92718}},
            ['主池 190588 条', '重合 97870 条，未命中 92718 条', '不等于训练准入或提分'])

    def test_list_uses_new_confirmed_score_when_canonical_is_pending(self):
        p = {'id': 'EXP-048', 'aliases': ['codex120'], 'remote': {
            'submission_id': 123, 'script_version_id': 456}}
        record = {'title': 'test', 'submission': {'status': 'PENDING'}}
        confirmed = {'submission_id': 123, 'script_version_id': 456,
                     'score': .665, 'observed_at': '2026-11-02T00:00:00Z'}
        output = io.StringIO()
        with patch.object(wb, 'read', return_value={'experiments': [p]}), \
             patch.object(wb, 'profile', return_value=({}, p, record)), \
             patch.object(wb, 'latest', return_value=confirmed), \
             patch('sys.stdout', output):
            self.assertEqual(wb.main(['list']), 0)
        self.assertIn('0.665', output.getvalue())
        self.assertNotIn('未确认', output.getvalue())

    def test_local_lock_rejects_overlap_and_releases(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(core, 'WORK', Path(tmp)):
            with core.local_lock('EXP-049'):
                with self.assertRaisesRegex(core.GateError, 'LOCAL_OPERATION_ALREADY_ACTIVE'):
                    with core.local_lock('EXP-049'):
                        self.fail('overlapping operation admitted')
            with core.local_lock('EXP-049'):
                pass

    def test_dependency_change_rejected_before_child_or_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / 'script.py'
            dep = root / 'dep.py'
            script.write_text('pass', encoding='utf-8')
            dep.write_text('changed', encoding='utf-8')
            p = {'id': 'EXP-049', 'local_code_hashes': {
                'script.py': core.sha(script), 'dep.py': '0' * 64}}
            with patch.dict(wb.JOB_SPECS, {('EXP-049', 'run-local'): ('script.py', [], 1)}), \
                 patch.object(wb, 'inside', lambda name: root / name), \
                 patch.object(wb, 'new_observation') as observation, \
                 patch.object(wb.subprocess, 'run') as child:
                with self.assertRaisesRegex(core.GateError, 'LOCAL_JOB_DEPENDENCY_CHANGED'):
                    wb._local_job({}, p, 'run-local')
                observation.assert_not_called()
                child.assert_not_called()

    def _failed_child_case(self, timeout):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / 'script.py'
            script.write_text('pass', encoding='utf-8')
            p = {'id': 'EXP-049', 'local_code_hashes': {'script.py': core.sha(script)}}
            with patch.dict(wb.JOB_SPECS, {('EXP-049', 'run-local'): ('script.py', [], 1)}), \
                 patch.object(wb, 'inside', lambda name: root / name), \
                 patch.object(wb, 'new_observation', return_value=root), \
                 patch.object(wb, 'interpreter', return_value='python'), \
                 patch.object(wb, 'remember') as remember, \
                 patch.object(wb.subprocess, 'run') as child:
                if timeout:
                    child.side_effect = subprocess.TimeoutExpired(['python'], 1)
                else:
                    child.return_value = subprocess.CompletedProcess(['python'], 1)
                result = wb._local_job({}, p, 'run-local')
                self.assertEqual(result['status'], 'ERROR')
                self.assertNotIn('verified_summary', result)
                self.assertEqual(remember.call_args.args[2]['status'], 'ERROR')
                self.assertTrue((root / 'intent.json').is_file())

    def test_local_timeout_never_records_pass(self):
        self._failed_child_case(timeout=True)

    def test_local_nonzero_never_records_pass(self):
        self._failed_child_case(timeout=False)

    def test_add120_preflight_fields_fail_closed(self):
        p = {'id': 'EXP-048', 'receipt_identity': {'added_data_sha256': 'a' * 64}}
        valid = {'status': 'PASS', 'new_rows': 120, 'new_groups': 60,
                 'holdout_overlap': 0, 'new120_sha256': 'a' * 64}
        verify_extra_receipt(p, valid)
        for key, value in [('status', 'FAIL'), ('new_rows', 119),
                           ('new_groups', 59), ('holdout_overlap', 1),
                           ('new120_sha256', 'b' * 64)]:
            with self.subTest(key=key), self.assertRaises(core.GateError):
                verify_extra_receipt(p, dict(valid, **{key: value}))
        for key in valid:
            invalid = copy.deepcopy(valid)
            del invalid[key]
            with self.subTest(missing=key), self.assertRaises(core.GateError):
                verify_extra_receipt(p, invalid)


if __name__ == '__main__':
    unittest.main()
