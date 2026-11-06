"""Synthetic-only tests for EXP044 concurrency, immutable resume and guards."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


r = module(HERE / 'runner.py', 'test_exp044_runner')
g = module(HERE / 'guard.py', 'test_exp044_guard')


class GuardTests(unittest.TestCase):
    def test_same_run_budget_accumulates_and_lost_guard_never_resets(self):
        ledger = {'hard_seconds': 6600, 'attempts': [
            {'charged_seconds': 100., 'started_wall': 10},
            {'charged_seconds': None, 'started_wall': 200}]}
        self.assertEqual(g.charge_previous(ledger, 500), 400)
        self.assertEqual(g.charge_previous(ledger, 900), 400)
        self.assertIn('CONSERVATIVELY', ledger['attempts'][1]['status'])
        lost = {'hard_seconds': 6600, 'attempts': [{'charged_seconds': 100.}, {'charged_seconds': None, 'started_wall': 100.}]}
        self.assertEqual(g.charge_previous(lost, 100000), 6600)
        reversed_clock = {'hard_seconds': 6600, 'attempts': [{'charged_seconds': None, 'started_wall': 1000.}]}
        self.assertEqual(g.charge_previous(reversed_clock, 10), 6600)
        with self.assertRaises(ValueError):
            g.charge_previous({'hard_seconds': 6600, 'attempts': [{'charged_seconds': float('nan')}]}, 0)

    def test_claim_mutex_rejects_second_live_claimer(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            # Separate process exercises actual OS locking, not a mocked mutex.
            code = ('import importlib.util,sys; from pathlib import Path; '
                    's=importlib.util.spec_from_file_location("g",sys.argv[1]);g=importlib.util.module_from_spec(s);s.loader.exec_module(g);'
                    'c=g.claim_mutex(Path(sys.argv[2]));c.__enter__();c.__exit__(None,None,None)')
            with g.claim_mutex(output):
                child = subprocess.run([sys.executable, '-X', 'utf8', '-c', code, str(HERE / 'guard.py'), str(output)], capture_output=True, text=True, encoding='utf-8', timeout=30)
                self.assertNotEqual(child.returncode, 0, child.stdout)
            child = subprocess.run([sys.executable, '-X', 'utf8', '-c', code, str(HERE / 'guard.py'), str(output)], capture_output=True, text=True, encoding='utf-8', timeout=30)
            self.assertEqual(child.returncode, 0, child.stderr)

    def test_still_live_synthetic_worker_blocks_resume_after_guard_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            worker = output / 'selftests/attempt-0001/jobs/f0-raw120/attempt-0001/process.json'
            r.save(worker, {'kind': 'live-worker'})
            ticket = {'attempt': 'attempt-0001', 'nonce': 'old', 'process': {'kind': 'dead-guard'}}
            with mock.patch.object(g.r, 'process_live', side_effect=lambda record: record['kind'] == 'live-worker'):
                with self.assertRaisesRegex(ValueError, 'still active'):
                    g.confirm_inactive(output, ticket)

    def test_remote_unknown_process_needs_terminal_tree_stop_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            ticket = {'attempt': 'attempt-0001', 'nonce': 'old', 'process': {}}
            with mock.patch.object(g.r, 'process_live', side_effect=ValueError('Cannot confirm process inactivity across host/boot')):
                with self.assertRaisesRegex(ValueError, 'terminal'):
                    g.confirm_inactive(output, ticket)
                r.save(output / 'attempts/attempt-0001/guard.json', {'nonce': 'old', 'status': 'ERROR', 'tree_stopped': True})
                g.confirm_inactive(output, ticket)

    def test_complete_inventory_rejects_corruption_and_unlisted_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / 'scores').write_bytes(b'original')
            files = r.inventory(output)
            r.validate_inventory(output, files)
            (output / 'extra').write_bytes(b'new')
            with self.assertRaisesRegex(ValueError, 'not closed'):
                r.validate_inventory(output, files)
            (output / 'scores').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                r.validate_inventory(output, files)


class NumericalAndResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.output = cls.root / 'selftest'
        cls.report = r.self_test(HERE, cls.output)
        cls.config, cls.frames, cls.legacy, cls.core, cls.identity = r.synthetic_setup(HERE)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_true_stylegrams_parallel_matches_frozen_serial_path(self):
        self.assertEqual(self.report['status'], 'PASS')
        self.assertEqual(self.report['parallel_workers_observed'], 2)
        self.assertLessEqual(self.report['max_abs_score_difference'], 1e-12)
        self.assertTrue(self.report['completed_resume_reused'])
        self.assertTrue(self.report['corruption_rejected'])

    def test_changed_identity_and_corrupted_job_refuse_resume(self):
        plan = [(0, arm) for arm in r.ARMS]
        changed = copy.deepcopy(self.identity)
        changed['config_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'identity'):
            r.find_completed(self.output, changed, plan)
        # Test a copy to preserve the shared successful numerical evidence.
        import shutil
        source = self.output / 'jobs/f0-raw120/attempt-0001'
        copied = self.root / 'corrupt-copy'
        shutil.copytree(source, copied)
        with (copied / 'dense.npz').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            r.validate_job(copied, self.identity, 0, 'raw120')

    def test_unfinished_attempt_is_preserved_and_new_attempt_used(self):
        output = self.root / 'resume-incomplete'
        old = output / 'jobs/f0-raw120/attempt-0001'
        r.save(old / 'identity.json', {'identity': self.identity, 'fold': 0, 'arm': 'raw120'})
        (old / 'interrupted-evidence.txt').write_text('preserve', encoding='utf-8')
        before = r.inventory(old)
        complete, maximum = r.schedule(HERE, output, self.identity, [(0, 'raw120')], 'new-explicit-attempt', synthetic=True, resume=True)
        self.assertEqual(complete[(0, 'raw120')].name, 'attempt-0002')
        self.assertEqual(maximum, 1)
        self.assertEqual(before, r.inventory(old))
        with self.assertRaisesRegex(ValueError, 'require --resume'):
            r.schedule(HERE, output, self.identity, [(0, 'raw120')], 'unauthorized-repeat', synthetic=True)

    def test_failed_worker_stops_other_workers_without_refill_or_retry(self):
        output = self.root / 'failed-worker'
        wrong = copy.deepcopy(self.identity)
        wrong['config_sha256'] = 'f' * 64
        with self.assertRaisesRegex((RuntimeError, ValueError), 'JOB_FAILED'):
            r.schedule(HERE, output, wrong, [(0, 'raw120'), (0, 'clean120'), (1, 'raw120')], 'fail-once', synthetic=True)
        self.assertFalse((output / 'jobs/f1-raw120').exists())
        self.assertEqual(len(list(output.glob('jobs/*/attempt-*'))), 2)
        self.assertEqual(len(list(output.rglob('completion.json'))), 0)
        for path in output.glob('jobs/*/attempt-*/process.json'):
            self.assertFalse(r.process_live(r.read(path)))


if __name__ == '__main__':
    unittest.main()
