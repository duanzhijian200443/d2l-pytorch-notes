"""Exercise the real guard around fake subprocesses in isolated temp dirs.

No model/data files are loaded. Run:
python -B -X utf8 -m unittest discover -s cloud/svc4-raw-public -p test_runtime.py -v
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

import psutil


GUARD = Path(__file__).with_name('guard.py')
PRELUDE = '''\
import hashlib, json, os, time
from pathlib import Path
out = Path(os.environ['NTL_OUTPUT_DIR'])
out.mkdir(parents=True, exist_ok=True)
(out/'child.pid').write_text(str(os.getpid()))
'''
COMPLETE = '''\
(out/'stage.txt').write_text('synthetic completed checkpoint', encoding='utf-8')
(out/'result.json').write_text(json.dumps({'status':'COMPLETE','trained_svms':20}), encoding='utf-8')
(out/'state.json').write_text(json.dumps({'status':'COMPLETE'}), encoding='utf-8')
files = [{'path':p.name, 'bytes':p.stat().st_size, 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
         for p in (out/'child.pid', out/'stage.txt', out/'result.json')]
(out/'artifacts.json').write_text(json.dumps({'files':files}), encoding='utf-8')
print('FAKE COMPLETE', flush=True)
'''
PARTIAL = '''\
(out/'stage.txt').write_text('synthetic retained checkpoint', encoding='utf-8')
(out/'failure.json').write_text(json.dumps({'cause':'ORIGINAL_RUNNER_DETAIL'}), encoding='utf-8')
print('FAKE PARTIAL', flush=True)
'''


class GuardRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='ntl-svc4-guard-test-')
        self.root = Path(self.temporary.name).resolve()
        self.assertEqual(self.root.parent, Path(tempfile.gettempdir()).resolve())
        self.assertTrue(self.root.name.startswith('ntl-svc4-guard-test-'))
        self.addCleanup(self.temporary.cleanup)
        self.payload = self.root/'payload'
        self.payload.mkdir()
        (self.payload/'guard.py').write_bytes(GUARD.read_bytes())
        self.output = self.root/'output'
        self.output.mkdir()
        self.addCleanup(self.stop_leftover_child)
        self.configure()

    def stop_leftover_child(self):
        # If the guard regresses, the test timeout must not leave its fake child
        # alive. Only kill a PID still running this test's exact runner path.
        for name in ('child.pid', 'tracked-child.pid'):
            pid_file = self.output/name
            if not pid_file.exists():
                continue
            try:
                process = psutil.Process(int(pid_file.read_text()))
                if str(self.payload/'runner.py') not in process.cmdline():
                    continue
                children = process.children(recursive=True)
                for item in [process]+children:
                    try:
                        item.kill()
                    except psutil.NoSuchProcess:
                        pass
                psutil.wait_procs([process]+children, timeout=5)
            except psutil.NoSuchProcess:
                pass

    def configure(self, budget=10, rss=512*1024**2):
        (self.payload/'protocol.json').write_text(json.dumps({
            'work_budget_seconds':budget, 'rss_limit_bytes':rss,
        }), encoding='utf-8')

    def fake_runner(self, body):
        (self.payload/'runner.py').write_text(PRELUDE+textwrap.dedent(body), encoding='utf-8')

    def run_guard(self, launcher=None):
        environment = os.environ.copy()
        environment.update({'NTL_OUTPUT_DIR':str(self.output), 'PYTHONDONTWRITEBYTECODE':'1', 'PYTHONUTF8':'1'})
        command = [sys.executable, '-B', '-X', 'utf8']
        command += ['-c', launcher] if launcher else [str(self.payload/'guard.py')]
        try:
            return subprocess.run(command, cwd=self.payload, env=environment, capture_output=True,
                                  text=True, encoding='utf-8', timeout=25)
        except subprocess.TimeoutExpired:
            self.fail('Guard did not terminate within the synthetic-test safety timeout')

    def read(self, name):
        return json.loads((self.output/name).read_text(encoding='utf-8'))

    def assert_child_stopped(self):
        pid = int((self.output/'child.pid').read_text())
        self.assertFalse(psutil.pid_exists(pid), f'Fake child still running: {pid}')

    def test_complete_requires_valid_manifest(self):
        self.fake_runner(COMPLETE)
        result = self.run_guard()
        self.assertEqual(result.returncode, 0, result.stderr)
        guard = self.read('guard.json')
        self.assertEqual(guard['status'], 'COMPLETE')
        self.assertTrue(guard['artifacts_verified'])
        self.assertIsNone(guard['termination_reason'])
        self.assertFalse((self.output/'failure.json').exists())
        self.assertTrue((self.output/'.run.lock').exists())
        self.assertIn('FAKE COMPLETE', (self.output/'training.log').read_text(encoding='utf-8'))
        self.assert_child_stopped()

    def test_wall_budget_preserves_partial_artifact_and_original_error(self):
        self.configure(budget=1)
        self.fake_runner(PARTIAL+'time.sleep(30)\n')
        result = self.run_guard()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.read('guard.json')['termination_reason'], 'WALL_BUDGET_EXCEEDED')
        self.assertEqual(self.read('guard.json')['status'], 'ERROR')
        self.assertEqual(self.read('state.json')['status'], 'ERROR')
        self.assertEqual(self.read('failure.json'), {'cause':'ORIGINAL_RUNNER_DETAIL'})
        self.assertEqual((self.output/'stage.txt').read_text(), 'synthetic retained checkpoint')
        self.assert_child_stopped()

    def test_rss_budget_preserves_partial_artifact(self):
        self.configure(budget=10, rss=64*1024**2)
        self.fake_runner(PARTIAL+'allocation = bytearray(96*1024**2)\ntime.sleep(30)\n')
        result = self.run_guard()
        self.assertNotEqual(result.returncode, 0)
        guard = self.read('guard.json')
        self.assertEqual(guard['termination_reason'], 'RSS_BUDGET_EXCEEDED')
        self.assertEqual(guard['status'], 'ERROR')
        self.assertGreater(guard['peak_rss_bytes'], 64*1024**2)
        self.assertEqual(self.read('failure.json'), {'cause':'ORIGINAL_RUNNER_DETAIL'})
        self.assert_child_stopped()

    def test_child_start_failure_records_terminal_error(self):
        self.fake_runner(COMPLETE)
        launcher = "import runpy,subprocess\ndef fail(*args,**kwargs): raise OSError('synthetic launch failure')\nsubprocess.Popen=fail\nrunpy.run_path('guard.py',run_name='__main__')"
        result = self.run_guard(launcher)
        self.assertNotEqual(result.returncode, 0)
        guard = self.read('guard.json')
        self.assertEqual(guard['status'], 'ERROR')
        self.assertEqual(guard['termination_reason'], 'GUARD_FAILED')
        self.assertIsNone(guard['child_exit'])
        self.assertIn('synthetic launch failure', guard['traceback'])
        self.assertEqual(self.read('failure.json')['cause'], 'GUARD_FAILED')
        self.assertFalse((self.output/'child.pid').exists())

    def test_bad_manifest_fails_closed(self):
        self.fake_runner(COMPLETE+"(out/'stage.txt').write_text('tampered')\n")
        result = self.run_guard()
        self.assertNotEqual(result.returncode, 0)
        guard = self.read('guard.json')
        self.assertEqual(guard['status'], 'ERROR')
        self.assertEqual(guard['termination_reason'], 'OUTPUT_VALIDATION_FAILED')
        self.assertFalse(guard['artifacts_verified'])
        self.assertEqual(self.read('state.json')['status'], 'ERROR')
        self.assertEqual(self.read('failure.json')['cause'], 'OUTPUT_VALIDATION_FAILED')
        self.assert_child_stopped()

    def test_invalid_manifest_structure_fails_closed(self):
        self.fake_runner(COMPLETE+"(out/'artifacts.json').write_text('{\"files\":{}}')\n")
        result = self.run_guard()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.read('guard.json')['termination_reason'], 'OUTPUT_VALIDATION_FAILED')
        self.assertEqual(self.read('state.json')['status'], 'ERROR')
        self.assert_child_stopped()

    def test_monitor_failure_stops_child_and_records_error(self):
        self.fake_runner('time.sleep(30)\n')
        launcher = '''\
import os, runpy, subprocess
from pathlib import Path
original = subprocess.Popen
def tracked(*args, **kwargs):
    child = original(*args, **kwargs)
    (Path(os.environ['NTL_OUTPUT_DIR'])/'tracked-child.pid').write_text(str(child.pid))
    return child
subprocess.Popen = tracked
namespace = runpy.run_path('guard.py', run_name='_guard_test')
def fail(child):
    raise OSError('synthetic monitoring failure')
namespace['main'].__globals__['tree_rss'] = fail
namespace['main']()
'''
        result = self.run_guard(launcher)
        self.assertNotEqual(result.returncode, 0)
        guard = self.read('guard.json')
        self.assertEqual(guard['status'], 'ERROR')
        self.assertEqual(guard['termination_reason'], 'GUARD_FAILED')
        self.assertIn('synthetic monitoring failure', guard['traceback'])
        self.assertEqual(self.read('state.json')['status'], 'ERROR')
        pid = int((self.output/'tracked-child.pid').read_text())
        self.assertFalse(psutil.pid_exists(pid))

    def test_duplicate_lock_does_not_overwrite_existing_run(self):
        self.fake_runner(COMPLETE)
        preserved = {'.run.lock':b'existing lock', 'guard.json':b'{"status":"RUNNING"}', 'training.log':b'existing log'}
        for name, content in preserved.items():
            (self.output/name).write_bytes(content)
        result = self.run_guard()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('OUTPUT_ALREADY_LOCKED', result.stderr)
        for name, content in preserved.items():
            self.assertEqual((self.output/name).read_bytes(), content)
        self.assertFalse((self.output/'child.pid').exists())
        self.assertFalse((self.output/'failure.json').exists())

    def test_dirty_output_cannot_reuse_stale_success(self):
        self.fake_runner(COMPLETE)
        (self.output/'result.json').write_text('{"status":"COMPLETE"}', encoding='utf-8')
        (self.output/'artifacts.json').write_text('{"files":[]}', encoding='utf-8')
        result = self.run_guard()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('OUTPUT_DIRECTORY_NOT_EMPTY', result.stderr)
        self.assertEqual((self.output/'result.json').read_text(), '{"status":"COMPLETE"}')
        self.assertFalse((self.output/'child.pid').exists())
        self.assertFalse((self.output/'.run.lock').exists())


if __name__ == '__main__':
    unittest.main()
