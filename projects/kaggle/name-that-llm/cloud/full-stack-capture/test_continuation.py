"""Offline fake-API checks of the one-shot continuation; never authenticate.

Every controller write and fake download is confined to a fresh temp directory.
The kaggle import is stubbed before loading the controller, so importing or
running this test cannot contact the service or start a real CPU/GPU job.
"""
from collections import deque
from contextlib import ExitStack, redirect_stdout
from datetime import timedelta
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().with_name('launch_after_preflight.py')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_controller():
    packages = {name: ModuleType(name) for name in ('kaggle', 'kaggle.api', 'kaggle.api.kaggle_api_extended')}
    packages['kaggle'].__path__ = []
    packages['kaggle.api'].__path__ = []

    class ForbiddenRealApi:
        def __init__(self):
            raise AssertionError('A fake API must be explicitly installed before running the controller')

    packages['kaggle.api.kaggle_api_extended'].KaggleApi = ForbiddenRealApi
    with patch.dict(sys.modules, packages):
        spec = importlib.util.spec_from_file_location('offline_continuation_under_test', SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class FakeApi:
    def __init__(self, controller, root):
        self.controller, self.root = controller, root
        self.calls = []
        self.statuses = deque(['COMPLETE'])
        self.gpu_status = 'QUEUED'
        self.remaining_seconds = 72000
        self.manifest_count = 1
        self.manifest_status = 'PASS'
        self.existing = []
        self.fail_save = None
        self.fail_run = None
        self.fail_pull_number = None
        self.bad_pull_number = None
        self.pull_count = 0
        self.save_error = None
        self.pushes = []

    def authenticate(self):
        self.calls.append(('authenticate',))

    def kernels_status(self, ref):
        self.calls.append(('status', ref))
        if ref == self.controller.CPU:
            status = self.statuses[0]
            if len(self.statuses) > 1:
                self.statuses.popleft()
            return SimpleNamespace(status=status)
        if ref.startswith(self.controller.GPU + '/'):
            return SimpleNamespace(status=self.gpu_status)
        raise AssertionError('Unexpected remote target: ' + ref)

    def kernels_list(self, **kwargs):
        self.calls.append(('list', kwargs))
        return list(self.existing)

    def kernels_output(self, ref, destination, quiet=True):
        self.calls.append(('output', ref))
        if ref != self.controller.CPU:
            raise AssertionError('Only the already-running CPU V1 may be downloaded')
        for index in range(self.manifest_count):
            directory = Path(destination) / ('result-' + str(index))
            directory.mkdir(parents=True, exist_ok=True)
            (directory / 'preflight_manifest.json').write_text(
                json.dumps({'status': self.manifest_status, 'synthetic_fixture': True}), encoding='utf-8')

    def quota_view(self):
        self.calls.append(('quota',))
        return SimpleNamespace(gpu_quota=SimpleNamespace(total_time_allowed=timedelta(seconds=self.remaining_seconds),
                                                        time_used=timedelta(0), time_reserved=timedelta(0)))

    def kernels_push(self, folder, timeout=None, acc=None, no_run=False):
        self.calls.append(('push', no_run, folder, timeout, acc))
        self.pushes.append(no_run)
        failure = self.fail_save if no_run else self.fail_run
        if failure:
            raise failure
        return SimpleNamespace(error=self.save_error if no_run else None, ref=self.controller.GPU,
                               version_number=1 if no_run else 2,
                               invalid_dataset_sources=[], invalid_kernel_sources=[],
                               invalid_model_sources=[], invalid_competition_sources=[])

    def kernels_pull(self, ref, destination, metadata=True, quiet=True):
        self.pull_count += 1
        self.calls.append(('pull', ref))
        if self.pull_count == self.fail_pull_number:
            raise ConnectionError('Synthetic readback failure')
        kernel = self.controller.HERE / 'launch-v1/kernel'
        settings = json.loads((kernel / 'kernel-metadata.json').read_text(encoding='utf-8'))
        for key in ('dataset_sources', 'kernel_sources'):
            settings[key] = [value.rsplit('/', 1)[0] for value in settings[key]]
        if self.pull_count == self.bad_pull_number:
            settings['enable_gpu'] = False
        destination = Path(destination)
        (destination / 'kernel-metadata.json').write_text(json.dumps(settings), encoding='utf-8')
        (destination / settings['code_file']).write_bytes((kernel / settings['code_file']).read_bytes())


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ntl-continuation-test-')
        self.root = Path(self.temp.name).resolve()
        self.assertEqual(self.root.parent, Path(tempfile.gettempdir()).resolve())
        self.assertTrue(self.root.name.startswith('ntl-continuation-test-'))
        self.addCleanup(self.temp.cleanup)
        self.controller = load_controller()
        self.controller.ROOT = self.root
        self.controller.HERE = self.root / 'controller'
        self.controller.HERE.mkdir()
        self.frozen = self.root / 'frozen.py'
        self.frozen.write_text('# immutable synthetic input\n', encoding='utf-8')
        (self.controller.HERE / 'continuation-bindings.json').write_text(
            json.dumps({'frozen.py': digest(self.frozen)}), encoding='utf-8')
        (self.controller.HERE / 'execution-state.json').write_text(
            json.dumps({'status': 'CPU_PREFLIGHT_RUNNING', 'gpu_started': False, 'gpu_run_requests': 0}), encoding='utf-8')
        self.api = FakeApi(self.controller, self.root)
        self.build_calls = []
        self.build_returncode = 0

    def fake_prepare(self, command, **kwargs):
        self.build_calls.append(command)
        self.assertEqual(Path(command[4]), self.controller.HERE / 'prepare_capture.py')
        manifest = Path(command[command.index('--preflight-manifest') + 1])
        if self.build_returncode or json.loads(manifest.read_text())['status'] != 'PASS':
            return SimpleNamespace(returncode=self.build_returncode or 1, stdout='Synthetic packaging rejection')
        destination = Path(command[command.index('--destination') + 1])
        kernel = destination / 'kernel'
        kernel.mkdir(parents=True)
        settings = {'id': self.controller.GPU, 'code_file': 'capture.py', 'is_private': True,
                    'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False,
                    'docker_image': 'synthetic-pinned-image', 'machine_shape': 'NvidiaTeslaT4',
                    'dataset_sources': ['owner/dataset/3'], 'kernel_sources': ['owner/kernel/1'],
                    'model_sources': ['owner/model/Transformers/variation/1'], 'competition_sources': ['name-that-llm']}
        (kernel / 'kernel-metadata.json').write_text(json.dumps(settings), encoding='utf-8')
        (kernel / 'capture.py').write_text('# fake payload; never executed\n', encoding='utf-8')
        (destination / 'launch-package.json').write_text(json.dumps({
            'status': 'PREPARED_NOT_LAUNCHED', 'run_id': 'synthetic-run',
            'bootstrap_sha256': digest(kernel / 'capture.py'), 'metadata_sha256': digest(kernel / 'kernel-metadata.json'),
            'preflight_manifest_sha256': digest(manifest), 'payload_zip_sha256': 'e' * 64,
            'plan_sha256': 'f' * 64}), encoding='utf-8')
        return SimpleNamespace(returncode=0, stdout='Synthetic package ready')

    def run_controller(self, succeeds=False, monotonic=None):
        with ExitStack() as stack:
            stack.enter_context(patch.object(self.controller, 'KaggleApi', return_value=self.api))
            stack.enter_context(patch.object(self.controller.subprocess, 'run', side_effect=self.fake_prepare))
            self.sleeper = stack.enter_context(patch.object(self.controller.time, 'sleep'))
            stack.enter_context(patch.object(sys, 'argv', ['continuation', '--build-python', 'synthetic-python']))
            stack.enter_context(redirect_stdout(io.StringIO()))
            if monotonic is not None:
                stack.enter_context(patch.object(self.controller.time, 'monotonic', side_effect=monotonic))
            if succeeds:
                self.controller.main()
            else:
                with self.assertRaises(SystemExit) as error:
                    self.controller.main()
                self.assertEqual(error.exception.code, 1)

    def state(self):
        return json.loads((self.controller.HERE / 'execution-state.json').read_text(encoding='utf-8'))

    def test_success_is_one_quick_save_then_exactly_one_run(self):
        self.api.statuses = deque(['RUNNING', 'QUEUED', 'COMPLETE'])
        self.run_controller(succeeds=True)
        self.assertEqual(self.api.pushes, [True, False])
        self.assertEqual(len(self.build_calls), 1)
        self.assertEqual(self.sleeper.call_count, 2)
        self.assertEqual(self.state()['gpu_run_requests'], 1)
        self.assertTrue(self.state()['gpu_readback_verified'])
        self.assertFalse(self.state()['automatic_gpu_continuation'])
        self.assertEqual([call[1] for call in self.api.calls if call[0] == 'status'][:3], [self.controller.CPU] * 3)
        for call in self.api.calls:
            if call[0] == 'push':
                self.assertEqual(call[-2:], (28800, 'NvidiaTeslaT4'))

    def test_cpu_failed_means_no_gpu_save_or_run(self):
        self.api.statuses = deque(['ERROR'])
        self.run_controller()
        self.assertEqual(self.api.pushes, [])
        self.assertEqual(self.build_calls, [])
        self.assertFalse(self.state()['gpu_started'])

    def test_cpu_observation_timeout_means_no_gpu_calls(self):
        self.api.statuses = deque(['RUNNING'])
        times = iter([0, 6000])
        self.run_controller(monotonic=lambda: next(times, 6000))
        self.assertEqual(self.api.pushes, [])
        self.assertEqual(self.build_calls, [])

    def test_duplicate_cpu_manifest_is_rejected(self):
        self.api.manifest_count = 2
        self.run_controller()
        self.assertEqual(self.api.pushes, [])
        self.assertEqual(self.build_calls, [])

    def test_invalid_cpu_manifest_packaging_cannot_launch_gpu(self):
        self.api.manifest_status = 'FAILED'
        self.run_controller()
        self.assertEqual(self.api.pushes, [])
        self.assertEqual(len(self.build_calls), 1)

    def test_insufficient_quota_stops_before_packaging(self):
        self.api.remaining_seconds = 28799
        self.run_controller()
        self.assertEqual(self.api.pushes, [])
        self.assertEqual(self.build_calls, [])

    def test_exactly_eight_hours_quota_is_admitted(self):
        self.api.remaining_seconds = 28800
        self.run_controller(succeeds=True)
        self.assertEqual(self.api.pushes, [True, False])

    def test_changed_frozen_source_is_rejected(self):
        self.frozen.write_text('# changed\n', encoding='utf-8')
        self.run_controller()
        self.assertEqual(self.api.calls, [])
        self.assertEqual(self.api.pushes, [])

    def test_quick_save_rejection_never_sends_run(self):
        self.api.save_error = 'synthetic invalid input'
        self.run_controller()
        self.assertEqual(self.api.pushes, [True])
        self.assertEqual(self.state()['gpu_run_requests'], 0)

    def test_ambiguous_quick_save_is_not_retried(self):
        self.api.fail_save = ConnectionError('Synthetic lost quick-save response')
        self.run_controller()
        self.assertEqual(self.api.pushes, [True])
        self.assertFalse(self.state()['gpu_started'])
        self.assertEqual(self.state()['gpu_run_requests'], 0)

    def test_existing_exact_gpu_target_prevents_all_saves(self):
        self.api.existing = [SimpleNamespace(ref=self.controller.GPU)]
        self.run_controller()
        self.assertEqual(self.api.pushes, [])

    def test_nonidentical_search_result_does_not_block_target(self):
        self.api.existing = [SimpleNamespace(ref=self.controller.GPU + '-unrelated')]
        self.run_controller(succeeds=True)
        self.assertEqual(self.api.pushes, [True, False])

    def test_kernel_changed_after_quick_save_readback_is_not_run(self):
        original = self.controller.verify_saved

        def readback_then_change(api, version, expected_dir, out):
            original(api, version, expected_dir, out)
            if version == 1:
                (expected_dir / 'capture.py').write_text('# changed after verified readback\n', encoding='utf-8')

        with patch.object(self.controller, 'verify_saved', side_effect=readback_then_change):
            self.run_controller()
        self.assertEqual(self.api.pushes, [True])
        self.assertEqual(self.state()['gpu_run_requests'], 0)

    def test_quick_save_readback_mismatch_never_sends_run(self):
        self.api.bad_pull_number = 1
        self.run_controller()
        self.assertEqual(self.api.pushes, [True])
        self.assertEqual(self.state()['gpu_run_requests'], 0)

    def test_ambiguous_run_is_not_retried_and_stays_unknown(self):
        self.api.fail_run = ConnectionError('Lost POST response https://example.invalid/run?token=SECRET')
        self.run_controller()
        self.assertEqual(self.api.pushes, [True, False])
        state = self.state()
        self.assertEqual(state['gpu_run_requests'], 1)
        self.assertIsNone(state['gpu_started'])
        self.assertTrue(state['gpu_may_be_active'])
        self.assertEqual(state['status'], 'CONTINUATION_REQUIRES_REVIEW')
        self.assertFalse(state['automatic_gpu_continuation'])
        text = (self.controller.HERE / 'continuation-failure.json').read_text(encoding='utf-8')
        self.assertNotIn('token=SECRET', text)
        self.assertIn('[redacted]', text)

    def test_acknowledged_run_readback_failure_does_not_relaunch(self):
        self.api.fail_pull_number = 2
        self.run_controller()
        self.assertEqual(self.api.pushes, [True, False])
        self.assertTrue(self.state()['gpu_started'])
        self.assertEqual(self.state()['gpu_run_requests'], 1)

    def test_existing_lock_preserves_old_state_and_prevents_all_calls(self):
        path = self.controller.HERE / 'execution-state.json'
        original = path.read_bytes()
        (self.controller.HERE / 'continuation.lock').write_text('previous owner')
        with patch.object(self.controller, 'KaggleApi', return_value=self.api), \
                patch.object(sys, 'argv', ['continuation', '--build-python', 'synthetic-python']):
            with self.assertRaises(FileExistsError):
                self.controller.main()
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(self.api.calls, [])


if __name__ == '__main__':
    unittest.main()
