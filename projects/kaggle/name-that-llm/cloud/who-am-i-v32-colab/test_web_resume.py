"""Local CPU evidence only: frozen math, stage recovery, and resource cleanup."""
import ast
import contextlib
import gc
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import re
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from web_checkpoint import Checkpoints, signature
from web_instrument import instrument_infer

ROOT = Path(__file__).resolve().parent
SOURCE = (ROOT / 'upstream-infer.py').read_text(encoding='utf-8')
ORIGINAL = ast.parse(SOURCE)
PATCHED = ast.parse(instrument_infer(SOURCE))
STAGES = ['tfidf', 'svc4', 'distort']
TESTS = ['a.tfidf_w > 0', 'a.svc_w > 0', 'a.distort_w > 0']


def run_ast(nodes, namespace):
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), 'fixture', 'exec'), namespace)


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_roundtrip_and_rng(self):
        ck = Checkpoints(self.root, 'same')
        random.seed(19); np.random.seed(41)
        value = {'array': np.arange(30, dtype=np.float32).reshape(3, 10), 'tuple': (None, True, [3, 'x'])}
        ck.save('a', value, 12.5)
        expected = (random.random(), np.random.rand())
        random.seed(111); np.random.seed(222)
        callbacks = []
        other = Checkpoints(self.root, 'same', callbacks.append)
        restored = other.load('a')
        np.testing.assert_array_equal(value['array'], restored['array'])
        self.assertEqual(value['tuple'], restored['tuple'])
        self.assertEqual(expected, (random.random(), np.random.rand()))
        self.assertEqual(callbacks, [12.5])
        self.assertEqual(other.replayed_seconds, 12.5)

    def test_corruption_context_and_unfinished(self):
        ck = Checkpoints(self.root, 'same')
        with self.assertRaises(RuntimeError):
            ck.call('failed', lambda: (_ for _ in ()).throw(RuntimeError('interrupted')))
        self.assertFalse(ck.marker('failed').exists())
        ck.save('done', {'x': np.ones(2)}, 3)
        self.assertIsNone(Checkpoints(self.root, 'different').load('done'))
        manifest = json.loads(ck.marker('done').read_text())
        (ck.root / manifest['payload']).write_bytes(b'partial write')
        self.assertIsNone(ck.load('done'))
        self.assertEqual(ck.call('done', lambda: {'replacement': 42}), {'replacement': 42})

    def test_nested_logical_duration(self):
        ck = Checkpoints(self.root, 'same')
        ck.save('inner', {'ok': True}, 120)
        ck.call('outer', lambda: ck.call('inner', lambda: self.fail('must restore')))
        outer = json.loads(ck.marker('outer').read_text())
        self.assertGreaterEqual(outer['seconds'], 120)
        self.assertLess(outer['seconds'], 125)
        fresh = Checkpoints(self.root, 'same')
        fresh.call('outer', lambda: self.fail('must restore'))
        self.assertAlmostEqual(fresh.replayed_seconds, outer['seconds'])

    def test_frozen_function_and_text_math(self):
        for name in ('l24_predict', 'atlas_scores'):
            before = next(x for x in ORIGINAL.body if isinstance(x, ast.FunctionDef) and x.name == name)
            after = next(x for x in PATCHED.body if isinstance(x, ast.FunctionDef) and x.name == name)
            self.assertEqual([ast.dump(n) for n in before.body], [ast.dump(n) for n in after.body])
        for test, stage in zip(TESTS, STAGES):
            before = next(x for x in ORIGINAL.body if isinstance(x, ast.If) and ast.unparse(x.test) == test)
            function = next(x for x in PATCHED.body if isinstance(x, ast.FunctionDef) and x.name == '_v32_compute_' + stage)
            self.assertEqual(ast.dump(before), ast.dump(function.body[1]))

    def test_actual_cpu_classifiers_interrupted_resume_matches_original(self):
        labels = ['class' + str(i) for i in range(18)]
        frame = pd.DataFrame([{'text': f'This is MODEL{c} prose, token{c} item{j}. We explain result{c} carefully!', 'model': labels[c]}
                              for c in range(18) for j in range(6)])
        def namespace(ck):
            return dict(np=np, pd=pd, sp=sp, gc=gc, re=re, os=os, TfidfVectorizer=TfidfVectorizer,
                        LogisticRegression=LogisticRegression, train=frame, test=frame.iloc[[1, 13, 51]].copy(),
                        LABELS=labels, L2I={s: i for i, s in enumerate(labels)},
                        FIT_T=frame.text.values, FIT_Y=np.repeat(np.arange(18), 6), FIT_W=np.ones(len(frame)),
                        EXTRA_DF=pd.DataFrame(columns=['text', 'model']), LP={}, log=lambda *a: None,
                        _norm_text_key=lambda s: re.sub(r'\s+', ' ', s).strip(), _v32_ck=ck,
                        a=types.SimpleNamespace(tfidf_w=.25, svc_w=1.5, distort_w=.10, perpassage=True,
                                                extra_w=.5, pool='', sim=False))
        original_blocks = [x for x in ORIGINAL.body if isinstance(x, ast.If) and ast.unparse(x.test) in TESTS]
        original = namespace(None)
        run_ast(original_blocks, original)
        wrappers = []
        for node in PATCHED.body:
            if isinstance(node, ast.FunctionDef) and node.name in ['_v32_compute_' + s for s in STAGES]:
                wrappers.append(node)
            elif isinstance(node, ast.Expr) and ast.unparse(node).startswith("LP.update(_v32_ck.call('text/"):
                wrappers.append(node)
        class Interrupted(Checkpoints):
            def save(self, key, value, seconds):
                super().save(key, value, seconds)
                if key == 'text/svc4':
                    raise RuntimeError('simulated process death after committed SVC')
        first = namespace(Interrupted(self.root, 'cpu'))
        with self.assertRaisesRegex(RuntimeError, 'simulated'):
            run_ast(wrappers, first)
        class VerifyRestores(Checkpoints):
            def save(self, key, value, seconds):
                if key in ('text/tfidf', 'text/svc4'):
                    raise AssertionError('completed stage recomputed')
                super().save(key, value, seconds)
        second = namespace(VerifyRestores(self.root, 'cpu'))
        run_ast(wrappers, second)
        self.assertEqual(set(original['LP']), set(second['LP']))
        for name in original['LP']:
            np.testing.assert_allclose(original['LP'][name], second['LP'][name], rtol=1e-12, atol=1e-12)

    def test_model_continue_nested_loop_and_memoized_resume(self):
        source = '''
import os, time
T0 = time.time()
LP = {}
if a.tfidf_w > 0: LP['text1'] = [1]
if a.svc_w > 0: LP['text2'] = [2]
if a.distort_w > 0: LP['text3'] = [3]
def l24_predict(spec, texts):
    counts[spec['name']] = counts.get(spec['name'], 0) + 1
    return np.array([spec['value']])
def atlas_scores(bundle, rows, budget):
    return np.array([budget])
for spec in specs:
    value = l24_predict(spec, ['text'])
    if fail and spec['name'] == 'second': raise RuntimeError('inside unfinished stage')
    if spec.get('skip'):
        LP[spec['name']] = value
        continue
    for i in range(3):
        if i == 1: continue
        value = value + i
    LP[spec['name']] = value
'''
        counts = {}
        def ns(fail):
            return {'np': np, 'counts': counts, 'fail': fail,
                    'a': types.SimpleNamespace(tfidf_w=1, svc_w=1, distort_w=1),
                    'specs': [{'name': 'first', 'value': 2, 'skip': True}, {'name': 'second', 'value': 4}]}
        baseline = ns(False); exec(source, baseline); counts.clear()
        code = instrument_infer(source)
        with patch.dict(os.environ, V32_CHECKPOINT_ROOT=str(self.root), V32_CHECKPOINT_CONTEXT='model'):
            with self.assertRaisesRegex(RuntimeError, 'unfinished'):
                exec(code, ns(True))
            resumed = ns(False); exec(code, resumed)
        self.assertEqual(counts, {'first': 1, 'second': 1})
        for name, expected in baseline['LP'].items():
            np.testing.assert_array_equal(expected, resumed['LP'][name])

    def test_runner_releases_on_success_error_and_interrupt(self):
        fake_fcntl = types.SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *a: None)
        fake_drive = types.SimpleNamespace(mount=lambda *a: None)
        google = types.ModuleType('google'); colab = types.ModuleType('google.colab'); colab.drive = fake_drive
        modules = {'fcntl': fake_fcntl, 'google': google, 'google.colab': colab}
        with patch.dict(sys.modules, modules):
            spec = importlib.util.spec_from_file_location('web_runner_test', ROOT / 'web_runner.py')
            runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
        for outcome in ('ok', 'error', 'interrupt', 'mount_error'):
            case = self.root / outcome; case.mkdir()
            drive = case / 'drive'; drive.mkdir()
            (drive / 'cli_wheels_result.json').write_text('{"status":"READY"}')
            work = case / 'work'
            calls = []
            def paths(value):
                string = str(value)
                return case / string[len('/content/'):] if string.startswith('/content/') else Path(value)
            class Process:
                pid = 999999
                returncode = 0
                def poll(self): return 0
            def popen(command, **kwargs):
                if str(command[-1]).endswith('web_execute.py'):
                    if outcome == 'interrupt': raise KeyboardInterrupt()
                    if outcome == 'error': raise RuntimeError('synthetic inference failure')
                    (runner.WORK / 'reproduction_result.json').write_text('{"sample_predictions_match":true}')
                return Process()
            fake_drive.mount = lambda *a: (_ for _ in ()).throw(RuntimeError('mount failed')) if outcome == 'mount_error' else None
            with patch.dict(sys.modules, modules), patch.object(runner, 'Path', paths), \
                 patch.object(runner, 'DRIVE', drive), patch.object(runner, 'WORK', work), \
                 patch.object(runner, 'prepare_cached_inputs', lambda: None), \
                 patch.object(runner, 'release', lambda: calls.append('release')), \
                 patch.object(runner.shutil, 'which', lambda _: 'nvidia-smi'), \
                 patch.object(runner.subprocess, 'check_output', lambda *a, **k: 'NVIDIA A100-SXM4-40GB, 580.82.07'), \
                 patch.object(runner.subprocess, 'Popen', popen), \
                 patch.object(runner.subprocess, 'run', lambda *a, **k: types.SimpleNamespace(returncode=0)), \
                 patch.object(runner.os, 'sync', lambda: None, create=True):
                if outcome == 'ok': runner.main()
                else:
                    with self.assertRaises(KeyboardInterrupt if outcome == 'interrupt' else RuntimeError): runner.main()
            self.assertEqual(calls, ['release'], outcome)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ResumeTests)
        result = unittest.TestResult(); suite.run(result)
    report = {'status': 'PASS' if result.wasSuccessful() else 'FAIL', 'tests_run': result.testsRun,
              'failures': [{'test': str(t), 'details': e} for t, e in result.failures + result.errors],
              'scope': 'local CPU and mocked lifecycle only; full A100 execution is not performed'}
    (ROOT / 'web-validation.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)
