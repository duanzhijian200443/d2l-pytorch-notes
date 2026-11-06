"""Small synthetic A1 parity and strict patch/input gates; no real-data fit."""
import ast
import contextlib
import gc
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


a1 = load(HERE / 'a1_stylegram.py', 'test_a1')
patcher = load(HERE / 'patch_comp.py', 'test_patch_a1')


def fixture():
    official, extra = [], []
    for prompt in range(3):
        for i, label in enumerate(a1.LABELS):
            official.append({'text': f'Official topic {prompt}; style{i} gives helpful details and careful reasons, example {i}.',
                             'model': label, 'prompt_id': f'o{prompt}'})
    for prompt in range(2):
        for i, label in enumerate(a1.LABELS):
            extra.append({'text': f'Public topic {prompt}; style{i} supplies another answer with explicit details and example {i}.',
                          'model': label, 'prompt_id': f'e{prompt}'})
    evaluation = pd.DataFrame({'text': ['HIDDENSENTINEL garden soil watering flowers',
                                       'HIDDENSENTINEL garden soil watering flowers',
                                       'HIDDENSENTINEL nebula star orbital telescope']})
    return pd.DataFrame(official), pd.DataFrame(extra), evaluation


class A1Tests(unittest.TestCase):
    def test_cloud_small_selftest_has_no_dependency_on_training_csv(self):
        with contextlib.redirect_stdout(io.StringIO()):
            receipt = a1.self_test(ROOT / 'experiments/EXP-010-012-stylegram-share/payload/stylegram_core.py')
        self.assertEqual(receipt['status'], 'PASS')
        self.assertLessEqual(receipt['maximum_raw_margin_difference'], 1e-12)
        self.assertLessEqual(receipt['maximum_probability_difference'], 1e-12)
        self.assertTrue(receipt['both_inference_routes_exercised'])
        self.assertEqual(receipt['training_rows'], 54)

    def test_exact_exp027_ast_training_and_prediction_path_on_synthetic_inputs(self):
        reference_path = ROOT / 'cloud/fixed-blend-multiclass/runner.py'
        reference = load(reference_path, 'test_exp027_reference')
        core = a1.load_core(ROOT / 'experiments/EXP-010-012-stylegram-share/payload/stylegram_core.py')
        official, extra, evaluation = fixture()
        index = {label: i for i, label in enumerate(a1.LABELS)}
        y, ey = [frame.model.map(index).to_numpy(dtype=int) for frame in (official, extra)]
        tree = ast.parse(reference_path.read_text(encoding='utf-8'))
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
        start = next(i for i, node in enumerate(run.body) if isinstance(node, ast.Assign)
                     and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'texts')
        end = next(i for i, node in enumerate(run.body) if isinstance(node, ast.Assign)
                   and isinstance(node.targets[0], ast.Tuple) and any(isinstance(item, ast.Name) and item.id == 'style_scores' for item in node.targets[0].elts))
        # Execute the original EXP027 AST slice, not a hand-rewritten baseline.
        original_math = ast.Module(body=run.body[start:end+1], type_ignores=[])
        with tempfile.TemporaryDirectory() as directory, threadpool_limits(limits=2), contextlib.redirect_stdout(io.StringIO()):
            namespace = dict(vars(reference))
            namespace.update(official=official, extra=extra, evaluation=evaluation, y=y, extra_y=ey,
                             labels=list(a1.LABELS), core=core, timings={}, begun=time.monotonic(),
                             out=Path(directory), completed=[], phase=lambda *args: None, gc=gc)
            exec(compile(original_math, str(reference_path), 'exec'), namespace)
            fitted_on = []
            original_fit = core.StyleGrams.fit
            def record_fit(instance, texts):
                fitted_on.append(list(texts))
                return original_fit(instance, texts)
            with mock.patch.object(core.StyleGrams, 'fit', record_fit):
                probability, metadata = a1._fit_predict(official.text.tolist(), y, official.prompt_id.to_numpy(),
                    evaluation.text.tolist(), extra.text.tolist(), ey, extra.prompt_id.to_numpy(), core)
        self.assertEqual(fitted_on, [official.text.tolist()+extra.text.tolist()])
        self.assertFalse(any('HIDDENSENTINEL' in text for text in fitted_on[0]))
        for actual, expected in ((metadata['raw_margins'], namespace['style_scores']),
                                 (metadata['plain_margins'], namespace['plain_scores'])):
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
            np.testing.assert_array_equal(actual.argmax(1), expected.argmax(1))
        np.testing.assert_array_equal(metadata['groups'], namespace['groups'])
        np.testing.assert_array_equal(metadata['centred_mask'], namespace['mask'])
        self.assertTrue(metadata['centred_mask'].any())
        self.assertTrue((~metadata['centred_mask']).any())
        shifted = namespace['style_scores']-namespace['style_scores'].max(axis=1, keepdims=True)
        expected_probability = np.exp(shifted)
        expected_probability /= expected_probability.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(probability, expected_probability, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(a1._weights(y, ey), reference.style_sample_weights(y, ey))
        self.assertEqual(a1._classifier().get_params(), reference.make_style_classifier().get_params())
        self.assertEqual(metadata['receipt']['evaluation_rows_in_classifier_tfidf_fit'], 0)

    def test_public_api_rejects_unpinned_rows_labels_and_extra_before_fit(self):
        tr = ['synthetic official'] * 2699
        y = np.arange(2699) % 18
        groups = np.array([f'p{i % 150}' for i in range(2699)])
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(a1, '_fit_predict') as fit:
            extra = Path(directory) / 'extra.csv'
            extra.write_text('not the frozen CSV', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'CSV hash'):
                a1.predict(tr, y, groups, ['hidden text'], extra)
            with self.assertRaisesRegex(ValueError, 'Caller changed'):
                a1.predict(tr, y, groups, ['hidden text'], extra, extra_sha256='0'*64)
            with self.assertRaisesRegex(ValueError, 'label order'):
                a1.predict(tr, y, groups, ['hidden text'], extra, labels=list(reversed(a1.LABELS)))
            with self.assertRaisesRegex(ValueError, 'cardinality'):
                a1.predict(tr[:-1], y[:-1], groups[:-1], ['hidden text'], extra)
            fit.assert_not_called()

    def test_core_hash_rejects_changed_source_before_import(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'core.py'
            path.write_text('raise RuntimeError("must not execute")', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'core hash'):
                a1.load_core(path)

    def test_patch_is_two_regions_only_and_preserves_family_gate_bytes(self):
        original = (ROOT / 'experiments/EXP-009-baseline-data-lineage/comp_stack.py').read_bytes()
        patched, receipt = patcher.patch_source(original)
        self.assertTrue(receipt['family_gate_bytes_equal'])
        self.assertTrue(receipt['global_extra_loader_bytes_equal'])
        self.assertTrue(receipt['suffix_after_cleanup_bytes_equal'])
        self.assertEqual(receipt['patched_sha256'], hashlib.sha256(patched).hexdigest())
        self.assertEqual(patched.count(b"add_member('stylegram', P_style)"), 1)
        self.assertEqual(patched.count(b'EX_T, EX_Y, EX_G ='), 1)
        self.assertNotIn(patcher.CLEANUP, patched)
        self.assertIn(b"os.environ['NTL_A1_EXTRA']", patched)
        self.assertIn(b"os.environ['NTL_A1_MODULE_DIR']", patched)
        self.assertIn(b"pathlib.Path('/kaggle/working/ntl_a1_payload')", patched)
        start, gate, cleanup = original.index(patcher.START), original.index(patcher.GATES), original.index(patcher.CLEANUP)
        self.assertEqual(patched[:start], original[:start])
        self.assertEqual(patched[patched.index(patcher.GATES):patched.index(b'# A1 Stylegram matrices')], original[gate:cleanup])
        # Python compilation confirms no syntax damage; undefined old matrices
        # disappear because both their block and sole later cleanup were removed.
        compile(patched, 'patched_comp.py', 'exec')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            patcher.patch_source(patched)
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            patcher.patch_source(original + b'\n')


if __name__ == '__main__':
    unittest.main()
