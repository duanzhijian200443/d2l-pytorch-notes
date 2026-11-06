"""Synthetic-only checks for external-panel recipes, routing and serialization."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPEC = importlib.util.spec_from_file_location('fixed_external_runner', HERE / 'runner.py')
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


def original_nodes(path, predicate, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    selected = [node for node in tree.body if predicate(node)]
    if not selected:
        raise AssertionError('No reference nodes selected')
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class WeightAndMetadataTests(unittest.TestCase):
    def test_style_weights_equal_original_ast_function(self):
        source = ROOT / 'experiments/EXP-010-012-stylegram-share/runner.py'
        reference = original_nodes(source, lambda node: isinstance(node, ast.FunctionDef) and node.name == 'weights', {'np': np})['weights']
        y = np.repeat(np.arange(18), np.arange(18) + 2)
        extra = np.repeat(np.arange(18), np.arange(18) + 4)
        actual = R.style_sample_weights(y, extra)
        np.testing.assert_array_equal(actual, reference(y, extra, .5))
        self.assertEqual(actual.dtype, np.float64)
        self.assertTrue(np.all(actual[:len(y)] == .5))

    def test_style_parameters_match_original_factory(self):
        source = ROOT / 'experiments/EXP-010-012-stylegram-share/runner.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'fit_model')
        construction = next(node for node in function.body if isinstance(node, ast.Assign))
        namespace = {'LinearSVC': R.LinearSVC, 'P': {'C': .5, 'max_iter': 4000, 'seed': 42, 'dual': True}}
        exec(compile(ast.Module(body=[construction], type_ignores=[]), str(source), 'exec'), namespace)
        self.assertEqual(R.make_style_classifier().get_params(), namespace['model'].get_params())
        self.assertIsNone(R.make_style_classifier().class_weight)

    def test_gate_refuses_before_needing_any_data(self):
        for protocol in ({}, {'evaluation_accepted': False}):
            with self.subTest(protocol=protocol), self.assertRaisesRegex(ValueError, 'not accepted'):
                R.validate_protocol(protocol)

    def test_metadata_handles_numpy_vocabulary_indices(self):
        vectorizer = SimpleNamespace(vocabulary_={'词': np.int64(1), 'abc': np.int32(0)}, idf_=np.array([1., 2.], dtype=np.float32))
        receipt = R.vectorizer_receipts([('word', vectorizer, .7)])
        json.dumps(receipt, allow_nan=False)
        self.assertEqual(receipt[0]['columns'], 2)


class RouteAndMetricTests(unittest.TestCase):
    def test_style_route_matches_original_prediction_ast(self):
        matrix = csr_matrix(np.array([[2., 1.], [0., 3.], [4., 2.]]))
        texts = ['first unlabelled text', 'second unlabelled text', 'third unlabelled text']
        called = []

        class Features:
            def transform(self, values):
                self.values = values
                return matrix.copy()

        class Model:
            def __init__(self, coefficients):
                self.coefficients = coefficients

            def decision_function(self, values):
                return np.asarray(values @ self.coefficients)

        source = ROOT / 'experiments/EXP-010-012-stylegram-share/payload/stylegram_core.py'
        namespace = {'np': np, 'csr_matrix': csr_matrix, 'GROUP_MIN': 2, 'GROUP_MAX': 18}
        from scipy.sparse import diags
        namespace['diags'] = diags
        original_nodes(source, lambda node: isinstance(node, ast.FunctionDef) and node.name in ('sparse_group_centre', 'eligible'), namespace)

        def group(values):
            called.append(values)
            return np.array([1, 1, 2])

        core = SimpleNamespace(normalised_content_groups=group, eligible=namespace['eligible'], sparse_group_centre=namespace['sparse_group_centre'])
        plain = Model(np.array([[1., 0., -1.], [0., 1., 1.]]))
        centered = Model(np.array([[2., 0., 1.], [0., 2., -1.]]))
        actual, raw, groups, mask = R.predict_stylegram(Features(), plain, centered, texts, core)
        reference = {'plain': plain, 'center': centered, 'xv': matrix, 'mask': mask,
                     'xvc': core.sparse_group_centre(matrix[mask], groups[mask])}
        runner = ROOT / 'experiments/EXP-010-012-stylegram-share/runner.py'
        tree = ast.parse(runner.read_text(encoding='utf-8'))
        nodes = [node for node in ast.walk(tree) if isinstance(node, (ast.Assign, ast.If)) and node.lineno in (104, 105)]
        # Keep only outer statements; the assignment inside line 105 belongs
        # to the selected If and must not execute a second time independently.
        nodes = [node for node in nodes if not (isinstance(node, ast.Assign) and node.lineno == 105)]
        nodes.sort(key=lambda node: (node.lineno, node.col_offset))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(runner), 'exec'), reference)
        np.testing.assert_array_equal(actual, reference['scores'])
        np.testing.assert_array_equal(actual[~mask], raw[~mask])
        self.assertEqual(called, [texts])
        self.assertEqual(mask.tolist(), [True, True, False])

    def test_four_true_classes_keep_all_eighteen_prediction_columns(self):
        truth = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        prediction = np.array([0, 1, 17, 3, 0, 1, 2, 3])
        metrics = R.covered_label_metrics(truth, prediction, R.LABELS)
        self.assertEqual(metrics['score_columns'], 18)
        self.assertEqual(metrics['covered_label_count'], 4)
        self.assertEqual(metrics['macro_recall'], .875)
        self.assertEqual(metrics['correct'], 7)
        self.assertNotIn('balanced_accuracy', metrics)

    def test_paired_interval_is_covered_class_metric(self):
        truth = np.tile(np.arange(4), 10)
        prompts = np.repeat([f'p{i}' for i in range(10)], 4)
        reference = truth.copy(); reference[0] = 17
        interval = R.paired_interval(truth, reference, truth.copy(), prompts, R.LABELS)
        self.assertEqual(interval['invalid_draws'], 0)
        self.assertEqual(interval['delta_definition'], 'blend_minus_stylegram_covered_label_macro_recall')
        self.assertGreaterEqual(interval['ci95'][0], 0)
        self.assertGreater(interval['ci95'][1], 0)


class SafeSerializationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='fixed-blend-external-test-')
        self.root = Path(self.temp.name).resolve()
        self.assertEqual(self.root.parent, Path(tempfile.gettempdir()).resolve())
        self.assertTrue(self.root.name.startswith('fixed-blend-external-test-'))
        self.addCleanup(self.temp.cleanup)

    def test_json_scalars_and_npz_roundtrip_without_pickle(self):
        path = self.root / 'receipt.json'
        R.save_json(path, {'index': np.int64(2), 'time': np.float64(.125), 'weights': np.array([.5, .25])})
        self.assertEqual(json.loads(path.read_text()), {'index': 2, 'time': .125, 'weights': [.5, .25]})
        array_path = self.root / 'scores.npz'
        R.save_npz(array_path, ids=np.array(['例一', 'two']), scores=np.eye(2), alpha=np.array(.1))
        with np.load(array_path, allow_pickle=False) as stored:
            self.assertEqual(stored['ids'].tolist(), ['例一', 'two'])
            np.testing.assert_array_equal(stored['scores'], np.eye(2))
        self.assertFalse(list(self.root.glob('*.tmp')))
        with self.assertRaises(ValueError):
            R.save_npz(self.root / 'unsafe.npz', ids=np.array(['x'], dtype=object))
        with self.assertRaises(ValueError):
            R.save_json(self.root / 'nonfinite.json', {'value': np.float64(np.nan)})

    def test_unaccepted_panel_records_error_before_data_or_fit(self):
        (self.root / 'protocol.json').write_text('{"evaluation_accepted": false}', encoding='utf-8')
        out = self.root / 'output'
        out.mkdir()
        (out / '.run.lock').write_text('synthetic guard-owned lock', encoding='utf-8')
        with patch.object(R, 'HERE', self.root), patch.dict(os.environ, {'NTL_OUTPUT_DIR': str(out)}):
            with self.assertRaisesRegex(ValueError, 'not accepted'):
                R.main()
        self.assertEqual(json.loads((out / 'state.json').read_text())['status'], 'ERROR')
        self.assertIn('not accepted', json.loads((out / 'failure.json').read_text())['traceback'])
        self.assertFalse(list(out.glob('fit-*.json')))
        self.assertFalse((out / 'result.json').exists())

    def test_existing_result_and_state_are_preserved(self):
        out = self.root / 'output'
        out.mkdir()
        existing = {'result.json': b'{"status":"COMPLETE"}', 'state.json': b'{"status":"COMPLETE"}'}
        for name, data in existing.items():
            (out / name).write_bytes(data)
        with patch.dict(os.environ, {'NTL_OUTPUT_DIR': str(out)}):
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                R.main()
        for name, data in existing.items():
            self.assertEqual((out / name).read_bytes(), data)
        self.assertFalse((out / 'failure.json').exists())


if __name__ == '__main__':
    unittest.main()
