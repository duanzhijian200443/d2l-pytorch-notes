"""Synthetic-only checks; never fit real data or execute the payload builder."""
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
SPEC = importlib.util.spec_from_file_location('fixed_multiclass_runner', HERE / 'runner.py')
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)
BUILD_SPEC = importlib.util.spec_from_file_location('fixed_multiclass_builder', HERE / 'build_payload.py')
B = importlib.util.module_from_spec(BUILD_SPEC)
BUILD_SPEC.loader.exec_module(B)


def accepted_fixture():
    labels = R.EVALUATION_LABELS
    models = labels + [labels[0]] * 3
    frame = pd.DataFrame({'id': [f'sample-{i}' for i in range(len(models))],
                          'prompt_id': ['complete'] * 17 + ['incomplete'] * 3,
                          'model': models, 'text': [f'synthetic text {i}' for i in range(len(models))]})
    hashes = dict(R.FROZEN_INPUTS)
    hashes.update({name: 'a' * 64 for name in ('runner.py', 'guard.py', 'evaluation.csv', 'data_gate.json')})
    protocol = {'evaluation_accepted': True, 'evaluation_gate_status': R.GATE_STATUS,
                'labels': R.LABELS, 'required_versions': R.REQUIRED_VERSIONS, 'train_sha256': R.TRAIN_SHA256,
                'alpha': .1, 'style_share': .5, 'public_row_weight': .35,
                'evaluation_labels': labels, 'evaluation_rows': len(frame), 'evaluation_groups': 2,
                'evaluation_class_counts': frame.model.value_counts().to_dict(),
                'evaluation_sha256': hashes['evaluation.csv'], 'data_gate_sha256': hashes['data_gate.json'],
                'work_budget_seconds': 3300, 'platform_wall_seconds': 3600, 'rss_limit_bytes': 12 * 1024**3,
                'threads': 2, 'gpu': False, 'internet': False, 'bootstrap_draws': 2000, 'bootstrap_seed': 42,
                'input_hashes': hashes}
    gate = {'status': R.GATE_STATUS, 'evaluation': {
        'rows': len(frame), 'groups': 2, 'labels': labels,
        'class_counts': protocol['evaluation_class_counts'], 'sha256': protocol['evaluation_sha256']}}
    cfg = {'evaluation': {'labels': labels, 'max_rows': 749, 'alpha': .1, 'retune': False,
                         'output_classes': 18, 'sparse_sweep': False, 'excluded_label': 'gemini-3.7-flash',
                         'bootstrap_draws': 2000, 'bootstrap_seed': 42},
           'model': {'parameters': {'alpha': .1, 'retune': False}},
           'training': {'official_rows': 2699, 'stylegram_extra_rows': 10073, 'svc4_public_rows': 5400,
                        'svm_fits': 3, 'stylegram': {'C': .5, 'max_iter': 4000, 'dual': True,
                        'random_state': 42, 'share': .5, 'dtype': 'float64'},
                        'svc4': {'C': .5, 'max_iter': 5000, 'dual': 'auto', 'class_weight': 'balanced',
                        'random_state': 42, 'public_row_weight': .35, 'dtype': 'float32'}},
           'environment': {'required_versions': R.REQUIRED_VERSIONS, 'threads': 2,
                           'accelerator': None, 'internet': False, 'container': B.CONTAINER},
           'budget': {'work_budget_seconds': 3300, 'platform_wall_seconds': 3600, 'rss_limit_bytes': 12 * 1024**3}}
    return frame, protocol, gate, cfg


def original_nodes(path, predicate, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    selected = [node for node in tree.body if predicate(node)]
    if not selected:
        raise AssertionError('No reference nodes selected')
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class WeightAndMetadataTests(unittest.TestCase):
    def test_original_cpu_math_and_training_helpers_are_ast_identical(self):
        old = ast.parse((ROOT / 'cloud/fixed-blend-external/runner.py').read_text(encoding='utf-8'))
        new = ast.parse((HERE / 'runner.py').read_text(encoding='utf-8'))
        names = ('style_sample_weights', 'make_style_classifier', 'fit_checked', 'predict_stylegram',
                 'covered_label_metrics', 'paired_interval', 'weight_receipt', 'matrix_receipt', 'vectorizer_receipts')
        for name in names:
            with self.subTest(name=name):
                a = next(node for node in old.body if isinstance(node, ast.FunctionDef) and node.name == name)
                b = next(node for node in new.body if isinstance(node, ast.FunctionDef) and node.name == name)
                self.assertEqual(ast.dump(a), ast.dump(b))
        self.assertEqual(B.FROZEN_INPUTS, R.FROZEN_INPUTS)

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


class PanelGateTests(unittest.TestCase):
    def test_accepted_unequal_incomplete_panel_passes_without_fit(self):
        frame, protocol, gate, cfg = accepted_fixture()
        R.validate_protocol(protocol)
        np.testing.assert_array_equal(R.validate_evaluation(frame, protocol, gate), frame.id.to_numpy())
        self.assertEqual(B.validate_readiness(gate, cfg, frame, R.LABELS), gate['evaluation'])

    def test_protocol_rejects_label_counts_gate_and_budget_changes(self):
        _, protocol, _, _ = accepted_fixture()
        for field, value in [('evaluation_gate_status', 'PENDING'), ('evaluation_labels', R.EVALUATION_LABELS[::-1]),
                             ('evaluation_rows', protocol['evaluation_rows'] + 1), ('threads', 4),
                             ('gpu', True), ('bootstrap_draws', 1000), ('bootstrap_seed', 43),
                             ('evaluation_sha256', 'b' * 64), ('data_gate_sha256', 'b' * 64)]:
            changed = copy.deepcopy(protocol)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                R.validate_protocol(changed)

    def test_preflight_rejects_gate_or_class_support_changes(self):
        frame, protocol, gate, cfg = accepted_fixture()
        changed = frame.copy()
        changed.loc[len(changed) - 1, 'model'] = R.EVALUATION_LABELS[1]
        with self.assertRaisesRegex(ValueError, 'class support'):
            R.validate_evaluation(changed, protocol, gate)
        with self.assertRaisesRegex(ValueError, 'class counts'):
            B.validate_readiness(gate, cfg, changed, R.LABELS)
        gate['status'] = 'PENDING'
        with self.assertRaisesRegex(ValueError, 'not accepted'):
            R.validate_evaluation(frame, protocol, gate)
        with self.assertRaisesRegex(ValueError, 'not accepted'):
            B.validate_readiness(gate, cfg, frame, R.LABELS)

    def test_builder_rejects_retuning_or_recipe_drift(self):
        frame, _, gate, cfg = accepted_fixture()
        for section, key, value in [('evaluation', 'alpha', .2), ('evaluation', 'sparse_sweep', True),
                                    ('environment', 'threads', 4), ('budget', 'work_budget_seconds', 6600),
                                    ('training', 'svc4_public_rows', 5401)]:
            changed = copy.deepcopy(cfg)
            changed[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                B.validate_readiness(gate, changed, frame, R.LABELS)


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

    def test_seventeen_imbalanced_true_classes_keep_all_prediction_columns(self):
        covered = np.array([R.LABELS.index(name) for name in R.EVALUATION_LABELS])
        truth = np.r_[covered, np.repeat(covered[0], 3)]
        prediction = truth.copy()
        prediction[0] = R.LABELS.index('gemini-3.7-flash')
        metrics = R.covered_label_metrics(truth, prediction, R.LABELS)
        self.assertEqual(metrics['score_columns'], 18)
        self.assertEqual(metrics['covered_label_count'], 17)
        self.assertAlmostEqual(metrics['macro_recall'], (16 + .75) / 17)
        self.assertEqual(metrics['accuracy'], .95)
        self.assertNotEqual(metrics['macro_recall'], metrics['accuracy'])
        self.assertEqual(metrics['correct'], 19)
        self.assertNotIn('balanced_accuracy', metrics)

    def test_paired_interval_is_covered_class_metric(self):
        covered = np.array([R.LABELS.index(name) for name in R.EVALUATION_LABELS])
        truth = np.tile(covered, 10)
        prompts = np.repeat([f'p{i}' for i in range(10)], 17)
        reference = truth.copy(); reference[0] = 7
        interval = R.paired_interval(truth, reference, truth.copy(), prompts, R.LABELS)
        self.assertEqual(interval['invalid_draws'], 0)
        self.assertEqual(interval['delta_definition'], 'blend_minus_stylegram_covered_label_macro_recall')
        self.assertGreaterEqual(interval['ci95'][0], 0)
        self.assertGreater(interval['ci95'][1], 0)

    def test_any_missing_class_bootstrap_draw_nulls_whole_interval(self):
        truth = np.array([R.LABELS.index(name) for name in R.EVALUATION_LABELS])
        prompts = np.array(['common'] * 16 + ['rare'])
        interval = R.paired_interval(truth, truth.copy(), truth.copy(), prompts, R.LABELS)
        self.assertEqual(interval['draws'], 2000)
        self.assertEqual(interval['seed'], 42)
        self.assertGreater(interval['invalid_draws'], 0)
        self.assertEqual(interval['valid_draws'] + interval['invalid_draws'], 2000)
        self.assertIsNone(interval['ci95'])


class SafeSerializationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='fixed-blend-multiclass-test-')
        self.root = Path(self.temp.name).resolve()
        self.assertEqual(self.root.parent, Path(tempfile.gettempdir()).resolve())
        self.assertTrue(self.root.name.startswith('fixed-blend-multiclass-test-'))
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
