"""Small independent metrics, pairing, artifact and weight gates; no training."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('exp042_verifier', HERE / 'verify_outputs.py')
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


class VerifierTests(unittest.TestCase):
    def test_fixed_class_metrics_do_not_drop_absent_classes(self):
        result = v.metrics(np.array([0, 0, 1]), np.array([0, 1, 1]), classes=2)
        self.assertAlmostEqual(result['balanced_accuracy'], .75)
        self.assertEqual(result['correct'], 2)
        self.assertAlmostEqual(result['accuracy'], 2/3)
        missing = v.metrics(np.array([0, 0]), np.array([0, 0]), classes=2)
        self.assertIsNone(missing['balanced_accuracy'])
        self.assertEqual(missing['missing_class_indices'], [1])

    def test_paired_bootstrap_reproducible_and_null_without_redraw(self):
        y = np.array([0, 0, 1, 1])
        base = np.array([0, 1, 0, 1])
        add = y.copy()
        groups = np.array(['a', 'a', 'b', 'b'])
        result = v.paired_metrics(y, base, add, groups, classes=2)
        self.assertEqual(result, v.paired_metrics(y, base, add, groups, classes=2))
        self.assertEqual(result['delta_ba'], .5)
        self.assertEqual(result['fixed_errors'], 2)
        self.assertEqual(result['introduced_errors'], 0)
        boot = result['bootstrap']
        self.assertEqual(len(boot['deltas']), 2000)
        self.assertGreater(boot['null_draws'], 0)
        self.assertEqual(boot['null_draws'], boot['deltas'].count(None))
        self.assertEqual(boot['redraws'], 0)
        self.assertEqual(boot['ci95'], [.5, .5])
        reverse = v.paired_metrics(y, add, base, groups, classes=2)
        self.assertEqual(reverse['delta_ba'], -.5)
        self.assertEqual(reverse['introduced_errors'], 2)

    def test_all_bootstrap_draws_missing_class_returns_null_interval(self):
        r = v.paired_metrics(np.array([0, 0]), np.array([0, 1]), np.array([0, 0]), np.array(['p', 'p']), classes=2)
        self.assertEqual(r['bootstrap']['null_draws'], 2000)
        self.assertIsNone(r['bootstrap']['ci95'])
        self.assertIsNone(r['delta_ba'])

    def test_strict_matrix_dtype_shape_and_finiteness(self):
        valid = np.zeros((3, 18), dtype=np.float64)
        v.score_matrix(valid, 3)
        for bad in (valid.astype(np.float32), np.zeros((3, 17)), valid + np.nan, valid + np.inf):
            with self.assertRaises(ValueError):
                v.score_matrix(bad, 3)
        for bad in (np.array([0., 1.]), np.array([0, 18]), np.array([True, False])):
            with self.assertRaises(ValueError):
                v.integer_vector(bad, 2, 'truth', maximum=17)

    def test_route_gate_requires_eligible_groups_and_plain_score_identity(self):
        data = {'ids': np.array(['a', 'b', 'c']), 'true': np.array([0, 1, 0]),
                'fold': np.zeros(3, dtype=int), 'arm': np.array('control'),
                'scores': np.zeros((3, 2)), 'plain_scores': np.zeros((3, 2)),
                'group_ids': np.array([1, 1, 2]), 'centred_mask': np.array([True, True, False]),
                'route': np.array(['centred', 'centred', 'plain'])}
        v.check_prediction(data, data['ids'], data['true'], data['fold'], 'control', classes=2)
        changed = copy.deepcopy(data)
        changed['scores'][2, 0] = 1
        with self.assertRaisesRegex(ValueError, 'plain route'):
            v.check_prediction(changed, data['ids'], data['true'], data['fold'], 'control', classes=2)
        changed = copy.deepcopy(data)
        changed['centred_mask'][2] = True
        with self.assertRaisesRegex(ValueError, 'eligible group'):
            v.check_prediction(changed, data['ids'], data['true'], data['fold'], 'control', classes=2)

    def test_independent_weight_check_rejects_renormalized_addition(self):
        labels = ['a', 'b']
        oy, ey, ny = np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1, 1, 1]), np.array([0, 1])
        receipt = {'old_row_weights_identical': True, 'classes': []}
        for i, n in enumerate([2, 4]):
            receipt['classes'].append({'class_index': i, 'label': labels[i], 'official_rows': 2,
                'base_extra_rows': n, 'added_rows': 1, 'official_per_row_weight': .5,
                'public_per_row_weight': 1/n, 'official_mass': 1., 'base_extra_mass': 1.,
                'control_mass': 2., 'added_mass': 1/n, 'add120_mass': 2+1/n})
        v.check_weights(receipt, oy, ey, ny, labels)
        changed = copy.deepcopy(receipt)
        changed['classes'][0]['public_per_row_weight'] = 1/3
        with self.assertRaisesRegex(ValueError, 'public_per_row_weight'):
            v.check_weights(changed, oy, ey, ny, labels)
        changed = copy.deepcopy(receipt)
        changed['classes'][0]['added_rows'] = 0
        with self.assertRaises(ValueError):
            v.check_weights(changed, oy, ey, ny, labels)

    def test_artifact_anchor_and_full_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'result.json').write_text('{"status":"COMPLETE"}')
            entries = [{'path': 'result.json', 'bytes': (root/'result.json').stat().st_size, 'sha256': v.sha(root/'result.json')}]
            (root / 'artifacts.json').write_text(json.dumps({'files': entries}))
            anchor = v.sha(root / 'artifacts.json')
            _, files = v.verify_manifest(root, anchor)
            self.assertEqual(files, {'result.json'})
            with self.assertRaisesRegex(ValueError, 'anchor'):
                v.verify_manifest(root, '0' * 64)
            (root / 'unexpected.txt').write_text('extra')
            with self.assertRaisesRegex(ValueError, 'inventory'):
                v.verify_manifest(root, anchor)


if __name__ == '__main__':
    unittest.main()
