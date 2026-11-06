"""Reject cache routing/truth mistakes before reusing grouped predictions."""
import copy
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('complement_input_analysis', Path(__file__).with_name('analyze.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CachedRouteTests(unittest.TestCase):
    def setUp(self):
        self.truth = pd.DataFrame({'model': ['a', 'b', 'a']})
        self.plain = np.array([[2., 0.], [0., 2.], [2., 0.]])
        self.z = {'scores': np.array([[0., 2.], [3., 0.], [2., 0.]]),
                  'true': np.array([0, 1, 0]), 'group_ids': np.array([1, 1, 2]),
                  'centred_mask': np.array([True, True, False])}

    def test_centred_predictions_can_change_but_plain_must_not(self):
        sizes, mask = module.route_checks(self.z, self.truth, ['a', 'b'], self.plain)
        self.assertEqual(sizes.tolist(), [2, 2, 1])
        altered = copy.deepcopy(self.z); altered['scores'][2] = [0., 2.]
        with self.assertRaisesRegex(ValueError, 'Plain-route'):
            module.route_checks(altered, self.truth, ['a', 'b'], self.plain)

    def test_wrong_membership_mask_is_rejected(self):
        self.z['centred_mask'][2] = True
        with self.assertRaisesRegex(ValueError, 'eligible'):
            module.route_checks(self.z, self.truth, ['a', 'b'], self.plain)

    def test_shifted_truth_and_nonfinite_scores_are_rejected(self):
        altered = copy.deepcopy(self.z); altered['true'] = np.array([1, 0, 0])
        with self.assertRaisesRegex(ValueError, 'truth/order'):
            module.route_checks(altered, self.truth, ['a', 'b'], self.plain)
        altered = copy.deepcopy(self.z); altered['scores'][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Invalid style scores'):
            module.route_checks(altered, self.truth, ['a', 'b'], self.plain)

    def test_non_integer_or_out_of_range_truth_is_rejected(self):
        for values in (np.array([0., 1., 0.]), np.array([0, 2, 0]), np.array([0, -1, 0])):
            altered = copy.deepcopy(self.z); altered['true'] = values
            with self.assertRaises(ValueError):
                module.route_checks(altered, self.truth, ['a', 'b'], self.plain)


if __name__ == '__main__':
    unittest.main()
