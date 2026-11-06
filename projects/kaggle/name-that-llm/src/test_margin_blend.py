"""Synthetic numeric checks only; never load cached experiment scores."""
import unittest

import numpy as np

from margin_blend import blend_margins, normalize_margins


class MarginBlendTests(unittest.TestCase):
    def test_float64_population_normalization_and_positive_affine_invariance(self):
        scores = np.array([[1., 5., 7., 9.], [-2., 0., 5., 10.]])
        normalized = normalize_margins(scores)
        self.assertEqual(normalized.dtype, np.dtype("float64"))
        self.assertEqual(normalized.shape, scores.shape)
        np.testing.assert_allclose(normalized.mean(axis=1), 0., atol=1e-14)
        np.testing.assert_allclose(normalized.std(axis=1, ddof=0), 1., atol=1e-14)
        changed = scores * np.array([[2.75], [.03125]]) + np.array([[17.], [-4.]])
        np.testing.assert_allclose(normalize_margins(changed), normalized, rtol=1e-13, atol=1e-13)

    def test_constant_rows_and_single_column_return_zero(self):
        np.testing.assert_array_equal(normalize_margins([[0., 0., 0.], [7., 7., 7.], [-1e308, -1e308, -1e308]]), np.zeros((3, 3)))
        np.testing.assert_array_equal(normalize_margins([[0.], [3.], [-1e308]]), np.zeros((3, 1)))

    def test_extreme_finite_and_subnormal_values_do_not_overflow(self):
        maximum = np.finfo(np.float64).max
        minimum = np.nextafter(np.float64(0), np.float64(1))
        scores = np.array([[maximum, -maximum, 0.], [maximum, maximum / 2, maximum / 4], [minimum, -minimum, 0.]])
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result = normalize_margins(scores)
            mixed = blend_margins(scores, -scores, .1)
        self.assertTrue(np.isfinite(result).all() and np.isfinite(mixed).all())
        np.testing.assert_allclose(result.std(axis=1, ddof=0), 1., atol=1e-14)
        np.testing.assert_allclose(result.mean(axis=1), 0., atol=1e-14)
        np.testing.assert_allclose(mixed, .8 * result, atol=1e-14)

    def test_endpoints_preserve_argmax_of_each_member(self):
        style = np.array([[1., 4., 2.], [9., 3., -4.]])
        svc = np.array([[7., 2., 3.], [-1., 2., 5.]])
        np.testing.assert_array_equal(blend_margins(style, svc, 0).argmax(axis=1), style.argmax(axis=1))
        np.testing.assert_array_equal(blend_margins(style, svc, 1).argmax(axis=1), svc.argmax(axis=1))
        np.testing.assert_array_equal(blend_margins(style, svc, 0), normalize_margins(style))
        np.testing.assert_array_equal(blend_margins(style, svc, 1), normalize_margins(svc))

    def test_each_member_contributes_its_actual_normalized_weight(self):
        style = np.array([[1., 2., 3.]])
        svc = np.array([[0., 1., 0.]])
        style_expected = np.array([[-np.sqrt(1.5), 0., np.sqrt(1.5)]])
        svc_expected = np.array([[-1 / np.sqrt(2), np.sqrt(2), -1 / np.sqrt(2)]])
        expected = .8 * style_expected + .2 * svc_expected
        np.testing.assert_allclose(blend_margins(style, svc, .2), expected, atol=1e-14)
        np.testing.assert_allclose(blend_margins(np.ones((1, 3)), svc, .2), .2 * svc_expected, atol=1e-14)
        self.assertEqual(blend_margins(style, svc, np.array(.2)).dtype, np.dtype("float64"))

    def test_invalid_score_arrays_are_rejected(self):
        invalid = ([], [1., 2.], 1., np.empty((0, 3)), np.empty((2, 0)), np.zeros((1, 2, 3)),
                   [[np.nan, 0.]], [[np.inf, 0.]], [[-np.inf, 0.]], [[1 + 1j, 0]], [["1", "2"]])
        for scores in invalid:
            with self.subTest(scores=repr(scores)), self.assertRaises(ValueError):
                normalize_margins(scores)

    def test_shape_mismatch_and_nonfinite_unused_endpoint_member_are_rejected(self):
        for style, svc, alpha in ((np.ones((2, 3)), np.ones((3, 2)), .1),
                                  (np.ones((1, 2)), [[np.nan, 0.]], 0),
                                  ([[np.inf, 0.]], np.ones((1, 2)), 1)):
            with self.assertRaises(ValueError):
                blend_margins(style, svc, alpha)

    def test_invalid_alpha_inputs_are_rejected(self):
        for alpha in (-.1, 1.1, np.nan, np.inf, -np.inf, [.1], np.array([.1]), [[.1]], True, "0.1", .1 + 0j, None):
            with self.subTest(alpha=repr(alpha)), self.assertRaises(ValueError):
                blend_margins([[1., 2.]], [[2., 1.]], alpha)

    def test_input_arrays_are_not_mutated(self):
        style = np.array([[1, 2, 3]], dtype=np.int64)
        svc = np.array([[3., 2., 1.]], dtype=np.float32)
        before_style, before_svc = style.copy(), svc.copy()
        blend_margins(style, svc, .1)
        np.testing.assert_array_equal(style, before_style)
        np.testing.assert_array_equal(svc, before_svc)


if __name__ == "__main__":
    unittest.main()
