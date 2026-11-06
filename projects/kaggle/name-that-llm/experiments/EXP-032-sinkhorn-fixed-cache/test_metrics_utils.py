"""Synthetic statistics only; never opens the real prediction cache."""
import unittest

import numpy as np
import pandas as pd

import metrics_utils as m


def complete(groups=2):
    return pd.DataFrame([
        {"id": f"{g}-{i}", "prompt_id": f"prompt-{g}", "model": label,
         "prediction_045": label, "prediction_030": label}
        for g in range(groups) for i, label in enumerate(m.COVERED_LABELS)
    ])


class MetricsTests(unittest.TestCase):
    def test_full_18_argmax_keeps_absent_class_and_fixed_tie_order(self):
        scores = np.zeros((2, 18), dtype=np.float64)
        scores[0, m.OUTPUT_LABELS.index("gemini-3.7-flash")] = 1
        self.assertEqual(m.prediction_labels(scores).tolist(), ["gemini-3.7-flash", m.OUTPUT_LABELS[0]])

    def test_score_shape_and_nonfinite_rejected(self):
        for scores in [np.zeros((2, 17)), np.zeros((0, 18)), np.zeros(18), np.full((2, 18), np.nan)]:
            with self.subTest(shape=scores.shape), self.assertRaises(ValueError):
                m.prediction_labels(scores)

    def test_four_cells_fixed_new_errors_and_null_absent_class(self):
        frame = complete(1)
        frame.loc[0, "prediction_045"] = m.COVERED_LABELS[1]
        frame.loc[1, "prediction_030"] = m.COVERED_LABELS[0]
        frame.loc[2, ["prediction_045", "prediction_030"]] = "gemini-3.7-flash"
        result = m.compare_predictions(frame)
        summary = result["summary"]
        self.assertEqual([summary[k] for k in ["both_correct", "only_045_correct", "only_030_correct", "both_wrong"]], [14, 1, 1, 1])
        self.assertEqual((summary["fixed_old_errors"], summary["new_errors"]), (1, 1))
        self.assertEqual(summary["delta_030_minus_045"]["macro_recall"], 0)
        self.assertEqual({x["change"] for x in result["errors"]}, {"fixed_by_030", "new_error_030", "both_wrong_same"})
        absent = next(x for x in result["per_class"] if x["model"] == "gemini-3.7-flash")
        self.assertEqual(absent["support"], 0)
        self.assertIsNone(absent["recall_030"])
        self.assertEqual(absent["predicted_030"], 1)
        self.assertNotIn("macro_recall", result["per_prompt"][0])

    def test_macro_and_accuracy_use_different_denominators(self):
        frame = complete(1)
        extras = pd.DataFrame([{"id": f"extra-{i}", "prompt_id": f"extra-prompt-{i}",
                                "model": m.COVERED_LABELS[0], "prediction_045": m.COVERED_LABELS[0],
                                "prediction_030": m.COVERED_LABELS[1]} for i in range(9)])
        frame.loc[0, "prediction_030"] = m.COVERED_LABELS[1]
        frame = pd.concat([frame, extras], ignore_index=True)
        result = m.compare_predictions(frame)["summary"]
        self.assertAlmostEqual(result["delta_030_minus_045"]["macro_recall"], -1 / 17)
        self.assertAlmostEqual(result["delta_030_minus_045"]["accuracy"], -10 / 26)

    def test_paired_bootstrap_matches_shared_prompt_draws_and_row_reordering(self):
        frame = complete(2)
        frame.loc[frame.prompt_id.eq("prompt-0"), "prediction_045"] = "gemini-3.7-flash"
        frame.loc[frame.prompt_id.eq("prompt-1"), "prediction_030"] = "gemini-3.7-flash"
        observed = m.compare_predictions(frame)["summary"]["paired_prompt_bootstrap"]
        choices = np.random.default_rng(42).integers(0, 2, size=(2000, 2))
        differences = np.where(choices == 0, 1.0, -1.0).mean(1)
        self.assertEqual(observed["ci95"], np.quantile(differences, [.025, .975], method="linear").tolist())
        self.assertEqual(observed["invalid_draws"], 0)
        reordered = m.compare_predictions(frame.sample(frac=1, random_state=9))["summary"]["paired_prompt_bootstrap"]
        self.assertEqual(observed, reordered)

    def test_missing_class_draws_null_without_replacement(self):
        frame = complete(1)
        frame["prompt_id"] = ["left" if i < 8 else "right" for i in range(len(frame))]
        result = m.compare_predictions(frame)["summary"]["paired_prompt_bootstrap"]
        choices = np.random.default_rng(42).integers(0, 2, size=(2000, 2))
        invalid = int((choices[:, 0] == choices[:, 1]).sum())
        self.assertEqual(result["invalid_draws"], invalid)
        self.assertEqual(result["valid_draws"], 2000 - invalid)
        self.assertIsNone(result["ci95"])

    def test_identity_truth_and_label_fail_closed(self):
        variations = []
        frame = complete(); frame.loc[1, "id"] = frame.loc[0, "id"]; variations.append(frame)
        frame = complete(); frame.loc[17, "prompt_id"] = frame.loc[0, "prompt_id"]; variations.append(frame)
        frame = complete(); frame.loc[0, "prediction_030"] = "unknown"; variations.append(frame)
        frame = complete(); frame = frame[frame.model.ne(m.COVERED_LABELS[0])]; variations.append(frame)
        frame = complete(); frame.loc[0, "model"] = "gemini-3.7-flash"; variations.append(frame)
        frame = complete(); frame.loc[0, "prompt_id"] = ""; variations.append(frame)
        for index, frame in enumerate(variations):
            with self.subTest(case=index), self.assertRaises(ValueError):
                m.compare_predictions(frame)


if __name__ == "__main__":
    unittest.main()
