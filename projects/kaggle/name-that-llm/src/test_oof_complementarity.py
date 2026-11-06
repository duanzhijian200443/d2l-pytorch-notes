"""Small cached-prediction fixtures; never train or read real OOF results."""
import unittest

import numpy as np
import pandas as pd

from oof_complementarity import analyze_scenario


LABELS = [f"model-{number:02d}" for number in range(18)]


def fixture():
    rows = []
    for prompt in range(10):
        for model in LABELS:
            rows.append({"id": f"p{prompt}-{model}", "prompt_id": f"p{prompt}", "model": model,
                         "fold": prompt // 2, "stylegram_prediction": model, "svc4_prediction": model,
                         "stylegram_centred": prompt % 2 == 0})
    return pd.DataFrame(rows)


class ComplementarityTests(unittest.TestCase):
    def test_four_cells_disagreements_and_ba_class_denominators(self):
        frame = fixture().iloc[:-1].copy()  # Model 17 has nine, all other labels ten.
        frame.loc[17, "svc4_prediction"] = LABELS[0]  # stylegram only, class 17
        frame.loc[0, "stylegram_prediction"] = LABELS[1]  # svc4 only, class 0
        frame.loc[2, ["stylegram_prediction", "svc4_prediction"]] = LABELS[3]  # both wrong, agree
        frame.loc[3, "stylegram_prediction"] = LABELS[4]
        frame.loc[3, "svc4_prediction"] = LABELS[5]  # both wrong, disagree
        before = frame.copy(deep=True)
        result = analyze_scenario(frame, LABELS, "full")
        summary = result["summary"]
        self.assertEqual([summary[key] for key in ("both_correct", "stylegram_only_correct", "svc4_only_correct", "both_wrong")], [175, 1, 1, 2])
        self.assertEqual(summary["disagreements"], 3)
        self.assertEqual(summary["delta_accuracy"], 0)
        self.assertAlmostEqual(summary["delta_balanced_accuracy"], (1 / 10 - 1 / 9) / 18)
        self.assertEqual(summary["stylegram_centred_rows"], 90)
        self.assertEqual(summary["stylegram_plain_rows"], 89)
        self.assertEqual((summary["svc4_only_correct_prompt_count"], summary["svc4_only_correct_class_count"]), (1, 1))
        self.assertEqual([len(result[key]) for key in ("per_class", "per_fold", "per_prompt", "per_class_fold")], [18, 5, 10, 90])
        for key in ("per_class", "per_fold", "per_prompt", "per_class_fold"):
            self.assertEqual(sum(row["support"] for row in result[key]), len(frame))
        self.assertNotIn("stylegram_balanced_accuracy", result["per_prompt"][0])
        pd.testing.assert_frame_equal(frame, before)

    def test_identical_predictions_share_each_bootstrap_prompt_draw(self):
        frame = fixture()
        mask = frame.prompt_id.isin(["p0", "p3", "p8"])
        wrong = {label: LABELS[(number + 1) % 18] for number, label in enumerate(LABELS)}
        frame.loc[mask, "stylegram_prediction"] = frame.loc[mask, "model"].map(wrong)
        frame["svc4_prediction"] = frame.stylegram_prediction
        result = analyze_scenario(frame, LABELS, "identical")
        boot = result["summary"]["paired_prompt_bootstrap"]
        self.assertEqual((boot["draws"], boot["seed"], boot["invalid_draws"]), (2000, 42, 0))
        self.assertEqual((boot["ci95_lower_delta_ba"], boot["ci95_upper_delta_ba"]), (0., 0.))

    def test_bootstrap_matches_independent_prompt_resampling_reference(self):
        frame = fixture()
        frame.loc[frame.prompt_id.eq("p0"), "stylegram_prediction"] = frame.loc[frame.prompt_id.eq("p0"), "model"].map({label: LABELS[(i + 1) % 18] for i, label in enumerate(LABELS)})
        boot = analyze_scenario(frame, LABELS, "one-prompt-errors")["summary"]["paired_prompt_bootstrap"]
        draws = np.random.default_rng(42).integers(0, 10, size=(2000, 10))
        # Every prompt has all 18 classes; each sampled copy of p0 adds 1/10 BA.
        expected = np.quantile((draws == 0).sum(axis=1) / 10, [.025, .975], method="linear")
        np.testing.assert_allclose([boot["ci95_lower_delta_ba"], boot["ci95_upper_delta_ba"]], expected)

    def test_missing_class_in_fold_is_null_ba_and_empty_class_cell_is_null_recall(self):
        frame = fixture()
        frame = frame.loc[~(frame.fold.eq(0) & frame.model.eq(LABELS[-1]))]
        result = analyze_scenario(frame, LABELS, "fold-missing-class")
        fold = result["per_fold"][0]
        self.assertEqual(fold["classes_present"], 17)
        self.assertIsNone(fold["stylegram_balanced_accuracy"])
        self.assertIsNone(fold["delta_balanced_accuracy"])
        self.assertEqual(fold["stylegram_observed_class_recall"], 1.)
        cell = next(row for row in result["per_class_fold"] if row["fold"] == 0 and row["model"] == LABELS[-1])
        self.assertEqual((cell["support"], cell["classes_present"]), (0, 0))
        self.assertIsNone(cell["delta_recall"])
        self.assertEqual(result["summary"]["stylegram_balanced_accuracy"], 1.)

    def test_missing_class_bootstrap_draws_null_the_ci_without_replacement(self):
        frame = fixture()
        frame = frame.loc[~(frame.model.eq(LABELS[-1]) & frame.prompt_id.ne("p0"))]
        boot = analyze_scenario(frame, LABELS, "rare-class")["summary"]["paired_prompt_bootstrap"]
        self.assertGreater(boot["invalid_draws"], 0)
        self.assertEqual(boot["valid_draws"] + boot["invalid_draws"], 2000)
        self.assertIsNone(boot["ci95_lower_delta_ba"])
        self.assertIsNone(boot["ci95_upper_delta_ba"])
        self.assertEqual(boot["status"], "UNDEFINED_MISSING_CLASS_DRAWS")

    def test_duplicate_id_cross_fold_and_nonboolean_route_are_rejected(self):
        for kind in ("duplicate", "cross-fold", "route"):
            frame = fixture()
            if kind == "duplicate":
                frame.loc[1, "id"] = frame.loc[0, "id"]
            elif kind == "cross-fold":
                frame.loc[0, "fold"] = 1
            else:
                frame["stylegram_centred"] = frame.stylegram_centred.astype(str)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                analyze_scenario(frame, LABELS, "invalid")

    def test_incomplete_scenario_and_incorrect_m_are_rejected(self):
        for frame in (fixture().loc[lambda d: d.model.ne(LABELS[-1])], fixture().loc[lambda d: d.fold.ne(4)]):
            with self.assertRaises(ValueError):
                analyze_scenario(frame, LABELS, "incomplete")
        with self.assertRaises(ValueError):
            analyze_scenario(fixture(), LABELS, "wrong-m", m=3)


if __name__ == "__main__":
    unittest.main()
