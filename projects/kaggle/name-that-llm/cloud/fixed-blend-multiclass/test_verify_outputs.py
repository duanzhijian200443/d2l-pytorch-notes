"""Synthetic-only rejection tests; no real run outputs or training are used."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


SPEC = importlib.util.spec_from_file_location("multiclass_output_verifier", Path(__file__).with_name("verify_outputs.py"))
V = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V)
LABELS = V.LABELS
COVERED_INDICES = np.array([LABELS.index(label) for label in V.COVERED], dtype=np.int64)
EXCLUDED_INDEX = LABELS.index("gemini-3.7-flash")


def incomplete_panel():
    # 17 classes on p0; p1 lacks the last class; p2 contains only the first.
    truth = np.r_[COVERED_INDICES, COVERED_INDICES[:-1], COVERED_INDICES[:1]]
    prompts = ["p0"] * 17 + ["p1"] * 16 + ["p2"]
    return pd.DataFrame({"id": [f"row-{i}" for i in range(len(truth))], "prompt_id": prompts,
                         "model": np.asarray(LABELS)[truth], "text": [f"synthetic {i}" for i in range(len(truth))]})


class TemporaryFiles(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="multiclass-verifier-test-")
        self.root = Path(self.temporary.name).resolve()
        self.assertEqual(self.root.parent, Path(tempfile.gettempdir()).resolve())
        self.assertTrue(self.root.name.startswith("multiclass-verifier-test-"))
        self.addCleanup(self.temporary.cleanup)

    def save_json(self, name, value):
        (self.root / name).write_text(json.dumps(value), encoding="utf-8")


class ManifestTests(TemporaryFiles):
    def test_inventory_and_unsafe_duplicate_missing_manifest_entries(self):
        self.assertEqual(len(V.STABLE), 18)
        (self.root / "one.bin").write_bytes(b"synthetic")
        entry = {"path": "one.bin", "bytes": 9, "sha256": V.sha(self.root / "one.bin")}
        self.save_json("artifacts.json", {"files": [entry]})
        V.verify_manifest(self.root, {"one.bin"})
        for entries in ([entry, entry], [], [{**entry, "path": "../one.bin"}], [{**entry, "path": "C:/one.bin"}], [{**entry, "path": "a\\one.bin"}]):
            self.save_json("artifacts.json", {"files": entries})
            with self.assertRaises(V.VerificationError):
                V.verify_manifest(self.root, {"one.bin"})

    def test_artifact_tamper_and_failure_file_are_rejected(self):
        path = self.root / "one.bin"; path.write_bytes(b"synthetic")
        self.save_json("artifacts.json", {"files": [{"path": "one.bin", "bytes": 9, "sha256": V.sha(path)}]})
        path.write_bytes(b"different")
        with self.assertRaises(V.VerificationError):
            V.verify_manifest(self.root, {"one.bin"})
        path.write_bytes(b"synthetic")
        self.save_json("failure.json", {})
        with self.assertRaises(V.VerificationError):
            V.verify_manifest(self.root, {"one.bin"})

    def test_changed_local_protocol_fails_before_loading_data(self):
        self.save_json("protocol.json", {"evaluation_accepted": True})
        with patch.object(V, "PAYLOAD", self.root), self.assertRaisesRegex(V.VerificationError, "protocol changed"):
            V.local_inputs()


class EvaluationTests(unittest.TestCase):
    def test_imbalanced_incomplete_seventeen_class_panel_is_accepted(self):
        evaluation = incomplete_panel()
        support = evaluation.model.value_counts().to_dict()
        V.validate_evaluation(evaluation, 34, 3, support)
        self.assertEqual(set(support.values()), {1, 2, 3})
        self.assertEqual(evaluation.groupby("prompt_id").size().tolist(), [17, 16, 1])

    def test_duplicate_pair_changed_support_and_excluded_truth_are_rejected(self):
        evaluation = incomplete_panel()
        support = evaluation.model.value_counts().to_dict()
        duplicate = evaluation.copy(); duplicate.loc[len(duplicate) - 1, "prompt_id"] = "p0"
        excluded = evaluation.copy(); excluded.loc[0, "model"] = LABELS[EXCLUDED_INDEX]
        changed_support = {**support, V.COVERED[0]: 4}
        for table, counts in ((duplicate, support), (excluded, support), (evaluation, changed_support)):
            with self.subTest(case=counts), self.assertRaises(V.VerificationError):
                V.validate_evaluation(table, 34, table.prompt_id.nunique(), counts)


class ScoreTests(TemporaryFiles):
    def score_fixture(self):
        evaluation = pd.DataFrame({"id": ["a", "b", "c"], "prompt_id": ["p0", "p0", "p1"],
                                   "model": LABELS[12:15], "text": ["a", "b", "c"]})
        plain = np.tile(np.arange(18, dtype=float), (3, 1))
        style = plain.copy(); style[:2] += np.linspace(-.5, .5, 18)
        svc = -plain
        groups = np.array([1, 1, 2]); mask = np.array([True, True, False])
        first = {"ids": evaluation.id.to_numpy(dtype=str), "labels": np.array(LABELS), "scores": style,
                 "plain_scores": plain, "group_ids": groups, "centred_mask": mask}
        second = {"ids": first["ids"], "labels": first["labels"], "stylegram_scores": style,
                  "svc4_scores": svc, "blend_scores": V.independent_blend(style, svc), "alpha": np.array(.1),
                  "group_ids": groups, "centred_mask": mask}
        return evaluation, first, second

    def save_scores(self, first, second):
        np.savez_compressed(self.root / "stylegram-scores.npz", **first)
        np.savez_compressed(self.root / "component-and-blend-scores.npz", **second)

    def test_label_order_width_nonfinite_and_blend_tamper_are_rejected(self):
        evaluation, first, second = self.score_fixture()
        self.save_scores(first, second)
        V.verify_score_files(self.root, evaluation, LABELS)
        mutations = [
            {**second, "labels": np.array(LABELS[::-1])},
            {**second, "svc4_scores": second["svc4_scores"][:, :4]},
            {**second, "blend_scores": second["blend_scores"] + .001},
            {**second, "alpha": np.array(.2)},
            {**second, "svc4_scores": np.full((3, 18), np.nan)},
        ]
        for index, mutated in enumerate(mutations):
            self.save_scores(first, mutated)
            with self.subTest(case=index), self.assertRaises(V.VerificationError):
                V.verify_score_files(self.root, evaluation, LABELS)

    def test_inferred_group_mask_and_plain_fallback_are_checked(self):
        evaluation, first, second = self.score_fixture()
        wrong_mask = np.array([False, False, False])
        self.save_scores({**first, "centred_mask": wrong_mask}, {**second, "centred_mask": wrong_mask})
        with self.assertRaisesRegex(V.VerificationError, "group size"):
            V.verify_score_files(self.root, evaluation, LABELS)
        altered_plain = first["plain_scores"].copy(); altered_plain[2, 0] += 1
        self.save_scores({**first, "plain_scores": altered_plain}, second)
        with self.assertRaisesRegex(V.VerificationError, "plain fallback"):
            V.verify_score_files(self.root, evaluation, LABELS)

    def test_near_tie_argmax_flip_within_allclose_tolerance_is_rejected(self):
        evaluation, first, second = self.score_fixture()
        near_tie = np.full(18, -1., dtype=np.float64)
        near_tie[0] = 1.
        near_tie[1] = np.nextafter(1., 0.)
        first["scores"][0] = near_tie
        second["svc4_scores"][0] = near_tie
        expected = V.independent_blend(second["stylegram_scores"], second["svc4_scores"])
        second["blend_scores"] = expected.copy()
        self.save_scores(first, second)
        V.verify_score_files(self.root, evaluation, LABELS)
        second["blend_scores"][0, 1] += 5e-13
        self.assertTrue(np.allclose(second["blend_scores"], expected, rtol=1e-12, atol=1e-12))
        self.assertEqual(expected[0].argmax(), 0)
        self.assertEqual(second["blend_scores"][0].argmax(), 1)
        self.save_scores(first, second)
        with self.assertRaisesRegex(V.VerificationError, "Blend argmax differs"):
            V.verify_score_files(self.root, evaluation, LABELS)

    def test_masking_argmax_to_seventeen_true_labels_is_rejected(self):
        evaluation = pd.DataFrame({"id": [f"id-{i}" for i in range(17)], "prompt_id": ["p"] * 17, "model": V.COVERED})
        values = np.zeros((17, 18)); values[:, EXCLUDED_INDEX] = 10
        values[np.arange(17), COVERED_INDICES] = 9
        predictions = V.prediction_indices({member: values for member in ("stylegram", "svc4", "blend")})
        groups = np.ones(17, dtype=int); mask = np.ones(17, dtype=bool)
        saved = evaluation.copy()
        for member in predictions:
            saved[member + "_prediction"] = [LABELS[EXCLUDED_INDEX]] * 17
            saved[member + "_correct"] = ["False"] * 17
        saved["stylegram_group_id"] = ["1"] * 17
        saved["stylegram_centred"] = ["True"] * 17
        truth = V.validate_prediction_frame(saved, evaluation, LABELS, predictions, groups, mask)
        self.assertEqual(V.covered_metrics(truth, predictions["blend"], LABELS)["macro_recall"], 0.)
        masked = saved.copy()
        masked["blend_prediction"] = evaluation.model
        masked["blend_correct"] = "True"
        with self.assertRaisesRegex(V.VerificationError, "18-way argmax"):
            V.validate_prediction_frame(masked, evaluation, LABELS, predictions, groups, mask)

    def test_independent_blend_has_fixed_point_one_contribution(self):
        style = np.tile(np.arange(18, dtype=float), (2, 1))
        normalized = (style - style.mean(axis=1, keepdims=True)) / style.std(axis=1, keepdims=True)
        np.testing.assert_allclose(V.independent_blend(style, -style), .8 * normalized, atol=1e-14)

    def test_population_variance_constant_rows_and_first_argmax_tie(self):
        style = np.vstack([np.arange(18, dtype=float), np.ones(18)])
        svc = style.copy()
        expected = np.vstack([(style[0] - 8.5) / np.sqrt(np.mean((style[0] - 8.5) ** 2)), np.zeros(18)])
        actual = V.independent_blend(style, svc)
        np.testing.assert_allclose(actual, expected, atol=1e-14)
        np.testing.assert_array_equal(actual.argmax(axis=1), [17, 0])
        self.assertFalse(np.allclose(actual[0], (style[0] - 8.5) / style[0].std(ddof=1)))


class WeightTests(TemporaryFiles):
    def fixture(self, kind):
        official = np.repeat(np.arange(3), [2, 3, 4])
        public = np.repeat(np.arange(3), [300, 300, 300])
        labels = ["A", "B", "C"]; target = np.r_[official, public]
        counts = np.bincount(target); original = np.bincount(official)
        if kind == "stylegram":
            weights = np.r_[np.full(len(official), .5), .5 * original[public] / 300]
            cw = np.ones(3)
            extra = {"share": .5, "semantics": "official .5 per row; public .5 * official class count / public class count; no class_weight"}
        else:
            weights = np.r_[np.ones(len(official), dtype=np.float32), np.full(len(public), .35, dtype=np.float32)]
            cw = len(target) / (3 * counts)
            extra = {"public_row_weight": .35, "semantics": "sklearn 1.6.1 balanced from unweighted class counts, multiplied by float32 per-row weights"}
        om = np.bincount(official, weights=weights[:len(official)])
        pm = np.bincount(public, weights=weights[len(official):])
        metadata = {"labels": labels, "class_counts": counts.tolist(), "class_weights": cw.tolist(),
                    "official_sample_mass": om.tolist(), "public_sample_mass": pm.tolist(),
                    "effective_class_mass": ((om + pm) * cw).tolist(), "sample_weight_dtype": str(weights.dtype), **extra}
        np.savez_compressed(self.root / f"weights-{kind}.npz", labels=np.array(labels), target=target, weights=weights)
        self.save_json(f"weights-{kind}.json", metadata)
        return official, public, labels, metadata

    def test_original_style_and_float32_svc_weights_reconcile(self):
        for kind in ("stylegram", "svc4"):
            official, public, labels, _ = self.fixture(kind)
            V.verify_weights(self.root, kind, official, public, labels)

    def test_svc_mass_rounding_and_weighted_class_balance_are_rejected(self):
        official, public, labels, metadata = self.fixture("svc4")
        for key, value in (("public_sample_mass", [105., 105., 105.]), ("class_weights", [1., 1., 1.]), ("sample_weight_dtype", "float64")):
            self.save_json("weights-svc4.json", {**metadata, key: value})
            with self.subTest(key=key), self.assertRaises(V.VerificationError):
                V.verify_weights(self.root, "svc4", official, public, labels)


class TerminalAndMetricTests(unittest.TestCase):
    def records(self):
        protocol = {"work_budget_seconds": 3300, "platform_wall_seconds": 3600, "rss_limit_bytes": 12 * 1024**3, "labels": LABELS}
        result = {"status": "COMPLETE", "trained_svms": 3, "completed_fits": list(V.MEMBERS), "convergence_warnings": 0,
                  "protocol_sha256": V.PROTOCOL_SHA, "gpu_used": False, "models_saved": False, "calibrated_probabilities": False,
                  "sparse_scenarios_run": False, "alpha": .1, "score_columns": 18, "metric": "covered_label_macro_recall",
                  "scope": V.RESULT_SCOPE,
                  "evaluation_rows": V.EVALUATION_ROWS, "evaluation_groups": V.EVALUATION_GROUPS,
                  "evaluation_class_counts": V.CLASS_ROWS, "training_labels": LABELS, "evaluation_labels": V.COVERED,
                  "total_seconds": 10.}
        guard = {"status": "COMPLETE", "child_exit": 0, "termination_reason": None, "artifacts_verified": True,
                 "elapsed_seconds": 11., "wall_seconds": 3300, "rss_limit_bytes": 12 * 1024**3, "peak_rss_bytes": 1000}
        state = {"status": "COMPLETE", "completed_fits": list(V.MEMBERS)}
        return result, guard, state, protocol

    def test_incomplete_warning_and_over_budget_runs_are_rejected(self):
        records = self.records(); V.verify_terminal(*records)
        for position, field, value in ((0, "trained_svms", 2), (0, "convergence_warnings", 1), (1, "child_exit", 1),
                                       (1, "termination_reason", "WALL_BUDGET_EXCEEDED"), (2, "status", "RUNNING"),
                                       (0, "total_seconds", 3301.), (1, "elapsed_seconds", 3601.),
                                       (1, "peak_rss_bytes", 12 * 1024**3 + 1),
                                       (0, "evaluation_rows", -1), (0, "evaluation_groups", -1),
                                       (0, "evaluation_class_counts", {"wrong": 1}),
                                       (0, "evaluation_labels", V.COVERED[:-1]), (0, "training_labels", LABELS[::-1])):
            changed = copy.deepcopy(records); changed[position][field] = value
            with self.subTest(field=field), self.assertRaises(V.VerificationError):
                V.verify_terminal(*changed)

    def test_paired_prompt_bootstrap_matches_direct_complete_panel_reference(self):
        truth = np.tile(COVERED_INDICES, 10)
        prompts = np.repeat([f"p{i}" for i in range(10)], 17)
        reference = truth.copy(); reference[:17] = EXCLUDED_INDEX
        result = V.paired_bootstrap(truth, reference, truth, prompts, LABELS)
        choices = np.random.default_rng(42).integers(0, 10, size=(2000, 10))
        expected = np.quantile((choices == 0).sum(axis=1) / 10, [.025, .975], method="linear")
        np.testing.assert_allclose(result["ci95"], expected)
        self.assertEqual(result["invalid_draws"], 0)

    def test_macro_recall_is_not_micro_accuracy_for_imbalanced_support(self):
        evaluation = incomplete_panel()
        truth = evaluation.model.map({label: i for i, label in enumerate(LABELS)}).to_numpy()
        prediction = truth.copy(); prediction[truth == COVERED_INDICES[0]] = EXCLUDED_INDEX
        metrics = V.covered_metrics(truth, prediction, LABELS)
        self.assertEqual(metrics["covered_label_count"], 17)
        self.assertEqual(metrics["covered_labels"], V.COVERED)
        self.assertAlmostEqual(metrics["macro_recall"], 16 / 17)
        self.assertAlmostEqual(metrics["accuracy"], 31 / 34)
        self.assertNotEqual(metrics["macro_recall"], metrics["accuracy"])

    def test_missing_class_resample_nulls_interval_without_dropping_draws(self):
        evaluation = incomplete_panel()
        truth = evaluation.model.map({label: i for i, label in enumerate(LABELS)}).to_numpy()
        reference = np.full(len(truth), EXCLUDED_INDEX)
        result = V.paired_bootstrap(truth, reference, truth, evaluation.prompt_id.to_numpy(), LABELS)
        choices = np.random.default_rng(42).integers(0, 3, size=(2000, 3))
        missing_rare_class = int(np.all(choices != 0, axis=1).sum())
        self.assertGreater(missing_rare_class, 0)
        self.assertEqual(result["invalid_draws"], missing_rare_class)
        self.assertEqual(result["valid_draws"], 2000 - missing_rare_class)
        self.assertEqual(result["draws"], 2000)
        self.assertEqual(result["seed"], 42)
        self.assertIsNone(result["ci95"])


class MetricReceiptTests(TemporaryFiles):
    def fixture(self):
        evaluation = incomplete_panel()
        truth = evaluation.model.map({label: i for i, label in enumerate(LABELS)}).to_numpy()
        predictions = {member: truth.copy() for member in ("stylegram", "svc4", "blend")}
        predictions["stylegram"][truth == COVERED_INDICES[0]] = EXCLUDED_INDEX
        predictions["blend"][:] = predictions["stylegram"]
        predictions["blend"][0] = truth[0]
        scores = {}
        for member, prediction in predictions.items():
            scores[member] = np.zeros((len(truth), 18))
            scores[member][np.arange(len(truth)), prediction] = 1.
        groups = np.arange(1, len(truth) + 1); mask = np.zeros(len(truth), dtype=bool)
        saved = evaluation[["id", "prompt_id", "model"]].copy()
        for member, prediction in predictions.items():
            saved[member + "_prediction"] = np.asarray(LABELS)[prediction]
            saved[member + "_correct"] = prediction == truth
        saved["stylegram_group_id"] = groups
        saved["stylegram_centred"] = mask
        saved.to_csv(self.root / "predictions.csv", index=False)
        metrics = {member: V.covered_metrics(truth, prediction, LABELS) for member, prediction in predictions.items()}
        comparison = {"alpha": .1, "delta_macro_recall": 1 / 51, "fixed_errors": 1, "new_errors": 0,
                      "prediction_disagreements": 1,
                      "paired_prompt_bootstrap": V.paired_bootstrap(truth, predictions["stylegram"], predictions["blend"],
                                                                    evaluation.prompt_id.to_numpy(), LABELS)}
        receipt = {"scope": V.METRICS_SCOPE, "metrics": metrics, "comparison": comparison}
        self.save_json("metrics.json", receipt)
        rows = []
        for member, prediction in predictions.items():
            for actual in COVERED_INDICES:
                counts = np.bincount(prediction[truth == actual], minlength=18)
                for predicted in range(18):
                    rows.append({"member": member, "true_model": LABELS[actual],
                                 "predicted_model": LABELS[predicted], "count": int(counts[predicted])})
        pd.DataFrame(rows).to_csv(self.root / "confusion.csv", index=False)
        return evaluation, scores, groups, mask, receipt

    def test_full_17_by_18_confusion_and_original_prediction_rows_reconcile(self):
        evaluation, scores, groups, mask, receipt = self.fixture()
        metrics, comparison, per_class, errors = V.verify_metrics(self.root, receipt, evaluation, LABELS, scores, groups, mask)
        self.assertEqual(len(per_class), 51)
        self.assertEqual(len(errors), 1)
        self.assertIsNone(comparison["paired_prompt_bootstrap"]["ci95"])
        self.assertNotEqual(metrics["stylegram"]["macro_recall"], metrics["stylegram"]["accuracy"])

    def test_micro_accuracy_and_filtered_bootstrap_interval_are_rejected(self):
        evaluation, scores, groups, mask, receipt = self.fixture()
        micro = copy.deepcopy(receipt)
        micro["metrics"]["blend"]["macro_recall"] = micro["metrics"]["blend"]["accuracy"]
        filtered = copy.deepcopy(receipt)
        filtered["comparison"]["paired_prompt_bootstrap"]["ci95"] = [0., 1.]
        for changed in (micro, filtered):
            self.save_json("metrics.json", changed)
            with self.subTest(case=changed["comparison"]), self.assertRaises(V.VerificationError):
                V.verify_metrics(self.root, changed, evaluation, LABELS, scores, groups, mask)

    def test_missing_confusion_cell_and_reordered_prediction_rows_are_rejected(self):
        evaluation, scores, groups, mask, receipt = self.fixture()
        confusion = pd.read_csv(self.root / "confusion.csv")
        confusion.iloc[:-1].to_csv(self.root / "confusion.csv", index=False)
        with self.assertRaisesRegex(V.VerificationError, "918"):
            V.verify_metrics(self.root, receipt, evaluation, LABELS, scores, groups, mask)
        confusion.to_csv(self.root / "confusion.csv", index=False)
        saved = pd.read_csv(self.root / "predictions.csv")
        saved.iloc[::-1].to_csv(self.root / "predictions.csv", index=False)
        with self.assertRaisesRegex(V.VerificationError, "mapping differs"):
            V.verify_metrics(self.root, receipt, evaluation, LABELS, scores, groups, mask)


if __name__ == "__main__":
    unittest.main()
