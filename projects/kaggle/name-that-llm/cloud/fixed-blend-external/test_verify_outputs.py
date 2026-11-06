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


SPEC = importlib.util.spec_from_file_location("external_output_verifier", Path(__file__).with_name("verify_outputs.py"))
V = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V)
LABELS = [f"label-{index:02d}" for index in range(18)]


class TemporaryFiles(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="external-verifier-test-")
        self.root = Path(self.temporary.name).resolve()
        self.assertEqual(self.root.parent, Path(tempfile.gettempdir()).resolve())
        self.assertTrue(self.root.name.startswith("external-verifier-test-"))
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

    def test_masking_argmax_to_four_labels_is_rejected(self):
        evaluation = pd.DataFrame({"id": ["one", "two", "three", "four"], "prompt_id": ["p"] * 4, "model": LABELS[12:16]})
        values = np.zeros((4, 18)); values[:, 0] = 10
        values[np.arange(4), np.arange(12, 16)] = 9
        predictions = V.prediction_indices({member: values for member in ("stylegram", "svc4", "blend")})
        groups = np.ones(4, dtype=int); mask = np.ones(4, dtype=bool)
        saved = evaluation.copy()
        for member in predictions:
            saved[member + "_prediction"] = [LABELS[0]] * 4
            saved[member + "_correct"] = ["False"] * 4
        saved["stylegram_group_id"] = ["1"] * 4
        saved["stylegram_centred"] = ["True"] * 4
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
                  "scope": "fixed alpha=.10 full-data refit evaluated once on an accepted external four-label panel",
                  "evaluation_rows": 5200, "evaluation_groups": 1300, "training_labels": LABELS, "evaluation_labels": V.COVERED,
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
                                       (1, "peak_rss_bytes", 12 * 1024**3 + 1)):
            changed = copy.deepcopy(records); changed[position][field] = value
            with self.subTest(field=field), self.assertRaises(V.VerificationError):
                V.verify_terminal(*changed)

    def test_paired_prompt_bootstrap_matches_direct_complete_panel_reference(self):
        truth = np.tile(np.arange(12, 16), 10)
        prompts = np.repeat([f"p{i}" for i in range(10)], 4)
        reference = truth.copy(); reference[:4] = 0
        result = V.paired_bootstrap(truth, reference, truth, prompts, LABELS)
        choices = np.random.default_rng(42).integers(0, 10, size=(2000, 10))
        expected = np.quantile((choices == 0).sum(axis=1) / 10, [.025, .975], method="linear")
        np.testing.assert_allclose(result["ci95"], expected)
        self.assertEqual(result["invalid_draws"], 0)


if __name__ == "__main__":
    unittest.main()
