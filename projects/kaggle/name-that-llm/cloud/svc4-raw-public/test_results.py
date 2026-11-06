"""Small synthetic rejection/metric tests; these are not real-run acceptance.

No model fitting and no analysis/verified.json are produced by these tests.
"""
from collections import Counter
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd


SPEC = importlib.util.spec_from_file_location("svc4_result_verifier", Path(__file__).with_name("verify_results.py"))
V = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V)
CANONICAL_SPEC = importlib.util.spec_from_file_location("canonical_validation_protocol", V.ROOT / "validation/protocol.py")
CANONICAL = importlib.util.module_from_spec(CANONICAL_SPEC)
CANONICAL_SPEC.loader.exec_module(CANONICAL)
TEST_VARIANTS = {"EXP-017": None, "EXP-018": .35, "EXP-019": .175, "EXP-020": .7}
TEST_PAIRS = [["EXP-018", "EXP-017"], ["EXP-019", "EXP-018"], ["EXP-020", "EXP-018"]]


class ManifestRejectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="svc4-verifier-test-")
        self.root = Path(self.temp.name).resolve()
        self.assertEqual(self.root.parent, Path(tempfile.gettempdir()).resolve())
        self.assertTrue(self.root.name.startswith("svc4-verifier-test-"))
        self.addCleanup(self.temp.cleanup)
        (self.root / "sample.bin").write_bytes(b"small synthetic artifact")
        self.entry = {"path": "sample.bin", "bytes": (self.root / "sample.bin").stat().st_size,
                      "sha256": V.digest_file(self.root / "sample.bin")}

    def manifest(self, entries):
        (self.root / "artifacts.json").write_text(json.dumps({"files": entries}), encoding="utf-8")

    def test_duplicate_unsafe_missing_and_extra_manifest_paths_are_rejected(self):
        for path in ("../sample.bin", "/sample.bin", "C:/sample.bin", "dir\\sample.bin", "./sample.bin"):
            with self.subTest(path=path):
                self.manifest([{**self.entry, "path": path}])
                with self.assertRaises(V.VerificationError):
                    V.verify_artifacts(self.root, {"sample.bin"})
        self.manifest([self.entry, self.entry])
        with self.assertRaises(V.VerificationError):
            V.verify_artifacts(self.root, {"sample.bin"})
        self.manifest([])
        with self.assertRaises(V.VerificationError):
            V.verify_artifacts(self.root, {"sample.bin"})
        self.manifest([self.entry])
        (self.root / "unlisted.bin").write_bytes(b"unexpected")
        with self.assertRaises(V.VerificationError):
            V.verify_artifacts(self.root, {"sample.bin"})

    def test_tampered_bytes_and_failure_are_rejected(self):
        self.manifest([self.entry])
        V.verify_artifacts(self.root, {"sample.bin"})
        (self.root / "sample.bin").write_bytes(b"tampered")
        with self.assertRaises(V.VerificationError):
            V.verify_artifacts(self.root, {"sample.bin"})
        (self.root / "sample.bin").write_bytes(b"small synthetic artifact")
        (self.root / "failure.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(V.VerificationError):
            V.verify_artifacts(self.root, {"sample.bin"})

    def test_duplicate_json_keys_and_nonfinite_constants_are_rejected(self):
        path = self.root / "bad.json"
        for content in ('{"status":"COMPLETE","status":"ERROR"}', '{"score":NaN}'):
            path.write_text(content, encoding="utf-8")
            with self.assertRaises(V.VerificationError):
                V.read_json(path)


class FoldAndWeightRejectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="svc4-fold-test-")
        root = Path(self.temp.name).resolve()
        self.assertEqual(root.parent, Path(tempfile.gettempdir()).resolve())
        self.assertTrue(root.name.startswith("svc4-fold-test-"))
        self.addCleanup(self.temp.cleanup)
        self.path = root / "fold.npz"
        self.feature = {"train_matrix_sha256": "a" * 64, "valid_matrix_sha256": "b" * 64}
        self.arrays = {"ids": np.array(["row-a", "row-b"]), "labels": np.array(["A", "B", "C"]),
                       "true": np.array([0, 2]), "scores": np.array([[2., 1., 0.], [0., 1., 2.]]),
                       **{key: np.array(value) for key, value in self.feature.items()}}

    def check(self, arrays):
        np.savez_compressed(self.path, **arrays)
        V.verify_fold_npz(self.path, ["row-a", "row-b"], ["A", "B", "C"], np.array([0, 2]), self.arrays["scores"], self.feature)

    def test_bad_fold_identity_nonfinite_score_and_cross_cache_mismatch_rejected(self):
        self.check(self.arrays)
        cases = []
        for key, value in (("ids", np.array(["row-a", "row-a"])), ("labels", np.array(["B", "A", "C"])),
                           ("true", np.array([0, 1])), ("train_matrix_sha256", np.array("c" * 64)),
                           ("scores", np.array([[np.nan, 1., 0.], [0., 1., 2.]])),
                           ("scores", np.array([[2.1, 1., 0.], [0., 1., 2.]]))):
            cases.append({**self.arrays, key: value})
        cases.append({key: value for key, value in self.arrays.items() if key != "true"})
        for index, arrays in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(V.VerificationError):
                self.check(arrays)

    def weight_record(self):
        counts = np.array([2, 3, 4])
        full_counts = counts + 300
        cw = full_counts.sum() / (3 * full_counts)
        public_mass = np.full(3, float(np.float32(.35)) * 300)
        return {"fold": 0, "public_row_weight": .35, "labels": ["A", "B", "C"],
                "sample_weight_dtype": "float32", "semantics": V.SEMANTICS,
                "class_counts": full_counts.tolist(), "class_weights": cw.tolist(),
                "official_sample_mass": counts.tolist(), "public_sample_mass": public_mass.tolist(),
                "effective_class_mass": ((counts + public_mass) * cw).tolist()}

    def test_weighted_balanced_semantics_and_decimal_mass_are_rejected(self):
        record = self.weight_record()
        V.verify_weights(record, 0, .35, np.array([2, 3, 4]), ["A", "B", "C"])
        for column, value in (("public_sample_mass", [105., 105., 105.]),
                              ("class_weights", [1., 1., 1.]), ("sample_weight_dtype", "float64")):
            with self.subTest(column=column), self.assertRaises(V.VerificationError):
                V.verify_weights({**record, column: value}, 0, .35, np.array([2, 3, 4]), ["A", "B", "C"])


class TerminalAndAnalysisTests(unittest.TestCase):
    def test_feature_receipt_rejects_heldout_id_public_row_and_dtype(self):
        train = pd.DataFrame({"id": [f"id{i}" for i in range(10)]})
        folds = pd.DataFrame({"fold": [i // 2 for i in range(10)]})
        info = {"fold": 0, "mode": "official-only", "training_official_ids": train.id.iloc[2:].tolist(),
                "validation_ids": train.id.iloc[:2].tolist(), "public_rows_in_feature_fit": 0,
                "training_rows": 8, "validation_rows": 2, "dtype": "float32", "features": 4, "nnz": 8,
                "csr_bytes": 100, "validation_csr_bytes": 50, "liblinear_float64_csr_estimate": 200,
                "feature_seconds": .01, "train_matrix_sha256": "a" * 64, "valid_matrix_sha256": "b" * 64,
                "views": [{"view": name, "scale": scale, "columns": 1,
                           "vocabulary_sha256": "c" * 64, "idf_sha256": "d" * 64}
                          for name, scale in (("char", 1.), ("word", .7), ("shape", .5), ("function", .5))]}
        V.verify_feature(info, 0, "official-only", train, folds)
        for key, value in (("training_official_ids", train.id.iloc[:8].tolist()),
                           ("public_rows_in_feature_fit", 5400), ("dtype", "float64")):
            with self.subTest(key=key), self.assertRaises(V.VerificationError):
                V.verify_feature({**info, key: value}, 0, "official-only", train, folds)

    def terminal_records(self):
        protocol = {"scope": "synthetic unit fixture", "work_budget_seconds": 6900, "platform_wall_seconds": 7200, "rss_limit_bytes": 1000,
                    "variants": [{"id": exp, "public_row_weight": weight} for exp, weight in TEST_VARIANTS.items()], "comparisons": TEST_PAIRS}
        result = {"status": "COMPLETE", "trained_svms": 20, "convergence_warnings": 0,
                  "gpu_used": False, "fitted_weights_saved": False, "scope": protocol["scope"],
                  "total_seconds": 1., "fold_timings": [{"fold": f, "wall_seconds": .1} for f in range(5)]}
        guard = {"status": "COMPLETE", "child_exit": 0, "termination_reason": None, "artifacts_verified": True,
                 "wall_seconds": 6900, "rss_limit_bytes": 1000, "peak_rss_bytes": 500, "elapsed_seconds": 2.}
        state = {"status": "COMPLETE", "completed": [[e, f] for e in TEST_VARIANTS for f in range(5)]}
        return result, guard, state, protocol

    def test_terminal_failure_and_missing_fit_are_rejected(self):
        records = self.terminal_records()
        V.verify_terminal(*records)
        for index, key, value in ((0, "trained_svms", 19), (0, "convergence_warnings", 1),
                                  (1, "child_exit", 1), (1, "termination_reason", "RSS_BUDGET_EXCEEDED"),
                                  (1, "artifacts_verified", False), (2, "status", "RUNNING")):
            broken = copy.deepcopy(records); broken[index][key] = value
            with self.subTest(key=key), self.assertRaises(V.VerificationError):
                V.verify_terminal(*broken)
        broken = copy.deepcopy(records); broken[2]["completed"][-1] = broken[2]["completed"][0]
        with self.assertRaises(V.VerificationError):
            V.verify_terminal(*broken)

    def test_retry_work_budget_overrun_is_rejected_even_with_complete_receipts(self):
        result, guard, state, protocol = self.terminal_records()
        protocol.update(work_budget_seconds=6780, platform_wall_seconds=7080)
        guard.update(wall_seconds=6780, elapsed_seconds=7090)
        result["total_seconds"] = 6800
        with self.assertRaisesRegex(V.VerificationError, "Runner total exceeded protocol work budget"):
            V.verify_terminal(result, guard, state, protocol)

    def test_retry_platform_budget_overrun_is_rejected_with_runner_in_budget(self):
        result, guard, state, protocol = self.terminal_records()
        protocol.update(work_budget_seconds=6780, platform_wall_seconds=7080)
        guard.update(wall_seconds=6780, elapsed_seconds=7090)
        result["total_seconds"] = 6700
        with self.assertRaisesRegex(V.VerificationError, "Guard elapsed exceeded protocol platform budget"):
            V.verify_terminal(result, guard, state, protocol)

    def test_balanced_accuracy_and_paired_bootstrap_preserve_class_denominators(self):
        # Ten prompt groups, three labels each except the last missing C.
        y = np.array([0, 1, 2] * 9 + [0, 1])
        prompts = np.array([f"p{i:02d}" for i in range(10) for _ in range(3)][:29])
        train = pd.DataFrame({"id": [f"id{i}" for i in range(29)], "prompt_id": prompts})
        folds = pd.DataFrame({"fold": np.array([i // 2 for i in range(10) for _ in range(3)])[:29]})
        ref = y.copy(); ref[-2] = 1
        scores = {"EXP-017": np.eye(3)[ref], "EXP-018": np.eye(3)[y]}
        left = V.compare_predictions(y, scores, train, folds, ["A", "B", "C"], pairs=TEST_PAIRS[:1], variant_weights=TEST_VARIANTS)
        right = V.compare_predictions(y, scores, train, folds, ["A", "B", "C"], pairs=TEST_PAIRS[:1], variant_weights=TEST_VARIANTS)
        self.assertEqual(left, right)
        result = left[0][0]
        self.assertAlmostEqual(result["delta_ba"], 1 / 30)
        self.assertNotAlmostEqual(result["delta_ba"], 1 / 29)
        self.assertEqual((result["fixed_errors"], result["new_errors"]), (1, 0))
        self.assertEqual((result["bootstrap_draws"], result["bootstrap_seed"]), (2000, 42))

    def sparse_fixture(self):
        return pd.DataFrame({"id": [f"sample-{31-i:03d}" for i in range(32)],
                             "prompt_id": [f"p{i}" for i in range(4) for _ in range(8)]})

    def selected_ids(self, rows, sizes=(2, 3, 6)):
        return {(m, repeat): rows.iloc[indices].id.tolist()
                for m, repeat, indices in V.sparse_selections(rows, sizes=sizes)}

    def test_sparse_selection_exactly_matches_canonical_id_sets_and_order(self):
        rows = self.sparse_fixture()
        actual = self.selected_ids(rows)
        self.assertEqual(len(actual), 15)
        for (m, repeat), ids in actual.items():
            expected = CANONICAL.select_sparse_rows(rows, m, repeat, seed=42)
            self.assertEqual(ids, expected.id.tolist())
            self.assertEqual(len(ids), 4 * m)
            self.assertEqual(set(Counter(rows.loc[rows.id.isin(ids), "prompt_id"]).values()), {m})

    def test_sparse_selection_is_invariant_to_input_row_order(self):
        rows = self.sparse_fixture()
        # Keep nonsequential original dataframe indices to check positional mapping.
        shuffled = rows.sample(frac=1, random_state=837)
        self.assertEqual(self.selected_ids(rows), self.selected_ids(shuffled))

    def test_sparse_selection_is_invariant_to_sizes_order(self):
        rows = self.sparse_fixture()
        self.assertEqual(self.selected_ids(rows), self.selected_ids(rows, sizes=(6, 2, 3)))

    def test_production_expected_inventory_is_exactly_82(self):
        inventory = V.expected_artifacts(TEST_VARIANTS)
        self.assertEqual(len(inventory), 82)
        self.assertEqual(sum(path.endswith(".npz") for path in inventory), 20)
        self.assertEqual(sum("weights-fold" in path for path in inventory), 20)
        self.assertEqual(sum("metrics-fold" in path for path in inventory), 20)
        self.assertEqual(sum(path.startswith("feature-") for path in inventory), 10)
        self.assertIn("EXP-020/oof.csv", inventory)
        self.assertNotIn("EXP-016/oof.csv", inventory)

    def test_retry_protocol_rejects_completed_ids_from_prior_attempt(self):
        records = self.terminal_records()
        records[2]["completed"][0] = ["EXP-013", 0]
        with self.assertRaises(V.VerificationError):
            V.verify_terminal(*records)


if __name__ == "__main__":
    unittest.main()
