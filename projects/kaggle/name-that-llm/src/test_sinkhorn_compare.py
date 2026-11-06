"""CPU synthetic arithmetic tests only; no model or effectiveness evaluation.

The independent reference executes only two numerical AST excerpts from the
hash-pinned original source. It executes no source imports, model definitions,
data loading, estimator fitting, output writes, or evaluation-label handling.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import scipy.special

from full_stack_cache import read_cache
from full_stack_observer import decode_events
import sinkhorn_compare as module
from sinkhorn_compare import (LABELS, MARKS, SOURCE_PATH, SOURCE_SHA256, SinkhornInputError,
                              compare_sinkhorn, compute_sinkhorn_variant)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "experiments/EXP-028-full-stack-cache/fixtures"
TRAIN_SHA256 = "6b9eacf979969da4d3e3154d64ff5c3b0df2b45e66ab37647447b23d36ae8991"
FIXTURE_HASHES = {
    "n60_full-r2": {
        "manifest": "cf22fc04fb6072d0c33a9b910135c96ed5706930c150a99bb7c683af18814ea0",
        "test": "a80bfe553ef6610f75160eef98d741a3456f998d25220f850fb4cdeeb4eab393",
    },
    "n3_full-r2": {
        "manifest": "d0820407b7283a5652bd821a7747b6359b0e9ff5ab51e6f9cef19b2c465fc40d",
        "test": "64d2f04c328d311741c615112c1faa153dfb9e2034a62d36ad85f54e3ed08ae2",
    },
}


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reference_code():
    source_path = ROOT / SOURCE_PATH
    if _digest(source_path) != SOURCE_SHA256:
        raise AssertionError("Original EXP-009 source changed; do not update expected results")
    source = ast.parse(source_path.read_text(encoding="utf-8"))
    combine = [node for node in source.body if isinstance(node, ast.If) and node.lineno == 573]
    typography = [node for node in source.body if isinstance(node, ast.If) and node.lineno == 635]
    if len(combine) != 1 or len(typography) != 1:
        raise AssertionError("Original numerical excerpt anchors changed")
    sinkhorn = [node for node in combine[0].body if 592 <= node.lineno <= 606]
    typo = [node for node in typography[0].body if 640 <= node.lineno <= 672]
    if len(sinkhorn) != 3 or sinkhorn[-1].end_lineno != 606 or typo[-1].end_lineno != 672:
        raise AssertionError("Original numerical excerpt boundaries changed")
    if tuple(ast.literal_eval(typo[0].value)) != MARKS:
        raise AssertionError("Original typography marker order changed")
    return tuple(compile(ast.Module(body=copy.deepcopy(nodes), type_ignores=[]), str(source_path), "exec")
                 for nodes in (sinkhorn, typo))


REFERENCE_CODE = _reference_code()


def _reference(p, y, train_texts, test_texts, weight):
    namespace = {
        "np": np, "re": re, "_v38sp": scipy.special, "LABELS": list(LABELS),
        "P": np.array(p, copy=True, order="K"), "y": np.asarray(y, dtype=np.int64),
        "train": pd.DataFrame({"text": train_texts}), "test": pd.DataFrame({"text": test_texts}),
        "a": SimpleNamespace(sinkhorn=weight, sinkhorn_target_mix=0.35),
    }
    exec(REFERENCE_CODE[0], namespace)
    namespace["_v38_before_hard"] = namespace["tot"].copy()
    exec(REFERENCE_CODE[1], namespace)
    names = {
        "balanced_probability": "Q", "predicted_marginal": "_pred_marg", "train_prior": "_train_prior",
        "target_marginal": "_target", "pre_typography_logits": "_raw", "typography_evidence": "_evidence",
        "confidence_gate": "_confidence_gate", "length_gate": "_length_gate", "coverage_gate": "_coverage_gate",
        "final_gate": "_gate", "final_logits": "_safe_logits",
    }
    result = {key: namespace.get(name) for key, name in names.items()}
    result["argmax"] = namespace["_safe_logits"].argmax(axis=1)
    result["pre_typography_argmax"] = namespace["_raw"].argmax(axis=1)
    return result


def _fixture(name):
    path = FIXTURES / name
    if _digest(path / "inputs/train.csv") != TRAIN_SHA256:
        raise AssertionError("Synthetic fixture training text changed")
    if _digest(path / "inputs/test.csv") != FIXTURE_HASHES[name]["test"]:
        raise AssertionError("Synthetic fixture test text changed")
    train = pd.read_csv(path / "inputs/train.csv")
    test = pd.read_csv(path / "inputs/test.csv")
    snapshot = read_cache(path / "cache", "main", test.id.tolist(), test.text.tolist(), LABELS,
                          manifest_sha256=FIXTURE_HASHES[name]["manifest"])
    events = dict(decode_events(snapshot))
    if events["data"]["scope"] != "synthetic_math":
        raise AssertionError("This test may only consume synthetic fixtures")
    return {
        "p": events["pre_sinkhorn"]["probability"],
        "y": train.model.map({label: i for i, label in enumerate(LABELS)}).to_numpy(dtype=np.int64),
        "label_strings": train.model.tolist(), "train": train.text.tolist(), "test": test.text.tolist(),
        "events": events,
    }


class SinkhornCompareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = {name: _fixture(name) for name in FIXTURE_HASHES}
        cls.results = {name: compare_sinkhorn(f["p"], f["y"], f["train"], f["test"])
                       for name, f in cls.fixtures.items()}

    def exact(self, actual, expected):
        if expected is None:
            self.assertIsNone(actual)
            return
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.tobytes(), expected.tobytes())

    def reference_equal(self, result, reference):
        for key, expected in reference.items():
            with self.subTest(array=key):
                self.exact(result[key], expected)

    def test_control_reproduces_hash_verified_n60_cache_exactly(self):
        f = self.fixtures["n60_full-r2"]
        control = self.results["n60_full-r2"]["control"]
        sk = f["events"]["sinkhorn_applied"]
        for result_key, cache_key in (("balanced_probability", "balanced_probability"),
                                      ("pre_typography_logits", "logits"),
                                      ("predicted_marginal", "predicted_marginal"),
                                      ("train_prior", "train_prior"), ("target_marginal", "target")):
            self.exact(control[result_key], sk[cache_key])
        typo = f["events"]["typography"]
        for result_key, cache_key in (("typography_evidence", "evidence"), ("confidence_gate", "confidence_gate"),
                                      ("length_gate", "length_gate"), ("coverage_gate", "coverage_gate"),
                                      ("final_gate", "gate"), ("final_logits", "logits_after")):
            self.exact(control[result_key], typo[cache_key])
        self.exact(control["argmax"], typo["logits_after"].argmax(axis=1))
        self.assertTrue(control["sinkhorn_applied"])
        self.assertEqual(control["iterations"], 100)

    def test_both_fixed_weights_match_independent_original_ast(self):
        for name, f in self.fixtures.items():
            for key, weight in (("control", 0.45), ("candidate", 0.30)):
                with self.subTest(case=name, weight=weight):
                    reference = _reference(f["p"], f["y"], f["train"], f["test"], weight)
                    self.reference_equal(self.results[name][key], reference)

    def test_candidate_recalculates_confidence_and_final_gate(self):
        result = self.results["n60_full-r2"]
        control, candidate = result["control"], result["candidate"]
        self.assertTrue(np.any(candidate["confidence_gate"] != control["confidence_gate"]))
        self.assertTrue(np.any(candidate["final_gate"] != control["final_gate"]))
        incorrect = candidate["pre_typography_logits"] + control["final_gate"][:, None] * candidate["typography_evidence"]
        self.assertTrue(np.any(incorrect != candidate["final_logits"]))

    def test_n3_skips_sinkhorn_and_matches_original_cache(self):
        result = self.results["n3_full-r2"]
        f = self.fixtures["n3_full-r2"]
        self.assertNotIn("sinkhorn_applied", f["events"])
        for key in ("control", "candidate"):
            self.assertFalse(result[key]["sinkhorn_applied"])
            self.assertEqual(result[key]["iterations"], 0)
            self.exact(result[key]["balanced_probability"], f["p"])
            self.exact(result[key]["final_logits"], f["events"]["typography"]["logits_after"])
        self.exact(result["control"]["final_logits"], result["candidate"]["final_logits"])

    def test_float32_and_non_c_layout_retain_source_arithmetic(self):
        f = self.fixtures["n60_full-r2"]
        for rows in (17, 18):
            for dtype in (np.float32, np.float64):
                for order in ("C", "F"):
                    p = np.array(f["p"][:rows], dtype=dtype, order=order)
                    with self.subTest(rows=rows, dtype=dtype, order=order):
                        result = compare_sinkhorn(p, f["y"], f["train"], f["test"][:rows])
                        for key, weight in (("control", 0.45), ("candidate", 0.30)):
                            self.reference_equal(result[key], _reference(p, f["y"], f["train"], f["test"][:rows], weight))
                            self.assertEqual(result[key]["balanced_probability"].dtype, p.dtype)

    def test_exact_zero_probabilities_follow_original_clipping(self):
        f = self.fixtures["n60_full-r2"]
        p = np.zeros((18, 18), dtype=np.float64)
        p[:, 0] = 1.0
        result = compare_sinkhorn(p, f["y"], f["train"], f["test"][:18])
        for key, weight in (("control", 0.45), ("candidate", 0.30)):
            self.reference_equal(result[key], _reference(p, f["y"], f["train"], f["test"][:18], weight))
            self.assertTrue(np.isfinite(result[key]["final_logits"]).all())

    def test_inputs_and_variants_do_not_share_mutable_arrays(self):
        f = self.fixtures["n60_full-r2"]
        p, y = f["p"].copy(), f["y"].copy()
        p.setflags(write=False)
        y.setflags(write=False)
        result = compare_sinkhorn(p, y, f["train"], f["test"])
        self.exact(p, f["p"])
        self.exact(y, f["y"])
        for key, array in result["control"].items():
            if isinstance(array, np.ndarray):
                self.assertFalse(np.shares_memory(array, p))
                self.assertFalse(np.shares_memory(array, result["candidate"][key]))

    def test_string_training_labels_equal_index_training_labels(self):
        f = self.fixtures["n60_full-r2"]
        result = compare_sinkhorn(f["p"], f["label_strings"], f["train"], f["test"])
        self.exact(result["candidate"]["final_logits"], self.results["n60_full-r2"]["candidate"]["final_logits"])
        metadata = json.loads(json.dumps(result["metadata"], allow_nan=False))
        self.assertEqual(metadata["reference_source_sha256"], SOURCE_SHA256)
        self.assertEqual(metadata["weights"], {"control": 0.45, "candidate": 0.30})
        self.assertEqual(metadata["sinkhorn_target_mix"], 0.35)

    def test_rejects_shapes_unsupported_dtype_and_mismatched_rows(self):
        f = self.fixtures["n60_full-r2"]
        invalid = (f["p"].ravel(), f["p"][:, :17], f["p"][:-1], f["p"].astype(np.float16),
                   f["p"].astype(np.int64), f["p"].astype(object), f["p"].astype(np.complex128))
        for p in invalid:
            with self.subTest(shape=p.shape, dtype=p.dtype), self.assertRaises(SinkhornInputError):
                compare_sinkhorn(p, f["y"], f["train"], f["test"])

    def test_rejects_nonfinite_out_of_bounds_or_unnormalized_probabilities(self):
        f = self.fixtures["n60_full-r2"]
        for value in (np.nan, np.inf, -np.inf, -0.01, 1.01, 0.5):
            p = f["p"].copy()
            p[0, 0] = value
            with self.subTest(value=value), self.assertRaises(SinkhornInputError):
                compare_sinkhorn(p, f["y"], f["train"], f["test"])

    def test_rejects_invalid_training_labels(self):
        f = self.fixtures["n60_full-r2"]
        invalid = (f["y"][:-1], f["y"][:, None], f["y"].astype(float), f["y"].astype(bool),
                   [-1] + f["y"][1:].tolist(), [18] + f["y"][1:].tolist(),
                   ["unknown"] + f["label_strings"][1:], [None] + f["label_strings"][1:])
        for y in invalid:
            with self.subTest(labels=type(y).__name__), self.assertRaises(SinkhornInputError):
                compare_sinkhorn(f["p"], y, f["train"], f["test"])

    def test_rejects_nontext_empty_rows_and_wrong_label_order(self):
        f = self.fixtures["n60_full-r2"]
        for text in ("one scalar", None, [None] + f["test"][1:], [["nested"]] + f["test"][1:]):
            with self.subTest(text=type(text).__name__), self.assertRaises(SinkhornInputError):
                compare_sinkhorn(f["p"], f["y"], f["train"], text)
        with self.assertRaises(SinkhornInputError):
            compare_sinkhorn(f["p"], [], [], f["test"])
        with self.assertRaises(SinkhornInputError):
            compare_sinkhorn(f["p"][:0], f["y"], f["train"], [])
        with self.assertRaises(SinkhornInputError):
            compare_sinkhorn(f["p"], f["y"], f["train"][:-1], f["test"])
        with self.assertRaises(SinkhornInputError):
            compare_sinkhorn(f["p"], f["y"], f["train"], f["test"], labels=LABELS[::-1])

    def test_no_parameter_sweep_or_test_truth_argument(self):
        f = self.fixtures["n60_full-r2"]
        for unsupported in ({"sinkhorn": 0.6}, {"target_mix": 0.5}, {"test_labels": [0] * len(f["test"])}):
            with self.assertRaises(TypeError):
                compare_sinkhorn(f["p"], f["y"], f["train"], f["test"], **unsupported)

    def test_staged_api_evaluates_only_the_requested_fixed_variant(self):
        f = self.fixtures["n60_full-r2"]
        for name, weight in (("control", 0.45), ("candidate", 0.30)):
            with patch.object(module, "_variant", wraps=module._variant) as evaluated:
                result = compute_sinkhorn_variant(f["p"], f["y"], f["train"], f["test"], variant=name)
                self.assertEqual(evaluated.call_count, 1)
                self.assertEqual(evaluated.call_args.args[-1], weight)
                self.assertEqual(result["metadata"]["computed_variant"], name)
                self.exact(result["final_logits"], self.results["n60_full-r2"][name]["final_logits"])
        for unsupported in (0.45, 0.30, "0.45", "scan", None, ["control"]):
            with self.assertRaises(SinkhornInputError):
                compute_sinkhorn_variant(f["p"], f["y"], f["train"], f["test"], variant=unsupported)


if __name__ == "__main__":
    unittest.main()
