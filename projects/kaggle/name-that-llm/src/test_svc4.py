"""Tiny CPU equivalence checks against selected AST nodes of frozen svc4.

Run from the project root: ``python -X utf8 -m unittest src.test_svc4``.
No real data, full inference-script import, network or GPU work is performed.
"""

from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import hstack, isspmatrix_csr
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from src.svc4 import (
    FourViewFeatures,
    REFERENCE_SOURCE_SHA256,
    fit_features,
    function_view,
    make_classifier,
    public_sample_weights,
    shape_view,
)


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "experiments/EXP-009-baseline-data-lineage/infer.py"
)
TRAIN = [
    "I AM HELPFUL. The answer is **blue**; I can explain it!",
    "I AM HELPFUL. The answer is **green**; I can explain it!",
    "I AM HELPFUL. The answer is **red**; I can explain it!",
    "First, we consider the question.\nThen we give a concise answer.",
    "First, we consider the example.\nThen we give a concise answer.",
    "## Result\nAn example: café Café ＡＢＣ 123 _value — yes.",
    "I AM HELPFUL. The answer is **yellow**; I can explain it!",
    "First, we consider the input.\nThen we give a concise answer.",
    "First, we consider the details.\nThen we give a concise answer.",
    "First, we consider the text.\nThen we give a concise answer.",
    "## Result\nAn example: café Café ＡＢＣ 456 _value — yes.",
    "## Result\nAn example: café Café ＡＢＣ 789 _value — yes.",
]
VALID = [
    "I AM HELPFUL. The answer is blue — zzvalidationtokenzz!",
    "First, we consider the question. Then we give an answer.",
    "## Result\nAn example: café ＡＢＣ _value — yes.",
]
TARGET = np.asarray([0, 0, 0, 1, 1, 2, 0, 1, 1, 1, 2, 2])


def execute_nodes(nodes, namespace):
    tree = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(tree, str(SOURCE), "exec"), namespace)


def named_assignment(nodes, name):
    matches = [
        node for node in nodes
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    ]
    if len(matches) != 1:
        raise AssertionError(f"Expected one original assignment to {name}")
    return matches[0]


def source_weight_assignment(nodes, frame):
    matches = [
        node for node in nodes
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == frame
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == "_srcw"
            for target in node.targets
        )
    ]
    if len(matches) != 1:
        raise AssertionError(f"Expected one original source-weight assignment to {frame}")
    return matches[0]


class Svc4EquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source_bytes = SOURCE.read_bytes()
        if hashlib.sha256(source_bytes).hexdigest() != REFERENCE_SOURCE_SHA256:
            raise AssertionError("Frozen svc4 reference source changed")
        tree = ast.parse(source_bytes.decode("utf-8"))
        cls.block = next(node for node in tree.body if isinstance(node, ast.If) and node.lineno == 183)
        cls.pure_nodes = [named_assignment(cls.block.body, name) for name in ("_FW", "_shape", "_func")]
        cls.feature_loop = next(node for node in cls.block.body if isinstance(node, ast.For) and node.lineno == 222)
        cls.estimator_node = named_assignment(cls.block.body, "_svc")

    def reference(self):
        namespace = {
            "_re": re,
            "np": np,
            "TfidfVectorizer": TfidfVectorizer,
            "LinearSVC": LinearSVC,
            "_fit": TRAIN,
            "test": SimpleNamespace(text=SimpleNamespace(values=np.asarray(VALID))),
            "_A": [],
            "_B": [],
        }
        execute_nodes(self.pure_nodes + [self.feature_loop, self.estimator_node], namespace)
        return namespace

    def assert_sparse_equal(self, actual, expected):
        self.assertTrue(isspmatrix_csr(actual))
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        np.testing.assert_array_equal(actual.toarray(), expected.toarray())

    def test_raw_unicode_views_match_reference(self):
        namespace = {"_re": re}
        execute_nodes(self.pure_nodes, namespace)
        for text in ["", *TRAIN, *VALID, "**ABC**\tCafé\r\n she’s abc_def 123!? "]:
            with self.subTest(text=text):
                self.assertEqual(shape_view(text), namespace["_shape"](text))
                self.assertEqual(function_view(text), namespace["_func"](text))

    def test_four_views_match_original_matrix_and_keep_idf_fixed(self):
        reference = self.reference()
        actual, features = fit_features(iter(TRAIN))
        expected = hstack(reference["_A"], format="csr")
        self.assertEqual(actual.dtype, np.float32)
        self.assert_sparse_equal(actual, expected)
        offset = 0
        for matrix in reference["_A"]:
            self.assert_sparse_equal(actual[:, offset:offset + matrix.shape[1]], matrix.tocsr())
            offset += matrix.shape[1]
        before = [(dict(v.vocabulary_), v.idf_.copy()) for v in features.vectorizers_]
        self.assert_sparse_equal(features.transform(VALID), hstack(reference["_B"], format="csr"))
        for vectorizer, (vocabulary, idf) in zip(features.vectorizers_, before, strict=True):
            self.assertEqual(vectorizer.vocabulary_, vocabulary)
            np.testing.assert_array_equal(vectorizer.idf_, idf)
        self.assertNotIn("zzvalidationtokenzz", features.vectorizers_[1].vocabulary_)
        self.assertEqual(features.n_features_out_, sum(features.view_feature_counts_.values()))

    def test_public_weights_are_per_row_reference_weights(self):
        official = source_weight_assignment(self.block.body, "_official")
        pool_block = next(node for node in self.block.body if isinstance(node, ast.If) and node.lineno == 203)
        public = copy.deepcopy(source_weight_assignment(pool_block.body, "_P"))
        self.assertEqual(ast.literal_eval(public.value), 0.35)
        # Substitute only the candidate scalar; execute the original assignments
        # and float32 conversion rather than a second implementation of weights.
        public.value = ast.Name(id="candidate_weight", ctx=ast.Load())
        conversion = named_assignment(self.block.body, "_sw")
        for weight in (0.175, 0.35, 0.70):
            with self.subTest(weight=weight):
                namespace = {
                    "np": np,
                    "_official": pd.DataFrame(index=range(6)),
                    "_P": pd.DataFrame(index=range(6)),
                    "candidate_weight": weight,
                }
                execute_nodes([official, public], namespace)
                namespace["_fitdf"] = pd.concat([namespace["_official"], namespace["_P"]], ignore_index=True)
                execute_nodes([conversion], namespace)
                actual = public_sample_weights(6, 6, weight)
                np.testing.assert_array_equal(actual, namespace["_sw"])
                self.assertEqual(actual.dtype, np.float32)
                self.assertGreater(float(actual.sum()), 6.0)
        np.testing.assert_array_equal(public_sample_weights(6, 0, 0.35), np.ones(6, dtype=np.float32))

    def test_tiny_classifier_matches_original_with_sample_weights(self):
        reference = self.reference()
        matrix, features = fit_features(TRAIN)
        weights = public_sample_weights(6, 6, 0.35)
        actual = make_classifier()
        expected = reference["_svc"]
        self.assertEqual(actual.get_params(), expected.get_params())
        self.assertEqual(actual.dual, "auto")
        with threadpool_limits(limits=1), warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            expected.fit(hstack(reference["_A"], format="csr"), TARGET, sample_weight=weights)
            actual.fit(matrix, TARGET, sample_weight=weights)
        np.testing.assert_array_equal(actual.classes_, expected.classes_)
        np.testing.assert_allclose(
            actual.decision_function(features.transform(VALID)),
            expected.decision_function(hstack(reference["_B"], format="csr")),
            rtol=0, atol=1e-12,
        )

    def test_input_and_weight_contract(self):
        with self.assertRaises(NotFittedError):
            FourViewFeatures().transform(VALID)
        for bad in ("one string", ["text", None]):
            with self.subTest(texts=bad), self.assertRaises(TypeError):
                FourViewFeatures().fit_transform(bad)
        for args in ((-1, 2, 0.35), (1.5, 2, 0.35), (True, 2, 0.35), (1, 2, -1), (1, 2, float("nan")), (1, 2, float("inf"))):
            with self.subTest(args=args), self.assertRaises(ValueError):
                public_sample_weights(*args)


if __name__ == "__main__":
    unittest.main()
