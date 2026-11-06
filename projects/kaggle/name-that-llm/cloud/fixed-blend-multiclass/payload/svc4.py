"""CPU-only features and estimator for the main svc4 member.

Reference: ``experiments/EXP-009-baseline-data-lineage/infer.py``:
lines 187-189 define the shape/function views, 219 fixes sample-weight dtype,
and 222-230 define the four TF-IDF views and LinearSVC. The original source
SHA-256 is ``022cafd2e6504dd7cf38c8675b9597fb4fbd4c9dd0be69d38918293d9caac9b3``.
Tests extract those AST nodes without importing the full inference script.

Callers own data selection, prompt isolation, row order and fold boundaries.
Pass unmodified strings: these features do not clean, deduplicate or centre
text. Fit one feature object per fold, then reuse its matrices for the weight
candidates. Importing this module does not load data, train or contact services.

This is the main svc4, whose TF-IDF input is float32, rather than the float64
Stylegram member. LinearSVC can internally convert that matrix to float64.
Freeze and record the scikit-learn version: the interpretation of
``class_weight='balanced'`` with sample weights is version dependent.
"""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral
import re

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


REFERENCE_SOURCE_SHA256 = (
    "022cafd2e6504dd7cf38c8675b9597fb4fbd4c9dd0be69d38918293d9caac9b3"
)

FUNCTION_WORDS = frozenset(
    "a an the and or but if while although because as so than that which who "
    "whom whose this these those it its they them their theirs he him his she "
    "her hers we us our ours you your yours i me my mine in on at to from with "
    "without for of by over under between through into about is are was were "
    "be been being have has had do does did can could will would shall should "
    "may might must not no nor all any some each every both either neither "
    "other another such only also just very too more most less least here "
    "there when where why how what whether then now still yet even much many "
    "few little".split()
)


def shape_view(text: str) -> str:
    """Preserve the reference's Unicode word-shape substitution."""
    return re.sub(r"[^\W\d_]+", lambda match: "X" if match[0].isupper() else "x", text)


def function_view(text: str) -> str:
    """Keep reference function words and punctuation; mask content words."""
    return " ".join(
        word.lower()
        if word.lower() in FUNCTION_WORDS or not word.isalnum()
        else "CONTENT"
        for word in re.findall(r"\w+|[^\w\s]", text)
    )


def _text_list(texts: Iterable[str]) -> list[str]:
    if isinstance(texts, str):
        raise TypeError("Expected an iterable of text strings, not one string")
    values = list(texts)
    if not all(isinstance(text, str) for text in values):
        raise TypeError("Feature input must contain strings; prepare missing values upstream")
    return values


class FourViewFeatures:
    """Fit the reference vocabulary/IDF on training rows and transform new rows.

    ``vectorizers_`` and ``view_feature_counts_`` are available after fitting
    for recording the actual vocabulary sizes and feature configuration.
    """

    VIEWS = (
        ("char", "char", str, (3, 5), 250_000, 1.0),
        ("word", "word", str, (1, 3), 180_000, 0.7),
        ("shape", "char", shape_view, (2, 6), 60_000, 0.5),
        ("function", "word", function_view, (1, 3), 50_000, 0.5),
    )

    def fit_transform(self, texts: Iterable[str]) -> csr_matrix:
        values = _text_list(texts)
        vectorizers = []
        matrices = []
        feature_counts = {}
        for name, analyzer, prepare, ngrams, maximum, scale in self.VIEWS:
            options = dict(
                analyzer=analyzer,
                ngram_range=ngrams,
                lowercase=False,
                sublinear_tf=True,
                min_df=2,
                max_features=maximum,
                dtype=np.float32,
            )
            if analyzer == "word":
                options["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
            vectorizer = TfidfVectorizer(**options)
            matrix = vectorizer.fit_transform([prepare(text) for text in values]) * scale
            vectorizers.append(vectorizer)
            matrices.append(matrix)
            feature_counts[name] = matrix.shape[1]
        result = hstack(matrices, format="csr")
        self.vectorizers_ = vectorizers
        self.view_feature_counts_ = feature_counts
        self.n_features_out_ = result.shape[1]
        return result

    def transform(self, texts: Iterable[str]) -> csr_matrix:
        if not hasattr(self, "vectorizers_"):
            raise NotFittedError("Call fit_transform on this fold's training texts first")
        values = _text_list(texts)
        matrices = [
            vectorizer.transform([prepare(text) for text in values]) * scale
            for vectorizer, (_, _, prepare, _, _, scale) in zip(
                self.vectorizers_, self.VIEWS, strict=True
            )
        ]
        return hstack(matrices, format="csr")


def fit_features(texts: Iterable[str]) -> tuple[csr_matrix, FourViewFeatures]:
    """Return the training CSR matrix and its fitted four-view transformer."""
    features = FourViewFeatures()
    return features.fit_transform(texts), features


def make_classifier() -> LinearSVC:
    """Construct the reference estimator without fitting it.

    ``dual='auto'`` is explicit; it is the default in the installed 1.8.0
    environment and the recorded Stylegram 1.6.1 environment. That historical
    Stylegram environment does not prove the original hidden svc4 run version.
    """
    return LinearSVC(
        C=0.5,
        random_state=42,
        max_iter=5000,
        class_weight="balanced",
        dual="auto",
    )


def public_sample_weights(
    n_official: int, n_public: int, weight: float
) -> np.ndarray:
    """Return float32 weights for official rows followed by public rows.

    Each official row has weight 1; each public row has ``weight``. This does
    not normalise total weight or use the Stylegram per-class share formula.
    Class weighting remains the estimator's separate responsibility.
    """
    for count in (n_official, n_public):
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
            raise ValueError("Row counts must be non-negative integers")
    value = float(weight)
    if not np.isfinite(value) or value < 0:
        raise ValueError("Public sample weight must be finite and non-negative")
    if value > np.finfo(np.float32).max:
        raise ValueError("Public sample weight must be representable as float32")
    result = np.ones(n_official + n_public, dtype=np.float32)
    result[n_official:] = value
    return result
