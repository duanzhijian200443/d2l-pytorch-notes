"""Fixed .45/.30 Sinkhorn comparison from already verified cached probabilities.

The arithmetic follows EXP-009 ``infer.py`` lines 592-606 and 640-672. Only
official-training labels/texts and unlabelled test texts are inputs; there is no
evaluation-label, prompt, estimator, model, file-I/O, or parameter-search path.
The caller must verify cache coverage, row/column identities, and the provenance
of the official training data before calling this module. Arithmetic alone does
not establish those facts or any improvement on an evaluation panel.
"""
from __future__ import annotations

import re

import numpy as np
from scipy.special import softmax


SOURCE_PATH = "experiments/EXP-009-baseline-data-lineage/infer.py"
SOURCE_SHA256 = "022cafd2e6504dd7cf38c8675b9597fb4fbd4c9dd0be69d38918293d9caac9b3"
LABELS = (
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "glm-5.3", "glm-5.3-flash", "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
    "gpt-6-astra", "kimi-k3", "qwen3.8-max",
)
CONTROL_WEIGHT = 0.45
CANDIDATE_WEIGHT = 0.30
TARGET_MIX = 0.35
ITERATIONS = 100
TYPOGRAPHY_STRENGTH = 1.5
TYPOGRAPHY_CAP = 0.30
MARKS = (
    " — ", "—", "–", "’", "‘", "“", "”", "«", "»", '"', ";", ":", "(", ")",
    "[", "]", "**", "`", "…", "•", "\n", "\n\n", "e.g.", "i.e.", "However,", "Overall,",
    "In conclusion", "Therefore,", "Moreover,", "\t", " - ", " = ", "\n- ", "\n1.", "###", "```", "> ",
)


class SinkhornInputError(ValueError):
    """Invalid probability, training-label, text, or class-order input."""


def _require(condition, message):
    if not condition:
        raise SinkhornInputError(message)


def _texts(values, name):
    _require(not isinstance(values, (str, bytes)), name + " must be a sequence of strings")
    try:
        values = tuple(values)
    except TypeError as exc:
        raise SinkhornInputError(name + " must be a sequence of strings") from exc
    _require(all(isinstance(value, str) for value in values), name + " contains a non-string")
    return values


def _training_indices(values, rows):
    try:
        y = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise SinkhornInputError("train_labels must be a one-dimensional sequence") from exc
    _require(y.ndim == 1 and len(y) == rows and rows > 0,
             "train_labels and train_texts must have matching nonempty rows")
    if y.dtype.kind in "iu":
        _require(np.all((y >= 0) & (y < len(LABELS))), "train_labels has an invalid class index")
        return y.astype(np.int64, copy=True)
    _require(y.dtype.kind in "UO" and all(isinstance(value, str) for value in y),
             "train_labels must be integer class indices or exact label strings")
    lookup = {label: index for index, label in enumerate(LABELS)}
    _require(all(value in lookup for value in y), "train_labels has an unknown class label")
    return np.array([lookup[value] for value in y], dtype=np.int64)


def _probabilities(values, rows):
    try:
        p = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise SinkhornInputError("pre_sinkhorn must be a floating probability matrix") from exc
    _require(p.ndim == 2 and p.shape == (rows, len(LABELS)) and rows > 0,
             "pre_sinkhorn must have shape (nonzero test rows, 18)")
    _require(p.dtype in (np.dtype("float32"), np.dtype("float64")),
             "pre_sinkhorn must retain its original float32 or float64 dtype")
    _require(np.isfinite(p).all(), "pre_sinkhorn contains a nonfinite value")
    _require(np.all((p >= 0) & (p <= 1)), "pre_sinkhorn is outside probability bounds")
    tolerance = 32 * np.finfo(p.dtype).eps
    _require(np.allclose(p.sum(axis=1), 1.0, rtol=0.0, atol=tolerance),
             "pre_sinkhorn rows must already sum to one; logits are not accepted")
    # Preserve the input's memory order for reductions and its dtype. The source
    # subsequently uses P.copy() (C order) for the iteratively balanced matrix.
    return np.array(p, copy=True, order="K")


def _balance(p, y):
    q = p.copy()
    if len(p) < 18:
        return q, None, None, None
    rho = float(np.clip(TARGET_MIX, 0.0, 1.0))
    predicted = np.clip(p.mean(0), 1e-6, None)
    predicted /= predicted.sum()
    prior = np.bincount(y, minlength=18).astype(np.float64) + 1.0
    prior /= prior.sum()
    target = (1.0 - rho) * predicted + rho * prior
    target = np.clip(target, 1e-6, None)
    target /= target.sum()
    for _ in range(ITERATIONS):
        q *= (len(q) * target[None, :]) / np.clip(q.sum(0, keepdims=True), 1e-12, None)
        q /= np.clip(q.sum(1, keepdims=True), 1e-12, None)
    return q, predicted, prior, target


def _typography_statistics(y, train_texts, test_texts):
    xtr = np.array([[float(mark in text) for mark in MARKS] for text in train_texts], dtype=np.float32)
    xte = np.array([[float(mark in text) for mark in MARKS] for text in test_texts], dtype=np.float32)
    frequency = np.bincount(y, minlength=len(LABELS)).astype(np.float64)
    hits = np.stack([xtr[y == k].sum(axis=0) for k in range(len(LABELS))], axis=0).astype(np.float64)
    global_rate = (xtr.sum(axis=0) + 2.0) / (len(xtr) + 4.0)
    kappa = 24.0
    rate = (hits + kappa * global_rate[None, :]) / (frequency[:, None] + kappa)
    rate = np.clip(rate, 0.01, 0.99)
    global_rate = np.clip(global_rate, 0.01, 0.99)
    log_rate = np.log(rate)
    log_not_rate = np.log1p(-rate)
    base_log_rate = np.log(global_rate)
    base_log_not_rate = np.log1p(-global_rate)
    signal = np.sqrt(np.mean((rate - global_rate[None, :]) ** 2, axis=0))
    fw = signal / max(float(np.median(signal[signal > 0])) if np.any(signal > 0) else 1.0, 1e-6)
    fw = np.clip(fw, 0.25, 2.0)
    evidence = (xte * fw[None, :]) @ log_rate.T + ((1.0 - xte) * fw[None, :]) @ log_not_rate.T
    evidence -= (((xte * fw[None, :]) @ base_log_rate) + (((1.0 - xte) * fw[None, :]) @ base_log_not_rate))[:, None]
    evidence -= evidence.mean(axis=1, keepdims=True)
    evidence = np.clip(TYPOGRAPHY_STRENGTH * evidence / max(float(fw.sum()), 1e-6),
                       -TYPOGRAPHY_CAP, TYPOGRAPHY_CAP)
    word_count = np.array([len(re.findall(r"\b\w+\b", str(text))) for text in test_texts], dtype=np.float32)
    length_gate = np.clip(np.sqrt(np.maximum(word_count, 1.0) / 80.0), 0.40, 1.0)
    active_markers = xte.sum(axis=1)
    coverage_gate = np.clip(np.sqrt((active_markers + 1.0) / 4.0), 0.45, 1.0)
    return evidence, length_gate, coverage_gate


def _variant(p, q, predicted, prior, target, evidence, length_gate, coverage_gate, weight):
    applied = len(p) >= 18
    if applied:
        raw = (1 - weight) * np.log(p + 1e-12) + weight * np.log(q + 1e-12)
    else:
        raw = np.log(p + 1e-12)
    # This is recomputed from this variant's logits, never borrowed from control.
    probabilities = softmax(raw, axis=1)
    ordered = np.sort(probabilities, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    confidence = np.clip((0.60 - margin) / 0.60, 0.0, 1.0)
    gate = confidence * length_gate * coverage_gate
    final = raw + gate[:, None] * evidence
    return {
        "weight": weight, "sinkhorn_applied": applied,
        "iterations": ITERATIONS if applied else 0,
        "balanced_probability": q.copy(),
        "predicted_marginal": None if predicted is None else predicted.copy(),
        "train_prior": None if prior is None else prior.copy(),
        "target_marginal": None if target is None else target.copy(),
        "pre_typography_logits": raw,
        "pre_typography_argmax": raw.argmax(axis=1),
        "typography_evidence": evidence.copy(),
        "confidence_gate": confidence,
        "length_gate": length_gate.copy(), "coverage_gate": coverage_gate.copy(),
        "final_gate": gate, "final_logits": final, "argmax": final.argmax(axis=1),
    }


def _prepare(pre_sinkhorn, train_labels, train_texts, test_texts, labels):
    try:
        class_order = tuple(labels)
    except TypeError as exc:
        raise SinkhornInputError("labels must use the fixed 18-class order") from exc
    _require(class_order == LABELS, "labels must use the fixed 18-class order")
    training = _texts(train_texts, "train_texts")
    testing = _texts(test_texts, "test_texts")
    y = _training_indices(train_labels, len(training))
    p = _probabilities(pre_sinkhorn, len(testing))
    q, predicted, prior, target = _balance(p, y)
    evidence, length_gate, coverage_gate = _typography_statistics(y, training, testing)
    return p, q, predicted, prior, target, evidence, length_gate, coverage_gate, len(y)


def _metadata(rows, training_rows):
    return {
        "schema_version": 1, "scope": "fixed_cached_probability_math",
        "reference_source": SOURCE_PATH, "reference_source_sha256": SOURCE_SHA256,
        "labels": list(LABELS), "rows": rows, "training_rows": training_rows,
        "weights": {"control": CONTROL_WEIGHT, "candidate": CANDIDATE_WEIGHT},
        "sinkhorn_target_mix": TARGET_MIX, "sinkhorn_iterations": ITERATIONS,
        "typography_strength": TYPOGRAPHY_STRENGTH, "typography_cap": TYPOGRAPHY_CAP,
        "typography_kappa": 24.0, "confidence_recomputed_per_variant": True,
        "limits": [
            "Cache coverage, row identities, column order, and official-training provenance are caller preconditions.",
            "No evaluation labels, model inference, estimator fit, parameter scan, or effectiveness claim.",
            "An imbalanced external 17-class panel is exploratory and does not establish official 18-class performance.",
        ],
    }


def compute_sinkhorn_variant(pre_sinkhorn, train_labels, train_texts, test_texts, *, variant, labels=LABELS):
    """Compute exactly one fixed variant: ``control`` (.45) or ``candidate`` (.30).

    Use this entry point to verify control against the original cache before
    computing candidate. The result contains the variant's array fields plus
    JSON-compatible ``metadata``; the other variant is never evaluated.

    Inputs and array fields have the same contract as ``compare_sinkhorn``.
    """
    _require(isinstance(variant, str) and variant in ("control", "candidate"),
             "variant must be 'control' or 'candidate'; arbitrary weights are not supported")
    prepared = _prepare(pre_sinkhorn, train_labels, train_texts, test_texts, labels)
    weight = CONTROL_WEIGHT if variant == "control" else CANDIDATE_WEIGHT
    result = _variant(*prepared[:-1], weight)
    result["metadata"] = _metadata(len(prepared[0]), prepared[-1])
    result["metadata"]["computed_variant"] = variant
    return result


def compare_sinkhorn(pre_sinkhorn, train_labels, train_texts, test_texts, *, labels=LABELS):
    """Compute both fixed variants; return ``control``, ``candidate``, ``metadata``.

    ``pre_sinkhorn`` is the unmodified, verified N-by-18 cached probability array
    in ``LABELS`` order. ``train_labels`` accepts integer indices 0..17 or the
    exact official class names, aligned to ``train_texts``. ``test_texts`` must
    retain the verified cache's row order. Callers own all identity/hash checks.

    Both variants expose balanced probabilities, marginals, pre-typography
    logits/argmax, typography evidence, all three gates and their product, and
    final logits/argmax. Outputs are independent snapshots. N < 18 follows the
    source's skip branch (identical variants, zero Sinkhorn iterations).
    For staged real-cache acceptance, use ``compute_sinkhorn_variant`` instead.
    """
    prepared = _prepare(pre_sinkhorn, train_labels, train_texts, test_texts, labels)
    variants = {name: _variant(*prepared[:-1], weight)
                for name, weight in (("control", CONTROL_WEIGHT), ("candidate", CANDIDATE_WEIGHT))}
    variants["metadata"] = _metadata(len(prepared[0]), prepared[-1])
    return variants
