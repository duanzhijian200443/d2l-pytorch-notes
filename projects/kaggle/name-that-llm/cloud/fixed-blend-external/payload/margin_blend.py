"""Label-free row normalization and blending of uncalibrated decision margins.

These functions neither fit calibration nor produce probabilities. Candidate
weights and selection of cached rows belong to the caller's frozen protocol.
"""
import numpy as np


__all__ = ["normalize_margins", "blend_margins"]


def _score_matrix(scores, name):
    try:
        values = np.asarray(scores)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a rectangular real numeric matrix") from error
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError(f"{name} must be a nonempty two-dimensional matrix")
    if values.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain real numeric values")
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite float64-representable values")
    return values


def normalize_margins(scores):
    """Return float64 row margins with mean zero and population std one.

    Divide each row by its maximum absolute value before centering to prevent
    overflow on large finite input. Constant rows, including all-zero rows,
    return zeros. Input values are never mutated.
    """
    values = _score_matrix(scores, "scores")
    maximum = np.max(np.abs(values), axis=1, keepdims=True)
    scaled = np.divide(values, maximum, out=np.zeros_like(values), where=maximum != 0)
    centred = scaled - scaled.mean(axis=1, keepdims=True)
    # np.std(scaled) uses the population sum of squared centered deviations.
    deviation = scaled.std(axis=1, ddof=0, keepdims=True)
    return np.divide(centred, deviation, out=np.zeros_like(centred), where=deviation != 0)


def _alpha_scalar(alpha):
    try:
        value = np.asarray(alpha)
    except (TypeError, ValueError) as error:
        raise ValueError("alpha must be one finite real scalar in [0, 1]") from error
    if value.ndim != 0 or value.dtype.kind not in "iuf":
        raise ValueError("alpha must be one finite real scalar in [0, 1]")
    number = float(value)
    if not np.isfinite(number) or not 0 <= number <= 1:
        raise ValueError("alpha must be one finite real scalar in [0, 1]")
    return number


def blend_margins(style_scores, svc_scores, alpha):
    """Blend normalized margins as (1-alpha)*style + alpha*svc.

    Both matrices must have identical nonempty two-dimensional shapes and
    finite real values, including at the endpoints alpha=0 and alpha=1.
    Alpha accepts a real numeric scalar (or zero-dimensional numeric array).
    Boolean, complex, string and nonscalar alpha inputs are rejected.
    """
    weight = _alpha_scalar(alpha)
    style = _score_matrix(style_scores, "style_scores")
    svc = _score_matrix(svc_scores, "svc_scores")
    if style.shape != svc.shape:
        raise ValueError("style_scores and svc_scores must have identical shapes")
    return (1.0 - weight) * normalize_margins(style) + weight * normalize_margins(svc)
