"""Pure analysis of aligned cached predictions; no model fitting or inference.

The caller owns cache hashes, experiment identity and scenario row selection.
No oracle predictions, fusion predictions or prospective fusion gains are made.
"""
from numbers import Integral

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "id", "prompt_id", "model", "fold", "stylegram_prediction",
    "svc4_prediction", "stylegram_centred",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _nonnegative_integer(value, name, positive=False):
    _require(isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)), f"{name} must be an integer")
    _require(value >= (1 if positive else 0), f"{name} is outside its allowed range")
    return int(value)


def _stats(indices, y, style, svc, centred, prompts, labels, include_ba=True):
    sy, ss, sv = y[indices], style[indices], svc[indices]
    sc, vc = ss == sy, sv == sy
    counts = np.bincount(sy, minlength=len(labels))
    present = counts > 0
    style_counts = np.bincount(sy[sc], minlength=len(labels))
    svc_counts = np.bincount(sy[vc], minlength=len(labels))
    support = len(indices)
    style_observed = float(np.mean(style_counts[present] / counts[present])) if present.any() else None
    svc_observed = float(np.mean(svc_counts[present] / counts[present])) if present.any() else None
    row = {
        "support": support,
        "classes_present": int(present.sum()),
        "observed_labels": [label for label, found in zip(labels, present) if found],
        "prompt_count": int(len(set(prompts[indices]))),
        "both_correct": int(np.sum(sc & vc)),
        "stylegram_only_correct": int(np.sum(sc & ~vc)),
        "svc4_only_correct": int(np.sum(~sc & vc)),
        "both_wrong": int(np.sum(~sc & ~vc)),
        "stylegram_correct": int(sc.sum()), "svc4_correct": int(vc.sum()),
        "disagreements": int(np.sum(ss != sv)),
        "stylegram_centred_rows": int(centred[indices].sum()),
        "stylegram_plain_rows": int((~centred[indices]).sum()),
        "stylegram_accuracy": float(sc.mean()) if support else None,
        "svc4_accuracy": float(vc.mean()) if support else None,
        "delta_accuracy": float((vc.astype(float) - sc.astype(float)).mean()) if support else None,
        "stylegram_observed_class_recall": style_observed,
        "svc4_observed_class_recall": svc_observed,
        "delta_observed_class_recall": svc_observed - style_observed if present.any() else None,
    }
    if include_ba:
        complete = bool(present.all())
        row.update(stylegram_balanced_accuracy=style_observed if complete else None,
                   svc4_balanced_accuracy=svc_observed if complete else None,
                   delta_balanced_accuracy=svc_observed - style_observed if complete else None,
                   balanced_accuracy_scope="all_18_classes" if complete else "undefined_missing_classes")
    return row


def _paired_bootstrap(y, style, svc, prompts, n_classes, draws, seed):
    names = sorted(set(prompts))
    code = {name: index for index, name in enumerate(names)}
    group = np.asarray([code[name] for name in prompts], dtype=np.int64)
    counts = np.zeros((len(names), n_classes), dtype=np.int64)
    delta_correct = np.zeros_like(counts)
    np.add.at(counts, (group, y), 1)
    difference = (svc == y).astype(np.int64) - (style == y).astype(np.int64)
    np.add.at(delta_correct, (group, y), difference)
    samples = np.random.default_rng(seed).integers(0, len(names), size=(draws, len(names)))
    multiplicity = np.zeros((draws, len(names)), dtype=np.int64)
    np.add.at(multiplicity, (np.arange(draws)[:, None], samples), 1)
    denominators = multiplicity @ counts
    valid = (denominators > 0).all(axis=1)
    invalid = int((~valid).sum())
    lower = upper = None
    if invalid == 0:
        deltas = ((multiplicity @ delta_correct) / denominators).mean(axis=1)
        lower, upper = [float(value) for value in np.quantile(deltas, [.025, .975], method="linear")]
    return {
        "status": "OK" if invalid == 0 else "UNDEFINED_MISSING_CLASS_DRAWS",
        "draws": draws, "seed": seed, "sampling_unit": "prompt_id", "prompt_groups": len(names),
        "paired": True, "valid_draws": int(valid.sum()), "invalid_draws": invalid,
        "ci95_lower_delta_ba": lower, "ci95_upper_delta_ba": upper,
        "delta_definition": "svc4_minus_stylegram_balanced_accuracy",
        "class_coverage_policy": "CI is null if any draw lacks a declared class; no dropping or replacement draws",
        "conditional_on_existing_models": True, "includes_retraining_uncertainty": False,
        "multiple_comparison_adjustment": "none",
    }


def analyze_scenario(frame, labels, scenario, m=None, repeat=None, seed=42, bootstrap_draws=2000):
    """Return four-cell overlap and paired prompt statistics for one cached scenario.

    The complete input scenario must cover all 18 labels and folds 0..4. Folds
    may individually lack labels: their 18-class BA is then null. Prompt rows
    expose accuracy and explicitly named observed-class recall, never 18-class
    BA. Empty class/fold cells have zero support and null recalls.

    ``stylegram_centred`` must contain actual booleans, not strings or 0/1.
    ``m``, when provided, is checked against every prompt's observed row count.
    This function does not mutate the input, select rows, or load any model.
    """
    _require(isinstance(frame, pd.DataFrame), "frame must be a pandas DataFrame")
    _require(set(REQUIRED_COLUMNS).issubset(frame.columns), "Missing required prediction columns")
    _require(frame.columns.is_unique, "Duplicate column names")
    _require(isinstance(scenario, str) and bool(scenario.strip()), "scenario must be a nonempty string")
    labels = list(labels)
    _require(len(labels) == 18 and len(set(labels)) == 18 and all(isinstance(label, str) and label for label in labels), "Exactly 18 unique nonempty labels are required")
    _require(len(frame) > 0 and not frame[list(REQUIRED_COLUMNS)].isna().any().any(), "Empty frame or missing required value")
    for column in ("id", "prompt_id", "model", "stylegram_prediction", "svc4_prediction"):
        _require(frame[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all(), f"Invalid string column: {column}")
    _require(frame.id.is_unique, "Duplicate sample IDs")
    _require(frame.fold.map(lambda value: isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))).all(), "fold must contain integer values")
    _require(set(frame.fold) == set(range(5)), "Complete scenario must include folds 0..4")
    _require(frame.groupby("prompt_id").fold.nunique().eq(1).all(), "A prompt crosses folds")
    _require(set(frame.model) == set(labels), "Complete scenario must include all 18 true classes")
    for column in ("stylegram_prediction", "svc4_prediction"):
        _require(set(frame[column]).issubset(labels), f"Unknown predicted label: {column}")
    _require(frame.stylegram_centred.map(lambda value: isinstance(value, (bool, np.bool_))).all(), "stylegram_centred must contain actual booleans")
    seed = _nonnegative_integer(seed, "seed")
    bootstrap_draws = _nonnegative_integer(bootstrap_draws, "bootstrap_draws", positive=True)
    if m is not None:
        m = _nonnegative_integer(m, "m", positive=True)
        _require(frame.groupby("prompt_id").size().eq(m).all(), "Observed prompt sizes differ from m")
    if repeat is not None:
        repeat = _nonnegative_integer(repeat, "repeat")
    index = {label: position for position, label in enumerate(labels)}
    y = frame.model.map(index).to_numpy(dtype=np.int64)
    style = frame.stylegram_prediction.map(index).to_numpy(dtype=np.int64)
    svc = frame.svc4_prediction.map(index).to_numpy(dtype=np.int64)
    centred = frame.stylegram_centred.to_numpy(dtype=bool)
    prompts = frame.prompt_id.to_numpy(dtype=str)
    folds = frame.fold.to_numpy(dtype=np.int64)

    def stats(mask, include_ba=True):
        return _stats(np.flatnonzero(mask), y, style, svc, centred, prompts, labels, include_ba)

    summary = stats(np.ones(len(frame), dtype=bool))
    svc_only = (svc == y) & (style != y)
    summary.update(scenario=scenario, m=m, repeat=repeat, seed=seed, fold_count=5,
                   svc4_only_correct_prompt_count=int(len(set(prompts[svc_only]))),
                   svc4_only_correct_class_count=int(len(set(y[svc_only]))),
                   svc4_only_correct_classes=[label for number, label in enumerate(labels) if number in set(y[svc_only])],
                   paired_prompt_bootstrap=_paired_bootstrap(y, style, svc, prompts, len(labels), bootstrap_draws, seed))
    per_class, per_class_fold = [], []
    for number, label in enumerate(labels):
        row = stats(y == number, include_ba=False)
        row.update(model=label, stylegram_recall=row["stylegram_accuracy"], svc4_recall=row["svc4_accuracy"], delta_recall=row["delta_accuracy"])
        per_class.append(row)
        for fold in range(5):
            row = stats((y == number) & (folds == fold), include_ba=False)
            row.update(model=label, fold=fold, stylegram_recall=row["stylegram_accuracy"], svc4_recall=row["svc4_accuracy"], delta_recall=row["delta_accuracy"])
            per_class_fold.append(row)
    per_fold = [{"fold": fold, **stats(folds == fold)} for fold in range(5)]
    per_prompt = [{"prompt_id": prompt, "fold": int(folds[prompts == prompt][0]), **stats(prompts == prompt, include_ba=False)}
                  for prompt in sorted(set(prompts))]
    return {"summary": summary, "per_class": per_class, "per_fold": per_fold,
            "per_prompt": per_prompt, "per_class_fold": per_class_fold}
