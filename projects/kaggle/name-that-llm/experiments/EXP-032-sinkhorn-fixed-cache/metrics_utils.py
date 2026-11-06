"""Fixed, local descriptive statistics for EXP032; no model or file access.

The caller must first validate the accepted EXP031 cache, source hashes, all
729 IDs and the local truth sidecar. These functions do not certify that gate.
They deliberately accept smaller synthetic panels for unit tests, while still
requiring all 17 predeclared true classes for the overall macro recall.
"""
import numpy as np
import pandas as pd


OUTPUT_LABELS = (
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "glm-5.3", "glm-5.3-flash", "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
    "gpt-6-astra", "kimi-k3", "qwen3.8-max",
)
COVERED_LABELS = tuple(label for label in OUTPUT_LABELS if label != "gemini-3.7-flash")
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 42
REQUIRED_COLUMNS = ("id", "prompt_id", "model", "prediction_045", "prediction_030")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def prediction_labels(scores):
    """Argmax over every original class, including the absent true class."""
    _require(isinstance(scores, np.ndarray) and scores.ndim == 2 and scores.shape[0] > 0
             and scores.shape[1] == 18 and scores.dtype.kind == "f"
             and np.isfinite(scores).all(), "Expected nonempty finite N x 18 floating scores")
    return np.asarray(OUTPUT_LABELS)[scores.argmax(axis=1)]


def _four_cells(good_045, good_030):
    return {
        "both_correct": int(np.sum(good_045 & good_030)),
        "only_045_correct": int(np.sum(good_045 & ~good_030)),
        "only_030_correct": int(np.sum(~good_045 & good_030)),
        "both_wrong": int(np.sum(~good_045 & ~good_030)),
    }


def _paired_bootstrap(y, good_045, good_030, prompts):
    names = sorted(set(prompts))
    code = {name: i for i, name in enumerate(names)}
    group = np.asarray([code[name] for name in prompts], dtype=np.int64)
    support = np.zeros((len(names), len(COVERED_LABELS)), dtype=np.int64)
    difference = np.zeros_like(support)
    np.add.at(support, (group, y), 1)
    np.add.at(difference, (group, y), good_030.astype(np.int64) - good_045.astype(np.int64))
    choices = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(names), size=(BOOTSTRAP_DRAWS, len(names)))
    multiplicity = np.zeros((BOOTSTRAP_DRAWS, len(names)), dtype=np.int64)
    np.add.at(multiplicity, (np.arange(BOOTSTRAP_DRAWS)[:, None], choices), 1)
    denominators = multiplicity @ support
    valid = (denominators > 0).all(axis=1)
    invalid = int((~valid).sum())
    interval = None
    if invalid == 0:
        changes = (multiplicity @ difference / denominators).mean(axis=1)
        interval = np.quantile(changes, [0.025, 0.975], method="linear").tolist()
    return {
        "status": "OK" if invalid == 0 else "UNDEFINED_MISSING_CLASS_DRAWS",
        "delta_definition": "macro_recall_030_minus_045_on_fixed_17_covered_classes",
        "sampling_unit": "source_prompt_id", "prompt_groups": len(names), "paired": True,
        "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
        "valid_draws": int(valid.sum()), "invalid_draws": invalid, "ci95": interval,
        "missing_class_policy": "CI is null if any draw lacks a declared true class; no dropping or replacement draws",
        "conditional_on_existing_models_and_panel": True, "includes_retraining_uncertainty": False,
        "multiple_comparison_adjustment": "none",
    }


def compare_predictions(frame):
    """Describe fixed 0.45 vs 0.30 predictions after the caller's input gate.

    Returns summary/per_class/per_prompt/errors. Per-prompt entries use accuracy,
    never an 18-class or incomplete 17-class balanced-accuracy denominator.
    All reported differences are 0.30 minus 0.45.
    """
    _require(isinstance(frame, pd.DataFrame) and len(frame) > 0, "Expected nonempty prediction DataFrame")
    _require(set(REQUIRED_COLUMNS).issubset(frame.columns), "Missing prediction columns")
    for column in REQUIRED_COLUMNS:
        _require(frame[column].map(lambda x: isinstance(x, str) and bool(x.strip())).all(),
                 "Blank or non-string value in " + column)
    _require(frame.id.is_unique, "Duplicate prediction ID")
    _require(not frame.duplicated(["prompt_id", "model"]).any(), "Duplicate source prompt/model row")
    _require(set(frame.model) == set(COVERED_LABELS), "Overall truth must cover exactly the fixed 17 classes")
    for column in ("prediction_045", "prediction_030"):
        _require(set(frame[column]).issubset(OUTPUT_LABELS), "Unknown output label")
    truth = frame.model.to_numpy(dtype=str)
    pred_045 = frame.prediction_045.to_numpy(dtype=str)
    pred_030 = frame.prediction_030.to_numpy(dtype=str)
    good_045, good_030 = pred_045 == truth, pred_030 == truth
    code = {label: i for i, label in enumerate(COVERED_LABELS)}
    y = np.asarray([code[label] for label in truth], dtype=np.int64)
    support = np.bincount(y, minlength=17)
    correct_045 = np.bincount(y[good_045], minlength=17)
    correct_030 = np.bincount(y[good_030], minlength=17)
    macro_045, macro_030 = float(np.mean(correct_045 / support)), float(np.mean(correct_030 / support))
    accuracy_045, accuracy_030 = float(good_045.mean()), float(good_030.mean())
    prompts = frame.prompt_id.to_numpy(dtype=str)
    summary = {
        "scope": "exploratory_external_fixed_cache", "metric": "covered_17_class_macro_recall",
        "rows": len(frame), "prompt_groups": len(set(prompts)), "output_classes": 18,
        "covered_classes": 17, "covered_labels": list(COVERED_LABELS),
        "baseline_045": {"macro_recall": macro_045, "accuracy": accuracy_045, "correct": int(good_045.sum())},
        "candidate_030": {"macro_recall": macro_030, "accuracy": accuracy_030, "correct": int(good_030.sum())},
        "delta_030_minus_045": {"macro_recall": macro_030 - macro_045, "accuracy": accuracy_030 - accuracy_045},
        **_four_cells(good_045, good_030),
        "fixed_old_errors": int(np.sum(~good_045 & good_030)),
        "new_errors": int(np.sum(good_045 & ~good_030)),
        "prediction_disagreements": int(np.sum(pred_045 != pred_030)),
        "paired_prompt_bootstrap": _paired_bootstrap(y, good_045, good_030, prompts),
        "is_clean_oof": False, "full_stack_independence_certified": False,
    }
    per_class = []
    for label in OUTPUT_LABELS:
        selected = truth == label
        count = int(selected.sum())
        a, b = good_045[selected], good_030[selected]
        per_class.append({"model": label, "support": count, "covered": label in COVERED_LABELS,
                          "correct_045": int(a.sum()), "correct_030": int(b.sum()),
                          "recall_045": float(a.mean()) if count else None,
                          "recall_030": float(b.mean()) if count else None,
                          "delta_recall_030_minus_045": float(b.mean() - a.mean()) if count else None,
                          "predicted_045": int(np.sum(pred_045 == label)), "predicted_030": int(np.sum(pred_030 == label)),
                          **_four_cells(a, b)})
    per_prompt = []
    for prompt in sorted(set(prompts)):
        selected = prompts == prompt
        a, b = good_045[selected], good_030[selected]
        per_prompt.append({"prompt_id": prompt, "support": int(selected.sum()),
                           "observed_classes": int(len(set(truth[selected]))),
                           "accuracy_045": float(a.mean()), "accuracy_030": float(b.mean()),
                           "delta_accuracy_030_minus_045": float(b.mean() - a.mean()), **_four_cells(a, b)})
    errors = []
    for i, row in enumerate(frame.itertuples(index=False)):
        if good_045[i] and good_030[i]:
            continue
        change = ("fixed_by_030" if good_030[i] else "new_error_030" if good_045[i]
                  else "both_wrong_changed" if pred_045[i] != pred_030[i] else "both_wrong_same")
        errors.append({"id": row.id, "prompt_id": row.prompt_id, "model": row.model,
                       "prediction_045": pred_045[i], "prediction_030": pred_030[i], "change": change})
    return {"summary": summary, "per_class": per_class, "per_prompt": per_prompt, "errors": errors}
