"""Reusable checks for prompt-group-v1. No training or cloud side effects."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_fold_table(train: pd.DataFrame, folds: pd.DataFrame, labels: list[str]) -> None:
    require(train.id.is_unique and folds.id.is_unique, "Duplicate sample IDs")
    require(train.id.tolist() == folds.id.tolist(), "Fold sample order differs from train")
    require(len(labels) == len(set(labels)), "Duplicate labels")
    require(set(train.model) == set(labels), "Unknown or missing labels")
    require(not folds.isna().any().any(), "Missing fold metadata")
    for column in ("prompt_id", "model"):
        require(train[column].tolist() == folds[column].tolist(), f"Mismatched {column}")
    require(train.text.map(text_sha256).tolist() == folds.text_sha256.tolist(), "Changed text")
    require(folds.groupby("prompt_id").fold.nunique().eq(1).all(), "Prompt crosses folds")
    require(set(folds.fold) == set(range(5)), "Expected folds 0..4")
    for fold in range(5):
        for selected in (folds.fold.eq(fold), folds.fold.ne(fold)):
            require(set(folds.loc[selected, "model"]) == set(labels), "Fold lacks classes")


def validate_score_cache(
    train: pd.DataFrame, folds: pd.DataFrame, predictions: pd.DataFrame,
    scores: np.ndarray, labels: list[str], score_labels: list[str],
) -> None:
    validate_fold_table(train, folds, labels)
    require(predictions.id.is_unique, "Duplicate prediction IDs")
    require(predictions.id.tolist() == train.id.tolist(), "Prediction row order mismatch")
    require(score_labels == labels, "Decision-score label order mismatch")
    require(scores.shape == (len(train), len(labels)), "Score shape mismatch")
    require(np.isfinite(scores).all(), "Non-finite scores")
    for column in ("prompt_id", "model", "fold"):
        require(predictions[column].tolist() == folds[column].tolist(), f"OOF {column} mismatch")
    expected = np.asarray(labels)[scores.argmax(axis=1)].tolist()
    require(expected == predictions.predicted_model.tolist(), "Scores disagree with predictions")
    if "correct" in predictions:
        require(predictions.correct.tolist() == predictions.model.eq(predictions.predicted_model).tolist(),
                "Incorrect stored correctness flags")


def load_split(project: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    directory = project / "validation/prompt-group-v1"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    require(sha256(project / "train.csv") == manifest["train_sha256"], "Train file hash changed")
    require(sha256(directory / "folds.csv") == manifest["folds_sha256"], "Fold file hash changed")
    train = pd.read_csv(project / "train.csv", dtype={"id": str, "prompt_id": str})
    folds = pd.read_csv(directory / "folds.csv", dtype={"id": str, "prompt_id": str})
    validate_fold_table(train, folds, manifest["labels"])
    return train, folds, manifest["labels"]


def select_sparse_rows(folds: pd.DataFrame, per_prompt: int, repeat: int = 0, seed: int = 42) -> pd.DataFrame:
    """Select BEFORE group-dependent inference. Never uses ground-truth labels.

    Select within a validation fold in production. Returning a cross-fold table
    here is a selection manifest, not permission to mix folds during inference.
    """
    require(per_prompt > 0 and repeat >= 0, "Invalid sparse sampling parameters")
    require(folds.id.is_unique, "Duplicate sample IDs")
    ordered = folds.copy()
    ordered["_selection_key"] = ordered.id.map(
        lambda sample: text_sha256(f"{seed}:{repeat}:{sample}"))
    selected = ordered.sort_values(["prompt_id", "_selection_key"]).groupby("prompt_id", sort=True).head(per_prompt)
    return selected.drop(columns="_selection_key").sort_values("id").reset_index(drop=True)
