"""Independent cache acceptance. No classifier training or source-function imports.

Only writes independent-review.json next to this script. Content TF-IDF clustering
is reconstructed from held-out texts to audit inference routing, not fitted as a
classifier feature extractor. Use --parquet-python when this interpreter lacks
pyarrow; the alternate interpreter only reads the existing risk-flag parquet.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfVectorizer


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
ROOT = EXPERIMENT.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def regroup(texts):
    """Independent text-only reconstruction; accepts no labels or prompt IDs."""
    x = TfidfVectorizer(
        stop_words="english", sublinear_tf=True, max_features=100000,
        dtype=np.float32,
    ).fit_transform(texts)
    distances = np.clip(1 - (x @ x.T).toarray(), 0, 1)
    median = max(float(np.median(distances[np.triu_indices(len(texts), 1)])), 1e-6)
    distances = np.asarray(distances / median, dtype=np.float64)
    np.fill_diagonal(distances, 0)
    groups = fcluster(
        linkage(squareform(distances, checks=False), method="average"),
        0.90, criterion="distance",
    )
    sizes = np.bincount(groups)[groups]
    return groups, (sizes >= 2) & (sizes <= 18)


def ba(truth, predictions):
    support = np.bincount(truth, minlength=18)
    correct = np.bincount(truth, weights=(truth == predictions), minlength=18)
    return float((correct[support > 0] / support[support > 0]).mean())


def close(left, right):
    assert abs(float(left) - float(right)) < 1e-12, (left, right)


def public_flag_audit(python):
    code = """
from pathlib import Path
import csv, json, sys
import pyarrow.parquet as pq
root = Path(sys.argv[1])
extra = list(csv.DictReader((root / 'experiments/EXP-010-012-stylegram-share/payload/extra.csv').open(encoding='utf-8')))
ids = {row['row_id'] for row in extra}
risk = ['official_exact_match', 'official_high_similarity', 'official_shingle_review', 'label_conflict', 'identity_mismatch', 'provenance_issue', 'linked_official_prompt']
rows = pq.read_table(root / 'experiments/EXP-008-public-pool-audit/outputs/row_flags.parquet', columns=['row_id', 'text_hash'] + risk).to_pylist()
selected = {row['row_id']: row for row in rows if row['row_id'] in ids}
assert len(selected) == len(extra)
assert all(selected[row['row_id']]['text_hash'] == row['audit_text_hash'] for row in extra)
counts = {key: sum(bool(row[key]) for row in selected.values()) for key in risk}
assert not any(counts.values())
print(json.dumps({'public_rows_checked': len(selected), 'risk_counts': counts}))
"""
    result = subprocess.run(
        [str(python), "-B", "-X", "utf8", "-c", code, str(ROOT)],
        check=True, capture_output=True, text=True, encoding="utf-8",
    )
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=EXPERIMENT / "download-v2/results")
    parser.add_argument("--parquet-python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    start = time.monotonic()
    src = args.input.resolve()
    payload = EXPERIMENT / "payload"
    protocol = json.loads((payload / "protocol.json").read_text("utf-8"))
    labels = np.array(protocol["labels"])
    shares = protocol["shares"]
    label_index = {value: index for index, value in enumerate(labels)}
    hashes = {name: sha(payload / name) == expected for name, expected in protocol["input_hashes"].items()}
    assert all(hashes.values())
    assert sha(ROOT / "train.csv") == protocol["train_sha256"]
    assert sha(ROOT / "validation/prompt-group-v1/folds.csv") == protocol["input_hashes"]["folds.csv"]
    assert sha(EXPERIMENT / "runner.py") == protocol["input_hashes"]["runner.py"]
    train = pd.read_csv(ROOT / "train.csv", dtype={"id": str, "prompt_id": str})
    folds = pd.read_csv(ROOT / "validation/prompt-group-v1/folds.csv", dtype={"id": str, "prompt_id": str})
    extra = pd.read_csv(payload / "extra.csv", dtype={"row_id": str, "prompt_id": str})
    assert train.id.is_unique and folds.id.is_unique and extra.row_id.is_unique
    for column in ["id", "model", "prompt_id"]:
        assert train[column].tolist() == folds[column].tolist()
    assert train.text.map(lambda text: hashlib.sha256(text.encode()).hexdigest()).tolist() == folds.text_sha256.tolist()
    assert folds.groupby("prompt_id").fold.nunique().eq(1).all()
    assert set(folds.fold) == set(range(5))
    normalize = lambda text: " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    official_text = train.text.map(normalize)
    public_text = extra.text.map(normalize)
    assert not official_text.duplicated().any()
    assert not set(official_text) & set(public_text)
    assert extra.groupby("prompt_id").canonical_prompt.nunique().eq(1).all()
    assert extra.groupby("canonical_prompt").prompt_id.nunique().eq(1).all()
    risk_audit = public_flag_audit(args.parquet_python)
    y = train.model.map(label_index).to_numpy()
    dense_metrics = pd.read_csv(src / "fold_metrics.csv").set_index(["fold", "share"])
    sparse_metrics = pd.read_csv(src / "sparse_fold_metrics.csv").set_index(["fold", "share", "m", "repeat"])
    assert len(dense_metrics) == 15 and dense_metrics.index.is_unique
    assert len(sparse_metrics) == 225 and sparse_metrics.index.is_unique
    pooled = {(a, m, r): [] for a in shares for m in protocol["sparse_per_prompt"] for r in range(protocol["sparse_repeats"])}
    counters = {key: 0 for key in ["unique_grouping_batches", "dense_cache_blocks", "sparse_cache_blocks", "plain_route_rows_checked", "sparse_score_rows_different_from_dense", "sparse_prediction_rows_different_from_dense", "class_weight_cases"]}
    coverage = []
    for fold in range(5):
        vi = np.flatnonzero(folds.fold.to_numpy() == fold)
        valid = train.iloc[vi].reset_index(drop=True)
        truth = y[vi]
        groups, mask = regroup(valid.text.tolist())
        counters["unique_grouping_batches"] += 1
        dense = {}
        training_counts = train.loc[folds.fold.ne(fold)].model.value_counts().reindex(labels)
        public_counts = extra.model.value_counts().reindex(labels)
        for share in shares:
            total_mass = training_counts * (1 - share) + share * training_counts / public_counts * public_counts
            assert np.allclose(total_mass, training_counts)
            counters["class_weight_cases"] += 1
            with np.load(src / f"fold{fold}-share{share}.npz", allow_pickle=False) as z:
                assert z["ids"].tolist() == valid.id.tolist() and z["labels"].tolist() == labels.tolist()
                assert np.array_equal(z["true"], truth)
                assert z["scores"].shape == z["plain_scores"].shape == (len(valid), 18)
                assert np.isfinite(z["plain_scores"]).all() and np.isfinite(z["scores"]).all()
                assert np.array_equal(z["group_ids"], groups)
                assert z["centred_mask"].dtype == bool and np.array_equal(z["centred_mask"], mask)
                assert np.array_equal(z["scores"][~mask], z["plain_scores"][~mask])
                dense[share] = {"scores": z["scores"].copy(), "plain": z["plain_scores"].copy()}
            row = dense_metrics.loc[(fold, share)]
            close(row.ba, ba(truth, dense[share]["scores"].argmax(1)))
            close(row.plain_ba, ba(truth, dense[share]["plain"].argmax(1)))
            assert int(row.centred_rows) == int(mask.sum()) and int(row.rows) == len(valid)
            counters["dense_cache_blocks"] += 1
            counters["plain_route_rows_checked"] += int((~mask).sum())
        coverage.append({"fold": fold, "m": "dense", "repeat": None, "rows": len(valid), "centred_rows": int(mask.sum())})
        for m in protocol["sparse_per_prompt"]:
            for repeat in range(protocol["sparse_repeats"]):
                ordered = valid.assign(_k=valid.id.map(lambda value: hashlib.sha256(f"42:{repeat}:{value}".encode()).hexdigest()))
                selected = np.sort(ordered.sort_values(["prompt_id", "_k"]).groupby("prompt_id").head(m).index.to_numpy())
                groups, mask = regroup(valid.text.iloc[selected].tolist())
                counters["unique_grouping_batches"] += 1
                coverage.append({"fold": fold, "m": m, "repeat": repeat, "rows": len(selected), "centred_rows": int(mask.sum())})
                for share in shares:
                    with np.load(src / f"sparse-f{fold}-a{share}-m{m}-r{repeat}.npz", allow_pickle=False) as z:
                        assert z["ids"].tolist() == valid.id.iloc[selected].tolist()
                        assert np.array_equal(z["true"], truth[selected])
                        assert z["scores"].shape == (len(selected), 18) and np.isfinite(z["scores"]).all()
                        assert np.array_equal(z["group_ids"], groups)
                        assert z["centred_mask"].dtype == bool and np.array_equal(z["centred_mask"], mask)
                        assert np.array_equal(z["scores"][~mask], dense[share]["plain"][selected][~mask])
                        scores = z["scores"].copy()
                    pooled[(share, m, repeat)].append((truth[selected], scores.argmax(1)))
                    row = sparse_metrics.loc[(fold, share, m, repeat)]
                    close(row.ba, ba(truth[selected], scores.argmax(1)))
                    assert int(row.centred_rows) == int(mask.sum()) and int(row.rows) == len(selected)
                    counters["sparse_cache_blocks"] += 1
                    counters["plain_route_rows_checked"] += int((~mask).sum())
                    counters["sparse_score_rows_different_from_dense"] += int(np.any(scores != dense[share]["scores"][selected], axis=1).sum())
                    counters["sparse_prediction_rows_different_from_dense"] += int((scores.argmax(1) != dense[share]["scores"][selected].argmax(1)).sum())
    sparse = []
    for (share, m, repeat), parts in pooled.items():
        truth = np.concatenate([truth for truth, prediction in parts])
        prediction = np.concatenate([prediction for truth, prediction in parts])
        assert set(truth) == set(range(18))
        sparse.append({"share": share, "m": m, "repeat": repeat, "rows": len(truth), "balanced_accuracy": ba(truth, prediction)})
    sparse_summary = pd.DataFrame(sparse).groupby(["share", "m"]).balanced_accuracy.agg(["mean", "std", "min", "max"]).reset_index().to_dict("records")
    dense_summary, predictions = [], {}
    for share in shares:
        scores = np.load(src / f"oof-share{share}.npy", allow_pickle=False)
        prediction = scores.argmax(1)
        predictions[share] = prediction
        dense_summary.append({"share": share, "balanced_accuracy": ba(y, prediction), "correct": int((prediction == y).sum())})
    # Independent prompt/class counts, with shared draws for each paired contrast.
    groups = sorted(train.prompt_id.unique())
    group_index = pd.Categorical(train.prompt_id, categories=groups).codes
    support = np.zeros((len(groups), 18))
    np.add.at(support, (group_index, y), 1)
    counts = {}
    for share, prediction in predictions.items():
        count = np.zeros_like(support)
        np.add.at(count, (group_index, y), (prediction == y).astype(float))
        counts[share] = count
    draws = np.random.default_rng(42).integers(0, len(groups), size=(2000, len(groups)))
    denominator = support[draws].sum(axis=1)
    paired = []
    for share in [.25, .75]:
        delta = ((counts[share][draws].sum(axis=1) - counts[.5][draws].sum(axis=1)) / denominator).mean(axis=1)
        baseline, candidate = predictions[.5] == y, predictions[share] == y
        paired.append({"share": share, "delta_ba": ba(y, predictions[share]) - ba(y, predictions[.5]), "bootstrap_low": float(np.quantile(delta, .025)), "bootstrap_high": float(np.quantile(delta, .975)), "fixed_errors": int((~baseline & candidate).sum()), "introduced_errors": int((baseline & ~candidate).sum())})
    centered = {str(m): {"rows": int(part.rows.sum()), "centred_rows": int(part.centred_rows.sum())} for m, part in pd.DataFrame(coverage).groupby("m")}
    result = {
        "status": "PASS", "observed_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Independent actual-cache routing and metric audit; source functions were not imported or called",
        "classifier_trained": False, "source_result_sha256": sha(src / "result.json"),
        "review_script_sha256": sha(Path(__file__)), "input": str(src),
        "input_audit": {"payload_hashes": hashes, "official_rows": len(train), "official_prompts": train.prompt_id.nunique(), "official_normalised_duplicate_rows": 0, "train_public_normalised_overlap": 0, "public_rows": len(extra), "public_groups": extra.prompt_id.nunique(), "canonical_prompt_mapping_one_to_one": True, "public_row_flags": risk_audit},
        "counters": counters, "dense": dense_summary, "paired": paired,
        "sparse_pooled_metrics": sparse, "sparse_summary": sparse_summary,
        "centred_coverage_unique_batches": centered,
        "local_versions": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "elapsed_seconds": time.monotonic() - start,
        "findings": [],
        "limitations": [
            "No SVM weights were saved, so centred numerical inference cannot be regenerated without training; this check verifies the clustering, eligibility masks, plain-score routing and metrics.",
            "Known public prompt/text risks were checked against the prior audit; exhaustive semantic isolation is not established.",
            "Prompt bootstrap is conditional on fitted models and development folds, without candidate-selection correction; it is not training-repeat or leaderboard uncertainty.",
            "Sparse repeats share training models and prompts; their standard deviations describe scenario variation, not independent experimental replication.",
        ],
    }
    destination = HERE / "independent-review.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "counters": counters, "dense": dense_summary, "paired": paired, "output": str(destination)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
