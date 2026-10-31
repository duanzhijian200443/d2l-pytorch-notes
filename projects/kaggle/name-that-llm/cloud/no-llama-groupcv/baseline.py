# %% [markdown]
# # Name That LLM：无 Llama 的 prompt 分组基线
#
# 在 Kaggle CPU 上训练词和字符 n-gram + LinearSVC，建立可追溯的第一条基线。
# 仅使用比赛训练数据，无外部模型、额外生成数据或 API 调用。
#
# 数据与评估：[Name That LLM](https://www.kaggle.com/competitions/name-that-llm)。
# 指标为 Balanced Accuracy，即 18 个类别召回率的算术平均。
# 这份 Notebook 用于建立实验对照，不代表已复现公开方案的 0.654 分数。
#
# **验证约定**：5 折 GroupKFold，以 `prompt_id` 分组；每折独立拟合词表、
# IDF 和分类器，验证 prompt 及文本不进入该折训练。参数预先固定，不在这些折上搜索。

# %% [markdown]
# ## 1. 固定配置与运行环境

# %%
import gc
import hashlib
import json
import os
import platform
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from IPython.display import Markdown, display
from matplotlib.ticker import PercentFormatter
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

CONFIG = {
    "experiment": "no-llama-tfidf-groupcv-v1",
    "seed": 42,
    "n_splits": 5,
    "word_ngram_range": [1, 2],
    "char_ngram_range": [3, 5],
    "word_max_features": 50000,
    "char_max_features": 120000,
    "min_df": 2,
    "C": 1.0,
    "class_weight": "balanced",
    "max_iter": 12000,
    "tol": 0.0001,
}
SEED = CONFIG["seed"]
np.random.seed(SEED)
START = time.monotonic()
OUT = Path("/kaggle/working")
OUT.mkdir(parents=True, exist_ok=True)
VERSIONS = {
    "python": platform.python_version(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scikit_learn": sklearn.__version__,
}
print(json.dumps({"config": CONFIG, "versions": VERSIONS}, indent=2))

# %% [markdown]
# ## 2. 读取数据并检查边界
#
# `test.csv` 在普通运行中仅包含少量示例，正式评分会替换为隐藏测试集。
# 本轮输出文件检查不等于获得了 Public LB 分数。

# %%
DATA = next(
    p for p in [
        Path("/kaggle/input/competitions/name-that-llm"),
        Path("/kaggle/input/name-that-llm"),
    ] if (p / "train.csv").is_file()
)
train = pd.read_csv(DATA / "train.csv", dtype={"id": str, "prompt_id": str})
test = pd.read_csv(DATA / "test.csv", dtype={"id": str})
assert {"id", "prompt_id", "text", "model"} <= set(train.columns)
assert {"id", "text"} <= set(test.columns)
assert not train[["id", "prompt_id", "text", "model"]].isna().any().any()
assert not test[["id", "text"]].isna().any().any()
assert train.id.is_unique and test.id.is_unique
assert not set(train.id).intersection(test.id)
assert len(train) > 0 and len(test) > 0
assert train.text.str.strip().str.len().gt(0).all()
assert test.text.str.strip().str.len().gt(0).all()
LABELS = sorted(train.model.unique().tolist())
assert len(LABELS) == 18, f"Unexpected class count: {len(LABELS)}"
assert train.prompt_id.nunique() >= CONFIG["n_splits"]
TEXTS = train.text.astype(str).to_numpy()
TARGETS = train.model.astype(str).to_numpy()
GROUPS = train.prompt_id.astype(str).to_numpy()
INPUT_HASHES = {
    name: hashlib.sha256((DATA / name).read_bytes()).hexdigest()
    for name in ("train.csv", "test.csv")
}
display(pd.DataFrame({
    "dataset": ["train", "test"],
    "rows": [len(train), len(test)],
    "prompt_groups": [train.prompt_id.nunique(), pd.NA],
}))
print("Exact duplicate training texts:", int(train.text.duplicated().sum()))

# %% [markdown]
# ## 3. 每折从零建立分类器
#
# 保留大小写和标点；词 n-gram 捕捉用词习惯，字符 n-gram 捕捉拼写与格式模式。
# 文本特征和分类器均只在当前训练折拟合。SVM 输出为决策分数，不解释为概率。

# %%
def make_classifier():
    common = dict(
        min_df=CONFIG["min_df"], lowercase=False, sublinear_tf=True,
        norm="l2", dtype=np.float32,
    )
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=tuple(CONFIG["word_ngram_range"]),
            max_features=CONFIG["word_max_features"], **common,
        )),
        ("char", TfidfVectorizer(
            analyzer="char", ngram_range=tuple(CONFIG["char_ngram_range"]),
            max_features=CONFIG["char_max_features"], **common,
        )),
    ], n_jobs=1)
    return Pipeline([
        ("features", features),
        ("classifier", LinearSVC(
            C=CONFIG["C"], class_weight=CONFIG["class_weight"],
            dual=True, max_iter=CONFIG["max_iter"], tol=CONFIG["tol"],
            random_state=SEED,
        )),
    ])


def fit_checked(model, texts, labels):
    # A convergence warning invalidates this run rather than silently accepting it.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(texts, labels)
    assert list(model.named_steps["classifier"].classes_) == LABELS
    return model

# %% [markdown]
# ## 4. 五折 prompt 分组 OOF
#
# 每条训练样本只产生一次未见过该 prompt 的预测。
# 全量训练模型在 OOF 结束后另行训练，不参与任何验证指标。

# %%
oof_pred = np.empty(len(train), dtype=object)
oof_scores = np.full((len(train), len(LABELS)), np.nan, dtype=np.float32)
fold_id = np.full(len(train), -1, dtype=np.int16)
fold_records = []
split_records = []
splitter = GroupKFold(n_splits=CONFIG["n_splits"])
for fold, (fit_idx, valid_idx) in enumerate(splitter.split(TEXTS, TARGETS, GROUPS)):
    fold_start = time.monotonic()
    fit_groups = set(GROUPS[fit_idx])
    valid_groups = set(GROUPS[valid_idx])
    assert fit_groups.isdisjoint(valid_groups)
    assert set(TEXTS[fit_idx]).isdisjoint(set(TEXTS[valid_idx])), "Text overlap across folds"
    assert np.all(fold_id[valid_idx] == -1)
    assert set(TARGETS[fit_idx]) == set(LABELS)
    assert set(TARGETS[valid_idx]) == set(LABELS)
    model = fit_checked(make_classifier(), TEXTS[fit_idx], TARGETS[fit_idx])
    scores = model.decision_function(TEXTS[valid_idx])
    assert scores.shape == (len(valid_idx), len(LABELS))
    assert np.isfinite(scores).all()
    predictions = np.asarray(LABELS)[scores.argmax(axis=1)]
    oof_pred[valid_idx] = predictions
    oof_scores[valid_idx] = scores
    fold_id[valid_idx] = fold
    record = {
        "fold": fold,
        "training_rows": len(fit_idx),
        "validation_rows": len(valid_idx),
        "training_prompts": len(fit_groups),
        "validation_prompts": len(valid_groups),
        "balanced_accuracy": float(balanced_accuracy_score(TARGETS[valid_idx], predictions)),
        "accuracy": float(accuracy_score(TARGETS[valid_idx], predictions)),
        "seconds": round(time.monotonic() - fold_start, 3),
    }
    fold_records.append(record)
    split_records.append({"fold": fold, "validation_prompt_ids": sorted(valid_groups)})
    print(json.dumps(record), flush=True)
    del model
    gc.collect()

assert np.all(fold_id >= 0) and np.isfinite(oof_scores).all()
assert set(oof_pred).issubset(set(LABELS))
fold_metrics = pd.DataFrame(fold_records)
oof_balanced_accuracy = float(balanced_accuracy_score(TARGETS, oof_pred))
oof_accuracy = float(accuracy_score(TARGETS, oof_pred))
display(fold_metrics)
print(f"OOF balanced accuracy: {oof_balanced_accuracy:.6f}", flush=True)

# %% [markdown]
# ## 5. 类别诊断与独立指标复算
#
# 各类别召回率的均值必须与 sklearn 的 Balanced Accuracy 相同。
# 折间差异是这次切分的观测波动，不是置信区间，也不是 Public LB 分数。

# %%
matrix = confusion_matrix(TARGETS, oof_pred, labels=LABELS)
support = matrix.sum(axis=1)
assert np.all(support > 0) and int(support.sum()) == len(train)
recall = np.diag(matrix) / support
assert np.isclose(recall.mean(), oof_balanced_accuracy, atol=1e-12)
class_metrics = pd.DataFrame({"model": LABELS, "support": support, "recall": recall})
display(class_metrics.sort_values("recall", ascending=False).reset_index(drop=True))

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})
ordered = class_metrics.sort_values("recall")
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(ordered.model, ordered.recall, color="#356C9B", height=0.7)
for index, value in enumerate(ordered.recall):
    ax.text(min(value + 0.008, 0.98), index, f"{value:.1%}", va="center", fontsize=10)
ax.axvline(1 / len(LABELS), color="#505050", linestyle="--", linewidth=1,
           label="Uniform-guess reference (5.6%)")
ax.set_xlim(0, 1)
ax.xaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_xlabel("Recall on held-out prompts")
ax.set_title("Name That LLM: 5-fold OOF recall by class", loc="left", pad=14)
ax.grid(axis="x", alpha=0.2)
ax.set_axisbelow(True)
ax.legend(loc="lower right", frameon=False)
fig.tight_layout()
fig.savefig(OUT / "class_recall.png", dpi=160, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. 全量训练、生成并检查提交文件
#
# 在全部训练数据上重新拟合一个模型。测试数据只参与 transform/predict，
# 不参与词表、IDF、分类器训练或参数选择。

# %%
final_model = fit_checked(make_classifier(), TEXTS, TARGETS)
test_scores = final_model.decision_function(test.text.astype(str).to_numpy())
assert test_scores.shape == (len(test), len(LABELS)) and np.isfinite(test_scores).all()
submission = pd.DataFrame({
    "id": test.id.to_numpy(),
    "model": np.asarray(LABELS)[test_scores.argmax(axis=1)],
})
assert submission.columns.tolist() == ["id", "model"]
assert submission.id.is_unique and submission.notna().all().all()
assert submission.id.tolist() == test.id.tolist()
assert submission.model.isin(LABELS).all()
submission.to_parquet(OUT / "submission.parquet", index=False)
saved_submission = pd.read_parquet(OUT / "submission.parquet")
pd.testing.assert_frame_equal(submission, saved_submission)
display(submission.head(10))

# %% [markdown]
# ## 7. 保存实验记录与结果小结
#
# 保留逐行 OOF 预测和原始决策分数，后续模型可使用同一份分组进行比较。
# 此 Notebook 的运行只生成文件，不执行正式竞赛提交。

# %%
oof = train[["id", "prompt_id", "model"]].copy()
oof["predicted_model"] = oof_pred
oof["fold"] = fold_id
oof["correct"] = oof.model.eq(oof.predicted_model)
oof.to_parquet(OUT / "oof_predictions.parquet", index=False)
np.save(OUT / "oof_decision_scores.npy", oof_scores)
np.save(OUT / "test_decision_scores.npy", test_scores.astype(np.float32))
fold_metrics.to_csv(OUT / "fold_metrics.csv", index=False)
class_metrics.to_csv(OUT / "class_metrics.csv", index=False)
pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(OUT / "confusion_matrix.csv")
(OUT / "folds.json").write_text(json.dumps(split_records, indent=2), encoding="utf-8")
metrics = {
    "status": "COMPLETE",
    "config": CONFIG,
    "versions": VERSIONS,
    "data_source": "https://www.kaggle.com/competitions/name-that-llm/data",
    "input_sha256": INPUT_HASHES,
    "train_rows": len(train),
    "train_prompt_groups": int(train.prompt_id.nunique()),
    "test_rows": len(test),
    "labels": LABELS,
    "oof_balanced_accuracy": oof_balanced_accuracy,
    "oof_accuracy": oof_accuracy,
    "fold_results": fold_records,
    "fold_score_std_descriptive": float(fold_metrics.balanced_accuracy.std(ddof=1)),
    "prompt_overlap_count": 0,
    "validation_text_overlap_count": 0,
    "convergence_warning_count": 0,
    "submission_checks_passed": True,
    "leaderboard_score": None,
    "total_seconds": round(time.monotonic() - START, 3),
}
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
display(Markdown(
    f"**本轮 OOF Balanced Accuracy：{oof_balanced_accuracy:.4f}。**\n\n"
    f"覆盖 {len(train):,} 条训练数据、{train.prompt_id.nunique()} 个 prompt、"
    f"{len(LABELS)} 类。五折分数范围为 "
    f"{fold_metrics.balanced_accuracy.min():.4f}–{fold_metrics.balanced_accuracy.max():.4f}。\n\n"
    f"生成的 submission.parquet 含 {len(submission):,} 行，已通过结构和读取回验。"
    "本轮尚无 Public LB 分数；该 OOF 结果用于后续实验对照。"
))
print("RUN_COMPLETE", json.dumps({
    "oof_balanced_accuracy": oof_balanced_accuracy,
    "submission_rows": len(submission), "seconds": metrics["total_seconds"],
}), flush=True)
