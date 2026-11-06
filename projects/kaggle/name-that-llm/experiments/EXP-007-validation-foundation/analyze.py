"""Audit existing EXP-001 OOF; never trains a classifier or starts cloud jobs.

Run from any directory with the local scientific Python environment.
The historical parquet is retained; its CSV conversion needs pandas+pyarrow.
"""
from __future__ import annotations

import json
import platform
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from validation.protocol import require, sha256, text_sha256, validate_score_cache

REFERENCE = HERE / "reference/exp-001-v4"
OUT = HERE / "outputs"
SPLIT = ROOT / "validation/prompt-group-v1"
THRESHOLD = 0.90


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def immutable_write(path, content):
    if path.exists():
        require(path.read_bytes() == content, f"Frozen validation file differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def table(frame, columns=None, limit=10):
    frame = frame if columns is None else frame[columns]
    rows = ["| " + " | ".join(map(str, frame.columns)) + " |",
            "|" + "---|" * len(frame.columns)]
    for row in frame.head(limit).itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(x).replace("|", "\\|") for x in row) + " |")
    return "\n".join(rows)


def find_near_duplicates(train, folds):
    # Quality audit only: the fitted vectorizer never enters any classifier.
    # Exhaustive pair scan in blocks; the 100k feature cap is an explicit limitation.
    vec = TfidfVectorizer(analyzer="char", ngram_range=(5, 5), lowercase=False,
                          min_df=2, max_features=100000, dtype=np.float32, norm="l2")
    x = vec.fit_transform(train.text.map(normalized))
    rows = []
    for start in range(0, len(train), 128):
        sim = (x[start:start + 128] @ x.T).toarray()
        a, b = np.where(sim >= THRESHOLD)
        for local, right in zip(a.tolist(), b.tolist()):
            left = start + local
            if left >= right:
                continue
            rows.append({
                "id_a": train.id.iloc[left], "id_b": train.id.iloc[right],
                "prompt_a": train.prompt_id.iloc[left], "prompt_b": train.prompt_id.iloc[right],
                "model_a": train.model.iloc[left], "model_b": train.model.iloc[right],
                "cosine": float(sim[local, right]),
                "cross_prompt": bool(train.prompt_id.iloc[left] != train.prompt_id.iloc[right]),
                "cross_fold": bool(folds.fold.iloc[left] != folds.fold.iloc[right]),
                "text_a": train.text.iloc[left], "text_b": train.text.iloc[right],
            })
    columns = ["id_a", "id_b", "prompt_a", "prompt_b", "model_a", "model_b", "cosine",
               "cross_prompt", "cross_fold", "text_a", "text_b"]
    result = pd.DataFrame(rows, columns=columns).sort_values("cosine", ascending=False)
    result.to_csv(OUT / "near_duplicate_candidates.csv", index=False)
    return {
        "method": "NFKC+casefold+whitespace; char 5-gram TF-IDF cosine",
        "min_df": 2, "max_features": 100000, "actual_features": len(vec.vocabulary_),
        "threshold": THRESHOLD, "all_pair_count": len(train) * (len(train) - 1) // 2,
        "candidate_pairs": len(result),
        "cross_prompt_pairs": int(result.cross_prompt.sum()),
        "cross_fold_pairs": int(result.cross_fold.sum()),
        "scope": "Lexical candidate screen, not semantic paraphrase detection or label adjudication",
    }


def source_inventory(observed, hashes):
    source_manifest = ROOT / "cloud/atlas20-drive/drive-inputs.json"
    manifest = read_json(source_manifest)
    raw = {
        "seyitkaangunes/ntl-extra-api-passages": ["ntl_extra_passages_claude.csv"],
        "abdullah1afify/ntl-generations": ["ntl_generations.csv"],
        "dophuchung/ntl-proxy-generations": ["ntl_proxy_generations.csv"],
        "dhruvtibarewal/nlm-matched-clean": ["nlm_matched_clean.parquet", "nlm_pool_v5.parquet"],
        "lucasboesen/ntl-free-proxy-generations": ["gen_passages.csv"],
        "prvsiyan/prvsiyans-extended-name-that-llm-dataset": ["passages_clean.csv"],
    }
    sources = [dict(source_id="official-train", kind="competition", ref="name-that-llm",
                    url="https://www.kaggle.com/competitions/name-that-llm/data",
                    role="labeled development / fixed OOF", local_path="train.csv",
                    sha256=hashes["train.csv"], rows=2699, row_audit="completed"),
               dict(source_id="official-test-example", kind="competition", ref="name-that-llm",
                    url="https://www.kaggle.com/competitions/name-that-llm/data",
                    role="three public schema examples; no labels, no validation score",
                    local_path="test.csv", sha256=hashes["test.csv"], rows=3, row_audit="completed")]
    assets = []
    for asset in manifest["inputs"]:
        item = {key: asset[key] for key in ("kind", "ref", "version", "legacy", "relocations") if key in asset}
        item["evidence"] = "cloud/atlas20-drive/drive-inputs.json"
        item["live_cloud_verified_in_exp007"] = False
        item["byte_identity_to_exp004"] = "not established"
        assets.append(item)
        if asset["ref"] in raw:
            sources.append(dict(source_id=asset["ref"], kind="external_generation_pool", ref=asset["ref"],
                url=f"https://www.kaggle.com/datasets/{asset['ref']}/versions/{asset['version']}",
                version=asset["version"], candidate_files=raw[asset["ref"]], role="candidate training data",
                drive_path="/content/drive/MyDrive/Kaggle/" + asset["legacy"],
                sha256=None, rows=None, row_audit="not_read_or_audited_in_exp007",
                historical_expected_file_hashes=[entry for entry in manifest.get("small_file_hashes", [])
                    if entry["path"].startswith(asset["target"] + "/") and entry["path"].endswith((".csv", ".parquet"))],
                exact_generator_version_verified=False,
                next_audit="schema, exact generator identity, duplicates across pools/official data, prompt leakage, source confounding"))
    write_json(ROOT / "data/source-inventory.json", {
        "schema_version": 1, "observed_at": observed, "origin_experiment": "EXP-007",
        "manifest_sha256": sha256(source_manifest), "text_sources": sources, "input_assets": assets,
        "scope": "Official CSVs audited; six named external text sources registered from local manifest only. Other assets can also contain data and require later inspection.",
    })
    return sources


def plots(classes, matrix, labels):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    ordered = classes.sort_values("recall")
    fig, ax = plt.subplots(figsize=(10, 7.8))
    ax.barh(ordered.model, ordered.recall, color="#2c7896")
    for i, value in enumerate(ordered.recall):
        ax.text(value + .012, i, f"{value:.1%}", va="center", fontsize=9)
    ax.set(xlim=(0, 1), xlabel="Recall on held-out prompts")
    ax.set_title("EXP-001 diagnostic: TF-IDF / LinearSVC\nReused 5-fold OOF; not ATLAS20", loc="left", pad=14)
    ax.axvline(1 / 18, color="#666666", linestyle="--", linewidth=1, label="Uniform-guess reference (5.6%)")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "class_recall.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(13.5, 11.5))
    normalized_cm = matrix / matrix.sum(axis=1, keepdims=True)
    im = ax.imshow(normalized_cm, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=65, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    ax.set(xlabel="Predicted model", ylabel="True model")
    ax.set_title("TF-IDF / LinearSVC OOF confusion\nColor = row fraction; numbers = counts. Not ATLAS20.", loc="left", pad=16)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if matrix[i, j] > 0:
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if normalized_cm[i, j] > .55 else "#202020")
    fig.colorbar(im, ax=ax, shrink=.75, label="Fraction within true class")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def main():
    started = time.monotonic()
    observed = datetime.now(timezone.utc).isoformat()
    OUT.mkdir(parents=True, exist_ok=True)
    historical = read_json(REFERENCE / "metrics.json")
    hashes = {name: sha256(ROOT / name) for name in ("train.csv", "test.csv")}
    require(hashes == historical["input_sha256"], "Historical input hashes differ")
    train = pd.read_csv(ROOT / "train.csv", dtype={"id": str, "prompt_id": str})
    test = pd.read_csv(ROOT / "test.csv", dtype={"id": str})
    labels = historical["labels"]
    require(len(labels) == 18 and labels == sorted(train.model.unique()), "Label set changed")
    require(train.id.is_unique and test.id.is_unique, "Repeated official ID")
    require(not train.isna().any().any() and not test.isna().any().any(), "Missing official values")
    require(train.text.str.strip().ne("").all() and test.text.str.strip().ne("").all(), "Blank text")
    require(set(train.id).isdisjoint(test.id), "Train/test ID overlap")
    oof = pd.read_csv(REFERENCE / "oof_predictions.csv", dtype={"id": str, "prompt_id": str})
    scores = np.load(REFERENCE / "oof_decision_scores.npy", allow_pickle=False)
    fold_map = {}
    for item in read_json(REFERENCE / "folds.json"):
        for prompt in item["validation_prompt_ids"]:
            require(prompt not in fold_map, "Prompt repeated in historical folds")
            fold_map[prompt] = item["fold"]
    require(set(fold_map) == set(train.prompt_id), "Historical prompt coverage differs")
    folds = train[["id", "prompt_id", "model"]].copy()
    folds["fold"] = train.prompt_id.map(fold_map)
    folds["text_sha256"] = train.text.map(text_sha256)
    validate_score_cache(train, folds, oof, scores, labels, historical["labels"])
    immutable_write(SPLIT / "folds.csv", folds.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    split_manifest = {
        "schema_version": 1, "split_id": "prompt-group-v1", "rows": len(train),
        "prompts": train.prompt_id.nunique(), "n_splits": 5, "labels": labels,
        "train_sha256": hashes["train.csv"], "folds_sha256": sha256(SPLIT / "folds.csv"),
        "origin": {"experiment": "EXP-001", "notebook_version": 4, "script_version_id": 363168447,
                   "folds_sha256": sha256(REFERENCE / "folds.json"), "splitter": "GroupKFold(n_splits=5), shuffle=False"},
        "scope": "Frozen development OOF split, not an untouched holdout; seed 42 belongs to historical classifier, not GroupKFold",
    }
    immutable_write(SPLIT / "manifest.json", (json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print("Historical sample IDs, labels, decision scores, folds and source hashes PASS", flush=True)

    norm = train.text.map(normalized)
    duplicates = train.assign(normalized_text=norm).loc[norm.duplicated(keep=False)]
    duplicates.to_csv(OUT / "normalized_duplicate_rows.csv", index=False)
    prompt_counts = train.groupby("prompt_id").agg(rows=("id", "size"), classes=("model", "nunique"))
    class_counts = train.groupby("model").size().reindex(labels)
    quality = {
        "input_sha256": hashes, "rows": len(train), "prompt_count": int(train.prompt_id.nunique()),
        "class_counts": {k: int(v) for k, v in class_counts.items()},
        "missing_values": int(train.isna().sum().sum()), "duplicate_ids": int(train.id.duplicated().sum()),
        "duplicate_prompt_model_pairs": int(train.duplicated(["prompt_id", "model"]).sum()),
        "exact_duplicate_extra_rows": int(train.text.duplicated().sum()),
        "normalized_duplicate_extra_rows": int(norm.duplicated().sum()),
        "normalized_duplicate_cross_prompt_groups": int(duplicates.groupby("normalized_text").prompt_id.nunique().gt(1).sum()),
        "train_test_normalized_text_overlap": len(set(norm).intersection(test.text.map(normalized))),
        "incomplete_prompt_groups": [{"prompt_id": p, "rows": int(r.rows),
            "missing_labels": sorted(set(labels) - set(train.loc[train.prompt_id.eq(p), "model"]))}
            for p, r in prompt_counts.iterrows() if r.rows != len(labels)],
    }
    require(quality["normalized_duplicate_cross_prompt_groups"] == 0, "Cross-prompt duplicates need review before validation")
    quality["near_duplicates"] = find_near_duplicates(train, folds)
    write_json(OUT / "data-quality.json", quality)
    print("Lexical near-duplicate screen:", quality["near_duplicates"], flush=True)

    ba = float(balanced_accuracy_score(oof.model, oof.predicted_model))
    require(abs(ba - historical["oof_balanced_accuracy"]) < 1e-12, "Historical metric mismatch")
    matrix = confusion_matrix(oof.model, oof.predicted_model, labels=labels)
    classes = pd.DataFrame({"model": labels, "support": matrix.sum(axis=1), "correct": matrix.diagonal(),
                            "recall": matrix.diagonal() / matrix.sum(axis=1)})
    classes.to_csv(OUT / "per_class.csv", index=False)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(OUT / "confusion_matrix.csv")
    pairs = pd.DataFrame([{"true_model": labels[i], "predicted_model": labels[j],
        "count": int(matrix[i, j]), "fraction_of_true_class": float(matrix[i, j] / matrix[i].sum())}
        for i in range(len(labels)) for j in range(len(labels)) if i != j and matrix[i, j]])
    pairs = pairs.sort_values(["count", "true_model", "predicted_model"], ascending=[False, True, True])
    pairs.to_csv(OUT / "top_confusions.csv", index=False)
    diag = oof.copy()
    diag["words"] = train.text.map(lambda t: len(re.findall(r"\S+", t)))
    diag["characters"] = train.text.str.len()
    diag["same_family"] = diag.model.str.split("-").str[0].eq(diag.predicted_model.str.split("-").str[0])
    sorted_scores = np.sort(scores, axis=1)
    diag["top1_minus_top2_decision_score"] = sorted_scores[:, -1] - sorted_scores[:, -2]
    diag["top2_hit"] = [truth in np.asarray(labels)[row] for truth, row in zip(diag.model, np.argsort(scores, axis=1)[:, -2:])]
    diag["length_bin"] = pd.cut(diag.words, [-np.inf, 80, 140, 200, np.inf],
                                labels=["<=80", "81-140", "141-200", ">200"])
    diag.to_csv(OUT / "oof_diagnostics.csv", index=False)
    wrong = diag.loc[~diag.correct].copy()
    wrong["text"] = train.loc[wrong.index, "text"]
    wrong.sort_values("top1_minus_top2_decision_score", ascending=False).head(50).to_csv(OUT / "error_examples.csv", index=False)
    length_rows = []
    for name, part in diag.groupby("length_bin", observed=True):
        length_rows.append({"length_bin": str(name), "rows": len(part), "classes_present": part.model.nunique(),
                           "accuracy": float(part.correct.mean()),
                           "macro_recall_present_classes": float(part.groupby("model").correct.mean().mean())})
    lengths = pd.DataFrame(length_rows)
    lengths.to_csv(OUT / "length_slices.csv", index=False)
    prompts = diag.groupby("prompt_id").agg(fold=("fold", "first"), rows=("id", "size"),
                                            correct=("correct", "sum"), accuracy=("correct", "mean"))
    prompts.sort_values(["accuracy", "prompt_id"]).to_csv(OUT / "prompt_difficulty.csv")
    fold_metrics = diag.groupby("fold").apply(lambda p: pd.Series({
        "rows": len(p), "prompts": p.prompt_id.nunique(), "accuracy": p.correct.mean(),
        "balanced_accuracy": p.groupby("model").correct.mean().mean()}))
    fold_metrics.to_csv(OUT / "fold_metrics.csv")
    for row in historical["fold_results"]:
        require(abs(fold_metrics.loc[row["fold"], "balanced_accuracy"] - row["balanced_accuracy"]) < 1e-12,
                "Historical per-fold metric mismatch")
    sources = source_inventory(observed, hashes)
    plots(classes, matrix, labels)
    source_nb = REFERENCE / "name-that-llm-no-llama-groupcv.ipynb"
    cache_manifest = {
        "experiment": "EXP-001", "notebook_ref": "zhjianduan/name-that-llm-no-llama-groupcv",
        "notebook_version": 4, "script_version_id": 363168447,
        "artifact_sha256": {p.name: sha256(p) for p in sorted(REFERENCE.iterdir()) if p.is_file()},
        "score_type": "LinearSVC decision_function; not calibrated probabilities",
        "score_rows": "oof_predictions.csv in original train.csv order", "score_labels": labels,
        "source_review": {"notebook_sha256": sha256(source_nb),
            "findings": ["code cell 6 defines a new TF-IDF + LinearSVC Pipeline",
                "code cell 8 fits a fresh Pipeline only on each fold's fit_idx",
                "code cell 8 uses held-out rows only with decision_function",
                "code cell 12 full-data fitting is after OOF and only creates public example predictions"],
            "cell_numbering": "zero based", "scope": "Source and saved-artifact audit; no retraining"},
    }
    write_json(OUT / "reference-integrity.json", cache_manifest)
    metrics = {
        "experiment": "EXP-007", "observed_at": observed, "status": "COMPLETE",
        "mode": "reuse_historical_oof_no_classifier_training", "source_experiment": "EXP-001",
        "validation_split_id": "prompt-group-v1", "balanced_accuracy": ba,
        "accuracy": float(diag.correct.mean()), "top2_accuracy": float(diag.top2_hit.mean()),
        "same_family_fraction_of_errors": float(wrong.same_family.mean()),
        "rows": len(train), "correct": int(diag.correct.sum()), "wrong": len(wrong),
        "prompt_overlap_count": 0, "score_artifact_alignment": "PASS",
        "historical_metric_recomputed": True, "classifier_retrained": False,
        "gpu_seconds": 0, "api_usd": 0, "colab_cu": 0, "new_leaderboard_score": None,
        "audit_wall_seconds": round(time.monotonic() - started, 3),
        "timing_note": "Local audit from main() entry; excludes Python/import startup, artifact download and historical classifier runtime",
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__},
        "analysis_sha256": sha256(Path(__file__)), "protocol_sha256": sha256(ROOT / "validation/protocol.py"),
        "scope": "Lightweight TF-IDF/LinearSVC diagnostic only; no ATLAS20 OOF or external pool row audit",
    }
    write_json(OUT / "metrics.json", metrics)
    display_classes = classes.sort_values("recall").head(6).copy()
    display_classes["recall"] = display_classes.recall.map(lambda x: f"{x:.1%}")
    display_pairs = pairs.head(8).copy()
    display_pairs["fraction_of_true_class"] = display_pairs.fraction_of_true_class.map(lambda x: f"{x:.1%}")
    external = pd.DataFrame([{"来源": s["ref"], "固定版本": s["version"], "审计": "来源登记；未读原文"} for s in sources if "version" in s])
    report = f"""# EXP-007：数据与固定验证基础

本轮完成于 {observed}。沿用 EXP-001 V4 的历史 OOF 并独立复算；**没有重训分类器、重跑 ATLAS20、启用 GPU、调用付费 API 或提交比赛**。0.662 正式基线保持为 EXP-004。

## 已完成与关键结果

- 官方训练数据：{len(train)} 条，150 个 prompt、18 类。149 个 prompt 各 18 条；p002 缺 `claude-fable-5`，因此该类 149 条，其余每类 150 条。
- 空值、重复 ID、重复 `(prompt_id, model)`、原文完全重复、NFKC/大小写/空白规范化重复：均为 0。训练与公开测试样例的规范化文本交集为 {quality['train_test_normalized_text_overlap']}。
- 近似重复筛查：比较 {quality['near_duplicates']['all_pair_count']:,} 对文本，阈值 {THRESHOLD}，得到 {quality['near_duplicates']['candidate_pairs']} 对候选，其中跨 prompt {quality['near_duplicates']['cross_prompt_pairs']} 对、跨折 {quality['near_duplicates']['cross_fold_pairs']} 对。候选表保存完整原文，未自动删除。字符词表上限 100,000；这项检查不能发现所有语义改写或证明模型训练数据无泄漏。
- 数据哈希与历史运行完全一致；2699 行样本 ID、标签、决策分数、预测、正确标记和五折位置一一对应；各折指标也与原记录一致。
- 轻量模型 OOF Balanced Accuracy **{ba:.6f}**；正确 {metrics['correct']}/{len(train)}。这是重算历史结果，**不是一次新提升，也不是 ATLAS20 的验证成绩**。
- 固定产物：[划分清单](../../validation/prompt-group-v1/manifest.json)、[逐行折表](../../validation/prompt-group-v1/folds.csv)、[使用规则](../../validation/README.md)、[数据来源清单](../../data/source-inventory.json)。

## 轻量模型暴露的困难

以下只支持 TF-IDF/LinearSVC 的诊断，不能据此认定强集成也存在同样弱项。

召回率最低的六类：

{table(display_classes)}

最常见的有方向误判：

{table(display_pairs)}

全部错例中，同一厂商/家族前缀之间的混淆占 {metrics['same_family_fraction_of_errors']:.1%}。Top-2 命中率为 {metrics['top2_accuracy']:.1%}；它只是排名诊断，不是提交指标。分差较大的前 50 个错例见 [error_examples.csv](outputs/error_examples.csv)，分差不是校准置信度。

![18 类召回](outputs/class_recall.png)

按空白分词的长度切片如下。切片间类别、题目难度可能不同，不能当作“长度导致正确率变化”的因果证据。

{table(lengths.round(4))}

更多产物：[混淆矩阵图](outputs/confusion_matrix.png)、[逐行诊断](outputs/oof_diagnostics.csv)、[按 prompt 的难度](outputs/prompt_difficulty.csv)、[近似重复候选](outputs/near_duplicate_candidates.csv)。官方文件没有 prompt 原文，本轮没有推断或编造题材标签。

## 外部数据：已登记，尚未清洗

已登记 Drive 输入清单中的全部 {len(read_json(ROOT / 'cloud/atlas20-drive/drive-inputs.json')['inputs'])} 个资产引用，单独列出 6 个明确的生成文本来源：

{table(external)}

版本来自已有本地缓存清单，本轮没有重新连接云端核查，也没有读取这些公共池的全部原文；其他模型资产也可能包含训练表，后续需继续检查。部分文件的历史期望哈希已随来源登记，但本轮尚未对实际内容核验；外部样本数、准确生成模型版本与去重结果仍为未知，不能声称已建立干净的大训练池。

## 后续实验沿用的约束

1. 五折严格按 prompt 分组；拟合 TF-IDF、分类器和监督统计量只用训练折。现成 LoRA 的训练范围要另外核查，不能用这套折表把全量训练适配器的输出变成干净 OOF。
2. 这一套折已经用于开发，不是未触碰的最终 holdout。限制调参候选；需要更强效果证据时再做嵌套验证或新增独立 prompt。
3. 稀疏回答压力测试预设每 prompt 2/3/6 条、5 次固定抽样；本轮只提供选样函数，没有跑稀疏推理。组相关后处理要先选文本、再重新计算，不能抽取完整组预测。
4. 原文保持大小写、标点和排版；规范化只用于审计。外部数据必须核对生成模型的准确版本，并排查跨来源重复、验证 prompt 和来源特征与标签的混淆。

**推荐下一步：固定这套折与现有轻量分类器，审计并小规模比较公共池的原始版、清洗版、等量抽样版。先检验数据是否有用，再把有希望的数据配方交给强模型确认。** 每轮另登记 EXP，不用这次诊断替代强集成的逐成员错误分析。

## 复核

本轮只下载 EXP-001 的固定 V4 源码和产物。来源：[历史 Notebook V4](https://www.kaggle.com/code/zhjianduan/name-that-llm-no-llama-groupcv?scriptVersionId=363168447)。完整来源哈希和源码检查记录见 [reference-integrity.json](outputs/reference-integrity.json)。原 parquet 保留；本地以其 CSV 转换件读取，18 列分数来自原始 NPY。

`python -X utf8 experiments/EXP-007-validation-foundation/analyze.py` 可重算报告；已有固定划分如发生变化会拒绝覆盖。`python -X utf8 -m unittest discover -s validation -p test_protocol.py` 检查泄漏和缓存错位保护。运行环境与实际审计耗时见 [metrics.json](outputs/metrics.json)。

方法参考：[sklearn 数据泄漏与 Pipeline](https://scikit-learn.org/stable/common_pitfalls.html)、[按组交叉验证](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)、[Balanced Accuracy 定义](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html)。
"""
    (HERE / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "balanced_accuracy": ba, "audit_seconds": metrics["audit_wall_seconds"],
                      "lowest_recall": display_classes.to_dict("records"), "top_confusions": pairs.head(5).to_dict("records")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
