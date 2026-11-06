# 固定验证协议

`prompt-group-v1` 固定 EXP-001 V4 的五折分组：同一个 `prompt_id` 的回答全部留在同一折。源数据哈希、逐行文本哈希、标签顺序和折文件哈希都在 manifest 中；修改数据或划分需要另建协议版本，不能覆盖此版本。

```python
from pathlib import Path
from validation.protocol import load_split, select_sparse_rows

train, folds, labels = load_split(Path("."))
for fold in range(5):
    fit_rows = train.loc[folds.fold.ne(fold)]
    valid_rows = train.loc[folds.fold.eq(fold)]
    # 只在 fit_rows 拟合特征提取器、分类器、校准参数和有监督统计量。
    # valid_rows 只用于 transform / predict；按原样本 ID 保存结果。
```

每个模型缓存必须记录：样本 ID、标签顺序、fold、数据与源码哈希、checkpoint 的训练范围、分数类型、逐行预测和阶段耗时。LinearSVC 的 decision scores **不是概率**；不能直接作为概率做融合或 Sinkhorn。

这是一套**开发验证集**，不是从未看过的最终 holdout。大量调参仍可能过拟合这五折；正式比较应限制候选、使用配对预测检查稳定性，必要时用嵌套验证或新增独立 prompt。不能把现成全量训练的适配器事后切折称为 OOF；外部数据需按验证 prompt/近似 prompt 与文本隔离，并核查其标签来源。

稀疏场景的候选设置为每个 prompt 取 2/3/6 条，repeat=0..4，seed=42。`select_sparse_rows` 只按样本 ID 的稳定散列选取，不读取标签。对每个验证折，必须**先选择文本，再重新执行**聚类、中心化、Gaussian、Sinkhorn 等组相关计算。不能从完整组的最终预测里抽行冒充稀疏测试。这些设置是压力测试，不是已确认的隐藏测试集分布。

EXP-007 只复核历史轻量模型 OOF，并进行数据质量诊断；没有执行这些稀疏推理实验，也没有跑 ATLAS20。近似重复候选仅供审阅，不自动删除或修改原文。

运行检查：`python -X utf8 -m unittest discover -s validation -p test_protocol.py`。

方法参考：[sklearn 数据泄漏与 Pipeline](https://scikit-learn.org/stable/common_pitfalls.html)、[按组交叉验证](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)。
