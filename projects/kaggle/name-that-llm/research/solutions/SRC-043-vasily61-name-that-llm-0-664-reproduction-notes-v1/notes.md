# SRC-043 · vasily61 V1：SRC-021同代码与Claude覆盖负结果

原始记录：[正式台账](../../../tracking/records/sources/SRC-043.json)。
状态：固定源码及作者增补已读；与 SRC-021 完全代码同源；本页未显示分数，已正式归档。


## 值得保留的结论

这份 V1 与 judith007 得分 V2（SRC-021）代码完全相同。作者明确说明 0.664 属于上游固定版本，本副本尚未评分；因此它不是另一个已验证的0.664方法。

作者单独补充的 Claude specialist 负结果值得保留：350行外部样本中4个原错改对、68个原对改错；样本仅14类、重叠未查、ATLAS未完整运行，不能推广成所有Claude专门模型都无效。

## 来源与原件

- [Notebook](https://www.kaggle.com/code/vasily61/name-that-llm-0-664-reproduction-notes)；MCP 确认当前 V1，再经 CLI 固定拉取 v1，采集 2026-11-01 07:34:36 UTC。
- [source.ipynb](source.ipynb)、[metadata.json](metadata.json)、[采集哈希](download-receipt.json)、[原文增补](description.md)、[代码](code.txt)、[对 SRC-002 差异](diff-vs-SRC-002.patch)、[主窗口页面观察](snapshots/browser-observation-20261101.json)。
- 主窗口本轮（2026-11-01 UTC，逐页时刻未记录，采集截至07:35:57 UTC）比赛Code列表未显示本项Score；本条 `public_score=null` 不表示零分或失败。未取得本副本 scriptVersionId。

## 作者声称与实际设置

作者“Reproduction and our experimental notes”明确链接上游 [judith V2/363834272](https://www.kaggle.com/code/judith007/0-664-hyperparameter-tuned-who-are-u?scriptVersionId=363834272)，并说当前fork尚未评价；此前其V31提交为0.660。增补仅报告外部Claude专家覆盖的负结果，明确没有包含在当前代码中。

实际参数：Sinkhorn0.30、target mix0.35、标点1.5、ATLAS20%、互补栈15%、SVC1.5、pool-v5；相对 SRC-002 仅改 Sinkhorn 0.45→0.30。

## 我们核查了什么

- 完整 code SHA256 为 `f9fbf9845de6c62c9bca9a753d2fe5aac1d35e106541c1e07ca5755a6449bce9`，与 SRC-021 完全一致；不同作者出处保留，但不算独立算法。
- 读到了作者对分数归属与负结果条件的明确限定；其余“All Public / What Failed”大段为继承正文，不自动视为此作者亲自运行的历史实验。
- 未取得350行数据、逐行预测或提交ID，负结果仅为作者报告；没有运行或核验该副本的完整隐藏测试。

## 对当前方案的用途与限制

复用 SRC-021 的方法与固定参数，不凭0.664标题重复计数。Claude覆盖失败提醒 IDEA-004 应检查对/错互换而不只看专门模型的单项精度；350行且仅14类的结果只作风险线索。

## 待验证想法

关联 IDEA-004：若后续研究专门模型，使用训练范围可查的分组OOF和完整类别，统计原集成/专家互换矩阵与小权重融合；先核查样本重叠，再评估收益。本次不执行实验。

## 更新记录

- 2026-11-01：固定V1，确认与SRC-021同代码，保留作者明确的未评分声明及有边界的负结果。
