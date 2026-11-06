# SRC-060 · Pablo：自适应中心化与收缩边际Sinkhorn的OOF声明

原始记录：[正式台账](../../../tracking/records/sources/SRC-060.json)。
状态：已读取原帖与当前可返回评论；已归档，正式记录见下方链接。

## 值得保留的结论

作者给轻量管线OOF阶梯0.3145→0.4345；无逐行OOF/源码版本绑定，且全prompt Hungarian提升不代表稀疏测试收益。

## 来源与原件

- 原帖：[Kaggle 讨论 754026](https://www.kaggle.com/competitions/name-that-llm/discussion/754026)；作者 `pablodiegoo`。
- 采集时间：2026-11-01T07:32:15.226936+00:00；原帖发布：2026-10-30T03:43:11.882121800Z。
- [原帖与评论快照](snapshots/20261101T073215Z.json)；SHA-256 `66fde552e1471afe1492480ddea7814282947310aa2d5b9391428bd7301f0b4d`。这是公开只读 API 响应，不是独立实验验证。

## 作者声称与实际设置

2026-10-30原帖关联 [Starter Notebook](https://www.kaggle.com/code/pablodiegoo/name-that-llm-starter-notebook)。作者在prompt_id的5折GroupKFold给出阶梯：word TF-IDF0.3145；加char/char_wb与20风格比率0.3367；加LinearSVC与top100功能词外掩码0.3686；自适应中心化0.3812；收缩目标边际Sinkhorn0.3834；组内Hungarian0.4345。

参数声明：聚类sim_thr0.15，中心化alpha0.65；Sinkhorn alpha0.45、rho0.35、100轮。另称V40A+V31的供应商家族内残差用logsumexp保持家族质量，但没有独立量化该项收益。

## 我们核查了什么

已读原帖，无评论。帖子给的是作者OOF而非0.664得分配方；没有把表格绑定特定scriptVersionId，也没有逐行OOF或训练/特征构造隔离证据。所链Notebook的已登记SRC应由合并窗口以URL查重；关联旧源码不代表它已支持当前表格。

原文将2699条描述为18模型×180prompt，数目并非完整笛卡尔积，应保留缺失组合而不能按完整组假设实现。作者关于Public/Private类计数不均的解释不是官方统计。

## 对当前方案的用途与限制

全组Hungarian尤其可能受每组类别齐全程度影响，+0.0511的OOF增量不能直接外推隐藏稀疏组；功能词掩码和概率边际是两个不同干预，分别验证。

## 待验证想法

关联IDEA-002：先核对源码与折内fit范围，抽稀后重算聚类、中心化和assignment，再分别打开各组件；报告组大小分层收益和违反单类唯一假设的样本。

## 更新记录

- 2026-11-01：读取原帖和当前评论，保存时间快照。新来源，正式编号已登记。 本轮未训练、未提交。
