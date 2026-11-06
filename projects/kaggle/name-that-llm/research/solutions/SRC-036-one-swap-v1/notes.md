# SRC-036 · One Swap V1：Open Pool + SVC12，绑定0.664
状态：已核验固定源码、平台得分版本；未独立复现。
原始记录：[正式台账](../../../tracking/records/sources/SRC-036.json)。

## 值得保留的结论
平台 0.664 明确绑定 V1 / scriptVersionId 364073396。相对 SRC-002 alexchilton V8，代码新增 Open Pool 读取/过滤/写出块，并将 svc_w 从 1.5 改为 12；ATLAS 仍为 20%、Sinkhorn 0.45、标点系数 1.5。它同时改变数据和融合权重，不能视为已证明“只换池子”的单变量收益。

## 来源与原件
- [作者得分 V1](https://www.kaggle.com/code/zbigniewkirchner/0-664-one-swap-198k-clean-pool-fully-public?scriptVersionId=364073396)
- [固定源码](source.ipynb)、[下载元数据](metadata.json)、[获取时间与哈希](receipt.json)、[对 alex V8 的代码差分](vs-alex-v8.diff)。
- 平台版本观察由本批次 one-swap-versions-dom-observation.json 留证；作者原文是源码 Markdown，V1 含从 alexchilton 继承的说明，第一人称失败实验不得归给新作者。

## 作者声称与实际设置
作者最新版说明声称 198507 条清洗池、SVC 单模型五折从0.4805到0.5008（193k池、pool_weight0.35）；该 CV 不属于当前V1源码中执行出的结果。V3说明还列 svc_w=1.5/6/12/20 的 LB 为0.664/0.659/0.664/0.664，是作者报告而非本次逐个绑定的成绩。V1平台成绩已核验，其他权重梯度只能引用关联新版说明。
FAST_COMMIT 在无比赛重跑标志且测试样本少于50时跳过重计算；V1保存运行11分29秒，不能当完整隐藏评分耗时。约8小时仅为作者在V3的估计。

## 我们核查了什么
下载固定 v1，计算全部代码单元拼接SHA256，保存差分。Open Pool 从 flag_train_prompt_regen 过滤后写入 /kaggle/working/open_pool.parquet；之后 svc4 仍会按标签、规范化文本和官方/外部数据去重。打印 len(_op) 是过滤前行数，不是最终拟合样本数。常规运行 POOL 主要流向 svc4；TF-IDF/distort/comp并未整体改用新池。代码仍先寻找 nlm_pool_v5 原文件，再覆盖 POOL 变量。
版本历史显示 V1 为2026-10-31 01:30 UTC，V3为当天16:46 UTC quick version；API metadata.lastRunTime 返回继承的旧时间，不能用它替代网页版本历史。

## 对当前方案的用途与限制
优先作为主svc4公共池对照候选，关联 IDEA-003；先沿用已有来源/官方prompt隔离审计。仅有代码相同/公开声明不能证明输入内容、许可证和模型权重全部可复现。数据卡叫v10，Kaggle数据版本实际11；本次未下载全量池或逐项固定所有附件。
计时有耦合：infer的T0在SVC训练前，ATLAS按 time.time()-T0 和预算退出；ATLAS_BUDGET=8.3h-COMP_SEC-600。扩大池可能改变ATLAS覆盖量。svc_w又会改变混合概率与margin<0.3选行，不能只比较总分而忽略被ATLAS处理的样本。
代码Apache2.0、Open Pool卡CC BY-SA4.0、Llama另有其许可；不视作统一授权。本笔记仅记录来源声明，未作法律许可审查。

## 待验证想法
固定prompt-group验证、成员预测与输入版本，分别比较旧池/清洗池和svc_w1.5/12；记录各来源权重、去重后行数、单成员错误交集、ATLAS选行与完成覆盖量。没有启动训练、GPU或正式提交。

## 更新记录
- 2026-11-01：新增固定V1与平台绑定核查；关联SRC-002、SRC-027和同来源V3、Open Pool数据条目。

关联来源：[SRC-002](../../../tracking/研究索引.md#src-002) · [SRC-027](../../../tracking/研究索引.md#src-027) · [SRC-037](../../../tracking/研究索引.md#src-037) · [SRC-038](../../../tracking/研究索引.md#src-038)。
