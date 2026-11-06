# SRC-068 · chyibo V1：0.665 与 Apex E 共用推理配方

原始记录：[正式台账](../../../tracking/records/sources/SRC-068.json)。[Kaggle 得分版本 V1 / 364709540](https://www.kaggle.com/code/chyibo/name-that-llm-0663-fork?scriptVersionId=364709540)。

## 值得保留的结论

Kaggle 页面显示 Version 1 of 1，Public / Best 0.665 均属于 V1。固定源码与 alexchilton 更新后的 **V9 全文件逐字相同**，包括 Markdown、绘图及 Notebook 内嵌元数据。与另一条 0.665 的 synthreaper V1 比较，chyibo 将标点系数从 1.5 改到 1.2 的补丁应用后，实际生成的主推理程序、其余推理单元和最终运行命令逐字相同。这两个出处应归入同一推理配方。[固定源码](source.ipynb) · [对比凭据](receipt.json) · [alexchilton V9 元数据](upstream-alex-v9-metadata.json)

## 来源、版本和分数

- 作者账号 `chyibo`，Notebook ID 142199340。Kaggle MCP 的默认与显式 `v1` 响应源码一致，另从 Kaggle API 用 `version_label=v1` 保存原件。采集 2026-11-02 07:10:15 UTC。源码 SHA-256 `07cb21ed6f9b2b2d0920756a8fea64ec5a5d37637d7ea4fad07ab6eb85ce5140`，10 个代码单元以两个换行连接的 SHA-256 `e3b8fa97495e45873b1297b782488819b4a04ac53cd08baa74e3637505b54605`。[元数据](metadata.json) · [哈希记录](receipt.json)
- 主窗口实读页面及运行日志，版本链接给出全局 ID 364709540，保存运行 6 分 35 秒、T4×2。Notebook 正文称“在我们账号得 0.663”，这段来自 alexchilton V9 的继承说明。chyibo 的绑定分数以页面这次 0.665 为准。隐藏耗时及 Private 分数未知。[页面与日志字段](platform-observation.json)
- alexchilton V9 与旧 SRC-002 V8 的 Notebook ID 都是 139599456，slug 随标题改变。直接复取固定 V9 后源码与 chyibo 整份相同；V8 的源码 SHA 与之不同。此处不把 chyibo 的分数迁移给 alexchilton。[(上游) alexchilton](https://www.kaggle.com/code/alexchilton/name-that-llm-0-663-all-public-what-failed) · [旧 SRC-002](../../../tracking/研究索引.md#src-002)

## 作者说明与实际得分路径

1. 在 Open Pool 文件存在时，源码选择 `ntl_open_pool.parquet`，按官方标签集合筛选，并在列存在时删去 `flag_train_prompt_regen` 和 `flag_gemini37_routing_risk`。保存日志报后一项删除 0 行，准备池为 **190,588** 行。页面 Input 概览为平台 V13、作者 v11。该数位于主 SVC 合并官方、matched 及池并规范化去重之前，不能据此填写最终拟合行数。[保存日志字段](platform-observation.json) · [固定池已有审计](../../../tracking/研究索引.md#src-065)
2. 主 SVC 保持四视图 LinearSVC，`svc_w=12` 是 log 分数的融合系数；官方、matched、池的训练行权重为 1、0.5、0.35。ATLAS 权重 0.20、候选 margin 阈值 0.3，互补栈权重 0.15。六个主成员保持 q14v5 / q8v5 / ll8v5 / n14b / q25_14 / deb。扩池会改变 SVC 拟合耗时与 logits，进而改变 ATLAS 候选行及能在预算内完成的行数。隐藏评分日志与覆盖 mask 未取得。[源码](source.ipynb)
3. 最终命令传 Sinkhorn 0.60、target mix 0.35。独立单元断言锚点唯一并把主推理脚本的 Bayesian typography 系数改为 1.2。在常规非模拟路径，V40A 在标点修正后拟合家族内风格残差，最大 alpha 0.08，按家族保留概率质量。成功时它的 argmax 写回最终 submission，失败时回退到残差前预测。保存日志确认补丁生效，未能从隐藏日志证明残差在那里成功运行。[源码](source.ipynb) · [页面与日志字段](platform-observation.json)

## 与已有固定版本的区别

| 对照 | 来源分数 | 真正有区别的部分 |
|---|---:|---|
| EXP-047 自有 V3 | 0.664 | 原 v5 池、svc_w 1.5、Sinkhorn 0.30、标点 1.5；chyibo 同时换池、svc_w 12、Sinkhorn 0.60、标点 1.2，并增加家族残差。主模型清单、ATLAS20%与互补15%相同。 |
| EXP-004 正式基线 | 0.662 | 原 v5 池、svc_w 1.5、Sinkhorn 0.45、标点 1.5，无该末端家族残差。chyibo 使用公开 Llama 模型输入，且容器摘要不同。 |
| SRC-036 One Swap V1 | 0.664 | 同有 Open Pool、svc_w 12。One Swap 的过滤路径只检查重建标记，Sinkhorn 0.45、标点 1.5，无 V40A 末端残差；其池历史字节不能按相同 ref 默认等同于本条。 |
| SRC-040 Apex E V1 | 0.663 | chyibo 新增一格绘制历史分数的图，输入单元仅多检查 `KAGGLE_IS_COMPETITION_RERUN`，其余八个代码单元逐字相同。触发完整推理时共用配方。 |
| synthreaper Sovereign Gold Apex V1 | 0.665 | 七个非主脚本的有效代码单元逐字一致。另一 V1 在主脚本内直接写 1.2，chyibo 通过补丁取得相同字节；比较包含最终命令及回退路径。 |

[EXP-047](../../../tracking/records/experiments/EXP-047.json) · [EXP-004](../../../tracking/records/experiments/EXP-004.json) · [SRC-036](../../../tracking/研究索引.md#src-036) · [SRC-040](../../../tracking/研究索引.md#src-040) · [synthreaper](https://www.kaggle.com/code/synthreaper/ntl-sovereign-gold-apex) · [比较哈希](receipt.json)

## 输入公开性与可复核边界

15 个数据集、8 个上游 Notebook、8 个模型版本的 ref 集合与 SRC-040 及 synthreaper V1 相同，两份 0.665 源码使用同一 Docker 摘要。匿名 Kaggle 接口逐项返回匹配页面和公开状态，8 个模型实例的版本号都与请求的版本 1 一致。源码没有引入单独训练的新 LoRA 或一组 chyibo 独有的数据。模型名称相同不保证历史附件字节相同：除 Open Pool 的 Input/保存日志证据，其他数据和 Notebook 的历史版本没有逐一绑定；当前版本清单仅表示采集时的可见性。所有 31 项输入均未下载权重或执行推理。Open Pool 标签 CC BY-SA 4.0；其余附件各有许可条目。[逐项匿名访问核查](public-input-check.json)

## 对当前方案的用途与验证

这次 0.665 支持把 Apex E 作为一套完整的对照配方。它与已有 Apex E 得分的差异既可能涉及附件更新，也可能涉及时间预算和隐藏执行覆盖。本次没有同输入的逐行隐藏预测，无法把相对 EXP-047 的 +0.001 归给任一组件。下一步可复用 EXP-049 的后处理缓存与 EXP-050 的池审计，先对齐输入、成员及预算，再单独测试池、svc_w、0.60/1.2或家族残差。每次固定其他条件并记录 ATLAS 选行、完成行和成对改判。现有资料归档没有启动训练或正式提交。

## 更新记录

- 2026-11-02：固定得分源码，核对 Apex E 与 alexchilton 的版本关系、保存阶段 Open Pool 数量和输入可见性。该条源码级推理配方与另一条 0.665 同源。

## 同配方的固定得分版本

Alex V9（SRC-069，0.663）与 chyibo V1（SRC-068，0.665）整份 Notebook 字节相同。旧 Apex E（SRC-040）和 Alex 的输入卡均为 Open Pool 平台 V11，公开保存时各写出 190,210 行。两份 0.665 使用 V13，均写出 190,588 行。四个版本公开产出的四份推理脚本哈希一致。隐藏评分的完整输入、ATLAS 覆盖与逐行预测未取得，分差无法按组件分摊。[新版本输入与日志](../../../tracking/snapshots/20261102T071145Z-public0665/kaggle-browser-observation.json) · [旧版本输入卡](../../../tracking/snapshots/20261102T071145Z-public0665/historical-input-card-observations.json) · [V1 脚本比对](../../../tracking/snapshots/20261102T071145Z-public0665/fixed-v1-module-comparison.json) · [Alex V9 脚本回执](../SRC-069-alexchilton-all-public-v9/receipt.json)
