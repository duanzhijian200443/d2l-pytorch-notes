# SRC-040 · Apex E V1：Open Pool、SVC12与V40A多项组合

原始记录：[正式台账](../../../tracking/records/sources/SRC-040.json)。
状态：已读取固定源码、完成静态对比并绑定 V1 页面成绩；正式编号已登记。

## 值得保留的结论

Apex E 同时组合 Open Pool、SVC 权重 12、Sinkhorn 0.60、标点系数 1.2 与 V40A 家族内风格残差。它是多个公开改动的组合，不能把页面表现归因于其中某一项，更不能把正文中的“0.670 pipeline”当成平台成绩。

与我们当前 ATLAS20 参照相比，其研究价值在于：给出了 Open Pool 有条件风险过滤的具体路径，以及保留家族概率总质量的残差实现。两者都仍需独立验证。

## 来源与原件

- 来源：[NTL Apex E](https://www.kaggle.com/code/bgunshui/ntl-apex-e)。本次用 Kaggle API 先取当前版本，再用 `version_label=v1` 固定拉取，二者源码一致。
- 采集：2026-11-01 07:32:08 UTC / 15:32:08 北京时间。
- [固定 source.ipynb](source.ipynb)、[API 元数据](metadata.json)、[采集与哈希](download-receipt.json)、[原 Markdown](description.md)、[代码提取](code.txt)、[对 SRC-002 的逐行差异](diff-vs-SRC-002.patch)、[静态检查证据](analysis-evidence.json)。
- 主窗口在本轮（2026-11-01 UTC，逐页时刻未记录，采集截至07:35:57 UTC） 实读网页：Version 1 of 1，Public Score 0.663，Best Score 0.663 V1；V1 与 Copy/Edit 链接均绑定 **scriptVersionId=364347706**。保存 [主窗口 DOM 观察快照](snapshots/browser-observation-20261101.json)；[固定得分版本](https://www.kaggle.com/code/bgunshui/ntl-apex-e?scriptVersionId=364347706)。正文的 0.670 声称不构成得分证据。

## 作者声称与实际设置

正文称其从 synthreaper 的公开方案出发，叠加“一次替换”方案的 Open Pool/SVC 12，以及 judith007 的 0.60/1.2。后者对应已存 SRC-022 的 V6 组合，**不是已绑定 0.664 的 SRC-021 V2（0.30/1.5）**。

代码落实的变化：

1. 原始输入仍先定位 `nlm_pool_v5.parquet`，再寻找首个排序后的 `ntl_open_pool.parquet`。只保留官方 18 类；存在列时排除 `flag_train_prompt_regen` 和 `flag_gemini37_routing_risk`；转为 text/model 后写 `pool_open_safe.parquet` 供主 SVC 使用。缺少 Open Pool 文件时退回原池，因此不能只凭标题断言实际用了 198507 行。
2. 主四视图 `LinearSVC(C=0.5, class_weight=balanced)` 本身不变，`svc_w` 从 1.5 升至 12；源权重仍为官方 1.0、matched-clean 0.5、池 0.35，并逐文本规范化去重。Open Pool 替换只进入主 SVC 的 `--pool`；TF-IDF/文本变形仍使用既有 `FIT_T`，不能描述为全部成员重训了新池。
3. Sinkhorn 0.45→0.60，target mix 仍为 0.35；通过字符串锚点断言将 `infer.py` 内标点证据系数 1.5→1.2，clip 范围仍为 ±0.30。
4. 新增 `sibling_style.py`：标点、常用词与句长等风格特征；在 Claude/Gemini/GPT/GLM 家族内，按 prompt 的 5 折 GroupKFold 测 balanced accuracy；LogisticRegression 的 C 分别 0.003/0.003/0.003/0.03。可靠度为 `clip((BA - chance)/0.15, 0, 1)`；再按原家族内 margin 门控，最大混合强度 `max_alpha=0.08`。保持各家族 logsumexp 总量，并有断言；失败时退回残差前预测。调用在 Bayesian typography 之后。
5. ATLAS 20%、互补栈 15%、原有 LoRA 主成员及大多数基础流程沿用。Llama 输入增加两来源回退；alexchilton 分支仅检查分片存在，Meta 分支额外检查大小。FAST_COMMIT 仅检查测试行数小于 50，未保留 SRC-002 的 `KAGGLE_IS_COMPETITION_RERUN` 条件。

正文末尾仍写旧的 Sinkhorn 0.45，而实际命令是 0.60；参数判断以源码为准。

## 我们核查了什么

- 保存固定 V1 原件，code SHA256 为 `176028c830935d9ed88a36e659fd55294cc07ccf398d874161f20ff7207e93e6`。未与当前正式 SRC 找到相同代码哈希或相同来源/固定版本。
- API 输入清单确实包含 `crtproduction/name-that-llm-open-pool`，但不提供该项的明确数据版本；未下载/统计其全量内容，也未核验过滤后的行数。
- Notebook 内嵌 `metadata.kaggle.dataSources` 集合与 SRC-002 完全相同，和 API 所列新增 Open Pool 不一致；不能用继承的内嵌 ID 列表认定新数据版本。
- 下载源码 Notebook 没有执行输出；网页另显示 12 个输出文件、保存运行 385.7 秒（显示 6m26s）、T4×2。本次仅源码与说明核对，未执行 SVC、GPU 前向、正式提交或付费调用；保存时长不代表完整隐藏评分耗时，ATLAS 覆盖及完整运行成功未独立复现。
- 可关联 SRC-002/SRC-016（共同基础与 V40A）、SRC-022（实际引用的后处理组合）、SRC-027（Open Pool 讨论）；One Swap 已关联 SRC-036/V1、SRC-037/V3；固定池版本见 SRC-038。

## 对当前方案的用途与限制

这份组合可用于提出拆分实验，但不能证明 Open Pool、SVC 12 或风格残差分别有效。过滤依赖标记列的存在；列缺失时不会执行相应过滤，且过滤前后数据量尚无运行日志。公共池原有清洗是否利用全部官方标签、各来源重复与 prompt 关联仍需沿用项目数据审计。

风格残差的家族内 OOF 仅用于可靠度门控，并非最终 ATLAS 集成的干净 OOF 评价。它保留提供商总质量，只重排家族内身份；机制限制与代码断言清楚，收益仍待测。作者宣传的半时延、分钟内保存、0.670 都不能替代本版本的实际隐藏评分证据。

## 待验证想法

关联 IDEA-003/IDEA-002/IDEA-004：先固定 Open Pool 的确切版本与过滤结果，按统一 prompt 分组做池替换和 SVC 权重的拆分对照；在同一强模型预测缓存上再分别测试 0.60/1.2 和家族残差开关。记录每一步而非把五项组合直接用于归因；监测 SVC 额外耗时对 ATLAS 覆盖的影响。本次未启动实验。

## 更新记录

- 2026-11-01：下载 V1 并逐行对照 SRC-002，记录多项组合、数据回退、继承 metadata 及正文参数过期；主窗口已绑定 364347706/0.663，已正式归档。Open Pool 数据卡内容标签 v10 不等于 Kaggle 数据版本 10，主窗口另核实当前固定数据版本为 11；本 Notebook 的实际附件版本仍未独立绑定。

关联来源：[SRC-002](../../../tracking/研究索引.md#src-002) · [SRC-016](../../../tracking/研究索引.md#src-016) · [SRC-022](../../../tracking/研究索引.md#src-022) · [SRC-027](../../../tracking/研究索引.md#src-027) · [SRC-036](../../../tracking/研究索引.md#src-036) · [SRC-037](../../../tracking/研究索引.md#src-037) · [SRC-038](../../../tracking/研究索引.md#src-038)。

- 2026-11-02：固定 V1 的 Input 卡已核实挂载平台 V11（作者 v10，198,507 行）。公开保存时按 Gemini 3.7 路由风险过滤 8,297 行，写出 190,210 行。SRC-067 与 SRC-068 同配方保存时挂载 V13，写出 190,588 行，均绑定 0.665。Apex E 和这两份 V1 实际写出的四个推理脚本字节相同，主命令参数相同。各自的隐藏拟合规模及 ATLAS 完成行仍未知。[历史输入卡](../../../tracking/snapshots/20261102T071145Z-public0665/historical-input-card-observations.json) · [三份 V1 输出比对](../../../tracking/snapshots/20261102T071145Z-public0665/fixed-v1-module-comparison.json)
