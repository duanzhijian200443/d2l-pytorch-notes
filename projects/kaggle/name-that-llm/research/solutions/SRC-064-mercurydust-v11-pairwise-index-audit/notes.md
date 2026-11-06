# SRC-064 · Mercurydust V11 / V34.1：配对修正与批次索引缺陷

已归档；正式索引：[来源记录](../../../tracking/records/sources/SRC-064.json)。下方保留采集时的范围、证据及交接说明；其中PENDING_IMPORT描述的是原始交接阶段。

# Mercurydust 平台 V11 / 正文 V34.1：配对分支核查

- 来源：https://www.kaggle.com/code/mercurydust/name-that-llm-submission
- 核查时间：2026-11-02 约 12:00 Asia/Shanghai。
- 角色：待导入资料；关联旧 SRC-017（平台 V3），没有分配正式新 SRC，没有改台账。
- 已使用 code-review 技能，按实际调用链核查。仅取原件、元数据及运行隔离的合成微测试。

## 结论与问题

**[P1] 第二批起配对特征混用首批文本。** `code.txt:2068-2069` 对完整测试矩阵索引 `Cte[q-start]` / `Wte[q-start]`，`code.txt:2073-2074` 对完整文本数组索引 `test_texts[z-start]`。上游 `q=np.repeat(np.arange(start,stop),R)` 已是全局测试索引（2066），而 Cte/Wte 由完整 C[n:]/W[n:] 定义（2003），从未被批次重新赋值。故 start=128 时第 129 条取第 1 条字符/词向量、长度和标点。参考选择 `Cte[start:stop]`、A 特征 `a_prof[n+q]` 和 B 属性 `b_test[q]` 仍取正确行，构成同一 pair 内的特征混用。

这段代码直接构造 `XF` 并送入 `verifier.predict_proba`（2076-2077），不是未调用 helper 中的疑似问题。旁边定义的 `feats()` 未被主训练/推理调用，本 finding 不依赖它。

隔离 AST 微测试实际执行该原始 for-loop：

| 测试条数 | 与仅修正索引的 comparator 差异 |
|---|---|
| 3 | 无 |
| 128 | 无 |
| 129 | 第129条，第0/1/8/9四列 |
| 257 | 第129至257条，同四列 |

第129条示例：原版 char/word=1/1，修正版=0/0；长度比约1变0.2；标点相似度1变0.375；A/B 六列保持相同。模型接口为记录特征的 stub，无拟合、无GPU、无真实预测/分数评价。公开三条示例不会触发此错误；本微测试证明特征来源错误，不证明具体 LB 降幅或最终标签必定变化。

## 作者主张与实际代码

正文标题是 V34.1，平台固定版本是 V11；固定 V11 与当前 latest 源码和代码哈希完全一致。作者声称以原 V31 / Public LB 0.660 为基线；这不是 V11 新分数绑定。正文明确 0.10 权重和 0.35 gate 是初始值，完整 GPU 运行和 LB 仍待测试（description.md:30）。本次没有获得绑定 V11 的 submission/scriptVersionId/新 LB。

实际分支以 A 的 exact evidence quote 建特征原型；B 是任务/领域/结构/显式程度四项弱标签，经 prompt-group OOF 分类。用同模型跨 prompt 正对、同 prompt 不同模型/同家族不同模型/跨家族负对，训练 `LogisticRegression(C=0.35,class_weight='balanced')`。推理为每类取6条参考，经10维 pair 特征判同模型概率，再取 top-3 均值组成18类分数。

**0.10 是对中心化 log probability 的门控修正系数，不等于固定10%概率平均。** 在 Bayesian Typography 后，gate = clip((0.35 - top1_top2_margin)/0.35,0,1)，最终 logits = base + 0.10*gate*centered_log_pair_prob（code.txt:874-892）。它也不是强制每组18模型一一匹配/匈牙利分配。

## 附件公开性与版本

已在 Kaggle MCP `search_datasets(group='Public',user='mercurydust',search='NTLLM-ChatGPT-A_B')` 返回中找到精确资源；官方 SDK 的 metadata 与首20文件列表亦可读。因此目前有明确公开检索证据，不能继续称“公开性未知”或“附件缺失”。

- 资源：https://www.kaggle.com/datasets/mercurydust/ntllm-chatgpt-a-b
- datasetId 12958299；平台 V1，卡片自称 Revision03（两种版本号不可混用）。
- lastUpdated：2026-10-30T01:53:51.670Z；32,053,319 bytes；licenseName = Unknown。
- 卡片称 2699 answers / 150 prompt groups / 36703 evidence spans。此处仅引用作者卡片，不是本次逐行重数结果。
- 未下载整份附件，未核对根 manifest、所有必需 CSV、准确行数或 canonical train 全量对齐。源码有这些检查不等于它们本次已通过。

## 版本日期与哈希

固定 V11 与 latest 的 source SHA256 均为 `5335c55713f17f5cebf7b262ce7bfac9b86ccb79366d3cb8c1c4907628b020d8`；code SHA256 均为 `d7d0a8a91052183c9d6677c7eebd106540c259053270680939ad2b7cef575f23`。

API metadata 没有可核实的版本发布时间或 scriptVersionId。`lastRunTime` 返回 2026-10-30T03:59:25.433Z / 03:59:28.136Z，两次响应值不同；字段本身是运行时间而非发布时间，不据此认定 V11 今天发布或哪天发布。观察时间与发布时间分开。

## 对我们有什么用、怎样验证

可作为 A/B 弱标注和 pairwise residual 的独立研究候选，不可当作已验证榜单提升。先修正全局索引并做批次边界一致性检查；再固定 prompt-group 外层验证、在训练折内确定特征/原型及权重，区分跨折辅助预测与完整 pair verifier 的 OOF。现代码特征词表包含全 train+test，仅凭辅助 OOF 不足以宣称严格非传导端到端泛化验证。修 bug 与改变权重应分别验证。

## 文件与未做事项

- source.ipynb / code.txt / description.md：固定 V11 原件与提取件；latest-*：独立 latest 获取。
- metadata.json / latest-metadata.json / fetch-receipt.json：原始公开元数据与哈希。
- attachment-metadata-check.json / attachment-public-search-mcp.json：小型附件查询证据。
- minimal_batch_index_proof.py / minimal-proof-result.json：可重跑隔离证明。
- factcheck.json / manifest.json：核查摘要和 PENDING_IMPORT 清单，非权威台账。
- 未完整执行外部 Notebook、未训练、未用 GPU、未正式提交、未下载大权重、未修改现有源码/台账。
