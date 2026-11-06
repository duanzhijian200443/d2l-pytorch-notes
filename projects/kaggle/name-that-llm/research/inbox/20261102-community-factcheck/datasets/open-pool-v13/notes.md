# Open Pool 平台 V13 / 作者 v11 修订：待导入核查

状态：PENDING_IMPORT。仅采集资料和读取元数据；未训练、提交或修改云端任务。关联旧资料 SRC-038（平台 V11 / 作者 v10）；正式编号由合并窗口分配。

## 来源、固定版本及采集

- [平台 V13](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool/versions/13)；[比较版本 V12](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool/versions/12)。
- `api-info.json` 是官方 Kaggle 已认证 Python SDK GetDataset 原始响应；`dataset-metadata.json` 是该 SDK 最新数据卡响应，**不是独立取得的历史 V12 卡片**。采集时 currentVersionNumber=13。
- `files-v12.json`、`files-v13.json` 是固定版本文件清单。`files/` 为 V13 八个小 CSV 原件。`range-fetch-log.json` 保存每次 HTTP Range 读取的版本、普通来源链接、UTC时间、字节范围和 SHA256，不保存签名 URL。
- 后续动态暴露 Kaggle MCP 后，已实际调用 `get_dataset_files_summary`，19文件/3 parquet 的结果见 `mcp-files-summary-v13.json`。早期未认证 REST view 请求返回404；使用现有已认证 SDK成功，无重新登录。
- 每个 parquet 只取末尾 65536 字节，通过已安装 thrift 库解析 Parquet FileMetaData（`inspect-footers.py`）；这核实文件元数据，不等于逐行审计。`parquet-metadata-check.json` 为结果，尾部原始字节保留。未下载整个主池。

## 作者声称与我们实际核查

1. **版本与时间核实。** API history：V12 创建于2026-11-01 20:37:38.553Z，即北京时间11月2日04:37:38；notes标“v11”。V13创建于21:34:18.227Z，即北京时间05:34:18；notes为“v11 fix: flag_kb_proxy corrected on 5,501 rows ...; no rows changed”。因此作者发行名 v11 不能作为 Kaggle 下载版本号。
2. **主池190588行已获得文件级支持。** V12及V13 parquet footer都为190588行、23字段、1个row group；V13 `per_model.csv` 18标签的 passages之和与 `sources.csv` 55来源的 passages之和均为190588。隔离文件footer为10593行；prompts footer为13021行。仍未验证每行正文/标签、去重和来源真实性。
3. **增删关系算术一致，但迁移身份未逐行验证。** 作者称198507 + 10294 − 10593 − 7620 = 190588。新增10294由`v11_append_log.csv`终行及数据卡支持；10593由隔离文件footer支持；7620是本次增量删除，不能与`removal_summary.csv`累计source_unavailable=9815混同。未取旧主池逐行join，不能证明所有新增/删除ID迁移恰好符合这组数。
4. **V13的5501 flag修正与正文不变仍是作者声明。** 固定历史说明明确如此，数据卡拆分为alperen Vertex 2573、arxitekton OpenRouter 2846、earthmaker OpenRouter 82，和为5501。V13 per_model的kb_proxy和为69745。只验证两版row count/schema相同，不凭文件体积或footer推出正文一致；没有逐行比较V12/V13。
5. **Gemini 3.7 “Vertex-only”是文档溯源过滤，不是服务端模型鉴定。** 卡片称仅保留有来源文档支持的Vertex路线，主池3278行/4来源；无证明路线的10593另存隔离文件，未删除。卡片也明确labels为发布者声明、路线来自各来源自己的文档。我们没有独立认证实际served model。
6. **prompt工具存在，但分组仍需审计。** 主池footer确认`prompt_canon`、`panel_labels`、`prompt_evidence`、`match_weight`列。`panel_gaps.csv`实际4472行，panel_labels范围10–17；来源卡片称592组覆盖18标签（本次未逐行核查prompts）。`p`跨来源、`s`仅来源内、空值未知；centroid匹配作者报告约2%误配。不能把该列直接当完美prompt身份真值。

## OOF与训练使用边界

- 卡片建议`match_weight`用于类内抽样，不应未经试验直接当loss weight。发布列使用全部官方train拟合；官方train上的验证会偏高。严格prompt OOF应每折仅用训练prompt估计权重，并将全部同源/改写/重复prompt归在同一折。
- **仅重算权重仍不够恢复独立性。** 卡片称沿用v10质量过滤；`removal_summary.csv`含1278条typography_mismatch；新版新增审计`v11_append_log.csv`明确1024候选因“typography the label never uses in train (em dash)”删除，并有264条因与train内容cos>=0.3删除。清洗已接触官方train统计/内容，重算权重无法恢复预先删去的行。若要声称严格独立OOF，须从足够原始候选按折重做相关筛选，或另设从未参与筛选/权重拟合的外部验证集；否则明确是已受上游选择影响的开发指标。
- **未核实所谓标签classifier/suspicion另一路筛选。** 新卡片有style-only train-vs-pool分类器`style_score`描述，但明确不按style_score筛选。现有证据只支持上述逐标签标点/内容接触风险，不能凭旧印象补写为已证实的额外标签分类器剔除。
- 45k等量、Qwen3-1.7B、两seed、单项变化的实验为作者报告。实际下载10行`dataq4_runs.csv`（B/P/T/G/X × seed0/1）与20行`dataq4_paired.csv`，没有训练日志、预测原件或独立复跑。作者明确新版全组合与新增10294行未参与这些消融；不能外推为全量强模型或本项目ATLAS20收益。
- T报告整体balanced delta +0.0263，G +0.0093；两答案场景T/G区间都跨0。CSV中的整体delta并不等于所列两个seed delta的简单平均，例如T seed0/1为+0.0134/+0.0219。文档未在本次材料中明确该估计量关系，因此保留原值、不称作简单两seed平均增益、不据此重算显著性。置信区间作者明确不包含训练随机性。

## 对我们有用与下一步

可作为公共池候选版本固定到V13，优先核查与现有训练数据/验证集的文本及prompt祖先重叠、来源许可与路线字段，再设计等量和全量独立对照。现阶段证据足以确认版本更新、文件结构及规模，尚不足以宣布模型收益或整个池训练安全。正式导入应保留SRC-038历史结论，本条关联新版而不是覆盖旧版。

完整哈希、范围与限制见`manifest.json`和`factcheck.json`。未运行全局ledger check/render：本子任务不改台账，交由合并窗口完成。
