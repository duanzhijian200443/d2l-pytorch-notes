# SRC-062 · Enai V3：9719条OpenAI文本的数据卡与许可边界

原始记录：[正式台账](../../../tracking/records/sources/SRC-062.json)。
状态：已读取数据卡与固定版本文件清单；未审计行内容；资料已归档。

## 值得保留的结论

这是四个GPT标签的候选公共数据；需要统一prompt祖先分组并核对生成文本权利状态。不能把exact messages误认为主办方原始system prompt，也不能把新增面板误认为新主题。作者明确尚未建立下游精度收益。

## 来源与原件

- [数据版本V3](https://www.kaggle.com/datasets/enaimaan/ntl-openai-generated-responses/versions/3)，作者Enai / `enaimaan`；关联发布讨论753974。
- 采集于2026-11-01 07:34:34 UTC（北京时间15:34:34），使用Kaggle MCP的`get_dataset_info`、`get_dataset_metadata`和指定版本3的`list_dataset_files`。
- [原始工具响应快照](snapshots/20261101T073434Z.json)，SHA-256 `3a4e34109298c54a923a0b7a0665aeb1b4dfd63359a91698b3f5ebe3fa763dee`。
- 数据ID12956120，V3创建于2026-10-29 23:03:14 UTC；V3说明为修订下载校验和覆盖范围。当前description是采集时数据卡，并非本轮下载的固定README字节。

## 作者声称与实际设置

数据卡称9719条：Luna2341、Sol2338、Terra2336、Astra2704；4311条补社区prompt缺失模型，5408条组成额外面板，共2739个prompt组。Luna/Sol/Terra reasoning为none，Astra为low，其余sampling留给服务默认。组内不同模型的目标字数可能不同。

精确messages是本次应用发出的请求，不是恢复的比赛system prompt；gap请求可能由社区模板重构，历史发送字节未验证。新增请求使用作者候选system指令。提供`cv_group`、`prompt_group`、`prompt_hash`和祖先别名映射；hash是规范化来源body，不是每条完整请求的hash。

## 我们核查了什么

固定V3清单有10文件：train.csv、train.parquet、prompts.jsonl、lineage-aliases.json、README.md、LICENSE.md、attribution.json、SHA256SUMS及两份许可证文本。API总大小27,005,758字节。本轮仅读卡片和清单，没有下载这些文件、核验SHA256SUMS、读取行内容或验证9719条统计。

卡片将Kaggle许可标成Other；汇编/metadata/原创指令仅在作者持有权利范围内提供CC BY4.0，生成响应拟采用CC BY4.0但当前审查状态须看LICENSE.md/response_license；上游Dolly保留CC BY-SA3.0，不能把全包视为统一CC BY。我们没有将卡片中“verified serving label”提升为独立服务商证明。

## 对当前方案的用途与限制

与SRC-027公开池讨论关联：该池曾排除生成文本许可未建立的OpenAI-response来源；此处数据卡也明确保留未验证权利状态。完整许可判断和逐行审计都仍待完成。合并时应将别名映射同时应用于上游数据，按canonical cv_group切分；保守分组仍不能证明无语义重叠。

## 待验证想法

先取固定V3的LICENSE、attribution与校验和，再核行数、返回模型标识/设置、祖先别名、官方prompt关联与重复；通过后再做按prompt隔离的单来源加入对照。本次未生成文本、未训练、未提交。

## 更新记录

- 2026-11-01：保存数据卡、metadata和V3文件清单；关联公告而不复制其正文，行内容未核验。
