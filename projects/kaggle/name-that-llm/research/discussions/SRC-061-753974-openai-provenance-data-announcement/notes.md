# SRC-061 · Enai：9719条OpenAI生成文本的发布公告

原始记录：[正式台账](../../../tracking/records/sources/SRC-061.json)。
状态：已读取原帖与当前可返回评论；已归档，正式记录见下方链接。

## 值得保留的结论

公告称9719条、四模型且含exact prompts/设置/溯源；数据本体与生成文本许可本轮未独立核验。

## 来源与原件

- 原帖：[Kaggle 讨论 753974](https://www.kaggle.com/competitions/name-that-llm/discussion/753974)；作者 `enaimaan`。
- 采集时间：2026-11-01T07:32:15.438860+00:00；原帖发布：2026-10-29T23:25:13.996281900Z。
- [原帖与评论快照](snapshots/20261101T073215Z.json)；SHA-256 `ea4df16acf2e063f7680cbfcf701b1ba5e4f79af2fc6fef1f5935f0b1fcb6782`。这是公开只读 API 响应，不是独立实验验证。

## 作者声称与实际设置

公告发于2026-10-29 23:25 UTC，10月31日01:34 UTC有一条鼓励投票评论，没有补充技术证据。链接 [OpenAI Generated Responses](https://www.kaggle.com/datasets/enaimaan/ntl-openai-generated-responses)。

声称总9719条：GPT-5.6 Luna2341、Sol2338、Terra2336、GPT-6 Astra2704；4311条补已有社区prompt缺失类，5408条组成新增prompt面板，使用社区库与部分Dolly指令。称提供CSV/Parquet、prompt JSONL、精确system/user消息、调用设置、来源和许可通知。

## 我们核查了什么

本条读取的是发布公告，不是dataset版本；没有下载文本或模型权重，也没有验证逐行模型返回ID、完整调用日志、prompt授权和去重。若本轮另有dataset条目，应关联它并将详细数据审计留在该条，避免复制主体。

## 对当前方案的用途与限制

SRC-027的Open Pool v10作者明言暂未加入一份新的OpenAI-response数据，因为生成文本许可未建立；应核对是否为本数据及具体数据卡，不把有license notice等同许可已明确。不能把本公告称为新0.664方法或已证明增量收益。

## 待验证想法

先固定版本、文件哈希、上游prompt来源与模型/设置，再查与现有公共池的文本/prompt重叠；有明确许可与足够溯源后才作按prompt隔离的单来源加入对照。

## 更新记录

- 2026-11-01：读取原帖和当前评论，保存时间快照。新来源，正式编号已登记。 本轮未训练、未提交。
