# EXP-034 · 显式格式要求对齐

**配置对齐与离线核验已完成；尚未发起任何新生成请求。** 用户要求先对齐公共来源的显式要求。当前只落地共同格式约束，不推定新增采样条数、重复次数或API预算。

## 已对齐的内容

- 原250题逐字保留，统一在末尾追加同一段格式要求。
- 只输出所要求的正文；恰好一段纯文本；不使用标题、项目列表、编号列表、Markdown或代码围栏，不插入换行。
- 邮件、对话、步骤、计算仍保留任务本意，在同一段表达。没有额外禁止称呼/落款、引号、普通标点或邮件正文中的礼貌语。
- 目标词数保持原80/120/160/200设计；四模型标签和请求low设置沿用原控制条件，不改成各公共来源混杂的默认值。
- base_prompt_id、prompt_group、train/holdout均继承原值。全部变体、渠道与后续重复都应留在同一个原始题组中，不得随机拆行进入不同验证组。

统一追加文本：

> Write the requested response as one self-contained paragraph of plain text. Do not use a title, headings, bullet points, numbered lists, Markdown formatting, code fences, or line breaks. Include only the requested content, without any introductory or closing commentary from the assistant. Keep the approximate word target given in the task. If the task calls for an email, dialogue, steps, or calculations, express that content within the same paragraph.

## 文件与后续可用面板

- [完整250题新版本](prompts.jsonl)：每行保留base_prompt、其SHA256、原分组及protocol_version。
- [格式要求](format-instructions.txt)：唯一文本定义；两条渠道共用同一user字符串。
- [配置](config.json)：说明对齐范围、保留参数与仍不同的项目。
- [10题对照面板](pilot/prompts.jsonl)：复用此前Kaggle试采的10题，覆盖10类，全部来自train。
- [Kaggle请求预览](kaggle-request-preview.jsonl)：10题×Luna/Terra=20份JSON请求草案，**未发送**。
- [Codex面板配置](pilot/config.json)：同10题、同两模型、low、最多2并发；干跑目标20条，未创建collection目录。
- [离线生成脚本](code/prepare_alignment.py)：只生成配置/预览，不含网络请求或模型调用。`--check-only`校验输出与生成规则一致，不生成回答。

此面板不等于已授权运行。重复次数、两路实际总请求数、API费用边界尚未设定。没有启动新监控或后台采集。

## 兼容性与离线校验

250题逐条兼容性复核未发现必须分段、列表或代码块的硬要求。25封邮件、25个创作题、25个推理题可在单段内保留语义；cp0152仅要求顺序清晰，不要求编号列表。[兼容性复核](metrics/prompt-compatibility-review.md)。

原题字段、词数目标、200个train组和50个holdout组全部保留。生成文件逐字重算一致；API预览中的user字符串与Codex面板prompt逐字一致。已有1000条原文SHA256保持不变。

格式筛查函数用纯文本、邮件、引号对话、行内算术、跨行加粗、斜体、代码标记、编号列表和换行样例验证。输出检查只做标记，不自动修改、删除或重新生成；带有可能是正文内容的标记仍须按原题复核。不能仅靠正则证明语义上没有标题或助理评论。

[离线对齐结果](metrics/offline-alignment.json) · [校验及干跑](metrics/validation.json)。

## 尚未对齐、不能声称等价的项目

这是依据Lucas与Prvs等来源共同格式要求建立的新协议，**不是逐字复刻其中某个公共数据集**。

- 原同题对照的API最小system仍为`Answer the user's request.`；Codex使用同文本的基础指令文件。两者角色和完整运行时上下文不能保证一致，因此不称为“无system”。
- API预览保留1024 completion-token上限；现有Codex采集器没有对应的显式输出上限。没有为对齐格式而顺带更改公共Lucas的4096或我们的旧参数。
- temperature/top_p/seed继续省略，实际默认未知；模型底层snapshot未知。目标词数也没有改成公共来源的每模型随机抽样分布。
- 旧1000条继续标记原协议，不做事后改写或删除。后续新数据必须使用新协议标签单独保存。

## 来源

- `data/public-pools/lucasboesen/ntl-free-proxy-generations/v7/README.md`：单user、单段、无标题、low、原样保存，仅strip首尾。
- `data/public-pools/prvsiyan/prvsiyans-extended-name-that-llm-dataset/v6/README.md`：单段、无标题/列表；其他采样参数采用不同默认值。
- [EXP-033来源协议核查](../EXP-033-codex-source-contamination-audit/metrics/source-provenance-notes.md)。
- 本次已调用OpenAI Docs MCP搜索并读取[Codex配置参考](https://learn.chatgpt.com/docs/config-file/config-reference)；本次未修改全局Codex配置或旧基础指令文件。
