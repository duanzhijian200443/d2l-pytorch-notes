# 已交接至EXP-040：用户将总并发改为3

21:11旧单并发控制器等待在途回答保存后退出，20条成功原文已保留，四模型各5条。它不再继续派发；后续剩余280条由EXP040以2工作进程续接，双渠道EXP038仍为1，因此合计3。

旧state保留原始LOCAL_ERROR：这是交接时预占尚未派发任务的目录，在attempt.mkdir处产生的FileExistsError，发生于request.json/Popen之前。闸门仅有handover-gate.json，0模型请求；没有取消模型请求，也没有将其当作已采样数据。原20次真实请求全部SUCCEEDED且正文哈希一致，旧PID已退出、锁已释放。原日志、状态、未完成审计和标记均保留，交接凭据见EXP040/recovery-receipt.json。

以下为本批原启动观察及设计历史。

# 已启动：北京时间2026-11-01 21:06观察

当前run_id为`2f5308a40ed74d748cf6354c77612627`，状态RUNNING，已保存6/300。四个请求模型均已保存首条原文；初始记录无格式标记、无重复job_key，保存正文与final.txt哈希匹配。后台继续单并发；现有EXP038仍在运行，快照合计2个采集请求在途，没有中断它。

这是首批运行核验，不是300条采集完成或底层模型snapshot证明。统一自动化15每15分钟汇报两项任务，任一结束后继续另一项，两项均终态后删除监控。

以下为已冻结的选题与执行说明。

# EXP-039 四模型Codex显式格式对齐增采300条

用户授权同时采集300条，并选择原四模型各75条：gpt-5.6-luna、gpt-5.6-terra、gpt-5.6-sol、gpt-6-astra。全部使用现有Codex订阅，low reasoning，最小基础指令与原CLI覆盖参数保持不变；不走API key、不新增付费API。

## 输入与分组

从EXP034已离线校验的250题中固定选75题，每个模型使用完全相同的75题。6train/类别，共60train；5类2holdout、5类1holdout，共15holdout。合计回答240train、60holdout。覆盖全部10类，每类7或8题；在各类train中覆盖原本可用的字数档，保留每题目标词数。种子20261101039只决定题目选择与顺序，不是模型采样seed。

已排除EXP038双渠道计划的10个原题ID，与其当前和未来请求均不重叠。75题的user文本逐字复用EXP034；prompt_group、split、base_prompt/hash和protocol_variant都保留。本批与旧1000条没有相同user文本+模型组合，但共享75个原始题组，属于新格式变体，不能当成75个全新独立题组；所有变体保持原train/holdout归属。

格式要求：单段纯文本，无标题、列表、Markdown、代码围栏或额外助理说明。这是显式要求对齐，不能据此声称完整运行上下文、输出上限、底层模型版本或裸API等价已经对齐。

## 执行边界

新批次硬限制1个并发；现有EXP038串行1在途继续运行，所以本聊天两项采集合计最多2个请求在途。新批次独立目录和空workspace，不修改、停止或重启现有采集器。

最多300次请求，目标300条，四模型各75。控制器不自动重试；失败、未知结果、限流/访问异常或本地持久化异常则停止新派发并保留现场。旧成功记录不重发。每条保留未经清理的最终原文、JSON事件、CLI参数、请求/输出哈希、分组和模型请求标签。格式违规、长度偏差、相同输出只标记，不删除、改写或补采。

模型标识可能仅有requested_only证据，后台snapshot未知；Codex内部有界重连行为不等于控制器重试策略。单次超时180秒，批次最长6小时。完成须同时满足当前run_id终态、300条、四模型各75、75题组/240与60分组、完整性审计。采集完成不等于训练有效性验证。

## 位置

prompts.jsonl和request-plan.jsonl固定本批输入；metrics/selection-audit.json保存选择、排除与查重结果。collection/state.json是实时控制状态，collection/records.jsonl为成功原文索引，metrics/collection-audit.json为最终审计。后台运行前核验冻结manifest和现有ChatGPT登录，不读取或输出凭据。不训练、不提交比赛、不公开数据。

官方CLI行为通过OpenAI Docs MCP核验：https://learn.chatgpt.com/docs/developer-commands#codex-exec 。
