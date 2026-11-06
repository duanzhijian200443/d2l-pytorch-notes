# 250题与显式单段协议的兼容性复核

本次只读审查EXP-021的原始250题及拟采用的共同格式后缀，没有调用生成接口、读取凭据、训练、改写原题或操作采集器。复核日期：2026-11-01。

## 结论

**拟定后缀可用于这250题，未发现必须修改原题才能消除的硬格式冲突。** 原题没有要求输出多段、项目列表、表格、代码块、逐行诗歌或多文件产物。每个题型均25题，train 200组、holdout 50组；明确的approximate word target应逐题保留。

核查由关键词筛查和逐条阅读全部250题共同完成，不能把关键词命中直接当作冲突。例如“dining table”“QR code”“a list of rulers”“numbered pages”描述的是题目对象，不是输出格式要求。

## 拟定后缀

> Write the requested response as one self-contained paragraph of plain text. Do not use a title, headings, bullet points, numbered lists, Markdown formatting, code fences, or line breaks. Include only the requested content, without any introductory or closing commentary from the assistant. Keep the approximate word target given in the task. If the task calls for an email, dialogue, steps, or calculations, express that content within the same paragraph.

建议保持这段共同后缀，不再额外增加固定开头、禁用普通标点、ASCII化、统一语气等约束。那些改动会抹去额外的模型风格信息，并改变当前“只检验格式协议对齐”的问题。

## 特殊题型与逐项判断

| 题型/题目 | 是否冲突 | 处理原则 |
|---|---|---|
| 25题short_email，包括cp0161–cp0180、cp0241–cp0245 | 无硬冲突；常见邮件排版会改变 | 保留邮件的收件关系、请求、感谢和礼貌语义，允许在同一段内表达。不要额外禁止称呼、落款或正文中的“let me know”等礼貌句；它们不一定是助理追加套话。原题没有要求主题行。 |
| 25题creative_short_fiction，包括cp0102、cp0117、cp0226、cp0229 | 无硬冲突；对话需内联 | 引号、人物话语、动作与叙事可放在一个段落内。不要把“单段”误实现成删除对话或把创作题改成摘要。 |
| cp0152要求将房间说明改写为clear sequence | 无硬冲突 | 用First/Then/Finally等自然连接词表达顺序即可；原题未指定编号列表。不能把禁止编号列表解释为禁止顺序说明。 |
| 25题reasoning_explanation，包括cp0182中间步骤、cp0188计数、cp0247座位数 | 无硬冲突 | 保留中间步骤、数值、单位与正确性；计算可内联。禁止代码围栏和独立公式段不等于禁止数字或普通数学符号。 |
| cp0223讨论诗歌中的silence/empty space | 无冲突 | 任务是解释诗歌效果，不是要求创作依赖留白的诗。无需保留多行诗体。 |
| cp0144、cp0154、cp0239明确要求paragraph | 一致 | 后缀加强既有格式要求，没有改变任务语义。 |
| cp0245要求虚构项目交接及工作文件位置 | 无硬冲突 | 允许在邮件正文内写虚构、通用的位置；不能把所有“文件位置”一律删掉或判成运行环境泄漏。 |

## 对齐的准确范围

- 两路使用逐字相同的`user`字符串：原题原文 + 固定分隔符 + 固定后缀；原始题干应另存或通过哈希可回溯。
- 保持`target_words`、`prompt_group`、train/holdout归属和模型请求标签，原题与变体属于同一语义题组。不能将变体随机分进另一个折。
- 保留当前最小基础指令控制条件时，不得把Kaggle请求描述为“无system”；Codex基础指令文件与API消息角色也不能宣称完全等价。
- 此协议对齐的是已查公共源的共同显式格式要求，不是逐字复现任一公共来源的全部协议。Lucas的4096输出上限、按每次生成抽取长度等设置仍不同；Prvs默认reasoning也与当前low不同。
- 输出长度仍是approximate target，不应把80/120/160/200目标悄悄改成硬阈值或根据输出挑选更合意的样本。若后续做格式合规筛查，应另存标记，保留所有原始返回和失败记录。
- 词汇、引号、em dash、冒号、正常数学符号等不应额外统一或删除。显式格式对齐不证明底层snapshot、完整上下文、采样默认值或实际token上限已对齐。

## 建议的离线核验

1. 原250题文件哈希保持不变；变体仍为250个唯一prompt_id、10类各25、200 train/50 holdout。
2. 每行可证明去掉统一后缀与分隔符后与原题逐字相同；元数据和原题哈希逐行一致。
3. 两路序列化后的实际user内容与同一变体逐字相同，尤其保留引号、真实换行和Unicode；不要把JSON转义文本误当题干发送。
4. 同一原题的重复编号、模型、来源和协议版本应进入独立记录键；变体不能覆盖旧1000条记录或冒充旧请求的断点续跑。
5. 检查是否存在重复追加后缀、截掉原题、目标词数双重冲突、train/holdout改变等问题。禁止把默认配置或离线测试视为已发起采样。
6. 将来验收输出时，分开记录原文字节与可见正文的格式判定，区分传输末尾换行和正文内部换行。加粗检测应允许跨行，已知cp0103反例不能再次漏计；不要误判小数、减法、普通引号或邮件礼貌句。

本次仅确认题干与协议兼容，未验证模型会遵守协议的比例，也未评价训练收益或API等价性。台账和其他配置由主代理维护，本复核没有修改这些文件。
