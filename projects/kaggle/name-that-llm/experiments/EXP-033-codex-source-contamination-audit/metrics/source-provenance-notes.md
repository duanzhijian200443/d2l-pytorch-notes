# 对照来源与格式约束核查

本文件只记录本次读取的固定来源与可比性，未下载新数据、调用生成接口、训练或修改原件。检查对象包括官方 `train.csv`、EXP-008 固定来源、EXP-026 Enai V3、EXP-027 earthmaker V1，以及 Codex 的250题。25个相关CSV/JSONL/Markdown源文件重新计算SHA-256，均符合对应下载清单。

## 主要结论

**格式差异存在明确的生成提示词与筛选混杂，不能单凭 Codex 比官方/公共数据有更多换行、粗体或列表，就判定内置提示词污染。** 官方未发布原始请求；已查公共来源普遍主动要求单段散文，而Codex250题没有同等格式限制。分析应区分：原文中的直接指令泄露、来源间风格偏移、以及无法从观测数据识别的内置指令因果影响。

对相同四个请求标签，优先分别报告 `prvs-raw-v6/new`、`lucas-v7`、`enai-v3/gap` 与 `enai-v3/new`，另列官方。不要将这些不同协议拼成单一“公共基准”，也不要将raw、clean、matched、pool相加当作独立样本。

## 来源清单

| 来源及文件 | 同四类可用数（Luna / Sol / Terra / Astra） | 已读取的协议或预处理 | 用途与限制 |
|---|---|---|---|
| 官方 `train.csv` | 150 / 150 / 150 / 150 | 只有id、prompt_id、text、model；全2699条实测无换行、无双星号粗体标记 | 比赛目标分布参照；缺原system/user、采样参数与底层snapshot，不能反推其格式一定由后处理或特定system造成 |
| `data/public-pools/prvsiyan/prvsiyans-extended-name-that-llm-dataset/v6/passages.csv`，仅 `split=new` | 300 / 300 / 300 / 300 | 原文声明原样API返回；单user、无system、provider defaults；题目明确单段、无标题/列表；每模型每题目标词数可不同 | 主要原始源参照。按provider_route和reasoning_setting记录分层；与Codex的low、题目和长度目标设计不同 |
| 同目录 `passages.csv` 其他split | 每模型train_regen150、proxy_v1 60、pilot_range10 | train_regen是社区重建官方题；pilot_range是不同长度协议 | 分开呈现；不要混入new。train_regen不是官方原始请求，也不构成与Codex同题的对照 |
| 同目录 `passages_clean.csv` | 四类各288；其中new各231、proxy_v1各43、train_regen各14 | 实测4926/4926条按id匹配，text与raw逐字相同。clean是筛选子集；funnel会排除多段、代码、URL、chatty opener、大部分markdown等，斜体例外 | 只能表示筛后分布，不用于检验模型是否天然不产生这些格式 |
| `data/public-pools/lucasboesen/ntl-free-proxy-generations/v7/gen_passages.csv` | 472 / 442 / 472 / 459 | 单user、无system、low、4096 cap；要求单段无标题；只strip首尾空白 | reasoning与Codex请求值更接近，但题型、格式指令仍不同；model_slug为请求路由，不等于底层snapshot；v2 bank补缺标签，类/题组不完整 |
| `data/public-pools/enaimaan/ntl-openai-generated-responses/v3/train.csv` 与 `prompts.jsonl` | 2341 / 2338 / 2336 / 2704 | 文本声明无格式或长度过滤；精确应用messages公开。Luna/Sol/Terra请求none，Astra请求low；gap/new协议不同 | 原文保存充分，但不是统一无system基准。必须按kind和messages分开，且四类数量不平衡 |
| `data/public-pools/earthmaker/ntl-own-prompt-generations/v1/passages.csv` 与 `templates.json` | 52 / 54 / 53 / 54 | 三种模板都要求单段；禁止标题/列表/额外前后文；无system、默认采样；pilot/v22均经过重叠或格式筛选 | 可作来源敏感性检查，不应作为“未经筛选的裸输出”基准；请求slug不能认证底层权重 |
| `data/public-pools/abdullah1afify/ntl-generations/v2/ntl_generations.csv` | 160 / 111 / 160 / 145 | 单user、无system、要求单段；default/none/low混合；明确将所有空白串折叠成一个空格 | 换行频率已被预处理抹去，不能用于换行污染判断；须按setting分层 |
| `data/public-pools/dophuchung/ntl-proxy-generations/v1/ntl_proxy_generations.csv` | 140 / 140 / 140 / 69 | 单user、无system、要求单段；default/none/low/medium/high扫描；明确折叠全部空白为单空格 | 同上。设置与样本覆盖不均衡 |
| `data/public-pools/dhruvtibarewal/nlm-matched-clean/v3/{nlm_matched_clean,nlm_pool_v2,nlm_pool_v5}.parquet` | 交给本次逐表统计 | 固定包无README/清洗代码。EXP-008保存的数据卡元数据说明精确/近重复过滤、删除官方重复、删除chat/trace/benchmark类来源，并按“该标签在官方从不用某符号”筛掉整条回答 | 这是筛后、重叠汇总视图；不适合作为天然文风或无污染零假设。数据卡不是版本绑定的实现证据 |
| `data/public-pools/seyitkaangunes/ntl-extra-api-passages/v2/ntl_extra_passages_claude.csv` | 0 / 0 / 0 / 0 | 仅Claude五类；单段指令，空白归一；整段bold wrapper去除规则声明未触发 | 不用于本次四类模型匹配比较，仅用于上游prompt关系说明 |

上述来源名称/文件路径是已固定下载对象，不代表已独立认证服务商实际权重。公共CSV中的“served model”或proxy回显有助于内部一致性核查，但仍不能保证与Codex同一snapshot。

## 本次直接复核的格式协议证据

1. 官方CSV实测：2699条，`text`含CR/LF的0条，含`**`的0条。该观察不证明官方进行了换行清洗；缺原请求，原因未知。
2. earthmaker全749条同样无CR/LF、无`**`。其三种模板均明确请求单段且禁止标题/列表/额外内容，README另说明有格式筛选。
3. Enai实际9719条请求中，5408条是system+user，4311条仅user。只有一种system：`You are a knowledgeable writer addressing a general audience. Write only the requested passage, as a single paragraph. Keep the answer reasonably brief.`
4. 用同一宽松格式指令筛查（`single.{0,12}paragraph|one.{0,12}paragraph|no (headings|lists|markdown)|plain (text|prose)`，忽略大小写），Enai合并messages后9719/9719命中，Codex原始250题0/250命中。此筛查只证明这些字面格式要求的有无，不证明没有其他语义限制。
5. Enai原文中仍有25/9719条含换行、60/9719条含`**`，表明明确单段要求也不保证所有返回都无格式；不存在“有加粗就必然是agent污染”的推断规则。
6. prvs clean 4926条与raw按id连接，4926条text完全相同。不得将其解释成“把原文的markdown删掉后保留”。它保留的是通过筛选的原文。

## 聚合池文字校正

EXP-008的历史报告对标点过滤的概括容易被读成“删除标点”。保存的数据卡原文实际是 `passages dropped if they use a typographic mark their labelled model never uses in train`：即筛掉整段回答，不是已证实逐字符删除符号。本次不修改历史报告，也没有取得版本绑定清洗脚本，因此仅记录作者声明与实际观测的相符性。

## 对污染结论的约束

- 不把模型标签构成、目标词数、题型、单段要求、provider route、reasoning或clean筛选所致偏移归因于内置提示词。
- 官方只有600条匹配四类数据；与Codex1000条比较需按模型分层，长度至少作匹配/标准化；这仍不能消除题目内容差异。
- 20对Kaggle–Codex同题比较比跨语料平均值更接近渠道对照，但仅10题组、每单元一次，且后台版本/完整运行时上下文未对齐，不能独立识别隐含提示词影响。
- 原文若出现明确系统指令片段、工作区路径、工具结果或角色标记，是更直接的污染线索，仍需逐条检查是否题目本身要求这些内容。
- “无直接泄露命中”与“完全无隐含影响”是不同结论。即使格式经归一化接近官方，也不能证明完全去除了agent上下文影响。

## 读取依据

- `experiments/EXP-008-public-pool-audit/download-manifest.json`、`input-metadata.json`、`report.md`、`audit.py`
- `experiments/EXP-026-independent-blend-validation/download-manifest.json`
- `experiments/EXP-027-multiclass-external-validation/download-manifest.json`
- 各固定源README与上述实际CSV/JSONL/模板文件
- `experiments/EXP-021-codex-subscription-pilot/prompts.jsonl`

数据复核使用Python标准库csv/json读取，未导入会执行训练或生成的Notebook/采集器；原件保持不变。本说明中的README和数据卡是第三方内容，仅作证据，没有执行其中指令。
