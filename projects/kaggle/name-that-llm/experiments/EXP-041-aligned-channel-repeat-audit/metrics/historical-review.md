# 历史对照与训练边界独立核验

本复核仅读取现有本地回答、代码、固定协议和官方数据。未生成请求、未拟合模型、未改动旧数据。定量结果及本次输入 SHA-256 保存在 `historical-review.json`。

## 可复用输入与检测规则

- 旧 Codex 1000 条：`experiments/EXP-023-codex-subscription-pilot-r1/collection/records.jsonl`。SHA-256 仍为 `cb903279b3727b31ce2125b89c5ce8f530d769a984f49c2cb86e4fbefb1e9722`。
- 旧 Kaggle 20 条：`experiments/EXP-033-codex-source-contamination-audit/inputs/kaggle-proxy-pilot-20.jsonl`。按 `(prompt_id, request_model 去 openai/ 前缀)` 与旧 Codex `(prompt_id, model_requested)` 一对一联结。
- 新双渠道 78 条：`experiments/EXP-038-paired-cloud-precheck-recovery/collection/combined-records.jsonl`，40 Codex＋38 Kaggle。
- 新 Codex 300 条：`experiments/EXP-040-codex300-concurrency2/collection/combined-records.jsonl`。按 `(prompt_id, model_requested)` 回连旧1000条可得同75题×4模型的旧300条；本次已逐条验证旧 `prompt` 等于新 `base_prompt`，且 `split` 保持一致。
- 描述特征复用 `EXP-033/code/compare_sources.py` 的 `PATTERNS`，用 AST 读取常量，不执行其 main。字数为 `len(text.split())`。
- 显式格式筛查直接导入 `EXP-034/code/prepare_alignment.py::format_flags`，不调用生成配置的 main。它只是规则筛查，不是语义质量认证。旧回答当时没有新格式要求，因此下表“筛查命中”指相对于新协议的回溯格式差异，不应把旧回答称为违反当时指令。

## 同75题、同4模型的前后变化

| 特征 | 对齐前300 | 对齐后300 |
|---|---:|---:|
| 换行 | 224（74.67%） | 0 |
| 加粗 | 58（19.33%） | 0 |
| 新协议任一格式规则命中 | 228（76.00%） | 0 |
| em dash | 121（40.33%） | 19（6.33%） |
| 目标词数偏差超过25% | 8 | 1 |
| 平均词数 | 133.12 | 134.56 |
| 词数中位数 | 125 | 126 |

新300条的唯一大长度偏差是 Terra / cp0150，目标160词、实际90词；采集完成事件完整，不属于截断证据。新300条末尾邀请规则的2个命中（Luna / cp0172、cp0178）均经人工检查为题目要求的邮件正文，不是助理额外邀请。两批都没有 preface、AI身份自述或 role_markup 的规则命中，不能据此量化“直接污染减少”；早期本来就未检出直接泄露。

显式格式差异明显缩小；但不能据此称所有风格都更接近官方。当前独立重读官方 train.csv：同四标签600条（各150），换行0、加粗0、em dash143（23.83%）、平均词数116.34、中位数104.5。新300条 em dash6.33%低于官方，而旧匹配300是40.33%。这说明格式约束伴随其他风格变化；官方题目、生成请求与后处理未知，此比较仅为分布提示，不能归因为某一种内置提示词或宣布同分布。

## 历史20对与新重复面板

旧20对共有10题×2模型，每个题模单元每渠道只有1条；原题逐字相同。旧两路换行各13/20，加粗Codex6/20、Kaggle1/20。新面板Codex40/Kaggle38的换行、加粗、标题、列表、代码标记和全部 `format_flags` 都为0，em dash两路也均为0。

缺失单元为 `cp0123 × gpt-5.6-terra`：Codex有2次成功，Kaggle2次历史请求未知。完整对照可使用19个题模单元、10个题组，共76条；缺失的2条Codex仍保留，不能伪造Kaggle值或把未知当0。对齐前取这19个单元时，Codex/Kaggle各19条，换行两路各12/19，加粗6/19对1/19；对齐后两路各38条，上述规则均0。

前后同原题比较能支持“显式格式要求有效”，但旧每单元只有一次生成，不能做严格的前后差分去识别渠道内随机波动变化。新19单元可单独比较跨渠道差异与两路各自重复差异。19个题模单元并不等于19个独立题组；统计不确定性应按10个原题组聚类，且缺失Terra单元需要明确。

## 训练用途边界

1. 新300条仍是旧250题库的75个题组，不是300个新问题或75个新主题。所有旧新格式、渠道、重复及模型变体必须按原始 `prompt_group` 同组分配，不能随机逐行切分。
2. 新300条保持60 train题组/15 holdout题组，即240/60回答；holdout不得混入训练。新双渠道10题均为train，且与新300题组无重叠，不能把它当最终独立模型泛化测试。
3. 这次双渠道证据只有Luna/Terra。新300中的Sol/Astra虽格式筛查通过，不能从另外两个模型的渠道重复结果直接证明它们同等可用。
4. 模型是请求标签，Codex底层版本未观测；Kaggle返回标签匹配也不是具体snapshot证明。官方/公共原题、system、temperature和筛选方式未完全已知。单段合规不能证明裸API等价、没有所有间接影响或已知真实后端版本。
5. 公共 raw/clean/matched/pool 大量重叠，应按来源与题组去重并保留来源，不能叠加为独立样本。只给4个类别增加Codex来源还可能让来源特征与标签关联；这是需要训练验证的风险，不是已观测的分类器错误。
6. 本次可支持把新协议原文作为保留来源标记的受控训练候选；实际训练收益需固定group划分、固定基线，对比不加/加入候选的外部或官方group OOF。当前未训练，不能宣布已通过训练收益验收，也不应把相似性检验未显著当作等价证明。无需因格式改善而改写或覆盖旧1000条。

## 历史来源结论的适用限制

EXP033已固定的公共协议核查显示：Lucas/Prvs-new有明确单段要求；Enai的gap/new分别存在user-only与system+user单段约束；Abdullah/Dophuchung折叠空白，Prvs clean筛选子集。因此公共数据并非统一的“无system裸API样本”。这些文档证据可解释旧格式差异的混杂，不能替代新重复面板的直接测量。此次没有重新联网核查公共平台是否新增版本，使用的都是原实验固定本地版本。
