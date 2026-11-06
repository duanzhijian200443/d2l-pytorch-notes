# EXP-008：公共数据池检查

**结论：有可用数据，但不要将公共文件全部拼接。先使用来源清楚、能隔离官方 prompt 的小池做对照；大汇总池保留为后续独立实验。**

检查时间：2026-10-31T16:57:27.086868+00:00。已读取 6 个固定版本数据集、27 个文件（约 191 MB），包括 9 张回答表；固定版本由 Kaggle MCP 列出，官方 Kaggle CLI 的 Python 客户端下载，全部文件校验大小与 SHA-256。部分历史缓存期望哈希也一致。此次本地 CPU 审计，没有训练模型、调用生成 API、消耗 GPU、提交比赛或发布资源。

## 结果概览

| 文件视图 | 行数 | 类别数 | 本轮初试候选 | 判断 |
|---|---|---|---|---|
| seyit-v2 | 4,998 | 5 | 4,998 | Claude 5 类；作者声明官方 API，缺逐行原始响应日志 |
| abdullah-v2 | 1,033 | 8 | 1,033 | 8 类；有 proxy 返回标识；297 条 finish_reason 缺失，设置混合 |
| dophuchung-v1 | 1,029 | 7 | 1,029 | 7 类；有 proxy 返回标识；包含 reasoning 设置扫描 |
| lucas-v7 | 4,255 | 8 | 4,255 | 8 类；记录调用 slug；v2 prompt bank 主要补缺类，单表组不完整 |
| prvs-raw-v6 | 9,290 | 18 | 0 | 全量带溯源；含 2,700 条 train_regen；不能整表加入每折 |
| prvs-clean-v6 | 4,926 | 18 | 4,674 | clean 仍含 252 条 train_regen；初试只用 new + proxy_v1 |
| matched-v3 | 24,443 | 18 | 0 | 只有 text/model；原 prompt 与来源列丢失，暂不作首轮验证训练池 |
| pool-v2 | 37,252 | 18 | 0 | 23 个 src；5,273 条 prompt_key 空缺；仅作单独汇总池对照 |
| pool-v5 | 119,899 | 18 | 0 | 46 个 src；至少 121 条关联官方 prompt；先追溯，再作单独对照 |

`matched-v3`、`pool-v2`、`pool-v5` 是同一个数据集 V3 中的三个不同文件，不是三个独立来源。prvs raw/clean 也不是独立数据。每张表内部都没有本轮规范化规则下的重复文本；重复主要发生在这些文件之间。

## 1. 必须先处理：重建官方 prompt 的回答

prvsiyan V6 原表包含 **2,700 条 `train_regen`**，覆盖官方 150 个 prompt；其 `passages_clean.csv` 仍保留 **252 条**。`source_prompt_id` 明确关联官方 `p000` 等 ID。这些回答虽然是新生成的、并未逐字复制官方响应，仍然是同一道题。

例如，验证折中有 p008 时，任何来自重建 p008 的公共回答都不能进入该折训练。可在其他训练折使用，但必须继承原 prompt 的分组；首轮小试先全部排除，减少混杂因素。详细排除折号已保存到 [official_prompt_linked_rows.csv](outputs/official_prompt_linked_rows.csv)。这针对**本地验证的泄漏风险**，不等于认定它们不能用于正式提交前的全量训练。

通过明确的 `tr_p###` 键及原文哈希溯源，在 `pool-v5` 中也找到 **至少 121 条**，对应 32 个官方 prompt。其中一些 prompt 键已经变成 `pt:...`，只看前缀会漏掉。未知来源和改写内容尚可能隐藏更多关联，因此 121 是已确认下限。

prvs README 一处笼统说 prompt 与官方不同，但同文件的 split 表与实际 CSV 明确包含 `train_regen`；本轮以具体行与关联字段为准。证据：固定版本 README 第 32 / 163 / 192 行及实际 CSV。

## 2. 版本重叠很大，不能把行数相加当新增数据

9 张表共有 **207,125 条记录**，按 NFKC、大小写和空白规范化后，只对应 **125,411 段不同文本**；额外出现 **81,714 次**（约 39.5%）。没有发现同一规范化文本被标为不同模型的冲突。

具体例子：

- pool-v2 的 37,252 条中，**36,779 条**已在 pool-v5。
- matched 的 24,443 条中，**23,213 条**已在 pool-v5。
- Seyit 的 4,998 条中，**4,924 条**已在 pool-v5。
- prvs clean 的 4,926 条全部属于 prvs raw。

这主要是版本和汇总关系，不代表这些重复记录本身都是坏文本。ATLAS20 的已有加载代码也做了一些 `drop_duplicates`；上述数字描述本次九视图的原始拼接，不能直接用来判定当前基线重复计权或成绩有问题。完整矩阵见 [cross_table_exact_overlap.csv](outputs/cross_table_exact_overlap.csv)。

## 3. “模型标签一致”与“实际服务版本有证据”要分开

- **prvsiyan 的记录最完整。** 9,290 条已发布回答均能关联到原始调用日志，标签、文本、请求 ID、返回 ID 与原始响应模型字段全部内部一致。9,291 条日志中另有 1 条失败调用，没有混进发布回答。Gemini 3.7 的 Google 路由记录保留 `model_version`。
- **abdullah / dophuchung** 保存 proxy 返回标识，且与目标标签在已列出的别名规则下吻合；但没有逐行原始服务商响应日志。
- **lucas** 保存调用用的 `model_slug`，不等于底层实际权重版本。
- **Seyit** README 声明官方 API 且每次返回模型匹配，但下载包没有逐行原始响应，不能把作者声明升级为独立核实。
- **汇总池** 大量行只剩 `text/model/prompt_key/src`，甚至只有 `text/model`。能按文本哈希追溯一部分，其他来源仍需单独核查。

即使存在原始日志，本轮核实的也是所提供记录的内部一致性，不是对服务商底层权重的独立认证。Proxy 可能只回显路由名称；模型大版本相近也不能随意合并标签。溯源结果见 [provenance-audit.json](outputs/provenance-audit.json) 与 [逐行对照](outputs/prvs_provenance_checks.csv)。

## 4. 数据风格和类分布确实发生偏移

下面是含至少一个 em dash（—）的回答比例；单位是“回答占比”，不是每百词标点数。

| table | model | rows | em_dash_rate |
|---|---|---|---|
| lucas-v7 | gemini-3.6-flash | 813 | 20.8% |
| lucas-v7 | gemini-3.8-flash | 438 | 18.7% |
| official-train | gemini-3.6-flash | 150 | 0.0% |
| official-train | gemini-3.8-flash | 150 | 0.0% |
| pool-v5 | gemini-3.6-flash | 6521 | 0.0% |
| pool-v5 | gemini-3.8-flash | 9088 | 0.0% |
| prvs-clean-v6 | gemini-3.6-flash | 288 | 20.1% |
| prvs-clean-v6 | gemini-3.8-flash | 288 | 24.7% |

这说明额外数据不必然服从官方样本的风格分布，不能因为数量多就假定偏差会消失。pool-v5 在这两个 Gemini 类上恢复为 0，与其作者声明的“删去某类在官方训练集从不用的标点”清洗方式相符；这种清洗也可能强化官方训练集的偶然规律。不要直接全局删破折号，应该把保留与过滤做成对照；涉及类别统计的规则应在训练折内确定。

类覆盖也不均衡：Seyit 只有 5 个 Claude 类；Lucas V7 的 952 个 prompt 在本表里每组中位数只有 **2 类**，因为部分批次专门为其他来源补缺类。需要根据共同 prompt 来源合组，不能把每个文件的 prompt ID 当作独立题目。来源、主题和标签可能纠缠；详见 [source_class_counts.csv](outputs/source_class_counts.csv) 和 [prompt_completeness.csv](outputs/prompt_completeness.csv)。

## 5. 逐字没重复，不代表验证安全

公共表与官方 train / 公开 test 三条样例之间，规范化后完全相同的回答为 **0**。全量扫描共同的连续 8 词片段，找到 **324 对**候选，涉及 **179 段**公共文本；再计算字符 5-gram TF 余弦，候选中没有达到 0.90 的对。

该方法是“共享至少 3 个不同 8-gram → 再算相似度”的候选筛查，不是穷举所有字符相似度，更不检测全部语义改写。已有 `train_regen` 就是逐字去重挡不住的题目级关联。179 段共享片段文本保留为人工复查线索，不能全部认定为抄录；本轮初试候选也避开了这些标记。见 [official_overlap_candidates.csv](outputs/official_overlap_candidates.csv)。

## 建议怎样开始下一轮

已经保存 **15,989 条**初试候选的逐行索引：[initial_candidate_index.csv](outputs/initial_candidate_index.csv)。原文继续保存在固定版本原文件里，用 `table + row_number` 读取；本轮没有移动、覆盖或发布数据。

候选组成：Seyit 4,998、abdullah 1,033、dophuchung 1,029、Lucas 4,255、prvs clean 排除 train_regen 后 4,674。该索引只是保守的对照实验起点，不是已经证明能提分的“最终干净训练集”。它保留来源、设置和风险标记，18 类数量从 231 到 1,391 条，相差约 6 倍，训练时仍需控制各类总权重。

建议下一次单独登记轻量 CPU 实验，固定 prompt-group-v1、特征、分类器和同一运行环境：

1. 官方数据作为对照；旧 OOF 可作参考，但新旧依赖版本不同，因此本次比较应在同一环境重算轻量对照。
2. 加入 **prvs clean 的 `new` 子集 4,158 条**（231 个完整的 18 模型 prompt 组），先检查可追溯、类平衡的数据是否有效。
3. 再比较上述 15,989 条混合候选，控制每类增量总权重，检查收益是否只来自某个家族。

将 119,899 条汇总池整包加入放在后续独立对照，先处理已知官方 prompt、未知 prompt 与来源缺失。单模型提升之后还需要验证是否有益于强集成；本轮没有训练、没有新 OOF 或排行榜分数。

## 检查范围与复现

- 固定版本与文件 SHA：[download-manifest.json](download-manifest.json)；完整字段、空值及逐表统计：[profiles.json](outputs/profiles.json)。
- 全部逐行筛查标记：[row_flags.parquet](outputs/row_flags.parquet)。空 prompt 键表示未知，不能把 5,273 条空键视为同一个真实 prompt。
- 算法脚本：[audit.py](audit.py)；运行：`uv run --no-project experiments/EXP-008-public-pool-audit/audit.py`。审计约 26.64 秒（不含下载和依赖准备），环境见 [summary.json](outputs/summary.json)。
- 跨池精确匹配覆盖全部 9 表；没有完成跨池全量语义去重，也没有逐个下载 pool-v5 的其余上游来源。
- 当前在线数据卡可能描述更晚版本；本报告计数来自下载的固定版本文件，不能用数据卡当前行数替换。
- 来源许可与引用说明随固定 README / 数据卡保存，汇总池作者声明沿用上游许可；本轮没有重新分发数据。

固定来源：

- [seyitkaangunes/ntl-extra-api-passages V2](https://www.kaggle.com/datasets/seyitkaangunes/ntl-extra-api-passages/versions/2)
- [abdullah1afify/ntl-generations V2](https://www.kaggle.com/datasets/abdullah1afify/ntl-generations/versions/2)
- [dophuchung/ntl-proxy-generations V1](https://www.kaggle.com/datasets/dophuchung/ntl-proxy-generations/versions/1)
- [lucasboesen/ntl-free-proxy-generations V7](https://www.kaggle.com/datasets/lucasboesen/ntl-free-proxy-generations/versions/7)
- [prvsiyan/prvsiyans-extended-name-that-llm-dataset V6](https://www.kaggle.com/datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset/versions/6)
- [dhruvtibarewal/nlm-matched-clean V3](https://www.kaggle.com/datasets/dhruvtibarewal/nlm-matched-clean/versions/3)
