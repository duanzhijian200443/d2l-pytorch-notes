# SRC-038 · Open Pool：作者v10 / Kaggle V11 数据卡与清洗对照
状态：已读数据卡、版本历史、文件清单；未审计全量数据。
原始记录：[正式台账](../../../tracking/records/sources/SRC-038.json)。

## 值得保留的结论
作者称198507条、18标签、54来源、44601 prompt组的清洗池。Kaggle API明确 current_version_number=11，版本11的notes称“v10”；192926条clean v9实际是平台版本10。引用与下载必须按平台版本11固定，不能把作者发行名当平台版本号。

## 来源与原件
- [Kaggle版本11](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool/versions/11)
- [数据卡原始元数据](dataset-metadata.json)、[MCP版本历史](api-info.json)、[版本11文件清单](files-v11.json)。
- 关联SRC-027清洗与配对比较讨论、SRC-033旧聚合池和本轮One Swap方案。卡片抓取于2026-11-01，数据版本发布时间2026-10-30 07:53 UTC。

## 作者声称与实际设置
作者称v10在v9上加5581条，旧行保持不变；清除垃圾、体裁偏差、溯源风险、官方prompt重建等，发布removed_rows、source attribution和dataq_results。45k等量Qwen3-1.7B/两seed测试，clean比raw balanced accuracy约+0.027；DeBERTa全量raw v7/clean v9报告Public LB0.585/0.587。上述不是ATLAS20增益，也不是本项目已复现结论。
卡片称flag_train_prompt_regen与flag_near_train_prompt从v9全False；标签仍是上游作者声明。CC BY-SA4.0是本数据卡许可声明，不证明每条上游来源已独立完成许可审查。

## 我们核查了什么
已核对平台版本历史和12个文件名。ntl_open_pool.parquet列表大小99263691字节；本次未下载/读取parquet，因此198507行、覆盖/去重/标记和全部上游许可均属于作者报告。元数据原文和文件清单有本地SHA256，数据本体哈希尚缺。

## 对当前方案的用途与限制
主svc4换池是近期有依据的候选，但必须先恢复跨来源prompt身份、检查与现有data及官方train的重复。清洗使用官方train标点统计，不能仅凭“无原文复制”认定严格OOF独立。类不均衡、proxy路由标签、标点筛选偏差仍需独立检查。One Swap代码宣称包含所有nlm_pool_v5，卡片却描述清洗删行，不能不核实就据此认定严格超集。

## 待验证想法
关联IDEA-003：复用EXP-008/009的方法和现有原件，未来固定版本11并核查hash/逐行来源，再做相同分类器、等量与全量、相同权重的比较；先区分上游清洗对验证独立性的影响。本次没有下载大数据或训练。

## 更新记录
- 2026-11-01：新增数据资料，明确作者v10与平台11差异；链接原讨论，未把卡片声明写成逐行审计结论。

关联来源：[SRC-027](../../../tracking/研究索引.md#src-027) · [SRC-033](../../../tracking/研究索引.md#src-033) · [SRC-036](../../../tracking/研究索引.md#src-036) · [SRC-037](../../../tracking/研究索引.md#src-037)。
