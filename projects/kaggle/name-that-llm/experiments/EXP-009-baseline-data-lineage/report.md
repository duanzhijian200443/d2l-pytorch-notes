# EXP-009：ATLAS20 0.662 基线到底用过哪些数据

**审计结论：前一轮的 4,158 条 prvs clean/new 全部已被这套基线规则选中，不应再以“给强基线加新数据”立项。15,989 条候选中，14,604 条已进入至少一条被审计的运行时训练支路；余下 1,385 条只是在这些支路里没出现，不能保证现成权重从未见过。**

## 证据与适用范围

- 固定目标：EXP-004，私有 Notebook V2，scriptVersionId 363569852，历史正式 Public LB 0.662。源码 SHA 与既有记录一致：`6c9145f038ccffbe08dc6da6db3d1dcf3d2e0e7d2286927bfc8800ef5148cfb8`。
- 本次实时 MCP 读取固定 V2，31 个输入引用与已有 outer metadata 一致。但接口没有给出 dataset / kernel 的版本号；内嵌 Notebook metadata 有复制遗留，不能用于补齐实际运行版本。
- **以下精确行数来自：0.662 原版过滤规则 + EXP-008 下载的固定公共文件。它们是可重算的当前数据投影，不是原来隐藏评分运行的实测计数。** 只覆盖已审计的六个公共数据集；其他附件里是否存在 `ntl_generated*.csv` 等同名输入仍未核实。
- 未运行完整 Notebook、未训练分类器、未启动 GPU、未提交比赛。只抽取并执行已检查过的 loader/权重函数，避免导入会自动推理的整份脚本。

## 1. 主流程：公共池用于三条现场拟合支路

`official` 是官方 train；`matched` 是 nlm_matched_clean；`pool` 是 nlm_pool_v5。

| 成员 | 本轮数据用途 | 样本权重 | 主 log 系数 | 中心化 twin 系数 | Gaussian 系数 |
|---|---|---|---|---|---|
| tfidf | official+matched | {'official': 1.0, 'matched': 0.5} | 0.25 | — | — |
| distort | official+matched | {'official': 1.0, 'matched': 0.5} | 0.1 | — | — |
| svc4 | official+matched+pool | {'official': 1.0, 'matched': 0.5, 'pool': 0.35} | 1.5 | — | — |
| q14v5 | 现成权重推理；原始训练范围另查 | 没有现场重训 | 3.0 | 1.05 | 0.7 |
| q8v5 | 现成权重推理；原始训练范围另查 | 没有现场重训 | 2.0 | 0.7 | 0.44 |
| ll8v5 | 现成权重推理；原始训练范围另查 | 没有现场重训 | 1.0 | 0.35 | 0 |
| n14b | 现成权重推理；原始训练范围另查 | 没有现场重训 | 2.5 | 0.875 | 0 |
| q25_14 | 现成权重推理；原始训练范围另查 | 没有现场重训 | 1.0 | 0.35 | 0 |
| deb | 现成权重推理；原始训练范围另查 | 没有现场重训 | 1.0 | — | — |

主流程去重保留大小写：NFKC + 空白归一化，按 **官方 → matched → pool** 先到先得。与 EXP-008 用于审计的 casefold 键不同，本次按原函数重算并校验。

- TF-IDF LR / distort LR：官方 2,699 + matched 24,443 = **27,142 行**。官方样本权重 1，matched 0.5。
- 四视图 SVC：再从 pool 119,899 行中保留 **96,686 行**，另 23,213 行被已出现的文本去掉；总计 **123,828 行**。pool 新保留行权重 0.35，并另有 `class_weight=balanced`。
- 0.5 / 0.35 是每行 loss 权重，不是数据源影响百分比，也不是最终融合权重。词表与 IDF 在选中的全部文本上无权重拟合；大量公共数据仍会显著改变特征空间。
- q14/q8/ll8 v5 使用五折加全量的六套 adapter；n14b/q25_14 是各一套；deb 是现成 encoder/head。改 CSV 不会重训这些权重。
- q14/q8 的 Gaussian 头在官方特征、标签和 prompt 上现场拟合；其特征来自全量 adapter，因此不能事后切折就称干净 OOF。

源码与精确引用：[infer.py](infer.py)、[main-members.json](main-members.json)、[逐行 SVC 清单](outputs/main_svc_rows.parquet)。这里的普通四视图 `svc4` 与其他分支的 pool-centred SVC 不是同一个实现，不能混用先前的 bug 结论。

## 2. 互补流程：挂载不等于进入共享公共池

原顺序为 Seyit → Lucas → abdullah → dophuchung → `ntl_generated*` → prvs。先筛结束状态（指定列存在才做），再限制标签、清理空白/整段粗体、取 30–300 词，按 `(model,text)` 去重，按 `(prompt_id,model)` 保留首次，最后只保留覆盖至少 12 类的 prompt。

| origin | input_rows | after_finish_filter | after_label_and_length | after_model_text_dedup | after_prompt_model_dedup | after_min_12_labels |
|---|---|---|---|---|---|---|
| seyit-v2 | 4998 | 4998 | 4998 | 4998 | 4998 | 2099 |
| lucas-v7 | 4255 | 4255 | 4252 | 4252 | 4252 | 2880 |
| abdullah-v2 | 1033 | 736 | 736 | 736 | 360 | 0 |
| dophuchung-v1 | 1029 | 1029 | 1029 | 1029 | 540 | 420 |
| prvs-clean-v6 | 4926 | 4926 | 4926 | 4926 | 4926 | 4926 |

最终共享 EXTRA 是 **10,325 行 / 708 个 prompt**。其中 prvs 的 4,158 条 new 全部在内；同样也包含 252 条 train_regen。abdullah 这条共享支路保留 0 条，**不代表其文本没被主流程汇总池使用**。

这个 loader 会丢掉 `setting`、原 source_prompt_id 等字段；同题同模型先到者胜出，文件顺序会替代明确的设置选择策略。将 `tr_p###` 改成 `pv_tr_p###` 并没有消除与官方题目的关系。

### 各成员用途

以下系数仅是 comp 内部的 log-probability 系数，不是最终全栈占比。所有行号指 [comp_stack.py](comp_stack.py) 的 1-based 行号；它比 Notebook cell 8 内行号少 1（去掉 writefile 行）。

| 成员 | 运行时拟合/预测 | comp log 系数 | 源码行 |
|---|---|---|---|
| stylegram | official train + filtered EXTRA | 1.516 | 239, 260, 269, 274, 277, 307 |
| stylometry_lgb | official train only | 0.063 | 363, 368 |
| debv3l | official train only | 0.05 | 56, 59, 607, 617, 644, 650, 700, 707 |
| electl | official train only | 0.032 | 56, 59, 607, 617, 644, 650, 700, 707 |
| desk | official train only | 0.132 | 56, 59, 607, 617, 644, 650, 700, 707 |
| q3_8b_twin | official train only, centred by official prompt_id | 0.3285 | 37, 53, 55, 653, 658, 679, 707 |
| q25_7b_twin | official train only, centred by official prompt_id | 0.2115 | 37, 53, 55, 653, 658, 679, 707 |
| q3_4b_twin | official train only, centred by official prompt_id | 0.1245 | 37, 53, 55, 653, 658, 679, 707 |
| glm9b_twin | official train only, centred by official prompt_id | 0.165 | 37, 53, 55, 653, 658, 679, 707 |
| debv3l_twin | official train only, centred by official prompt_id | 0.25 | 37, 53, 55, 653, 658, 679, 707 |
| gauss_q3_8b_L22 | official train features, labels, prompt_id | 0.3285 | 44, 48, 665, 672, 679, 707 |
| gauss_glm9b_L8 | official train features, labels, prompt_id | 0.165 | 44, 48, 665, 672, 679, 707 |
| gauss_q3_4b_L4 | official train features, labels, prompt_id | 0.1245 | 44, 48, 665, 672, 679, 707 |
| gauss_debv3l_L24 | official train features, labels, prompt_id | 0.25 | 44, 48, 665, 672, 679, 707 |
| lora24_public | No runtime supervised fit; TE inference. TR[:3] only checks feature parity when anchor available. | 0.177 | 60, 67, 711, 794 |
| lora24_seed7 | No runtime supervised fit; TE inference. TR[:3] only checks feature parity when anchor available. | 0.113 | 60, 67, 711, 794 |
| lora24_claude | No runtime supervised fit; TE inference. TR[:3] only checks feature parity when anchor available. | 0.177 | 60, 67, 711, 794 |
| lora20_q25_instyle | No runtime supervised fit; TE inference. TR[:3] only checks feature parity when anchor available. | 0.17 | 60, 67, 711, 794 |
| lora24_public_twin | No runtime fit; centre test features within eligible groups | 0.15 | 38, 779, 781 |
| lora24_claude_twin | No runtime fit; centre test features within eligible groups | 0.15 | 38, 779, 781 |
| tokattn_q35 | TE only; optional TINY in-sample TR[:40] check is not training | 0.153 | 68, 70, 808, 857, 888, 904 |
| tokattn_deb | TE only; optional TINY TR[:40] training-fit parity check | 0.185 | 68, 808, 857, 888, 904 |
| ultrabert | No runtime supervised fit; TE inference only | 0.095 | 860, 885, 905 |

stylegram 在官方 + EXTRA 上现场拟合。每类有公共样本时，官方每行权重 0.5；外部每行权重为 `0.5 × 官方该类行数 / 外部该类行数`，使两者各占该类总权重的一半。plain/centred 按测试分组替换预测后，**只加一次 1.516**，不是两票。

四个 LM plain head 系数为 0，但相同底座仍用于 twin、Gaussian 或聚类，不能列为整体停用。Qwen3.5 token-attention 的配置是 `NTL_DROP_Q35=0`（开启），之后仍要检查 head 和底座。已声明模型附件没有 Qwen3.5 底座，可能被跳过；历史实际激活状态需要运行日志确认，不能写成开关禁用。

### 家族内细分 gates

| gate | 数据 | 样本权重 | λ | 插值上限 |
|---|---|---|---|---|
| claude | official Claude rows + raw first matched Seyit CSV | all rows equal | 1.5 | 0.45 |
| gpt | official GPT rows + filtered EXTRA GPT rows | official=1, extra=0.25 | 0.25 | 0.2 |
| gemini | official Gemini rows + filtered EXTRA Gemini rows | official=1, extra=0.25 | 0.25 | 0.2 |
| glm | official GLM rows + filtered EXTRA GLM rows | official=1, extra=0.25 | 0.25 | 0.2 |
| claude_lora | checkpoint claude_coef readout on TE | no runtime fitting | 1.0 | 0.45 |

Claude gate 直接读取完整原始 Seyit CSV，绕过共享 loader，投影为 **749 官方 + 4,998 公共行**，同等样本权重。GPT/Gemini/GLM gate 分别使用 3,540 / 2,309 / 490 条过滤后的公共行，加相应官方行；公共行权重 0.25，另外使用 balanced class weights。gate 只改家族内部排序并保留家族总概率，λ 不等于恒定的最终占比。

完整信息：[complement-members.json](complement-members.json)、[共享公共行清单](outputs/complement_extra_rows.csv)、[逐类权重总量](outputs/stylegram_class_weight_mass.csv)。

## 3. 现成权重里已经包含的数据

本次读取作者当前数据卡所得证据如下；它们不替代固定 checkpoint 的逐行训练清单。

| 成员 | 能确认的作者说明 | 尚不能确认 |
|---|---|---|
| n14b / q25_14 / deb | 公共池 75,783 行后再用官方数据训练；Qwen3-14B 另用一个公开来源 | 精确输入哈希、每个 checkpoint 的逐行/折绑定 |
| q14v5 / q8v5 / ll8v5 | 当前标题写 120k pool，正文为空 | 不能仅凭标题认定就是本次 pool-v5 的全部119,899行 |
| ATLAS | 作者称全部2,699官方行训练5轮，公布train SHA匹配 | 验证折专用权重与历史运行覆盖 |
| Claude LoRA | 18类头用official；Claude辅助头还用Seyit；基线选择final | 不能用final代替fold0做OOF |
| UltraBERT | 作者称按prompt五折训练，消费者平均五个模型 | 平均所有fold对训练集预测不等于OOF |
| 其余LoRA/token-attention/encoder readouts | 找到消费者选择的checkpoint标记 | 完整训练数据版本、选择规则与折范围仍待生产Notebook核实 |

因此，删除一个运行时 CSV 不会消除它在已训练权重中的影响。证据与实时来源链接见 [checkpoint-evidence.json](checkpoint-evidence.json)。

## 4. 前一轮候选是否真的新增

| table | candidate_rows | in_main | in_comp | in_either | in_any_audited_runtime_branch |
|---|---|---|---|---|---|
| abdullah-v2 | 1033 | 1023 | 0 | 1023 | 1023 |
| dophuchung-v1 | 1029 | 1029 | 420 | 1029 | 1029 |
| lucas-v7 | 4255 | 2742 | 2880 | 2880 | 2880 |
| prvs-clean-v6 | 4674 | 4673 | 4674 | 4674 | 4674 |
| seyit-v2 | 4998 | 4998 | 2099 | 4998 | 4998 |

**4,158/4,158 条 prvs new 已在上述运行时配方中。** 剩余未进入已审计现场训练表的 **1,385 条**由 Lucas 1,375 + abdullah 10 组成；其中不少是覆盖条件/状态或选择顺序造成的差异，不能立刻当作值得添加的好数据，也可能在上游权重中出现过。

分别保存：[候选重叠逐行表](outputs/candidate_overlap.csv)、[仅未命中现场训练表的候选](outputs/potential_increment_rows.csv)。匹配依据是 EXP-008 审计哈希；该统计没有宣称检查了所有上游权重的训练语料。

## 5. 验证需要先做的隔离

当前投影中，主 SVC 的公共训练行含已知官方题目重建回答 121 条；comp stylegram 含 252 条。家族 gates 也继承其中一部分。直接把原版现场训练逻辑放进我们的五折验证，会产生已知的题目跨折风险。

这不否定正式隐藏集评分的 0.662：正式训练本来就可使用官方训练题目的公开扩充。约束针对本地开发验证。后续共同底座必须保留原始 row_id/source_prompt_id、按官方折隔离；必要时先在所有对照里一致排除这部分，而不是只给实验组删掉后比较。

## 6. 修正后的实验顺序（尚未执行训练）

| 优先级 | 对照 | 唯一主要变化 | 判断依据 |
|---|---|---|---|
| P0 | 同样做已知prompt隔离后的原stylegram配方，EXTRA_SHARE=0.5 | 比较0.25、0.75；行集、特征、C、折和seed保持一致 | 五折配对BA差、逐类召回、错误互补性；不是最终LB提升 |
| P1 | 主SVC原数据配方、pool每行0.35 | 只比较pool来源权重，保持matched权重和每折同一词表/IDF | 看大公共池是否过强；不同时删源或加新模型 |
| P2 | 固定权重与来源配额的原版first-row选择 | 按明确setting选择同题回答，保持数量/分组可比 | 评估文件顺序带来的设置偏差 |
| P3 | 已验证有效的现场分类器修改 | 固定其他成员再确认完整集成增量 | 检查单成员提高是否被ensemble抵消；现成权重不能伪造OOF |

之前“官方only vs +4,158 vs +15,989”的计划只适合从零搭轻量诊断分类器，**不再作为给ATLAS20增加新数据的主优化路线**。接下来最值得先做的是 P0 的现有 stylegram 公共权重对照，不是再次复制整份 Notebook。

## 复核与产物

`uv run --no-project experiments/EXP-009-baseline-data-lineage/project_rows.py` 只执行数据筛选与统计。已与固定源码抽取的原 `load_extra_passages` 函数逐行比对一致，主去重键也与原函数一致；没有导入原始完整推理模块。统计与源码哈希保存在 [projection.json](outputs/projection.json)。

主融合顺序是主 LP 加和/温度缩放 → softmax → comp 0.15 → 低margin且预算内完成的ATLAS 0.20 → Sinkhorn 0.45（目标混入官方先验0.35）→ 最终Bayes标点修正1.5。`post=0` 不关闭最后一步。COMP/ATLAS可因错误被跳过、ATLAS可能只覆盖部分行，因此静态配置不证明历史运行每个成员的实际参与率。
