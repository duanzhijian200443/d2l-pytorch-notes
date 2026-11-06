# 主 svc4 公共池权重对照草案

**状态：草案，未执行；前置的 EXP-010/011/012 完整验收仍被 NPY/NPZ 下载缺口阻塞。**

EXP-004 / ATLAS20 V2 / scriptVersionId 363569852 / Public LB 0.662 继续作为正式固定基线，不修改源码、附件或提交选择。本草案不占用新 EXP 编号，不授予训练权限。

## 1. 研究问题及当前依据

Stylegram 三组完整 CSV 已核对：50% OOF BA=0.468242605，25%=0.459709172，75%=0.464916729。CSV-only prompt 配对区间均跨0，完整二进制验收未完成。当前不将它们视作已通过的实验结论，更不能推广为所有公共池都应占50%。

下一轮仅问：在相同数据隔离、相同训练行集和四视图特征下，只改变主 svc4 的 pool 逐行训练权重，是否改善固定分组开发验证？

## 2. 主 svc4 原配方，不是 Stylegram

按 EXP-004 固定 infer.py 第183–231行：

| 项目 | 主 svc4 原配方 |
|---|---|
| 数据 | official + matched + pool |
| 每行 sample_weight | official=1.0；matched=0.5；pool=0.35 |
| 额外类别权重 | class_weight='balanced' |
| 分类器 | LinearSVC，C=0.5，random_state=42，max_iter=5000 |
| 特征输入类型 | float32 |
| 四视图缩放 | char=1.0；word=0.7；shape=0.5；function=0.5 |
| 最终主集成 log 系数 | 1.5，不是本轮训练权重变量 |

Stylegram按类别分配官方/公共样本总权重份额，并包含plain/centred路由；主svc4的0.35是每条pool样本的loss权重，没有相同的中心化双模型。不能把Stylegram的0.5直接移植成svc4的0.5。

原数据投影：official 2699、matched 24443、保留pool 96686，共123828行。它是EXP-008固定文件按原规则筛选的投影，不是历史隐藏评分实测计数，也不是隔离后每折行数。

## 3. 尚未解决的启动阻塞

### B1 官方 prompt 关联不完整
主svc4公共行至少121条关联官方prompt，这只是下限。按来源、row ID、文本哈希和官方prompt映射定位，不能只查tr_p前缀。所有组必须共享同一隔离规则：统一排除全部已知关联，或逐折排除该验证折关联，执行前固定一种，不可仅为候选清洗。

### B2 matched/pool 来源字段不全
matched仅text/model，部分prompt来源已丢失；pool-v5有46个src、至少2470条空/占位prompt_key。未知键不能当真实组，回答不重复也不证明题目隔离。建立可追溯映射或共同隔离未知行；若不能做到，只能称“已知关联隔离后的探索性验证”，不能宣称严格无泄漏。清理明显改变配方时，对照必须命名为“隔离后的svc4”，不是原榜单基线的原样复现。

### B3 上游清洗使用全量官方标签统计
EXP-008记录pool-v5含按官方类别标点规律进行的清洗。单删121条不能消除构建阶段验证标签依赖。严格验证需回到清洗前来源，在每个训练折确定标签相关规则，或使用不依赖验证标签的固定规则。仅切分清洗后文件不能恢复已丢失内容，也不能消除此风险。

### B4 跨源重复和顺序
原流程NFKC+空白归一化、大小写敏感，按official→matched→pool先到先得；不同于EXP008的casefold审计键。保留两个键的用途，核对每行最终来源和权重，不能把一段文本同时算matched和pool。4158条prvs new已经在基线中，不是新增数据。

### B5 强集成干净 OOF 缺失
部分adapter/分类头/ATLAS权重见过全部官方数据。训练集预测事后分折不能证明svc4对ATLAS20的增量。不阻止隔离后的单成员试验，但阻止“完整集成无泄漏OOF”的结论。

### B6 字节与版本绑定
冻结新实验输入版本、SHA-256、来源和逐行隔离清单。EXP004历史31项输入仍缺部分dataset/kernel版本与字节绑定，不能把新开发实验说成完全复刻旧隐藏运行。

## 4. 单变量方案（仅建议，需新的训练授权）

| 组别 | official每行 | matched每行 | pool每行 |
|---|---:|---:|---:|
| 原配方对照 | 1.0 | 0.5 | 0.35 |
| 降低 | 1.0 | 0.5 | 0.175 |
| 提高 | 1.0 | 0.5 | 0.70 |

这些是原pool权重减半/翻倍，不是公共数据总份额。不同时改变来源、matched权重、模型或融合系数。

固定：prompt-group-v1五折、官方ID/标签/顺序/折文件哈希；每折共同隔离去重并固定行顺序。每折四TF-IDF只拟合训练官方文本+允许的公共文本，三组共用词表/IDF/矩阵，不用官方验证文本拟合特征。

四视图：char 3–5/max_features250000；word 1–3/180000；shape 2–6/60000；function 1–3/50000；min_df=2、lowercase=False、sublinear_tf=True和原token_pattern均不动。C、seed、max_iter、float32与依赖版本固定；记录原默认dual等参数，不擅改。

三组同一训练行计数，class_weight保持一致；保存每类/来源行数、sample_weight总量及乘class_weight的有效总权重。不额外归一化总sample_weight，避免暗改实验定义；改变pool权重也改变总损失规模，解释中必须承认这一点。

## 5. 未来预算边界

当前不训练。后续获准才可：CPU、三组×五折=15次单模型拟合；每折一次特征供三组复用，不加载LLM/LoRA/Gaussian/ATLAS20。记录输入清单、矩阵指纹、阶段耗时、峰值内存和收敛。

不把Stylegram约64分钟当svc4耗时承诺。主svc4训练表更大，四视图理论上限540000列，需按实际稀疏矩阵尺寸估算内存。执行前冻结墙钟/内存上限；初始可建议7200秒，但必须容量评估可行且用户授权后采用。到预算或OOM保留产物停止，不自动GPU、重跑，也不只比较先完成的组。

## 6. 验收规范

保存2699行OOF决策分数/预测/ID/标签/prompt/fold/标签顺序和每折缓存。decision_function不是校准概率。

1. 三组同一2699唯一ID与顺序，18列分数有限，argmax与CSV吻合。
2. prompt不跨折；逐折公共隔离清单、来源计数、哈希可追溯。
3. 全量BA/accuracy、逐类召回、逐折差、混淆和错误互补性。
4. 0.175/0.70分别对0.35作同prompt配对，报告修正旧错、引入新错、分歧样本。
5. 固定seed的prompt成组bootstrap，区间条件于当前模型/开发折，不代表重训或隐藏榜单不确定性。
6. 稀疏压力场景若保留m=2/3/6×5次：每次合并五折算BA，不平均缺类折BA。主svc4若仍为纯逐行推理，可以经确认后子选现有OOF分数；若加聚类、中心化或Sinkhorn，需按场景重算且属于新实验定义。
7. 稀疏多次抽样共用模型/prompt，不当成独立训练重复。

## 7. 决策门槛

任何输入/缓存/隔离/类别检查失败，不判通过。单次最高只保留候选；区间跨0、折方向不稳或收益来自极少prompt，不替换原设置。即使区间不跨0，仍是探索性开发验证，存在多候选选择与构建偏差，需后续稳定性/集成验证授权。

不设“提高0.001就提交”的通用门槛。EXP004始终保持0.662不变，不将单成员BA与它相减，不保证私榜收益。获新的训练授权后才登记新EXP；草案不执行，不提交。

## 证据来源

- [EXP009基线成员与数据报告](https://drive.google.com/file/d/1O634leziE02aCXlAhL-CdaYKeCBtlIxp/view)
- [主成员结构化记录](https://drive.google.com/file/d/1mqn989H0XAM18FlBpk5I17WvNe_JbgRL/view)
- [固定infer.py](https://drive.google.com/file/d/1yAHsjS18I-Ffn_rZ_HzwycbMlf49-sM5/view)
- [EXP008公共池审计](https://drive.google.com/file/d/1Ncbw5PvKuTLWI3hTQi_rpPJjWVo5sqwD/view)
- [方法与决策DEC008–010](https://drive.google.com/file/d/11xBdTWtEVELorqCRTaEw68-Olh95GoL6/view)

这些来源本次已只读核对；历史投影不能替代实时隐藏运行证据。与本次acceptance-partial.md一同阅读。
