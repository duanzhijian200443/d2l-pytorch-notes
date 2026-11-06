# SRC-067 · NTL Sovereign Gold Apex V1 · Public 0.665

[正式来源记录](../../../tracking/records/sources/SRC-067.json)

Kaggle 将 **0.665** 绑定于 synthreaper 的 **V1 / scriptVersionId 364684268**。与已归档的 [Apex E V1](https://www.kaggle.com/code/bgunshui/ntl-apex-e?scriptVersionId=364347706) 相比，两个保存版本产出的四个推理模块逐字节相同，实际准备的 Open Pool 数据行数却不同。相对本项目 EXP-047 的 0.664，Apex 同时改动了主 SVC 的池和融合权重、Sinkhorn、标点修正及家族内风格残差。0.001 是两次公开成绩的差，单项收益仍待对照。

## 来源与原件

- 作者为 Aditya Goyal（synthreaper）。固定 V1 与最新源码分别从 Kaggle API 获取，两份原始文件的 SHA256 一致。Notebook ID 为 142184076。[固定得分页](https://www.kaggle.com/code/synthreaper/ntl-sovereign-gold-apex?scriptVersionId=364684268) · [页面版本与输入记录](snapshots/browser-observation-20261102.json)
- 原始 V1 源码 SHA256 为 `ca4c9e18ed32dcd8168b8118b192d39d657803d30dcd89a564b1c2d1d9c846dd`。拼接代码单元 SHA256 为 `5e2bbb0482edbc63ff2b9afc59c8855a8d90e1385b91036b11549a9fd6c72b28`。采集于 2026-11-02 UTC。[原始 Notebook](source.ipynb) · [API 元数据](metadata.json) · [下载回执](receipt.json) · [作者正文](description.md) · [代码文本](code.txt)
- 逐行比较覆盖 [EXP-047](vs-EXP-047.diff)、[EXP-004](vs-EXP-004.diff)、[One Swap V1 / SRC-036](vs-SRC-036.diff) 和 [Apex E / SRC-040](vs-SRC-040.diff)。[两份固定保存版本的模块哈希与日志](commit-output-comparison.json)

## 作者声称与实际设置

作者把标题写作“198k Clean Pool”，列出单 SVC CV 0.4805→0.5008、双 T4 节省约 40 分钟和 sub-minute 保存。这份 V1 的输入卡实际挂载 Open Pool 平台 V13（数据作者 v11）。公开保存日志显示准备池 190,588 行，页面显示保存耗时 9m60s。CV 数值也见 One Swap 作者对旧 193k 池的报告。此处未取得这次完整隐藏评分的 CV、计时或逐模块日志。

| 环节 | 本项目 EXP-047 V3 · 0.664 | Sovereign Gold Apex V1 · 0.665 |
|---|---|---|
| 主 SVC 训练数据 | 官方 + matched + nlm_pool_v5 | 官方 + matched + Open Pool V13。先保留官方 18 类，有相应列时剔除 train-prompt regen、Gemini 3.7 路由风险。缺池文件则退回 nlm_pool_v5 |
| 主 SVC 权重 | 官方每行 1、matched 0.5、pool 0.35。svc4 的 log 分数乘 1.5 | 三种样本权重相同，svc4 的 log 分数改乘 **12**。仍为四视图 char、word、shape、function 的 LinearSVC（C=0.5、balanced） |
| 类频率和标点 | Sinkhorn **0.30**、target mix 0.35、标点系数 **1.5** | Sinkhorn **0.60**、target mix 0.35、标点系数 **1.2**、clip ±0.30，保留 `kappa=24`。正文末段沿用的 0.45 与执行命令不符 |
| 家族残差 | 已有 comp 内的家族 gates。最后一步为 Bayesian typography | 在 typography 后增加 V40A 残差，按官方训练数据的 `prompt_id` 分组校准 Claude/Gemini/GPT/GLM 的家族内重排 |
| 继承的主模型/混合 | q14v5、q8v5、ll8v5、n14b、q25_14、deb 六成员；两种 8B 的 dp 已启用，comp 15%，低 margin 的 ATLAS 20% | 六成员及主权重 3/2/1/2.5/1/1 相同；tfidf .25、distort .10、comp .15、ATLAS .20、margin .30 相同。Apex 的 Llama 输入改成优先 Meta、失败后找 alexchilton，两份公开保存均选中了 alexchilton |

Apex V1 的 `MODELS` 使用已有权重。SVC 在官方、matched、pool 的顺序归一化去重并训练，改变的 `--pool` 只供这个成员常规训练，TF-IDF、distort 仍用官方和 matched。家族残差从官方文本提取词长句长、词频和标点等特征，各家族分别拟合 StandardScaler + balanced LogisticRegression，正则参数 Claude/Gemini/GPT 为 0.003，GLM 为 0.03。五折 GroupKFold 得到家族内 BA，用 `clip((BA−chance)/0.15,0,1)` 算可靠度。每行强度为 `0.08 × 可靠度 × clip((0.65−家族margin)/0.65,0,1)`，只重排该家族并保持其总概率。可靠度为零会跳过家族，异常会退回残差前 logits。完整隐藏数据上的激活家族、改判数和实际权重未知。

## 与 Apex E 的同源关系

SRC-040 固定 V1 已绑定 0.663。代码文本差分只有两组变化。Sovereign Gold 的样例快路增加 `KAGGLE_IS_COMPETITION_RERUN` 判断，在比赛重跑标志存在时即使行数小于 50 也要走全流程。另一个变化是产生 `infer.py` 的写法。SRC-040 先写 1.5 再改为 1.2，此版本直接写 1.2。两份 V1 公开保存实际产生的 **infer.py、comp_stack.py、gauss.py、sibling_style.py** 的长度和 SHA256 均相等。四份模块、模型定义和运行参数一致。两份 API 清单的 15 个数据集、8 个上游 Notebook 和 8 个模型 ref 集合，以及容器摘要也一致，这些集合未固定每个附件的数据字节。[源码差分](vs-SRC-040.diff) · [实际输出比对](commit-output-comparison.json)

| 固定保存版本 | 快路 | Gemini 风险剔除 | 写出的安全池 | 公开三行结果 |
|---|---|---:|---:|---|
| Apex E / SRC-040 | True | 8,297 | 190,210 | 全部 gpt-5.5 |
| Sovereign Gold Apex V1 | True | 0 | 190,588 | 全部 gpt-5.5 |

两份都是实际固定版本日志中的数字，相差 **378 行净额**。行数无法说明两池仅有 378 个不同样本。安全池写出之后还要与官方和 matched 去重，最终 svc4 拟合行数未知。Input 和日志说明当前作者实际保存使用了 V13，页面的 0.665 与 SRC-040 的 0.663 无法分解成数据版本、输入权重、数值差异或隐藏执行覆盖的各自贡献。

## 开销、用途与验证

公开保存中 `FAST_COMMIT True`，comp 用时为 0，实际输出只有三个占位预测。隐藏评分按代码应运行完整主流程。ATLAS 预算依赖已有 comp 耗时，还包括从 `infer` 开始到 SVC 训练和后续成员推断所消耗的时间。svc4 分数乘数的变化还会改变 margin<0.30 的入选行。只有完成评分的 ATLAS 行实际以 20% 混合，分支故障时可跳过。公开 V1 的 9m60s 是保存耗时，隐藏评分的 GPU 时长和各阶段日志尚未取得。

复现时先复用已审计的 [Open Pool V13](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool/versions/13) 及 SRC-065 / EXP-050 的来源与 prompt 隔离记录。建议先将 EXP-047 的固定成员缓存作为对照，分别检查四项改动：同一池下的 svc4 1.5/12、同一 svc4 权重下的换池、同一预测下的 0.30/0.60 与 1.5/1.2，以及固定 typography 输出的家族残差开关。残差应在外层分折内拟合校准，同时记录各家族纠错和引错、ATLAS 入选/完成行数及时间。CPU 家族内 OOF 仅用于校准，不能代替完整 ATLAS 的留出验证。本条只读取源码和保存产物。

## 更新记录

- 2026-11-02：固定 V1 与 0.665 的版本绑定，保存 EXP-047、EXP-004、SRC-036、SRC-040 四份差分及两份 Apex V1 的公开保存日志和实际模块哈希。

## 同配方的固定得分版本

Alex V9（SRC-069，0.663）与 chyibo V1（SRC-068，0.665）整份 Notebook 字节相同。旧 Apex E（SRC-040）和 Alex 的输入卡均为 Open Pool 平台 V11，公开保存时各写出 190,210 行。两份 0.665 使用 V13，均写出 190,588 行。四个版本公开产出的四份推理脚本哈希一致。隐藏评分的完整输入、ATLAS 覆盖与逐行预测未取得，分差无法按组件分摊。[新版本输入与日志](../../../tracking/snapshots/20261102T071145Z-public0665/kaggle-browser-observation.json) · [旧版本输入卡](../../../tracking/snapshots/20261102T071145Z-public0665/historical-input-card-observations.json) · [V1 脚本比对](../../../tracking/snapshots/20261102T071145Z-public0665/fixed-v1-module-comparison.json) · [Alex V9 脚本回执](../SRC-069-alexchilton-all-public-v9/receipt.json)
