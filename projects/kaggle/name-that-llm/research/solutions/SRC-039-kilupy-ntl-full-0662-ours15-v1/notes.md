# SRC-039 · Kilupy ours8 V1：加权1.5的LoRA成员，依赖私有数据

原始记录：[正式台账](../../../tracking/records/sources/SRC-039.json)。
状态：已读取固定源码、完成静态对比并绑定 V1 页面成绩；正式编号已登记。

## 值得保留的结论

这份代码是在 SRC-002（alexchilton All Public V8）的完整集成上加一个 `ours8` 成员。`ours15` 对应代码里的 **log-probability 权重 1.5**，不能解读为新增模型占最终概率的 15%。基础 SVC、公共池、Sinkhorn、标点和 ATLAS 参数均未改。

源码新成员很明确，但其训练数据、折划分和权重产物版本尚不明确。网页明确列有 `[Private Dataset]`，因此这是公开源码且含私有依赖的方案，不能称为完整公开可复现。它是“已有集成加独立成员”的候选，不能仅凭页面小幅分差判断新增成员产生了稳定收益。

## 来源与原件

- 来源：[Kilupy Notebook](https://www.kaggle.com/code/kilupy/ntl-full-0662-ours15)。本次用 Kaggle API 先取当前版本，再用 `version_label=v1` 固定拉取，二者源码一致。
- 采集：2026-11-01 07:32:08 UTC / 15:32:08 北京时间。
- [固定 source.ipynb](source.ipynb)、[API 元数据](metadata.json)、[采集与哈希](download-receipt.json)、[原 Markdown](description.md)、[代码提取](code.txt)、[对 SRC-002 的逐行差异](diff-vs-SRC-002.patch)、[静态检查证据](analysis-evidence.json)。
- 主窗口在本轮（2026-11-01 UTC，逐页时刻未记录，采集截至07:35:57 UTC） 实读网页：Version 1 of 1，Public Score 0.664，Best Score 0.664 V1；V1 与 Copy/Edit 链接均绑定 **scriptVersionId=364235852**。保存 [主窗口 DOM 观察快照](snapshots/browser-observation-20261101.json)；[固定得分版本](https://www.kaggle.com/code/kilupy/ntl-full-0662-ours15?scriptVersionId=364235852)。该快照明确标注由主窗口采集，子任务没有伪装成直接浏览器抓取。

## 作者声称与实际设置

Notebook 的全部 Markdown 与 SRC-002 原文完全相同，包含其全公开声明、历史失败实验和稀疏验证相关性旧说法。它们是继承正文，**不能视为 Kilupy 对新增 ours8 的训练或消融报告**；稀疏验证的后来纠正另见 SRC-026。

实际新增只有两处：

1. 扫描 `/kaggle/input/**/ours8_fold*/heads.pt`，排序后把每个父目录放入 `OURS`。源码未固定预期折数，也未检查列表非空。
2. 向 `MODELS` 加 `ours8`：`type=l24`、`path=B8`（Qwen3-8B-Base）、`keep=24`、`layers=[2,4,6,8,12,16,20,24]`、`w=1.5`、`bs=16`、`max_len=384`、`twin=0.0`。不同 adapter 的 plain log-probabilities 在既有 `l24_predict` 内等权平均；未给此成员设置 Gaussian 或双卡 `dp`。

沿用设置：主 SVC 权重 1.5，池为 `nlm_pool_v5.parquet`；`extra_w=0.5`、`tfidf_w=0.25`、`distort_w=0.10`；Sinkhorn 0.45、target mix 0.35；标点证据系数 1.5；ATLAS 概率混合 20%，互补栈 15%。这里的互补栈 15% 与 `ours8` 权重 1.5 是不同机制。

## 我们核查了什么

- 保存了固定 V1 原件，code SHA256 为 `f608191f7677839c48f8ebeee5482ee40a033af0d59763db2e108f11da6f809f`；完整文件哈希见采集凭据。
- 对照 SRC-002 生成完整 patch，除上述成员注册与输入扫描外，无代码改动；因此并非另一套独立基础算法。
- 未与当前正式 SRC 找到相同代码哈希或相同来源/固定版本，可新建条目并关联 SRC-002、SRC-001；适用验证方向关联 IDEA-004。
- API `datasetDataSources` 有一个空字符串，其余为已有公共输入；随后主窗口网页核对明确看见 `[Private Dataset]`。网页与 API 缺口共同表明依赖并非全公开，但不能仅凭空项把它的名称或内容猜成某个具体资源。继承的“All public”声明不适用于当前完整依赖集。
- Notebook 内嵌 `metadata.kaggle.dataSources` 的集合与 SRC-002 完全相同，而 API 输入清单多出空项。内嵌 metadata 很可能被继承；不能据它确认 ours8 附件已绑定。
- 下载的源码 Notebook 没有执行输出；网页另显示 10 个输出文件和 9m17s、T4×2 保存运行。没有读取或执行模型权重，未验证 adapter 数量、标签序、训练范围或完整隐藏评分时长；页面保存时长不能代替隐藏评分时长。

## 对当前方案的用途与限制

最有用的是一个清楚的加成员接口，可为 IDEA-004 提供小权重加入和互补性对照。固定 plain 特征/预测后比较单成员、原集成、加成员三者的错误重合，比根据文件名推断收益可靠。

复现前必须取得 `ours8_fold*` 的确切数据资源与版本、adapter/heads 哈希、标签顺序、训练数据与折隔离证明。源码对缺失 `OURS` 无提前断言；空 adapters 时预测累加器保留标量，后续按矩阵/成员最大值处理并不保证正常运行。这是静态依赖风险，未实际执行复现。

新增前向推理也消耗整条管线的时间预算，可能改变 ATLAS 实际覆盖的样本；即使代码其余不变，也不能假设所有下游预测严格保持不变。10 分钟级保存快路属于三条公开样例，不证明完整隐藏测试时长。

## 待验证想法

关联 IDEA-004：拿到可核查权重与训练范围后，以固定 prompt 分组验证和一致下游缓存，对比 ours8 权重 0、较小值与 1.5；同时记录 OOF 单成员、集成增量、错误相关性、耗时与 ATLAS 覆盖。未经权重溯源不登记为干净 OOF，本次未启动实验。

## 更新记录

- 2026-11-01：下载 V1，核对与 SRC-002 的两处增量，识别 `ours15` 的权重语义及继承正文；主窗口已绑定 364235852/0.664 并确认页面列出私有数据依赖，已正式归档。
