# SRC-041 · bgunshui One Swap V1：原作者V3同代码副本

原始记录：[正式台账](../../../tracking/records/sources/SRC-041.json)。
状态：固定源码已读、同源关系已核查；页面未显示分数，已正式归档。


## 值得保留的结论

本条完整代码与 zbigniewkirchner One Swap **V3 完全相同**，保留作者出处，但不计为独立方法。正文的 0.664、权重阶梯与 SVC CV 说明也不能直接当作此副本自己的评分记录。

## 来源与原件

- [Notebook](https://www.kaggle.com/code/bgunshui/ntl-one-swap)；MCP 确认当前 V1，再经 CLI 固定拉取 v1，采集 2026-11-01 07:34:36 UTC。
- [source.ipynb](source.ipynb)、[metadata.json](metadata.json)、[采集哈希](download-receipt.json)、[description.md](description.md)、[code.txt](code.txt)、[对 SRC-002 的差异](diff-vs-SRC-002.patch)、[对 One Swap V1 的差异](diff-vs-original-one-swap-v1.patch)、[对 One Swap V3 的空差异](diff-vs-original-one-swap-v3.patch)。
- 主窗口 本轮（2026-11-01 UTC，逐页时刻未记录，采集截至07:35:57 UTC） 实读比赛 Code 搜索列表，此项**没有显示 Score**；见 [观察快照](snapshots/browser-observation-20261101.json)。`public_score=null` 不表示零分或失败；本次未绑定 scriptVersionId。

## 作者声称与实际设置

正文继承 One Swap 的说明：将主四视图 SVC 的 `nlm_pool_v5` 换为 Open Pool，并把 `svc_w` 从 1.5 调到 12。正文列作者权重阶梯 1.5/6/12/20 对应 0.664/0.659/0.664/0.664，并称 193k 数据版上单 SVC 五折 prompt-grouped CV 从 0.4805 到 0.5008；这是正文自报，未见本副本单独的训练/提交证据。

源码读取 `ntl_open_pool.parquet` 的 text/model/flag_train_prompt_regen，排除重建官方 prompt 的标记行后给主 SVC；打印的 `len(_op)` 是过滤前行数，不能当成实际训练行数。未新增 Apex E 的 Gemini routing-risk 条件过滤或家族残差。沿用 Sinkhorn 0.45、标点 1.5、ATLAS20%、互补栈15%、`post=0`，其余成员未重训。

## 我们核查了什么

- 完整 code SHA256：`d8c630d3b9fd1b3b0453de6fffb9e5d613f7a9c1c79cf47ab42f7dd9ad30321f`，与本轮原作者 V3 相同。相对原作者 V1 只去掉绘图单元、改两处署名注释，不形成新预测方法。
- API 输入列有 `crtproduction/name-that-llm-open-pool`，未取得此副本附件精确数据版本或内容哈希。源码公开不表示各依赖许可统一，代码/数据/Llama 许可需分别保留。
- 本次只读下载与静态比较；未执行训练或评分，未独立验证正文的 CV、耗时或分差。

## 对当前方案的用途与限制

用于核对原作者版本扩散与复现依赖，方法分析复用 One Swap 原条目。不要把同代码副本增加到“独立有效改进数量”，也不要用继承正文的 0.664 填补未显示的页面分数。

## 待验证想法

关联 IDEA-003：固定精确 Open Pool 版本、prompt 隔离与来源去重后，分别比较换池和 SVC 权重。主 SVC 耗时变化会挤占 ATLAS 覆盖，应同时记录。本次未创建实验。

## 更新记录

- 2026-11-01：固定 V1、与原作者 V3 做完整代码哈希去重；记录本页未显示分数。原作者 V3 已登记为 SRC-037，并已建立同源关联。

关联来源：[SRC-002](../../../tracking/研究索引.md#src-002) · [SRC-027](../../../tracking/研究索引.md#src-027) · [SRC-037](../../../tracking/研究索引.md#src-037)。
