# SRC-044 · maverick V9：输入诊断、失败保底与预测缓存

原始记录：[正式台账](../../../tracking/records/sources/SRC-044.json)。
状态：固定源码与重构说明已读；页面实际0.651未绑定V9，已正式归档。


## 值得保留的结论

V9主要是输入诊断、参数集中、失败保底、超时控制和成员预测保存的工程重构。正文声称默认严格复制0.664配方，但实际默认 Sinkhorn0.60、标点1.2 对应 SRC-022，而非已经绑定0.664的 SRC-021 V2。因此不能将这些工程改动表述为0.664的验证配方。

## 来源与原件

- [当前Notebook](https://www.kaggle.com/code/maverickss26/kaggle-name-the-llm-v4)；MCP 确认V9，再经CLI固定拉取v9，采集2026-11-01 07:34:36 UTC。
- [source.ipynb](source.ipynb)、[metadata](metadata.json)、[采集哈希](download-receipt.json)、[原文](description.md)、[代码](code.txt)、[对SRC-002差异](diff-vs-SRC-002.patch)、[对该作者旧V8差异](diff-vs-own-v8.patch)、[页面观察](snapshots/browser-observation-20261101.json)。
- 旧证据 `reviews/public-top20-20261029/13-maverickss26/` 记录 ref 为 `maverickss26/kaggle-name-the-llm-v3`、版本8；其 kernel id **139393444** 与当前v4 slug相同。是同一个Notebook改名后的新固定版本，现有正式SRC尚未登记旧条目，不重复复制旧证据。
- 主窗口本轮（2026-11-01 UTC，逐页时刻未记录，采集截至07:35:57 UTC）比赛Code列表显示Score0.651，未核对具体得分版本，`bound_to_source=false`。

## 作者声称与实际设置

作者称只调整工程结构、输入/失败处理及潜在post覆盖问题，默认模型和混合与“0.664 notebook”完全相同；同时承认没有分组OOF验证。其正文估计的噪声阈值和两次最终选择建议均是作者观点，本次不采纳为比赛官方规则或通用统计准则。

实际设置及工程改动：

- 集中 `W` 与运行配置，SVC1.5、ATLAS20%、互补栈15%、Sinkhorn0.60/target0.35、标点1.2、POST0。`FULL_COMMIT_TEST=False` 默认仍为公开样例快路。
- 删除未使用输入的硬要求，汇总缺失输入；Llama校验公共与官方来源分片大小；无法找到匹配Llama时**删除ll8v5成员继续运行**，所以依赖异常时不是完整相同模型。
- 先从互补栈写保底提交；若其不存在，写全部gpt-5.5的占位。主阶段输出到另一路径，仅退出码0才覆盖最终提交。时间预算默认9小时，watchdog在安全余量15分钟前停止主阶段；这是代码配置与作者假设，未独立核对当前比赛限时。
- 增加 `--v31_typo`；`if a.post: _v38_before_hard=tot.copy()` 让开启post时的改动不被后续typography覆盖。默认POST0时此修复不生效。
- 保存 `members_logprob.npz`（ids、labels、before_typography及LP字典），用于后续离线混合；缓存/DeBERTa副本移到temp路径。该缓存不是训练隔离的OOF证据。

## 我们核查了什么

code SHA256 `011698096a6d1f2b46f1b4199c941884c77e0ddf1fe947f8834797476d7bb12f`，未与正式SRC完全相同。保留与旧V8及SRC-002的完整差异，并核实同kernel id的改名关系。

未执行模型、watchdog或失败分支，没有独立验证时长、缓存数值、保底分数或正式隐藏评分。页面0.651与正文0.664是不同证据，且前者尚未绑定V9。

## 对当前方案的用途与限制

输入诊断、明确阶段状态、分成员缓存与post写回修复值得参考，主要对应工程可靠性和IDEA-002的缓存前提。保底“有文件”可能掩盖主集成失败：必须读取 `STAGE B` 与保底类型，不能把生成submission当成完整方案运行成功。Llama被删除、互补栈缺失或主阶段超时都会改变实际配方。

## 待验证想法

若未来采用其工程思路，先固定本项目真实基线参数，给主阶段成功、互补栈保底、常量占位分别记录状态；用小型固定预测验证post/typography次序和缓存标签行序，再考虑GPU运行。不要把继承的0.664标题作为验收目标。

## 更新记录

- 2026-11-01：固定V9、关联同kernel的旧V8，保留实际页面0.651未绑定，记录默认参数归属、Llama删除与失败保底限制。
