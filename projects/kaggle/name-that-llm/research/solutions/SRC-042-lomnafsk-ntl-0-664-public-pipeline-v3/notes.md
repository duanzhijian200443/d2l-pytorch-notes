# SRC-042 · lomnafsk V3：0.60/1.2与被覆盖的guard

原始记录：[正式台账](../../../tracking/records/sources/SRC-042.json)。
状态：固定源码已读，页面实际 0.659、与 V3 未绑定；已正式归档。


## 值得保留的结论

标题 0.664 与本轮页面 Score 0.659 不同。当前源码采用 Sinkhorn 0.60/标点 1.2，并试图启用仅 hard guard 的 `post=1`；但最终 typography 写回使用 guard 之前的快照，因此该 guard 在正常最终提交路径被覆盖。这是静态数据流发现，不是运行验证或分差归因。

## 来源与原件

- [Notebook](https://www.kaggle.com/code/lomnafsk/ntl-0-664-public-pipeline)；MCP 确认当前 V3，再经 CLI 固定拉取 v3，采集 2026-11-01 07:34:35 UTC。
- [source.ipynb](source.ipynb)、[API metadata](metadata.json)、[采集哈希](download-receipt.json)、[原文](description.md)、[代码](code.txt)、[对 SRC-002 的差异](diff-vs-SRC-002.patch)、[主窗口页面观察](snapshots/browser-observation-20261101.json)。
- 本轮（2026-11-01 UTC，逐页时刻未记录，采集截至07:35:57 UTC） 搜索列表显示 Score 0.659；没有查其历史得分版本，`bound_to_source=false`，scriptVersionId 未取得。

## 作者声称与实际设置

标题称 0.664；正文主要沿用基础流水线介绍，还写着旧的 Sinkhorn0.45/post0，与代码不一致。

相比 SRC-002，代码改三处：标点 1.5→1.2；命令 Sinkhorn0.45→0.60 且 `post=1`；去掉 post 块内 dense-cluster Hungarian，只保留 GUARDS。主 SVC 仍 1.5、pool-v5、ATLAS20%、互补栈15%，无新模型或新公共池。

## 我们核查了什么

- code SHA256 `ea3905785055fd2ed8517531f09c04574b96c8a6111fb6d51ffc3ff3b3c51cd9`，未与既有 SRC 完全相同；实际后处理参数对应 SRC-022 的0.60/1.2，而非 SRC-021 得分V2。
- `code.txt` 中先在 post 前存 `_v38_before_hard = tot.copy()`，随后 GUARDS 改 tot；最终 `if not a.sim` 内用 `_raw = _v38_before_hard.copy()` 重新计算 `_safe_logits` 并覆盖同一 submission 文件。故普通完整最终输出的 typography 不继承该 guard。模拟路径未进入该写回块，不能把两路径的结果等同。
- 源码已固定，模型权重、实际输入版本和完整隐藏运行未验证。本次未执行源码。

## 对当前方案的用途与限制

适合作为“标题、源码、得分版本必须拆开”的实例，以及后处理覆盖次序的静态反例。页面0.659不能归因为 guard、Sinkhorn 或标点中的任一项；即使静态预测路线与旧组合接近，运行预算和环境也可能改变下游覆盖。

## 待验证想法

关联 IDEA-002：用冻结预测构造明确的 post→typography 顺序对照，先核验最终保存产物是否真正包含改动，再评价分组验证。未据此创建训练或提交。

## 更新记录

- 2026-11-01：固定 V3，保留实际页面0.659但不绑定V3，记录正文过期与post改动被覆盖的静态发现。
