# SRC-069 · Alex Chilton · All Public + What Failed V9

[正式来源记录](../../../tracking/records/sources/SRC-069.json)

Kaggle 把 Public **0.663** 绑定到 Alex 的 **V9 / 364580223**。这与 SRC-002 是同一 Notebook（ID `139599456`）的不同版本。V8 发布时的 slug 含 `0-662`，V9 的 slug 含 `0-663`。[固定得分版本](https://www.kaggle.com/code/alexchilton/name-that-llm-0-663-all-public-what-failed?scriptVersionId=364580223) · [版本与页面观察](snapshots/kaggle-page-observation.json)

## 来源与原件

2026-11-02 通过 Kaggle MCP 和官方 SDK 固定读取 `v9`。SDK 的 12 个输出路径均指向同一个全局版本号 `364580223`。下载时直接比较原始 UTF-8 字节，141003 字节全部等于 chyibo 的 `name-that-llm-0663-fork` V1。两条记录共用一份 Notebook 原件。全文件 SHA256 为 `07cb21ed6f9b2b2d0920756a8fea64ec5a5d37637d7ea4fad07ab6eb85ce5140`，代码单元 SHA256 为 `e3b8fa97495e45873b1297b782488819b4a04ac53cd08baa74e3637505b54605`。[下载与哈希凭据](receipt.json) · [V9 元数据](metadata.json)

## 作者声明与实际设置

Alex 的 V9 正文把这次运行归为 bgunshui Apex E 配方，报告自己得到 0.663。他还报告分别复现 judith 的 0.60/标点 1.2 组合得到两次 0.659，以及一次 One Swap 在隐藏评分时失败。后两项来自作者说明，本次只绑定了 Alex V9 的 0.663。

V9 延续主 LoRA 成员、互补栈 15% 与 ATLAS 20%，将主四视图 SVC 训练池切到 Open Pool，并将其融合系数从 1.5 调至 12。Sinkhorn 为 0.60，target mix 为 0.35，标点证据系数为 1.2，V40A 风格残差在标点步骤后按家族重排，强度上限 0.08。SRC-002 V8 使用 `nlm_pool_v5`、SVC 1.5、Sinkhorn 0.45 与标点 1.5。多项配置一起变化，两个版本的分差无法分配给单一参数。

## 我们核查了什么

- V9 挂载的 Open Pool 输入卡显示 **Kaggle V11 / 作者 v10**。日志显示排除 `flag_gemini37_routing_risk` 8297 行，写出筛过的 Open Pool **190210 行**。卡片声称原池 198507 行。日志计数发生在主 SVC 去重拟合之前。[历史输入卡和日志摘录](snapshots/kaggle-page-observation.json)
- 从 Alex 固定 `/9` 下载实际写出的 `infer.py`、`gauss.py`、`comp_stack.py`、`sibling_style.py`。四个 SHA256 分别与固定的 chyibo V1、synthreaper V1 和 bgunshui V1 同名文件相等。Alex 保存日志 `FAST_COMMIT True`，互补栈计时为 0；Kaggle 页面显示 8分17秒、T4×2。[四脚本与日志证据](receipt.json)
- chyibo V1 与 Alex V9 全文件相等，chyibo 的挂载版本为 V13，保存日志的相应计数为删除 0 行、写出 190588 行。两份保存版本存在可观察的数据差异。完整训练输入、隐藏评分中的动态 ATLAS 选行与实际覆盖均未逐行绑定。

## 对当前方案的用途与限制

这个出处把同源码的两次观察联系起来，也暴露出影响复现的输入版本变化。相同源码未提供二者的直接 Fork 操作历史，保存阶段的池差异也无法独自解释正式分差。和我们自己的 EXP-047 比较时，应分别保留 SVC池、融合系数、后处理和家族残差的差异。[我们的固定得分版本](https://www.kaggle.com/code/zhjianduan/who-are-you?scriptVersionId=364274017)

## 怎样验证

固定模型输入、其余参数与 ATLAS 完成行，先核 V11/V13 原文件、风险过滤、与官方/外部数据去重后的有效行数，再做只换池的对照。另对固定输入检查 SVC 1.5/12、标点/Sinkhorn 及残差开关。这些对照本次未运行。

[与 chyibo V1 共用的固定源码](../SRC-068-chyibo-name-that-llm-0663-fork-v1/source.ipynb)

## 同配方的固定得分版本

Alex V9（SRC-069，0.663）与 chyibo V1（SRC-068，0.665）整份 Notebook 字节相同。旧 Apex E（SRC-040）和 Alex 的输入卡均为 Open Pool 平台 V11，公开保存时各写出 190,210 行。两份 0.665 使用 V13，均写出 190,588 行。四个版本公开产出的四份推理脚本哈希一致。隐藏评分的完整输入、ATLAS 覆盖与逐行预测未取得，分差无法按组件分摊。[新版本输入与日志](../../../tracking/snapshots/20261102T071145Z-public0665/kaggle-browser-observation.json) · [旧版本输入卡](../../../tracking/snapshots/20261102T071145Z-public0665/historical-input-card-observations.json) · [V1 脚本比对](../../../tracking/snapshots/20261102T071145Z-public0665/fixed-v1-module-comparison.json) · [Alex V9 脚本回执](../SRC-069-alexchilton-all-public-v9/receipt.json)
