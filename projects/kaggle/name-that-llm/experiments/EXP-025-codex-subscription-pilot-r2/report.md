# EXP-025 · Codex 试采续采与本地状态写入修复

执行状态以 [共享控制器状态](../EXP-023-codex-subscription-pilot-r1/collection/state.json) 为准；本次续采已有 6 条，最多再生成 994 条。最终以本页自动收尾与 metrics/collection-audit.json 为准。

用户授权的总量仍是 1,000 条、最多两个并发。采样配置和提示词没有改变，继续使用 EXP-023 的冻结配置、EXP-021 的 250 个原创提示词与最小基础指令。四模型各250条，200 train/50 holdout提示词组，共800/200条。

EXP-023续采在 Windows state.json 原子替换阶段发生 PermissionError。已有6条全部真实成功，两条已从各自原件核验哈希和turn.completed后补入聚合索引，没有重发模型请求。首次smoke与失败运行的原件均保留。

本次仅修复本地文件写入：对 Windows 短暂占用做有界磁盘重试；状态写入失败则停止新调度并收集已在途结果。模型请求自动重试仍为零。收尾同时核对退出码、本次run_id、state/summary目标及保存数、1000个成功attempt、原文哈希、分组和并发，禁止误用旧smoke终态。

[原文数据](../EXP-023-codex-subscription-pilot-r1/collection/records.jsonl) 保持一个位置，不重复复制。配置/代码版本见 manifest.json，前次恢复见 EXP-023/metrics/manual-reconciliation.json。

请求模型标识与底层snapshot不等价；残余用户级指令/权限提示的可能性保留。数据是Codex订阅来源，尚未训练或确认验证收益。本任务不提交比赛、不公开数据、不使用Google/Kaggle或付费API额度。需要本机持续运行及联网。

## 自动收尾结果

观察时间：2026-11-01T10:14:33.494345+00:00。状态：**STOPPED_INCOMPLETE**。保存 **558/1000** 条。

| 请求模型 | 原始回答数 |
|---|---:|
| gpt-5.6-luna | 140 |
| gpt-5.6-sol | 139 |
| gpt-5.6-terra | 140 |
| gpt-6-astra | 139 |

train/holdout：{'train': 558}；观测最大请求并发：2；字数等标记：{}。

[完整性审计](metrics/collection-audit.json) · [原始数据](../EXP-023-codex-subscription-pilot-r1/collection/records.jsonl) · [控制器结果](../EXP-023-codex-subscription-pilot-r1/collection/summary.json)

这些是 Codex 订阅管线原文；没有进行训练、榜单提交、公开发布或 API 等价性验收。
