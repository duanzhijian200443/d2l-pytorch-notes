# EXP-023 · Codex 订阅原文试采重试

执行状态以 [实时控制器状态](collection/state.json) 为准。四个模型首轮各一条已成功，后台续采已启动；终止时自动更新下方审计结果。

用户授权 1,000 条、最多 2 并发。四个目标模型各回答同一组 250 个原创英文提示词；200 train 组和 50 holdout 组，共预计 800/200 条。使用现有 ChatGPT 订阅认证，未调用付费 API、GPU、Kaggle 或 Google Cloud。

EXP-021 在本地配置阶段失败，零事件流、零回答。本次移除内置 provider 不允许的覆盖，保留原失败证据。输入复用 [原提示词](../EXP-021-codex-subscription-pilot/prompts.jsonl) 与 [最小基础指令](../EXP-021-codex-subscription-pilot/model-instructions.txt)，本次代码另行冻结到 code/。

四模型请求 slug：gpt-5.6-luna、gpt-5.6-terra、gpt-5.6-sol、gpt-6-astra；显式 low reasoning。底层 snapshot 未独立确认。采集器自动重试为零，CLI 内部重试行为未证实。任一失败即停止新调度，保留在途结果；不自动重发失败或未定任务。

[预检](../EXP-021-codex-subscription-pilot/runtime-preflight.json) 发现可能残留用户级 AGENTS、权限和工具描述。debug 与 exec(ignore-user-config) 路径不完全等价，不宣称裸 API 等价；任何实际工具调用会拒收并停止。每条保留原文、usage、哈希、来源及请求参数；字数异常只标记，不改写。

离线控制器 11 项测试通过。先采每模型一条，确认成功后断点续采至总计 1,000。数据在 collection/，本实验不训练、不提交、不公开；采集完成不代表数据已有验证增益。

## 自动收尾结果

观察时间：2026-11-01T08:25:32.613319+00:00。状态：**STOPPED_INCOMPLETE**。保存 **4/1000** 条。

| 请求模型 | 原始回答数 |
|---|---:|
| gpt-5.6-luna | 1 |
| gpt-5.6-sol | 1 |
| gpt-5.6-terra | 1 |
| gpt-6-astra | 1 |

train/holdout：{'train': 4}；观测最大请求并发：2；字数等标记：{}。

[完整性审计](metrics/collection-audit.json) · [原始数据](collection/records.jsonl) · [控制器结果](collection/summary.json)

这些是 Codex 订阅管线原文；没有进行训练、榜单提交、公开发布或 API 等价性验收。
