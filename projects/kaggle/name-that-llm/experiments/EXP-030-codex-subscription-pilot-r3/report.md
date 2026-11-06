# EXP-030 · Codex 网络中断后的断点续采

用户在本聊天明确要求“继续采集”。原目标仍为1000条、四模型各250条、最多2并发；本次已保存558条，最多补442条。已采集数据不重新生成，不训练、不提交、不公开，不使用付费API。

实际采样配置继续使用EXP-023/config.json；250题和最小基础指令继续使用EXP-021原件。冻结collector与EXP-025字节一致，无采样逻辑改变。恢复前成功记录前缀已固定并在开跑/收尾时核验。

前次cp0140的Sol和Astra请求均未保存最终文本、agent message或turn.completed。旧控制器和请求子进程均已退出，run.lock不存在。两份失败原件已移动到本EXP的recovery/failed-attempts/并逐文件核验哈希，旧路径映射、原run_id与用户续采授权均在[恢复凭据](metrics/resume-preflight.json)。558个成功attempt和records.jsonl原字节保持不变。仅这两个失败请求在用户本次指示下重发；新失败不自动重试。

DNS解析恢复，chatgpt.com匿名HTTPS HEAD返回403，说明连接可达，不表示模型授权已验证；模型访问由本次前两个实际请求验证。

状态以[共享state.json](../EXP-023-codex-subscription-pilot-r1/collection/state.json)为准。原始数据仍在[records.jsonl](../EXP-023-codex-subscription-pilot-r1/collection/records.jsonl)。本次收尾审计位于metrics/collection-audit.json，并包含历史两次失败的并发与证据核验。

台账登记时存在不相关的EXP-029 source hash mismatch，标准new-experiment全局检查因此退出。使用ledger现有create_experiment函数独占创建新记录和目录；未改EXP-029或手改生成视图。[登记校验记录](metrics/registration-validation.json)。check/render仍执行并如实保存结果，不声称全局通过。

已于北京时间18:20启动。观察时间2026-11-01T10:21:55.723266+00:00：RUNNING，保存563/1000条；原cp0140的Sol/Astra两条请求均已成功，原始输出及哈希通过核验，最多2并发。每15分钟监控已恢复（15-codex）。Codex请求模型标签不等于底层snapshot；数据保留Codex订阅来源标签，训练收益和API等价性未验收。

## 自动收尾结果

观察时间：2026-11-01T11:33:42.587268+00:00。状态：**COLLECTED_UNVALIDATED**。保存 **1000/1000** 条。

| 请求模型 | 原始回答数 |
|---|---:|
| gpt-5.6-luna | 250 |
| gpt-5.6-sol | 250 |
| gpt-5.6-terra | 250 |
| gpt-6-astra | 250 |

train/holdout：{'train': 800, 'holdout': 200}；观测最大请求并发：2；字数等标记：{}。

已核验原有558条数据前缀保持不变。前次失败2次的原件已归档并校验哈希；并发审计包含这些历史尝试。[恢复凭据](metrics/resume-preflight.json)。

[完整性审计](metrics/collection-audit.json) · [原始数据](../EXP-023-codex-subscription-pilot-r1/collection/records.jsonl) · [控制器结果](../EXP-023-codex-subscription-pilot-r1/collection/summary.json)

这些是 Codex 订阅管线原文；没有进行训练、榜单提交、公开发布或 API 等价性验收。
