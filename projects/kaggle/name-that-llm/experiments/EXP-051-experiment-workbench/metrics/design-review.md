# EXP-051 统一入口设计审查

范围：只读检查现有控制器、quick-commit 验收、ledger 同步与 EXP-048 源绑定；没有远端调用、启动任务或执行训练。此文是实现前建议，不是新入口验收通过。

## 全局影响评估

固定 EXP ID → canonical record + 受控 registry → 独立远端只读观察或本地白名单操作 → 带身份的不可覆盖收据 → 中文报告。入口不应拥有远端 save/run/submit 能力，也不应把下载、核验、正式评分合成一个“成功”状态。

## 必须隔离的风险

1. **历史分数被未知响应清空。** `tracking/ledger.py:379` 对非 COMPLETE 或暂缺 publicScore 的返回写入 `public_score=None`。不要直接复用 `sync_kaggle` 作为单实验 status。新观察需独立保存；缺字段、查询失败、目标未出现在页内都表示 UNKNOWN。last confirmed score 和来源/时间保留，失败不能刷新其确认时间。缺 scriptVersionId 的响应不得产生新的版本绑定分数。
2. **样例 COMPLETE 被误当隐藏完成。** `control.py:300-310` 是 Notebook 保存运行状态；`verify_commit.py:26-38` 明确验证 public_placeholder、fast_commit=True，结果是 PASS_QUICK_COMMIT_ONLY。submission 状态、commit 状态、公开样例、合成一致性、OOF/外部验证必须分别呈现。正式 COMPLETE+分数只能证明平台完成评分，不等于下载到了 hidden 输出，也不自动证明全部模型/ATLAS覆盖。
3. **下载过滤不等于大小约束。** `control.py:313-337` 使用文件正则，SDK 另外自动下载 Notebook 日志，没有字节上限。统一 collect 应先只读列举全部页、检查精确文件名、路径、数量、单文件及总字节上限；下载时再流式硬限额，不能先整包下载再判断。若无法阻止 SDK 自动日志或提前约束体积，禁用该快捷下载分支，明确 NOT_SUPPORTED。
4. **成功与失败路径均可能暴露凭据。** `control.py:307` 保存 response_dict；`:361` 仅清理 URL query，不能处理 Authorization/Bearer/password 等字段。新入口应只保存允许的公开字段，签名 URL 只留在内存；从 URL 提取 scriptVersionId 后立即丢弃原地址。不要打印 SDK 对象、原始异常、stderr、完整日志或 requests PreparedRequest。安全错误码和脱敏消息足够供用户定位。
5. **泛化运行旧验收脚本会产生错误适用范围。** verify_commit 的 OUT、EXP-046、32输入、T4及样例标签均固定，不能作为 EXP-048/任意实验通用验收器。脚本用 assert，若 PYTHONOPTIMIZE 开启会跳过关键断言；run-local 必须禁止优化模式或选择显式 require 的受审计任务。运行旧脚本还会覆盖旧 quick-commit-verification.json，因此默认 verify 应读取并重新核对收据，不隐式重跑旧 verifier。
6. **时间与字段缺省不能伪造状态。** control.state 缺 notebook_ref 时默认接受当前 REF；update 会无条件刷新总 observed_at，不能据此推断 last_status 或 formal_score 刚核查过。registry/record ID、ref、version 必须同时匹配，缺必要字段失败关闭；本地快照标明“截至时间/本次未查询”，新 query_attempt 时间与 evidence_observed_at 分开。

## EXP-048 绑定与边界

固定记录为 Notebook `zhjianduan/name-that-llm-a1-codex120-1102` V2，scriptVersionId 364733823，submission 57893384。仅作为本次仓库所登记目标，不从记忆或标题猜测后续版本。`source-audit.json` 是 PASS_STATIC_SOURCE_AUDIT，明确没有执行 Notebook；绑定借助已有 SDK output-URL receipt，而不是该审计员独立取回历史源码。`inputs.content_hashes_verified=false` 与完整大输入尚未复核一致，不能因小型 new120 静态哈希通过而将它翻成 true。

下载新结果前固定上述目标。metadata/latest 变化、提交ID相同但脚本版本不符、下载receipt缺身份均应停止并保留诊断。把 public_placeholder 显示为“公开样例占位输出（非隐藏集预测）”；report 不覆盖 canonical conclusion 或 baseline。

## 应复用与实现建议

- 复用 ledger 的 records/schema、project_path 路径约束及 validate 思路；复用控制器独立观察目录、锁、SHA256、显式版本、无覆盖意图。不要复用混有 push_once 的 main 分发，也不要复制远端写操作进统一类。
- registry 只声明具体 EXP 的 allowed remote reads、固定版本、exact artifact allowlist、下载限额、known local task ID。用户仅输入 EXP ID，不输入任意 shell 命令、脚本路径、正则或远端ref。
- list/report 默认离线，status 明示只读查询；collect 只下载允许的小结果；verify 以身份、schema、完整性为主，未登记验收器返回 NOT_SUPPORTED 而非泛化 PASS。
- run-local 只允许独立审核的无网络、无模型、短时 CPU 任务；固定 argv、shell=False、超时、线程/输出上限、可写路径范围。CUDA_VISIBLE_DEVICES=-1 本身不是隔离沙箱，不能把任意训练脚本注册为“CPU安全”。此轮可先只注册 ledger check 等轻量任务。
- stdout 中文摘要应区分“状态查询成功”“已下载”“文件校验通过”“静态源码核查”“OOF”“正式Public LB”，并列证据时间和明确未验证项。

## 最小验收边界

需覆盖：未知 EXP；registry 与 record 不符；缺 source/版本；旧 RUNNING 快照；远端失败/缺分数/缺版本/错submission；COMPLETE但只有public_placeholder；缺文件/错哈希/重复文件/分页未完；路径穿越或绝对路径；声明大小不足但流实际超限；签名URL及Authorization出现在异常中；本地任务超时或非零退出。失败不得覆盖历史score、旧结果或制造PASS；所有测试应使用模拟传输，不消耗云额度。
