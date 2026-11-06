# 续采已恢复：实时观察

北京时间2026-11-01 20:56，官方私有Benchmark云端Luna和Terra均已实际成功返回，模型标识匹配、finish_reason=stop、费用字段完整。首条Luna为$0.0003778，首条Terra为$0.001414。

本次观察累计保存12条、累计派发15/80，当前RUNNING且本轮失败0；已确认费用$0.0025668，历史2次费用仍未知。既有2条成功原文哈希未变，没有重复成功作业编号。以上仅是实时观察，不是全量采集完成或训练效果验证。

当前run_id：`75230ec19d8d4d4daf6e771359fd475b`，实时controller_pid以collection/state.json为准。后台继续串行执行，原预算和上限不变；自动化15按用户此前要求每15分钟只读汇报，完成或停止后自删。更多逐模型凭据见metrics/recovery-live-verification.json。

以下为修复与恢复设计。

# EXP-038 私有任务查询403兼容修复与续采

用户授权保持原条数和预算，排查修复后续采。EXP035/036两个Kaggle请求未得到成功回答，费用仍未知；EXP035与EXP037各保存1条Codex，二者不重采。EXP037第三次Kaggle尝试仅在GET私有任务查询阶段返回403，尚未发出创建POST/模型请求。这一点由冻结源码的stage顺序、原失败记录、缺少cloud-task/run/result/raw产物，以及官方SDK已知私有任务可读/目标不存在的只读核验共同支持。

本次只将该GET-only失败重新置为可执行；它的原记录不删除、不篡改。原state的5个控制器尝试意图中4个属于实际/未知模型派发，1个只读失败。按原80次模型请求上限最多再发76次（Codex38、Kaggle38）。若全部成功，累计78条成功、80次模型派发、19个完整题目×模型单元。未为凑齐样本补发历史2次真正的API失败。

传输修复遵循KaggleApi._get_benchmark_task(allow_not_found=True)已有逻辑，仅在查询尚不存在任务时将403/404当作未找到；创建请求仍独立鉴权、只发一次、无重试。401及其他查询错误仍停止。官方Benchmarks私有CPU执行、1个指定模型验证阶段、无缓存/额外judge/另行task run；其余提示词、模型、参数及顺序保持原计划。

仍仅用现有免费额度，0.25美元目标与累计已确认费用0.10美元停发线不变。历史2次费用仍未知，新的未知结果/费用立即停发；不能把软阈值称为总费用硬上限。不会训练、提交比赛、公开资源或改动其他任务。

recovery-receipt.json与manifest.json绑定原件、快照、GET-only失败及源码；collection/state.json为实时控制状态，records.jsonl仅新成功，combined-records.jsonl终态关联历史成功，metrics/collection-audit.json为最终审计。
