# EXP-031 · 完整ATLAS20缓存重跑：修正可选成员预检

状态：**Kaggle 第2版完成，真实缓存验收通过**。最新观测及运行标识见 [execution-state.json](../../cloud/full-stack-capture-r1/execution-state.json)。

- 台账原始记录：[EXP-031](../../tracking/records/experiments/EXP-031.json)
- 实验总览：[实验台账](../../tracking/实验台账.md#exp-031)
- 对照基线：`EXP-004`；登记时间：`2026-11-01T11:04:13.572772+00:00`。
- 参数来源：[config.json](config.json)；文件清单：[manifest.json](manifest.json)。

## 假设与主要改动

修正 EXP-029 的可选成员预检错误。旧运行器错误地强制要求原算法允许省略的 public LoRA。新本地 CPU/GPU 共用路径解析规则，按同一531文件清单得到20个comp成员目标，省略public/public_twin/Q35。其他成员、权重、分组与后处理不变；20成员目标仍需至少一个有效comp分组，不造组或重新分配权重。

## 运行前核验

复用已完成的 CPU 全文件哈希：531文件、190.823GB、清单 SHA256 `de7a305b5d26a5ad0cca0ed41cf31e2af1f85037dd354286aeef77ac67aaabf5`。729条盲化原文和31个固定附件不变。GPU端核对全部路径/大小并重算小文件哈希；不是在GPU重新全量哈希。

共享策略13项测试、运行器14项测试、独立缺失/新增路径检查通过。新策略拒绝真正缺失的必需输入和输入清单漂移。原main/comp等算法文件与EXP029同字节，运行预算/进程退出/归档逻辑的AST对照通过。见 [package-preflight.json](metrics/package-preflight.json)、[fix-validation.json](metrics/fix-validation.json) 和 [独立复核](metrics/data-reuse-and-input-policy-review.json)。

## 执行与结果

Notebook `zhjianduan/name-that-llm-cache-capture-031`：第1版保存回读后，第2版启动完整运行。源码、31附件、私有、禁网、T4×2配置与原容器镜像已回读。实际设备与模型阶段现已通过输出核验，详情见下方收尾记录。工作上限28200秒，平台上限28800秒；不自动重跑，不正式提交比赛。

旧 EXP-029 的第3版ERROR、失败日志和代码均保留，本次作为独立重跑记录，不改写旧实验结果。

## 结论、限制与下一步

输入解析、组件分数、ATLAS覆盖、重放及Parquet格式已通过核验；固定后处理比较与诊断指标见EXP032。尚无新的榜单提分证据。729条面板已经查看过且仅覆盖17类，全成员训练隔离仍未知，不是独立确认、干净OOF或榜单成绩。

## 实际完成与收尾（2026-11-01）

Kaggle官方CLI实时确认固定V2 COMPLETE。运行于北京时间22:09:21结束；守护总耗时10654.553秒（约2小时57分35秒），setup523.070秒、comp3840.104秒、main6138.553秒。此为执行墙钟耗时，不是精确计费GPU小时。

实际T4×2；main16、comp20、family gate5均匹配预定清单；comp适用728/729行，ATLAS206/206请求行完成。143项缓存重放检查最大误差0。全部归档与稳定产物哈希通过EXP032入口验收，未发现新增遗漏或异常降级。public/public_twin/Q35按预先固定的原算法可选分支跳过。

最终result、guard与state确认完整结束；官方查询不再运行，账户quota返回reserved0。缓存已保存本地[输出](../../cloud/full-stack-capture-r1/outputs-v2/results/)，[下载收据](metrics/download-receipt.json)独立固定清单SHA。保持未正式提交、无重开GPU；0.45/0.30真实统计见[EXP032报告](../EXP-032-sinkhorn-fixed-cache/report.md)。
