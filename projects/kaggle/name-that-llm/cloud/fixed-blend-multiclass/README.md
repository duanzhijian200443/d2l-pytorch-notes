# 固定10%融合 · 多家族外部检查

本控制器对应[EXP-027](../../experiments/EXP-027-multiclass-external-validation/report.md)。运行与验收状态以[execution-state.json](execution-state.json)为准。

Notebook：`zhjianduan/name-that-llm-fixed-blend-multiclass-027`。V1仅保存；V2为唯一CPU执行，私有、断网、原容器；Dataset固定私有V1。输入729条/56个规范化题干组，17个被覆盖真类，预测保留18类。

代码和输入在启动前通过[冻结协议](payload/protocol.json)绑定。内部3300秒、平台3600秒、RSS12GiB，2CPU线程；不自动重试或开启GPU。`runner.py`和payload保持冻结，后续尝试另建EXP。

下载V2产物完成后，可以离线核对：

```powershell
python -B -X utf8 cloud/fixed-blend-multiclass/verify_outputs.py --input cloud/fixed-blend-multiclass/outputs-v2/results --check-only
```

配对区间按真实题干组抽样，真标签只用于计分与统计；模型推断分组只看未标记文本。指标是17真类宏召回，不能称完整18类BA、完整ATLAS20或榜单成绩。完整集成接入方案见[分离对照设计](../../experiments/EXP-027-multiclass-external-validation/integration-plan.md)。
