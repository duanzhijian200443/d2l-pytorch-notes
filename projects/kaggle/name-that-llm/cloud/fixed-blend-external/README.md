# 固定10%融合 · 外部验证控制器

EXP-026已完成，结果与解释见[实验报告](../../experiments/EXP-026-independent-blend-validation/report.md)。私有Notebook `zhjianduan/name-that-llm-fixed-blend-external-026` 的V1仅保存，V2为唯一CPU执行，终态COMPLETE，无GPU/互联网/API支出或正式提交。

- [执行状态](execution-state.json)、[最终V2回读](complete-v2-readback.json)：终态与运行来源。
- [冻结协议](payload/protocol.json)、[开跑预检](preflight.json)：代码/输入/环境和预算。
- [原始下载](outputs-v2/)、[验收结果](analysis/verified.json)：远端产物与本地复算。
- [按实验归档清单](../../experiments/EXP-026-independent-blend-validation/metrics/archive-manifest.json)：本地小型结果副本及哈希。

离线复核，不启动训练：

```powershell
python -B -X utf8 cloud/fixed-blend-external/verify_outputs.py --input cloud/fixed-blend-external/outputs-v2/results --check-only
python -B -X utf8 -m unittest discover -s cloud/fixed-blend-external -p test_verify_outputs.py -v
```

`runner.py`及payload已冻结，不在原目录改算法重跑；后续尝试需新EXP。指标仅为四个真类上的宏召回，预测仍在18类中选择；不能称为完整18类BA或榜单结果。
