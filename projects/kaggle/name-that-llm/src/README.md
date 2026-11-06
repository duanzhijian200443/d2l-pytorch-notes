# 共用代码

[项目入口](../README.md) · [配置模板](../configs/README.md) · [实验产物](../experiments/README.md)

后续多个实验共用的数据读取、特征、训练、推理和融合函数放在这里。只有需要复用的逻辑才提取成模块，不预先创建没有实现的 `train.py`、`inference.py` 或 `ensemble.py`。

共用代码接收明确的配置和输入路径，结果写入指定 EXP 目录；模块被 import 时不自动训练、不联网下载、不分配 GPU。一次实验独有的分析脚本可以留在该实验目录中。

现有可复用能力继续使用原入口：

- 日常实验入口：[experiment_workbench.py](experiment_workbench.py)，由根目录`experiment.py`调用；固定四个对象的只读云端状态/小文件下载、已有结果核验与中文报告。用法见[EXP051指南](../experiments/EXP-051-experiment-workbench/guide.md)。不提供云端写入/提交。
- 公共池逐行审计：[open_pool_audit.py](open_pool_audit.py)，按冻结config核验文件、重建实际用数与候选索引；[EXP050](../experiments/EXP-050-open-pool-v13-increment-audit/report.md)已执行，`--verify`只读复核，不把候选批准为训练数据。
- 固定分折与抽样：[validation/protocol.py](../validation/protocol.py)，规则见 [验证协议](../validation/README.md)。
- 实验登记与目录生成：[tracking/ledger.py](../tracking/ledger.py)。
- Drive 输入映射：[cloud/atlas20-drive](../cloud/atlas20-drive/README.md)。
- 历史报告的本地链接迁移：[report_links.py](report_links.py)，供归档和历史报告生成器使用；不执行训练或云端操作。
- 主 svc4 四视图与 LinearSVC：[svc4.py](svc4.py)，由 EXP-009 固定源码提取；[test_svc4.py](test_svc4.py) 比较合成输入上的参考行为。EXP-017～020固定 sklearn 1.6.1；不能默认与1.8的balanced/sample_weight行为相同。
- OOF互补性统计：[oof_complementarity.py](oof_complementarity.py)只接收已对齐预测，计算错误重叠、逐类/折/prompt指标及prompt配对区间，不训练或产生融合预测；输入字节与场景路由由调用方核验。EXP-022使用记录见[报告](../experiments/EXP-022-oof-complementarity/report.md)。
- 固定margin融合：[margin_blend.py](margin_blend.py)逐行统一分数尺度后加权，不读取标签、拟合参数或输出校准概率；[test_margin_blend.py](test_margin_blend.py)覆盖端点、极值、常数行和输入约束。开发筛选结果见[EXP-024](../experiments/EXP-024-oof-margin-blend/report.md)。
- 完整集成缓存：[full_stack_cache.py](full_stack_cache.py)保存带身份和哈希的原精度数组；[full_stack_observer.py](full_stack_observer.py)只观测固定源码；[full_stack_replay.py](full_stack_replay.py)独立重放缓存数学。使用与边界见[缓存入口](../cloud/full-stack-cache/README.md)；合成工程一致性不代表已有真实全模型预测或干净 OOF。
- 固定Sinkhorn对比：[sinkhorn_compare.py](sinkhorn_compare.py)只计算0.45控制与0.30候选，使用同一上游概率及原标点公式，各自重算门控；模型/文件操作和验证集真标签不进入数学模块。实际执行的完整性门槛见[EXP-032](../experiments/EXP-032-sinkhorn-fixed-cache/report.md)，未取得真实缓存前只完成合成测试。

已经验收的 EXP-010/011/012 训练脚本及原公开源码仍是历史证据，不因目录整理而移动或改写。后续需要复用时，从固定来源提取实现，在新实验中绑定共用代码的版本/哈希并做必要的行为检查。
