# 实验目录

[项目入口](../README.md) · [实验台账](../tracking/实验台账.md) · [默认配置](../configs/README.md)

每次独立尝试使用一个 EXP 编号。该实验的配置、预测、指标、日志和报告放在同一个目录里，通过台账索引找到；不建立混装所有实验结果的全局 `oof/` 或 `logs/`。

## 新建一次实验

在项目根目录执行以下命令。它只登记计划、生成目录并刷新本地总览，不执行训练。

```powershell
python -X utf8 tracking/ledger.py new-experiment --title "实验名称" --slug experiment-name
```

默认使用 `configs/experiment-defaults.json`。需要其他模板时，添加 `--config configs/你的配置.json`，路径相对于项目根目录。`--slug` 可省略；填写时只使用小写英文、数字与连字符。新编号由现有台账分配，不手工复用旧 EXP 编号。

生成结构：

```text
experiments/EXP-编号-experiment-name/
  config.json       当次参数快照；草稿阶段先完善
  manifest.json     文件清单；初始 DRAFT，不代表已核验输入
  report.md         假设、执行范围、结果、限制和下一步
  oof/              逐行预测、分数与分折缓存
  metrics/          汇总、逐类、误差分析和验收结果
  logs/             运行日志与报错
```

同时建立 `tracking/records/experiments/EXP-编号.json`，关联实验目录、配置和清单。`config.json` 是新实验的参数来源；台账负责状态、结果和证据索引。不要再在其他“最新记录”文件中维护一份独立的状态。

## 开跑前与收尾

1. 完善台账的假设、对照和唯一主要改动，以及 `config.json` 的模型、参数、环境与预算。
2. 在 `manifest.json` 中绑定真实代码版本/哈希、数据与权重位置/版本/哈希、分折与标签顺序；检查通过后才固定这次运行配置。刚生成的文件清单只是草稿。
3. 保存 OOF 的样本 ID、类别顺序、折号与分数类型。日志、数值和报告都必须明确属于哪个 EXP；结果未知时留空。
4. 验收后更新台账，再运行 `python -X utf8 tracking/ledger.py check` 与 `render`。状态和分数总览由脚本生成。

## 现有实验

- [EXP-007：验证基础](EXP-007-validation-foundation/report.md)
- [EXP-008：公共池审计](EXP-008-public-pool-audit/report.md)
- [EXP-009：基线用数追溯](EXP-009-baseline-data-lineage/report.md)
- [EXP-010/011/012：Stylegram 权重对照](EXP-010-012-stylegram-share/report.md) · [完整验收](EXP-010-012-stylegram-share/acceptance-20261101/README.md)

已有实验目录保持原样。EXP-010/011/012 共享一次固定 sweep 的产物目录，并由三条台账分别关联，不拆出重复文件。历史原件与派生分析各自保留，下载包不冒充验收结果。

大权重继续使用既有 Drive/Kaggle 资产，目录中记录位置和内容标识。小型 OOF、指标、配置和日志保留本地副本及必要的云端备份；备份位置写入清单，不为每个实验重复复制大模型。
