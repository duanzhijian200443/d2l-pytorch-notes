# Notebook 入口

[项目入口](../README.md) · [当前状态](../tracking/当前状态.md)

新开发 Notebook 放在这里，负责选择实验配置、调用共用代码、展示结果。训练和推理中可复用的逻辑逐步放到 [src](../src/README.md)，每次运行的预测和日志写进对应 [EXP 目录](../experiments/README.md)。

## 现有入口

| 用途 | 文件 | 使用范围 |
|---|---|---|
| 当前 VS Code / Colab 开发副本 | [name-that-llm-atlas20-repro.ipynb](name-that-llm-atlas20-repro.ipynb) | 已适配 Drive；这是当前唯一开发副本 |
| Drive 输入适配说明 | [atlas20-drive](../cloud/atlas20-drive/README.md) | 说明挂载、检查和目录映射 |
| EXP-004 固定基线源码 | [baseline-v2.ipynb](../reviews/own-scheme-readiness-20261030/baseline-v2.ipynb) | 历史证据；开发时不覆盖 |
| V32 Colab 网页实验入口 | [Who_am_I_V32_A100.ipynb](../cloud/who-am-i-v32-colab/Who_am_I_V32_A100.ipynb) | 历史云端实验入口，当前状态查台账；不是默认启动任务 |

开发副本已于 2026-11-01 从根目录移入本目录，文件字节保持一致；相关构建脚本和入口链接已更新。固定基线与历史云端入口保留原位置。未来如要迁移其他 Notebook，需同时更新构建脚本、相对路径和记录引用，并验证后再切换，不能长期维护两份都叫“最新”的文件。

新 Notebook 命名使用用途，例如 `stylegram-development.ipynb`；它可以服务多个 EXP。明确记录本次使用的实验编号，结果不只写在单元格输出中。
