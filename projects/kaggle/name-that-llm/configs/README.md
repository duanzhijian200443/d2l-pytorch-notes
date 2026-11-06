# 配置模板

[项目入口](../README.md) · [新建实验](../experiments/README.md)

这里放可复用的配置模板。默认模板是 [experiment-defaults.json](experiment-defaults.json)，包含已使用的验证协议与 seed；模型、环境和资源上限留空，表示尚未确定，不表示无限预算或已获运行授权。

[experiment-workbench.json](experiment-workbench.json)是日常工具注册表：固定EXP046/048的远端版本/提交/源码哈希，EXP049/050的本地任务及代码哈希，以及本机已有解释器路径。它不包含凭据，不是模型训练参数；当前交付快照由EXP051的manifest绑定。新增对象或更换环境要重新核验，不可直接改哈希绕过失败。用法见[命令指南](../experiments/EXP-051-experiment-workbench/guide.md)。

创建实验时把选定模板复制到 `experiments/EXP-编号-名称/config.json`。该文件是这次实验的实际参数来源；以后修改本目录模板，不会自动改变已创建的实验。

- 模型参数放 `model.parameters`，样本训练权重与最终融合权重使用不同字段。
- 数据来源和字节哈希放该实验的 `manifest.json`，凭据放安全存储。
- 配置在实验启动前完善并固定；启动后修改主要参数或独立重跑，要另建 EXP，保留原始快照。
- 清单初始为 `DRAFT`。存在配置文件不等于数据、代码和环境已经核验。

当前 [svc4-raw-public.json](svc4-raw-public.json) 是主 svc4 原始公共数据对照的模板；实际运行以各EXP已冻结的config/manifest为准，重试剩余预算也记在各EXP内。以后按配方新增明确命名的模板，不用反复覆盖 `latest.json` 保存结果。

[oof-complementarity.json](oof-complementarity.json)用于固定两成员和完整/稀疏场景的只读预测诊断，默认不训练、不拟合融合或校准。单独成员的公共份额与每行权重字段保持各自含义。

[oof-margin-blend.json](oof-margin-blend.json)固定逐行标准化margin的0/5%/10%/20%融合筛选，以10%为主要候选。系数不是校准概率或完整ATLAS20权重；按既有开发OOF筛选，不自动替换正式基线。
