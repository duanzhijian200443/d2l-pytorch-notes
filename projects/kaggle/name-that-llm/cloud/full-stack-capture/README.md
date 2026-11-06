# EXP-029：真实完整基线预测缓存

[实验记录](../../experiments/EXP-029-full-stack-capture/report.md) · [当前运行状态](execution-state.json) · [上一步缓存工程](../full-stack-cache/README.md)

目的：复用 EXP-004 的固定 ATLAS20 代码，对同一批 729 条外部原文生成真实成员分数，随后在 CPU 上复算。面板只上传 `id,text`；标签和题组侧车留在本地。这是探索性外部诊断，不是干净 OOF、独立提分证据或比赛提交。

## 执行顺序

1. `preflight-kernel/` 是已启动的私有 CPU V1，只挂载 31 个固定输入，逐文件计算 SHA256。内部截止 3450 秒、子进程超时 3500 秒、Kaggle 上限 3600 秒。
2. 下载 V1 的 `preflight/preflight_manifest.json`；必须为 PASS，源码/配置哈希、已知输入哈希及保存配置回读都匹配。
3. `prepare_capture.py --preflight-manifest <文件> --destination <新目录>` 在新目录生成冻结运行包与 GPU 上传脚本。它不启动任务。脚本内嵌压缩后的允许文件，仍只使用原 31 个附件；不把本地侧车或整个实验目录上传。
4. 将 GPU 版本保存并回读源码、输入和资源设置，确认后只启动一次。平台限时 28800 秒、T4×2、私有、禁网；工作进程最多 28200 秒，结束或出错后进程组终止，由 Kaggle 结束该次批处理资源。
5. 下载 `results/`，先验收两分支身份、成员、阶段、ATLAS 覆盖、重放与 Parquet，再用本地侧车计算指标。无正式提交动作。

旧单次接续 `launch_after_preflight.py` 已因保存错误退出，句柄仅作历史证据保留于 [controller-session.json](controller-session.json)。恢复时发现返回 ref 带 `/code/` 前缀，且脚本 Quick Save 的页面导出按 Notebook JSON 解析；已保留原代码，改为单代码单元 Notebook 外包装，第2版回读后发起第3版完整GPU运行。**不要重新启动旧控制器。** 当前状态及恢复凭据见 [execution-state.json](execution-state.json) 和 [recovery-result.json](recovery-result.json)。GPU V3 已失败结束，尚未进入模型推理；新增检查误将可选 public LoRA 设为必需。当前不占用 GPU 预留，禁止重启旧控制器；详见实验报告和失败审计。

本机隐藏后台进程启动曾被工具策略拒绝，当前使用的是受控执行会话，并非另开的隐藏进程。无需重复运行接续脚本；先查 `execution-state.json`，再检查该会话和 Kaggle 真实状态。

## 运行器保证与限制

`build_runtime.py` 只生成新副本：comp 的非模拟 `DATA_DIR` 指向 `NTL_CAPTURE_DATA`，main 的 `--data` 同目录；官方训练文件不变。原 Notebook 的常量占位分支不执行，离线安装和每个子进程退出码都会检查。

`runner.py` 先验证包、CPU 文件凭据和 GPU 实际环境。大于 16 MiB 的模型文件复用 CPU 全哈希及相同固定附件版本，GPU 端检查路径/大小；小文件重新计算哈希。不能将此描述成 GPU 重新读取全部权重做哈希。

本次预先要求 16 个主成员、22 个补充分支成员和 5 个家族 gate。Q35 token head 因原底座未附加而明确排除；其余所需输入必须存在。22 键要求至少一个有效补充分支分组，未满足就记录覆盖不足，不改算法造组或事后缩减名单。

comp 有本次 COMMITTED 凭据后才运行 main。ATLAS 实际预算取原预算与全局剩余工作时间减 120 秒的较小值；不足 180 秒则停止。原预算、实际预算和覆盖 mask 都记录。超时或部分覆盖不冒充完整验收。

`result.json` 是整体终态，`worker-result.json` 是工作进程结果，`component-receipts.json` 绑定组件缓存，`replay.json` 是数学核验，`cache-artifacts.zip` 是小型结果包。Kaggle 显示 COMPLETE 只代表脚本进程完成，仍须检查这些内容；GPU真实执行与覆盖合格、独立验证、榜单得分是不同结论。

## 本地核验

不启动模型的包检查：`python -B -X utf8 -m unittest discover -s cloud/full-stack-capture -p test_prepare_capture.py`。运行器与数据流的独立检查记录在 `../full-stack-capture-planning/`。EXP-028 和原 Notebook 的已冻结文件保持不变。
