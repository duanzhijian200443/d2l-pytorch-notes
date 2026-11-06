# EXP-031：修正可选成员预检后的缓存重跑

[实验记录](../../experiments/EXP-031-full-stack-capture-r1/report.md) · [实时查询前先读运行状态](execution-state.json)

本次修复 EXP-029 的输入检查错误：原算法会跳过未挂载的 `lora24_public` 及其 twin，但旧运行器强制要求该分支，约 42 秒即退出。这里保留旧运行、原 Notebook 和全部失败证据，使用新的实验和 Notebook 运行。

`input_policy.py` 是本地 CPU 预检与 GPU 启动检查共用的规则。它基于已核验的 531 文件清单解析成员，确认固定缺席的 public LoRA、public twin 与 Q35，得到 20 个补充分支成员目标。必需文件缺失、路径增删、意外新增可选模型、重复或歧义路径仍然拒绝。20 成员覆盖仍要求至少一个有效 comp 分组；不为满足覆盖要求造组或重新分配权重。

GPU 运行器会将实际挂载路径的解析结果、旧 CPU 清单的解析结果分别与冻结的 `availability.json` 比较，保留 Q35 的配置内容检查。路径存在性不能代替文件内容校验；本次继续使用旧 CPU 全文件哈希与相同固定附件版本，GPU 端重算小文件哈希并核对大文件路径/大小。

原有 setup、模型命令、main、comp、gauss 和缓存重放代码按字节复用。只修输入策略和实验标识，预算及进程退出/归档守护不变。使用 729 条盲化外部文本，真标签侧车不上传；该面板仍为探索性诊断，不能称为干净全栈 OOF。

`build_package.py` 生成实际 `.ipynb` 上传包装，避免旧 Python 脚本 Quick Save 被按 Notebook JSON 解析。新 Notebook 为 `zhjianduan/name-that-llm-cache-capture-031`；版本1仅保存回读，版本2为本次唯一完整运行。运行后查 `results/result.json`、`input-availability-check.json`、组件凭据和重放报告，不能只凭平台 COMPLETE 认定验证成功。

策略13项测试、运行器14项测试及独立缺失输入检查通过，具体源码与输入哈希见实验 `metrics/`。没有重新下载权重、重新读取190GB，也没有正式提交比赛。运行最多8小时，当前结果以实时Kaggle状态和下载产物为准。
