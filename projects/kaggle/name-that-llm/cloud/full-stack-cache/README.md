# 完整基线分数缓存

对应 [EXP-028](../../experiments/EXP-028-full-stack-cache/report.md)。本目录是导出与重放工具，**不是已经生成的真实全模型预测**。

## 做什么

从 EXP-004 固定基线提取的 `infer.py`、`comp_stack.py` 生成新副本，只增加观测点。每个成员的分数、Stylegram 原始 margin、家族 gate 前后分数、主集成、ATLAS 请求行/完成 mask、Sinkhorn、标点修正分别保存。数组保留原 dtype、形状和标签顺序；每份缓存绑定样本 ID、正文哈希、代码版本和组件清单。

原 Notebook、原脚本与正式 0.662 基线不变。输入不齐或缓存写入失败会终止捕获，不静默生成看似完整的缓存。`COMMITTED` 仅表示已保存字节完整，不能证明模型真实执行、覆盖完整或没有训练泄漏。

## 本地工程核验

在项目根目录执行；指定新的输出目录/运行标签，保留已有尝试。

```powershell
python -B -X utf8 cloud/full-stack-cache/instrument.py --output-dir <新的副本目录>
python -B -X utf8 -m unittest discover -s src -p test_full_stack_cache.py
python -B -X utf8 -m unittest discover -s src -p test_full_stack_observer.py
python -B -X utf8 experiments/EXP-028-full-stack-cache/run_parity.py --case all --run-tag <新标签>
```

合成核验实际执行原源码与副本的选定后处理数学，模型输出由固定合成数组代替。包含 60 行的部分选行及 100 次 Sinkhorn、3 行短面板、ATLAS 未完成三种情况。比较原/新数组与最终预测表，再从落盘缓存独立重放。它不加载模型、不训练、不用 GPU、不测准确率；Parquet 写入在该核验中被截获，因此只验证表内容，不验证 Parquet 引擎。

`assemble.py` 把 `gauss.py` 及共用模块放入 `generated/`，创建带哈希的 `package-manifest.json`。仅供本次固定包组装；不会启动云端或模型，已有不同内容不会被覆盖。

## 未来真实捕获

`capture-context-template.json` 默认 `launch_ready:false`，不能直接启动真实捕获。先固定实际输入字节/版本、激活成员与允许缺省项、输入面板、环境和预算，再生成新 run_id 的捕获上下文。训练暴露限制见 [data-scope.json](data-scope.json)。现有全部 adapter 在官方训练集上的预测不能称为 OOF；已有外部面板也尚未证明与所有大模型成员隔离。

运行脚本时通过环境变量 `NTL_CACHE_EXPORT` 指定缓存根目录，`NTL_CACHE_CONTEXT` 指向上下文文件。未设置导出变量时为无操作观测器。两个脚本使用同一 run_id、同一 ID/正文/18 类顺序。

必须在原 Notebook 的所有 `%%writefile` 完成后替换为新副本，避免随后被覆盖。公开三行样例会触发原 Notebook 的 `FAST_COMMIT` 常量占位输出；需要显式启动真正的计算流程才有模型缓存，保存版本成功不能证明已执行模型。

从日志保留 `CACHE_COMPONENT_COMMITTED` 中 main/comp 的 manifest SHA256，作为缓存以外的读回依据。成员清单在捕获前冻结，不能按实际缺失结果反向缩减验收范围。

```powershell
python -B -X utf8 cloud/full-stack-cache/verify_cache.py --cache-root <缓存根目录> --test-csv <同一输入.csv> --receipts-json <外部哈希凭据.json> --inventory-json <事前成员清单.json> --output <新的核验报告.json>
```

凭据格式为 `{"main":{"manifest_sha256":"..."},"comp":{"manifest_sha256":"..."}}`，SHA 必须是非空 64 位小写十六进制；成员清单含非空、无重复的 `expected_main_members` 与 `expected_comp_members` 数组。真实捕获前将这两项写入上下文；读回时同时检查两组件的上下文哈希与内容一致，成员清单等于事前上下文。工具只复算缓存数学，不认证模型来源或准确率。

ATLAS 是限时计算，记录开销可能影响完成行数；不同覆盖必须单独报告。未完成或新候选需要的行不在缓存内时，不能补零、猜测或宣称全流程等价，应返回 `NEEDS_ADDITIONAL_ATLAS_SCORES` 并另行取得缺失预测。原算法下的部分结果可以记录，完整覆盖验收仍不通过。

旧 V30 的三条 `comp_logit.npy` 已独立导入 EXP-028，属于旧配方的后 gate 汇总，缺主分支和成员明细；不能由它倒推出缺失分数。现有固定 Kaggle/Drive 输出的查询依据见 EXP-028 `logs/`。
