# EXP-042 独立代码审查

结论：**PASS，仅限开跑前代码与合成验证**。绑定运行 `exp042-20261101T143057Z`；四份最终源码哈希及详细证据见 [code-review.json](code-review.json)。未训练、未启动云端任务、未联网。

## 已解决的问题

- 原 prepare 会把含保留集样本的完整 data-review.json 嵌入云端包。现仅嵌入无样本文本的 data-gate.json；完整审核留本地。最终 Notebook ZIP 解码后只有 7 个允许文件，71850 字节，658 条保留集及配对诊断原文检查零命中。
- launch 现将自身纳入 manifest，并在请求之前核对 runner/test_runner/prepare/launch 四份审查哈希，避免旧 PASS 用于变化后的源码。

## 核验结果

- 训练词表只拟合每折官方训练部分、原 public pool，以及候选臂的 120 条新增数据；验证仅 transform。验证文本独立聚类是原有无标签推理流程，未给分类词表或 SVM 训练使用。
- 五折控制臂旧权重逐元素等于旧实现；候选保留每一条旧数据的权重，新增行继承原对应类 public 每行权重，不重归一化。
- 新数据为 60 个 Luna/Terra 成对组，以独立 codex 分块中心化；不混入公共组。官方折和行序保持冻结。
- 实际官方数据上的 75 个稀疏选取索引全部等于旧实现；先选择再按选中文本聚类及中心化，两臂共享选择。
- 最终 test_runner.py：10/10 合成测试通过。guard 的独占锁、拒绝重复、完整清单、拒绝字节篡改和未列出文件检查通过。
- 使用本机实际 kagglesdk.ApiSaveKernelResponse 做纯内存回执检查，确认 .url 及所有使用字段存在。没有发起 API 请求。
- 固定 core、float64、C=.5/dual=True/max_iter=4000/seed42/两线程保持旧控制配方。云端严格检查四个包版本且拒绝 GPU；本地非严格 preflight 不代表云端环境已验收。

## 运行后仍需验收

必须取得本运行 guard/result/state COMPLETE、child_exit=0、无失败或终止原因、严格实际环境、完整两臂五折 20 次拟合、2699×18 OOF 和全部 150 份稀疏产物，并核对清单字节。随后比较 control 与 EXP010 share0.5 的历史 dense/sparse 分数和预测；任何无法解释的漂移先阻断效果解释，不能自动重跑。

正式基线不变。本次只检验 Stylegram 增量数据；官方开发折并非新 holdout，known/exact 隔离不认证语义独立或真实后端模型快照。新增数据会同时影响候选词表/IDF和两类总权重，此为冻结的数据增量干预。没有结果前不声称收益，RSS 限制仍为采样式。

