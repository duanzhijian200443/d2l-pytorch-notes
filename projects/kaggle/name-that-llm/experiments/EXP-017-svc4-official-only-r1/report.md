# EXP-017 · 重试：主svc4：仅官方数据五折基准

**COMPLETE，产物验收PASS**。Kaggle固定V4，核查于2026-11-01T08:06:13.842221+00:00。

独立svc4官方only对照OOF BA=0.384484216，作为raw公共数据加入对照；不是ATLAS20整体。

- [台账](../../tracking/records/experiments/EXP-017.json) · [配置](config.json) · [冻结输入清单](manifest.json)。
- [逐行OOF](oof/oof.csv) · [18类决策分数](oof/oof_decision.npy) · [逐类召回](metrics/per_class_recall.csv) · [归档哈希](metrics/artifact-index.json)。
- [共用验收报告](../../cloud/svc4-raw-public-r1/analysis/report.md) · [验收凭据](../../cloud/svc4-raw-public-r1/analysis/verified.json) · [运行日志](logs/README.md)。

固定2699条、150个prompt、18类、五折；本组答对1038条，Balanced Accuracy=0.384484216。官方only为参考组。

区间为2000次按prompt配对bootstrap，seed42，未作多重比较校正，只反映这套已拟合开发模型下的抽样变化，不含重新训练或新数据源不确定性。m=2/3/6各5次的稀疏场景均按固定ID协议重算，结果在metrics内；不能事后按单个稀疏场景挑权重。

四组共享主程序546.266秒，守护进程总计550.998秒（约9分11秒），不是每组用时；CPU2线程，峰值采样RSS约0.983GiB，无GPU/API使用。排队、下载和本地验收另计。

本组重试EXP-013，只修复元数据整数序列化；失败原件仍保留。82项稳定产物、20次拟合、2699×18分数、折/ID/标签、权重及矩阵指纹均通过检查；没有收敛警告。模型权重、完整词表及特征矩阵未保存，指纹/代码与成员关系核验不能代替独立重建训练过程。

本轮是独立主svc4原始公共数据诊断，不是原matched/pool配方或完整ATLAS20。加数据对照同时改变词表/IDF；三公共组之间才是纯权重对照。公共生成参考过官方长度统计，语义prompt关联未穷尽；5376/5400条已命中基线数据投影，不是新增5400条。正式0.662基线没有新的得分证据。

下一步：保留0.35作本raw配方开发对照，再检查类别收益和错误互补性。当前未启动下一组训练。
