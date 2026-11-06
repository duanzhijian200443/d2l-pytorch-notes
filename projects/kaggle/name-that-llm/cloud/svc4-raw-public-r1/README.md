# 主svc4 原始公共数据：第一次兼容性重试

EXP-017/018/019/020分别重试EXP-013/014/015/016。首轮在分类器拟合前因词表指纹中的numpy.int64序列化退出；本轮只转换指纹索引为Python int，不改变原文、特征、参数和权重。

共享剩余平台预算7080秒，内部工作期限6780秒；与首轮预留120秒合计不超过原7200秒上限。RSS仍12GiB，CPU2线程，不自动增加第二次重试。

[原实验定义](../svc4-raw-public/README.md) · [原数据门槛](../../experiments/EXP-014-svc4-raw-w035/metrics/data-gate.json)

数据准备输出逐字复用；本目录prepare_inputs.py仅作为原准备源码证据，不重复执行。运行以execution-state.json及固定Kaggle保存版本实时状态为准。

| 实验 | 数据 | 官方每行权重 | 公共每行权重 | 预先确定的比较 |
|---|---|---:|---:|---|
| EXP-017 | 官方训练折 | 1 | 不加入 | 独立主svc4基准 |
| EXP-018 | 官方训练折+5400条raw/new | 1 | 0.35 | 对EXP-017比较加数据的整体效果 |
| EXP-019 | 与EXP-018相同 | 1 | 0.175 | 对EXP-018比较减小公共权重 |
| EXP-020 | 与EXP-018相同 | 1 | 0.70 | 对EXP-018比较增大公共权重 |

2026-11-01 [Kaggle V4](https://www.kaggle.com/code/zhjianduan/name-that-llm-svc4-raw-public-cv)已COMPLETE且**完整产物验收PASS**。[终态回读](complete-v4-readback.json)确认私有、CPU、互联网关闭、原容器和固定源码；运行时协议哈希绑定Dataset V2输入包。全部20次SVM拟合完成，82项稳定产物、2699×18分数、折/标签/样本权重和特征指纹一致性通过。见[机器验收](analysis/verified.json)、[详细结果](analysis/report.md)、[独立CSV复算](independent-review.md)、[控制记录](execution-state.json)。V3仅保存未运行。

已通过官方CLI按固定V4下载88文件（约4.40MB），见[下载清单](download-receipt.json)。可在项目根目录重验：`python -B cloud/svc4-raw-public/verify_results.py --input cloud/svc4-raw-public-r1/outputs-v4/results --protocol cloud/svc4-raw-public-r1/payload/protocol.json --check-only`。验收器只读取缓存结果，不训练或提交比赛。

| 实验 | 公共每行权重 | OOF Balanced Accuracy | 答对数/2699 |
|---|---:|---:|---:|
| EXP-017 | 不加入 | 0.384484216 | 1038 |
| EXP-018 | 0.35 | **0.402289336** | 1086 |
| EXP-019 | 0.175 | 0.397099180 | 1072 |
| EXP-020 | 0.70 | 0.390800398 | 1055 |

保留0.35作为本raw配方开发对照。相对官方only提高1.7805个百分点，prompt配对95%区间[+0.2998,+3.2222]个百分点，五折4正1负；0.175相对0.35的区间跨0，没有提升依据；0.70比0.35更差，区间[-2.0370,-0.2965]个百分点。区间采用2000次prompt bootstrap、seed42，未校正多重比较，也未计入重新训练不确定性。稀疏m=2/3/6的五次均值也以0.35最高，但这些是复用同一模型的抽样情景，不是独立训练重复。

四组共享主计算546.266秒，守护进程550.998秒，约9分11秒；20次SVM拟合共335.244秒。采样峰值RSS约0.984GiB，无GPU/API使用。这里不含排队、下载和本地验收。每组OOF、指标、日志索引已归档到各EXP目录，原冻结config/manifest未改。

主指标为固定五折2699条OOF的Balanced Accuracy，并复算逐类、逐折及三项prompt配对区间；m=2/3/6各5次稀疏场景按固定ID抽样，只作压力检查。决策分数不等于概率，单成员开发OOF不等于完整ATLAS20增量。公共生成参考过官方长度分布，未知语义关联仍有限制，不能称严格未触碰验证。
