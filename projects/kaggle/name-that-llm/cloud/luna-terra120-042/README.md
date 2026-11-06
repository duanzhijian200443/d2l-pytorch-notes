# EXP-042 私有CPU实验

[实验协议与结果](../../experiments/EXP-042-luna-terra120-ablation/report.md) · [执行状态](execution-state.json)

比较固定Stylegram配方是否加入120条Luna/Terra原文，复用原官方五折。仅私有CPU、关闭互联网，无GPU、无生成API、无正式比赛提交。

`payload/`保存固定输入与代码。Notebook内只嵌入新增训练数据、代码和无原文的审核收据；既有公共数据、官方训练及分折按哈希从固定挂载读取。新holdout和诊断面板原文不上传。

启动入口为`experiments/EXP-042-luna-terra120-ablation/launch.py`。只有PREPARED状态、输入/源码哈希和独立代码审查均通过，才先Quick Save回读，然后请求一次真实运行。已发起任务不能再次执行启动入口。

输出`results/`包含逐折/OOF/稀疏预测、权重、收据、日志和guard；结束时另写`exp042-results.zip`便于一次下载。完整结果验收使用本实验`verify_outputs.py`。首次保存版本不等于实际训练版本，以execution-state和run-ack为准。
