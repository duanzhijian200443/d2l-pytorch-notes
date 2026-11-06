# EXP048：A1加原文Luna/Terra120

用户在其他窗口启动，本窗口仅只读核验并登记，没有新开运行、重投、停止或改动云端任务。

北京时间 11 月 2 日 18:12，Kaggle 已确认 [V2 正式提交 57893384](https://www.kaggle.com/code/zhjianduan/name-that-llm-a1-codex120-1102?scriptVersionId=364733823) 完成，Public LB 为 **0.663**。与不加数据的 EXP046 在榜单显示精度下同分；隐藏集逐行预测是否相同未知。[官方状态快照](metrics/formal-latest.json)

直接对照为EXP046/A1正式0.663。唯一算法输入改动是向Stylegram加入EXP040原文Luna60+Terra60；与EXP042的new120.csv字节相同，全部属于60个train原题组，保留原holdout。旧样本每行权重固定，新增行继承同类别公共每行权重，不重新归一化。新增样本用于重新拟合该分支的TF-IDF、普通及组内中心化SVC；其他模型、family gates和融合设置不变。

本次给 A1 加两模型 120 条，没有观察到可见的榜单收益。Apex 的四模型 120 条使用不同题组和基线，另以 EXP054 作对照。

静态核查不证明实际隐藏推理覆盖；新增拟合耗时可能改变原有ATLAS剩余预算。明细与限制见config.json、manifest.json及metrics/source-audit。
