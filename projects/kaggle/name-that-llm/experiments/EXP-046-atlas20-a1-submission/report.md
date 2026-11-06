# EXP046：A1公共数据Stylegram正式提交

本实验接续EXP045 A1：在完整ATLAS20中，只替换Stylegram为已冻结公共数据配方。保留EXP004其余模型与融合设置，不加入自制120条、不清除文本换行加粗、不采用A2/A3。

本地729条探索集上A0/A1均为0.601075宏召回、488条正确，最终类别无变化；这不是干净OOF或正式榜单分数。此次向Kaggle提交代码，在隐藏集重新计算所有需要的模型预测，不上传本地729条预测缓存。

固定输入为EXP031的31附件加公共Stylegram输入V1，合计32。两个历史公共CSV镜像已核对字节，不能宣称所有历史模型权重都逐字节复现。详细参数见config.json，源代码/附件绑定见manifest.json，独立审查见metrics。

快速保存运行使用小公开样例占位，仅用于云端输入、T4×2环境、数值自检与输出格式检查。只要KAGGLE_IS_COMPETITION_RERUN键存在，无论它的值或样本数如何，都执行完整算法。完整运行仍遵守9小时上限。

当前远端进展与具体版本以cloud/a1-public-stylegram-046/execution-state.json和提交回执为准。未获得绑定的COMPLETE分数前，不宣称提升。

## 已提交

正式提交ID：57879463；自有Notebook V2；scriptVersionId：364624288。

提交时间：2026-11-01T17:48:25.780Z。最新核查状态：COMPLETE；正式Public分数0.663。

[固定提交版本](https://www.kaggle.com/code/zhjianduan/name-that-llm-a1-public-stylegram-046?scriptVersionId=364624288)。详见submission-ack.json及submission-binding.json。

## 正式结果核验（2026-11-02）

官方API确认submission57879463、scriptVersionId364624288已COMPLETE，Public0.663。相对固定EXP004的0.662，显示分数增加0.001（0.1个百分点）；仅是单次正向结果，不能据此证明稳定或显著增益。

正在评分的Codex120提交57893384应与本次0.663直接比较。自己的另一配方EXP047/0.664仍为已核实最好正式分数，不替换本组受控对照。详细证据见metrics/formal-result.json。
