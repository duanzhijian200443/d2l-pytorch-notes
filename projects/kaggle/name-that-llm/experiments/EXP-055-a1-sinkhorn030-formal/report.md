# EXP-055 · A1 的 Sinkhorn 0.30 正式候选

在 2026-11-02T09:10:47.597954+00:00，公开快保存验收通过，状态为 **READY_FOR_SUBMISSION**。Kaggle 私有 Notebook V2 的 scriptVersionId 为 **364819470**。[核验收据](metrics/quick-commit.json)

候选使用 EXP049 冻结 Notebook。相比 EXP046 的 A1，仅将主命令 Sinkhorn 从 0.45 调为 0.30。Stylegram 使用官方 2699 条与原 public10073 配方，未添加自产 120 条。EXP046 的正式 Public LB 0.663 是比较对象。EXP049 的外部面板诊断另见[原实验](../EXP-049-a1-sinkhorn030-ready/report.md)。

## 公开执行

官方 SDK 查询了 74 个自有 Notebook 和 7 条比赛提交。随后各发送一次 V1 保存请求与 V2 运行请求。两个版本均已固定读回，对比完整 Notebook 规范化 JSON、附件集合、私有状态、断网设置、容器和机器规格。[保存、读回与执行证据](manifest.json)

V2 的 preflight 校验 398 个基线小文件及 5 个额外小文件的 SHA256，对 130 个基线大文件核对尺寸。运行环境确认两张 Tesla T4，A1 合成分数与推理路由核对通过。Kaggle 回传部分 dataset/kernel 引用时省略版本，实际字节依据本次运行的分级校验解释。

公开测试集三条 ID 按顺序输出，预测均为快保存占位类。收据标记 EXP-049、Sinkhorn 0.30、ATLAS 0.20、comp 0.15。submission.parquet SHA256 为 `ef94c7669e36a465bbd3e2dd8cf4905b5e1c2e51e18053a588cb56623086170a`。正式提交和隐藏评分由统一提交控制器另行记录，本收据仅核验公开快保存。

正式提交：北京时间 11 月 2 日 17:12，Kaggle 已受理 [V2](https://www.kaggle.com/code/zhjianduan/name-that-llm-a1-sinkhorn030-049?scriptVersionId=364819470)，提交号 **57903087**；首次回查为 PENDING，尚无公开分数。[提交收据](metrics/formal-dispatch.json)
