# EXP-035 双渠道重复对照：描述性结果

采集状态 `STOPPED_FAILURE`；保存1条，发送2次；已确认Kaggle费用$0，费用完整性=False。
完整题目×模型单元0/20，不完整20；参与成对汇总的独立原题组0个。
存在费用未知的请求；已确认金额不是本次总费用，不能把未知部分当作0。
没有完整配对单元，无法估计跨渠道差异或同渠道随机波动。

## 格式与长度

| 渠道 | 阶段 | 条数 | 格式筛查通过 | 含换行 | 含加粗 | 平均词数 | 词数偏离目标>25% |
|---|---|---:|---:|---:|---:|---:|---:|
| Codex订阅 | 旧未对齐 | 20 | 6/20 | 13 | 6 | 113.4000 | 1 |
| Codex订阅 | 新已对齐 | 1 | 1/1 | 0 | 0 | 147.0000 | 0 |
| Kaggle Model Proxy | 旧未对齐 | 20 | 7/20 | 13 | 1 | 118.4000 | 1 |
| Kaggle Model Proxy | 新已对齐 | 0 | 0/0 | 0 | 0 | 未知 | 0 |

旧样本只是历史参照，格式变化不能解释为同时随机化实验得到的因果效果。格式筛查是规则检查，不等于内容全部合规。

## 同渠道与跨渠道波动

词汇距离是词组三元组的Jaccard距离，0表示表面词组相同，1表示没有共享词组；不是语义差异百分比。每单元先计算两个同渠道重复距离与四个跨渠道距离，再按原题组汇总。

| 范围 | 同渠道词汇距离 | 跨渠道词汇距离 | 跨−同差值 | 按题组bootstrap 95%区间 |
|---|---:|---:|---:|---|
| 两模型按题等权 | 未知 | 未知 | 未知 | 不足2个题组，未估计 |
| gpt-5.6-luna | 未知 | 未知 | 未知 | 不足2个题组，未估计 |
| gpt-5.6-terra | 未知 | 未知 | 未知 | 不足2个题组，未估计 |

平均绝对词数差：同渠道未知词，跨渠道未知词。
完整文本完全相同的额外行数为0；这些真实重复均保留。

## 解释边界

- Only10 independent original prompt groups, two repeats per route; estimates may be unstable.
- All successful raw records retained, including duplicates and flagged format/length results; paired distances use complete cells only, with incomplete cells listed.
- Missing or failed cells may cause selection bias; bootstrap does not correct it.
- Historical unaligned samples are not a simultaneous randomized format-control arm; cannot isolate a causal effect of format instructions.
- Channel differences may reflect hidden context, backend snapshots, output caps, timing or random variation; cannot identify built-in prompt contamination.
- Codex model labels may be requested_only; matching label text does not prove identical backend snapshot.
- No evidence here establishes training benefit, leaderboard improvement or raw-API equivalence.

完整逐单元、逐模型、缺失请求、失败结果、格式标记和输入哈希见 [paired-descriptive.json](paired-descriptive.json)。
