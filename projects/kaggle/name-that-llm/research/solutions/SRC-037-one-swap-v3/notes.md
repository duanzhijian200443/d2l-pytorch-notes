# SRC-037 · One Swap V3：说明修订与权重梯度，未绑定成绩
状态：已读固定源码与作者说明；未独立复现。
原始记录：[正式台账](../../../tracking/records/sources/SRC-037.json)。

## 值得保留的结论
最新版V3 / scriptVersionId364294976 是11秒 Quick Version、0个输出。页面Best Score仍指V1 /364073396的0.664；不能给V3继承该成绩。V3较V1删除开头绘图、修订署名注释与Markdown，主推理数学代码未变化。

## 来源与原件
- [固定V3](https://www.kaggle.com/code/zbigniewkirchner/0-664-one-swap-198k-clean-pool-fully-public?scriptVersionId=364294976)
- [源码](source.ipynb)、[作者说明](description.md)、[元数据](metadata.json)、[哈希](receipt.json)、[与V1差分](vs-swap-v1.diff)。

## 作者声称与实际设置
作者称SVC清洗池在193k版本、五折、pool_weight0.35时从0.4805到0.5008。其权重梯度报告为 svc_w1.5→0.664、6→0.659、12→0.664、20→0.664，没有清楚的单调趋势。正式方案代码为Open Pool、svc_w12、ATLAS20%。完整隐藏跑约8小时只是作者陈述。
作者引用alex的相同代码波动幅度；这不是可以通用采用的0.01显著性阈值。

## 我们核查了什么
最新API返回V3，代码及来源SHA256已保存；网页版本号和fork-version链接对应364294976。主推理相对V1一致，但整份code SHA不同。作者文字已明确更高svc_w并未证明改善。V3无本次独立得分证据。

## 对当前方案的用途与限制
可用其说明理解新池和权重对照、继承来源和限制；执行复现仍应固定得分V1及附件。与V1为同一推理配方，不算第二种独立0.664方法。保留V1笔记中的去重、时间预算、ATLAS选行耦合问题。

## 待验证想法
关联IDEA-003，先固定缓存拆开池与svc_w影响，再记录ATLAS实际覆盖量。保存快速路径成功不等于隐藏数据完整推理成功。

## 更新记录
- 2026-11-01：新增V3条目并关联V1；分数留为未核验，保留作者负结果。

关联来源：[SRC-002](../../../tracking/研究索引.md#src-002) · [SRC-027](../../../tracking/研究索引.md#src-027) · [SRC-036](../../../tracking/研究索引.md#src-036) · [SRC-038](../../../tracking/研究索引.md#src-038) · [SRC-041](../../../tracking/研究索引.md#src-041)。

## 2026-11-02 固定版本复核

官方API当前仍为V3；以同一规范提取全部代码单元，与已归档源码完全一致。代码SHA256 `d8c630d3b9fd1b3b0453de6fffb9e5d613f7a9c1c79cf47ab42f7dd9ad30321f`。仅更新版本/代码观察，不把标题或历史最佳分绑定到当前版本。详见`snapshots/20261102-community-factcheck/receipt.json`。
