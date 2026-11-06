# SRC-045 · 0.662 | Qwen3.8-27B LoRA + Weighted Ensemble

原始记录：[正式台账](../../../tracking/records/sources/SRC-045.json)。
状态：已取得固定源码并作静态对比；未运行；正式 SRC 已登记。

## 来源与原件

- 作者：Vladislav Lavrentev（`lightningv08`）。[来源页面](https://www.kaggle.com/code/lightningv08/0-662-qwen3-8-27b-lora-weighted-ensemble)。
- 采集时间：`2026-11-01T07:32:19.318819+00:00`。Notebook 版本：V43；全局 scriptVersionId：未取得。
- [源码](source.ipynb) · [API 元数据](metadata.json) · [下载凭据与哈希](receipt.json) · [原正文提取](description.md)。
- 页面分数0.662由主窗口实时网页观察；与本源码版本未绑定，以主窗口保留的页面证据为准。该分数不作为本版本已得分证明。

## 值得保留的结论

V43 是进一步关闭类频率平衡的候选。与已登记 SRC-003 V33 相比，同时改变 GSQ、ATLAS 份额和频率平衡强度；不能把跨版本的任何分数差单独归因于某一参数。作者正文明确说明这个候选没有已确认榜单成绩。

## 作者声称与实际设置

- 正文称 GSQ 模型用完整 119899 行公共池后接两轮官方数据训练；推理保留完整 token 序列，batch size 1，两张 T4 分担。此处只核查了声明和推理代码，未下载权重或训练日志证明训练过程。
- core/complementary 先按 85/15 混合；被选中行混入 10% ATLAS；随后全部行混入 7% GSQ；边际频率修正关闭。最后保留 37-marker 分层 Bayesian typography，未新增末尾 Hungarian 强制分配。
- 公开三条样例是严格按 ID、字节长度和文本 fingerprint 匹配的 CPU 格式预览；不代表完整隐藏集性能或耗时。
- 数据集附件为数值缓存/权重。代码中的通过标记和文件 inventory 是作者设置，不是我们的运行验收。

## 我们核查了什么

- 静态读了正文与全部代码单元。相对 SRC-003 V33 的 [逐行差分](vs-SRC-003.diff)：`gsq_share` 0.075→0.07；ATLAS 0.20→0.10；`balance_logits` 默认 strength 0.45→0.0，且只有 `strength > 0` 才进入修正分支；执行契约新增 `balance000`，记录强度为 0。
- 与 SRC-014 谱集成旧版本也高度同源，见 [差分](vs-SRC-014.diff)；相近行数比例只用于定位，不证明独立创新或等价。
- 元数据返回固定 Notebook V43；本轮未取得全局 scriptVersionId。源文件与代码 SHA256 见下载凭据。未执行代码。

## 对当前方案的用途与限制

对 IDEA-002（固定预测后比较后处理）有直接参考意义，但 V43 同时变三项，不是可用于归因的单变量实验。源码明确区分 Shiner 的得分 V2（SRC-021，0.664）与本候选，归档保留这个边界。网页列表0.662是 Notebook 页面分数，尚未绑定 V43；本条不登记已复现的收益。

## 待验证想法

固定训练范围可核查的分折预测、相同 GSQ 与 ATLAS 份额，先比较 balancing 0/0.30/0.45；再固定最佳候选单独比较 ATLAS 10%/20% 和 GSQ 7%/7.5%。使用 prompt 分组及稀疏场景，报告错误分歧和逐类召回；未启动这些实验。

## 更新记录

- 2026-11-01：首次纳入本批正式资料。关联旧来源：SRC-003, SRC-014, SRC-021。权威来源信息见本页正式台账链接。

## 2026-11-02 固定版本复核

官方API当前仍为V43；以同一规范提取全部代码单元，与已归档源码完全一致。代码SHA256 `a5946bdcdc4f8a7e2f971f37eaa90ba6e57083301c6d3cc5659d04f19f66f52e`。仅更新版本/代码观察，不把标题或历史最佳分绑定到当前版本。详见`snapshots/20261102-community-factcheck/receipt.json`。
