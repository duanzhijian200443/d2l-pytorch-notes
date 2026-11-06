# SRC-047 · NTL Apex SOTA Atlas Hungarian

原始记录：[正式台账](../../../tracking/records/sources/SRC-047.json)。
状态：已取得固定源码并作静态对比；未运行；正式 SRC 已登记。

## 来源与原件

- 作者：ItsmeSujan（`sujanmajhisuzan`）。[来源页面](https://www.kaggle.com/code/sujanmajhisuzan/ntl-apex-sota-atlas-hungarian)。
- 采集时间：`2026-11-01T07:32:19.753501+00:00`。Notebook 版本：V12；全局 scriptVersionId：未取得。
- [源码](source.ipynb) · [API 元数据](metadata.json) · [下载凭据与哈希](receipt.json) · [原正文提取](description.md)。
- 页面分数0.662由主窗口实时网页观察；与本源码版本未绑定，以主窗口保留的页面证据为准。该分数不作为本版本已得分证明。

## 值得保留的结论

本页虽名为 Hungarian，但代码与 SRC-002 完全相同，不能从标题推断其新增了 Hungarian 算法。 保留出处和固定版本，方法上归入已存在的 alexchilton ATLAS20 同代码家族。

## 作者声称与实际设置

正文沿用 alexchilton 全公开版：V31 主体、ATLAS 20%、公开 Llama 路径、8B 双 GPU 与池化优化。文内的失败实验和相同代码分数波动表均为上游已有文字，不能据此声称当前页面作者又独立做了一轮实验。

代码仍使用 complementary 15%、ATLAS 20%、Sinkhorn 0.45、target mix 0.35；正文中的标题或历史0.663说明不覆盖当前页面0.662。公开三条样例快速路径与隐藏评分路径分开。

## 我们核查了什么

- Kaggle API 返回 Notebook V12，公开状态为 false isPrivate；原始文件、元数据和哈希随条目保存。全局 scriptVersionId 未取到。
- 全部代码单元按顺序以双换行拼接后 SHA256=`d164e570c03550b08337e9934b5f18af1255aeacdf1e0a0d279547bfac693c6f`，与 SRC-002（alexchilton V8）、SRC-009（junaid V1）、SRC-010（dfenglei V1）完全一致。
- 本次新收集的 junaid V3、sujan V12、dazhijiang V1、gloria V1 同属这一家族。后三者完整 Notebook 源文件 SHA256 也一致；不是四种新方法。
- 未执行下载代码，没有重跑训练或正式提交。

## 对当前方案的用途与限制

用途是保存公开复现出处和潜在附件可用性线索。页面分数0.662由主合并窗口在列表观察，尚未绑定此源码版本；附件引用可能随发布版本变化，代码相同不能证明输入数据/权重完全相同，更不能证明单次成绩可复现。

## 待验证想法

不因同代码副本另开算法实验。若将来需要重跑，先固定源版本对应的得分、所有附件版本与哈希，再复用同一基线验证；若只有挂载变化，作为依赖修复记录。

## 更新记录

- 2026-11-01：首次纳入本批正式资料。关联旧来源：SRC-002, SRC-009, SRC-010。权威来源信息见本页正式台账链接。
