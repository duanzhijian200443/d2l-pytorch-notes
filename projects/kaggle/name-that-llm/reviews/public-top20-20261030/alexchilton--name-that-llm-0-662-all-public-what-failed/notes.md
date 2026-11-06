# SRC-002 · Name That LLM 0.662: All Public + What Failed

原始记录：[正式台账](../../../tracking/records/sources/SRC-002.json)。
状态：已核查固定源码与部分页面证据；没有新运行。

## 值得保留的结论

全公开ATLAS20 V8为多份新页面的同代码上游；保持0.662页面分数未绑定V8。

## 来源与原件

[来源页面](https://www.kaggle.com/code/alexchilton/name-that-llm-0-662-all-public-what-failed)。复用已有原件，不为本次查询复制源码：

[source.ipynb](source.ipynb) · [metadata.json](metadata.json) · [description.md](description.md) · [ui-scores.json](../ui-scores.json)

固定版本 V8，代码 SHA256 `d164e570c03550b08337e9934b5f18af1255aeacdf1e0a0d279547bfac693c6f`。本轮归档日期2026-11-01；网页观察与原始证据已关联到本页正式台账。

## 作者声称与实际设置

作者报告自己ATLAS20运行0.662、若干失败消融与相同代码重复分数差；这些是作者自报。

本条保存全公开依赖版本的 ATLAS20：Llama输入改为alexchilton公开模型；集成仍含complementary15%、ATLAS20%、Sinkhorn0.45和原标点步骤。新出现的多个公开Notebook延用这一代码，页面数量不代表方法数量。

作者正文报告将同代码重跑得到不同分数，也报告若干替换/增加成员与Hungarian失败尝试。这些表是作者的历史报告，未附完整逐次submission绑定，本轮不升级为独立重现实验。作者从若干观察推导的约0.01阈值不能当作通用统计噪声界限；应保留为对微小差值谨慎解释的提醒。

本轮确认新收集 junaid V3、sujan V12、dazhijiang V1、gloria V1 的全部代码单元SHA256均为d164e570c03550b08337e9934b5f18af1255aeacdf1e0a0d279547bfac693c6f，与本条相同。其文内复制的第一人称失败实验不应被重复计作四次独立社区验证。

网页列表仍显示0.662；未将该页面成绩绑定V8。公共样例快速提交路径不证明隐藏评分完整执行成功。

## 我们核查了什么

固定V8源码/正文可读且hash匹配；新junaid V3、sujan V12、dazhijiang V1、gloria V1代码均与本条完全相同。页面仍显示0.662，但未查得分版本。 本轮静态读取与比对，没有执行下载的Notebook。

## 对当前方案的用途与限制

作为当前ATLAS20同源基线比较和失败实验线索；复用原件避免重复归档同一固定版本。 平台0.662未绑定V8；作者自报未独立重训；输入依赖内容仍需版本审核。

## 待验证想法

固定输入版本、训练来源与预测缓存，以prompt分组和稀疏场景对照；新方法先验证互补性而非只看单成员OOF。 本次整理不登记新实验或收益。

## 更新记录

- 2026-11-01：复用SRC-002与既有原件，补独立笔记、同源/版本边界。关联SRC-001, SRC-009, SRC-010, SRC-021, SRC-022。既有证据路径保持有效，独立笔记与正式台账已关联。
