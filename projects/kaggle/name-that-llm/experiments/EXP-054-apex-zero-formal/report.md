# EXP-054 · Apex 原版，0 条自采数据

父窗口已提交固定私有 Notebook V2。正式提交 **57903109** 绑定 `scriptVersionId=364820326`。2026-11-02 09:13 UTC 的官方记录为 **PENDING**，Public 分数尚未返回。该时刻是提交状态观测，隐藏评分的完成时间和正式分数应以后续官方查询为准。[正式提交绑定](../../cloud/apex-zero-054/formal-submit-binding.json)

## 固定配方与任务

试验目的是为自有账户取得 Apex 原版、0 条新增 Codex 样本的正式对照。原件为 SRC-067 作者 V1 / 364684268，Notebook 原始 SHA256 `ca4c9e18ed32dcd8168b8118b192d39d657803d30dcd89a564b1c2d1d9c846dd`。源代码、Markdown、环境分支和融合参数完整保留。作者源码使用 Open Pool V13、主 SVC 融合系数 12、Sinkhorn 0.60、标点修正 1.2、ATLAS 0.20 和 V40A 家族残差。作者 0.665 属于其公开提交。[固定来源](../../research/solutions/SRC-067-synthreaper-sovereign-gold-apex-v1/notes.md) · [本次配置](config.json)

## 私有执行与验收

账户查重时读取自有 Notebook 清单、本比赛已有提交，以及 23 个比赛关联 Notebook 的 51 个固定版本或未编号快照。未发现相同原始源字节的既有自有版本。新私有 ref 为 `zhjianduan/ntl-apex-zero-054`，kernelId 142259135。先 quick-save V1，确认回读后向 V2 发出一次运行请求。两个版本的远端源码原始字节都与作者原件逐字相同。[查重记录](../../cloud/apex-zero-054/owned-version-source-check.json) · [V2 回读](../../cloud/apex-zero-054/readback-v2.json)

V2 按作者同一 Docker 摘要运行，机型 `NvidiaTeslaT4`，私有、禁网。31 项外部附件由 15 个数据集、8 个上游 Notebook 和 8 个模型组成，另附本比赛。固定数据与上游 Notebook 的版本取自已审计的 EXP053 公开快照。Kaggle SDK 回读会省略部分版本后缀，已保留原始 pin 请求。Open Pool 的 V13 输入卡与日志有单独依据，其余附件未逐字节绑定作者历史隐藏运行。[运行请求](../../cloud/apex-zero-054/v2-intent.json) · [输入快照](../EXP-053-apex-stylegram-volume/inputs/README.md)

Kaggle 报告 V2 **COMPLETE**。固定版本输出对象绑定全局 ID 364820326。保存日志确认走 `FAST_COMMIT`、选择 alexchilton 的 Llama、Gemini 3.7 风险过滤删除 0 行、准备池为 190588 行。四份生成脚本 `infer.py`、`comp_stack.py`、`gauss.py`、`sibling_style.py` 与作者 V1 公开保存的文件逐字一致。公开 `submission.parquet` 具有 `id,model` 两列，3 个 ID 与公开 test 顺序一致，三行均是作者快路指定的 `gpt-5.5`。文件 SHA256 `ef94c7669e36a465bbd3e2dd8cf4905b5e1c2e51e18053a588cb56623086170a`。[产物核验](metrics/commit-verification.json) · [提交移交回执](../../cloud/apex-zero-054/ready-for-submission.json)

## 正式结果边界

父窗口通过 Kaggle 的代码提交入口提交 V2，官方受理时间为 2026-11-02 09:12:48 UTC。公开三行快路的产物用于固定正确 Notebook 版本，隐藏重跑走作者完整分支。本报告撰写时，隐藏预测、实际模型/ATLAS覆盖、分数和计费 GPU 时长均未取得。该提交可与后续不同自采数据量的完整配方作正式对照，单次显示分数的微小变化仍需独立证据解释。[提交受理](../../cloud/apex-zero-054/formal-submit-ack.json) · [固定版本](https://www.kaggle.com/code/zhjianduan/ntl-apex-zero-054?scriptVersionId=364820326)
