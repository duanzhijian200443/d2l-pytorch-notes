# SRC-056 · GoPain：TPU v5e-8 的 Qwen3-8B LoRA 吞吐探针

原始记录：[正式台账](../../../tracking/records/sources/SRC-056.json)。
状态：已读取原帖与当前可返回评论；已归档，正式记录见下方链接。

## 值得保留的结论

11月1日新帖报告8芯TPU探针12.5k padded tokens/s；约14.2倍为特定配置吞吐，完整训练收益尚未验证。

## 来源与原件

- 原帖：[Kaggle 讨论 754657](https://www.kaggle.com/competitions/name-that-llm/discussion/754657)；作者 `locdunhat`。
- 采集时间：2026-11-01T07:32:09.651508+00:00；原帖发布：2026-11-01T03:34:51.322208300Z。
- [原帖与评论快照](snapshots/20261101T073209Z.json)；SHA-256 `26947c583fff875a37630abae324df9bea90f601d0512b98d9d3ed44d84a7922`。这是公开只读 API 响应，不是独立实验验证。

## 作者声称与实际设置

2026-11-01 03:34:51 UTC 新帖：Qwen3-8B 截断至24层，所有投影 LoRA r16，约30M可训练参数；8层隐藏态均值池化后线性分类头，max_len384。使用 bf16、torch_xla2.8、transformers5.12，每个decoder层采用XLA checkpoint。

| 探针 | batch | step | padded tokens/s | 相对2×T4 fp16 |
|---|---:|---:|---:|---:|
| 单芯 | 8×384 | 1.90秒 | 1.6k | 1.8倍 |
| 8芯SPMD | 64×384 | 1.97秒 | 12.5k | 14.2倍 |

作者称所有loss/梯度范数有限，336个LoRA tensor均有梯度；每个输入形状编译50–80秒。把约7小时阶段折算为约0.5小时是估计，帖子开头说明是在迁移真实训练前先测量。

## 我们核查了什么

已读原帖与评论接口（无评论），未获取完整训练脚本/原始计时日志。作者称旧T4 fp16训练后期大量前向溢出且OOF下降0.10；这是其历史运行自报，不是本轮复现实验。

原帖提供 [序列桶基准 Notebook](https://www.kaggle.com/code/locdunhat/how-many-sequence-buckets-for-tpu-training)，称3–4个优化固定长度桶大约可将单一max_len形状的效率翻倍。完整Qwen probe脚本仅表示愿意分享，未在此帖提供链接。

## 对当前方案的用途与限制

固定形状可减少XLA重编译；作者用128/256/384三个长度与固定token预算、padding行零loss。数据依赖的group数量也需固定；SPMD虚拟设备的memory_info失败不能破坏已完成运行。首次排队约23小时后运行10分钟，吞吐快不等于交付快。作者的周TPU额度和预装库版本都不是官方保证。

## 待验证想法

先小规模测有效token而非padding token吞吐、编译次数、所有梯度有限性、样本覆盖和GPU加载权重兼容。随后才以相同训练步数/数据/seed作收敛与OOF比较。本次未申请TPU、未训练。

## 更新记录

- 2026-11-01：读取原帖和当前评论，保存时间快照。新来源，正式编号已登记。 本轮未训练、未提交。
