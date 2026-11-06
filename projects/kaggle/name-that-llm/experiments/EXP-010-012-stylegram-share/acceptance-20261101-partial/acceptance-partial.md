# EXP-010/011/012 结果验收：部分完成，完整验收被原始二进制缺失阻塞

记录日期：北京时间 2026-11-01。对象：[Stylegram Share CV](https://www.kaggle.com/code/zhjianduan/name-that-llm-stylegram-share-cv/output)，Version 2，scriptVersionId **364307522**。

## 结论

**训练成功不等于验收 PASS。本次没有通过完整验收，没有更新正式实验台账结论，没有改动 EXP-004 基线。**

已取得并核对三组完整 OOF CSV、fold_metrics.csv、result.json、environment.json。三组各 2699 个唯一 ID，顺序、标签、prompt_id、折号与官方 train.csv 和固定 prompt-group-v1 完全一致；固定折没有跨折 prompt。三个公共份额为 EXP-010=50%、EXP-011=25%、EXP-012=75%。原始标签预测与 correct 列一致；15 条折指标与 CSV 复算一致。

但是还没有下载到 3 份 OOF NPY、15 份完整 fold NPZ 和 225 份 sparse NPZ，故无法核对分数矩阵、argmax 与每折缓存一致性，也不能计算五折合并后的稀疏指标。CSV-only 诊断不能代替这些检查。

## 实际执行过的原脚本

从项目 Drive 取得原版 verify_results.py 与 write_report.py，未修改其逻辑。

- verify_results.py SHA-256：ea6fa44926c036da45e694b3b2fe1b64f32ecb0b44200f5496eb9a4c1d907bc2，与项目 postprocess-review.json 记录吻合。
- 原版 verify_results.py --input download-v2/results 实际执行，exit=1，在读取 oof-share0.5.npy 时 FileNotFoundError。此前 result COMPLETE/30 SVM/零收敛警告、environment 与 protocol 一致、官方 train 哈希已经通过。
- 原版 write_report.py 实际执行，exit=1：不存在 analysis/verified.json。没有伪造 PASS 文件，也未生成脚本的正式 report.md。
- 详细错误分别在 verify-attempt.log、report-attempt.log。
- 辅助 csv_diagnostics_only.py 只做已取得 CSV 的局部诊断，复用原核验脚本 matrix_counts 函数；它明确不输出 PASS、不更新台账，不是替代验收流程。

## 可以核实的主指标（仅 CSV 局部核验）

| 公共份额 | OOF BA | 正确行 | 相对 50% 的百分点变化 |
|---|---:|---:|---:|
| 50% | 0.4682426050 | 1264 | 0 |
| 25% | 0.4597091723 | 1241 | -0.8533 |
| 75% | 0.4649167288 | 1255 | -0.3326 |

50% 在这次主指标上最高，尚无理由替换原份额。这里是单独 Stylegram 的分组 OOF，不能与 EXP-004 Public LB 0.662 相减，也不能解释为整套方案下降。

## prompt 配对诊断（CSV-only，不是完整验收）

沿用原脚本方法：150 个 prompt，seed=42，2000 次按 prompt 的配对重采样；候选和对照共享同一批抽样。条件于本次已经拟合的模型和开发折，未作候选选择校正，不是重训不确定性或隐藏榜单置信区间。

| 候选对 50% | BA 差的 95% 抽样区间（百分点） | 修正旧错 / 引入新错 | 改善 / 下降折数 |
|---|---|---|---|
| 25% | [-1.8178, +0.1120] | 88 / 111 | 1 / 4 |
| 75% | [-1.3692, +0.7410] | 108 / 117 | 2 / 3 |

两项区间都跨 0；不能把单次差异说成稳定显著优劣。逐折数值见 partial-analysis/csv_per_fold.csv、csv_diagnostics.json。

逐类结果有明显得失：75% 对 Gemini-3.6-flash 召回 +6.0 个百分点，但对 Gemini-3.7-flash -15.33 个百分点；25% 对 Kimi-K3 +6.0、GPT-5.6-sol +5.33，但 Claude-opus-5 -6.67。逐类样本仅约150条，这些是开发集描述，不能据此事后再为每类挑份额。完整逐类表见 partial-analysis/csv_per_class_deltas.csv。

## 耗时与文件

- Kaggle保存版本页面显示 3844.3 秒 successful。
- result.json 主程序 total_seconds=3822.3624 秒；两者统计边界不同，不冲突。
- 结果写出 UTC 2026-10-31 18:38:36.635（北京时间 11-01 02:38:36）。
- result.json：30 个 SVM、convergence_warnings=0、gpu_used=false；未保存各 SVM 权重。
- 本地验收是 CPU 读取/统计，无训练、无 GPU、无比赛提交。
- 本次项目副本位于云端；用户关机电脑的本地目录没有被写入。相关原始文件与证据包同步到既有 Drive 项目 experiments/EXP-010-012-stylegram-share/acceptance-20261101-partial。

## 下载阻塞与续办

CSV/JSON 的网页 Download 会新开标签显示原文，已完整取回；NPY/NPZ 的下载按钮没有可读取的 href，受支持的下载事件等待超时，没有可核验本地下载结果。云端没有已配置 Kaggle CLI 凭据，未提取 cookie/token，未创建新凭据。用户桌面当前不在线。

恢复条件：从已有认证的 Kaggle CLI 下载固定 V2 全量输出（项目 README 已给命令），或通过正常浏览器取得完整输出包并放入此 Drive 项目。不需要重训。

取得原件后，按文件名补齐 download-v2/results，重新运行原 verify_results.py；只有其真正生成 PASS 的 analysis/verified.json，才运行 write_report.py、更新 EXP-010/011/012 原始记录并 check/render 台账。EXP-004 继续保持不变。

五折稀疏场景必须按每个 share × m × repeat 合并五折再计算 BA；不能平均缺类折的 BA，不能用完整 CSV 的随机删行冒充原来重新聚类/中心化的稀疏推理。
