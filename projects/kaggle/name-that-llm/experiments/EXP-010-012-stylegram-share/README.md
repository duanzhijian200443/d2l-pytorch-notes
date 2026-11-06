# Stylegram 权重对照：执行入口

2026-11-01 已完成固定 V2 输出的完整验收，原核验与独立复算均 PASS。三份额主指标保留 50%；本次没有重训或提交。见 [结果报告](report.md) 与 [验收记录](acceptance-20261101/README.md)。DOT 的旧部分验收保留为历史快照。

实际状态以 [execution-state.json](execution-state.json) 与实时 Kaggle 为准；不要仅看旧 RUNNING 值决定重启。

- 实验：EXP-010（0.5对照）、EXP-011（0.25）、EXP-012（0.75）。
- 运行：[私有 Kaggle CPU Notebook](https://www.kaggle.com/code/zhjianduan/name-that-llm-stylegram-share-cv)，V2。GPU/TPU/互联网关闭，CPU运行上限7200秒。
- 输入：私有 Dataset `zhjianduan/ntl-stylegram-cv-inputs-v1` V1 加比赛输入。所有输入文件哈希在 [preflight.json](preflight.json)，公共数据10073行/694组，排除252条已知重建回答。
- `runner.py` 不支持断点恢复；任务运行时不得重复推送。V1启动器失败在模型训练之前，已保存错误日志。
- 结果返回后用 `verify_results.py --input <下载的results目录>` 校验，再执行 `write_report.py`。只有核验PASS才更新实验结论，不能用CLI退出0或Kaggle COMPLETE代替结果完整性检查。
- 稀疏BA要先合并五折，不能直接平均存在缺类的折指标。三份额和同一prompt的多次抽样均不算独立实验重复。
- 完整ATLAS20源码、LoRA和GPU实验未修改。本轮结果不等于排行榜提升。

Kaggle CLI网络必要时复用Windows已有代理（本次为本机127.0.0.1:7890），不要重装或重新登录。上传ZIP会被Kaggle展开；bootstrap定位哈希固定的protocol.json并验证所有文件再运行。

## 后续接手

1. 先查固定 V2 的 `kaggle kernels status zhjianduan/name-that-llm-stylegram-share-cv/2`，运行中用 `kernels logs .../2 --follow` 查看进度；不要再次 push。
2. 终态后下载：`kaggle kernels output zhjianduan/name-that-llm-stylegram-share-cv/2 -p experiments/EXP-010-012-stylegram-share/download-v2 -q`。
3. 找到下载目录中的 `results/result.json`。只有其 status=COMPLETE 才执行 `verify_results.py --input <results绝对路径>`，随后执行 `write_report.py`。ERROR 时先读取失败日志，不自动重训。
4. 核验完成后更新 EXP-010/011/012 的环境、OOF、耗时和结论，再运行台账 check/render；不正式提交比赛，不改变 EXP-004 固定基线。

当前读取源码的子代理使用 gpt-6-astra / max；没有使用 ultra 子代理。首次运行和后处理方法检查分别在 review-checks.json 与 postprocess-review.json，检查通过不代表实际训练结果已经验收。
