# EXP-049：A1 仅将 Sinkhorn 0.45 改为 0.30

候选包状态 **READY_NOT_LAUNCHED**；本地 CPU 缓存诊断 **COMPLETE_LOCAL_DIAGNOSTIC**。未远端保存、运行或提交，未重新训练，未添加自产120条。

## 准备包

直接基于 EXP046 冻结 A1 Notebook（SHA256 `cea8f8cbfa382306dbdd7b7361b16d1d68febc09d7ef585dcb01ee1ef9e46625`）生成。唯一数学改动是主命令 `--sinkhorn '0.45'` → `'0.30'`；正文、bootstrap注释、experiment_id及sinkhorn收据更新为EXP049。原源码、嵌入payload、模型成员、Stylegram配方、原sharedgate数据、ATLAS权重及动态路由保持原样。隐藏test的动态快速提交判定和runtime检查也保持原样。

- 候选：[a1.ipynb](../../cloud/a1-sinkhorn030-049/kernel/a1.ipynb)，SHA256 `35ddcc24900d4bace4b3c4cc167475ad2c119af83a9eb687ef717f7bea1200b5`。
- metadata：[kernel-metadata.json](../../cloud/a1-sinkhorn030-049/kernel/kernel-metadata.json)，SHA256 `8c0a1ad41ecda521e57a0072bb68d93bbfa2ce65014a4ee42ffb0d09d679ba6f`；仅id/title更名，32项输入**请求字符串**和原runtime hash守卫一致。
- 精确允许差异：[candidate.diff](../../cloud/a1-sinkhorn030-049/candidate.diff)；修改cell为0、2、12；其余cell逐对象一致。
- [冻结manifest](manifest.json)在诊断前生成，其 `local_diagnostic_completed=false` 是冻结时状态；真实本次完成状态见[结果摘要](outputs/summary.json)。

32项请求一致不代表平台当前实际解析版本已全部认证。已知调查发现部分附件请求版本可能与原运行实际字节版本不符；本包忠实保留EXP046请求和守卫，不据此声称所有远端附件已经重新下载/重hash验证。大文件原守卫的size-only边界也保留。远端准备核验应由单独的输入审查包完成。

## 固定本地对照与结果

假设：在相同A1缓存预测、样本及ATLAS路由上，仅降低Sinkhorn权重是否改善覆盖17真类的macro recall。本次预先固定0.45与0.30两个点，不搜索其他参数。

复用 EXP045 的真实 A1 Stylegram缓存和 EXP031 完整成员缓存，由原 `math_replay.py` 严格模式重建A1 pre-Sinkhorn，再复用 `src/sinkhorn_compare.py` 的固定两个分支。没有重写数学。A1请求ATLAS共206条，全部有原缓存，缺失0；若有任何缺失则空分并失败，禁止旧mask或零值填补。A1的0.45 logits与EXP045原产物**逐元素、逐bit一致**。

| 本地诊断 | A1 / 0.45 | A1 / 0.30 |
|---|---:|---:|
| 覆盖17真类macro recall | 0.60107543 | 0.60329923 |
| 正确数 / 729 | 488 / 729 | 490 / 729 |
| accuracy | 0.66941015 | 0.67215364 |

0.30修正5条、新错3条，净多对2条；总计11条预测改变，另3条为两边都错但预测不同。Macro recall变化 **+0.22238个百分点**。仍保留全部18个输出类别（含无真值覆盖的gemini-3.7-flash），不把指标改成18类平均。

56个prompt整组配对bootstrap，2000次，seed42，95%差值区间 **[-0.38099, +0.88649]个百分点**，2000次均覆盖17真类。区间跨0，不能认定稳定增益，也不是等价证明。该区间条件于现有模型和已被多次查看的外部面板，不含重新训练不确定性。

完整证据：

- [729条逐行预测与正确性](outputs/predictions.csv)
- [11条改判，含改对/改错](outputs/changed_rows.csv)
- [18输出类逐类明细，缺真类recall为空](outputs/per_class.csv)
- [56题组明细](outputs/per_prompt.csv)
- [配对统计及bootstrap](outputs/summary.json)
- [原A1路由覆盖](outputs/coverage.json)、[固定分数NPZ](outputs/scores.npz)、[产物哈希](outputs/artifacts.json)

这是本地缓存诊断，**不是干净OOF、不是正式LB**。历史A1的Public LB 0.663属于另一评估口径，不能与此处0.601075/0.603299作数值差比较；本次没有新的正式分数。

## 执行与验证

本地CPU真实运行耗时 **4.965633秒**（包含输入验证、缓存重建、两点后处理、指标、末尾再次核验）。Python3.13环境的精确版本见summary。模型推理0次、模型拟合0次、GPU0秒、远端调用0次。

32项测试通过：候选差异和非授权变更拒绝4项、既有严格缓存数学/缺失覆盖测试7项、既有统计及bootstrap测试7项、固定Sinkhorn/typography测试14项；详见[测试收据](metrics/tests.json)。本地测试不代表云端Notebook或隐藏评分已成功。

可供统一CLI调用：

```powershell
python -B -X utf8 cloud/a1-sinkhorn030-049/prepare_verify.py verify
python -B -X utf8 cloud/a1-sinkhorn030-049/test_prepare_verify.py
```

`prepare` 已执行并冻结，再次执行会拒绝覆盖。`experiments/EXP-049-a1-sinkhorn030-ready/run_local.py` 已执行一次，有结果时拒绝覆盖/重调。该准备包没有远端save/run/submit入口，也没有复制旧控制器或旧intent。
