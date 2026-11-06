# 少额度时的日常实验入口

在项目目录 `C:\计算机学习\深度学习\projects\kaggle\name-that-llm` 打开终端。以下命令复用现有解释器与登录，不安装依赖，也不启动云端任务。

## 最常用的三个操作

```powershell
python -B -X utf8 experiment.py list
python -B -X utf8 experiment.py check codex120
python -B -X utf8 experiment.py run-local sinkhorn030
```

- `list`：看已登记的四个对象；显示带证据的正式分数，不会联网刷新。
- `check codex120`：实时查询 EXP048 的固定正式提交，取回并核验已经完成的保存版本的小型产物，生成中文报告。已有下载通过哈希检查后复用。
- `run-local sinkhorn030`：核验并复用 EXP049 已完成的固定缓存对照。结果已存在时不训练、不重新加载模型、不重复计算，也不覆盖旧结果。

报告都在本目录的 [reports](reports/)；每次查询/核验的原始小收据在 [observations](observations/)，保留时间、版本和哈希。命令会打印报告的完整路径。下载首次可能需要几分钟，复用文件后的查询通常更轻；不承诺网络耗时。

## 四个对象

| 简写 | 编号 | 含义 |
|---|---|---|
| `a1` | EXP046 | 已正式得到0.663的A1，Sinkhorn0.45，不加自产120条 |
| `codex120` | EXP048 | A1只加入Luna/Terra120条的既有任务；正式结果以实时查询为准 |
| `sinkhorn030` | EXP049 | A1只将Sinkhorn改为0.30；候选已准备，未远端运行/提交 |
| `pool-v13` | EXP050 | 190588行公共池增量审计；没有训练、没有模型分数 |

也可以直接使用 `EXP-049` 等编号。未登记对象会拒绝运行。正式对照基线仍为EXP004的0.662，另一配方EXP047的0.664保留独立记录。

## 分清三种“成功”

1. **保存运行通过**：`PASS_COMMIT_SAMPLE_ONLY`。核验的是公开三行占位输出、输入检查收据、T4×2环境和合成数值自检。它不能证明隐藏集完整推理成功。
2. **本地诊断通过**：`PASS_FIXED_LOCAL_JOB`。代码/输入/结果绑定成立；是否提分要看详细报告的同组对照。本次729条诊断不叫干净OOF。
3. **正式评分完成**：状态为 `COMPLETE`，并绑定自己的submission ID、scriptVersionId和平台返回分数。只有这里能写“正式0.xxx”。

EXP048可以同时出现“保存运行COMPLETE、正式评分PENDING、公开占位核验通过”，这不是冲突。命令成功退出也不代表比赛已经出分。

## 其他命令

```powershell
python -B -X utf8 experiment.py doctor
python -B -X utf8 experiment.py status a1
python -B -X utf8 experiment.py collect a1
python -B -X utf8 experiment.py verify a1
python -B -X utf8 experiment.py prepare sinkhorn030
python -B -X utf8 experiment.py verify pool-v13
python -B -X utf8 experiment.py report sinkhorn030
```

`doctor`检查解释器路径是否存在；不证明登录或网络可用。`status`只查询；`collect`会建立一份新的小文件下载快照，日常优先用自动复用的`check`。`verify`只核验已有产物；`report`只根据已有证据生成报告，不联网。`prepare`发现候选已存在时调用只读验证器，不改旧验收收据。

当前入口专门覆盖这四个固定对象，不是任意Notebook启动器。新实验用`tracking/ledger.py new-experiment`登记，再由维护者接入固定版本和核验器。不要仅替换slug复用另一实验的验收。

## 出错时

- 退出码2表示本次操作未通过或状态未知；看对应观察收据的`error_code`，不要把未知当0分。
- 查询失败会保留上一次确认的正式分数及其时间。旧结果不是“现在仍成功”的实时证明。
- `LOCAL_JOB_CODE_CHANGED`、`ARTIFACT_HASH_MISMATCH`、`SOURCE_BINDING_CHANGED`：先核对文件与版本，不能直接改哈希让检查通过。
- `LOCAL_OPERATION_ALREADY_ACTIVE`：同一实验已有本地操作在执行；等它结束。锁文件会保留，但OS锁在进程结束后释放，不需要删文件。
- `INTERPRETER_NOT_AVAILABLE`：检查`configs/experiment-workbench.json`所指向的既有环境。该文件绑定本机路径，换电脑需要重新核验；入口不会自动重新安装。
- 网络/登录失败：先恢复现有代理与授权，再执行一次只读查询。入口没有无限重试、自动登录或新建任务分支。

不下载大权重；每个允许产物上限16MiB、一次64MiB。平台返回的日志只保存固定安全事件摘要，不保存原始日志、签名下载URL或凭据。完整隐藏运行失败仍需另读相应完整日志，不能只靠这份摘要定位。

## 下一次真正改方案

先等EXP048出分并核验，区分120条数据的效果。EXP049是单独的无120条参数候选，若决定上Kaggle，先核实际附件字节与现有任务，再按当次授权启动；本入口不提供push、GPU启动、停止、提交或数据生成命令。

Open Pool筛出的C1/C2仍是候选索引。先选择一个固定数据变化，写清类/来源配额、题组隔离及评价边界，再登记新的CPU对照。公开`match_weight`和上游筛选接触过全部官方训练数据，不能直接据此声称严格OOF收益。详见[EXP050报告](../EXP-050-open-pool-v13-increment-audit/report.md)。

未来一个月的节奏见[轻量排期](month-plan.md)。本地文件与缓存保留在项目和既有云端资产中，日常运行这些脚本不依赖持续调用最高档模型。
