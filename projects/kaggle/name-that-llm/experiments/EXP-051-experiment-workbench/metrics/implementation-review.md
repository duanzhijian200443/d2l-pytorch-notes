# EXP-051 首版实现审查

状态：CHANGES_REQUESTED。只读源码审查与临时目录/mock验证；没有远端调用、模型执行或修改现有结果。

## [1] WB-R1 新确认的正式分数会被下一次失败查询从报告中抹去

位置：`src/workbench_core.py` — remember + historical_score; make_report in experiment_workbench.py。

status成功仅更新latest/status；canonical不写回。下一次ERROR替换latest后report只从canonical查历史，EXP048原PENDING将显示尚无确认分数。旧观察文件虽留存，用户入口不再展示已确认值。

修正建议：另存按固定submission/script绑定的last-confirmed score指针；UNKNOWN/ERROR/缺score均不能覆盖。report同时展示最近查询与最后确认结果。

证据：临时目录构造T1 COMPLETE score .664 + T2 ERROR：make_report不含 .664，已离线复现。

## [2] WB-R2 报告在核验失败后仍展示未经核验的本地摘要数值

位置：`src/experiment_workbench.py` — make_report local_summary block。

local_summary被直接read，没有绑定已验收收据的hash/身份。即使verify ERROR仍输出改后的数字；旧PASS也不证明摘要当前字节未改。

修正建议：成功本地验收保存summary path/hash/ID/scope；report重新核hash再显示，缺绑定/不符时仅说明未核验，不展示候选数值。

证据：临时summary候选correct=999并提供最近verify ERROR：报告仍显示999，已离线复现。

## [2] WB-R3 离线verify会覆盖已存在的包验收原件

位置：`src/experiment_workbench.py` — local_job + JOB_SPECS verify-package。

offline_verify(EXP049)先执行prepare_verify.py verify；该脚本verify()每次save metrics/package-verification.json。prepare已有结果也走同路径。固定旧收据被写入，不能称这条分支为只读验证。

修正建议：对已准备包用verify_results里现成只读source_gate/validate_candidate路线或新只读入口；只有prepare首次创建收据。重复check/verify不修改原包与其收据，观察结果写EXP051目录。

证据：静态调用链确认，未执行该写入分支。

## 已确认的边界与尚未验证

固定目标、下载文件白名单与流式限额、异常输出最小化、公开样例限定、受限本地命令均已实现。没有远端push/submit/cancel路径。EXP050脚本hash正在冻结，不将其临时不一致当缺陷。新入口真实远端collect尚未由本审查员验证。

还需检查：run-local动态import之前应核所有冻结本地依赖；并发prepare需按EXP互斥；EXP048如声称新增数据输入政策已验收，应读取add120_preflight而非仅检查最终receipt的声明。
