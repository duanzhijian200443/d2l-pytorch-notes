# EXP056 / EXP057 正式候选的构建与检查

两份 Notebook 直接替换 EXP054 原始 JSON 中零基索引 12、14 两个 cell 的 source 字符串。其他 15 个 cell 和其余原始 JSON 字节保持一致。metadata 仅更新 id、title、code_file。两份包均有 15 个数据集、8 个 Notebook、8 个模型，另有比赛输入，容器与机器形状继承 EXP054。[构建清单](manifest.json)

| 候选 | 新增行 / 题组 | Notebook 字节 | 生成的 comp_stack.py 字节 |
|---|---:|---:|---:|
| EXP056 | 120 / 30 | 212010 | 126175 |
| EXP057 | 240 / 60 | 280798 | 194426 |

第 12 个 cell 将 EXP052 原 CSV 压缩后内嵌。Stylegram 在官方、原公共和新增原文上共同拟合词表，新增四答题组独立居中。原行权重直接使用作者函数，新行使用同类原公共行权重，分母只计原公共行。原 `fit_svc`、测试分组和 `add_member` 的代码保持一致。成员成功加入后，收据记录新增行数、公共数据哈希、每个新增类别的实际样本权重、两次拟合及本次运行标识。家族门控继续读取作者原来的共享数组。[成员模板](volume_block.py.in) · [运行收据模板](member_receipt.py.in)

第 14 个 cell 在公开快保存时检查实际生成模块、内嵌数据、作者 loader 读取的 10,743 条 / 738 组公共数据，写入 volume_preflight.json。隐藏分支要求 comp 子进程成功退出，核验本次成员收据、测试 ID、18 类标签、数值型 seconds、comp_logit 的形状和有限性，然后按原位置记录 COMP_SEC。原 infer.py、gauss.py、sibling_style.py 的输出哈希由清单固定。[执行模板](run_cell.py.in)

本地测试将 Kaggle 路径映射到 EXP053 归档数据，实际执行两臂公开预检。隐藏分支测试用模拟子进程输出检查合法连接及非零退出、缺失收据、无效 seconds 的拒绝路径。真实隐藏执行及成绩由竞赛端独立产生。[本地预检](local-preflight.json)

从项目根目录执行：

```powershell
python -B cloud/apex-volume-formal/generate.py verify
python -B cloud/apex-volume-formal/test_generated.py
python -B cloud/apex-volume-formal/generate.py verify-materialized four120 path/to/comp_stack.py
```
