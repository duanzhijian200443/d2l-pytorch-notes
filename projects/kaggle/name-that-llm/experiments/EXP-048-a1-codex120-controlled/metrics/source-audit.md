# EXP-048 静态源码与120条输入审查

结论：**PASS_STATIC_SOURCE_AUDIT**。相对 EXP046 V2，仅 Stylegram 加入 EXP040 原文120条；其他集成计算源码与参数保留。只解析 JSON、AST、base64、ZIP、CSV并独立计算哈希及分组，没有执行 Notebook 或内嵌源码。本结论不代表隐藏集运行成功或成绩改善。

## 固定证据

- 新源码：[codex120-latest.ipynb](../../../tracking/snapshots/20261102T035507208171Z/codex120-latest.ipynb)，SHA256 `9ba9ab375efad299f8bc568674d30dba276ce72574ee237a784b0a6551a876d6`。
- 内嵌ZIP SHA256 `91f50161c84661387cc361965525f43368fb86c1e31d38e5e9021cab4a3798fc`；包内哈希清单全部匹配。
- 基线：[EXP046 V2](../../../cloud/a1-public-stylegram-046/readback-v2-20261101T174011837673Z-b9ca46/name-that-llm-a1-public-stylegram-046.ipynb)，scriptVersionId `364624288`。
- 父窗口取得的[官方SDK版本绑定](../../../tracking/snapshots/20261102T035507208171Z/codex120-version-binding.json)：新Notebook V2 → `364733823`，submission `57893384`。本审查读取该证据；未独立下载历史版本源码。

## 数据与隔离

内嵌 `new120.csv` 与 EXP042 既有候选逐字节相同，SHA256 `cbb0f2227f7748d9172f04d23c1653403103a192dbb7503d897153d24bb00acd`。独立复核 Luna60、Terra60，共60个原题组，每组两模型，全部train，原文哈希逐条匹配，row/job键唯一。协议为 `single-paragraph-plain-text-v1`，没有去格式处理。

holdout哈希列表与 EXP042 完全相同（原题库50个holdout题组hash），新增数据交集为0。EXP040新300中原holdout60条未进入新增CSV；不能把50个原题库holdout hash误称为此次60条回答或60个新题组。

## 改动范围与权重

- Notebook仅单元0（说明）、2（包与新增预检）、12（receipt元数据）变化，编号从0起算。Gaussian、main infer、comp stack和comp运行单元完全相同，单元12中的模型表和执行命令未变。
- 包内仅 `a1_stylegram.py`、哈希清单变化，并新增 `new120.csv`；其余7个原文件逐字节相同。
- Stylegram训练量2699官方 + 10073原公共 + 120新增 = 12892。新增数据参与TF-IDF及plain／centred SVC重拟合，新增题组单独居中。
- 官方每行仍为0.5；原公共每行为 `0.5 × 该类官方条数 / 该类原公共条数`。新增行继承同类原公共每行权重，不重新归一化；源码断言旧样本权重保持相同。
- 保留Stylegram原求解器、raw margin、其他main/comp成员及权重、family gates原共享训练池、main svc4、ATLAS路由；ATLAS0.20／comp0.15／Sinkhorn0.45／target_mix0.35／标点1.5不变。

## 限制

后续需独立核对隐藏执行、收敛及成员回执、正式输出和分数。ATLAS预算公式虽未改，新增训练耗时可能减少实际剩余预算。生成模型底层snapshot未知，语义重叠未穷尽排查。完整机器可读检查及每文件哈希见 [source-audit.json](source-audit.json)。
