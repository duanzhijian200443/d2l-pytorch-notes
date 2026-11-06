# EXP053 固定输入

`shared.csv` 由 SRC-068 得分版 V1（scriptVersionId 364709540）的原始 `load_extra_passages()` 函数生成。最终为 10,743 行、738 个题组，列序为 `prompt_id,model,text`。Apex 来源的公开 V1 API 提供附件 ref，Open Pool 的输入卡和保存日志还提供平台 V13 依据。其余 passage CSV 采用 SRC-068 采集时读到的当前公开版本，再以固定版本接口枚举和固化文件哈希。现存材料没有逐项绑定这些文件到作者隐藏评分运行的字节。[完整输入回执](input-receipt.json)

| 来源 | 本次固定版本 | 实际文件 | 原 loader 最终保留行 |
|---|---:|---|---:|
| Seyit | 2 | `ntl_extra_passages_claude.csv` | 2,209 |
| Lucas | 12 | `gen_passages.csv` | 3,012 |
| Abdullah | 9 | `ntl_generations.csv` | 1,102 |
| Dophuchung | 1 | `ntl_proxy_generations.csv` | 120 |
| prvs | 7 | `passages_clean.csv` | 4,300 |

官方 SDK 枚举了这套快照中全部 15 个数据集、8 个上游 Notebook 和 8 个模型的固定文件清单。六条 passage glob 共匹配上表五个 CSV，`ntl_generated*.csv` 匹配数为零。Seyit、Lucas、Dophuchung、prvs 复用已核哈希的本地固定原件，Abdullah 下载平台 V9 的单个实际匹配文件。大型权重只枚举文件名和大小。完整 SHA256、字节数和原件位置见[来源回执](source-snapshot/loader-file-receipt.json)。

EXP031 的实际共享表有 10,469 行、716 题。按规范化正文哈希比较，本次 10,743 行与它共有 9,399 条，本次独有 1,344 条，原表独有 1,070 条。净增 274 行掩盖了较多行替换。两次使用的 Lucas、Abdullah 版本不同，而且不同来源的同题同模型会按原 loader 顺序择首，不能把净增量解释为单一来源的新增训练资料。A1 Stylegram 专用的 10,073 行输入没有用于构造此表。

`panel.csv` 完整复用 EXP029 缓存的盲化顺序和字节，仅含 `id,text`。标签和题组在 `local-only/evaluation.csv`，以盲 id 和原文 SHA256 一对一连接。729 条面板覆盖 56 题、17 类，已多次用于探索。

- 主 SVC 的 V13 准备池为 190,588 行，按当前源码和固定文件投影，实际去重拟合表为官方 2,699、matched 24,443、V13 再入 173,336，共 200,478 条。729 条面板中有 **698 条**直接命中这张拟合表，涉及 **54/56 题**；按整题隔离后仅剩 15 行、2 题。698 条及各题分布见[本地重合索引](local-only/panel-overlap.csv)。
- 面板与官方原文、作者共享 EXTRA、four120、four240 的原样和规范化正文交集均为零。已有近题复核确认 four120 的 cp0029 与面板潮汐题接近，对应 12 行；four240 另有 cp0049 与版本控制题接近，两题共 24 行。这些检查只覆盖已知关系。[历史近题核查](../../EXP-042-luna-terra120-ablation/metrics/data-review.md)
- 这张面板可核对推理路径、逐行改判和覆盖缺口。不能据此选出泛化更好的 A/B/C 或预测 Public LB，也不能据剩下的两题给出独立收益结论。

共享表中有 14 组来自已知官方题目的重新生成回答，每组 18 条，共 252 条。`regen-fold-map.csv` 只有 `prompt_id,official_prompt_id,official_fold`。官方开发五折按对应折排除 36、54、36、72、54 行，每个实验臂执行同一隔离规则。完整面板的全量拟合保留原作者全部 10,743 行。该映射仅覆盖已知官方祖先。[映射与核验记录](regen-fold-map.receipt.json)

打包仅选训练所需 `shared.csv`、盲化 `panel.csv`、训练用 `regen-fold-map.csv`，另按运行配置引用已冻结的官方 train、folds 和 EXP052 两个新增臂。`local-only/` 和 `source-snapshot/` 只作本地评价及来源核查。输入准备没有训练、正式提交或改变远端资源。
