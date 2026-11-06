# Name That LLM 项目入口

日常从这里进入。实验、社区资料和当前状态各有一个入口；模型、数据、日志和历史源码继续保存在 `cloud/`、`reviews/`，通过链接关联。

新开发采用 **Notebook 入口 → 共用代码与配置 → 按 EXP 归档结果** 的结构。开发 Notebook 放在 `notebooks/`，旧长文放在 `archive/`，固定基线与实验原始证据继续保留。

| 要了解什么 | 打开哪里 | 维护方式 |
|---|---|---|
| 少额度时怎样用一条命令查状态、验收、复用缓存 | [日常命令指南](experiments/EXP-051-experiment-workbench/guide.md) · [本轮交付](experiments/EXP-051-experiment-workbench/report.md) | `python experiment.py check codex120`；四个固定对象，只读远端，不启动GPU或提交 |
| A1只改Sinkhorn0.30是否值得上云 | [EXP049候选与缓存对照](experiments/EXP-049-a1-sinkhorn030-ready/report.md) | 729条净多对2条，差值区间跨0；候选已准备，尚无新榜单成绩 |
| Open Pool V13相对实际用数有多少增量 | [EXP050逐行审计](experiments/EXP-050-open-pool-v13-increment-audit/report.md) | 92718条是已知运行用数之外的文本上界；C1/C2均未准入训练，实际prvs运行字节对应V7 |
| 现在以哪个版本为基线、哪些任务在等结果、下一步做什么 | [当前状态](tracking/当前状态.md) | 从台账生成；每条远端状态标注核查时间 |
| 我们跑过什么、改了什么、实际分数和成本是多少 | [自己的实验台账](tracking/实验台账.md) | 每次独立尝试一个 EXP 编号 |
| 别人的有效实验、公开方案差异、社区讨论及后续修正 | [研究索引](tracking/研究索引.md) | 每个固定版本或讨论一个 SRC 编号 |
| Open Pool V13 更新、190588行与验证独立性限制 | [SRC-065 核查](research/datasets/SRC-065-open-pool-kaggle-v13/notes.md) | 固定平台V13；作者声明与我们EXP050的逐行审计分开记录 |
| Mercurydust V11 配对分支为什么不能直接照跑 | [SRC-064 源码核查与小型复现](research/solutions/SRC-064-mercurydust-v11-pairwise-index-audit/notes.md) | 第129条起批次索引错位；附件公开性已核，新榜单成绩未绑定 |
| Lucas 公共数据新增192条是否是真实增量 | [SRC-066 固定版本差异](research/datasets/SRC-066-lucas-free-proxy-v12/notes.md) | V12对V11新增192；与当前池/Open Pool的净增量仍需去重 |
| 让其他 GPT/DOT 窗口收集社区资料，如何分类归档 | [社区资料规范](research/README.md) · [项目规则](AGENTS.md) | 按类别、按条目保存；Top20只作清单，不能合成一份长文交付 |
| 哪些方向值得试、怎么判断有效、优先顺序是什么 | [待验证想法](tracking/待验证想法.md) | IDEA 只表示待验证假设，不计入实验结果 |
| 为什么选择这条路线、哪些坑已有结论 | [方法与决策记录](tracking/方法与决策.md) | 保留理由；结论变化时补充依据 |
| 数据是否可靠、验证怎样划分、轻量模型错在哪里 | [第一步诊断报告](experiments/EXP-007-validation-foundation/report.md) · [固定验证协议](validation/README.md) | EXP-007 复核历史轻量 OOF；不代表强集成的验证结果 |
| 公共数据哪些能先用、哪些要隔离、来源之间重复多少 | [公共池检查报告](experiments/EXP-008-public-pool-audit/report.md) · [数据入口](data/README.md) | EXP-008 已检查六个固定版本；保留原始数据和逐行风险标记 |
| 0.662 基线每个成员已经用了哪些数据，什么才算增量 | [基线用数清单](experiments/EXP-009-baseline-data-lineage/report.md) | EXP-009 固定源码审计与当前固定数据投影；据此修正优化排期 |
| Stylegram 三组公共权重 OOF 是否可靠、哪组值得保留 | [权重对照结果](experiments/EXP-010-012-stylegram-share/report.md) · [完整验收](experiments/EXP-010-012-stylegram-share/acceptance-20261101/README.md) | EXP-010/011/012 已验收；保留50%，单成员结果不等于榜单变化 |
| 主 svc4 加入可追溯原文后是否改善、公共权重该多大 | [四组 CPU 结果](cloud/svc4-raw-public-r1/README.md) · [完整验收](cloud/svc4-raw-public-r1/analysis/verified.json) | EXP-017～020已验收；0.35保留作本raw配方开发对照，不能当作ATLAS20榜单提升 |
| Stylegram与svc4是否互补，回答变少后结论是否改变 | [互补性与稀疏场景报告](experiments/EXP-022-oof-complementarity/report.md) | EXP-022已验收；完整组Stylegram更强，m=2的五次抽样svc4均领先；该次仅作诊断，融合筛选另见EXP-024 |
| svc4以小比例参与融合是否值得保留 | [固定5%/10%/20%融合结果](experiments/EXP-024-oof-margin-blend/report.md) | EXP-024已验收；10%完整组净多对2条但区间跨0，稀疏均值改善，仅作为独立验证候选 |
| 固定10%候选在外部数据上有没有增益 | [四类外部验证](experiments/EXP-026-independent-blend-validation/report.md) | EXP-026已验收；5200条/1299归并题组净多对54条、四类宏召回+1.0385个百分点；题组勘误已保留，不是完整18类或榜单提升 |
| 增益能否延伸到另一多家族来源、怎样接入完整集成 | [17类外部检查](experiments/EXP-027-multiclass-external-validation/report.md) · [分离对照方案](experiments/EXP-027-multiclass-external-validation/integration-plan.md) | EXP-027已验收；729条/56题上宏召回−0.4335个百分点、区间跨0，暂不升级候选；完整GPU集成尚未执行 |
| 怎样保存完整集成中间分数，之后在CPU上复算 | [真实缓存验收](experiments/EXP-031-full-stack-capture-r1/report.md) · [固定参数对比](experiments/EXP-032-sinkhorn-fixed-cache/report.md) | EXP-031真实缓存已验收；EXP-032中0.30净多对4条，区间跨0。外部17类探索结果，保留正式0.662基线 |
| 新采集的Luna/Terra120条是否改善轻量分类器 | [新增数据对照结果](experiments/EXP-042-luna-terra120-ablation/report.md) | EXP-042完成：control精确复现；加120主OOF净少对2条、区间跨0，稀疏均值略升但不一致；暂不合入正式方案 |
| 旧120条去换行和加粗是否改善、两路并行耗时如何 | [格式对照与提速结果](experiments/EXP-044-format120-parallel/report.md) | EXP-044验收通过：原文0.466388、去格式0.465648，净少对2条、区间跨0；两路CPU守护28分14秒，不采用默认去格式 |
| 既有公共数据改动接入完整ATLAS20是否改善 | [完整缓存四臂对照](experiments/EXP-045-public-data-full-stack/report.md) | A0/A1已验收：17类宏召回均0.601075、最终改判0；标准化与10%融合两臂缺4条ATLAS预测，完整分数未计算 |
| 从哪里打开 Notebook、修改代码和选择参数 | [Notebook入口](notebooks/README.md) · [共用代码](src/README.md) · [配置模板](configs/README.md) | 现有开发副本由入口链接；新增共用逻辑和参数各有位置 |
| 新实验如何建目录、预测和日志放哪里 | [实验目录说明](experiments/README.md) | 新建命令同时登记 EXP 和创建产物目录 |
| 原来的长篇方案记录和 Top20 汇总在哪里 | [历史归档](archive/README.md) | 保留当时内容和证据；最新信息查台账 |

## 一次实验怎样记录

1. **开跑前**：用新建命令登记 EXP 并生成目录；完善当次 `config.json`、`manifest.json` 与台账，写清对照基线、假设和唯一主要改动，固定代码、数据版本、验证划分、环境、seed 与预算。新建清单为草稿，未知字段留空并说明，不填猜测值。
2. **运行中**：记录平台运行 ID、Notebook 保存版本、全局 scriptVersionId、状态与核查时间。`RUNNING` 是某次观察，不能永远当作实时状态。
3. **结束后**：补正式提交 ID、OOF、Public LB、逐类召回、耗时、成本、失败原因、日志路径和一句结论。重跑另建 EXP，通过 `baseline_id` 关联。
4. **刷新入口**：更新结构化记录后生成四个 Markdown 页面。不要同时手动维护多份“当前分数”。

例如：`EXP-005` 是 Sinkhorn=0.60、标点系数=1.2；`EXP-006` 是 0.30/1.5。标题都可以叫 Who are U，但台账会固定各自的参数和来源版本。

## 社区资料怎样进入我们的方案

每条 SRC 必须说明 **来源说了什么 → 我们核查了什么 → 有什么适用条件 → 准备怎样验证**，并链接具体版本、讨论或评论以及本地证据。完整帖子和源码保存为快照，摘要只保留有用信息。

新资料按 [research 规范](research/README.md) 分入公开方案、社区讨论、公共数据和官方资料四类，每个来源条目独立目录。已有资料先查重、复用原 SRC 和证据；不为每次“Top20收集”重建一份混合长文。并行窗口由一个合并者分配正式编号，远端窗口按同一结构交付待导入目录。

- 平台分数与下载的最新源码分别记录；没有确切版本绑定，就标记“未绑定”。
- 社区报告的实验不是我们的实验；不能拿它更新自己的最好分数。
- 同一代码的副本按代码哈希识别；Top20 不能直接当成 20 种独立方法。
- 发现作者更新、反例或复现失败，要补进原 SRC；重要的新源码版本单独建 SRC，并保留旧版。
- 判断有用后才建 IDEA；实际开跑才建 EXP。负结果和未解释的中断也要保留。

## 更新命令

在项目目录运行。`check` 和 `render` 只操作本地记录；`sync-kaggle` 只读取 Kaggle 并更新本地快照，**不会启动 GPU、提交比赛、发布资源或停止已有任务**。

```powershell
python -X utf8 tracking/ledger.py check
python -X utf8 tracking/ledger.py render
python -X utf8 tracking/ledger.py new-experiment --title "固定数据比较 Sinkhorn 0.30 与 0.45" --slug sinkhorn-comparison
```

`new-experiment` 只登记计划和生成文件，不启动训练。默认复制 `configs/experiment-defaults.json`；可用 `--config configs/你的配置.json` 选择其他模板。每次的配置独立保存，后续修改默认模板不会改变旧实验。

当前机器可使用已经配置好凭据的 Kaggle CLI Python 环境刷新：

```powershell
& 'C:\Users\34331\AppData\Roaming\uv\tools\kaggle\Scripts\python.exe' -X utf8 tracking/ledger.py sync-kaggle
```

刷新只关联**已登记的提交 ID**，并查询已登记 Notebook 的固定保存版本。新的正式提交要先补入对应 EXP；不要通过相似标题猜测关联。原始返回保存在 `tracking/snapshots/`。当前没有新增定时刷新任务。

## 文件的职责

```text
README.md                         日常入口
AGENTS.md                         本项目记录维护约定
notebooks/                        开发 Notebook 与入口说明
  name-that-llm-atlas20-repro.ipynb 当前 VS Code / Colab 开发副本
src/                              逐步提取的新共用计算逻辑
configs/                          可复用的默认配置模板
validation/                       固定分折、标签顺序与验证规则
data/                             数据清单、来源、原始公共池和审计索引
experiments/
  README.md                       新建与归档说明
  EXP-编号-名称/
    config.json                   当次参数快照
    manifest.json                 代码与输入清单，初始 DRAFT
    oof/                          逐行预测与分折缓存
    metrics/                      指标、误差分析和验收结果
    logs/                         执行日志
    report.md                     结果解释、限制和下一步
tracking/
  当前状态.md                     生成视图
  实验台账.md                     生成视图
  研究索引.md                     生成视图
  待验证想法.md                   生成视图
  方法与决策.md                   方法、决策与已知限制
  records/experiments/EXP-*.json   自己实验的唯一台账
  records/sources/SRC-*.json      研究资料的唯一登记
  ideas.json                     待验证事项
  settings.json                  固定对照基线等设置
  templates/                     新实验和资料的字段模板
  snapshots/                     带时间的 API/帖子证据
  ledger.py                      校验、生成、只读刷新
cloud/                            原有运行脚本、缓存清单与产物
reviews/                          原有公开源码、差异、审查证据
research/                         新社区资料逐条归档
  solutions/                      公开方案与固定源码版本
  discussions/                    讨论、他人实验及后续纠正
  datasets/                       公共数据资料与使用限制
  official/                       官方规则、公告和平台文档
archive/research-notes/            原根目录的两份历史研究长文
```

`cloud/*/execution-state.json` 仍由对应运行控制器维护。总览不靠扫描其中所有旧 RUNNING 字段判断是否仍在运行；实验结果按正式提交和终态证据登记。

## 历史材料

- [原公开方案与实验记录](archive/research-notes/公开方案与实验记录.md)：保留方法分析和历史过程，停止承担当前状态入口。
- [公开 Top20 历史收集](archive/research-notes/公开高分方案Top20-2026-10-29.md)：文件名沿用旧日期，内容包含 10 月 29 / 30 日快照；新增研究从 SRC 索引查找。

两份历史长文已迁入归档，正文只调整相对链接；模型和运行目录保持原路径。后续大量训练产生的权重留在云端，本地记录位置、版本、哈希和必要的小型指标文件。

已有 `EXP-007` 至 `EXP-012` 的历史目录结构继续有效；其中三份额共享的 sweep 不拆成三份重复产物。官方 CSV 暂留根目录以保持验证入口；开发 Notebook 已迁入 `notebooks/`，固定基线仍在 `reviews/`。本次迁移映射见 [迁移记录](tracking/migrations/20261101-folder-relocation.json)。
