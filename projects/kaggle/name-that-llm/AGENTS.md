# 本项目的实验与研究记录约定

先读 `README.md`、`tracking/当前状态.md` 和任务涉及的 EXP/SRC 原始记录。项目现状以当前仓库、具体证据和实时远端查询为准，不以聊天或长期记忆中的阶段状态为准。

## 记录职责

- 自己的独立尝试登记在 `tracking/records/experiments/EXP-*.json`；重跑另建 EXP，明确对照。
- 别人的 Notebook、实验和社区讨论登记在 `tracking/records/sources/SRC-*.json`。固定源码版本与得分版本分别记录，未经绑定不得混用。
- 想法与排期写 `tracking/ideas.json`，不把待办登记为已执行实验。
- 四个 Markdown 总览由 `tracking/ledger.py render` 生成。不要手改视图或同时在旧长文维护最新状态。
- 运行控制器继续维护自己的 `cloud/*/execution-state.json`；不为了美化总览擅改控制状态。
- 新收集的社区资料按下方规则放 `research/`；现有证据继续在 `reviews/`、`cloud/`、`tracking/snapshots/`，可直接引用，不重复复制。记录使用项目相对路径。已有文件不随意改名、搬动或删除。

## 社区资料必须分类、逐条归档

适用于本窗口、其他 GPT/Codex 窗口、DOT 和子代理。开始收集前读取本文件、[research/README.md](research/README.md) 与现有 SRC 记录。

1. **每个独立来源条目一个目录。** 新资料分别放 `research/solutions/`（公开方案）、`research/discussions/`（社区讨论与他人实验）、`research/datasets/`（公共数据来源与用法）、`research/official/`（官方规则、公告和平台文档）。正式条目命名 `SRC-编号-简短名称/`，含独立 `notes.md` 和原始证据或其已有位置引用。跨主题用 `tags` 和关联 SRC，不复制多份正文。
2. **禁止把一批收集结果只写成一份大长文。** “收集 Top20”必须把候选逐条放入相应目录；排名、日期、搜索批次不是长期分类。可在 `tracking/snapshots/时间/top20-index.json` 保存排序依据、抓取范围、排名、分数及 SRC/目录链接，但它只作清单，不承载各方案的完整分析，也不能代替逐条归档。不要再新增“Top20方案汇总.md”作为资料主体。
3. **先查重，再新增。** 同一来源 URL/Notebook ref 与同一固定版本复用已有 SRC；不同固定源码版本新建 SRC 并关联前版；同一讨论的补充和纠正归入原 SRC，新增带时间的快照。不同作者的同代码副本保留出处并标记同源，不计作独立方法。已有原件直接引用，不为新的搜索批次复制。
4. **只有一套权威记录。** `tracking/records/sources/SRC-*.json` 是来源、作者、版本、观察时间、分数绑定、分类和证据路径的权威索引；`notes.md` 保存详细分析，JSON 只保留简短结论。导入清单和下载元数据须标明其角色，不能再手工维护另一份“最新状态”。新增条目填写 `category/artifact_dir/notes_path/tags`，并把 notes 与证据路径加入 `evidence`。
5. **保留证据与限制。** 每条说明“作者声称 / 我们核查 / 适用条件 / 对我们有什么用 / 怎样验证”，保留来源链接、采集时间、固定版本和可取得的哈希。未读取原文、未取到源码或未绑定分数的部分明确标未核验，不能凭标题补造。自己的训练结果仍归 EXP，不放入社区资料冒充作者证据。
6. **多窗口只由一个合并窗口分配正式编号。** 并行收集者先用唯一任务目录和来源 slug 暂存（如 `research/inbox/唯一任务名/solutions/作者--slug--版本/`），不得各自猜“下一个 SRC”。合并窗口对照最新记录查重、独占创建编号和目标目录；发现冲突重新核对，不覆盖别人的文件。无法写本地的窗口按相同分类交付目录包和逐条导入清单，不交付一个混合长文。
7. **收尾要更新索引。** 完成条目后运行 `python -X utf8 tracking/ledger.py check` 与 `render`，交接新增/更新的 SRC、目录、重复项和未核验项。未能运行本地校验就明确写“待合并/待校验”。只有逐条资料落盘、索引关联和必要校验完成，才称为本次收集完成。

详细目录、模板与 Top20 交付示例见 [research/README.md](research/README.md)。历史 Top20 长文及旧批次证据保持归档；新任务不沿用旧的大长文生成方式。数据集大文件和模型权重继续存 `data/` 或现有云端资产，research 只记录研究资料和位置。

## 开跑与收尾

- 开跑前固定假设、基线、代码、输入版本/哈希、验证划分、标签顺序、环境、seed、预算。已知缺口明确记录，不能从标题推断配置。
- 正式 Public LB 必须绑定自己的 submission ID 和 scriptVersionId，并核实 COMPLETE 与分数。公开三条示例、模拟测试、OOF、正式 Public/Private 成绩分别记录。
- 状态与分数都附观察时间、来源和证据；查询失败不覆盖上一次真实状态，也不把未知值填成 0。
- 耗时区分加载、训练、推理、保存与隐藏评分。提交到查询的时间差不等于执行时间。未知 GPU 小时/API 花费/Colab CU 留空。
- 记录失败和负结果；原因不明就写原因不明。重复尝试必须先查旧任务的实际状态。
- 修改记录后执行 `python -X utf8 tracking/ledger.py check` 和 `render`。辅助代码有行为变更时执行其针对性测试。

## 权限与边界

资料整理和状态同步不自动授权运行 GPU、消费 API 额度、正式提交、公开资源、停止他人/现有任务、创建自动化。按当前聊天授权处理。

不把密钥、OAuth token、签名下载地址放进记录、快照或上传包。不要将项目的当前状态写入长期记忆。

## 新开发文件的组织

- 开发 Notebook 放 `notebooks/`，主要承担配置选择、调用与展示；`name-that-llm-atlas20-repro.ipynb` 已移入该目录。固定基线保留 `reviews/` 原路径，通过 `notebooks/README.md` 索引，不制造重复的可编辑副本。
- 可复用计算逻辑逐步放 `src/`；固定验证继续复用 `validation/`，云端运行和传输继续放 `cloud/`。不为整理目录改写已经绑定哈希的历史代码。
- `configs/` 放可修改的默认模板；新实验用 `tracking/ledger.py new-experiment --title ... --slug ...` 同时登记台账并建立 `experiments/EXP-编号-名称/`。
- 每个新实验的 `config.json` 是当次参数来源，台账通过 `config_path` 关联；`manifest.json` 固定代码、输入、分折与环境依据。生成时为 `DRAFT`，不能视作已冻结或已验收。
- 每次的 OOF、指标和日志分别放该 EXP 内的 `oof/metrics/logs`；共用一次运行的现有 sweep 保留原关联，不重复拷贝产物。
- 大权重继续复用现有 Drive/Kaggle 资产，本地保留清单；小型 OOF、指标和验收证据可本地与云端备份。历史文件迁移须同时更新所有已知引用并验证，不能只为美化目录移动。
- 原根目录两份研究长文归档在 `archive/research-notes/`；当前信息统一进入 tracking。历史审计凭据里的路径和哈希保留原值，用 `tracking/migrations/` 的迁移映射关联新位置。
