"""Build the Markdown research catalogue from captured evidence and reviewed notes."""
from datetime import datetime, timedelta
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
ui = json.loads((ROOT / 'ui-scores.json').read_text(encoding='utf-8'))
receipts = {x['ref']: x for x in json.loads((ROOT / 'sources-index.json').read_text(encoding='utf-8'))}
notes = json.loads((ROOT / 'curated-notes.json').read_text(encoding='utf-8'))
rows = ui['rows']
assert len(rows) == len({x['ref'] for x in rows}) == 20
assert set(receipts) == set(notes) == {x['ref'] for x in rows}
assert all(rows[i]['score'] >= rows[i+1]['score'] for i in range(len(rows)-1))
assert all(x['status'] == 'DOWNLOADED_NOT_EXECUTED' for x in receipts.values())
local_time = (datetime.fromisoformat(ui['observed_at_utc'].replace('Z', '+00:00')) + timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S')
prefix = 'reviews/public-top20-20261029'

def escape(s):
    return s.replace('|', '\\|').replace('\n', ' ')

text = f'''# Name That LLM：公开高分 Notebook Top 20 思路收集

核查时间：**{local_time} 北京时间**（UTC：`{ui['observed_at_utc']}`）。本轮只读取公开资料、下载源码和整理差异，没有启动模型实验或提交比赛。

## 1. 收录口径

- 对象是比赛 **Code 页面按 Public Score 排序的公开 Notebook**，不是比赛总排行榜前 20 支队伍，也不是 20 种互不重复的原创算法。
- 先通过 Kaggle MCP `search_notebooks(ScoreDescending)` 发现候选，再从网页逐条读取实际 `Score`。MCP 返回列表不带分数，且其中一项 0.660 出现在较低分项之后，所以本表以网页实际分数和显示顺序整理。
- 排除被固定在顶部、分数为 0.322 的官方 Starter；同分不分优劣。第 20 项处为 **0.651 并列**，后面还有同分条目，本次按页面顺序取满 20 项。
- 所有条目已保存源码、元数据、抓取时间及 SHA-256。**平台显示分数属于这个 Notebook 的得分记录；本轮没有逐一把分数绑定到最新下载源码的确切历史版本。** 最新源码可能包含尚未验证的新实验。
- 网页 URL 带平台默认 `excludeNonAccessedDatasources=true`。本轮网页列出的 20 项与 MCP 候选前 20 的集合一致；本文不声称覆盖未公开、不可访问或尚未评分的方案。
- 下文“思路”来自实际代码和作者说明，不表示我们已经独立复现其训练、数值一致性、成本或正式得分。

来源：[Kaggle Public Score 列表]({ui['source_url']}) · [网页逐条分数证据]({prefix}/ui-scores.json) · [原始 MCP 排序]({prefix}/search-order.json) · [源码及哈希索引]({prefix}/sources-index.json)。

## 2. 二十项总表

| 序号 | Notebook / 作者账号 | 平台 Public Score | 下载保存版本 | 核心增量或定位 |
|---|---|---:|---:|---|
'''
registry = []
for rank, row in enumerate(rows, 1):
    ref = row['ref']; r = receipts[ref]; n = notes[ref]
    version = r['metadata']['currentVersionNumber']
    url = 'https://www.kaggle.com/code/' + ref
    text += f"| {rank:02d} | [{escape(row['title'])}]({url}) · `{ref.split('/')[0]}` | **{row['score']:.3f}** | {version} | {escape(n['summary'])} |\n"
    registry.append({'display_order': rank, 'ref': ref, 'title': row['title'], 'platform_score': row['score'],
                     'score_observed_at_utc': ui['observed_at_utc'], 'downloaded_version': version,
                     'source_sha256': r['source_sha256'], 'code_only_sha256': r['code_only_sha256'],
                     'source_directory': r['directory'], 'source_and_score_exact_version_binding': 'NOT_CHECKED', **n})

text += '''
## 3. 最值得先理解的共同思路

### 3.1 公共多模型配方是共同基础

这些高分方案大量共享 Dhruv 的两阶段 LoRA、公共外部训练池、Qwen/Llama 适配器、SVC、Gaussian/twin、互补集成和 ATLAS。先读 **第 16 项 nlm-v27** 与 **第 19 项 Fully Reproducible**，比从头重复阅读 20 次相似代码更有效。

它们训练的是“识别作者模型的鉴定员”。使用 Llama 或 Qwen 骨干，不等于模拟或蒸馏出比赛的每一个目标模型。

### 3.2 小幅高分增量往往出现在融合和后处理

- **第 01 项：ATLAS 20%**，让疑难文本的复核证据获得更大权重。
- **第 12 项：Soft Sinkhorn 0.5**，减少强制均匀类别比例造成的偏移。
- **第 05 项：V31 家族门控与 Bayesian typography**，限制修正范围和强度。
- **第 20 项：SVC 题目偏差修正**，在密集 prompt 组内减少题目带来的干扰。

如果已有兼容的逐成员和 ATLAS 预测，可以优先做 CPU 融合对照；若调度改变了被 ATLAS 处理的样本，或者新增模型，就不能假定无需重新计算。

### 3.3 新模型需要提供不同的错误，而不只是更大

第 02 项加入 27B GSQ LoRA；第 07 项加入两个 GLM 模型；第 17 项加 Qwen2.5 v5；第 09 项把多条路线合在一起。更大的集成并未在本次列表中自动排到更前。这个列表不是受控实验，不能据分数差单独推断某个模型有效或无效；应观察新增成员与当前基线的错误相关性。

### 3.4 工程优化本身有实际价值

第 01/02 项的本地磁盘暂存和进程隔离、第 13 项的成员预测导出和减少无效拟合、第 20 项的总截止/提前保存，都能帮助降低试验成本和避免“跑完却不知道哪里出了问题”。快速三条样例预览只是格式/部分路径检查，不是完整复现。

### 3.5 去 Llama 已有可参考路线

第 07 项平台为 0.659，第 08 项为 0.658。二者都涉及去掉 Llama，但替代模型及其他设置不同，因此不能把它们与 0.663 的差值当成 Llama 的单独贡献。我们已有 Llama 缓存，不需要因此立即更换基线；这些条目适合作为成本与依赖权衡的候选。

## 4. 每项的具体思路与限制

'''
for row in registry:
    rank = row['display_order']; ref = row['ref']; directory = row['source_directory']
    text += f"### {rank:02d}. {row['title']} — 平台 {row['platform_score']:.3f}\n\n"
    text += f"作者账号：`{ref.split('/')[0]}`；本次下载保存版本 **{row['downloaded_version']}**。定位：**{row['family']}**。\n\n"
    text += '\n'.join('- ' + idea for idea in row['ideas']) + '\n\n'
    text += '**需要分清：**' + row['caveat'] + '\n\n'
    text += '**对我们的用处：**' + row['reuse'] + '\n\n'
    text += f"证据：[公开页](https://www.kaggle.com/code/{ref}) · [保存源码]({prefix}/{directory}/source.ipynb) · [原文说明]({prefix}/{directory}/description.md) · [抓取与版本记录]({prefix}/{directory}/receipt.json)。\n\n"

text += f'''## 5. 已确认的重复与容易误读的地方

| 观察 | 本次证据 | 解读 |
|---|---|---|
| 第 05 项 mercurydust 与第 06 项 ozertuu | 拼接全部代码单元后的 SHA-256 相同 | 不是两个独立的新算法；仍需分别检查输入附件 |
| 第 16 项 nlm-v27 与第 18 项 avazaku | 拼接全部代码单元后的 SHA-256 相同 | 后者主要贡献是依赖恢复、校验与复现文档 |
| 第 08 项 Llama-free 正文仍写 0.663 | 平台实际显示 0.658；代码去掉 ll8v5 | 继承的 README 成绩不能算在当前分支头上 |
| 第 10 项正文写 0.670+/0.700+ | 平台实际显示 0.655 | 目标分数不是实验结果 |
| 第 04 项 Who am I 已更新到保存版本 33 | 最新正文为 V40A；我们的固定 V32 正文为 V40B | 可变 URL 和内部方法编号都不能替代固定版本与源码哈希 |
| 第 14 项说明沿用 avazaku | 代码已经新增成员、调权、扩展外部数据路径 | 不可仅读 Markdown 就宣布“原样复现” |
| 第 20 项局部验证有 +0.01742 | 这是 train-only SVC 代理的分组验证 | 不是全模型或 Public LB 增益 |
| 多个方案跳过三条样例的重推理 | FAST_COMMIT、精确样例预览或 rerun 环境变量 | 几分钟保存完成不等于几分钟跑完整隐藏测试 |

差异证据位于本次收集目录的 `diff-*.txt`；其中数字是**最初 MCP 候选序号**，不是本报告按网页分数整理后的序号。请按 [sources-index.json]({prefix}/sources-index.json) 的 `ref` 对照，避免错配。

## 6. 结合我们现有资产的阅读与实验顺序

| 顺序 | 先看什么 | 为什么 |
|---|---|---|
| 1 | 第 16/19 项的公共训练与集成基础；第 05 项的 V31 改动 | 理解大多数条目的共同来源，避免重复部署 |
| 2 | 第 01 项的 ATLAS 权重、第 12 项的 Soft Sinkhorn、第 20 项的 CPU 修正 | 适合在固定基线和同一验证划分上做较便宜的消融 |
| 3 | 第 13/20 项的预测导出、截止、落盘；第 01/02 项的加载优化 | 直接改善我们的实验速度、可复用性和失败可诊断性 |
| 4 | 第 03/15 项的失败对照 | 尤其注意完整 prompt 组上的高 OOF 不一定适合稀疏隐藏组 |
| 5 | 第 02/07/09/14/17 项的新模型或大集成 | 新资源和更多计算应由独立增量证据支持，放到后面评估 |

以上是研究顺序，不是自动执行计划。当前我们自己已经核验的正式高分基线仍为 **V30 / Public LB 0.654**；本次收集的其他分数不加入自己的实验成绩。

## 7. 可复查材料

- [结构化二十项登记表]({prefix}/catalog.json)：分数、时间、下载版本、哈希、思路和限制。
- [Kaggle 网页分数快照]({prefix}/score-page-ax.txt)：包含页面实际显示的 Score。
- [数字分数提取]({prefix}/ui-scores.json)：保留页面顺序与来源 URL。
- [原始候选顺序]({prefix}/search-order.json)：MCP 返回 40 项，本次收录与网页吻合的前 20 项集合。
- [20 份源码索引]({prefix}/sources-index.json)：包含完整元数据、源码哈希与代码哈希；每个目录保留 `source.ipynb`、`code.txt`、`description.md`、`metadata.json` 和 `receipt.json`。
- [原实验总记录](公开方案与实验记录.md)：我们的实际执行异常、资源记录和正式结果。

当前没有检查这 20 项所有附件的实时可下载性，也没有运行它们。某个页面显示高分、某份代码可下载，以及今天能够重现那个分数，是三个不同的结论。
'''
out = PROJECT / '公开高分方案Top20-2026-10-29.md'
out.write_text(text, encoding='utf-8')
(ROOT / 'catalog.json').write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'report': str(out), 'entries': len(registry), 'characters': len(text), 'score_range': [rows[-1]['score'], rows[0]['score']]}, ensure_ascii=False))
