"""Build the 2026-10-30 update from saved, observed Kaggle evidence."""
import collections
import csv
import datetime as dt
import hashlib
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
REPORT = PROJECT / '公开高分方案Top20-2026-10-29.md'
prefix = 'reviews/public-top20-20261030/'
scores = json.loads((ROOT / 'ui-scores.json').read_text(encoding='utf-8'))
index = json.loads((ROOT / 'sources-index.json').read_text(encoding='utf-8'))
byref = {x['ref']: x for x in index}
observed = dt.datetime.fromisoformat(scores['observed_at'].replace('Z', '+00:00'))
local_time = observed.astimezone(dt.timezone(dt.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')

notes = {
'omerfarukmerey/0-663-0-660-v31-atlas-20': 'ATLAS 20%；Llama 本地暂存、双 T4 并行 8B、GPU 池化；仍是本次有效分数最高项。',
'alexchilton/name-that-llm-0-662-all-public-what-failed': 'ATLAS20 的全公开依赖复现；Llama 换为公开模型输入，并整理同代码分数波动和失败实验。',
'lightningv08/0-662-qwen3-8-27b-lora-weighted-ensemble': '新增 Qwen3.8-27B GSQ3 LoRA，最新源码概率份额 7.5%，ATLAS 20%；预计算 CPU 状态与进程隔离。',
'roserrll/name-that-llm-0663-v31-atlas20-alex': '相对 ATLAS20 主要增加 Llama 两种来源的解析与检查；属于依赖修复复现。',
'huesosinapoganaya/0-663-0-660-v31-atlas-20': 'ATLAS20 副本；更换 Llama 路径并保留分片大小检查；标题 0.663，平台实际 0.662。',
'youdian1s/ntl-0-662-all-public-fork': '沿用 alexchilton 全公开版；缺失路径解析改为返回 None，说明用于未使用的指令模型变量。',
'ye070301/ntl-0663-v31-atlas20': 'ATLAS20 加公开 Llama 路径、提交格式检查、隐藏重跑的集成输出核验。',
'negi69/ntl-0-663-v31-atlas20-repro': '最新源码增加显存上限、分块 LoRA 合并及真实样例预检；移除两个 8B 的 dp 标志。',
'junaid512/ntl-v1-baseline': '全部代码单元与本轮 alexchilton V8 完全相同；作为独立公开复现记录收录。',
'dfenglei/ntl-repro-0662': '全部代码单元与本轮 alexchilton V8 完全相同；页面得分 0.662。',
'otmaneelmahdaouy/ntl-v40-summit-pinnacle': 'ATLAS20 主体；Llama 来源回退与 FAST_COMMIT 跳过复制；正文写 0.670+，平台实际 0.662。',
'dosssi4nzk/muse-guard-ntl-youdian1s-0-662-api-copy': '全部代码单元与 youdian1s all-public fork 完全相同。',
'pablodiegoo/name-that-llm-starter-notebook': '完整多骨干 ATLAS20 集成，加 V40A 家族风格残差与分组验证展示；并非官方轻量 Starter。',
'lightningv08/0-661-qwen3-8-27b-lora-spectral-ensemble': '27B GSQ LoRA / Spectral 分支，概率份额 10%；与第 03 项 7.5% 分支分别保留。',
'pikaaa/name-that-llm-v20-pool-centred': '合并额外 open pool，增加 prompt 中心化 SVC 成员 svc4c，ATLAS 20%。',
'romantamrazov/name-that-llm-who-am-i': '最新保存版 33，正文为 V40A：按 prompt 分组 OOF 控制家族内风格残差。',
'mercurydust/name-that-llm-submission': 'V31 共同基础：来源加权文本分类器、家族门控、Soft Sinkhorn 与 Bayesian typography。',
'ozertuu/public-lb-0-66': '全部代码单元与 mercurydust 当前 V3 相同；输入附件仍应单独核验。',
'otmaneelmahdaouy/ntl-v39-summit-sovereign': '把公共池加入 TF-IDF/文本变形训练，并把家族分类器 C 调为 0.10；正文 0.670+ 未实现。',
'youdian1s/ntl-tuned-blend-calib-refit': '在 all-public fork 上给 Llama 增加 Gaussian 权重 0.7，ATLAS 调至 0.30；平台 0.660。',
}

eligible, excluded = [], []
for row in scores['rows']:
    ref = row['url'].split('/code/', 1)[1]
    score = float(re.search(r'Score:\s*([0-9.]+)', row['text']).group(1))
    if ref == 'martynaplomecka/starter-notebook-for-name-that-llm-competition' or not 0 <= score <= 1:
        excluded.append({'ref': ref, 'display_score': score, 'evidence_text': row['text']})
        continue
    eligible.append({'ref': ref, 'score': score, 'url': row['url'], 'page_order': len(eligible) + 1})
eligible.sort(key=lambda x: (-x['score'], x['page_order']))
catalog = []
for rank, row in enumerate(eligible[:20], 1):
    entry = {**byref[row['ref']], **row, 'rank': rank, 'observed_at': scores['observed_at'], 'note': notes[row['ref']],
             'score_is_bound_to_downloaded_version': False}
    assert entry['private'] is False
    catalog.append(entry)
    receipt = {**entry, 'source_reader': 'Kaggle MCP get_notebook_info',
               'score_reader': 'Kaggle Code page rendered Score via browser DOM',
               'retrieved_by_refresh_date': '2026-10-30', 'downloaded_source_executed': False}
    (ROOT / entry['directory'] / 'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT / 'catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT / 'excluded.json').write_text(json.dumps(excluded,ensure_ascii=False,indent=2),encoding='utf-8')
assert len(catalog) == 20 and len({x['ref'] for x in catalog}) == 20
counts = collections.Counter(x['score'] for x in catalog)
assert counts == {.663:1,.662:11,.661:3,.660:5}, counts

with (ROOT / 'own-submissions.csv').open(encoding='utf-8-sig', newline='') as f:
    own = list(csv.DictReader(f))
submission = next(x for x in own if x['ref']=='57778357')
assert submission['status']=='SubmissionStatus.COMPLETE' and float(submission['publicScore'])==.662
own_evidence = {'submission_id':57778357,'status':'COMPLETE','public_score':.662,'submission_date_utc':submission['date'],
                'submission_date_beijing':'2026-10-30 01:36:23','filename':submission['fileName'],
                'method_label':'ATLAS20 reproduction, identified by user; score independently verified by official CLI',
                'kernel_version_binding_verified_this_refresh':False}
(ROOT/'own-result.json').write_text(json.dumps(own_evidence,ensure_ascii=False,indent=2),encoding='utf-8')

groups=collections.defaultdict(list)
for row in catalog: groups[row['code_sha256']].append(row['rank'])
duplicates=[v for v in groups.values() if len(v)>1]
assert sorted(duplicates)==[[2,9,10],[6,12],[17,18]],duplicates

lines = [
'# Name That LLM：公开高分 Notebook Top 20 思路收集', '',
f'最新分数核查：**{local_time} 北京时间**（UTC：`{scores["observed_at"]}`）。在用户指定的原文件中更新，文件名沿用 2026-10-29；昨天的详细分析保留在文末历史附录。', '',
'## 1. 本次更新结论与我们的实测基线', '',
'- **你的 ATLAS20 复现已取得正式 Public Score 0.662。** 官方 CLI 返回提交 `57778357` 为 `COMPLETE`，提交时间为北京时间 **2026-10-30 01:36:23**，文件为 `submission.parquet`。这是正式比赛结果；本轮未重新核对该提交对应的 Notebook 历史版本。',
'- 本次可用公开高分榜最高仍为 **0.663**，Top20 中有 **11 项为 0.662**。你的成绩与这些条目同分，距公开最高显示值差 **0.001（0.1 个百分点）**；这个差值的原因尚未定位。',
'- Top20 截止分数从昨天的 **0.651** 上升到 **0.660**，且 0.660 还有并列未收录项。排序按网页实际 Score；标题、README 的 0.663/0.670+ 不用来代替实测分数。',
'- 有多组代码完全相同的副本。20 个 Notebook 不代表 20 种独立算法；值得优先阅读的是公开依赖修复、27B 加权成员、扩展公共池和 prompt 中心化 SVC 的差异。', '',
f'依据：[自己的正式提交记录]({prefix}own-submissions.csv) · [本次复现记录]({prefix}own-result.json) · [网页分数快照]({prefix}ui-scores.json)。', '',
'| 自有实验记录 | 正式 Public Score | 提交编号 | 状态 |',
'|---|---:|---:|---|',
'| 无 Llama 的 TF-IDF / LinearSVC 基线 | 0.308 | 57736953 | COMPLETE |',
'| Who am I V30 复现 | 0.654 | 57759707 | COMPLETE |',
'| 本次 ATLAS20 复现 | **0.662** | **57778357** | **COMPLETE** |', '',
'自有基线由 0.654 到 0.662 的观测差为 **+0.008（+0.8 个百分点）**；同时涉及方案与执行差异，不能归因为某一个参数。自己的私有 Notebook 不插入下面的公开榜。', '',
'## 2. 2026-10-30 公开高分 Top20', '',
'口径沿用原文：Name That LLM 的公开 Notebook，按平台实际显示的 Public Score 降序排列；同分按当前页面顺序取满 20 个，不再按算法去重。**“下载版本”只是本次保存的最新公开源码版本，不保证就是产生该页面分数的历史版本。**', '',
'| 序号 | Notebook / 作者账号 | 平台 Public Score | 下载版本 | 核心增量或定位 |',
'|---|---|---:|---:|---|'
]
for row in catalog:
    title=row['title'].replace('|','\\|')
    folder=prefix+row['directory']
    lines.append(f"| {row['rank']:02d} | [{title}]({row['url']}) · `{row['ref'].split('/')[0]}` | **{row['score']:.3f}** | [{row['version']}]({folder}/source.ipynb) | {row['note']} |")
lines += ['',
'第 20 项之后，`youdian1s/ntl-atlas-0-45-probe` 同为 **0.660**，本次按页面顺序不占用 Top20 名额；源码也已保存供比较。它同时保留 Llama Gaussian 0.7、把 ATLAS 调到 0.45，不能把其分数单独归因于 ATLAS 权重。', '',
'## 3. 新增条目中值得研究的差异', '',
'### 3.1 ATLAS20 的 0.662 公开复现群', '',
'第 02 项 alexchilton 当前保存版从昨天的 V7 / 0.660 更新为 V8 / 0.662，页面 slug 也由 `name-that-llm-0-660-full-ensemble-what-failed` 变为 `name-that-llm-0-662-all-public-what-failed`。作者明确记录：运行 omerfarukmerey 的 0.663 配方，在自己的账号得到 0.662。Llama 从不可读的旧 cache 换成公开同权重来源。', '',
'第 04/05 项主要修复 Llama 路径；第 07 项增加提交格式与集成输出检查。第 08 项最新源码更进一步：真实样例预检、显存预算、分块合并和取消 8B 双副本；因此不能把它当作只改文件路径的副本。当前没有逐条复跑这些方案，也没有证明 0.001 差异来自随机性、数值精度、输入版本还是 ATLAS 覆盖率。', '',
'### 3.2 27B 加权分支：0.662', '',
'第 03 项当前 V33 源码给自训 Qwen3.8-27B GSQ3 LoRA **7.5%** 的概率份额，在 ATLAS 混合后、类别平衡前加入；ATLAS 为 **20%**，保留长文本和分进程/CPU 缓存优化。第 14 项 Spectral 版本仍为 **10%** 自有模型份额、页面分数 **0.661**。这是有新增模型成本的分支，分数比较不是控制变量实验，也不证明 27B 必然比我们现有配方更好。', '',
'### 3.3 公共池与 prompt 中心化 SVC：0.661', '',
'第 15 项 pikaaa 把额外 open pool 合并进已有公共池，按模型限制规模；增加 **svc4c**，在官方及公共池的 prompt 中心化表示上拟合 SVC，并采用 ATLAS 20%。这是新的特征/训练数据组合，不只是改一个输出融合权重。README 仍沿用较早的 0.660 描述，本文采用网页实际 **0.661**。', '',
'### 3.4 风格残差与调权，并非所有修改都提高得分', '',
'第 13 项名为 Starter，但实际包含完整多骨干集成和 V40A 家族风格残差，平台为 **0.661**；其说明中的 0.662～0.663、线下 nested CV 数字不当作当前页面成绩。第 16 项 Who am I V33 的家族内风格残差思路继续保留，页面为 **0.660**。', '',
'第 19 项 v39 把公共池扩展至 TF-IDF/文本变形训练，并将若干家族读出的正则参数 C 调到 0.10；第 11 项 v40 的主体代码与 ATLAS20 更接近，主要可见差异是 Llama 解析与复制分支。两页正文的 **0.670+ 是标题/宣传值**，实际分别为 0.660、0.662。', '',
'第 20 项 tuned blend 在 all-public 基础上同时加入 Llama Gaussian 0.7 并把 ATLAS 改为 0.30，页面为 **0.660**；正文继承的 0.662 不属于这次调权结果。实验应保留逐成员预测，再固定数据/版本做单项比较。', '',
'## 4. 代码重复、异常值与收录边界', '',
'### 4.1 已核对的代码完全相同分组', '',
'| 当前序号 | 代码单元 SHA-256 比较 | 含义 |',
'|---|---|---|',
'| 02、09、10 | 完全相同 | alexchilton V8、junaid512、dfenglei 属于同代码复现 |',
'| 06、12 | 完全相同 | youdian1s all-public 与 Muse Guard API Copy 同代码 |',
'| 17、18 | 完全相同 | mercurydust V3 与 ozertuu Public LB 0.66 同代码 |', '',
'比较的是按顺序拼接的全部代码单元，未把 Markdown、执行输出计入。相同代码不证明挂载版本、环境、被 ATLAS 实际处理的行和得分版本都相同。', '',
'### 4.2 排除的列表项', '',
'| Notebook | 页面显示 Score | 处理与原因 |',
'|---|---:|---|',
'| [He-Who-Must-Not-Be-Named](https://www.kaggle.com/code/leoprovorov/he-who-must-not-be-named) | 1645.5 | 超出本比赛 Balanced Accuracy 的 0～1 范围。最新源码确实包含 Name That LLM 分类流程，但这个显示值尚无法绑定到本比赛有效成绩，单独存档，不冒充最高分。 |',
'| [177/180 Fresh Top-30 · v21.1 Conditional Memory](https://www.kaggle.com/code/ozertuu/177-180-fresh-top-30-v21-1-conditional-memory) | 1019.8 | 同样超出指标范围；源码开头是 route / market 的对局胜率分析，不能当作本比赛 1019.8 分方案。 |',
'| [官方 Starter](https://www.kaggle.com/code/martynaplomecka/starter-notebook-for-name-that-llm-competition) | 0.322 | 页面置顶条目，按实际分数不属于 Top20；不要与第 13 项同名近似的私人 Starter 混淆。 |', '',
'这两项异常值为何出现在按分数排序的比赛 Code 页面，尚未查明。本文保留原始列表与源码，不推断是作者作弊或平台的某个具体故障。', '',
'### 4.3 证据限制', '',
'- 网页使用 `excludeNonAccessedDatasources=true`；本文覆盖该可访问公开列表，不是比赛总排行榜前 20 名，也不保证包含所有被筛掉的 Notebook。',
'- 页面分数与当前下载源码没有逐项绑定到同一历史版本。标题分数、README 声称分数、平台 Score、自有正式复现分数分别记录。',
'- 本次只读取资料和比对源码，未执行这 20 份模型、未检查每个附件实时可下载性、未增加任何比赛提交。',
'- 自有 ATLAS20 0.662 已由官方提交记录核实；其与原作者 0.663 的差异原因仍待专门诊断。', '',
'## 5. 本次可复查材料', '',
f'- [Top20 结构化登记表]({prefix}catalog.json)：网页分数、排序、版本和思路。',
f'- [网页实际分数]({prefix}ui-scores.json) · [完整页面文本]({prefix}score-page-ax.txt) · [MCP 原始 40 项候选]({prefix}search-order.json)。',
f'- [源码/元数据/哈希索引]({prefix}sources-index.json)：保存 20 项正式条目、2 项异常候选、2 项边界比较，共 24 份；各目录另有代码与说明，部分有差异文件。',
f'- [排除记录]({prefix}excluded.json) · [自有正式提交]({prefix}own-submissions.csv) · [更新前原文备份]({prefix}report-before-update.md)。',
'- [Kaggle Public Score 列表](https://www.kaggle.com/competitions/name-that-llm/code?competitionId=170484&sortBy=scoreDescending&excludeNonAccessedDatasources=true)。', '',
'---', '',
'## 附录：2026-10-29 历史 Top20 与详细思路', '',
'以下保留上一轮核查内容；**序号、分数、版本和“本次/当前”等用语均对应 2026-10-29 的历史截面**。今天的排名和自有基线以本文前半部分为准。', ''
]
old=(ROOT/'report-before-update.md').read_text(encoding='utf-8-sig')
old=old.split('\n',1)[1].lstrip()
old=old.replace('以上是研究顺序，不是自动执行计划。当前我们自己已经核验的正式高分基线仍为 **V30 / Public LB 0.654**；本次收集的其他分数不加入自己的实验成绩。',
                '以上是上一轮的研究顺序，不是自动执行计划。**截至 2026-10-29 该轮核查**，自有正式高分基线为 V30 / Public LB 0.654；**2026-10-30 已核实 ATLAS20 / Public LB 0.662，见本文前半部分。** 公开作者的其他分数仍不计入自己的实验成绩。')
old=re.sub(r'(?m)^(#{2,5}) ',r'#\1 ',old)
REPORT.write_text('\n'.join(lines)+'\n'+old,encoding='utf-8')

# Verify rank evidence, preserved history and every relative Markdown link.
text=REPORT.read_text(encoding='utf-8')
missing=[]
for link in re.findall(r'\]\(([^)]+)\)',text):
    if '://' not in link and not link.startswith('#') and not (PROJECT/link.split('#',1)[0]).exists():missing.append(link)
assert not missing,missing
assert all(catalog[i]['score']>=catalog[i+1]['score'] for i in range(19))
assert not re.search(r'(?m)^### 01\. 0\.663', text)  # old detail headings are inside the history appendix
assert '#### 20. Name That LLM' in text
validation={'status':'PASS','ranked_rows':20,'sources_saved':len(index),'score_distribution':{f'{k:.3f}':v for k,v in counts.items()},
            'same_code_groups':duplicates,'own_score':.662,'own_submission_status':'COMPLETE','missing_relative_links':missing,
            'report_sha256':hashlib.sha256(REPORT.read_bytes()).hexdigest(),'history_preserved':True,'observed_at':scores['observed_at']}
(ROOT/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(validation,ensure_ascii=False,indent=2))
