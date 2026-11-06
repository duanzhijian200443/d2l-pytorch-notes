from pathlib import Path
from datetime import datetime,timezone
import json,os,hashlib
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
manifest=json.loads((BASE/'manifest.json').read_text(encoding='utf-8'))
configs={
'SRC-021':{
 'dest':'reviews/judith007-0664-20261031/v2-score-0664',
 'summary':'V2=363834272明确绑定0.664；相对alex V8仅将Sinkhorn .45改.30，保留标点1.5。',
 'related':['SRC-002','SRC-022'],
 'claim':'标题0.664；正文沿用上游ATLAS20说明。实时页面默认呈现得分V2，不代表API最新源码V6。',
 'verified':'原证据绑定V2/363834272/0.664；本轮静态重算代码hash匹配；主窗口2026-11-01T07:43Z再次看到Version 2 of 6、Best Score 0.664 V2。',
 'body':'''已登记得分版本仍是 **V2 / scriptVersionId 363834272 / 0.664**。2026-11-01 主窗口读取默认页面时，网页实际渲染的是 Version 2 of 6；API 最新源码返回 V6。默认URL没有版本参数，不足以认定读到的是最新代码。

相对 SRC-002 alexchilton V8 的代码变化仅为运行参数 `--sinkhorn 0.45` → `0.30`；标点证据系数仍为1.5，ATLAS份额仍20%。这里的单行差分不意味着榜单提升已经被因果验证；固定附件内容和运行轨迹尚未逐项审核。

页面 Runtime 9m16s（556.3秒）、10个输出文件对应公开样例快速提交路径，不是隐藏评分的完整耗时。不能把快速commit视作完整管线成功验收。''',
 'use':'作为IDEA-002的可定位后处理候选，固定概率后只比较平衡强度。',
 'test':'固定相同预测/类顺序/验证划分，仅比较0.30与0.45；报告稀疏prompt场景、错误分歧与重复运行差异。',
 'limits':'单次0.002差异不足以证明稳定提升；旧附件内容与隐藏全程运行未独立验收。'
},
'SRC-022':{
 'dest':'reviews/judith007-0664-20261031',
 'summary':'最新API仍为V6且代码不变：Sinkhorn .60、标点1.2；不能继承V2的0.664。',
 'related':['SRC-002','SRC-021'],
 'claim':'Notebook标题沿用0.664，不能用标题给最新V6登记成绩。',
 'verified':'本地旧V6代码hash匹配；主窗口本轮重新下载API最新V6也确认相同。网页默认选择V2而API最新为V6。',
 'body':'''V6 同时把 SRC-002 的 Sinkhorn 0.45 改成0.60、标点证据系数1.5改成1.2。V6与得分V2（SRC-021：0.30/1.5）是两种配置，应保留为两个来源条目。

主窗口2026-11-01重新拉取最新API仍是V6，代码hash与已存原件一致，因此复用SRC-022，不另造“11月1日版”。公开网页默认实际渲染Version2 of6，并显示0.664 V2；API最新版本与网页默认选中版本分开记录。

V6正式成绩仍未核验，本条public_score应保留未绑定/未知，不能填V2的0.664。本轮没有执行V6或更新自己的实验结果。''',
 'use':'保留为两参数组合候选，帮助避免默认页面版本与最新下载源码错绑。',
 'test':'固定缓存先分开验证Sinkhorn与标点系数，再评估组合，不能从一次两参数改变推导单项收益。',
 'limits':'V6没有已核验的版本成绩，附件内容和完整运行未验收。'
},
'SRC-002':{
 'dest':'reviews/public-top20-20261030/alexchilton--name-that-llm-0-662-all-public-what-failed',
 'summary':'全公开ATLAS20 V8为多份新页面的同代码上游；保持0.662页面分数未绑定V8。',
 'related':['SRC-001','SRC-009','SRC-010','SRC-021','SRC-022'],
 'claim':'作者报告自己ATLAS20运行0.662、若干失败消融与相同代码重复分数差；这些是作者自报。',
 'verified':'固定V8源码/正文可读且hash匹配；新junaid V3、sujan V12、dazhijiang V1、gloria V1代码均与本条完全相同。页面仍显示0.662，但未查得分版本。',
 'body':'''本条保存全公开依赖版本的 ATLAS20：Llama输入改为alexchilton公开模型；集成仍含complementary15%、ATLAS20%、Sinkhorn0.45和原标点步骤。新出现的多个公开Notebook延用这一代码，页面数量不代表方法数量。

作者正文报告将同代码重跑得到不同分数，也报告若干替换/增加成员与Hungarian失败尝试。这些表是作者的历史报告，未附完整逐次submission绑定，本轮不升级为独立重现实验。作者从若干观察推导的约0.01阈值不能当作通用统计噪声界限；应保留为对微小差值谨慎解释的提醒。

本轮确认新收集 junaid V3、sujan V12、dazhijiang V1、gloria V1 的全部代码单元SHA256均为d164e570c03550b08337e9934b5f18af1255aeacdf1e0a0d279547bfac693c6f，与本条相同。其文内复制的第一人称失败实验不应被重复计作四次独立社区验证。

网页列表仍显示0.662；未将该页面成绩绑定V8。公共样例快速提交路径不证明隐藏评分完整执行成功。''',
 'use':'作为当前ATLAS20同源基线比较和失败实验线索；复用原件避免重复归档同一固定版本。',
 'test':'固定输入版本、训练来源与预测缓存，以prompt分组和稀疏场景对照；新方法先验证互补性而非只看单成员OOF。',
 'limits':'平台0.662未绑定V8；作者自报未独立重训；输入依赖内容仍需版本审核。'
}}
updates=[]
for ident,c in configs.items():
 r=json.loads((ROOT/f'tracking/records/sources/{ident}.json').read_text(encoding='utf-8'))
 target=ROOT/c['dest']; stage=BASE/'updates'/ident;stage.mkdir(parents=True,exist_ok=True)
 links=[]
 for e in r['evidence']:
  rel=os.path.relpath(ROOT/e,target).replace('\\','/')
  links.append(f'[{Path(e).name}]({rel})')
 source=next(ROOT/e for e in r['evidence'] if e.endswith('.ipynb'))
 nb=json.loads(source.read_text(encoding='utf-8'));code='\n\n'.join(''.join(cell['source']) if isinstance(cell['source'],list) else cell['source'] for cell in nb['cells'] if cell['cell_type']=='code')
 assert hashlib.sha256(code.encode()).hexdigest()==r['source_version']['code_sha256']
 record_rel=os.path.relpath(ROOT/f'tracking/records/sources/{ident}.json',target).replace('\\','/')
 text=f'''# {ident} · {r['title']}

状态：已核查固定源码与部分页面证据；没有新运行。
原始记录：[{ident}.json]({record_rel})。

## 值得保留的结论

{c['summary']}

## 来源与原件

[来源页面]({r['url']})。复用已有原件，不为本次查询复制源码：

{' · '.join(links)}

固定版本 V{r['source_version']['notebook_version']}，代码 SHA256 `{r['source_version']['code_sha256']}`。本轮归档日期2026-11-01；页面再核查时点和完整证据由主合并窗口写入SRC JSON。

## 作者声称与实际设置

{c['claim']}

{c['body']}

## 我们核查了什么

{c['verified']} 本轮静态读取与比对，没有执行下载的Notebook。

## 对当前方案的用途与限制

{c['use']} {c['limits']}

## 待验证想法

{c['test']} 本次整理不登记新实验或收益。

## 更新记录

- 2026-11-01：复用{ident}与既有原件，补独立笔记、同源/版本边界。关联{', '.join(c['related'])}。正文链接按计划归档位置{c['dest']}/notes.md构造；临时包等待主窗口合并。
'''
 (stage/'notes.md').write_text(text,encoding='utf-8')
 updates.append({'existing_src':ident,'action':'UPDATE_EXISTING','staged_notes_path':(stage/'notes.md').relative_to(ROOT).as_posix(),'intended_notes_path':c['dest']+'/notes.md','proposed_update':{'category':'solutions','artifact_dir':c['dest'],'notes_path':c['dest']+'/notes.md','tags':['ensemble','version-binding','postprocessing'] if ident!='SRC-002' else ['ensemble','same-code','runtime','negative-results'],'related_sources':c['related'],'summary':c['summary'],'author_claim':c['claim'],'verified':c['verified'],'use_for_us':c['use'],'test_plan':c['test'],'limitations':c['limits']},'preserve_source_version_and_public_score':True,'source_hash_rechecked':True})
manifest['updates']=updates
(BASE/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print('3 existing-source update notes staged; existing source hashes verified.')
