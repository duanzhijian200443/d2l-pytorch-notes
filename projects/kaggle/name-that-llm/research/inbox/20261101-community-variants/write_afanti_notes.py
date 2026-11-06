from pathlib import Path
import json
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2];p=BASE/'manifest.json';m=json.loads(p.read_text(encoding='utf-8'))
configs={
'ntl-fork-664':('0.60','0.35',0.659,'与SRC-022 judith V6代码完全一致，没有新增算法。'),
'ntl-fork664-m25':('0.60','0.25',None,'相对SRC-022只将sinkhorn_target_mix从0.35改成0.25。'),
'ntl-fork664-s50':('0.50','0.35',None,'相对SRC-022只将sinkhorn从0.60改成0.50。'),
'ntl-fork664-s70':('0.70','0.35',0.658,'相对SRC-022只将sinkhorn从0.60改成0.70。')}
for r in m['entries']:
 if not r['ref'].startswith('afantihust/'):continue
 slug=r['ref'].split('/')[1];strength,mix,score,change=configs[slug];folder=ROOT/r['artifact_dir']
 r['public_score']={'value':score,'evidence_type':'platform_page' if score is not None else 'not_displayed','bound_to_source':False,'script_version_id':None}
 r['page_observation_received_from_merge_window']={'observed_at':'2026-11-01T07:49:00+00:00','scope':'Kaggle competition Code search 0.664','score_displayed':score,'notebook_version_binding_verified':False}
 r['tags']=['postprocessing','parameter-sweep','version-binding','same-family']
 r['summary']=change+('页面实际'+str(score)+'，不是标题0.664。' if score is not None else '页面未显示分数，不能按标题记0.664。')
 r['author_claim']='正文称fork自0.664上游且代码未改；该0.664属于上游得分V2，无法继承到本fork。'
 r['verified']=f'固定V1已重取匹配源码SHA；Sinkhorn={strength}、target_mix={mix}、ATLAS=.20、标点=1.2。'+change
 r['use_for_us']='作为同源后处理参数扫描与标题/得分错绑反例。'
 r['test_plan']='固定预测、输入与标点参数，单独比较Sinkhorn或target_mix；先绑定各页得分版本，不用标题推断。'
 r['limitations']='页面分数未绑定V1，未运行。未显示分数不等于0、失败或等待状态；依赖输入版本未逐一审计。'
 claim_check='其“代码未改”声明与本次V1对比相符，但相同的是上游V6而非得分V2。' if slug=='ntl-fork-664' else '正文仍称“代码未改”，但静态差分显示上述参数修改；本条以代码为准，保留声明与事实的区别。'
 score_text=f'主窗口于2026-11-01T07:49Z读取搜索列表时，本页实际显示 **{score}**；并非标题所暗示的0.664。该页面分数仍未绑定本V1。' if score is not None else '主窗口于2026-11-01T07:49Z读取搜索列表时，本页没有Score字段。这里只记录“未显示”，不能填0、失败或推断尚在运行，更不能填标题里的0.664。'
 text=f'''# 待合并 · {r['title']} · V1

状态：已读固定源码与正文；未运行；正式SRC待合并窗口分配。

## 值得保留的结论

{change} 它属于judith V6（SRC-022）的参数家族，而不是得分V2（SRC-021）原样复现。标题含“664”不能作为成绩证据。

## 来源与原件

- 作者：{r['author']}（afantihust）。[来源页面](https://www.kaggle.com/code/{r['ref']})。
- 源码采集：`{r['observed_at']}`，V1；以v1重新请求后文件SHA一致。全局scriptVersionId未取得。
- [原始源码](source.ipynb) · [原正文](description.md) · [API元数据](metadata.json) · [下载凭据](receipt.json) · [与SRC-022逐行差分](vs-SRC-022.diff)。
- 代码SHA256：`{r['code_sha256']}`。

## 作者声称与实际设置

正文称fork自judith的0.664公开方案，并说代码未改。{claim_check}

实际运行配置：Sinkhorn **{strength}**、target_mix **{mix}**、ATLAS **20%**、标点证据系数 **1.2**。得分上游V2配置是0.30/0.35/20%/1.5，因此当前fork不能直接继承其0.664。

## 我们核查了什么

{score_text}

全部代码单元经静态对比，确认{change} 没有执行Notebook；没有把提交预览或页面题名当作完整评分成功。来源页面分数观察由主合并窗口保留总快照。

## 对当前方案的用途与限制

可作为IDEA-002的参数扫描线索，以及“下载最新版本与读取最佳成绩不等价”的具体案例。单页分数差既未绑定固定输入，也未控制运行差异，不能据此宣称某参数更优。未验证所有附件版本和完整运行。

## 待验证想法

固定相同预测缓存和标点系数，先单独扫Sinkhorn，再扫target_mix；补齐原始提交/源码版本绑定后才讨论社区LB对比。当前只整理资料，未新增训练。

## 更新记录

- 2026-11-01：首次收集V1，与SRC-022/SRC-021关联。待主窗口分配正式SRC、合并页面快照并校验。
'''
 (folder/'notes.md').write_text(text,encoding='utf-8')
 r['notes_path']=(folder/'notes.md').relative_to(ROOT).as_posix()
 r['evidence']=[f.relative_to(ROOT).as_posix() for f in folder.iterdir() if f.name!='code.txt']
m['validation'].update(source_count=len(m['entries']),each_entry_has_notes=all((ROOT/r['notes_path']).exists() for r in m['entries']),fixed_version_refetch_hash_match=9,existing_source_updates=3)
p.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(m['validation']))
