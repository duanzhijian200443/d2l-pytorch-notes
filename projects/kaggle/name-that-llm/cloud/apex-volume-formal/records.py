"""Freeze the paired experiment evidence and register official scores for the fixed comparisons."""
import argparse
from pathlib import Path
import sys
import hashlib
from decimal import Decimal
from formal import ROOT, OUT, ARMS, ZERO_ID, ZERO_SID, props, inventory, compare
from workbench_core import atomic_json,read,sha,require,now,CONFIG
from experiment_workbench import child_env
from workbench_remote import api_client
import os

def path(p):return p.relative_to(ROOT).as_posix()
def info(p):return {'path':path(p),'sha256':sha(p),'bytes':p.stat().st_size}
def rec_path(n):return ROOT/f'tracking/records/experiments/EXP-{n:03}.json'

def plan(n):
 k,checksum,h,d,ref=props(n);f=ROOT/f'experiments/EXP-052-stylegram-four-model-volume/data/four{k}.csv';require(sha(f)==checksum,'RAW_ROWS_SHA')
 zero=read(ROOT/'cloud/apex-zero-054/formal-submit-binding.json');require(zero['id']==ZERO_ID and zero['script_version_id']==ZERO_SID,'ZERO_CONTROL')
 cp=d/'config.json'; rec=read(rec_path(n))
 require(not rec.get('submission') and 'candidate_sha256' not in read(cp),'ALREADY_FROZEN')
 cfg={'experiment_id':rec['id'],'baseline_experiment':'EXP-054','baseline_submission_id':ZERO_ID,'baseline_script_version_id':ZERO_SID,'mother_source_sha256':sha(ROOT/'cloud/apex-zero-054/kernel/apex-zero.ipynb'),'ref':ref,'arm_size':k,'labels':4,'rows_per_label':k//4,'complete_train_groups':k//4,'training_data':info(f),'training_protocol':'EXP052 frozen raw; append only inside Apex Stylegram; all pre-existing weights unchanged, new rows inherit existing class public row weight','excluded':'all 60 holdout rows; no validation/test rows used','unchanged':'EXP054 full recipe, non-Stylegram members and gates, Open Pool V13, main SVC w12, ATLAS .20, comp .15, Sinkhorn .60, punctuation 1.2, sibling residual','image':read(ROOT/'cloud/apex-zero-054/kernel/kernel-metadata.json')['docker_image'],'gpu':'NvidiaTeslaT4','private':True,'internet':False,'formal_submissions_max':1,'comparison':'own displayed Public Score vs actual EXP054, plus 240 minus 120'}
 atomic_json(cp,cfg);atomic_json(d/'manifest.json',dict(freeze_status='PREPARING_BEFORE_SOURCE_FREEZE',config=info(cp),inputs=[info(f)],experiment_id=rec['id']))
 rec['baseline_id']='EXP-054';rec['hypothesis']='在固定 Apex 0 条对照上，只改变 Stylegram 自采训练数据，比较正式 Public LB。';rec['changes']=[f'四模型各 {k//4} 条，合计 {k}；其他成员不读取新增数据。']
 rec['related_sources']=['SRC-067'];rec['inputs']['validation_split_id']='EXP052 raw nested four-model train groups, 60 holdout rows excluded'
 rec['execution'].update(scope='formal_submission_preparation',status='PREPARING',observed_at=now());rec['observed_at']=rec['execution']['observed_at']
 rec['next_action']='核验两个候选后，各提交一次；按本账户三份提交的最终分数计算差值。';rec['limitations']='完整公开保存只输出三行占位。正式成绩需自行绑定版本、提交 ID 与 COMPLETE 状态。'
 atomic_json(rec_path(n),rec)
 report=f"# {rec['id']} · Apex 四模型 {k} 条\n\n正式 0 条对照：[EXP-054 V2](https://www.kaggle.com/code/zhjianduan/ntl-apex-zero-054?scriptVersionId={ZERO_SID})，提交号 {ZERO_ID}。只在 Stylegram 加入冻结的 {k} 条原文，{k//4} 个训练原题组各含四类回答。对照采用同一账户的正式分数。\n\n训练范围与数据哈希见 [配置](config.json)，正式状态与证据见[本实验台账](../../tracking/records/experiments/{rec['id']}.json)。\n"
 (d/'report.md').write_text(report,encoding='utf8')

def release(n,audit):
 k,_,h,d,ref=props(n);cfg=read(d/'config.json');src=h/'kernel/apex.ipynb';meta=h/'kernel/kernel-metadata.json';sm=read(meta);zero=read(ROOT/'cloud/apex-zero-054/kernel/kernel-metadata.json')
 require(sm['id']==ref and sm['code_file']==src.name and all(sm[key]==v for key,v in zero.items() if key not in ('id','title','code_file')),'AUXILIARY_PARAMETER_DRIFT')
 require((ROOT/audit).is_file(),'INDEPENDENT_AUDIT_NOT_FOUND')
 review=read(ROOT/audit);arm=review['arms'][str(k)];require(review['status']=='PASS_LOCAL_CANDIDATE_PAIR' and arm['status']=='PASS' and arm['source_sha256']==sha(src) and arm['metadata_sha256']==sha(meta),'SOURCE_NOT_IN_INDEPENDENT_PASS')
 require(not (d/'metrics/release.json').exists(),'CONFIG_ALREADY_FROZEN')
 cfg['candidate_sha256']=sha(src);cfg['metadata_sha256']=sha(meta);cfg['frozen_at']=now();atomic_json(d/'config.json',cfg)
 gate={'status':'PASS','experiment':f'EXP-{n:03}','release_time':now(),'source_sha256':cfg['candidate_sha256'],'private':True,'internet':False,'audit':info(ROOT/audit),'meaning':'Independent source/row/weight and guard checks passed before cloud release, not yet hidden grading.'}
 atomic_json(d/'metrics/release.json',gate,exclusive=True)
 files=[src,meta,h.parent/'apex-zero-054/kernel/apex-zero.ipynb',ROOT/cfg['training_data']['path'],ROOT/audit,d/'metrics/release.json']
 atomic_json(d/'manifest.json',{'experiment_id':f'EXP-{n:03}','freeze_status':'FROZEN_BEFORE_CLOUD_EXECUTION','config':info(d/'config.json'),'mother_submission':ZERO_ID,'sources_and_data':[info(x) for x in files]})
 rec=read(rec_path(n));rec['source'].update(notebook_ref=ref,local_path=path(src),sha256=sha(src));rec['execution'].update(scope='commit',status='PREPARED',observed_at=now());rec['observed_at']=rec['execution']['observed_at'];rec['evidence']+=([audit,path(d/'metrics/release.json')]);atomic_json(rec_path(n),rec)

def refresh():
 os.environ.update(child_env(read(CONFIG)));rows=inventory(api_client());compare(rows);snapshot=read(OUT/'latest-comparison.json');obs=snapshot['observed_at']
 for n in [48,54,56,57]:
  record=read(rec_path(n));h=None
  if n==48:rid=57893384;sid=364733823
  elif n==54:rid=ZERO_ID;sid=ZERO_SID
  else:
   k,_,h,d,ref=props(n)
   if not (h/'formal-submit-binding.json').exists():continue
   b=read(h/'formal-submit-binding.json');rid,sid=b['id'],b['script_version_id']
  from urllib.parse import urlsplit,parse_qs
  hits=[r for r in rows if r['ref']==rid and parse_qs(urlsplit(r['url']).query).get('scriptVersionId')==[str(sid)]]
  require(len(hits)==1,'SCORE_BINDING');r=hits[0];v=float(r['public_score']) if r['status']=='COMPLETE' and r['public_score'] not in (None,'') else None
  metric=ROOT/Path(record['config_path']).parent/'metrics'/'formal-latest.json';atomic_json(metric,dict(submission_id=rid,script_version_id=sid,observed_at=obs,source='Kaggle official submissions list',**r))
  record['observed_at']=obs;sub=record.get('submission') or {}
  sub.update(id=rid,script_version_id=sid,status=r['status'],public_score=v,observed_at=obs,url='https://www.kaggle.com'+r['url'],submitted_at=r['date'])
  record['execution'].update(scope='competition_rerun',status=r['status'],observed_at=obs);record['submission']=sub
  record['evidence']=list(dict.fromkeys(record['evidence']+[path(metric)]))
  if h:
   rd=read(h/'ready-for-submission.json');record['source'].update(notebook_ref=rd['ref'],notebook_version=rd['version'],script_version_id=sid,local_path=path(h/'kernel/apex.ipynb'),sha256=rd['source_sha256'])
   record['execution']['receipt_path']=path(h/'formal-submit-binding.json')
   record['environment'].update(platform='Kaggle',private=True,internet=False,accelerator='T4 x2',container=read(h/'kernel/kernel-metadata.json')['docker_image'])
   sub.update(notebook_version=rd['version'],request_count=1)
   record['evidence']=list(dict.fromkeys(record['evidence']+[path(h/z) for z in ['ready-for-submission.json','formal-submit-intent.json','formal-submit-ack.json','formal-submit-binding.json'] if (h/z).exists()]))
  if n==48 and v is not None:
   record['conclusion']=f'正式 Public LB {v:.3f}，与不加 Luna/Terra 数据的 A1（EXP046，0.663）显示同分。';record['next_action']='这组使用两类数据，不代替 Apex 四模型实验的 0 条对照。';record['execution']['note']='正式评分完成；不从显示同分推断逐行预测相同。'
  else:
   record['conclusion']=f'正式提交 {rid}，状态 {r["status"]}。'+(f'Public LB {v:.3f}。' if v is not None else '暂无公开分数。')
   contrast=snapshot['score_differences'].get(str(k)+'_minus_0') if h else None
   if contrast is not None:record['conclusion']+=f' 与本账户 0 条对照的显示分差为 {Decimal(contrast):+.3f}。'
   record['next_action']='待 0/120/240 条各自完成，以绑定的实际分数比较。' if any(a.get('public_score') is None for a in snapshot['arms'].values()) else '三组正式成绩已取得；差值见 cloud/apex-volume-formal/comparison.md，不按单次微差推断稳定收益。'
  atomic_json(rec_path(n),record)
  if h:
   f=d/'report.md';report=f.read_text(encoding='utf8').split('\n## 正式结果')[0].rstrip()
   report+=f"\n\n## 正式结果\n\n[固定版本 V{rd['version']}]({sub['url']})，正式提交 **{rid}**。{obs}：**{r['status']}**，Public Score：{v if v is not None else '未返回'}。[三组实时记录](../../cloud/apex-volume-formal/latest-comparison.json)\n"
   f.write_text(report,encoding='utf8')
 lines=['# Apex：0、120、240 条自采数据的正式对照','',f'最近官方回查：{obs}。','',
        '| Stylegram 新增 | 本次提交 | Kaggle 状态 | Public Score | 相对本账户 0 条 |',
        '|---|---:|---|---:|---:|']
 for size in ['0','120','240']:
  a=snapshot['arms'][size];rid=a.get('id');ref=next((row['url'] for row in rows if row['ref']==rid),None)
  label=(f'[{rid}](https://www.kaggle.com{ref})' if ref else '尚未绑定')
  delta=snapshot['score_differences'].get(size+'_minus_0')
  gap=('对照' if size=='0' else (f'{Decimal(delta):+.3f}' if delta is not None else '待两者出分'))
  lines.append(f"| {size} 条 | {label} | {a['status']} | {a['public_score'] or '—'} | {gap} |")
 mid=snapshot['score_differences']['240_minus_120']
 lines+=['',f"240 条相对 120 条：{Decimal(mid):+.3f}。" if mid is not None else '240 条相对 120 条：待两组完成后计算。',
         '', '四类为 Luna、Terra、Sol、Astra。120 条为 30 个完整四答题组，240 条包含这些题组在内共 60 组。三组复用同一 Apex 配方，60 条留出样本未参与训练。', '',
         '差值基于同一账户、固定版本的正式 Public Score；作者的 0.665 不代替本账户零组实测，显示同分不能推断逐行预测完全一致。运行耗时会影响 ATLAS 剩余预算，隐藏预测覆盖未知。']
 (OUT/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','release','refresh']);p.add_argument('arm',nargs='?',type=int,choices=list(ARMS));p.add_argument('--audit');a=p.parse_args()
 if a.action=='plan':plan(a.arm)
 elif a.action=='release':release(a.arm,a.audit)
 else:refresh()
