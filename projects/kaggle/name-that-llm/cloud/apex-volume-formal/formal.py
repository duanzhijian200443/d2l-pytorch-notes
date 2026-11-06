"""Formal submission and read-only comparison for the two authorized Apex arms."""
import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlsplit, parse_qs

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from workbench_core import atomic_json, read, require, sha, now, CONFIG
from experiment_workbench import child_env
from workbench_remote import api_client

ARMS={56:(120,'a4e86314a77f7ed699dbfb107b32a8263a1ff09fc7c3ec92b98d4028d487005d'),57:(240,'694d91e2536db2d77d81f8d8b32a802763a750a2e9643fdb0ba6a10b76755a01')}
ZERO_REF='zhjianduan/ntl-apex-zero-054'
ZERO_SID=364820326
ZERO_ID=57903109
TARGET_DAY='2026-11-02'
OUT=ROOT/'cloud/apex-volume-formal'

def props(n):
 k,expected=ARMS[n];home=ROOT/f'cloud/apex-codex{k}-{n:03}';folder=ROOT/f'experiments/EXP-{n:03}-apex-four{k}-formal'
 return k,expected,home,folder,f'zhjianduan/ntl-apex-codex{k}-{n:03}'

def inventory(api):
 out=[]
 for page in range(1,6):
  rr=api.competition_submissions('name-that-llm',page_number=page,page_size=100)
  require(rr is not None,'UNAVAILABLE')
  for x in rr:
   d=x.to_dict();out.append(dict(ref=d['ref'],url=d['url'],date=str(d['date']),status=str(x.status).split('.')[-1],public_score=d.get('publicScore'),description=d.get('description')))
  if len(rr)<100:break
 else:raise ValueError('Incomplete inventory')
 require(len({r['ref'] for r in out})==len(out) and any(r['ref']==ZERO_ID and matches(r,ZERO_REF,ZERO_SID) for r in out),'WRONG_ACCOUNT_OR_BASELINE')
 return out

def matches(row,ref,sid):
 u=urlsplit(row['url'])
 return u.path=='/code/'+ref and parse_qs(u.query).get('scriptVersionId')==[str(sid)]

def score(row):
 return (Decimal(str(row['public_score'])) if row and row['status']=='COMPLETE' and row['public_score'] not in (None,'') else None)

def ready(n):
 k,expected,h,exp,ref=props(n);p=h/'ready-for-submission.json';r=read(p);cfg=read(exp/'config.json')
 require(r['status']=='READY_FOR_SUBMISSION' and r['ref']==ref and type(r['version'])==int and r['version']>0 and type(r['script_version_id'])==int and r['script_version_id']>0,'READY_TARGET')
 require(r['private'] is True and r['internet'] is False and r['asset_count']==31,'RECIPE_SETTINGS')
 src=h/'kernel/apex.ipynb'; require(sha(src)==r['source_sha256']==cfg['candidate_sha256'],'SOURCE_DRIFT')
 require(any(sha(x)==r['submission_sha256'] for x in h.rglob('submission.parquet')),'PLACEHOLDER')
 require(cfg['arm_size']==k and cfg['baseline_submission_id']==ZERO_ID,'INVALID_CONTROL')
 a=read(exp/'metrics/release.json');require(a['status']=='PASS' and a['source_sha256']==r['source_sha256'],'NO_INDEPENDENT_RELEASE')
 data=ROOT/f'experiments/EXP-052-stylegram-four-model-volume/data/four{k}.csv';require(sha(data)==expected,'DATA_DRIFT')
 return r

def preflight(n,rows):
 k,_,h,_,ref=props(n);r=ready(n)
 same=[x for x in rows if matches(x,ref,r['script_version_id'])]
 require(len(same)<=1,'MULTIPLE_MATCHES')
 if same:return r,same[0]
 require(not any(urlsplit(x['url']).path=='/code/'+ref for x in rows),'OTHER_VERSION_ALREADY_SUBMITTED')
 require(datetime.now(timezone.utc).date().isoformat()==TARGET_DAY,'DIFFERENT_QUOTA_DAY')
 require(sum(x['date'].startswith(TARGET_DAY) for x in rows)<5,'DAILY_LIMIT')
 require(not (h/'formal-submit-intent.json').exists() and not (h/'formal-submit-ack.json').exists(),'EXISTING_INTENT_RECONCILE')
 return r,None

def submit(api,n,rows):
 k,_,h,_,ref=props(n);r,existing=preflight(n,rows)
 if existing:
  print('EXISTING',existing['ref']);return
 desc=f'EXP{n:03} Apex V1; Stylegram only + four-model raw {k}; paired against 57903109; script {r["script_version_id"]}'
 atomic_json(h/'formal-submit-intent.json',dict(created_at=now(),ref=ref,version=r['version'],script_version_id=r['script_version_id'],ready_sha256=sha(h/'ready-for-submission.json'),description=desc,submission_ids_before=[x['ref'] for x in rows]),exclusive=True)
 response=api.competition_submit_code('submission.parquet',desc,'name-that-llm',kernel=ref,kernel_version=r['version'],quiet=True)
 require(isinstance(response.ref,int) and response.ref>0,'UNCONFIRMED_RESPONSE')
 atomic_json(h/'formal-submit-ack.json',dict(id=response.ref,source='official Kaggle SDK',observed_at=now()),exclusive=True)
 print('ACCEPTED',n,response.ref,flush=True)

def bind(n,rows):
 _,_,h,_,ref=props(n);r=ready(n)
 cand=[row for row in rows if matches(row,ref,r['script_version_id'])]
 require(len(cand)==1,'NO_UNIQUE_OFFICIAL_BINDING')
 x=cand[0];ack=h/'formal-submit-ack.json'
 if ack.exists():require(read(ack)['id']==x['ref'],'ACK_ID_MISMATCH')
 b=dict(id=x['ref'],script_version_id=r['script_version_id'],notebook_version=r['version'],url='https://www.kaggle.com'+x['url'],submitted_at=x['date'],observed_at=now(),status=x['status'],public_score=float(score(x)) if score(x) is not None else None)
 p=h/'formal-submit-binding.json'
 if p.exists():require(read(p)['id']==x['ref'] and read(p)['script_version_id']==r['script_version_id'],'PREVIOUS_BINDING_DRIFT')
 else:atomic_json(p,b,exclusive=True)
 print('BOUND',n,b['id'],b['status'],b['public_score'])

def compare(rows,save=True):
 arms={}
 for n,k in [(54,0),(56,120),(57,240)]:
  if n==54: ref,sid,id_=ZERO_REF,ZERO_SID,ZERO_ID
  else:
   _,_,home,_,ref=props(n);p=home/'formal-submit-binding.json'
   if not p.exists():arms[str(k)]={'status':'NO_LOCAL_BINDING','public_score':None};continue
   b=read(p);sid,id_=b['script_version_id'],b['id']
  hits=[r for r in rows if r['ref']==id_ and matches(r,ref,sid)]
  require(len(hits)==1,'REMOTE_BINDING_CHANGED')
  rr=hits[0];s=score(rr)
  arms[str(k)]=dict(id=id_,script_version_id=sid,status=rr['status'],public_score=str(s) if s is not None else None)
 def diff(l,r):
  a,b=arms[l]['public_score'],arms[r]['public_score']
  return str(Decimal(a)-Decimal(b)) if a is not None and b is not None else None
 data=dict(observed_at=now(),competition='name-that-llm',metric='displayed Public LB, not independent statistical evidence',arms=arms,score_differences={'120_minus_0':diff('120','0'),'240_minus_0':diff('240','0'),'240_minus_120':diff('240','120')},used_today=sum(x['date'].startswith(datetime.now(timezone.utc).date().isoformat()) for x in rows))
 if save:atomic_json(OUT/'latest-comparison.json',data)
 print(json.dumps(data,ensure_ascii=True))

def main():
 arg=argparse.ArgumentParser();arg.add_argument('op',choices=['inventory','submit','bind','compare']);arg.add_argument('arm',nargs='?',type=int,choices=ARMS)
 a=arg.parse_args();os.environ.update(child_env(read(CONFIG)));api=api_client();rows=inventory(api)
 if a.op=='inventory':print(json.dumps(rows,ensure_ascii=True))
 elif a.op=='compare':compare(rows)
 elif a.op=='submit':submit(api,a.arm,rows)
 else:bind(a.arm,rows)

if __name__=='__main__':main()
