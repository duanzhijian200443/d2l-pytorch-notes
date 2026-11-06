"""Inspect local downloaded EXP054 public fast-commit output and issue transfer receipt."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
EXP=ROOT/'experiments/EXP-054-apex-zero-formal'
REF='zhjianduan/ntl-apex-zero-054'
SOURCE=next((ROOT/'research/solutions').glob('SRC-067-*'))
RAW_SHA='ca4c9e18ed32dcd8168b8118b192d39d657803d30dcd89a564b1c2d1d9c846dd'

def need(ok,why):
    if not ok:raise ValueError(why)

def read(p):return json.loads(p.read_text(encoding='utf8'))
def entry(p):return {'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def write(p,data):
    with p.open('x',encoding='utf8') as f:f.write(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

def main():
    out=HERE/'outputs-v2';submission=out/'submission.parquet';test=pd.read_csv(ROOT/'test.csv',dtype={'id':str},keep_default_na=False)
    result=pd.read_parquet(submission)
    need(list(result.columns)==['id','model'] and len(result)==len(test)==3 and result.id.is_unique,'Unexpected public output schema')
    need(result.id.astype(str).tolist()==test.id.tolist() and result.model.tolist()==['gpt-5.5']*3,'Public placeholder IDs/labels differ')
    log=out/'ntl-apex-zero-054.log'
    log_text=log.read_text(encoding='utf8')
    signals=['FAST_COMMIT True','llama source alexchilton/llama-3-1-8b-base','variant E dropped flag_gemini37_routing_risk 0','open safe pool 190588']
    for signal in signals:need(signal in log_text,'Missing fixed commit signal: '+signal)
    fixed_modules=read(SOURCE/'commit-output-comparison.json')['results'][0]['materialized_module_hashes']
    modules=[]
    for name,expected in fixed_modules.items():
        item=entry(out/name)
        need(item['sha256']==expected['sha256'] and item['bytes']==expected['size'],'CPU/GPU-generated module differs: '+name)
        modules.append(item)
    binding=read(HERE/'script-identity.json');back=read(HERE/'readback-v2.json');ack=read(HERE/'v2-ack.json')
    need(binding['ref']==back['ref']==REF and binding['version']==back['version']==2 and binding['script_version_id']>0,'Fixed-version binding differs')
    need(back['private'] and back['internet'] is False and back['asset_count']==31 and back['source_bytes_sha256']==RAW_SHA,'Fixed metadata digest mismatch')
    need(ack['response']['versionNumber']==2,'Wrong run acknowledgment')
    now=datetime.now(timezone.utc).isoformat()
    evidence={'experiment_id':'EXP-054','status':'PASS_PUBLIC_FAST_COMMIT','verified_at':now,'ref':REF,'version':2,'script_version_id':binding['script_version_id'],
              'submission_artifact':entry(submission),'module_hashes':modules,'public_ids_exact_order':test.id.tolist(),'public_output_models':result.model.tolist(),
              'public_log':entry(log),'verified_log_signals':signals,'source':entry(HERE/'kernel/apex-zero.ipynb'),
              'source_exact_readback':entry(HERE/'readback-v2.json'),'global_id_evidence':entry(HERE/'script-identity.json'),
              'competition_submission_made_by_this_controller':False,'hidden_result_observed':False}
    write(EXP/'metrics/commit-verification.json',evidence)
    receipt={'status':'READY_FOR_SUBMISSION','ref':REF,'version':2,'script_version_id':binding['script_version_id'],
             'source_sha256':RAW_SHA,'submission_sha256':entry(submission)['sha256'],'private':True,'internet':False,'asset_count':31,'verified_at':now,
             'evidence_paths':[entry(EXP/'metrics/commit-verification.json')['path'],entry(HERE/'readback-v2.json')['path'],entry(HERE/'script-identity.json')['path'],entry(submission)['path'],entry(log)['path']],
             'attachment_version_limit':back['limit'],'hidden_run_not_yet_accepted':True,'competition_submission_owner':'root'}
    write(HERE/'ready-for-submission.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False))

if __name__=='__main__':main()
