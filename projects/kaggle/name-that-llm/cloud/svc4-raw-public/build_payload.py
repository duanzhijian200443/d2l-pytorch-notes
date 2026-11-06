"""Freeze local input/code/config snapshots and prepare one private CPU job."""
from datetime import datetime,timezone
import hashlib,json,re,sys,zipfile
from pathlib import Path
import pandas as pd
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
from svc4 import FourViewFeatures, REFERENCE_SOURCE_SHA256

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def resource_bound(train,public,folds):
    word=re.compile(r'(?u)\b\w+\b|[^\w\s]')
    def counts(texts):
        output=[]
        for text in texts:
            value=0
            for _,analyzer,prep,(lo,hi),cap,_ in FourViewFeatures.VIEWS:
                view=prep(text);length=len(view) if analyzer=='char' else len(word.findall(view))
                value+=min(cap,sum(max(0,length-n+1) for n in range(lo,hi+1)))
            output.append(value)
        return np.asarray(output,dtype=np.int64)
    official_bound=counts(train.text);public_bound=int(counts(public.text).sum())
    cases=[]
    for fold in range(5):
        tr=folds.fold.to_numpy()!=fold;valid=~tr
        for added in (False,True):
            rows=int(tr.sum())+(len(public) if added else 0)
            nnz=int(official_bound[tr].sum())+(public_bound if added else 0)
            valid_nnz=int(official_bound[valid].sum())
            # Loose bound for sparse values/indices, hstack copies, float64
            # liblinear copies and coefficient arrays; vocabulary strings are
            # covered by an additional 1 GiB allowance. Actual peak is guarded.
            csr32=nnz*8+(rows+1)*4
            csr64=nnz*12+(rows+1)*8
            estimate=2*csr32+2*csr64+valid_nnz*16+18*540000*8*3+1024**3
            cases.append({'fold':fold,'mode':'official-plus-raw' if added else 'official-only',
                          'rows':rows,'nnz_upper_bound':nnz,'csr32_upper_bytes':csr32,
                          'working_memory_estimate_bytes':estimate})
    limit=12*1024**3
    assert max(x['working_memory_estimate_bytes'] for x in cases)<limit, 'Capacity estimate above planned RSS limit'
    return {'status':'PASS_ESTIMATE','method':'token/character ngram count upper bound without vocabulary fitting or classifier training',
            'cases':cases,'rss_limit_bytes':limit,'note':'Not a hard bound on every library allocation; runtime RSS watchdog remains authoritative. No runtime prediction is claimed.'}

def main():
    state=HERE/'execution-state.json'
    if state.exists():
        assert read(state)['status'] not in ('RUNNING','QUEUED','LAUNCHING','COMPLETE'), 'Do not rebuild a live/completed run'
    now=datetime.now(timezone.utc).isoformat();registration=read(HERE/'registration.json');payload=HERE/'payload'
    audit=read(payload/'input_audit.json');assert audit['status']=='PASS_FOR_SCOPED_DEVELOPMENT'
    for source in audit['sources']:
        assert sha(ROOT/source['path'])==source['sha256'], source['path']
    assert sha(payload/'public.csv')==audit['outputs']['public_csv']['sha256']
    public=pd.read_csv(payload/'public.csv',dtype={'id':str,'prompt_id':str,'call_id':str})
    original=pd.read_csv(ROOT/'data/public-pools/prvsiyan/prvsiyans-extended-name-that-llm-dataset/v6/passages.csv',keep_default_na=False)
    raw=original.loc[original.split.eq('new')]
    assert public.id.tolist()==raw.id.tolist() and public.text.tolist()==raw.text.tolist()
    train=pd.read_csv(ROOT/'train.csv',dtype={'id':str,'prompt_id':str});folds=pd.read_csv(ROOT/'validation/prompt-group-v1/folds.csv',dtype={'id':str,'prompt_id':str})
    split=read(ROOT/'validation/prompt-group-v1/manifest.json')
    assert sha(ROOT/'train.csv')==split['train_sha256']
    assert sha(ROOT/'validation/prompt-group-v1/folds.csv')==split['folds_sha256']
    assert sha(ROOT/'experiments/EXP-009-baseline-data-lineage/infer.py')==REFERENCE_SOURCE_SHA256
    resource=resource_bound(train,public,folds)
    resource_path=ROOT/'experiments/EXP-014-svc4-raw-w035/metrics/resource-preflight.json';dump(resource_path,resource)
    copies={'svc4.py':ROOT/'src/svc4.py','runner.py':HERE/'runner.py','guard.py':HERE/'guard.py',
            'folds.csv':ROOT/'validation/prompt-group-v1/folds.csv'}
    for name,path in copies.items():(payload/name).write_bytes(path.read_bytes())
    variants=[];versions=None;image=None
    for row in registration['variants']:
        config_path=ROOT/row['directory']/'config.json';cfg=read(config_path)
        assert cfg['experiment_id']==row['id'] and cfg['model']['parameters']['public_row_weight']==row['public_row_weight']
        parameters=cfg['model']['parameters']
        for key,value in {'C':.5,'random_state':42,'max_iter':5000,'class_weight':'balanced','dual':'auto','feature_dtype':'float32'}.items():
            assert parameters[key]==value
        assert cfg['environment']['internet'] is False and cfg['environment']['accelerator'] is None
        if versions is None:versions=cfg['environment']['required_versions'];image=cfg['environment']['container']
        assert cfg['environment']['required_versions']==versions and cfg['environment']['container']==image
        config_name=f"config-{row['id']}.json";(payload/config_name).write_bytes(config_path.read_bytes())
        variants.append({**row,'config_file':config_name,'config_sha256':sha(config_path)})
    assert {v['public_row_weight'] for v in variants}=={None,.175,.35,.7}
    expected_files=set(copies)|{v['config_file'] for v in variants}|{'public.csv','input_audit.json','attribution.md'}
    assert {p.name for p in payload.iterdir() if p.is_file() and p.name!='protocol.json'}==expected_files
    hashes={name:sha(payload/name) for name in sorted(expected_files)}
    protocol={'schema_version':1,'created_at':now,'scope':'isolated svc4 raw-public development OOF, not original matched/pool recipe or full ATLAS20',
        'train_sha256':split['train_sha256'],'split_id':split['split_id'],'labels':split['labels'],
        'public_rows':5400,'public_prompts':300,'variants':variants,'required_versions':versions,
        'source_infer_sha256':REFERENCE_SOURCE_SHA256,'input_hashes':hashes,
        'comparisons':[['EXP-014','EXP-013'],['EXP-015','EXP-014'],['EXP-016','EXP-014']],
        'work_budget_seconds':6900,'platform_wall_seconds':7200,'rss_limit_bytes':12*1024**3,
        'gpu':False,'internet':False,'threads':2,'limits':audit['limitations']}
    dump(payload/'protocol.json',protocol)
    dataset=HERE/'dataset';kernel=HERE/'kernel';dataset.mkdir(exist_ok=True);kernel.mkdir(exist_ok=True)
    with zipfile.ZipFile(dataset/'svc4_payload.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(expected_files|{'protocol.json'}):archive.write(payload/name,name)
    dump(dataset/'dataset-metadata.json',{'id':'zhjianduan/ntl-svc4-raw-public-inputs-v1','title':'NTL SVC4 Raw Public CV Inputs v1',
        'licenses':[{'name':'other'}],
        'description':'Private CPU experiment payload. Public passage source: prvsiyan/prvsiyans-extended-name-that-llm-dataset V6, passages.csv raw split=new, 5400 unchanged texts, author CC BY 4.0 attribution preserved in attribution.md. No clean/funnel/punctuation filtering. Official texts loaded separately from competition input. Not a competition submission.'})
    bootstrap="""import hashlib,json,pathlib,runpy,sys,os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='2'
matches=[p for p in pathlib.Path('/kaggle/input').rglob('protocol.json') if hashlib.sha256(p.read_bytes()).hexdigest()==PROTOCOL_HASH]
assert len(matches)==1, 'Expected exactly one hash-pinned protocol'
target=matches[0].parent
protocol=json.loads(matches[0].read_text())
for name,digest in protocol['input_hashes'].items():
    assert hashlib.sha256((target/name).read_bytes()).hexdigest()==digest,name
sys.path.insert(0,str(target))
runpy.run_path(str(target/'guard.py'),run_name='__main__')
""".replace('PROTOCOL_HASH',repr(sha(payload/'protocol.json')))
    (kernel/'bootstrap.py').write_text(bootstrap,encoding='utf-8')
    dump(kernel/'kernel-metadata.json',{'id':'zhjianduan/name-that-llm-svc4-raw-public-cv','title':'Name That LLM SVC4 Raw Public CV',
        'code_file':'bootstrap.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':False,
        'enable_tpu':False,'enable_internet':False,'dataset_sources':['zhjianduan/ntl-svc4-raw-public-inputs-v1/1'],
        'competition_sources':['name-that-llm'],'kernel_sources':[],'model_sources':[],
        'docker_image_pinning_type':'original','docker_image':image})
    freeze={'status':'FROZEN','observed_at':now,'protocol_sha256':sha(payload/'protocol.json'),
            'payload_sha256':sha(dataset/'svc4_payload.zip'),'payload_bytes':(dataset/'svc4_payload.zip').stat().st_size,
            'source_files':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [ROOT/'src/svc4.py',HERE/'runner.py',HERE/'guard.py',HERE/'prepare_inputs.py',Path(__file__)]},
            'resource_preflight':resource,'runtime_training_started':False}
    dump(HERE/'preflight.json',freeze)
    for row in registration['variants']:
        exp=ROOT/row['directory'];manifest=read(exp/'manifest.json')
        manifest.update(freeze_status='FROZEN',frozen_at=now,note='Frozen for the shared CPU sweep; semantic/upstream data limits retained in input audit.',
            config={'path':row['directory']+'/config.json','sha256':sha(exp/'config.json')},
            code=[{'path':k,'sha256':v} for k,v in freeze['source_files'].items()],
            inputs=[{'path':'cloud/svc4-raw-public/payload/'+name,'sha256':value} for name,value in hashes.items()],
            protocol={'path':'cloud/svc4-raw-public/payload/protocol.json','sha256':freeze['protocol_sha256']})
        dump(exp/'manifest.json',manifest)
        record_path=ROOT/f"tracking/records/experiments/{row['id']}.json";record=read(record_path)
        record['observed_at']=now;record['execution'].update(scope='grouped_oof',status='PREPARED',observed_at=now)
        record['source'].update(notebook_ref='zhjianduan/name-that-llm-svc4-raw-public-cv',local_path='cloud/svc4-raw-public/runner.py',sha256=sha(HERE/'runner.py'))
        record['inputs']['content_hashes_verified']=True
        record['cost'].update(gpu_seconds=0,api_usd=0,colab_cu=0)
        record['evidence']+=['experiments/EXP-014-svc4-raw-w035/metrics/data-gate.json','experiments/EXP-014-svc4-raw-w035/metrics/resource-preflight.json','cloud/svc4-raw-public/preflight.json']
        record['evidence']=list(dict.fromkeys(record['evidence']))
        record['next_action']='上传私有输入，回读CPU/隐私/互联网/容器与附件，再启动唯一完整五折运行。'
        dump(record_path,record)
    dump(HERE/'execution-state.json',{'status':'PREPARED','observed_at':now,'notebook_ref':'zhjianduan/name-that-llm-svc4-raw-public-cv',
        'version':None,'gpu':False,'internet':False,'variants':[v['id'] for v in variants],
        'shared_wall_seconds':7200,'guard_work_seconds':6900,'rss_limit_gib':12,'duplicate_run_allowed':False,
        'protocol_sha256':freeze['protocol_sha256'],'result_status':'not_run'})
    print(json.dumps({'status':'FROZEN','payload_bytes':freeze['payload_bytes'],
         'max_working_memory_estimate_gib':max(c['working_memory_estimate_bytes'] for c in resource['cases'])/1024**3,
         'experiments':[v['id'] for v in variants]},ensure_ascii=False))

if __name__=='__main__':main()
