"""Save/read back, then request one private CPU run. Never retry automatically."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'cloud/luna-terra120-042'
REF='zhjianduan/name-that-llm-luna-terra120-042'

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def now():return datetime.now(timezone.utc).isoformat()
def ack(r):
    return {'raw_ref':r.ref,'ref':r.ref.removeprefix('/code/').strip('/'),
        'version':r.version_number,'url':r.url,'error':r.error,
        'invalid_datasets':r.invalid_dataset_sources or [],'invalid_kernels':r.invalid_kernel_sources or [],
        'invalid_models':r.invalid_model_sources or []}
def normalized(nb):
    return [''.join(c['source']) if isinstance(c['source'],list) else c['source'] for c in nb['cells']]

def readback(api,version):
    dest=HERE/f'readback-v{version}'
    api.kernels_pull(REF+f'/{version}',str(dest),metadata=True,quiet=True)
    meta=read(dest/'kernel-metadata.json');wanted=read(HERE/'kernel/kernel-metadata.json')
    actual=read(dest/meta['code_file']);expected=read(HERE/'kernel/experiment.ipynb')
    assert normalized(actual)==normalized(expected),'Source differs'
    assert meta['is_private'] and not meta['enable_gpu'] and not meta['enable_tpu'] and not meta['enable_internet']
    assert meta['dataset_sources']==['zhjianduan/ntl-stylegram-cv-inputs-v1']
    assert meta['competition_sources']==['name-that-llm']
    assert not meta['kernel_sources'] and not meta['model_sources']
    assert meta['docker_image']==wanted['docker_image'] and meta['machine_shape'] in ('None','',None)
    value={'status':'PASS','observed_at':now(),'ref':REF,'version':version,'source_equal':True,
        'private':True,'gpu':False,'internet':False,'input_count':2,'docker_image':meta['docker_image'],
        'source_sha256':sha(dest/meta['code_file']),'metadata_sha256':sha(dest/'kernel-metadata.json')}
    save(dest/'verified.json',value);return value

def main():
    sp=HERE/'execution-state.json';state=read(sp)
    assert state['status']=='PREPARED' and state['cloud_run_requests']==0,'Run already prepared/requested'
    ex=ROOT/'experiments/EXP-042-luna-terra120-ablation'
    mf=read(ex/'manifest.json')
    for v in [mf['config'],mf['protocol']]+mf['inputs']+mf['code']:
        assert sha(ROOT/v['path'])==v['sha256'],v['path']
    assert read(ex/'metrics/preflight/state.json')['status']=='PREFLIGHT_COMPLETE'
    review=read(ex/'metrics/code-review.json')
    assert review['status']=='PASS'
    reviewed=review['code_sha256']
    for name in ('runner.py','test_runner.py','prepare.py','launch.py'):
        path=ex/name
        assert sha(path)==reviewed[path.relative_to(ROOT).as_posix()], 'Review stale: '+name
    api=KaggleApi();api.authenticate()
    state.update(status='SAVE_REQUEST_IN_FLIGHT',observed_at=now());save(sp,state)
    saved=ack(api.kernels_push(str(HERE/'kernel'),timeout=7200,no_run=True));save(HERE/'save-ack.json',saved)
    assert saved['ref']==REF and saved['version']==1 and not saved['error']
    assert not any(saved[k] for k in ('invalid_datasets','invalid_kernels','invalid_models'))
    readback(api,1)
    state.update(status='SAVED_VERIFIED',version=1,observed_at=now());save(sp,state)
    state.update(status='RUN_REQUEST_IN_FLIGHT',cloud_run_requests=1,observed_at=now());save(sp,state)
    running=ack(api.kernels_push(str(HERE/'kernel'),timeout=7200,no_run=False));save(HERE/'run-ack.json',running)
    assert running['ref']==REF and running['version']==2 and not running['error']
    assert not any(running[k] for k in ('invalid_datasets','invalid_kernels','invalid_models'))
    readback(api,2)
    current=api.kernels_status(REF+'/2')
    state.update(status=str(current.status).split('.')[-1],version=2,observed_at=now(),readback_verified=True)
    save(sp,state)
    recordpath=ROOT/'tracking/records/experiments/EXP-042.json';record=read(recordpath)
    record['source'].update(notebook_ref=REF,notebook_version=2,local_path=(HERE/'kernel/experiment.ipynb').relative_to(ROOT).as_posix(),sha256=sha(HERE/'kernel/experiment.ipynb'))
    record['observed_at']=state['observed_at'];record['execution'].update(status=state['status'],observed_at=state['observed_at'],run_id=state['run_id'],notebook_version=2,cloud_run_requests=1)
    record['conclusion']='固定两臂、输入、权重及五折验证已冻结；私有CPU V2已发起并回读，无结果指标。'
    record['next_action']='跟进固定V2，完整产物/基线复现验收通过后做18类OOF及配对题组区间；不自动重跑。'
    record['evidence']=list(dict.fromkeys(record['evidence']+[(HERE/'run-ack.json').relative_to(ROOT).as_posix(),(HERE/'readback-v2/verified.json').relative_to(ROOT).as_posix(),sp.relative_to(ROOT).as_posix()]))
    save(recordpath,record)
    print(json.dumps({'status':state['status'],'ref':REF,'version':2,'gpu':False,'private':True,'internet':False}))

if __name__=='__main__':main()
