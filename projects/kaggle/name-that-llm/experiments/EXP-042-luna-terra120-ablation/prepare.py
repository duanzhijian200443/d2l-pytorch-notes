"""Build a private CPU comparison from frozen inputs; never start a run."""
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict, Counter
import base64
import hashlib
import io
import json
import unicodedata
import zipfile
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CLOUD=ROOT/'cloud/luna-terra120-042'
BASE=ROOT/'experiments/EXP-010-012-stylegram-share/payload'
REF='zhjianduan/name-that-llm-luna-terra120-042'
IMAGE='gcr.io/kaggle-images/python@sha256:17b2fc7f27c25c6c2fd4230ec8e66097ab2b89fc88e1e4d6f338b72e1ee0bc3e'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(t):return ' '.join(unicodedata.normalize('NFKC',t).casefold().split())
def text_sha(t):return hashlib.sha256(t.encode('utf-8')).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def entry(p):return {'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size}

def main():
    assert not (CLOUD/'execution-state.json').exists(),'Already prepared; do not overwrite or duplicate run'
    gate=read(HERE/'metrics/data-review.json')
    assert gate.get('status','').startswith('PASS'), 'Data review must pass before freezing'
    manifests=read(ROOT/'experiments/EXP-041-aligned-channel-repeat-audit/manifest.json')
    for spec in manifests['inputs']+manifests['code']+manifests.get('outputs',[]):
        assert sha(ROOT/spec['path'])==spec['sha256'],spec['path']
    src=ROOT/'experiments/EXP-040-codex300-concurrency2/collection/combined-records.jsonl'
    elig_path=ROOT/'experiments/EXP-041-aligned-channel-repeat-audit/metrics/training-eligibility.jsonl'
    oldpath=ROOT/'experiments/EXP-023-codex-subscription-pilot-r1/collection/records.jsonl'
    pairpath=ROOT/'experiments/EXP-038-paired-cloud-precheck-recovery/collection/combined-records.jsonl'
    rows=lines(src);bykey={x['job_key']:x for x in rows};assert len(bykey)==len(rows)==300
    eligibility=lines(elig_path);assert len({x['job_key'] for x in eligibility})==300
    chosen=[x for x in eligibility if x['assessment']=='CONTROLLED_TRAINING_CANDIDATE']
    assert len(chosen)==120
    group_splits=defaultdict(set);base_splits=defaultdict(set)
    oldrows=lines(oldpath)
    original_prompts={}
    for r in oldrows:
        original_prompts.setdefault(r['prompt_group'],r['prompt'])
        assert original_prompts[r['prompt_group']]==r['prompt']
    allrows=oldrows+rows+lines(pairpath)
    def base(r):
        original=original_prompts[r['prompt_group']]
        if 'base_prompt' in r:assert r['base_prompt']==original
        return original
    for r in allrows:
        group_splits[r['prompt_group']].add(r['split']);base_splits[norm(base(r))].add(r['split'])
    assert all(len(s)==1 for s in group_splits.values()) and all(len(s)==1 for s in base_splits.values())
    holdout_hashes=sorted({text_sha(norm(base(r))) for r in allrows if r['split']=='holdout'})
    selected=[]
    for e in chosen:
        r=bykey[e['job_key']];assert r['split']==e['split']=='train'
        assert r['model_requested']==e['model'] in ('gpt-5.6-luna','gpt-5.6-terra')
        assert r['prompt_group']==e['prompt_group'] and r['model_observation_scope']=='requested_only'
        final=ROOT/r['attempt_path_project']/'final.txt'
        assert sha(final)==r['raw_final_sha256']==r['text_sha256']==text_sha(r['text'])
        assert final.read_bytes()==r['text'].encode('utf-8')
        assert text_sha(norm(r['base_prompt'])) not in holdout_hashes
        selected.append({'row_id':'codex:'+r['job_key'],'origin':'codex_subscription_aligned_v1',
            'prompt_id':'codex:'+r['prompt_group'],'model':r['model_requested'],'text':r['text'],
            'base_prompt':r['base_prompt'],'text_sha256':r['text_sha256'],'job_key':r['job_key'],
            'split':r['split'],'source_protocol':r['protocol_version']})
    selected=sorted(selected,key=lambda r:(r['prompt_id'],r['model']))
    assert Counter(x['model'] for x in selected)=={'gpt-5.6-luna':60,'gpt-5.6-terra':60}
    assert len({x['prompt_id'] for x in selected})==60
    for d in (CLOUD/'payload',CLOUD/'kernel'):d.mkdir(parents=True,exist_ok=False)
    payload=CLOUD/'payload'
    source_files={'train.csv':ROOT/'train.csv','folds.csv':BASE/'folds.csv','extra.csv':BASE/'extra.csv',
        'stylegram_core.py':BASE/'stylegram_core.py','runner.py':HERE/'runner.py',
        'guard.py':ROOT/'cloud/svc4-raw-public-r1/guard.py'}
    for name,source in source_files.items():(payload/name).write_bytes(source.read_bytes())
    # The full local review contains protected holdout examples. Upload only an
    # allowlisted receipt without question/answer text, example snippets, or labels.
    save(payload/'data-gate.json',{'status':gate['status'],
        'local_review_sha256':sha(HERE/'metrics/data-review.json'),
        'candidate_rows':120,'candidate_groups':60,'holdout_rows_not_uploaded':60,
        'scope':'Known/exact checks for fixed official OOF; semantic independence not certified; EXP027 not evaluated.'})
    pd.DataFrame(selected).to_csv(payload/'new120.csv',index=False,lineterminator='\n')
    split=read(ROOT/'validation/prompt-group-v1/manifest.json');baseline=read(BASE/'protocol.json')
    for name in ('extra.csv','folds.csv','stylegram_core.py'):
        assert sha(payload/name)==baseline['input_hashes'][name],name
    assert sha(payload/'train.csv')==split['train_sha256']
    now=datetime.now(timezone.utc);runid='exp042-'+now.strftime('%Y%m%dT%H%M%SZ')
    cfg={'schema_version':1,'experiment_id':'EXP-042','run_id':runid,
        'baseline_id':'EXP-010','formal_ensemble_baseline_id':'EXP-004',
        'arms':['control','add120'],'labels':split['labels'],
        'new_labels':['gpt-5.6-luna','gpt-5.6-terra'],'validation_scheme':'prompt-group-v1',
        'C':.5,'dual':True,'max_iter':4000,'seed':42,'share':.5,
        'sparse_per_prompt':[2,3,6],'sparse_repeats':5,'threads':2,
        'weight_policy':'Freeze base official .5 and public .5*n_official_class/n_base_extra_class in both arms; new rows inherit the same per-class base public weight; no normalization after adding120.',
        'input_hashes':{name:sha(payload/name) for name in ('train.csv','folds.csv','extra.csv','new120.csv','stylegram_core.py','runner.py','data-gate.json')},
        'holdout_prompt_group_hashes':holdout_hashes,
        'identity':{'added_source':'EXP040 aligned Codex subscription','model_observation':'requested_only',
            'snapshot':'unknown','original_rows':120,'original_prompt_groups':60,'preserve_raw':True,
            'formal_pool_acceptance':False,'no_holdout_or_paired_diagnostic_text_uploaded':True},
        'environment':{'platform':'Kaggle CPU','accelerator':None,'internet':False,'container':IMAGE,
            'required_versions':{'numpy':'2.1.3','pandas':'2.3.3','scipy':'1.16.3','sklearn':'1.6.1'}},
        'budget':{'hard_seconds':6600,'platform_wall_seconds':7200,'rss_limit_bytes':20*1024**3,
            'new_cloud_runs':1,'gpu_seconds':0,'api_usd':0,'colab_cu':0},
        'evaluation':{'primary':'18-class pooled OOF balanced accuracy','bootstrap_draws':2000,'bootstrap_seed':42,
            'bootstrap_unit':'original official prompt_id','candidate_minus_control':True,
            'sparse_secondary':'m2/3/6 each repeats0..4 using original deterministic selections; regroup selected text before centering',
            'new_holdout_evaluated':False,'retune':False},
        'limits':['Official five-fold split is already inspected development validation, not new holdout.',
            'Known/exact data and prompt separation only; semantic overlap not exhaustively excluded.',
            'New class-specific extra weight mass increases naturally; old weights and algorithm remain fixed.',
            'Isolated Stylegram comparison, not complete ATLAS20 or leaderboard.']}
    save(HERE/'config.json',cfg);save(payload/'config.json',cfg)
    protocol={'experiment_id':'EXP-042','run_id':runid,'work_budget_seconds':6600,'rss_limit_bytes':20*1024**3,
        'input_hashes':{p.name:sha(p) for p in payload.iterdir() if p.is_file()}}
    save(payload/'protocol.json',protocol)
    # Reuse frozen existing mounted inputs. Only new120/code/receipts are embedded.
    mount_names={'extra.csv','folds.csv','train.csv'}
    bundle=io.BytesIO()
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(payload.iterdir()):
            if p.is_file() and p.name not in mount_names:z.writestr(p.name,p.read_bytes())
    encoded=base64.b64encode(bundle.getvalue()).decode('ascii')
    code="""import base64,hashlib,io,json,os,pathlib,runpy,sys,zipfile
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
target=pathlib.Path('/kaggle/working/exp042-payload')
assert not target.exists(),'Refusing repeated execution'
target.mkdir()
raw=base64.b64decode(ENCODED)
assert hashlib.sha256(raw).hexdigest()==BUNDLE_SHA
with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
 for name in bundle.namelist():
  assert pathlib.PurePosixPath(name).name==name and ':' not in name
  (target/name).write_bytes(bundle.read(name))
protocol=json.loads((target/'protocol.json').read_text())
for name in ('train.csv','extra.csv','folds.csv'):
 matches=[p for p in pathlib.Path('/kaggle/input').rglob(name) if hashlib.sha256(p.read_bytes()).hexdigest()==protocol['input_hashes'][name]]
 assert len(matches)==1,('Expected one frozen input',name,len(matches))
 (target/name).write_bytes(matches[0].read_bytes())
for name,digest in protocol['input_hashes'].items():assert hashlib.sha256((target/name).read_bytes()).hexdigest()==digest,name
sys.path.insert(0,str(target))
print('EXP042 frozen payload verified. CPU only, no network, no holdout text.',flush=True)
try:
 runpy.run_path(str(target/'guard.py'),run_name='__main__')
finally:
 root=pathlib.Path('/kaggle/working/results')
 if root.exists():
  with zipfile.ZipFile('/kaggle/working/exp042-results.zip','w',zipfile.ZIP_DEFLATED) as z:
   for p in sorted(root.rglob('*')):
    if p.is_file():z.write(p,p.relative_to(root).as_posix())
""".replace('ENCODED',repr(encoded)).replace('BUNDLE_SHA',repr(hashlib.sha256(bundle.getvalue()).hexdigest()))
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
        'cells':[{'cell_type':'code','metadata':{},'source':code.splitlines(True),'outputs':[],'execution_count':None,'id':'exp042-run'}]}
    save(CLOUD/'kernel/experiment.ipynb',nb)
    save(CLOUD/'kernel/kernel-metadata.json',{'id':REF,'title':'Name That LLM Luna Terra120 042',
        'code_file':'experiment.ipynb','language':'python','kernel_type':'notebook','is_private':True,
        'enable_gpu':False,'enable_tpu':False,'enable_internet':False,
        'dataset_sources':['zhjianduan/ntl-stylegram-cv-inputs-v1/1'],'competition_sources':['name-that-llm'],
        'kernel_sources':[],'model_sources':[],'docker_image_pinning_type':'original','docker_image':IMAGE})
    save(HERE/'manifest.json',{'schema_version':1,'experiment_id':'EXP-042','freeze_status':'FROZEN_BEFORE_TRAINING',
        'observed_at':now.isoformat(),'run_id':runid,'config':entry(HERE/'config.json'),
        'inputs':[entry(p) for p in (src,elig_path,oldpath,pairpath,HERE/'metrics/data-review.json',payload/'new120.csv',payload/'extra.csv',payload/'train.csv',payload/'folds.csv')],
        'code':[entry(p) for p in (Path(__file__),HERE/'launch.py',HERE/'runner.py',HERE/'test_runner.py',payload/'guard.py',payload/'stylegram_core.py',CLOUD/'kernel/experiment.ipynb')],
        'protocol':entry(payload/'protocol.json'),'model_calls':0,'formal_pool_ingestion':False})
    save(CLOUD/'execution-state.json',{'experiment_id':'EXP-042','status':'PREPARED','observed_at':now.isoformat(),
        'run_id':runid,'notebook_ref':REF,'version':None,'cloud_run_requests':0,'gpu':False,'internet':False,
        'platform_wall_seconds':7200,'work_budget_seconds':6600,'duplicate_run_allowed':False,'automatic_retry_allowed':False,
        'protocol_sha256':sha(payload/'protocol.json'),'config_sha256':sha(payload/'config.json')})
    print(json.dumps({'status':'FROZEN','run_id':runid,'embedded_zip_bytes':len(bundle.getvalue()),
        'new_rows':120,'holdout_hashes':len(holdout_hashes),'training_started':False}))

if __name__=='__main__':main()
