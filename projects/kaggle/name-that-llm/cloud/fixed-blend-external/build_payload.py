"""Freeze one private CPU external check after the data gate passes."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import zipfile

import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
EXP=ROOT/'experiments/EXP-026-independent-blend-validation'
STYLE=ROOT/'experiments/EXP-010-012-stylegram-share'
SVC=ROOT/'cloud/svc4-raw-public-r1'
EXTERNAL=ROOT/'data/public-pools/enaimaan/ntl-openai-generated-responses/v3'
REF='zhjianduan/name-that-llm-fixed-blend-external-026'
DATASET='zhjianduan/ntl-fixed-blend-external-inputs-026'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:json.loads(p.read_text(encoding='utf-8'))


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    gate=load(EXP/'metrics/data-gate.json')
    assert gate['status']=='PASS_FOR_EXTERNAL_FOUR_CLASS_CHECK','External data not accepted'
    assert sha(EXP/'evaluation.csv')==gate['evaluation']['sha256'],'Evaluation changed after audit'
    for entry in gate['inputs']:
        assert sha(ROOT/entry['path'])==entry['sha256'],entry['path']
    assert not (HERE/'execution-state.json').exists(),'Controller already prepared; inspect its state instead of creating a duplicate'
    cfg=load(EXP/'config.json');split=load(ROOT/'validation/prompt-group-v1/manifest.json')
    frame=pd.read_csv(EXP/'evaluation.csv',dtype={'id':str,'prompt_id':str})
    wanted=cfg['evaluation']['labels']
    assert frame.id.is_unique and not frame[['id','prompt_id','model','text']].isna().any().any()
    assert sorted(frame.model.unique())==sorted(wanted)
    assert frame.groupby('prompt_id').agg(rows=('id','size'),labels=('model','nunique')).eq(4).all().all()
    assert 0<len(frame)<=cfg['evaluation']['max_rows']
    assert sha(ROOT/'train.csv')==split['train_sha256']
    assert cfg['model']['parameters']['alpha']==.10 and cfg['evaluation']['alpha']==.10
    assert cfg['environment']['required_versions']=={'numpy':'2.1.3','pandas':'2.3.3','scipy':'1.16.3','sklearn':'1.6.1'}
    style_protocol=load(STYLE/'payload/protocol.json');svc_protocol=load(SVC/'payload/protocol.json')
    assert sha(STYLE/'payload/extra.csv')==style_protocol['input_hashes']['extra.csv']
    assert sha(SVC/'payload/public.csv')==svc_protocol['input_hashes']['public.csv']
    assert sha(ROOT/'src/svc4.py')==svc_protocol['input_hashes']['svc4.py']
    assert sha(STYLE/'payload/stylegram_core.py')==style_protocol['input_hashes']['stylegram_core.py']
    assert sha(SVC/'guard.py')=='eaf2c31758045488eb53ba3f9c7f824c848bfe0a33698fab816d96485f9b6d3b'
    payload=HERE/'payload';dataset=HERE/'dataset';kernel=HERE/'kernel'
    for folder in (payload,dataset,kernel):folder.mkdir(parents=True,exist_ok=True)
    sources={'runner.py':HERE/'runner.py','guard.py':SVC/'guard.py',
        'stylegram_core.py':STYLE/'payload/stylegram_core.py','svc4.py':ROOT/'src/svc4.py','margin_blend.py':ROOT/'src/margin_blend.py',
        'style_extra.csv':STYLE/'payload/extra.csv','public.csv':SVC/'payload/public.csv',
        'evaluation.csv':EXP/'evaluation.csv','data_gate.json':EXP/'metrics/data-gate.json','config.json':EXP/'config.json',
        'enai-LICENSE.md':EXTERNAL/'LICENSE.md','enai-attribution.json':EXTERNAL/'attribution.json',
        'CC-BY-4.0.txt':EXTERNAL/'CC-BY-4.0.txt','CC-BY-SA-3.0.txt':EXTERNAL/'CC-BY-SA-3.0.txt'}
    for name,source in sources.items():
        (payload/name).write_bytes(source.read_bytes());assert sha(payload/name)==sha(source)
    (payload/'attribution.md').write_text('''# Private external validation inputs

This package is private research material for EXP-026; no competition submission or public redistribution is performed.

- Training Stylegram public text is the byte-identical EXP-010 extra.csv, originally from seyitkaangunes/ntl-extra-api-passages V2, lucasboesen/ntl-free-proxy-generations V7, dophuchung/ntl-proxy-generations V1, and prvsiyan/prvsiyans-extended-name-that-llm-dataset V6. Their original attribution and source records remain in the project.
- Training svc4 public text is the byte-identical 5,400 raw/new responses from prvsiyan V6 used by EXP-018.
- External text comes from enaimaan/ntl-openai-generated-responses V3, selected before scoring by the documented data gate. Enai (2026) publishes compilation/metadata under CC BY4.0 to the extent held. Generated-response rights review is recorded as not independently established; this package does not assert blanket clearance or relicense the text.
- Selected prompt ancestry is Databricks Dolly, credited to Databricks, Inc. (2023), CC BY-SA3.0. Original enai license and attribution notices and both license texts are included unchanged. Answers are not rewritten; only approved complete-panel rows are selected.
- Official training text is mounted separately from the competition.

The external labels cover four GPT classes; this is not a full 18-class validation or leaderboard result.
''',encoding='utf-8')
    now=datetime.now(timezone.utc).isoformat()
    hashes={p.name:sha(p) for p in payload.iterdir() if p.is_file() and p.name!='protocol.json'}
    protocol={'schema_version':1,'experiment_id':'EXP-026','created_at':now,'scope':'fixed alpha0.10 external four-class check with 18-way predictions; not full ATLAS20 or leaderboard',
        'labels':split['labels'],'train_sha256':split['train_sha256'],'official_rows':2699,'style_extra_rows':10073,'public_rows':5400,
        'evaluation_accepted':True,'evaluation_rows':len(frame),'evaluation_groups':int(frame.prompt_id.nunique()),'evaluation_labels':wanted,
        'evaluation_sha256':sha(EXP/'evaluation.csv'),'data_gate_sha256':sha(EXP/'metrics/data-gate.json'),
        'alpha':.10,'style_share':.5,'public_row_weight':.35,'required_versions':cfg['environment']['required_versions'],
        'input_hashes':hashes,'work_budget_seconds':3300,'platform_wall_seconds':3600,'rss_limit_bytes':12*1024**3,'threads':2,'gpu':False,'internet':False,
        'limits':['Only four externally generated GPT labels; all predictions still choose among 18 classes.','Data independence is limited to audited known ancestry and exact/normalized prompt/response checks; semantic overlap is not exhaustively ruled out.','Publisher-provided labels/settings are not independent authentication of serving weights.','The model recipe is fixed but full-data classifiers are refitted; prior OOF classifiers were not persisted.','No alpha tuning, class-specific selection, formal submission or baseline change.']}
    save(payload/'protocol.json',protocol)
    with zipfile.ZipFile(dataset/'external_payload.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(payload.iterdir()):archive.write(path,path.name)
    save(dataset/'dataset-metadata.json',{'id':DATASET,'title':'NTL Fixed Blend External Inputs 026','licenses':[{'name':'other'}],
        'description':'Private fixed alpha0.10 verification inputs. Original Enai/Databricks attribution and license scopes preserved; four-class external check, not a competition submission. Original training inputs unchanged.'})
    code="""import hashlib,json,pathlib,runpy,sys,os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
matches=[p for p in pathlib.Path('/kaggle/input').rglob('protocol.json') if hashlib.sha256(p.read_bytes()).hexdigest()==PROTOCOL_HASH]
assert len(matches)==1,'Expected one pinned protocol'
target=matches[0].parent
protocol=json.loads(matches[0].read_text())
for name,digest in protocol['input_hashes'].items():assert hashlib.sha256((target/name).read_bytes()).hexdigest()==digest,name
sys.path.insert(0,str(target))
runpy.run_path(str(target/'guard.py'),run_name='__main__')
""".replace('PROTOCOL_HASH',repr(sha(payload/'protocol.json')))
    (kernel/'bootstrap.py').write_text(code,encoding='utf-8')
    save(kernel/'kernel-metadata.json',{'id':REF,'title':'Name That LLM Fixed Blend External 026','code_file':'bootstrap.py','language':'python','kernel_type':'script',
        'is_private':True,'enable_gpu':False,'enable_tpu':False,'enable_internet':False,'dataset_sources':[DATASET+'/1'],'competition_sources':['name-that-llm'],
        'kernel_sources':[],'model_sources':[],'docker_image_pinning_type':'original','docker_image':cfg['environment']['container']})
    manifest=load(EXP/'manifest.json');assert manifest['freeze_status']=='DRAFT'
    manifest.update(freeze_status='FROZEN',frozen_at=now,note='External row set accepted and frozen before fitting or scoring; model recipe and alpha fixed.',
        config={'path':(EXP/'config.json').relative_to(ROOT).as_posix(),'sha256':sha(EXP/'config.json')},
        inputs=[{'path':(payload/name).relative_to(ROOT).as_posix(),'sha256':digest} for name,digest in sorted(hashes.items())],
        code=[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for p in [HERE/'runner.py',Path(__file__),EXP/'audit_external.py']],
        protocol={'path':(payload/'protocol.json').relative_to(ROOT).as_posix(),'sha256':sha(payload/'protocol.json')})
    save(EXP/'manifest.json',manifest)
    save(HERE/'preflight.json',{'status':'FROZEN','observed_at':now,'protocol_sha256':sha(payload/'protocol.json'),'payload_sha256':sha(dataset/'external_payload.zip'),
        'payload_bytes':(dataset/'external_payload.zip').stat().st_size,'evaluation_rows':len(frame),'evaluation_groups':int(frame.prompt_id.nunique()),'training_started':False})
    save(HERE/'execution-state.json',{'status':'PREPARED','observed_at':now,'experiment':'EXP-026','notebook_ref':REF,'version':None,'dataset_ref':DATASET,'dataset_version':1,
        'gpu':False,'internet':False,'work_budget_seconds':3300,'platform_wall_seconds':3600,'rss_limit_gib':12,'duplicate_run_allowed':False,'automatic_retry_allowed':False,
        'protocol_sha256':sha(payload/'protocol.json'),'result_status':'NOT_RUN'})
    print(json.dumps({'status':'FROZEN','rows':len(frame),'groups':int(frame.prompt_id.nunique()),'payload_bytes':(dataset/'external_payload.zip').stat().st_size}))


if __name__=='__main__':main()
