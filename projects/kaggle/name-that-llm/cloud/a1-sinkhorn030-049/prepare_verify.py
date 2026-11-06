"""Offline preparation/verification only. No Kaggle client, save, run or submit path."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, copy, difflib, hashlib, json, sys
sys.dont_write_bytecode = True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
EXP=ROOT/'experiments/EXP-049-a1-sinkhorn030-ready'
BASE=ROOT/'cloud/a1-public-stylegram-046/kernel'
NB_SHA='cea8f8cbfa382306dbdd7b7361b16d1d68febc09d7ef585dcb01ee1ef9e46625'
META_SHA='74007dcac3b5e34c7886fb3e946147ddb2335a3e1e64b530a29e646983943c37'
INPUT_KEYS=['competition_sources','dataset_sources','kernel_sources','model_sources']
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def entry(p): return {'path':Path(p).relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':Path(p).stat().st_size}
def text(c): return ''.join(c['source'])
def expected_notebook(old):
    nb=copy.deepcopy(old)
    nb['cells'][0]['source']=text(old['cells'][0]).replace('EXP046: A1 public-data Stylegram + ATLAS20','EXP049: A1 public-data Stylegram + ATLAS20, Sinkhorn 0.30')+'\nEXP049 prepared locally; READY_NOT_LAUNCHED. Relative to frozen EXP046 A1, only the main Sinkhorn argument changes 0.45 to 0.30. No self-generated 120-row addition. Runtime receipts identify this candidate; no result is claimed.\n'
    boot=text(old['cells'][2]); assert boot.count('# EXP046:')==1
    nb['cells'][2]['source']=boot.replace('# EXP046:','# EXP049:')
    main=text(old['cells'][12])
    replacements={"'--sinkhorn', '0.45'":"'--sinkhorn', '0.30'","'experiment_id':'EXP-046'":"'experiment_id':'EXP-049'","'sinkhorn':.45":"'sinkhorn':.30"}
    for a,b in replacements.items():
        assert main.count(a)==1,(a,main.count(a)); main=main.replace(a,b)
    nb['cells'][12]['source']=main
    return nb
def expected_metadata(old):
    result=copy.deepcopy(old)
    result.update(id='zhjianduan/name-that-llm-a1-sinkhorn030-049',title='Name That LLM A1 Sinkhorn030 049')
    return result
def validate_candidate(old,oldmeta,new,newmeta):
    assert new==expected_notebook(old),'Notebook differs outside exact allowed change list'
    assert newmeta==expected_metadata(oldmeta),'Metadata differs outside identity fields'
    assert sum(len(newmeta[k]) for k in INPUT_KEYS)==32
    assert all(newmeta[k]==oldmeta[k] for k in INPUT_KEYS)
    assert newmeta['is_private'] and not newmeta['enable_internet']
    for cell in new['cells']:
        if cell['cell_type']=='code':
            s=text(cell);s=s.split('\n',1)[1] if s.startswith('%%writefile ') else s
            compile(s,'EXP049-candidate-cell','exec')
    return {'changed_cells':[i for i,(a,b) in enumerate(zip(old['cells'],new['cells'])) if a!=b],
            'attachment_count':32,'payload_and_model_source_cells_unchanged':True,
            'mathematical_change':"cell12 command --sinkhorn 0.45 -> 0.30",'other_changes':['cell0 description','cell2 bootstrap comment identity','cell12 experiment and Sinkhorn receipt','metadata id/title']}
def source_gate():
    assert sha(BASE/'a1.ipynb')==NB_SHA,'EXP046 frozen notebook changed'
    assert sha(BASE/'kernel-metadata.json')==META_SHA,'EXP046 frozen metadata changed'
    assert not any(HERE.glob('*intent*.json')),'Remote intent unexpectedly present'
    return read(BASE/'a1.ipynb'),read(BASE/'kernel-metadata.json')
def prepare():
    old,meta=source_gate()
    if (EXP/'manifest.json').exists():
        draft=read(EXP/'manifest.json')
        assert draft.get('freeze_status')=='DRAFT' and not draft.get('code') and not (HERE/'kernel').exists(),'Prepared package exists; use verify, do not overwrite'
    new,newmeta=expected_notebook(old),expected_metadata(meta)
    audit=validate_candidate(old,meta,new,newmeta)
    save(HERE/'kernel/a1.ipynb',new);save(HERE/'kernel/kernel-metadata.json',newmeta)
    lines=[]
    for i in audit['changed_cells']:
        lines.extend(difflib.unified_diff(text(old['cells'][i]).splitlines(True),text(new['cells'][i]).splitlines(True),fromfile=f'EXP046/cell{i}',tofile=f'EXP049/cell{i}'))
    (HERE/'candidate.diff').write_text(''.join(lines),encoding='utf-8')
    config={'schema_version':1,'experiment_id':'EXP-049','status':'READY_NOT_LAUNCHED','created_at':datetime.now(timezone.utc).isoformat(),
        'baseline_experiment':'EXP-046','only_math_change':{'parameter':'sinkhorn','control':.45,'candidate':.30},
        'arms':['A1_sinkhorn045','A1_sinkhorn030'],'self_generated_rows_added':0,'remote_action_authorized':False,
        'fixed':{'target_mix':.35,'typography_scale':1.5,'comp_weight':.15,'atlas_weight':.20,'atlas_margin':.30,'stylegram':'official2699+public10073_share0.5_raw_margin','original_sharedgate_data':True},
        'local_diagnostic':{'scope':'EXP045 A1 real cache; already inspected external panel; not clean OOF or LB','rows':729,'output_classes':18,'true_classes':17,'prompt_groups':56,'bootstrap_draws':2000,'bootstrap_seed':42,'atlas_policy':'strict complete coverage or no scores','refit':False},
        'remote_save_started':False,'remote_run_started':False,'submission_started':False}
    save(EXP/'config.json',config)
    bindings=[BASE/'a1.ipynb',BASE/'kernel-metadata.json',ROOT/'experiments/EXP-045-public-data-full-stack/manifest.json',
        ROOT/'experiments/EXP-045-public-data-full-stack/outputs/scores.npz',ROOT/'experiments/EXP-045-public-data-full-stack/outputs/coverage.json',
        ROOT/'experiments/EXP-032-sinkhorn-fixed-cache/metrics_utils.py']
    codes=[Path(__file__),HERE/'test_prepare_verify.py',EXP/'run_local.py']
    for item in bindings+codes: assert item.is_file(),item
    save(EXP/'manifest.json',{'schema_version':1,'experiment_id':'EXP-049','freeze_status':'FROZEN_PREPARED_BEFORE_DIAGNOSTIC','status':'READY_NOT_LAUNCHED','config':entry(EXP/'config.json'),
        'code':[entry(p) for p in codes],'source_bindings':[entry(p) for p in bindings],
        'candidate_files':[entry(HERE/'kernel/a1.ipynb'),entry(HERE/'kernel/kernel-metadata.json'),entry(HERE/'candidate.diff')],
        'change_audit':audit,'local_diagnostic_completed':False,'remote_execution':False,'submission':False})
    return verify()
def verify():
    old,meta=source_gate();m=read(EXP/'manifest.json')
    assert m['status']=='READY_NOT_LAUNCHED'
    for e in [m['config']]+m['code']+m['source_bindings']+m['candidate_files']:
        assert sha(ROOT/e['path'])==e['sha256'],'Frozen binding changed: '+e['path']
    audit=validate_candidate(old,meta,read(HERE/'kernel/a1.ipynb'),read(HERE/'kernel/kernel-metadata.json'))
    out={'status':'PASS_READY_NOT_LAUNCHED','experiment_id':'EXP-049','notebook_sha256':sha(HERE/'kernel/a1.ipynb'),'metadata_sha256':sha(HERE/'kernel/kernel-metadata.json'),'audit':audit,'remote_actions':0}
    save(EXP/'metrics/package-verification.json',out)
    return out
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','verify']);a=p.parse_args()
    print(json.dumps(prepare() if a.command=='prepare' else verify(),ensure_ascii=False,indent=2))
