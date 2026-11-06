"""Read-only verification of the completed diagnostic. Never rebuild or rescore it.

This post-run entry point is separate so the original frozen calculation code
and its pre-calculation manifest remain unchanged.
"""
from pathlib import Path
import hashlib,importlib.util,json,sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=ROOT/'experiments/EXP-045-public-data-full-stack'
CLOUD=ROOT/'cloud/a1-sinkhorn030-049'
CACHE=ROOT/'cloud/full-stack-capture-r1/outputs-v2/results'
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def check(entries):
    for e in entries:
        path=(ROOT/e['path']).resolve()
        assert path.is_relative_to(ROOT),'Manifest path outside repository'
        assert sha(path)==e['sha256'],'Hash changed: '+e['path']
        if 'bytes' in e:assert path.stat().st_size==e['bytes'],'Size changed: '+e['path']
def verify():
    manifest=read(HERE/'manifest.json')
    assert manifest['status']=='READY_NOT_LAUNCHED'
    check([manifest['config']]+manifest['code']+manifest['source_bindings']+manifest['candidate_files'])
    package=load(CLOUD/'prepare_verify.py','exp049_readonly_package')
    old,meta=package.source_gate()
    package.validate_candidate(old,meta,read(CLOUD/'kernel/a1.ipynb'),read(CLOUD/'kernel/kernel-metadata.json'))
    receipt=read(HERE/'metrics/package-verification.json')
    assert receipt['status']=='PASS_READY_NOT_LAUNCHED' and receipt['remote_actions']==0
    assert receipt['notebook_sha256']==sha(CLOUD/'kernel/a1.ipynb')
    assert receipt['metadata_sha256']==sha(CLOUD/'kernel/kernel-metadata.json')
    # Original inputs and mathematical helper bindings, including hidden cache shards.
    oldmanifest=read(OLD/'manifest.json')
    check([oldmanifest['config']]+oldmanifest['code']+oldmanifest['inputs'])
    review=read(OLD/'metrics/input-review.json')
    check([{'path':p,**e} for p,e in review['inputs'].items()])
    check([review['output']])
    cmp=load(ROOT/'experiments/EXP-032-sinkhorn-fixed-cache/compare_cached.py','exp049_readonly_cache')
    cmp.verify_artifact_inventory(CACHE,'1732849196220d023716a2dc912d660282c94022c2df370c69d9a6c2899bac63')
    closeout=read(HERE/'metrics/closeout.json');check(closeout['files'])
    assert closeout['remote_status']=='READY_NOT_LAUNCHED' and closeout['local_status']=='COMPLETE_LOCAL_DIAGNOSTIC'
    artifacts=read(HERE/'outputs/artifacts.json');check(artifacts['files'])
    summary=read(HERE/'outputs/summary.json');coverage=read(HERE/'outputs/coverage.json')
    assert summary['status']=='COMPLETE_LOCAL_DIAGNOSTIC' and summary['remote_status']=='READY_NOT_LAUNCHED'
    assert summary['baseline_A1_logits_bitwise_match_EXP045'] is True
    assert summary['rows']==729 and summary['prompt_groups']==56 and summary['covered_classes']==17 and summary['output_classes']==18
    assert coverage==summary['atlas_coverage'] and coverage['accepted'] and not coverage['missing_ids']
    assert coverage['requested_count']==206 and len(coverage['requested_ids'])==206
    assert coverage['requested_ids']==read(OLD/'outputs/coverage.json')['A1']['requested_ids']
    with np.load(HERE/'outputs/scores.npz',allow_pickle=False) as scores,np.load(OLD/'outputs/scores.npz',allow_pickle=False) as prior:
        assert np.array_equal(scores['ids'],prior['ids']) and np.array_equal(scores['labels'],prior['labels'])
        assert np.array_equal(scores['logits_045'],prior['A1']),'Stored baseline parity failed'
        assert scores['logits_030'].shape==scores['logits_045'].shape==(729,18)
        assert np.isfinite(scores['logits_030']).all() and np.isfinite(scores['pre_sinkhorn']).all()
        rows=pd.read_csv(HERE/'outputs/predictions.csv')
        assert rows.id.tolist()==scores['ids'].tolist() and rows.id.is_unique
        assert rows.prediction_045.tolist()==scores['labels'][scores['logits_045'].argmax(1)].tolist()
        assert rows.prediction_030.tolist()==scores['labels'][scores['logits_030'].argmax(1)].tolist()
    assert int(rows.correct_045.sum())==summary['baseline_045']['correct']==488
    assert int(rows.correct_030.sum())==summary['candidate_030']['correct']==490
    assert abs(summary['baseline_045']['macro_recall']-.6010754296632904)<1e-14
    assert summary['fixed_old_errors']==5 and summary['new_errors']==3 and summary['prediction_disagreements']==11
    assert summary['paired_prompt_bootstrap']['draws']==2000 and summary['paired_prompt_bootstrap']['seed']==42
    tests=read(HERE/'metrics/tests.json');assert tests['status']=='PASS' and tests['total_tests']==32
    return {'status':'PASS_REUSED_EXISTING_DIAGNOSTIC','experiment_id':'EXP-049','remote_status':'READY_NOT_LAUNCHED',
        'summary_path':(HERE/'outputs/summary.json').relative_to(ROOT).as_posix(),'summary_sha256':sha(HERE/'outputs/summary.json'),
        'candidate_notebook_sha256':receipt['notebook_sha256'],'checked_config_code_input_output_hashes':True,
        'baseline_parity':True,'atlas_complete':True,'readonly':True,'recomputed_postprocessing':False,'recomputed_bootstrap':False,
        'writes':0,'remote_calls':0,'verifier_sha256':sha(Path(__file__)),
        'baseline_045':summary['baseline_045'],'candidate_030':summary['candidate_030'],
        'paired_prompt_bootstrap':summary['paired_prompt_bootstrap']}
if __name__=='__main__':print(json.dumps(verify(),ensure_ascii=False,indent=2))
