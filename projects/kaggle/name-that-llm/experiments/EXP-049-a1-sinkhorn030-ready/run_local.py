"""Fixed A1 .45/.30 cache diagnostic. No fitting, inference, remote calls, or grid search."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.util,json,platform,sys,time
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import scipy
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
OLD=ROOT/'experiments/EXP-045-public-data-full-stack'
PRIOR=ROOT/'experiments/EXP-032-sinkhorn-fixed-cache'
CACHE=ROOT/'cloud/full-stack-capture-r1/outputs-v2/results'
CLOUD=ROOT/'cloud/a1-sinkhorn030-049'
ARTIFACT_SHA='1732849196220d023716a2dc912d660282c94022c2df370c69d9a6c2899bac63'
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def main():
    started=time.perf_counter()
    package=load(CLOUD/'prepare_verify.py','exp049_package');package.verify()
    old=load(OLD/'run_local.py','exp049_existing_run_functions');old.check_frozen()
    out=HERE/'outputs';assert not out.exists(),'Prior result exists; no overwrite or retuning'
    cmp=load(PRIOR/'compare_cached.py','exp049_acceptance')
    protocol=cmp.read_json(PRIOR/'protocol.json');cmp.validate_protocol(protocol)
    accepted=cmp.verify_real_artifacts(CACHE,protocol,ARTIFACT_SHA)
    panel=pd.read_csv(cmp.bound_input(protocol['inputs']['blind_panel']))
    train=pd.read_csv(cmp.bound_input(protocol['inputs']['official_train']))
    panel['text']=panel.text.fillna('');train['text']=train.text.fillna('')
    with np.load(OLD/'oof/aligned-candidate-margins.npz',allow_pickle=False) as data:
        assert set(data.files)=={'ids','source_ids','text_sha256','labels','stylegram_scores','svc4_scores'}
        assert data['ids'].tolist()==panel.id.tolist()
        assert data['text_sha256'].tolist()==[hashlib.sha256(t.encode()).hexdigest() for t in panel.text]
        labels=data['labels'].tolist();style=data['stylegram_scores'].copy();svc=data['svc4_scores'].copy()
    math=load(OLD/'math_replay.py','exp049_reused_math')
    with cmp.captured_helpers(CACHE,accepted['plan']) as helpers:
        snapshots={name:helpers['full_stack_cache'].read_cache(CACHE/'cache',name,panel.id.tolist(),panel.text.tolist(),labels,
            required_notes=['event-index'],manifest_sha256=accepted['receipts'][name]['manifest_sha256']) for name in ('main','comp')}
        events={name:helpers['full_stack_observer'].decode_events(snap) for name,snap in snapshots.items()}
        # Reuse exact EXP045 helper. A2/A3 are calculated internally but never scored here.
        replay=math.replay_four_arms(events['main'],events['comp'],style,svc,mode='strict',
            expected_main_members=accepted['plan']['expected_main_members'],expected_comp_members=accepted['plan']['expected_comp_members'])
        arm=replay['arms']['A1']
        route={'requested_count':len(arm['requested_rows']),'missing_ids':list(arm['missing_ids']),
            'requested_ids':panel.id.iloc[np.asarray(arm['requested_rows'],dtype=int)].tolist(),'accepted':bool(arm['accepted'])}
        out.mkdir()
        save(out/'coverage.json',route)
        if arm['pre_sinkhorn'] is None or arm['missing_ids'] or not arm['accepted']:
            save(out/'summary.json',{'status':'NOT_SCORED_MISSING_ATLAS','scores':None,'coverage':route,'local_wall_seconds':time.perf_counter()-started})
            raise RuntimeError('A1 ATLAS incomplete: fail closed; no substitution/scoring')
        oldroute=json.loads((OLD/'outputs/coverage.json').read_text())['A1']
        assert route['requested_ids']==oldroute['requested_ids'] and oldroute['missing_count']==0
        post=load(ROOT/'src/sinkhorn_compare.py','exp049_reused_post')
        computed={key:post.compute_sinkhorn_variant(arm['pre_sinkhorn'],train.model.tolist(),train.text.tolist(),panel.text.tolist(),variant=variant,labels=labels)
            for key,variant in [('045','control'),('030','candidate')]}
        scores={k:v['final_logits'] for k,v in computed.items()}
        with np.load(OLD/'outputs/scores.npz',allow_pickle=False) as previous:
            assert previous['ids'].tolist()==panel.id.tolist() and previous['labels'].tolist()==labels
            assert np.array_equal(previous['A1'],scores['045']),'A1 .45 logits not bitwise equal to EXP045'
        np.savez_compressed(out/'scores.npz',ids=panel.id.to_numpy(dtype=str),labels=np.array(labels),
            pre_sinkhorn=arm['pre_sinkhorn'],logits_045=scores['045'],logits_030=scores['030'])
    truth=pd.read_csv(cmp.bound_input(protocol['inputs']['truth_sidecar']),dtype=str,keep_default_na=False)
    assert len(truth)==729 and truth.prompt_id.nunique()==56 and truth.model.nunique()==17
    assert truth.id.tolist()==panel.id.tolist()
    assert truth.text_sha256.tolist()==[hashlib.sha256(t.encode()).hexdigest() for t in panel.text]
    metrics=load(PRIOR/'metrics_utils.py','exp049_reused_metrics')
    assert labels==list(metrics.OUTPUT_LABELS) and all(x.shape==(729,18) for x in scores.values())
    frame=truth[['id','source_id','prompt_id','model']].copy()
    frame['prediction_045']=metrics.prediction_labels(scores['045']);frame['prediction_030']=metrics.prediction_labels(scores['030'])
    results=metrics.compare_predictions(frame);summary=results['summary']
    assert summary['baseline_045']['correct']==488
    assert abs(summary['baseline_045']['macro_recall']-.6010754296632904)<1e-14
    frame['correct_045']=frame.prediction_045.eq(frame.model);frame['correct_030']=frame.prediction_030.eq(frame.model)
    frame['change']=np.select([~frame.correct_045 & frame.correct_030,frame.correct_045 & ~frame.correct_030,frame.prediction_045.ne(frame.prediction_030)],['fixed_by_030','new_error_030','both_wrong_changed'],default='unchanged')
    frame.to_csv(out/'predictions.csv',index=False)
    for key in ['per_class','per_prompt','errors']:pd.DataFrame(results[key]).to_csv(out/(key+'.csv'),index=False)
    frame.loc[frame.prediction_045.ne(frame.prediction_030)].to_csv(out/'changed_rows.csv',index=False)
    cmp.verify_artifact_inventory(CACHE,ARTIFACT_SHA);old.check_frozen();package.verify()
    summary.update(status='COMPLETE_LOCAL_DIAGNOSTIC',experiment_id='EXP-049',remote_status='READY_NOT_LAUNCHED',
        observed_at=datetime.now(timezone.utc).isoformat(),local_wall_seconds=time.perf_counter()-started,
        baseline_A1_logits_bitwise_match_EXP045=True,atlas_coverage=route,source_capture_artifact_sha256=ARTIFACT_SHA,
        is_leaderboard=False,clean_oof=False,self_generated_rows_added=0,model_inference_calls=0,model_fit_calls=0,gpu_seconds=0,remote_calls=0,
        bootstrap_interpretation='Conditional on existing models and previously inspected panel; no retraining uncertainty; crossing zero is not equivalence.',
        environment={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__})
    save(out/'summary.json',summary)
    save(out/'artifacts.json',{'files':[package.entry(p) for p in sorted(out.iterdir()) if p.is_file()]})
    print(json.dumps({k:summary[k] for k in ['status','baseline_045','candidate_030','fixed_old_errors','new_errors','prediction_disagreements','paired_prompt_bootstrap','local_wall_seconds']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
