"""Frozen, label-free margin blending of accepted OOF score caches.

This is development screening, not a new independent validation set.
No fitting, calibration, remote execution, submission or grid expansion.
"""
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = ROOT / 'experiments/EXP-022-oof-complementarity'
sys.path.insert(0, str(ROOT))
from src.margin_blend import blend_margins, normalize_margins
from src.oof_complementarity import analyze_scenario


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, data):
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(path)


def renamed(value):
    """Reuse frozen paired-statistics code with explicit reference/blend names."""
    if isinstance(value, dict):
        return {key.replace('stylegram', 'reference').replace('svc4', 'blend'): renamed(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [renamed(item) for item in value]
    if value == 'svc4_minus_stylegram_balanced_accuracy':
        return 'blend_minus_reference_balanced_accuracy'
    return value


def load_inputs():
    spec = importlib.util.spec_from_file_location('exp022_cache_loader', PARENT/'analyze.py')
    parent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parent)
    inputs, old_config, labels, frames = parent.load_inputs()
    prior_manifest = inputs.json(PARENT/'manifest.json')
    prior_input_hashes = {e['path']: e['sha256'] for e in prior_manifest['inputs']}
    require(prior_manifest['freeze_status'] == 'FROZEN', 'Parent is not frozen')
    for name, digest in prior_input_hashes.items():
        require(inputs.hashes.get(name) == digest, 'Parent input binding changed: '+name)
    for e in prior_manifest['code']:
        inputs.track(ROOT/e['path'], e['sha256'])
    acceptance = inputs.json(PARENT/'metrics/acceptance.json')
    require(acceptance['status'] == 'PASS' and acceptance['manifest_sha256'] == sha(PARENT/'manifest.json'), 'Parent acceptance binding failed')
    config = inputs.json(HERE/'config.json')
    params = config['model']['parameters']
    require(params['alphas'] == [0.0, 0.05, 0.10, 0.20] and params['primary_alpha'] == .10, 'Changed candidate grid')
    require(all(x is False for x in config['analysis'].values()), 'Scope includes unauthorized fitting/mutation')
    raw_a = np.load(parent.STYLE_RESULTS/'oof-share0.5.npy', allow_pickle=False)
    raw_b = np.load(parent.SVC_RESULTS/'EXP-018/oof_decision.npy', allow_pickle=False)
    dense_frame = frames[0][3]
    positions = dict(zip(dense_frame.id, range(len(dense_frame))))
    scores = []
    for name, m, repeat, frame in frames:
        ix = np.array([positions[x] for x in frame.id])
        b = raw_b[ix]
        if m is None:
            a = raw_a[ix]
        else:
            a = np.full((len(frame), len(labels)), np.nan, dtype=np.float64)
            row_of = dict(zip(frame.id, range(len(frame))))
            seen = []
            for fold in range(5):
                path = parent.STYLE_RESULTS/f'sparse-f{fold}-a0.5-m{m}-r{repeat}.npz'
                inputs.track(path, inputs.hashes[path.relative_to(ROOT).as_posix()])
                with np.load(path, allow_pickle=False) as z:
                    ids = z['ids'].tolist();where = np.array([row_of[x] for x in ids])
                    a[where] = z['scores'];seen.extend(ids)
            require(len(seen) == len(set(seen)) == len(frame), 'Duplicate/missing sparse score rows')
        require(a.shape == b.shape == (len(frame),18) and np.isfinite(a).all() and np.isfinite(b).all(), 'Invalid paired score matrix')
        require(np.array(labels)[a.argmax(1)].tolist() == frame.stylegram_prediction.tolist(), 'Style score order differs from accepted frame')
        require(np.array(labels)[b.argmax(1)].tolist() == frame.svc4_prediction.tolist(), 'svc4 score order differs from accepted frame')
        for alpha, original in ((0.0,a),(1.0,b)):
            require(np.array_equal(blend_margins(a,b,alpha).argmax(1),original.argmax(1)), 'Endpoint argmax changed')
        scores.append((name,m,repeat,frame,a,b))
    inputs.unchanged()
    return inputs, config, labels, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    require(args.prepare != args.run, 'Choose prepare or run')
    start = time.monotonic()
    inputs,config,labels,scenarios = load_inputs()
    code = [HERE/'run_blend.py', ROOT/'src/margin_blend.py', PARENT/'analyze.py',
            ROOT/'src/oof_complementarity.py',ROOT/'validation/protocol.py']
    frozen = {'config':{'path':(HERE/'config.json').relative_to(ROOT).as_posix(),'sha256':sha(HERE/'config.json')},
              'inputs':[{'path':name,'sha256':h} for name,h in sorted(inputs.hashes.items())],
              'code':[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for p in code]}
    manifest_path=HERE/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if args.prepare:
        require(manifest['freeze_status']=='DRAFT','Already frozen')
        manifest.update(frozen,freeze_status='FROZEN',frozen_at=stamp(),note='Fixed .05/.10/.20 candidates before seeing blend metrics; alpha .10 primary; existing development OOF only.',environment={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__})
        save(manifest_path,manifest)
        constants=[{'scenario':name,'stylegram_constant_rows':int(np.all(a==a[:,:1],axis=1).sum()),'svc4_constant_rows':int(np.all(b==b[:,:1],axis=1).sum())} for name,m,repeat,frame,a,b in scenarios]
        save(HERE/'metrics/input-preflight.json',{'status':'PASS','observed_at':stamp(),'scenarios':16,'rows':sum(len(x[3]) for x in scenarios),'parent_acceptance':'EXP-022 PASS','normalization_endpoints_match_cached_argmax':True,'constant_rows':constants,'inputs':len(inputs.hashes),'manifest_sha256':sha(manifest_path)})
        print(json.dumps({'status':'PREPARED','inputs':len(inputs.hashes),'scenarios':16}))
        return
    require(manifest['freeze_status']=='FROZEN','Not frozen')
    for key in frozen:require(frozen[key]==manifest[key],'Frozen '+key+' changed')
    require(not (HERE/'metrics/result.json').exists(),'Results already exist')
    tables={key:[] for key in ('summary','per_class','per_fold','per_prompt','per_class_fold')}
    baselines=[];prediction_rows=[]
    alphas=config['model']['parameters']['alphas']
    for name,m,repeat,frame,a,b in scenarios:
        require(time.monotonic()-start<config['budget']['wall_seconds'],'Analysis budget exceeded')
        matrices=[]
        for alpha in alphas:
            blended=blend_margins(a,b,alpha);matrices.append(blended)
            if alpha==0:
                baselines.append({'scenario':name,'m':m,'repeat':repeat,'rows':len(frame),'stylegram_argmax_unchanged':True})
                continue
            compare=frame.copy()
            compare['svc4_prediction']=np.array(labels)[blended.argmax(1)]
            analysis=renamed(analyze_scenario(compare,labels,name,m=m,repeat=repeat,seed=42,bootstrap_draws=config['validation']['bootstrap_draws']))
            meta={'scenario':name,'m':m,'repeat':repeat,'alpha':alpha,'primary':alpha==.10}
            tables['summary'].append({**meta,**analysis['summary']})
            for key in tables:
                if key!='summary':tables[key].extend({**meta,**row} for row in analysis[key])
            rows=frame[['id','prompt_id','model','fold','stylegram_prediction','svc4_prediction']].copy()
            rows=rows.rename(columns={'stylegram_prediction':'reference_prediction','svc4_prediction':'component_svc4_prediction'})
            rows.insert(0,'alpha',alpha);rows.insert(0,'scenario',name)
            rows['blend_prediction']=np.array(labels)[blended.argmax(1)]
            good_ref=rows.reference_prediction.eq(rows.model);good_blend=rows.blend_prediction.eq(rows.model)
            rows['change']=np.select([~good_ref&good_blend,good_ref&~good_blend,rows.reference_prediction.ne(rows.blend_prediction)],['fixed_error','new_error','both_wrong_changed'],default='unchanged')
            prediction_rows.append(rows)
        np.savez_compressed(HERE/'oof'/f'{name}.npz',ids=frame.id.to_numpy(dtype=str),labels=np.array(labels),alphas=np.array(alphas),scores=np.stack(matrices))
        print('BLENDED '+name,flush=True)
    inputs.unchanged()
    for e in manifest['code']:require(sha(ROOT/e['path'])==e['sha256'],'Code changed')
    for key,rows in tables.items():
        flat=[{k:json.dumps(v,ensure_ascii=False,allow_nan=False) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows]
        pd.DataFrame(flat).to_csv(HERE/'metrics'/f'{key}.csv',index=False,lineterminator='\n')
    pd.concat(prediction_rows).to_csv(HERE/'oof/predictions.csv',index=False,lineterminator='\n')
    summary=pd.DataFrame(tables['summary']);screens=[]
    for alpha in alphas[1:]:
        rows=summary[summary.alpha.eq(alpha)]
        dense=float(rows.loc[rows.scenario.eq('dense'),'delta_balanced_accuracy'].iloc[0])
        means={str(m):float(rows.loc[rows.m.eq(m),'delta_balanced_accuracy'].mean()) for m in (2,3,6)}
        values=[dense]+list(means.values())
        eps=config['screening']['tie_tolerance']
        screens.append({'alpha':alpha,'primary':alpha==.10,'dense_delta_ba':dense,'sparse_mean_delta_ba':means,'point_estimate_guardrail_met':all(v>=-eps for v in values) and any(v>eps for v in values),'tie_tolerance':eps,'role':'development screening only; no automatic promotion or independent gain claim'})
    result={'status':'COMPLETE','observed_at':stamp(),'scope':'fixed standardized-margin blend development screening on previously inspected OOF','scenarios':16,'candidate_scenario_count':48,'alphas':alphas,'primary_alpha':.10,'summaries':tables['summary'],'baseline_checks':baselines,'screening':screens,'calibrated_probabilities':False,'training':False,'official_baseline_changed':False,'elapsed_seconds':time.monotonic()-start,'manifest_sha256':sha(manifest_path),'limits':['No independent holdout or nested validation; input data has already informed development.','Weights apply to per-row standardized margins, not original raw margins, calibrated probabilities or final ATLAS weights.','Sparse samples share models/prompts; intervals are conditional and unadjusted.','No additional alpha search, data-dependent gating or class-specific weighting was performed.']}
    save(HERE/'metrics/result.json',result)
    outputs=sorted((HERE/'oof').glob('*'))+sorted((HERE/'metrics').glob('*.csv'))+[HERE/'metrics/result.json',HERE/'metrics/input-preflight.json']
    save(HERE/'metrics/output-manifest.json',{'files':[{'path':p.relative_to(HERE).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in outputs if p.is_file()]})
    print(json.dumps({'status':'COMPLETE','elapsed_seconds':result['elapsed_seconds'],'screening':screens}))


if __name__=='__main__':
    main()
