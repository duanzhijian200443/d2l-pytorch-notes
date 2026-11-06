"""Bounded aligned Codex batch; delegates generation to the frozen EXP030 core.

Default is offline dry-run. Exactly75 shared prompts x4 models, workers=1.
The frozen core's COMPLETE is hidden as FINALIZING until the final audit passes.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

EXP=Path(__file__).resolve().parents[1]
MODELS={'gpt-5.6-luna','gpt-5.6-terra','gpt-5.6-sol','gpt-6-astra'}
CORE='experiments/EXP-030-codex-subscription-pilot-r3/code/codex_collect.py'
FORMAT='experiments/EXP-034-explicit-format-alignment/code/prepare_alignment.py'
BANK='experiments/EXP-034-explicit-format-alignment/prompts.jsonl'
ALIGNED_CONFIG='experiments/EXP-034-explicit-format-alignment/config.json'
PILOT_CONFIG='experiments/EXP-034-explicit-format-alignment/pilot/config.json'
INSTRUCTIONS='experiments/EXP-021-codex-subscription-pilot/model-instructions.txt'
PROTOCOL='single-paragraph-plain-text-v1'

def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(data):return hashlib.sha256(data).hexdigest()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result

def verify_manifest(experiment):
    root=experiment.parents[1];manifest=load(experiment/'manifest.json')
    entries=[manifest['config']]+manifest.get('inputs',[])+manifest.get('code',[])
    entries += [e for e in manifest.get('outputs',[]) if e['path'].endswith(('/prompts.jsonl','/request-plan.jsonl'))]
    bound={}
    for item in entries:
        path=(root/item['path']).resolve()
        if not path.is_relative_to(root.resolve()):raise ValueError('Frozen path outside project')
        if item['path'] in bound and bound[item['path']]!=item['sha256']:raise ValueError('Conflicting hashes')
        if sha(path.read_bytes())!=item['sha256']:raise ValueError('Frozen hash mismatch: '+item['path'])
        bound[item['path']]=item['sha256']
    required=[CORE,FORMAT,BANK,ALIGNED_CONFIG,PILOT_CONFIG,INSTRUCTIONS]+[
        (experiment/p).relative_to(root).as_posix() for p in ['config.json','prompts.jsonl','code/run_batch.py']]
    if any(p not in bound for p in required):raise ValueError('Manifest lacks required runtime/input bindings')
    if bound[required[-1]]!=sha(Path(__file__).read_bytes()):raise ValueError('Executing wrapper differs from frozen wrapper')
    return manifest,bound

def validate_selection(experiment,codex,core):
    root=experiment.parents[1]
    prepared=core.prepare(experiment,codex)
    _,config,instructions,workspace,binding,jobs=prepared
    prompts=core.read_jsonl(experiment/'prompts.jsonl')
    bank={p['prompt_id']:p for p in core.read_jsonl(root/BANK)}
    paired=set(load(root/ALIGNED_CONFIG)['prepared_paired_panel']['prompt_ids'])
    if len(prompts)!=75 or len({p['prompt_id'] for p in prompts})!=75:raise ValueError('Exactly75 unique prompt groups required')
    if Counter(p['split'] for p in prompts)!={'train':60,'holdout':15}:raise ValueError('Expected60 train and15 holdout groups')
    if {p['prompt_id'] for p in prompts}&paired:raise ValueError('Paired panel must remain disjoint')
    for p in prompts:
        if p!=bank.get(p['prompt_id']) or p['prompt_group']!=p['prompt_id'] or p.get('protocol_version')!=PROTOCOL:
            raise ValueError('Selected prompt differs from canonical aligned bank')
    if len(config['models'])!=4 or {m['slug'] for m in config['models']}!=MODELS:
        raise ValueError('Exactly four approved models required')
    if any(m.get('reasoning_effort')!='low' for m in config['models']):raise ValueError('Reasoning setting changed')
    if config.get('cli_overrides')!=load(root/PILOT_CONFIG)['cli_overrides']:raise ValueError('Frozen CLI overrides changed')
    if instructions.resolve()!=(root/INSTRUCTIONS).resolve() or instructions.read_bytes()!=b"Answer the user's request.\n":
        raise ValueError('Minimal base instruction changed')
    if config.get('budget',{}).get('max_concurrent_requests',1)!=1:raise ValueError('Batch worker cap must be1')
    if len(jobs)!=300 or len({j['key'] for j in jobs})!=300:raise ValueError('Expected300 distinct jobs')
    plan_path=experiment/'request-plan.jsonl'
    if plan_path.exists():
        plan=core.read_jsonl(plan_path)
        keys=[p.get('job_key',p.get('key')) for p in plan]
        if keys!=[j['key'] for j in jobs]:raise ValueError('Request plan differs from derived frozen jobs')
    return prepared

def check_preflight(experiment,codex):
    p=load(experiment/'metrics/execution-preflight.json');now=datetime.now(timezone.utc)
    observed=datetime.fromisoformat(p['observed_at'].replace('Z','+00:00'))
    expires=datetime.fromisoformat(p['expires_at'].replace('Z','+00:00'))
    if p.get('status')!='PASS' or observed.tzinfo is None or expires.tzinfo is None or not observed<=now<expires or (now-observed).total_seconds()>1800:
        raise ValueError('Preflight failed or expired')
    if p.get('login_chatgpt') is not True:raise ValueError('ChatGPT subscription authentication not verified')
    if Path(p['codex_executable_path']).resolve()!=codex.resolve() or sha(codex.read_bytes())!=p['codex_executable_sha256']:
        raise ValueError('Codex executable changed')
    return p

def inspect_saved(core,original,checker,attempt,outcome,model):
    result=original(attempt,outcome,model)
    events=core.read_jsonl(attempt/'raw_stdout.jsonl')
    completed=[e for e in events if e.get('type')=='turn.completed']
    messages=[e.get('item',{}).get('text') for e in events if e.get('type')=='item.completed' and e.get('item',{}).get('type')=='agent_message']
    if len(completed)!=1:result['reasons'].append('COMPLETION_COUNT_MISMATCH')
    if not messages or messages[-1]!=result['text']:result['reasons'].append('FINAL_EVENT_TEXT_MISMATCH')
    result['success']=not result['reasons']
    result['flags']=sorted(set(result['flags']+checker(result['text'])))
    target=load(attempt/'request.json')['prompt'].get('target_words')
    if target is not None and not .75*target<=result['word_count']<=1.25*target:
        result['flags'].append('target_length_deviation_over_25_percent')
    return result

def audit_collection(experiment,core,original_inspect,checker,state,jobs,binding):
    verify_manifest(experiment)
    collection=experiment/'collection';rows=core.read_jsonl(collection/'records.jsonl')
    planned={j['key']:j for j in jobs};errors=[];counts=Counter();splits=Counter();flags=Counter()
    keys=[r['job_key'] for r in rows]
    if len(keys)!=len(set(keys)):errors.append('DUPLICATE_JOB_KEYS')
    for row in rows:
        try:
            job=planned[row['job_key']];attempt=collection/'attempts'/row['job_key']
            if any(row.get(k)!=v for k,v in job['prompt'].items() if k!='source'):errors.append('PROMPT_IDENTITY:'+row['job_key'])
            if row['model_requested']!=job['model']['slug'] or row['binding']!=binding:errors.append('MODEL_OR_BINDING:'+row['job_key'])
            req=load(attempt/'request.json');result=load(attempt/'result.json')
            if req['job_key']!=row['job_key'] or req['prompt']!=job['prompt'] or req['model']!=job['model']:errors.append('REQUEST_IDENTITY:'+row['job_key'])
            if result['status']!='SUCCEEDED' or load(attempt/'record.json')!=row:errors.append('ATTEMPT_RECORD:'+row['job_key'])
            raw=(attempt/'final.txt').read_bytes()
            if raw!=row['text'].encode('utf-8') or sha(raw)!=row['text_sha256'] or sha(raw)!=row['raw_final_sha256'] or sha(row['prompt'].encode())!=row['prompt_sha256']:
                errors.append('TEXT_HASH:'+row['job_key'])
            inspected=inspect_saved(core,original_inspect,checker,attempt,{'returncode':result['returncode']},row['model_requested'])
            if not inspected['success'] or sorted(inspected['flags'])!=sorted(row['flags']):errors.append('RAW_COMPLETION:'+row['job_key'])
            counts[row['model_requested']]+=1;splits[row['split']]+=1;flags.update(row['flags'])
        except Exception as exc:errors.append('ROW_'+type(exc).__name__)
    attempts=list((collection/'attempts').iterdir());statuses=Counter()
    for attempt in attempts:
        if not attempt.is_dir() or not (attempt/'result.json').exists():errors.append('UNFINISHED_ATTEMPT');continue
        result=load(attempt/'result.json');statuses[result['status']]+=1
        if attempt.name not in planned or result.get('job_key')!=attempt.name:errors.append('ATTEMPT_IDENTITY')
    for name in ['state.json','summary.json','run-'+state['run_id']+'.json']:
        if load(collection/name).get('run_id')!=state['run_id']:errors.append('TERMINAL_RUN_MISMATCH')
    complete=(state['status']=='COMPLETE' and len(rows)==300 and set(keys)==set(planned)
              and dict(counts)=={m:75 for m in MODELS} and dict(splits)=={'train':240,'holdout':60}
              and len({r['prompt_group'] for r in rows})==75 and len(attempts)==300
              and statuses=={'SUCCEEDED':300} and not errors and not state['active']
              and not state.get('failures') and not state.get('local_errors'))
    audit={'observed_at':core.utc_now(),'run_id':state['run_id'],'status':'COLLECTED_UNVALIDATED' if complete else 'INCOMPLETE',
           'responses':len(rows),'by_model':dict(counts),'by_split':dict(splits),'prompt_groups':len({r['prompt_group'] for r in rows}),
           'duplicate_job_keys':len(keys)-len(set(keys)),'duplicate_text_rows':len(rows)-len({r['text_sha256'] for r in rows}),
           'attempts_by_status':dict(statuses),'integrity_errors':errors,'flags':dict(flags),
           'model_observation_scopes':dict(Counter(r['model_observation_scope'] for r in rows)),
           'workers':1,'training_acceptance':'NOT_EVALUATED','bare_api_equivalence':'NOT_ESTABLISHED'}
    core.write_json(experiment/'metrics/collection-audit.json',audit)
    return audit

def run_batch(experiment=EXP,*,run=False,codex=None,workers=1,max_seconds=21600,runner=None):
    if workers!=1:raise ValueError('This batch requires exactly1 worker; existing paired collection may use the other slot')
    if not 0<max_seconds<=21600:raise ValueError('Maximum wall budget is21600 seconds')
    experiment=Path(experiment).resolve();root=experiment.parents[1]
    manifest,_=verify_manifest(experiment)
    core=module('frozen_codex_core',root/CORE);formatting=module('frozen_format',root/FORMAT)
    config=load(experiment/'config.json')
    codex=Path(codex or load(experiment/'metrics/execution-preflight.json')['codex_executable_path'])
    _,_,_,_,binding,jobs=validate_selection(experiment,codex,core)
    if not run:return core.run_collection(experiment,codex,limit=300,workers=1,run=False,max_seconds=max_seconds,runner=runner)
    if config.get('authorization',{}).get('generation') is not True or manifest.get('generation_authorized') is not True:
        raise ValueError('Generation authorization missing')
    check_preflight(experiment,codex)
    collection=experiment/'collection';collection.mkdir(exist_ok=True)
    original_write,original_inspect=core.write_json,core.inspect_response
    def guarded_write(path,value,exclusive=False):
        if isinstance(value,dict) and value.get('status')=='COMPLETE' and (path.name in ['state.json','summary.json'] or path.name.startswith('run-')):
            value=dict(value,status='FINALIZING')
        return original_write(path,value,exclusive)
    def inspected(attempt,outcome,model):return inspect_saved(core,original_inspect,formatting.format_flags,attempt,outcome,model)
    with core.RunLock(collection/'batch.lock'):
        core.write_json,core.inspect_response=guarded_write,inspected
        try:state=core.run_collection(experiment,codex,limit=300,workers=1,run=True,max_seconds=max_seconds,runner=runner)
        finally:core.write_json,core.inspect_response=original_write,original_inspect
        try:
            audit=audit_collection(experiment,core,original_inspect,formatting.format_flags,state,jobs,binding)
            state['audit_status']=audit['status'];state['audit_run_id']=audit['run_id']
            if state['status']=='COMPLETE' and (audit['status']!='COLLECTED_UNVALIDATED' or audit['run_id']!=state['run_id']):state['status']='AUDIT_FAILED'
        except Exception as exc:
            state.update(status='AUDIT_FAILED',audit_status='ERROR',audit_error_type=type(exc).__name__)
        for name in ['state.json','summary.json','run-'+state['run_id']+'.json']:
            state['observed_at']=core.utc_now();original_write(collection/name,state)
        return state

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment','--exp',type=Path,default=EXP)
    parser.add_argument('--codex',type=Path);parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--max-seconds',type=float,default=21600)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--run',action='store_true');mode.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    try:
        result=run_batch(args.experiment,run=args.run,codex=args.codex,workers=args.workers,max_seconds=args.max_seconds)
        print(json.dumps(result,ensure_ascii=False,indent=2));return 0 if result['status'] in ['DRY_RUN','COMPLETE'] else 2
    except Exception as exc:
        print(json.dumps({'status':'NOT_STARTED','error_type':type(exc).__name__,
                          'message':str(exc) if isinstance(exc,(ValueError,FileNotFoundError)) else 'Inspect local records; no service body printed'},ensure_ascii=False));return 2

if __name__=='__main__':raise SystemExit(main())
