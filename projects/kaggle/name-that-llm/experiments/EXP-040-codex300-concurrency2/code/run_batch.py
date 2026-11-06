"""Two-worker continuation of the same300 jobs; inherited successes are read-only."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timedelta,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

EXP=Path(__file__).resolve().parents[1]
BASE='experiments/EXP-039-codex-aligned-300/code/run_batch.py'
CORE='experiments/EXP-030-codex-subscription-pilot-r3/code/codex_collect.py'
FORMAT='experiments/EXP-034-explicit-format-alignment/code/prepare_alignment.py'
DEPENDENCIES=[CORE,FORMAT,BASE,'experiments/EXP-034-explicit-format-alignment/prompts.jsonl',
 'experiments/EXP-034-explicit-format-alignment/config.json','experiments/EXP-034-explicit-format-alignment/pilot/config.json',
 'experiments/EXP-021-codex-subscription-pilot/model-instructions.txt']
MODELS={'gpt-5.6-luna','gpt-5.6-terra','gpt-5.6-sol','gpt-6-astra'}

def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(data):return hashlib.sha256(data).hexdigest()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def project_path(root,relative):
    p=(root/relative).resolve()
    if not p.is_relative_to(root.resolve()):raise ValueError('Path outside project')
    return p
def pid_alive(pid):
    if type(pid) is not int or pid<=0:raise ValueError('Invalid prior PID')
    if os.name=='nt':
        command=f'if (Get-Process -Id {pid} -ErrorAction SilentlyContinue) {{ "ALIVE" }} else {{ "EXITED" }}'
        r=subprocess.run(['powershell.exe','-NoProfile','-Command',command],capture_output=True,text=True,timeout=15,
                         creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if r.returncode!=0 or r.stdout.strip() not in {'ALIVE','EXITED'}:raise ValueError('Prior process status unknown')
        return r.stdout.strip()=='ALIVE'
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False
    except PermissionError:return True

def verify_manifest(exp):
    root=exp.parents[1];m=load(exp/'manifest.json');entries=[m['config']]+m.get('inputs',[])+m.get('code',[])
    entries += [e for e in m.get('outputs',[]) if e['path'].endswith(('/prompts.jsonl','/request-plan.jsonl'))]
    bound={}
    for e in entries:
        p=project_path(root,e['path'])
        if sha(p.read_bytes())!=e['sha256']:raise ValueError('Frozen hash mismatch: '+e['path'])
        if e['path'] in bound and bound[e['path']]!=e['sha256']:raise ValueError('Conflicting frozen hashes')
        bound[e['path']]=e['sha256']
    required=DEPENDENCIES+[(exp/p).relative_to(root).as_posix() for p in ['config.json','prompts.jsonl','request-plan.jsonl','recovery-receipt.json','code/run_batch.py']]
    if any(p not in bound for p in required):raise ValueError('Manifest lacks required source/runtime bindings')
    if bound[required[-1]]!=sha(Path(__file__).read_bytes()):raise ValueError('Executing wrapper not frozen')
    return m,bound

def read_history(exp,codex,core,base,bound):
    root=exp.parents[1];receipt=load(exp/'recovery-receipt.json');config=load(exp/'config.json')
    source=project_path(root,receipt['prior_experiment'])
    if config.get('source_experiment')!=receipt['prior_experiment']:raise ValueError('Prior experiment changed')
    for p in [source/'config.json',source/'prompts.jsonl',source/'request-plan.jsonl']:
        if p.relative_to(root).as_posix() not in bound:raise ValueError('Original generation plan is not frozen')
    prepared=base.validate_selection(source,codex,core)
    _,original_config,instructions,workspace,original_binding,jobs=prepared
    if (exp/'prompts.jsonl').read_bytes()!=(source/'prompts.jsonl').read_bytes() or (exp/'request-plan.jsonl').read_bytes()!=(source/'request-plan.jsonl').read_bytes():
        raise ValueError('Continuation inputs differ from original')
    state_path=project_path(root,receipt['prior_state_path']);records_path=project_path(root,receipt['prior_records_path'])
    for p in [state_path,records_path]:
        if p.relative_to(root).as_posix() not in bound:raise ValueError('Prior terminal evidence not frozen')
    state=load(state_path);rows=core.read_jsonl(records_path);saved={r['job_key'] for r in rows}
    deadline=datetime.fromisoformat(receipt['original_wall_deadline_utc'].replace('Z','+00:00'))
    original_start=datetime.fromisoformat(state['started_at'].replace('Z','+00:00'))
    if deadline.tzinfo is None or original_start.tzinfo is None or deadline!=original_start+timedelta(seconds=21600):
        raise ValueError('Original wall deadline changed')
    if len(rows)!=receipt['saved_count'] or len(saved)!=len(rows) or saved!=set(receipt['saved_job_keys']):raise ValueError('Inherited success count mismatch')
    if state.get('status')!='LOCAL_ERROR' or state.get('active') or state['controller_pid']!=receipt['prior_controller_pid'] or state['saved_total']!=len(rows):
        raise ValueError('Old controller has not reached expected drained state')
    gate=receipt['gate_job_key'];gate_path=project_path(root,receipt['gate_attempt_path'])
    if gate in saved or gate not in {j['key'] for j in jobs}:raise ValueError('Invalid unsent gate job')
    if gate_path!=source/'collection/attempts'/gate or not gate_path.is_dir():raise ValueError('Gate path changed')
    if {p.name for p in gate_path.iterdir()}!={'handover-gate.json'}:raise ValueError('Gate may have dispatched a model; inspection required')
    gate_file=gate_path/'handover-gate.json'
    if gate_file.relative_to(root).as_posix() not in bound:raise ValueError('Gate proof not frozen')
    gate_meta=load(gate_file)
    if gate_meta.get('job_key',gate_meta.get('gate_job_key'))!=gate:raise ValueError('Gate identity mismatch')
    if len(state.get('failures',[]))!=1 or state['failures'][0].get('job_key')!=gate or state['failures'][0].get('reasons')!=['CONTROLLER_FileExistsError']:
        raise ValueError('Unexpected historical failure; do not replay')
    if not state.get('local_errors') or any(e.get('job_key')!=gate or e.get('error_type')!='FileExistsError' for e in state['local_errors']):
        raise ValueError('Unexpected historical local error')
    attempts=source/'collection/attempts'
    if {p.name for p in attempts.iterdir()}!=saved|{gate}:raise ValueError('Historical attempts do not match successes plus unsent gate')
    planned={j['key']:j for j in jobs}
    for row in rows:
        p=attempts/row['job_key'];job=planned[row['job_key']]
        for f in p.iterdir():
            if f.is_file() and f.relative_to(root).as_posix() not in bound:raise ValueError('Historical artifact not frozen')
        if load(p/'record.json')!=row or load(p/'result.json')['status']!='SUCCEEDED':raise ValueError('Inherited response lacks success evidence')
        if row['binding']!=original_binding or row['model_requested']!=job['model']['slug'] or any(row.get(k)!=v for k,v in job['prompt'].items() if k!='source'):
            raise ValueError('Inherited response identity changed')
        if (p/'final.txt').read_bytes()!=row['text'].encode('utf-8') or sha(row['text'].encode('utf-8'))!=row['text_sha256']:
            raise ValueError('Inherited successful raw text changed')
        request=load(p/'request.json')
        if request['job_key']!=row['job_key'] or request['prompt']!=job['prompt'] or request['model']!=job['model']:raise ValueError('Inherited request changed')
    return {'receipt':receipt,'source':source,'rows':rows,'saved':saved,'jobs':jobs,'original_binding':original_binding,
            'config':original_config,'instructions':instructions,'workspace':workspace,'state':state,'deadline':deadline}

def translated(state,history):
    result=dict(state)
    result.update(saved_new_total=state['saved_total'],saved_total=len(history['rows'])+state['saved_total'],
                  target_new_total=state['target_total'],target_total=300,planned_total=300,
                  already_saved_new=state.get('already_saved',0),already_saved=len(history['rows'])+state.get('already_saved',0),
                  saved_in_target=len(history['rows'])+state.get('saved_in_target',state['saved_total']),
                  inherited_saved=len(history['rows']),workers=2,combined_collection_concurrency_cap=3)
    result['original_wall_deadline_utc']=history['receipt']['original_wall_deadline_utc']
    return result

def audit_collection(exp,core,base,checker,state,history,binding):
    _,bound=verify_manifest(exp);read_history(exp,Path(binding['codex_path']),core,base,bound)
    collection=exp/'collection';new=core.read_jsonl(collection/'records.jsonl');allrows=history['rows']+new
    planned={j['key']:j for j in history['jobs']};keys=[r['job_key'] for r in allrows];errors=[];counts=Counter();splits=Counter();flags=Counter();events=[]
    if len(keys)!=len(set(keys)):errors.append('DUPLICATE_SUCCESSFUL_JOB_KEYS')
    for source,rows,expected_binding in [(history['source'],history['rows'],history['original_binding']),(exp,new,binding)]:
        for row in rows:
            try:
                job=planned[row['job_key']];attempt=source/'collection/attempts'/row['job_key'];result=load(attempt/'result.json');request=load(attempt/'request.json')
                if row['binding']!=expected_binding or row['model_requested']!=job['model']['slug'] or any(row.get(k)!=v for k,v in job['prompt'].items() if k!='source'):errors.append('ROW_IDENTITY')
                if request['job_key']!=row['job_key'] or request['prompt']!=job['prompt'] or request['model']!=job['model'] or load(attempt/'record.json')!=row or result['status']!='SUCCEEDED':errors.append('ARTIFACT_IDENTITY')
                raw=(attempt/'final.txt').read_bytes()
                if raw!=row['text'].encode('utf-8') or sha(raw)!=row['text_sha256'] or sha(raw)!=row['raw_final_sha256']:errors.append('TEXT_HASH')
                check=base.inspect_saved(core,core.inspect_response,checker,attempt,{'returncode':result['returncode']},row['model_requested'])
                if not check['success'] or sorted(check['flags'])!=sorted(row['flags']):errors.append('RAW_COMPLETION')
                counts[row['model_requested']]+=1;splits[row['split']]+=1;flags.update(row['flags'])
            except Exception as exc:errors.append('ROW_'+type(exc).__name__)
    statuses=Counter();attempts=list((collection/'attempts').iterdir())
    for attempt in attempts:
        if not attempt.is_dir() or not (attempt/'result.json').exists() or not (attempt/'request.json').exists():errors.append('INCOMPLETE_ATTEMPT');continue
        r=load(attempt/'result.json');q=load(attempt/'request.json');statuses[r['status']]+=1
        if attempt.name not in planned or attempt.name in history['saved'] or r['job_key']!=attempt.name or q['job_key']!=attempt.name:errors.append('ATTEMPT_IDENTITY')
        if r.get('started_at') and r.get('ended_at'):
            events.extend([(datetime.fromisoformat(r['started_at']),1),(datetime.fromisoformat(r['ended_at']),-1)])
    current=peak=0
    for _,delta in sorted(events,key=lambda item:(item[0],item[1])):current+=delta;peak=max(peak,current)
    if peak>2:errors.append('CONCURRENCY_EXCEEDED')
    for name in ['state.json','summary.json','run-'+state['run_id']+'.json']:
        if load(collection/name).get('run_id')!=state['run_id']:errors.append('RUN_ID_MISMATCH')
    remaining=300-len(history['rows']);complete=(state['status']=='COMPLETE' and len(allrows)==300 and set(keys)==set(planned)
        and dict(counts)=={m:75 for m in MODELS} and dict(splits)=={'train':240,'holdout':60}
        and len({r['prompt_group'] for r in allrows})==75 and len(attempts)==remaining and statuses=={'SUCCEEDED':remaining}
        and not errors and not state['active'] and not state.get('failures') and not state.get('local_errors'))
    audit={'observed_at':core.utc_now(),'run_id':state['run_id'],'status':'COLLECTED_UNVALIDATED' if complete else 'INCOMPLETE',
      'responses':len(allrows),'new_responses':len(new),'inherited_responses':len(history['rows']),'by_model':dict(counts),'by_split':dict(splits),
      'prompt_groups':len({r['prompt_group'] for r in allrows}),'duplicate_job_keys':len(keys)-len(set(keys)),
      'duplicate_text_rows':len(allrows)-len({r['text_sha256'] for r in allrows}),'attempts_by_status':dict(statuses),'integrity_errors':errors,
      'workers':2,'peak_new_attempt_concurrency':peak,'combined_collection_cap':3,'gate_model_dispatches':0,
      'model_observation_scopes':dict(Counter(r['model_observation_scope'] for r in allrows)),'flags':dict(flags),
      'training_acceptance':'NOT_EVALUATED','bare_api_equivalence':'NOT_ESTABLISHED'}
    core.write_json(exp/'metrics/collection-audit.json',audit)
    with (collection/'combined-records.jsonl').open('w',encoding='utf-8',newline='\n') as f:
        for source,rows in [(history['source'],history['rows']),(exp,new)]:
            for row in rows:f.write(json.dumps(dict(row,origin_experiment=source.name,attempt_path_project=(source/row['attempt_path']).relative_to(exp.parents[1]).as_posix()),ensure_ascii=False)+'\n')
    return audit

def run_batch(experiment=EXP,*,run=False,codex=None,workers=2,max_seconds=None,runner=None,pid_checker=pid_alive):
    if workers!=2:raise ValueError('This continuation requires exactly2 workers; companion is limited to1')
    exp=Path(experiment).resolve();root=exp.parents[1];manifest,bound=verify_manifest(exp);config=load(exp/'config.json')
    if config.get('workers')!=2:raise ValueError('Configured worker cap must be2')
    budget=config.get('max_seconds',config.get('budget',{}).get('wall_seconds',21600))
    max_seconds=budget if max_seconds is None else max_seconds
    if not 0<max_seconds<=min(21600,budget):raise ValueError('Wall budget exceeds authorization')
    core=module('handoff_core',root/CORE);base=module('frozen_batch39',root/BASE);fmt=module('frozen_format',root/FORMAT)
    codex=Path(codex or load(exp/'metrics/execution-preflight.json')['codex_executable_path'])
    history=read_history(exp,codex,core,base,bound)
    seconds_left=(history['deadline']-datetime.now(timezone.utc)).total_seconds()
    if seconds_left<=0:raise ValueError('Original wall deadline expired')
    max_seconds=min(max_seconds,seconds_left)
    binding=dict(history['original_binding'],execution_experiment=exp.relative_to(root).as_posix(),
                 execution_config_sha256=sha((exp/'config.json').read_bytes()),execution_manifest_sha256=sha((exp/'manifest.json').read_bytes()))
    remaining=[j for j in history['jobs'] if j['key'] not in history['saved']]
    def prepare_override(unused,codex_arg):return exp,history['config'],history['instructions'],history['workspace'],binding,remaining
    original_prepare=core.prepare
    core.prepare=prepare_override
    if not run:
        result=core.run_collection(exp,codex,limit=300,workers=2,run=False,max_seconds=max_seconds,runner=runner)
        return dict(result,target_total=300,remaining_in_this_run=result['remaining_in_this_run'],already_saved=len(history['rows'])+result['already_saved'],inherited_saved=len(history['rows']),workers=2,combined_collection_cap=3)
    if config.get('authorization',{}).get('generation') is not True or manifest.get('generation_authorized') is not True:raise ValueError('Generation authorization missing')
    base.check_preflight(exp,codex)
    if pid_checker(history['receipt']['prior_controller_pid']):raise ValueError('Old controller still exists; wait for handoff')
    if any((history['source']/'collection'/n).exists() for n in ['batch.lock','run.lock']):raise ValueError('Old collection lock remains')
    collection=exp/'collection';collection.mkdir(exist_ok=True)
    original_write,original_inspect=core.write_json,core.inspect_response
    def guarded_write(path,value,exclusive=False):
        if path.parent==collection and (path.name in ['state.json','summary.json'] or path.name.startswith('run-')) and isinstance(value,dict) and 'saved_total' in value:
            value=translated(value,history)
            if value.get('status')=='COMPLETE':value=dict(value,status='FINALIZING')
        return original_write(path,value,exclusive)
    def inspect(attempt,outcome,model):return base.inspect_saved(core,original_inspect,fmt.format_flags,attempt,outcome,model)
    with core.RunLock(collection/'batch.lock'):
        if pid_checker(history['receipt']['prior_controller_pid']):raise ValueError('Old controller still exists')
        if any((history['source']/'collection'/n).exists() for n in ['batch.lock','run.lock']):raise ValueError('Old collection lock remains')
        core.write_json,core.inspect_response=guarded_write,inspect
        try:state=core.run_collection(exp,codex,limit=300,workers=2,run=True,max_seconds=max_seconds,runner=runner)
        finally:core.write_json,core.inspect_response,core.prepare=original_write,original_inspect,original_prepare
        try:
            audit=audit_collection(exp,core,base,fmt.format_flags,state,history,binding)
            state.update(audit_status=audit['status'],audit_run_id=audit['run_id'])
            if state['status']=='COMPLETE' and (audit['status']!='COLLECTED_UNVALIDATED' or audit['run_id']!=state['run_id']):state['status']='AUDIT_FAILED'
        except Exception as exc:state.update(status='AUDIT_FAILED',audit_status='ERROR',audit_error_type=type(exc).__name__)
        state=translated(state,history)
        for name in ['state.json','summary.json','run-'+state['run_id']+'.json']:
            state['observed_at']=core.utc_now();original_write(collection/name,state)
        return state

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment','--exp',type=Path,default=EXP);p.add_argument('--codex',type=Path)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--max-seconds',type=float)
    modes=p.add_mutually_exclusive_group();modes.add_argument('--run',action='store_true');modes.add_argument('--dry-run',action='store_true');a=p.parse_args()
    try:
        result=run_batch(a.experiment,run=a.run,codex=a.codex,workers=a.workers,max_seconds=a.max_seconds)
        print(json.dumps(result,ensure_ascii=False,indent=2));return 0 if result['status'] in ['DRY_RUN','COMPLETE'] else 2
    except Exception as exc:
        print(json.dumps({'status':'NOT_STARTED','error_type':type(exc).__name__,'message':str(exc) if isinstance(exc,(ValueError,FileNotFoundError)) else 'Inspect local evidence; no service body printed'},ensure_ascii=False));return 2

if __name__=='__main__':raise SystemExit(main())
