"""Bounded, repeat-aware paired collection. Default mode is offline dry-run."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import threading
import time
import uuid


EXP = Path(__file__).resolve().parents[1]
MODELS = {'gpt-5.6-luna', 'gpt-5.6-terra'}
ROUTES = {'codex_subscription', 'kaggle_model_proxy'}
RUNTIME_FILES = [
    'experiments/EXP-030-codex-subscription-pilot-r3/code/codex_collect.py',
    'experiments/EXP-034-explicit-format-alignment/code/prepare_alignment.py',
    'experiments/EXP-034-explicit-format-alignment/pilot/config.json',
    'experiments/EXP-021-codex-subscription-pilot/model-instructions.txt',
]


class InspectionRequired(RuntimeError):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def write_json(path, value, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')
    target = path if exclusive else path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with target.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    if not exclusive:
        for delay in (0.05, 0.1, 0.2, 0.4, 0.8, None):
            try:
                os.replace(target, path)
                break
            except PermissionError:
                if delay is None:
                    raise
                time.sleep(delay)


def append_jsonl(path, value):
    with path.open('ab') as stream:
        stream.write((json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n').encode('utf-8'))
        stream.flush()
        os.fsync(stream.fileno())


class RunLock:
    def __init__(self, path):
        self.path, self.token = path, uuid.uuid4().hex

    def __enter__(self):
        try:
            write_json(self.path, {'controller_pid':os.getpid(), 'token':self.token, 'created_at':utc_now()}, True)
        except FileExistsError as exc:
            raise InspectionRequired('Collection lock exists; do not start another controller') from exc
        return self

    def __exit__(self, *unused):
        try:
            item = load(self.path)
            if item.get('controller_pid') == os.getpid() and item.get('token') == self.token:
                self.path.unlink()
        except (FileNotFoundError, ValueError):
            pass


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decimal_value(value):
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError('Invalid monetary value') from exc
    if not result.is_finite() or result < 0:
        raise ValueError('Monetary values must be finite and nonnegative')
    return result


def verify_manifest(experiment):
    root = experiment.parents[1]
    manifest = load(experiment/'manifest.json')
    entries = [manifest['config']]
    for name in ('inputs', 'code', 'outputs'):
        entries.extend(manifest.get(name, []))
    hashes = {}
    # Reports/metrics are updated at the end; input/config/code/request-plan bindings are immutable.
    for entry in entries:
        relative = entry['path']
        path = (root/relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Manifest path outside project root')
        if relative in hashes and hashes[relative] != entry['sha256']:
            raise ValueError('Conflicting manifest hashes')
        hashes[relative] = entry['sha256']
        if entry in manifest.get('outputs', []) and path != experiment/'request-plan.jsonl':
            continue
        if sha_bytes(path.read_bytes()) != entry['sha256']:
            raise ValueError('Frozen file hash mismatch: '+relative)
    required = RUNTIME_FILES + [
        (experiment/'config.json').relative_to(root).as_posix(),
        (experiment/'request-plan.jsonl').relative_to(root).as_posix(),
        (experiment/'code/run_trial.py').relative_to(root).as_posix(),
        (experiment/'code/kaggle_transport.py').relative_to(root).as_posix(),
    ]
    if any(path not in hashes for path in required):
        raise ValueError('Manifest must bind controller, transport, runtime and request plan')
    if hashes[required[-2]] != sha_bytes(Path(__file__).read_bytes()):
        raise ValueError('Executing controller does not match frozen controller')
    return manifest, {path:hashes[path] for path in required}


def validate_plan(config, jobs):
    if len(jobs) != 80 or len({r['job_id'] for r in jobs}) != 80:
        raise ValueError('Plan must contain exactly 80 unique requests')
    if config['design']['total_new_requests'] != 80 or config['design']['global_max_inflight'] != 1:
        raise ValueError('Authorized trial is 80 requests, globally serial')
    if config['design']['repeats_per_cell_per_route'] != 2:
        raise ValueError('Authorized repeat count is two')
    if set(config['design']['models']) != MODELS or set(config['design']['routes']) != ROUTES:
        raise ValueError('Unexpected models or routes')
    budget = config['proposed_budget']
    if decimal_value(budget['stop_dispatch_at_confirmed_cost_usd']) > Decimal('0.10'):
        raise ValueError('Stop threshold exceeds authorization')
    if decimal_value(budget['target_usd']) > Decimal('0.25'):
        raise ValueError('Budget target exceeds authorization')
    if budget['maximum_kaggle_requests'] != 40 or budget['maximum_codex_requests'] != 40:
        raise ValueError('Route dispatch caps must remain 40')
    cells, groups = {}, set()
    for index, job in enumerate(jobs, 1):
        if job['request_index'] != index or job['route'] not in ROUTES or job['model_label'] not in MODELS:
            raise ValueError('Invalid request sequence/model/route')
        if job['repeat_id'] not in (1,2) or job['split'] != 'train' or job['prompt_group'] != job['prompt_id']:
            raise ValueError('Invalid repeat or prompt group')
        if job['reasoning_effort'] != 'low' or job['base_instruction'] != "Answer the user's request.":
            raise ValueError('Sampling/base instructions changed')
        if sha_bytes(job['user_message'].encode('utf-8')) != job['user_message_sha256']:
            raise ValueError('User-message hash mismatch')
        if not re.fullmatch('[a-f0-9]{64}', job['job_id']):
            raise ValueError('Unsafe request identity')
        key = f"EXP-035|{job['route']}|{job['prompt_id']}|{job['model_label']}|{job['protocol_version']}|{job['user_message_sha256']}|r{job['repeat_id']}"
        if sha_bytes(key.encode('utf-8')) != job['job_id']:
            raise ValueError('Repeat-aware identity mismatch')
        groups.add(job['prompt_group'])
        cells.setdefault((job['prompt_id'],job['model_label']), []).append(job)
        if job['route'] == 'kaggle_model_proxy':
            expected = {'model':'openai/'+job['model_label'],
                        'messages':[{'role':'system','content':job['base_instruction']},
                                    {'role':'user','content':job['user_message']}],
                        'reasoning_effort':'low','max_completion_tokens':1024,'stream':False,'n':1}
            if job['api_payload'] != expected:
                raise ValueError('API payload differs from frozen paired protocol')
    if len(groups) != 10 or len(cells) != 20:
        raise ValueError('Expected 10 prompt groups and 20 model/prompt cells')
    for cell in cells.values():
        if len(cell)!=4 or len({r['user_message_sha256'] for r in cell})!=1:
            raise ValueError('Mismatched paired user messages')
        if {(r['route'],r['repeat_id']) for r in cell} != {(route,rep) for route in ROUTES for rep in (1,2)}:
            raise ValueError('Missing route/repeat combination')


def check_preflight(experiment, codex_path):
    pre = load(experiment/'metrics/execution-preflight.json')
    now = datetime.now(timezone.utc)
    observed = datetime.fromisoformat(pre['observed_at'].replace('Z','+00:00'))
    expires = datetime.fromisoformat(pre['expires_at'].replace('Z','+00:00'))
    if observed.tzinfo is None or expires.tzinfo is None:
        raise ValueError('Preflight timestamps require timezone')
    if pre.get('status')!='PASS' or not observed <= now < expires or (now-observed).total_seconds()>1800:
        raise ValueError('Preflight failed or expired')
    if pre.get('free_quota_only') is not True or decimal_value(pre['free_quota_remaining_usd'])<Decimal('0.25'):
        raise ValueError('Insufficient verified free allowance')
    if not {'openai/'+m for m in MODELS} <= set(pre['models_available']):
        raise ValueError('Required model routes not listed')
    if pre.get('requested_reasoning_effort') != 'low':
        raise ValueError('Requested low setting missing from preflight')
    if Path(pre['codex_executable_path']).resolve()!=codex_path.resolve():
        raise ValueError('Codex executable path changed')
    if sha_bytes(codex_path.read_bytes()) != pre['codex_executable_sha256']:
        raise ValueError('Codex executable changed')
    return pre


def api_cost(raw):
    try:
        value = raw['usage']['cost']
        parts = [value['input_tokens_cost_nanodollars'], value['output_tokens_cost_nanodollars']]
    except (KeyError, TypeError) as exc:
        raise ValueError('MISSING_COST') from exc
    if any(type(part) is not int or part < 0 for part in parts):
        raise ValueError('INVALID_COST')
    total = sum(parts)
    for key, component in value.items():
        if key in {'input_tokens_cost_nanodollars','output_tokens_cost_nanodollars'}:
            continue
        if key == 'total_cost_nanodollars':
            if type(component) is not int or component != total:
                raise ValueError('TOTAL_COST_MISMATCH')
        elif type(component) is not int or component != 0:
            raise ValueError('UNKNOWN_COST_COMPONENT')
    return total, Decimal(total)/Decimal(1_000_000_000)


def inspect_api(raw, job):
    reasons, flags = [], []
    try:
        nanos, cost = api_cost(raw)
    except ValueError as exc:
        nanos, cost = None, None
        reasons.append(str(exc))
    model = raw.get('model')
    if model not in (job['model_label'],'openai/'+job['model_label']):
        reasons.append('MODEL_IDENTIFIER_MISMATCH')
    choices = raw.get('choices')
    text, finish, refusal = '', None, None
    if not isinstance(choices,list) or len(choices)!=1:
        reasons.append('INVALID_CHOICES')
    else:
        choice=choices[0]
        finish=choice.get('finish_reason')
        message=choice.get('message') or {}
        content=message.get('content')
        refusal=message.get('refusal')
        if message.get('tool_calls') or message.get('function_call'):
            reasons.append('TOOL_USE')
        if isinstance(content,str):
            text=content
        elif content is not None:
            reasons.append('INVALID_CONTENT')
        if not text.strip() and isinstance(refusal,str) and refusal.strip():
            text=refusal
            flags.append('text_from_refusal_field')
        if refusal:
            flags.append('refusal')
        if finish!='stop':
            reasons.append('TRUNCATED' if finish=='length' else 'UNKNOWN_FINISH_REASON')
    if not text.strip():
        reasons.append('EMPTY_FINAL')
    response_id=raw.get('id')
    if not isinstance(response_id,str) or not response_id:
        reasons.append('MISSING_PROVIDER_RESPONSE_ID')
    return {'success':not reasons,'reasons':reasons,'flags':flags,'text':text,
            'text_sha256':sha_bytes(text.encode('utf-8')),'cost_nanodollars':nanos,
            'cost_usd':str(cost) if cost is not None else None,'usage':raw.get('usage'),
            'model_observed':[model] if model else [],'model_observation_scope':'returned_alias',
            'provider_response_id':response_id,'finish_reason':finish,'backend_snapshot':'unknown'}


class LiveCodex:
    def __init__(self, root, codex_path):
        self.root, self.codex_path = root, codex_path
        self.collector=import_file('frozen_codex_runtime',root/RUNTIME_FILES[0])
        self.runtime_config=load(root/RUNTIME_FILES[2])
        runtime_exp=(root/RUNTIME_FILES[2]).parent
        self.instructions=(runtime_exp/self.runtime_config['instructions_file']).resolve()
        self.workspace=(runtime_exp/self.runtime_config['workspace']).resolve()
        if self.instructions.read_text(encoding='utf-8').strip()!="Answer the user's request.":
            raise ValueError('Base instruction mismatch')
        self.runner=self.collector.SubprocessRunner()

    def __call__(self, job, attempt):
        command=[str(self.codex_path),'exec','--ignore-user-config','--ephemeral','--json',
                 '--skip-git-repo-check','-s','read-only','-m',job['model_label'],
                 '-c','approval_policy="never"','-c','model_reasoning_effort="low"',
                 '-c','model_instructions_file='+json.dumps(str(self.instructions)),
                 '-o',str(attempt/'final.txt')]
        for override in self.runtime_config['cli_overrides']:
            command+=['-c',override]
        command+=['-']
        write_json(attempt/'command.json',{'command':command,'cwd':str(self.workspace)},True)
        stop, cancel=threading.Event(), threading.Event()
        outcome=self.runner(command,job['user_message'],attempt,180,stop,cancel,str(self.workspace))
        response=self.collector.inspect_response(attempt,outcome,job['model_label'])
        response.update(model_observed=response.pop('observed_models'),cost_usd=None,cost_nanodollars=None,
                        finish_reason='turn.completed' if response['success'] else None,
                        backend_snapshot='unknown',returncode=outcome['returncode'])
        response['model_observation_scope']='cli_event' if response['model_observed'] else 'requested_only'
        events=read_jsonl(attempt/'raw_stdout.jsonl')
        thread_ids=[r.get('thread_id') for r in events if r.get('type')=='thread.started']
        response['provider_response_id']=thread_ids[0] if len(thread_ids)==1 else None
        if sum(e.get('type')=='turn.completed' for e in events)!=1:
            response['reasons'].append('COMPLETION_COUNT_MISMATCH')
            response['success']=False
        return response


def previous(collection, jobs, binding):
    if (collection/'binding.json').exists() and load(collection/'binding.json') != binding:
        raise InspectionRequired('Collection inputs/runtime changed')
    rows=read_jsonl(collection/'records.jsonl')
    by_id={r['job_id']:r for r in jobs}
    seen=set()
    for row in rows:
        ident=row['job_id']
        if ident in seen or ident not in by_id:
            raise InspectionRequired('Duplicate or unknown saved job')
        job=by_id[ident]
        if any(row[k]!=job[k] for k in ['route','prompt_id','prompt_group','model_label','repeat_id','user_message_sha256']):
            raise InspectionRequired('Saved record identity changed')
        if sha_bytes(row['text'].encode('utf-8'))!=row['text_sha256']:
            raise InspectionRequired('Saved text hash mismatch')
        seen.add(ident)
    attempts=collection/'attempts'
    if attempts.exists():
        for path in attempts.iterdir():
            if not path.is_dir() or not (path/'result.json').exists():
                raise InspectionRequired('Unfinished attempt exists; do not replay')
            result=load(path/'result.json')
            if path.name not in seen or result.get('job_id')!=path.name or result.get('status')!='SUCCEEDED':
                raise InspectionRequired('Failed/unknown/uncommitted attempt requires inspection')
    for row in rows:
        attempt=attempts/row['job_id']
        if not (attempt/'result.json').exists() or not (attempt/'record.json').exists():
            raise InspectionRequired('Saved attempt artifacts missing')
        if load(attempt/'record.json')!=row or (attempt/'final.txt').read_bytes()!=row['text'].encode('utf-8'):
            raise InspectionRequired('Saved artifact differs from index')
        if row['route']=='kaggle_model_proxy':
            nanos,cost=api_cost(load(attempt/'raw_response.json'))
            if nanos!=row['cost_nanodollars'] or cost!=decimal_value(row['cost_usd']):
                raise InspectionRequired('Saved billing evidence changed')
    return rows,seen


def audit_collection(experiment, jobs, state):
    collection=experiment/'collection'
    errors=[]
    try:
        rows=read_jsonl(collection/'records.jsonl')
    except Exception as exc:
        rows=[]
        errors.append('RECORDS_'+type(exc).__name__)
    expected={r['job_id']:r for r in jobs}
    counts=Counter()
    billed=Decimal(0)
    for row in rows:
        try:
            job=expected[row['job_id']]
            path=collection/'attempts'/row['job_id']
            if sha_bytes(row['text'].encode('utf-8'))!=row['text_sha256'] or (path/'final.txt').read_bytes()!=row['text'].encode('utf-8'):
                errors.append('TEXT_HASH:'+row['job_id'])
            if load(path/'result.json')['status']!='SUCCEEDED':
                errors.append('RESULT:'+row['job_id'])
            if any(row[k]!=job[k] for k in ['route','prompt_group','model_label','repeat_id','user_message_sha256']):
                errors.append('IDENTITY:'+row['job_id'])
            if row['route']=='kaggle_model_proxy':
                nanos,cost=api_cost(load(path/'raw_response.json'))
                if nanos!=row['cost_nanodollars'] or cost!=decimal_value(row['cost_usd']):
                    errors.append('BILLING:'+row['job_id'])
                billed+=cost
            counts[row['route']+'|'+row['model_label']]+=1
        except Exception as exc:
            errors.append('ROW_'+type(exc).__name__)
    duplicate_jobs=len(rows)-len({r['job_id'] for r in rows})
    results=[load(p) for p in (collection/'attempts').glob('*/result.json')]
    statuses=Counter(r['status'] for r in results)
    attempted_dirs=list((collection/'attempts').iterdir())
    if state['status']=='COMPLETE' and billed!=decimal_value(state['confirmed_cost_usd']):
        errors.append('TOTAL_BILLING_MISMATCH')
    complete=(state['status']=='COMPLETE' and state['cost_known'] and len(rows)==80 and not errors and not duplicate_jobs
              and len(counts)==4 and set(counts.values())=={20}
              and statuses=={'SUCCEEDED':80} and len(attempted_dirs)==80 and not state['active'])
    audit={'observed_at':utc_now(),'run_id':state['run_id'],'status':'COLLECTED_UNVALIDATED' if complete else 'INCOMPLETE',
           'responses':len(rows),'counts':dict(counts),'duplicate_job_ids':duplicate_jobs,
           'identical_text_extra_rows':len(rows)-len({r['text_sha256'] for r in rows}),
           'attempts_by_status':dict(statuses),'integrity_errors':errors,
           'confirmed_cost_usd':state['confirmed_cost_usd'],'cost_known':state['cost_known'],
           'raw_repeats_preserved':True,'training_or_api_equivalence':'NOT_EVALUATED'}
    write_json(experiment/'metrics/collection-audit.json',audit)
    return audit


def run_trial(experiment=EXP, *, run=False, codex_path=None, max_jobs=80, max_seconds=7200,
              kaggle_generate=None, kaggle_make_context=None, kaggle_preflight=None,
              codex_call=None, format_checker=None):
    experiment=Path(experiment).resolve()
    root=experiment.parents[1]
    config=load(experiment/'config.json')
    jobs=read_jsonl(experiment/'request-plan.jsonl')
    validate_plan(config,jobs)
    manifest,binding=verify_manifest(experiment)
    if not 1<=max_jobs<=80 or max_seconds<=0:
        raise ValueError('Invalid execution bound')
    collection=experiment/'collection'
    rows,seen=previous(collection,jobs,binding)
    plan={'status':'DRY_RUN','target_total':80,'already_saved':len(rows),'remaining':80-len(seen),
          'max_new_jobs_this_run':max_jobs,'global_max_inflight':1,'requests_sent':0}
    if not run:
        return plan
    if config.get('authorization',{}).get('generation') is not True or manifest.get('generation_authorized') is not True:
        raise ValueError('User generation authorization is not recorded')
    if not all(job.get('execution_authorized') is True for job in jobs):
        raise ValueError('Request plan remains preview-only')
    codex_path=Path(codex_path or load(experiment/'metrics/execution-preflight.json')['codex_executable_path'])
    preflight=check_preflight(experiment,codex_path)
    format_checker=format_checker or import_file('aligned_format_checker',root/RUNTIME_FILES[1]).format_flags
    if codex_call is None:
        codex_call=LiveCodex(root,codex_path)
    if kaggle_generate is None or kaggle_make_context is None or kaggle_preflight is None:
        transport=import_file('paired_kaggle_transport',experiment/'code/kaggle_transport.py')
        kaggle_generate=kaggle_generate or transport.generate
        kaggle_make_context=kaggle_make_context or transport.make_client
        kaggle_preflight=kaggle_preflight or transport.preflight
    collection.mkdir(exist_ok=True)
    with RunLock(collection/'run.lock'):
        rows,seen=previous(collection,jobs,binding)
        if not (collection/'binding.json').exists():
            write_json(collection/'binding.json',binding,True)
        (collection/'attempts').mkdir(exist_ok=True)
        context=None
        mutex=threading.RLock()
        done=threading.Event()
        halt=threading.Event()
        spent=sum((decimal_value(r['cost_usd']) for r in rows if r['route']=='kaggle_model_proxy'),Decimal(0))
        route_sent=Counter(r['route'] for r in rows)
        seen_response_ids={(r['route'],r['provider_response_id']) for r in rows if r.get('provider_response_id')}
        run_id=uuid.uuid4().hex
        state={'status':'RUNNING','phase':'PREFLIGHT','run_id':run_id,'controller_pid':os.getpid(),'started_at':utc_now(),
               'observed_at':utc_now(),'target_total':80,'saved_total':len(rows),'saved_this_run':0,
               'already_saved':len(rows),'launched_this_run':0,'requests_sent_total':len(rows),
               'requests_by_route':dict(route_sent),'active':[],'failures':[],'local_errors':[],
               'failed_this_run':0,'confirmed_cost_usd':str(spent),'cost_known':True,
               'global_max_inflight':1,'preflight_observed_at':preflight['observed_at']}
        started=time.monotonic()

        def snapshot():
            with mutex:
                state.update(observed_at=utc_now(),elapsed_seconds=round(time.monotonic()-started,3))
                write_json(collection/'state.json',state)

        def local_error(stage,exc):
            with mutex:
                state['local_errors'].append({'stage':stage,'error_type':type(exc).__name__,'observed_at':utc_now()})
                state['status']='LOCAL_ERROR'
                halt.set()

        def heartbeat():
            while not done.wait(1):
                try:
                    snapshot()
                except OSError as exc:
                    local_error('heartbeat',exc)
                    break

        snapshot()
        thread=threading.Thread(target=heartbeat,daemon=True)
        thread.start()
        try:
            if any(job['job_id'] not in seen and job['route']=='kaggle_model_proxy' for job in jobs):
                context=kaggle_make_context()
                live=kaggle_preflight(context)
                write_json(collection/('live-preflight-'+run_id+'.json'),live,True)
                if live.get('ready') is not True:
                    raise ValueError('Live model proxy preflight not ready')
                if not {'openai/'+m for m in MODELS} <= set(live.get('matched_requested_models',live.get('listed_models',[]))):
                    raise ValueError('Live required model list incomplete')
                balances=live.get('quota',{})
                if not {'daily','monthly'} <= set(balances) or any(decimal_value(balances[key]['remaining_usd'])<Decimal('0.25') for key in ['daily','monthly']):
                    raise ValueError('Live verified free quota insufficient')
            state['phase']='COLLECTION'
            for job in jobs:
                if job['job_id'] in seen:
                    continue
                if halt.is_set():
                    break
                if state['launched_this_run']>=max_jobs:
                    state['status']='STOPPED_REQUEST_LIMIT'
                    break
                if time.monotonic()-started>=max_seconds:
                    state['status']='STOPPED_TIME_BUDGET'
                    break
                if not state['cost_known'] or spent>=decimal_value(config['proposed_budget']['stop_dispatch_at_confirmed_cost_usd']):
                    state['status']='STOPPED_COST_UNKNOWN' if not state['cost_known'] else 'STOPPED_BUDGET'
                    break
                if sum(route_sent.values())>=80 or route_sent[job['route']]>=40:
                    state['status']='STOPPED_REQUEST_LIMIT'
                    break
                attempt=collection/'attempts'/job['job_id']
                attempt.mkdir()
                begin=utc_now()
                write_json(attempt/'request.json',{'job':job,'started_at':begin,'run_id':run_id,
                                                  'controller_pid':os.getpid()},True)
                with mutex:
                    route_sent[job['route']]+=1
                    state['launched_this_run']+=1
                    state['requests_sent_total']=sum(route_sent.values())
                    state['requests_by_route']=dict(route_sent)
                    state['active']=[job['job_id']]
                snapshot()
                response=None
                try:
                    if job['route']=='kaggle_model_proxy':
                        raw=kaggle_generate(context,job['api_payload'])
                        if not isinstance(raw,dict):
                            raise ValueError('Transport must return raw response JSON')
                        write_json(attempt/'raw_response.json',raw,True)
                        response=inspect_api(raw,job)
                        with (attempt/'final.txt').open('xb') as stream:
                            stream.write(response['text'].encode('utf-8'))
                    else:
                        response=codex_call(job,attempt)
                    if response.get('provider_response_id'):
                        identity=(job['route'],response['provider_response_id'])
                        if identity in seen_response_ids:
                            response['success']=False
                            response['reasons'].append('DUPLICATE_PROVIDER_RESPONSE_ID')
                        seen_response_ids.add(identity)
                    if job['route']=='kaggle_model_proxy':
                        if response.get('cost_usd') is None:
                            state['cost_known']=False
                        else:
                            spent+=decimal_value(response['cost_usd'])
                            state['confirmed_cost_usd']=str(spent)
                    end=utc_now()
                    result={'job_id':job['job_id'],'route':job['route'],'status':'SUCCEEDED' if response['success'] else 'FAILED',
                            'started_at':begin,'ended_at':end,'reasons':response['reasons'],
                            'cost_usd':response.get('cost_usd'),'cost_nanodollars':response.get('cost_nanodollars'),
                            'text_sha256':response.get('text_sha256'),'provider_response_id':response.get('provider_response_id')}
                    record=None
                    if response['success']:
                        flags=list(response.get('flags',[]))+list(format_checker(response['text']))
                        words=len(re.findall(r'\S+',response['text']))
                        if not .75*job['target_words']<=words<=1.25*job['target_words']:
                            flags.append('target_length_deviation_over_25_percent')
                        record={k:job[k] for k in ['job_id','request_index','cell_id','route','repeat_id','prompt_id','prompt_group','split','model_label','protocol_version','target_words','user_message_sha256']}
                        record.update(source=job['route'],model_requested=job['model_label'],
                                      model_observed=response.get('model_observed',[]),
                                      model_observation_scope=response['model_observation_scope'],
                                      backend_snapshot='unknown',started_at=begin,ended_at=end,
                                      text=response['text'],text_sha256=response['text_sha256'],
                                      word_count=words,flags=sorted(set(flags)),usage=response.get('usage'),
                                      finish_reason=response.get('finish_reason'),provider_response_id=response.get('provider_response_id'),
                                      cost_usd=response.get('cost_usd'),cost_nanodollars=response.get('cost_nanodollars'),
                                      attempt_path=attempt.relative_to(experiment).as_posix(),binding=binding)
                        write_json(attempt/'record.json',record,True)
                    write_json(attempt/'result.json',result,True)
                    if record is not None:
                        append_jsonl(collection/'records.jsonl',record)
                        with mutex:
                            rows.append(record)
                            seen.add(job['job_id'])
                            state['saved_total']=len(rows)
                            state['saved_this_run']+=1
                    else:
                        state['failures'].append(result)
                        state['failed_this_run']+=1
                        state['status']='STOPPED_FAILURE'
                        halt.set()
                except Exception as exc:
                    if isinstance(exc,OSError):
                        local_error('attempt_persistence_or_io',exc)
                    else:
                        state['status']='STOPPED_FAILURE'
                        halt.set()
                    if job['route']=='kaggle_model_proxy' and (response is None or response.get('cost_usd') is None):
                        state['cost_known']=False
                    failure={'job_id':job['job_id'],'route':job['route'],'status':'UNCERTAIN',
                             'started_at':begin,'ended_at':utc_now(),'reasons':['EXCEPTION_'+type(exc).__name__]}
                    http_status=getattr(exc,'http_status',None)
                    if http_status is None:
                        http_status=getattr(exc,'status_code',None)
                    if type(http_status) is int:
                        failure['http_status']=http_status
                    for name in ['stage','error_type']:
                        safe=getattr(exc,name,None)
                        if isinstance(safe,str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,100}',safe):
                            failure[name]=safe
                    if not (attempt/'result.json').exists():
                        try:
                            write_json(attempt/'result.json',failure,True)
                        except OSError as store_exc:
                            local_error('failure_persistence',store_exc)
                    state['failures'].append(failure)
                    state['failed_this_run']+=1
                finally:
                    with mutex:
                        state['active']=[]
                snapshot()
            # Stop status publication before selecting a candidate terminal state;
            # COMPLETE must never appear in heartbeat output before final audit.
            done.set()
            thread.join(timeout=5)
            if state['status']=='RUNNING':
                state['status']='COMPLETE' if len(seen)==80 else 'STOPPED_SIGNAL'
        except KeyboardInterrupt:
            state['status']='INTERRUPTED'
            halt.set()
        except Exception as exc:
            if isinstance(exc,OSError):
                local_error('controller',exc)
            else:
                state['status']='STOPPED_PREFLIGHT_OR_CONTROLLER'
                failure={'status':'NOT_DISPATCHED','reasons':['EXCEPTION_'+type(exc).__name__]}
                for name in ['stage','error_type']:
                    safe=getattr(exc,name,None)
                    if isinstance(safe,str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,100}',safe):
                        failure[name]=safe
                safe_status=getattr(exc,'http_status',None)
                if type(safe_status) is int:
                    failure['http_status']=safe_status
                state['failures'].append(failure)
        finally:
            done.set()
            thread.join(timeout=5)
            with mutex:
                state['active']=[]
                state['phase']='TERMINAL'
                if state['local_errors']:
                    state['status']='LOCAL_ERROR'
            # A terminal COMPLETE is published only after the same run passes
            # the complete artifact audit. Partial bounded runs stay partial.
            try:
                audit=audit_collection(experiment,jobs,state)
                state['audit_status']=audit['status']
                state['audit_run_id']=audit.get('run_id')
                if state['status']=='COMPLETE':
                    expected_counts={route+'|'+model:20 for route in ROUTES for model in MODELS}
                    audit_passed=(audit.get('status')=='COLLECTED_UNVALIDATED'
                                  and audit.get('run_id')==run_id
                                  and audit.get('responses')==80
                                  and audit.get('counts')==expected_counts
                                  and audit.get('cost_known') is True
                                  and decimal_value(audit['confirmed_cost_usd'])==spent
                                  and not audit.get('integrity_errors')
                                  and audit.get('duplicate_job_ids')==0)
                    if not audit_passed:
                        state['status']='AUDIT_FAILED'
            except Exception as exc:
                state['status']='AUDIT_FAILED'
                state['audit_status']='ERROR'
                state['audit_error_type']=type(exc).__name__
            for name in ['state.json','summary.json','run-'+run_id+'.json']:
                try:
                    state['observed_at']=utc_now()
                    write_json(collection/name,state,exclusive=name.startswith('run-'))
                except OSError as exc:
                    local_error('terminal_'+name,exc)
        return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment','--exp',type=Path,default=EXP)
    parser.add_argument('--codex',type=Path)
    parser.add_argument('--max-jobs',type=int,default=80)
    parser.add_argument('--max-seconds',type=float,default=7200)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--run',action='store_true')
    modes.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    try:
        result=run_trial(args.experiment,run=args.run,codex_path=args.codex,max_jobs=args.max_jobs,max_seconds=args.max_seconds)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0 if result['status'] in ('DRY_RUN','COMPLETE') else 2
    except Exception as exc:
        print(json.dumps({'status':'NOT_STARTED','error_type':type(exc).__name__,'message':str(exc) if isinstance(exc,(ValueError,InspectionRequired)) else 'See local preflight; no service response body printed'},ensure_ascii=False))
        return 2


if __name__=='__main__':
    raise SystemExit(main())
