"""Explicit authorized recovery; old dispatches count, old raw artifacts stay immutable.

Default is offline dry-run. This run preserves two historical unknown billing
outcomes under the user's explicit continuation authorization; all NEW unknown
costs immediately stop dispatch. Known spend is not represented as total spend.
"""
from __future__ import annotations
import argparse
from collections import Counter
from decimal import Decimal
import importlib.util
import json
import os
from pathlib import Path
import re
import threading
import time
import uuid

EXP=Path(__file__).resolve().parents[1]
BASE_REL='experiments/EXP-035-aligned-paired-repeat-pilot/code/run_trial.py'
_spec=importlib.util.spec_from_file_location('frozen_paired_trial',EXP.parents[1]/BASE_REL)
base=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
for _name in ['MODELS','ROUTES','RUNTIME_FILES','InspectionRequired','utc_now','sha_bytes','load','read_jsonl',
              'write_json','append_jsonl','RunLock','import_file','decimal_value','validate_plan','check_preflight',
              'inspect_api','api_cost','LiveCodex','previous']:
    globals()[_name]=getattr(base,_name)


def project_path(root,relative):
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Path outside project root')
    return path


def verify_manifest(experiment):
    root=experiment.parents[1]
    manifest=load(experiment/'manifest.json')
    entries=[manifest['config']]+manifest.get('inputs',[])+manifest.get('code',[])
    entries += [e for e in manifest.get('outputs',[]) if e['path'].endswith('/request-plan.jsonl')]
    hashes={}
    for item in entries:
        path=project_path(root,item['path'])
        if item['path'] in hashes and hashes[item['path']]!=item['sha256']:
            raise ValueError('Conflicting frozen hashes')
        if sha_bytes(path.read_bytes())!=item['sha256']:
            raise ValueError('Frozen file hash mismatch: '+item['path'])
        hashes[item['path']]=item['sha256']
    required=RUNTIME_FILES+[BASE_REL]+[(experiment/p).relative_to(root).as_posix() for p in
        ['config.json','request-plan.jsonl','code/run_recovery.py','code/kaggle_transport.py','recovery-receipt.json']]
    if any(p not in hashes for p in required):
        raise ValueError('Manifest must bind recovery code, receipt, plan, configuration and frozen runtime')
    if hashes[(experiment/'code/run_recovery.py').relative_to(root).as_posix()]!=sha_bytes(Path(__file__).read_bytes()):
        raise ValueError('Executing recovery controller differs from manifest')
    return manifest,hashes


def read_history(experiment,jobs,binding):
    root=experiment.parents[1]
    receipt=load(experiment/'recovery-receipt.json')
    if (receipt.get('authorization_to_continue_with_preserved_historical_uncertainty') is not True
            or receipt.get('authorization',{}).get('generation') is not True):
        raise ValueError('Explicit continuation authorization missing')
    if receipt.get('historical_cost_known') is not False:
        raise ValueError('Historical uncertain cost must remain unknown')
    state_path=project_path(root,receipt['prior_state_path'])
    records_path=project_path(root,receipt['prior_records_path'])
    old_collection=records_path.parent
    for p in [state_path,records_path]:
        if p.relative_to(root).as_posix() not in binding:
            raise ValueError('Manifest must bind historical state and records')
    old=load(state_path)
    if old.get('status')=='RUNNING' or old.get('active') or old.get('cost_known') is not False:
        raise ValueError('Historical terminal uncertainty not established')
    rows=read_jsonl(records_path)
    success=set(receipt['historical_success_job_ids'])
    unknown=set(receipt['historical_uncertain_job_ids'])
    if len(success)!=2 or len(unknown)!=2 or success & unknown:
        raise ValueError('Recovery scope is two historical successes and two unknowns')
    planned={j['job_id']:j for j in jobs}
    dispatched=set();counts=Counter()
    for path in sorted((old_collection/'attempts').iterdir()):
        if not path.is_dir():
            raise ValueError('Unexpected historical attempt entry')
        for f in path.iterdir():
            if f.is_file() and f.relative_to(root).as_posix() not in binding:
                raise ValueError('Manifest must bind every historical attempt artifact')
        req=load(path/'request.json');result=load(path/'result.json');job=req['job'];ident=path.name
        if ident not in planned or job!=planned[ident] or result['job_id']!=ident:
            raise ValueError('Historical attempt does not match frozen plan')
        if ident in success and result['status']!='SUCCEEDED':
            raise ValueError('Historical success not established')
        if ident in unknown and result['status']!='UNCERTAIN':
            raise ValueError('Historical uncertainty not preserved')
        dispatched.add(ident);counts[job['route']]+=1
    if dispatched!=success|unknown or dict(counts)!=receipt['historical_counts'] or dict(counts)!=old['requests_by_route']:
        raise ValueError('Historical dispatch counts mismatch')
    if dict(counts)!={'codex_subscription':2,'kaggle_model_proxy':2}:
        raise ValueError('Recovery must count two prior Codex and two prior Kaggle dispatches')
    if len(dispatched)!=old['requests_sent_total'] or {r['job_id'] for r in rows}!=success or len(rows)!=2:
        raise ValueError('Historical saved/dispatched mismatch')
    for row in rows:
        ident=row['job_id'];path=old_collection/'attempts'/ident;job=planned[ident]
        if any(row[k]!=job[k] for k in ['route','prompt_id','model_label','repeat_id','user_message_sha256']):
            raise ValueError('Historical success identity mismatch')
        if sha_bytes(row['text'].encode())!=row['text_sha256'] or (path/'final.txt').read_bytes()!=row['text'].encode():
            raise ValueError('Historical successful text changed')
        if load(path/'record.json')!=row:
            raise ValueError('Historical index differs from raw record')
        provenance=receipt.get('historical_attempt_origins',{}).get(ident)
        if not isinstance(provenance,dict):
            raise ValueError('Historical success original provenance missing')
        original=project_path(root,provenance['attempt_path'])
        if (original.name!=ident or provenance['experiment']!=original.parents[2].name
                or (original/'final.txt').relative_to(root).as_posix() not in binding
                or (original/'record.json').relative_to(root).as_posix() not in binding):
            raise ValueError('Historical original provenance is not frozen')
        if ((original/'final.txt').read_bytes()!=row['text'].encode()
                or load(original/'record.json')!=row):
            raise ValueError('Historical snapshot differs from original success')
    if decimal_value(receipt['historical_confirmed_cost_usd'])!=decimal_value(old['confirmed_cost_usd']):
        raise ValueError('Historical confirmed cost changed')
    # This one prior attempt failed during a read-only existence check, before
    # the transport advanced its stage to create_task or posted model work.
    # Keep its original evidence frozen without treating it as a model dispatch.
    non_dispatch=receipt.get('non_dispatch_failures',[])
    if len(non_dispatch)!=1:
        raise ValueError('Expected one proven metadata-only failure')
    proof=non_dispatch[0];ident=proof['job_id'];path=project_path(root,proof['attempt_path'])
    if ident not in planned or ident in dispatched or path.name!=ident:
        raise ValueError('Metadata-only failure identity conflicts with dispatch history')
    for name in ['request.json','result.json','cloud-submit-intent.json']:
        if (path/name).relative_to(root).as_posix() not in binding:
            raise ValueError('Metadata-only failure evidence is not frozen')
    req=load(path/'request.json');result=load(path/'result.json');intent=load(path/'cloud-submit-intent.json')
    if (req.get('job')!=planned[ident] or result.get('job_id')!=ident or intent.get('job_id')!=ident
            or result.get('stage')!='task_preexistence' or result.get('http_status')!=403
            or result.get('status')!='UNCERTAIN'
            or any((path/name).exists() for name in ['cloud-task.json','cloud-run.json','cloud-result.json','raw_response.json'])):
        raise ValueError('Prior metadata-only failure does not establish no model dispatch')
    return {'receipt':receipt,'rows':rows,'dispatched':dispatched,'route_counts':counts,
            'collection':old_collection,'state':old}


def safe_diagnostics(exc):
    # Whitelist types/codes/events only. Never serialize exception message/args.
    out={}
    for attr in ['cause_chain','trace_events','transport_trace','trace']:
        value=getattr(exc,attr,None)
        if isinstance(value,list):
            filtered=[]
            for entry in value[:100]:
                if isinstance(entry,str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,150}',entry):
                    filtered.append(entry)
                elif isinstance(entry,dict):
                    one={}
                    for k,v in entry.items():
                        if k in ['type','error_type','stage'] and isinstance(v,str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,150}',v):one[k]=v
                        elif k in ['errno','verify_code'] and (v is None or type(v) is int):one[k]=v
                    if one:filtered.append(one)
            out[attr]=filtered
    return out


def combined_rows(experiment,history,rows):
    root=experiment.parents[1];out=[]
    for origin,source in [(history['collection'].parent,history['rows']),(experiment,rows)]:
        for row in source:
            inherited=origin!=experiment
            provenance=history['receipt']['historical_attempt_origins'][row['job_id']] if inherited else None
            out.append(dict(row,origin_experiment=provenance['experiment'] if inherited else origin.name,
                            inherited_history_snapshot=inherited,
                            attempt_path_project=(origin/row['attempt_path']).relative_to(root).as_posix(),
                            attempt_original_path_project=provenance['attempt_path'] if inherited else
                                (origin/row['attempt_path']).relative_to(root).as_posix()))
    return out


def audit_collection(experiment,jobs,state,history,binding):
    collection=experiment/'collection';errors=[]
    # Recheck frozen source bytes and inherited raw evidence at terminal audit.
    verify_manifest(experiment)
    read_history(experiment,jobs,binding)
    rows=read_jsonl(collection/'records.jsonl');planned={j['job_id']:j for j in jobs}
    ids=[r['job_id'] for r in rows]
    if len(ids)!=len(set(ids)) or set(ids)&history['dispatched']:errors.append('DUPLICATE_JOB_ID')
    sent=Counter(history['route_counts']);new_results=[];spent=decimal_value(history['receipt']['historical_confirmed_cost_usd'])
    for path in (collection/'attempts').iterdir():
        if not path.is_dir() or not (path/'request.json').exists() or not (path/'result.json').exists():
            errors.append('INCOMPLETE_ATTEMPT');continue
        request=load(path/'request.json');result=load(path/'result.json');job=request['job']
        if path.name not in planned or job!=planned[path.name] or result['job_id']!=path.name or path.name in history['dispatched']:
            errors.append('ATTEMPT_IDENTITY')
        sent[job['route']]+=1;new_results.append(result)
        if job['route']=='kaggle_model_proxy' and result.get('cost_usd') is not None:spent+=decimal_value(result['cost_usd'])
    for row in rows:
        path=collection/'attempts'/row['job_id'];job=planned[row['job_id']]
        if any(row[k]!=job[k] for k in ['route','prompt_id','model_label','repeat_id','user_message_sha256']):errors.append('ROW_IDENTITY')
        if sha_bytes(row['text'].encode())!=row['text_sha256'] or (path/'final.txt').read_bytes()!=row['text'].encode():errors.append('TEXT_HASH')
        if load(path/'record.json')!=row or load(path/'result.json')['status']!='SUCCEEDED':errors.append('RECORD_RESULT_MISMATCH')
        if row['route']=='kaggle_model_proxy':
            raw=load(path/'raw_response.json')
            checked=inspect_api(raw,job)
            if not checked['success'] or checked['text_sha256']!=row['text_sha256']:
                errors.append('RAW_RESPONSE_MISMATCH')
            nanos,cost=api_cost(raw)
            if cost!=decimal_value(row['cost_usd']) or nanos!=row['cost_nanodollars']:errors.append('BILLING_MISMATCH')
    if sum(sent.values())!=state['requests_sent_total'] or dict(sent)!=state['requests_by_route']:errors.append('DISPATCH_COUNT_MISMATCH')
    if spent!=decimal_value(state['confirmed_cost_usd']):errors.append('CONFIRMED_COST_MISMATCH')
    if sum(sent.values())>80 or any(v>40 for v in sent.values()):errors.append('DISPATCH_CAP_EXCEEDED')
    joined=combined_rows(experiment,history,rows)
    counts=Counter(r['route']+'|'+r['model_label'] for r in joined)
    expected_counts={'codex_subscription|gpt-5.6-luna':20,'codex_subscription|gpt-5.6-terra':20,
                     'kaggle_model_proxy|gpt-5.6-luna':20,'kaggle_model_proxy|gpt-5.6-terra':18}
    cells={}
    for row in joined:
        cells.setdefault((row['prompt_id'],row['model_label']),set()).add((row['route'],row['repeat_id']))
    full_cell={(route,repeat) for route in ROUTES for repeat in (1,2)}
    complete_cells=sum(identities==full_cell for identities in cells.values())
    complete=(len(joined)==78 and sum(sent.values())==80 and set(sent.values())=={40}
              and len(new_results)==76 and all(r['status']=='SUCCEEDED' for r in new_results)
              and dict(counts)==expected_counts and complete_cells==19
              and state['new_cost_known'] and not errors and not state['active'])
    audit={'observed_at':utc_now(),'run_id':state['run_id'],
           'status':'DISPATCH_COMPLETE_PARTIAL_AUDITED' if complete else 'INCOMPLETE',
           'responses':len(joined),'new_responses':len(rows),'historical_successes':len(history['rows']),
           'requests_sent_total':sum(sent.values()),'requests_by_route':dict(sent),'counts':dict(counts),
           'complete_paired_cells':complete_cells,'expected_complete_paired_cells':19,
           'historical_uncertain':len(history['receipt']['historical_uncertain_job_ids']),
           'historical_cost_known':False,'new_cost_known':state['new_cost_known'],'cost_known':False,
           'confirmed_cost_usd':state['confirmed_cost_usd'],'integrity_errors':errors,
           'duplicate_job_ids':len(ids)-len(set(ids)),'raw_repeats_preserved':True,
           'training_or_api_equivalence':'NOT_EVALUATED'}
    write_json(experiment/'metrics/collection-audit.json',audit)
    with (collection/'combined-records.jsonl').open('w',encoding='utf-8',newline='\n') as stream:
        for row in joined:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    return audit


def run_recovery(experiment=EXP, *, run=False, codex_path=None, max_jobs=76, max_seconds=7200,
              kaggle_generate=None, kaggle_make_context=None, kaggle_preflight=None,
              codex_call=None, format_checker=None):
    experiment=Path(experiment).resolve()
    root=experiment.parents[1]
    config=load(experiment/'config.json')
    jobs=read_jsonl(experiment/'request-plan.jsonl')
    validate_plan(config,jobs)
    manifest,binding=verify_manifest(experiment)
    if not 1<=max_jobs<=76 or max_seconds<=0:
        raise ValueError('Invalid execution bound')
    collection=experiment/'collection'
    history=read_history(experiment,jobs,binding)
    rows,seen=previous(collection,jobs,binding)
    if seen & history['dispatched']:
        raise InspectionRequired('Historical dispatched request duplicated in recovery')
    plan={'status':'DRY_RUN','target_total':80,'already_saved':len(history['rows'])+len(rows),'remaining':80-len(history['dispatched'])-len(seen),
          'max_new_jobs_this_run':max_jobs,'global_max_inflight':1,'requests_sent':0,
          'historical_dispatched':len(history['dispatched']),'historical_cost_known':history['receipt']['historical_cost_known']}
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
        history=read_history(experiment,jobs,binding)
        rows,seen=previous(collection,jobs,binding)
        if seen & history['dispatched']:
            raise InspectionRequired('Historical dispatched request duplicated in recovery')
        seen |= history['dispatched']
        if not (collection/'binding.json').exists():
            write_json(collection/'binding.json',binding,True)
        (collection/'attempts').mkdir(exist_ok=True)
        context=None
        mutex=threading.RLock()
        done=threading.Event()
        halt=threading.Event()
        spent=decimal_value(history['receipt']['historical_confirmed_cost_usd'])+sum((decimal_value(r['cost_usd']) for r in rows if r['route']=='kaggle_model_proxy'),Decimal(0))
        route_sent=Counter(history['route_counts'])+Counter(r['route'] for r in rows)
        seen_response_ids={(r['route'],r['provider_response_id']) for r in history['rows']+rows if r.get('provider_response_id')}
        run_id=uuid.uuid4().hex
        state={'status':'RUNNING','phase':'PREFLIGHT','run_id':run_id,'controller_pid':os.getpid(),'started_at':utc_now(),
               'observed_at':utc_now(),'target_total':80,'saved_total':len(history['rows'])+len(rows),'saved_new_total':len(rows),'saved_this_run':0,
               'already_saved':len(history['rows'])+len(rows),'launched_this_run':0,'requests_sent_total':sum(route_sent.values()),
               'requests_by_route':dict(route_sent),'active':[],'failures':[],'local_errors':[],
               'failed_this_run':0,'confirmed_cost_usd':str(spent),'cost_known':history['receipt']['historical_cost_known'],
               'new_cost_known':True,'historical_cost_known':history['receipt']['historical_cost_known'],
               'historical_dispatched':len(history['dispatched']),'historical_unknown_job_ids':history['receipt']['historical_uncertain_job_ids'],
               'historical_saved':len(history['rows']), 'maximum_saved_total':78,
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
                if not state['new_cost_known'] or spent>=decimal_value(config['proposed_budget']['stop_dispatch_at_confirmed_cost_usd']):
                    state['status']='STOPPED_COST_UNKNOWN' if not state['new_cost_known'] else 'STOPPED_BUDGET'
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
                        raw=kaggle_generate(context,job['api_payload'],attempt_dir=attempt,job=job)
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
                            state['new_cost_known']=False
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
                            state['saved_new_total']=len(rows)
                            state['saved_total']=len(history['rows'])+len(rows)
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
                        state['new_cost_known']=False
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
                    failure.update(safe_diagnostics(exc))
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
                state['status']='DISPATCH_COMPLETE_PARTIAL' if len(seen)==80 else 'STOPPED_SIGNAL'
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
                audit=audit_collection(experiment,jobs,state,history,binding)
                state['audit_status']=audit['status']
                state['audit_run_id']=audit.get('run_id')
                if state['status']=='DISPATCH_COMPLETE_PARTIAL':
                    if not (audit.get('status')=='DISPATCH_COMPLETE_PARTIAL_AUDITED'
                            and audit.get('run_id')==run_id and audit.get('requests_sent_total')==80
                            and audit.get('responses')==78 and audit.get('new_cost_known') is True
                            and not audit.get('integrity_errors')):
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
        if context is not None and hasattr(context,'close'):
            context.close()
        return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment','--exp',type=Path,default=EXP)
    parser.add_argument('--codex',type=Path)
    parser.add_argument('--max-jobs',type=int,default=76)
    parser.add_argument('--max-seconds',type=float,default=7200)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--run',action='store_true');modes.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    try:
        result=run_recovery(args.experiment,run=args.run,codex_path=args.codex,max_jobs=args.max_jobs,max_seconds=args.max_seconds)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0 if result['status'] in ['DRY_RUN','DISPATCH_COMPLETE_PARTIAL'] else 2
    except Exception as exc:
        print(json.dumps({'status':'NOT_STARTED','error_type':type(exc).__name__,
                          'message':str(exc) if isinstance(exc,(ValueError,InspectionRequired)) else 'No service response body printed'},ensure_ascii=False))
        return 2


if __name__=='__main__':raise SystemExit(main())
