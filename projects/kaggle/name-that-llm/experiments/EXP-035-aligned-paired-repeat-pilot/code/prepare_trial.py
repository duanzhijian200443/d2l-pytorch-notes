"""Prepare an approval-ready paired trial offline. This file cannot send requests."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import random

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
ALIGN = ROOT/'experiments/EXP-034-explicit-format-alignment'
COST_RECEIPT = ROOT/'pending-handoff-kaggle-proxy-pilot-20261101T084539Z/authorization-v2-results/manifest.json'
MODELS = ['gpt-5.6-luna','gpt-5.6-terra']
ROUTES = ['codex_subscription','kaggle_model_proxy']
ORDER_SEED = 20261101


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    old_manifest = load(ALIGN/'manifest.json')
    for entry in old_manifest['inputs']+old_manifest['outputs']:
        if digest(ROOT/entry['path']) != entry['sha256']:
            raise ValueError('Frozen input changed: '+entry['path'])
    prompts = [json.loads(line) for line in (ALIGN/'pilot/prompts.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    assert len(prompts)==10 and len({r['category'] for r in prompts})==10
    assert all(r['split']=='train' for r in prompts)
    lookup = {r['prompt_id']:r for r in prompts}
    preview = [json.loads(line) for line in (ALIGN/'kaggle-request-preview.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    payloads = {(r['prompt_id'],r['model_label']):r['payload'] for r in preview}
    cells = [(row['prompt_id'],model) for row in prompts for model in MODELS]
    random.Random(ORDER_SEED).shuffle(cells)
    jobs = []
    for block_index,(prompt_id,model) in enumerate(cells):
        row=lookup[prompt_id]
        first=ROUTES if block_index%2==0 else list(reversed(ROUTES))
        for repeat in [1,2]:
            routes=first if repeat==1 else list(reversed(first))
            for route in routes:
                prompt_hash=hashlib.sha256(row['prompt'].encode('utf-8')).hexdigest()
                key=f'EXP-035|{route}|{prompt_id}|{model}|{row["protocol_version"]}|{prompt_hash}|r{repeat}'
                job={'request_index':len(jobs)+1,'block_index':block_index+1,
                     'job_id':hashlib.sha256(key.encode('utf-8')).hexdigest(),
                     'cell_id':f'{prompt_id}|{model}','route':route,'repeat_id':repeat,
                     'prompt_id':prompt_id,'prompt_group':row['prompt_group'],'split':row['split'],
                     'model_label':model,'protocol_version':row['protocol_version'],
                     'target_words':row['target_words'],'user_message':row['prompt'],
                     'user_message_sha256':prompt_hash,
                     'base_instruction':"Answer the user's request.",'reasoning_effort':'low',
                     'execution_authorized':False,'state':'PLANNED_NOT_SENT'}
                if route=='kaggle_model_proxy':
                    job['api_payload']=payloads[(prompt_id,model)]
                    assert job['api_payload']['messages'][1]['content']==job['user_message']
                else:
                    job['codex_execution_contract']={'fresh_ephemeral_exec':True,
                        'ignore_user_config':True,'sandbox':'read-only',
                        'frozen_runtime_config':'experiments/EXP-034-explicit-format-alignment/pilot/config.json',
                        'remove_api_key_env_overrides':True,
                        'attempt_path':f'collection/{route}/{prompt_id}/{model}/repeat-{repeat:02d}',
                        'note':'Use job_id/repeat_id for durable identity. Never put repeat identifiers into the model-visible prompt.'}
                jobs.append(job)
    assert len(jobs)==80 and len({r['job_id'] for r in jobs})==80
    groups=defaultdict(list)
    for row in jobs:groups[row['cell_id']].append(row)
    assert len(groups)==20
    for group in groups.values():
        assert len(group)==4 and len({r['user_message_sha256'] for r in group})==1
        assert Counter(r['route'] for r in group)==dict.fromkeys(ROUTES,2)
        assert Counter(r['repeat_id'] for r in group)=={1:2,2:2}
    receipt=load(COST_RECEIPT)
    historical=Decimal(str(receipt['total_cost_usd']))
    estimates={model:Decimal(str(receipt['cost_by_model_usd']['openai/'+model]))*2 for model in MODELS}
    base_estimate=historical*2
    assert sum(estimates.values())==base_estimate
    config={'schema_version':1,'experiment_id':'EXP-035','status':'PREPARED_AWAITING_USER_CONFIRMATION',
        'authorization':{'preparation':True,'generation':False,'basis':'User explicitly requested preparing a small trial and confirming count and budget before execution.'},
        'design':{'base_prompts':10,'models':MODELS,'routes':ROUTES,'repeats_per_cell_per_route':2,
                  'total_new_requests':80,'requests_per_route':40,'independent_prompt_groups':10,
                  'global_max_inflight':1,'route_order':'randomized cell order; ABBA/BAAB within each cell',
                  'order_seed':ORDER_SEED,'sampling_seed':'omitted; order seed is not a generation seed',
                  'all_variants_keep_base_prompt_group':True},
        'proposed_budget':{'funding':'Existing Kaggle free allowance only; no top-up or paid fallback',
                  'target_usd':0.25,'stop_dispatch_at_confirmed_cost_usd':0.10,
                  'hard_cap':False,'unknown_cost_stops_dispatch':True,
                  'maximum_kaggle_requests':40,'maximum_codex_requests':40,
                  'codex_funding':'Existing subscription allowance; no API key',
                  'historical_same_volume_cost_usd':str(base_estimate),
                  'historical_by_model_usd':{k:str(v) for k,v in estimates.items()},
                  'estimate_limit':'Uses prior actual spend, not current pricing or a guaranteed bound. Longer format prompt and generation variability may increase cost. Last request may cross the confirmed-spend threshold.'},
        'stop_rules':['Any transport/auth/rate/unknown-outcome failure stops all new dispatch',
                      'Missing or invalid cost, returned model mismatch, explicit truncation or unknown finish stops all new dispatch',
                      'Maximum request counts are dispatch limits, not a target that can be topped up after failures',
                      'No automatic retries; retain incomplete attempts and inspect before any resumption'],
        'result_handling':['Keep all raw results including format violations, target-length deviations, refusals and identical repeated outputs',
                           'Do not rewrite, filter away, replace or regenerate nonconforming responses',
                           'Report partial/failed/format-nonconforming outcomes separately'],
        'analysis_plan':['Check exact user-message hashes, route/model/repeat identities and completion evidence',
                         'Compare requested format adherence before and after as a descriptive historical comparison',
                         'Compare within-route repeat variation with cross-route differences at the cell level',
                         'Aggregate uncertainty by the 10 base prompt groups; never treat 80 answers or all cross-pairs as independent prompts',
                         'Use only complete paired cells for paired summaries and also report every incomplete cell',
                         'No classifier training, scoring, competition submission or publication'],
        'known_limits':['Two repeats provide only a preliminary variability baseline',
                        'Actual backend snapshot and complete runtime context remain unknown',
                        'Kaggle has explicit 1024 completion-token cap; existing Codex configuration has no equivalent cap',
                        'The earlier unaligned pilot is historical, not a simultaneous randomized format-control arm'],
        'execution_preflight':['Refresh model/low availability and free proxy quota before any request',
                               'Kaggle connector has no generation tool in this session; execution needs the official SDK or an explicitly authorized handoff',
                               'A separate execution runner must respect repeat-aware job_id; do not use the old single-repeat resume key unchanged']}
    (EXP/'request-plan.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in jobs),encoding='utf-8')
    save(EXP/'config.json',config)
    save(EXP/'metrics/preparation-checks.json',{'status':'PASS','observed_at':datetime.now(timezone.utc).isoformat(),
        'requests':80,'routes':dict(Counter(r['route'] for r in jobs)),
        'by_model_route':dict(Counter(r['route']+'|'+r['model_label'] for r in jobs)),
        'unique_jobs':80,'cells':20,'independent_prompt_groups':10,'same_visible_prompt_within_cell':True,
        'repeat_ids_only_in_metadata':True,'inputs_hash_verified':True,'requests_sent':0,
        'historical_cost_receipt_path':COST_RECEIPT.relative_to(ROOT).as_posix(),
        'historical_cost_receipt_sha256':digest(COST_RECEIPT),'historical_40_request_cost_usd':str(base_estimate)})
    print(json.dumps({'status':config['status'],'requests':80,'per_route':40,'estimated_from_history_usd':str(base_estimate),'requests_sent':0},ensure_ascii=False))


if __name__=='__main__':
    main()
