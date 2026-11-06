"""Offline fixed selection; never calls a model or alters existing collections."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import random

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
SEED = 20261101039
MODELS = ['gpt-5.6-luna','gpt-5.6-terra','gpt-5.6-sol','gpt-6-astra']
CODEX = Path(r'C:\Users\34331\AppData\Local\OpenAI\Codex\bin\22cfb1d302613eae\codex.exe')

def load(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def write(path, obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    if (EXP/'collection').exists(): raise RuntimeError('Never rebuild an active or existing collection')
    source = ROOT/'experiments/EXP-034-explicit-format-alignment'
    raw = (source/'prompts.jsonl').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == 'ae45936ba73373d551f9ac44ce13df906f1e32bfb468198b1683992cb9f02103'
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    raw_lines = {json.loads(line)['prompt_id']: line for line in raw.splitlines(keepends=True) if line.strip()}
    source_config = load(source/'config.json')
    excluded = set(source_config['prepared_paired_panel']['prompt_ids'])
    cats = sorted({r['category'] for r in rows})
    rng = random.Random(SEED)
    double_holdout = set(rng.sample(cats,5))
    selected = []
    for cat in cats:
        eligible = [r for r in rows if r['category']==cat and r['prompt_id'] not in excluded]
        train = [r for r in eligible if r['split']=='train']
        chosen = [rng.choice([r for r in train if r['target_words']==target]) for target in sorted({r['target_words'] for r in train})]
        chosen_ids = {r['prompt_id'] for r in chosen}
        chosen += rng.sample([r for r in train if r['prompt_id'] not in chosen_ids],6-len(chosen))
        holdout = rng.sample([r for r in eligible if r['split']=='holdout'],2 if cat in double_holdout else 1)
        selected += chosen+holdout
    rng.shuffle(selected)
    assert len(selected)==75 and len({r['prompt_id'] for r in selected})==75
    assert Counter(r['split'] for r in selected)=={'train':60,'holdout':15}
    assert not ({r['prompt_id'] for r in selected} & excluded)
    (EXP/'prompts.jsonl').write_bytes(b''.join(raw_lines[r['prompt_id']] for r in selected))
    (EXP/'workspace').mkdir(exist_ok=True)
    config = load(source/'pilot/config.json')
    config.update(experiment_id='EXP-039',retry_of=None,collection_authorized=True,
                  authorization={'generation':True,'basis':'同时用codex进行数据采集，采集300条；原四模型各75条','approved_at':datetime.now(timezone.utc).isoformat()},
                  models=[{'slug':m,'reasoning_effort':'low'} for m in MODELS],
                  instructions_file='../EXP-021-codex-subscription-pilot/model-instructions.txt',
                  workspace='workspace',prompts_file='prompts.jsonl',
                  sampling={'seed':SEED,'shared_prompt_groups':75,'train_groups':60,'holdout_groups':15,
                            'excluded_paired_prompt_ids':sorted(excluded),'category_sampling':'6train each;5categories2holdout,5categories1holdout;cover available wordtargets then sample remaining'},
                  concurrency={'this_batch':1,'existing_EXP038':1,'combined_maximum':2,'do_not_interrupt_companion':True})
    config['budget']={'max_successful_responses':300,'max_model_requests':300,'max_concurrent_requests':1,
                      'automatic_retries':0,'wall_seconds':21600,'api_usd':0,
                      'subscription_usage':'User authorized existing Codex subscription only; no API-key override'}
    config['validation']['responses_by_split']={'train':240,'holdout':60}
    config['validation']['training_use']='quarantined_pending_validation; preserve original base prompt split across all variants'
    write(EXP/'config.json',config)
    path=ROOT/'experiments/EXP-030-codex-subscription-pilot-r3/code/codex_collect.py'
    spec=importlib.util.spec_from_file_location('frozen_prepare_only',path)
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    *_,jobs=core.prepare(EXP,CODEX)
    assert len(jobs)==300 and len({j['key'] for j in jobs})==300
    (EXP/'request-plan.jsonl').write_text(''.join(json.dumps(j,ensure_ascii=False)+'\n' for j in jobs),encoding='utf-8')
    old=ROOT/'experiments/EXP-023-codex-subscription-pilot-r1/collection/records.jsonl'
    old_keys={(r['model_requested'],hashlib.sha256(r['prompt'].encode()).hexdigest()) for r in [json.loads(x) for x in old.read_text(encoding='utf-8').splitlines() if x.strip()]}
    new_keys={(j['model']['slug'],hashlib.sha256(j['prompt']['prompt'].encode()).hexdigest()) for j in jobs}
    assert not old_keys & new_keys
    stats={'observed_at':datetime.now(timezone.utc).isoformat(),'status':'PASS','generation_requests':0,
           'seed':SEED,'prompt_groups':75,'requests':300,'model_counts':dict(Counter(j['model']['slug'] for j in jobs)),
           'group_splits':dict(Counter(r['split'] for r in selected)),
           'group_categories':dict(Counter(r['category'] for r in selected)),
           'group_word_targets':dict(Counter(r['target_words'] for r in selected)),
           'excluded_paired_ids':sorted(excluded),'paired_panel_overlap':0,'exact_user_model_overlap_old1000':0,
           'identity_limit':'All75 base prompts existed in old1000; these are new format variants, not75 novel independent base prompts.'}
    write(EXP/'metrics/selection-audit.json',stats)
    print(json.dumps(stats,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
