"""Build identical user-message variants and previews offline; never calls a model."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
ORIGINAL = ROOT / 'experiments/EXP-021-codex-subscription-pilot/prompts.jsonl'
ORIGINAL_SHA256 = 'dc7495501a3412775902716a40d1b74d13e840b4b73a4c0c9530e28a6080a4d0'
OLD_RECORDS = ROOT / 'experiments/EXP-023-codex-subscription-pilot-r1/collection/records.jsonl'
OLD_RECORDS_SHA256 = 'cb903279b3727b31ce2125b89c5ce8f530d769a984f49c2cb86e4fbefb1e9722'
IDS = ['cp0001','cp0022','cp0043','cp0064','cp0081','cp0102','cp0123','cp0144','cp0161','cp0182']
MODELS = ['gpt-5.6-luna','gpt-5.6-terra']
PROTOCOL = 'single-paragraph-plain-text-v1'
BASE_INSTRUCTION = "Answer the user's request."


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def write_jsonl(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in data),encoding='utf-8')


def make_variant(row, requirements):
    if 'protocol_version' in row:
        raise ValueError('Do not apply a second wrapper to an already aligned prompt')
    return {**row, 'prompt':row['prompt']+'\n\n'+requirements,
            'base_prompt':row['prompt'], 'base_prompt_sha256':sha(row['prompt'].encode('utf-8')),
            'base_prompt_id':row['prompt_id'], 'protocol_version':PROTOCOL,
            'variant_id':row['prompt_id']+'::'+PROTOCOL,
            'format_instructions_sha256':sha(requirements.encode('utf-8'))}


def format_flags(text):
    """Screen only; never clean, reject or regenerate a saved response automatically."""
    checks = {
        'empty': not text.strip(),
        'line_break': bool(re.search(r'[\r\n]',text)),
        'heading': bool(re.search(r'(?m)^\s*#{1,6}\s+',text)),
        'bullet_list': bool(re.search(r'(?m)^\s*[-*+]\s+',text)),
        'numbered_list': bool(re.search(r'(?m)^\s*\d+[.)]\s+',text)),
        'bold': bool(re.search(r'\*\*[^*]+\*\*|__[^_]+__',text)),
        'italic': bool(re.search(r'(?<!\*)\*(?!\*)[^*\r\n]+(?<!\*)\*(?!\*)|(?<!\w)_[^_\r\n]+_(?!\w)',text)),
        'code_markup': '`' in text,
        'markdown_link': bool(re.search(r'\[[^\]]+\]\([^)]*\)',text)),
        'blockquote': bool(re.search(r'(?m)^\s*>\s+',text)),
        'assistant_preface_candidate': bool(re.search(r"\A\s*(?:Sure[!,.—:]|Certainly[!,.—:]|Here(?:'s| is| are)\b)",text,re.I)),
    }
    return [name for name, value in checks.items() if value]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    raw=ORIGINAL.read_bytes()
    if sha(raw)!=ORIGINAL_SHA256 or sha(OLD_RECORDS.read_bytes())!=OLD_RECORDS_SHA256:
        raise RuntimeError('Original prompts or previously collected data changed')
    rows=read_jsonl(ORIGINAL)
    requirements=(EXP/'format-instructions.txt').read_text(encoding='utf-8').strip()
    if not requirements or '\n' in requirements:
        raise ValueError('Formatting protocol must be one nonempty instruction paragraph')
    variants=[make_variant(row,requirements) for row in rows]
    assert len(variants)==250 and len({r['prompt_id'] for r in variants})==250
    assert Counter(r['split'] for r in variants)=={'train':200,'holdout':50}
    by_id={r['prompt_id']:r for r in variants}
    selected=[by_id[ident] for ident in IDS]
    assert all(r['split']=='train' for r in selected)
    for before,after in zip(rows,variants):
        for key,value in before.items():
            if key!='prompt':assert after[key]==value,(key,before['prompt_id'])
        assert after['base_prompt']==before['prompt']
        assert after['prompt']==before['prompt']+'\n\n'+requirements
    previews=[]
    for row in selected:
        for model in MODELS:
            messages=[{'role':'system','content':BASE_INSTRUCTION},{'role':'user','content':row['prompt']}]
            previews.append({'prompt_id':row['prompt_id'],'prompt_group':row['prompt_group'],'split':row['split'],
                'variant_id':row['variant_id'],'model_label':model,'target_words':row['target_words'],
                'user_message_sha256':sha(row['prompt'].encode('utf-8')),
                'payload':{'model':'openai/'+model,'messages':messages,'reasoning_effort':'low',
                           'max_completion_tokens':1024,'stream':False,'n':1},
                'execution_status':'PREVIEW_ONLY_NOT_SENT','execution_authorized':False})
    report={'observed_at':datetime.now(timezone.utc).isoformat(),'status':'OFFLINE_ALIGNMENT_PASS',
            'full_prompt_count':250,'pilot_prompt_count':10,'pilot_request_previews':20,
            'split_counts':dict(Counter(r['split'] for r in variants)),
            'categories':dict(Counter(r['category'] for r in variants)),
            'original_prompt_fields_preserved':True,'same_user_message_for_both_routes':True,
            'original_records_unchanged':True,'model_requests_sent':0,
            'original_prompts_sha256':ORIGINAL_SHA256,'original_records_sha256':OLD_RECORDS_SHA256,
            'protocol':PROTOCOL,'protocol_text_sha256':sha(requirements.encode('utf-8')),
            'limitations':['Only visible formatting requirements aligned; backend context/snapshot equivalence not established.',
                          'API minimal system retained from previous paired pilot; this is not a no-system replica of Lucas.',
                          'API 1024 cap retained; existing Codex collector has no equivalent explicit cap.',
                          'Temperature/top_p/seed remain omitted; effective defaults unknown.',
                          'Full source and raw output preserved; format screening is not automatic deletion/retry.']}
    if not args.check_only:
        write_jsonl(EXP/'prompts.jsonl',variants)
        write_jsonl(EXP/'pilot/prompts.jsonl',selected)
        write_jsonl(EXP/'kaggle-request-preview.jsonl',previews)
        old=json.loads((ROOT/'experiments/EXP-023-codex-subscription-pilot-r1/config.json').read_text(encoding='utf-8'))
        old.update(experiment_id='EXP-034',retry_of=None,models=[{'slug':m,'reasoning_effort':'low'} for m in MODELS],
                   instructions_file='../../EXP-021-codex-subscription-pilot/model-instructions.txt',
                   prompts_file='prompts.jsonl',workspace='workspace',protocol_version=PROTOCOL,
                   collection_authorized=False)
        old['budget']={'max_successful_responses':20,'max_concurrent_requests':2,'automatic_retries':0,
                       'wall_seconds':3600,'api_usd':None,'subscription_usage':'NOT_AUTHORIZED_FOR_THIS_NEW_PILOT'}
        old['validation']={'split_id':'codex-pilot-prompt-v1','training_use':'quarantined_pending_validation',
                           'variant_grouping':'Keep every variant/repeat/channel of a base prompt in its original group and split.'}
        write_json(EXP/'pilot/config.json',old)
        (EXP/'pilot/workspace').mkdir(exist_ok=True,parents=True)
        write_json(EXP/'metrics/offline-alignment.json',report)
    else:
        assert read_jsonl(EXP/'prompts.jsonl')==variants
        assert read_jsonl(EXP/'pilot/prompts.jsonl')==selected
        assert read_jsonl(EXP/'kaggle-request-preview.jsonl')==previews
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
