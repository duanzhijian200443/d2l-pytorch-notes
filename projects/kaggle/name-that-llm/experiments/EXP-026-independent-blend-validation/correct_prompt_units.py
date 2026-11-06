"""Correct a single case-equivalent prompt pair without changing model outputs."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=ROOT/'data/public-pools/enaimaan/ntl-openai-generated-responses/v3/prompts.jsonl'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
norm=lambda s:' '.join(unicodedata.normalize('NFKC',s).casefold().split())

def interval(frame):
    names=sorted(frame.prompt_id.unique())
    pos={k:i for i,k in enumerate(names)}
    labels=sorted(frame.model.unique()); cols={k:i for i,k in enumerate(labels)}
    support=np.zeros((len(names),len(labels)),dtype=np.int64)
    delta=np.zeros_like(support)
    for r in frame.itertuples():
        i,j=pos[r.prompt_id],cols[r.model]
        support[i,j]+=1
        delta[i,j]+=int(r.blend_prediction==r.model)-int(r.stylegram_prediction==r.model)
    rng=np.random.default_rng(42)
    draws=[]
    for _ in range(2000):
        take=rng.integers(0,len(names),size=len(names))
        den=support[take].sum(axis=0)
        assert np.all(den>0)
        draws.append(float(np.mean(delta[take].sum(axis=0)/den)))
    return np.quantile(draws,[.025,.975]).tolist()

def main():
    accepted=json.loads((HERE/'metrics/verified.json').read_text(encoding='utf-8'))
    assert sha(SOURCE)=='7a474ee6af8cdeae8e74f62a5f9e3b3ce65bb0eedc4056660925d02542be735b'
    pred=pd.read_csv(HERE/'predictions/predictions.csv',dtype={'id':str,'prompt_id':str})
    evaluation=pd.read_csv(HERE/'evaluation.csv',dtype={'id':str,'prompt_id':str})
    assert pred.id.tolist()==evaluation.id.tolist() and len(pred)==5200
    assert pred.prompt_id.tolist()==evaluation.prompt_id.tolist()
    wanted=set(pred.id); body={}
    for line in SOURCE.read_text(encoding='utf-8').splitlines():
        p=json.loads(line)
        if p['response_id'] not in wanted:continue
        request=p['messages'][-1]['content']; suffix=f"\n\nWrite about {p['target_words']} words."
        assert request.endswith(suffix)
        body[p['response_id']]=hashlib.sha256(norm(request[:-len(suffix)]).encode()).hexdigest()
    assert set(body)==wanted
    units=pred[['id','prompt_id']].copy();units['body_hash']=units.id.map(body)
    assert units.groupby('prompt_id').body_hash.nunique().eq(1).all()
    grouped=units.groupby('body_hash').prompt_id.unique()
    duplicate=[sorted(x.tolist()) for x in grouped if len(x)>1]
    assert duplicate==[['ext-20ab8f2ddfb9a2aa7dd5','ext-f646bbf21325a84376ad']]
    mapping={g:min(gs) for gs in grouped for g in gs}
    old=interval(pred)
    assert np.allclose(old,accepted['recomputed_comparison']['paired_prompt_bootstrap']['ci95'],atol=1e-15,rtol=0)
    corrected=pred.copy();corrected['prompt_id']=corrected.prompt_id.map(mapping)
    assert corrected.prompt_id.nunique()==1299
    new=interval(corrected)
    out={'schema_version':1,'observed_at_utc':datetime.now(timezone.utc).isoformat(),
      'status':'PASS_CORRECTED_PROMPT_UNITS','change':'Merge one case-equivalent source-body pair for statistical resampling only',
      'original_groups':1300,'normalized_groups':1299,'merged_source_groups':duplicate,
      'affected_rows':8,'rows':5200,'point_estimate_changed':False,
      'delta_macro_recall':accepted['recomputed_comparison']['delta_macro_recall'],
      'original_ci95':old,'corrected_ci95':new,'draws':2000,'seed':42,
      'sampling_unit':'normalized source-body components; stable canonical original group name',
      'prediction_or_model_bytes_changed':False,'scope':'Four-covered-class source check; no retraining uncertainty or semantic independence guarantee',
      'inputs':{str(p.relative_to(ROOT)):sha(p) for p in [SOURCE,HERE/'evaluation.csv',HERE/'predictions/predictions.csv',HERE/'metrics/verified.json']},
      'script_sha256':sha(Path(__file__))}
    target=HERE/'metrics/prompt-unit-correction.json'
    if target.exists():
        existing=json.loads(target.read_text(encoding='utf-8'));out['observed_at_utc']=existing['observed_at_utc']
        assert existing==out
    else:
        with target.open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(out,ensure_ascii=False))

if __name__=='__main__':main()
