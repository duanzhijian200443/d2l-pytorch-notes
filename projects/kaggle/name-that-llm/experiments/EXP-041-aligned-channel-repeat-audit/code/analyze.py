"""Offline paired/repeated channel analysis. No generation, model fitting or raw edits."""
from pathlib import Path
import collections as co
import datetime as dt
import hashlib
import importlib.util
import json
import math
import re
import unicodedata
import numpy as np

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
PAIRED = ROOT/'experiments/EXP-038-paired-cloud-precheck-recovery'
BATCH = ROOT/'experiments/EXP-040-codex300-concurrency2'
ROUTES = ('codex_subscription','kaggle_model_proxy')
MODELS = ('gpt-5.6-luna','gpt-5.6-terra')
SEED, BOOT = 2026110141, 20000
SOURCES = {}

def sha(b): return hashlib.sha256(b).hexdigest()
def read(path):
    path=Path(path); b=path.read_bytes(); SOURCES[path.relative_to(ROOT).as_posix()]=sha(b); return b
def js(path): return json.loads(read(path).decode('utf-8-sig'))
def jl(path): return [json.loads(s) for s in read(path).decode('utf-8-sig').splitlines() if s.strip()]
def save(name,data):
    p=EXP/name; p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def module(name,path):
    read(path); spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def norm(t):return ' '.join(unicodedata.normalize('NFKC',t).casefold().split())
def words(t):return len(t.split())
def tri(t):
    w=re.findall(r'\b\w+\b',t.lower()); return set(zip(w,w[1:],w[2:])) if len(w)>=3 else {tuple(w)}
def chartri(t):
    t=' '.join(t.lower().split());return co.Counter(t[i:i+3] for i in range(len(t)-2))
def lexical(a,b):
    x,y=tri(a),tri(b);return 1-len(x&y)/len(x|y)
def chardist(a,b):
    x,y=chartri(a),chartri(b)
    return 1-sum(v*y[k] for k,v in x.items())/math.sqrt(sum(v*v for v in x.values())*sum(v*v for v in y.values()))
DISTANCES={'word_trigram_jaccard':lexical,'char_trigram_cosine':chardist,'word_count_gap':lambda a,b:abs(words(a)-words(b))}
def features(t):
    n=words(t)
    return {'words':n,'sentence_endings':len(re.findall(r'[.!?](?:[\u201d\u2019\"\']*)(?=\s|$)',t)),
            'em_dash_per100w':100*t.count('—')/n,'semicolon_per100w':100*t.count(';')/n,
            'colon_per100w':100*t.count(':')/n,'comma_per100w':100*t.count(',')/n}

def cluster(cells,metric):
    grouped=co.defaultdict(list)
    for c in cells:grouped[c['prompt_id']].append(c[metric])
    vals=np.array([np.mean(v) for _,v in sorted(grouped.items())],float)
    if not len(vals):return {'mean':None,'ci95':None,'prompt_groups':0}
    boot=np.random.default_rng(SEED).choice(vals,size=(BOOT,len(vals)),replace=True).mean(axis=1)
    return {'mean':float(vals.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'prompt_groups':len(vals)}
def summarize(cells):
    keys=[k for k in cells[0] if k not in ['prompt_id','model','cross_pairs','within_pairs']] if cells else []
    return {'cells':len(cells),'prompt_groups':len({c['prompt_id'] for c in cells}),
            'metrics':{k:cluster(cells,k) for k in keys}}

def main():
    fmt=module('aligned_format',ROOT/'experiments/EXP-034-explicit-format-alignment/code/prepare_alignment.py')
    traces=module('trace_rules',ROOT/'experiments/EXP-033-codex-source-contamination-audit/code/direct_trace_audit.py')
    historical=module('paired_helpers',ROOT/'experiments/EXP-035-aligned-paired-repeat-pilot/code/analyze_trial.py')
    paired=jl(PAIRED/'collection/combined-records.jsonl'); batch=jl(BATCH/'collection/combined-records.jsonl')
    jobs=jl(ROOT/'experiments/EXP-035-aligned-paired-repeat-pilot/request-plan.jsonl')
    plans={j['job_id']:j for j in jobs}
    for exp,total,status in [(PAIRED,78,'DISPATCH_COMPLETE_PARTIAL'),(BATCH,300,'COMPLETE')]:
        state=js(exp/'collection/state.json');summary=js(exp/'collection/summary.json');audit=js(exp/'metrics/collection-audit.json')
        assert state['status']==summary['status']==status and state['saved_total']==total
        assert state['run_id']==summary['run_id']==audit['run_id'] and not audit['integrity_errors']
    assert len(paired)==78 and len(batch)==300
    assert len({r['job_id'] for r in paired})==78 and len({r['job_key'] for r in batch})==300
    integrity=[];screen=[];flags=[]
    for kind,rows in [('paired',paired),('batch300',batch)]:
        for r in rows:
            raw=read(ROOT/r['attempt_path_project']/'final.txt')
            assert raw==r['text'].encode() and sha(raw)==r['text_sha256']
            if kind=='paired':
                j=plans[r['job_id']]
                assert all(r[k]==j[k] for k in ('route','repeat_id','prompt_id','model_label','user_message_sha256'))
            else:
                j={'user_message':r['prompt']}
            hits={k:bool(re.search(v,r['text'],re.I|re.M)) for k,v in {**traces.STRONG_PATTERNS,**traces.CONTEXT_PATTERNS}.items()}
            screen.append({'dataset':kind,'id':r.get('job_id',r.get('job_key')),'prompt_id':r['prompt_id'],'model':r.get('model_label',r['model_requested']),
                           'route':r.get('route','codex_subscription'),'strong_hits':[k for k in traces.STRONG_PATTERNS if hits[k]],
                           'weak_hits':[k for k in traces.CONTEXT_PATTERNS if hits[k]],'format_flags':fmt.format_flags(r['text']),
                           'word_count':words(r['text']),'target':r['target_words'],'length_flag':abs(words(r['text'])/r['target_words']-1)>.25})
    old_c=jl(ROOT/historical.OLD_CODEX);old_k=jl(ROOT/historical.OLD_KAGGLE)
    old=historical.historical_rows(old_c,old_k,jobs)
    grouped=co.defaultdict(dict)
    for r in paired:grouped[(r['prompt_id'],r['model_label'])][(r['route'],r['repeat_id'])]=r
    cells=[];missing=[]
    for (pid,model),lookup in sorted(grouped.items()):
        absent=[{'route':rt,'repeat':rep} for rt in ROUTES for rep in (1,2) if (rt,rep) not in lookup]
        if absent:missing.append({'prompt_id':pid,'model':model,'missing':absent});continue
        c={'prompt_id':pid,'model':model,'within_pairs':{},'cross_pairs':{}}
        for metric,fn in DISTANCES.items():
            within=[fn(lookup[(rt,1)]['text'],lookup[(rt,2)]['text']) for rt in ROUTES]
            cross=[fn(lookup[(ROUTES[0],i)]['text'],lookup[(ROUTES[1],j)]['text']) for i in (1,2) for j in (1,2)]
            c['within_pairs'][metric]=within;c['cross_pairs'][metric]=cross
            c[metric+'_within_codex']=within[0];c[metric+'_within_kaggle']=within[1]
            c[metric+'_within']=float(np.mean(within));c[metric+'_cross']=float(np.mean(cross));c[metric+'_excess']=c[metric+'_cross']-c[metric+'_within']
        for ft in features(lookup[(ROUTES[0],1)]['text']):
            c[ft+'_signed_codex_minus_kaggle']=float(np.mean([features(lookup[(ROUTES[0],i)]['text'])[ft] for i in (1,2)])-np.mean([features(lookup[(ROUTES[1],i)]['text'])[ft] for i in (1,2)]))
        c['old_word_trigram_cross']=lexical(*[next(r['text'] for r in old if r['prompt_id']==pid and r['model_label']==model and r['route']==rt) for rt in ROUTES])
        c['new_minus_old_cross_word_trigram']=c['word_trigram_jaccard_cross']-c['old_word_trigram_cross']
        cells.append(c)
    assert len(cells)==19 and len(missing)==1
    balanced=[c for c in cells if sum(v['prompt_id']==c['prompt_id'] for v in cells)==2]
    summaries={'overall':summarize(cells),'balanced_9_prompts':summarize(balanced),**{m:summarize([c for c in cells if c['model']==m]) for m in MODELS}}
    profiles={}
    for name,rows in [('old_paired',old),('aligned_paired',paired)]:profiles[name]=historical.source_profiles(rows,fmt.format_flags)
    profiles['aligned300']={'all':historical.profile(batch,fmt.format_flags),'by_model':{m:historical.profile([r for r in batch if r['model_requested']==m],fmt.format_flags) for m in sorted({r['model_requested'] for r in batch})}}
    old_idx={(r['prompt_id'],r['model_requested']):r for r in old_c}
    matched300=[old_idx[(r['prompt_id'],r['model_requested'])] for r in batch]
    profiles['matched_old300']=historical.profile(matched300,fmt.format_flags)
    pair_groups={r['prompt_group'] for r in paired};batch_groups={r['prompt_group'] for r in batch}
    assert not pair_groups&batch_groups
    train_groups={r['prompt_group'] for r in batch if r['split']=='train'};hold_groups={r['prompt_group'] for r in batch if r['split']=='holdout'}
    assert not train_groups&hold_groups
    assert len(train_groups)==60 and len(hold_groups)==15
    bynorm=co.defaultdict(list)
    for r in paired+batch:bynorm[norm(r['text'])].append(r.get('job_id',r.get('job_key')))
    duplicates=[ids for ids in bynorm.values() if len(ids)>1]
    old_matches=[{'new_ids':bynorm[norm(r['text'])],'old_key':r['job_key']} for r in old_c if norm(r['text']) in bynorm]
    result={'experiment':'EXP-041','observed_at':dt.datetime.now(dt.timezone.utc).isoformat(),
            'methods':{'seed':SEED,'bootstrap_resamples':BOOT,'unit':'whole base prompt, equal prompt weight and equal available model cells within prompt','metrics':'word trigram Jaccard / normalized whitespace character trigram count cosine / absolute whitespace word-count gap','excess':'mean of four cross-route pairs minus mean of two within-route pairs in each cell','interpretation':'surface text distances, not semantic loss, causal contamination probability or equivalence test','sensitivity':'nine prompts with both models complete; missing cell omitted, never imputed','fit_or_generation':'none'},
            'paired_responses':78,'complete_cells':cells,'missing_cells':missing,'summaries':summaries,'profiles':profiles,
            'direct_screen':{'responses':len(screen),'strong_candidates':[r for r in screen if r['strong_hits']],'weak_candidates':[r for r in screen if r['weak_hits']],'format_candidates':[r for r in screen if r['format_flags']],'length_candidates':[r for r in screen if r['length_flag']]},
            'integrity':{'raw_and_hash_verified':378,'duplicate_normalized_text_groups':duplicates,'new_vs_old_exact_normalized_overlap':old_matches,'paired_vs_batch_prompt_overlap':list(pair_groups&batch_groups),'batch_groups':len(batch_groups),'train_groups':len(train_groups),'holdout_groups':len(hold_groups),'batch_splits':dict(co.Counter(r['split'] for r in batch))}}
    save('metrics/analysis.json',result);save('metrics/row-screen.json',screen)
    read(Path(__file__));read(EXP/'config.json')
    save('metrics/input-hashes.json',{'files':SOURCES,'source_bytes_rechecked_after_analysis':all(sha((ROOT/p).read_bytes())==h for p,h in SOURCES.items())})
    print(json.dumps({'summaries':{k:{'cells':v['cells'],'prompt_groups':v['prompt_groups'],'metrics':{m:z for m,z in v['metrics'].items() if m.endswith('_excess') or m.endswith('_within') or m.endswith('_cross') or m=='words_signed_codex_minus_kaggle'}} for k,v in summaries.items()},'missing':missing,'screen':result['direct_screen'],'integrity':result['integrity']},ensure_ascii=False))

if __name__=='__main__': main()
