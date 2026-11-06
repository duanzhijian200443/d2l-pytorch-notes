"""Create a traceable text-only view; never train, upload, or overwrite raw data."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter,defaultdict
import hashlib,json,re,unicodedata
from transform import transform

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SOURCE=ROOT/'experiments/EXP-023-codex-subscription-pilot-r1/collection/records.jsonl'
ALIGNED=ROOT/'experiments/EXP-040-codex300-concurrency2/collection/combined-records.jsonl'
EXPECTED='cb903279b3727b31ce2125b89c5ce8f530d769a984f49c2cb86e4fbefb1e9722'
def digest(v):return hashlib.sha256(v).hexdigest()
def sha(p):return digest(p.read_bytes())
def textsha(v):return digest(v.encode('utf-8'))
def norm(v):return ' '.join(unicodedata.normalize('NFKC',v).casefold().split())
def readlines(p):return [json.loads(x) for x in p.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def save_lines(p,rows):p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
def entry(p):return {'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size}
def duplicate_groups(rows,normalizer):
    groups=defaultdict(list)
    for r in rows:groups[textsha(normalizer(r['text']))].append(r['job_key'])
    return [keys for keys in groups.values() if len(keys)>1]

def main():
    assert sha(SOURCE)==EXPECTED
    assert sha(ALIGNED)=='2d6c691016a6ae6f789d5c6a625a207a3ec89e5093bf804c5052b11c90a6125b'
    out=HERE/'derived';assert not out.exists(),'Refusing to overwrite a derived version'
    config={'schema_version':1,'experiment_id':'EXP-043','view_id':'codex-original-newline-strong-v1',
        'scope':'Derived formatting only, not classifier training or model-source equivalence',
        'source':entry(SOURCE),'transforms':['Replace each CR/LF run with one ASCII space; preserve all other whitespace',
            'Remove paired ordinary ** delimiters outside inline code, preserving enclosed body'],
        'keep':['raw source bytes','word order and boundaries','all non-format body characters','single-star emphasis',
            'inline code','headings/list markers','labels','prompt groups','train/holdout assignment'],
        'reject':['Ambiguous unprotected double-star runs','Changed word token sequence','New duplicate/cross-split collisions'],
        'training_started':False,'formal_pool_accepted':False,'gpu_seconds':0,'api_usd':0}
    save(HERE/'config.json',config)
    source=readlines(SOURCE);assert len(source)==1000 and len({r['job_key'] for r in source})==1000
    by_pair={(r['prompt_group'],r['model_requested']):r for r in source};assert len(by_pair)==1000
    transformed=[];audits=[];counts=Counter();group_splits=defaultdict(set)
    for i,r in enumerate(source):
        assert textsha(r['text'])==r['text_sha256']
        a=transform(r['text']);assert not a['ambiguous_double_star_runs'],r['job_key']
        assert transform(a['text'])['text']==a['text']
        row={k:r[k] for k in ('job_key','prompt_id','prompt_group','split','category','target_words','prompt',
            'model_requested','model_observed','model_observation_scope','reasoning_effort')}
        row.update(text=a['text'],text_sha256=textsha(a['text']),raw_text_sha256=r['text_sha256'],
            source_record_index=i,source_artifact=config['source']['path'],source_artifact_sha256=EXPECTED,
            view_id=config['view_id'],source_protocol='original_without_explicit_single_paragraph_constraint',
            derived_formatting=True,formal_pool_accepted=False,training_started=False)
        transformed.append(row);group_splits[row['prompt_group']].add(row['split'])
        audits.append({'job_key':r['job_key'],'source_record_index':i,'split':r['split'],
            'raw_text_sha256':r['text_sha256'],'derived_text_sha256':row['text_sha256'],
            'changed':r['text']!=a['text'],'bold_pairs':a['bold_pairs'],'newline_runs':a['newline_runs'],
            'edits':a['edits'],'word_sequence_equal':True,'body_characters_equal':True,'idempotent':True})
        counts.update({'rows':1,'changed_rows':r['text']!=a['text'],'bold_rows':a['bold_pairs']>0,
            'bold_pairs':a['bold_pairs'],'newline_rows':a['newline_runs']>0,'newline_runs':a['newline_runs']})
    assert all(len(v)==1 for v in group_splits.values())
    assert (counts['rows'],counts['changed_rows'],counts['bold_rows'],counts['bold_pairs'],counts['newline_rows'])==(1000,727,215,537,709)
    assert Counter(r['split'] for r in transformed)=={'train':800,'holdout':200}
    assert not duplicate_groups(source,str) and not duplicate_groups(source,norm)
    assert not duplicate_groups(transformed,str) and not duplicate_groups(transformed,norm)
    for normalizer in (str,norm):
        original_hashes={textsha(normalizer(r['text'])):r['job_key'] for r in source}
        for r in transformed:
            match=original_hashes.get(textsha(normalizer(r['text'])))
            assert match is None or match==r['job_key'],'Cross-record source/view collision'
    aligned=readlines(ALIGNED);targets={'gpt-5.6-luna','gpt-5.6-terra'}
    matched=[]
    for r in aligned:
        if r['split']=='train' and r['model_requested'] in targets:
            original=by_pair[(r['prompt_group'],r['model_requested'])]
            assert original['split']=='train' and original['prompt']==r['base_prompt']
            matched.append({'job_key':original['job_key'],'aligned_job_key':r['job_key'],
                'prompt_group':r['prompt_group'],'model_requested':r['model_requested']})
    assert len(matched)==120
    # No row filtering by label/content metrics: all1000 are retained, separated
    # into their pre-existing train/holdout roles. Selections contain keys only.
    out.mkdir()
    for split in ('train','holdout'):save_lines(out/(split+'.jsonl'),[r for r in transformed if r['split']==split])
    save_lines(out/'row-changes.jsonl',audits)
    save(out/'selections.json',{'luna_terra_train400':[r['job_key'] for r in transformed if r['split']=='train' and r['model_requested'] in targets],
        'matched_with_aligned120':matched,'note':'Selection index only; no classifier has used this view.'})
    metrics={'status':'PASS_DERIVED_FORMAT_VIEW','observed_at':datetime.now(timezone.utc).isoformat(),
        'counts':dict(counts),'train_rows':800,'holdout_rows':200,'train_prompt_groups':200,'holdout_prompt_groups':50,
        'rows_per_model':dict(Counter(r['model_requested'] for r in transformed)),
        'raw_source_sha256_before_and_after':EXPECTED,'remaining_newline_rows':0,'remaining_bold_rows':0,
        'body_and_word_sequence_checks':1000,'idempotence_checks':1000,
        'exact_and_normalized_duplicate_groups':0,'cross_split_collision_groups':0,'cross_job_original_view_collisions':0,
        'training_started':False,'formal_pool_accepted':False,
        'limits':['Not identical to regeneration with a single-paragraph instruction.','Single-star emphasis and other Markdown markers remain.',
            'Formatting removal can also remove model attribution signal; efficacy is untested.','Source model labels remain requested_only.']}
    assert sha(SOURCE)==EXPECTED
    save(HERE/'metrics/derivation.json',metrics)
    save(HERE/'manifest.json',{'schema_version':1,'experiment_id':'EXP-043','freeze_status':'DERIVED_VIEW_COMPLETE_NOT_TRAINED',
        'config':entry(HERE/'config.json'),'inputs':[entry(SOURCE),entry(ALIGNED)],
        'code':[entry(HERE/n) for n in ('transform.py','test_transform.py','build_view.py')],
        'outputs':[entry(p) for p in sorted(out.iterdir()) if p.is_file()],
        'metrics':entry(HERE/'metrics/derivation.json')})
    print(json.dumps(metrics,ensure_ascii=False))

if __name__=='__main__':main()
