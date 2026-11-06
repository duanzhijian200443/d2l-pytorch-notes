"""Descriptive source comparison only: no model training or generation."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
OUT = EXP / 'metrics'
MODELS = ['gpt-5.6-luna', 'gpt-5.6-terra', 'gpt-5.6-sol', 'gpt-6-astra']
SPECS = [
    ('abdullah-v2', 'abdullah1afify/ntl-generations/v2/ntl_generations.csv', 'whitespace-collapsed'),
    ('dophuchung-v1', 'dophuchung/ntl-proxy-generations/v1/ntl_proxy_generations.csv', 'whitespace-collapsed'),
    ('lucas-v7', 'lucasboesen/ntl-free-proxy-generations/v7/gen_passages.csv', 'single-paragraph-request'),
    ('prvs-raw-v6', 'prvsiyan/prvsiyans-extended-name-that-llm-dataset/v6/passages.csv', 'raw-with-explicit-format-request'),
    ('prvs-clean-v6', 'prvsiyan/prvsiyans-extended-name-that-llm-dataset/v6/passages_clean.csv', 'selected-subset-of-prvs-raw'),
    ('matched-v3', 'dhruvtibarewal/nlm-matched-clean/v3/nlm_matched_clean.parquet', 'aggregate-selected'),
    ('pool-v5', 'dhruvtibarewal/nlm-matched-clean/v3/nlm_pool_v5.parquet', 'aggregate-selected'),
    ('enai-v3', 'enaimaan/ntl-openai-generated-responses/v3/train.csv', 'raw-with-custom-system'),
    ('earthmaker-v1', 'earthmaker/ntl-own-prompt-generations/v1/passages.csv', 'single-paragraph-request-and-selection'),
]
FEATURES = ['newline', 'paragraph', 'bold', 'heading', 'bullet', 'numbered_list',
            'code_fence', 'em_dash', 'curly_quote', 'preface', 'assistant_self_reference',
            'closing_offer', 'role_markup', 'cjk']
PATTERNS = {
    'newline': r'[\r\n]', 'paragraph': r'\n\s*\n',
    'bold': r'\*\*[^*]+\*\*|__[^_]+__',
    'heading': r'(?m)^\s*#{1,6}\s+\S', 'bullet': r'(?m)^\s*[-*+]\s+\S',
    'numbered_list': r'(?m)^\s*\d+[.)]\s+\S', 'code_fence': r'```',
    'em_dash': '—', 'curly_quote': '[‘’“”]',
    'preface': r'(?i)^\s*(?:sure\b|certainly\b|absolutely\b|here(?:\x27s| is| are)\b)',
    'assistant_self_reference': r'(?i)\b(?:as an ai(?: language model)?|as a language model|i am an ai|i\x27m an ai)\b',
    'closing_offer': r'(?is)\b(?:let me know if|if you\x27d like,? i can|if you would like,? i can)\b.{0,180}$',
    'role_markup': r'(?i)<\|(?:im_start|im_end|assistant|system)\|>|</?(?:analysis|final|tool_call|system|developer)>',
    'cjk': '[\u4e00-\u9fff]',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def normalized_hash(text):
    normalized = ' '.join(unicodedata.normalize('NFKC', text).casefold().split())
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def expected_inputs():
    expected = {}
    for name in ['EXP-008-public-pool-audit', 'EXP-026-independent-blend-validation', 'EXP-027-multiclass-external-validation']:
        value = json.loads((ROOT/'experiments'/name/'download-manifest.json').read_text(encoding='utf-8'))
        for dataset in value.get('datasets', [value.get('dataset')]):
            if dataset:
                for entry in dataset['files']:
                    expected[entry['local_path']] = entry['sha256']
    expected['train.csv'] = json.loads((ROOT/'validation/prompt-group-v1/manifest.json').read_text(encoding='utf-8'))['train_sha256']
    expected['experiments/EXP-023-codex-subscription-pilot-r1/collection/records.jsonl'] = 'cb903279b3727b31ce2125b89c5ce8f530d769a984f49c2cb86e4fbefb1e9722'
    expected[(EXP/'inputs/kaggle-proxy-pilot-20.jsonl').relative_to(ROOT).as_posix()] = '227ed0103dee52a9a6bea6680a77e8e6409c7e289e99d404f311337db2cac269'
    return expected


def prepare_frame(name, df):
    result = df.copy().fillna('')
    if 'model' not in result:
        result['model'] = result['label'] if 'label' in result else result['model_requested']
    result['model'] = result['model'].astype(str).str.replace(r'^openai/', '', regex=True)
    result['text'] = result['text'].astype(str)
    result['source'] = name
    result['row_number'] = range(len(result))
    result['row_id'] = [f'{name}:{i}' for i in range(len(result))]
    result['words'] = result['text'].str.count(r'\S+')
    result['length_bin'] = pd.cut(result['words'], [-1,80,120,160,250,np.inf], labels=['0-80','81-120','121-160','161-250','251+']).astype(str)
    result['text_hash_normalized'] = result['text'].map(normalized_hash)
    for name, pattern in PATTERNS.items():
        result[name] = result['text'].str.contains(pattern, regex=True)
    return result


def profile(df):
    return {'n':len(df), 'words_mean':float(df.words.mean()), 'words_median':float(df.words.median()),
            'words_q10':float(df.words.quantile(.1)), 'words_q90':float(df.words.quantile(.9)),
            'counts':{name:int(df[name].sum()) for name in FEATURES},
            'rates':{name:float(df[name].mean()) for name in FEATURES},
            'length_bins':{str(k):int(v) for k,v in df.length_bin.value_counts().items()}}


def main():
    OUT.mkdir(exist_ok=True)
    expectations = expected_inputs()
    provenance = []
    def read(path, view):
        relative = path.relative_to(ROOT).as_posix()
        actual = sha(path)
        if actual != expectations[relative]:
            raise ValueError(f'Input hash mismatch: {relative}')
        provenance.append({'source':view,'path':relative,'sha256':actual,'bytes':path.stat().st_size})
        if path.suffix == '.jsonl':
            return pd.DataFrame(load_jsonl(path))
        return pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path, low_memory=False, keep_default_na=False)

    official = prepare_frame('official-train', read(ROOT/'train.csv','official-train'))
    codex = prepare_frame('codex', read(ROOT/'experiments/EXP-023-codex-subscription-pilot-r1/collection/records.jsonl','codex'))
    assert len(codex)==1000 and codex.model.value_counts().to_dict()==dict.fromkeys(MODELS,250)
    frames = {'codex':codex,'official-train':official}
    kinds = {'codex':'raw-codex-subscription','official-train':'official-unknown-generation-and-preprocessing'}
    for name, relative, kind in SPECS:
        frames[name] = prepare_frame(name, read(ROOT/'data/public-pools'/relative,name))
        kinds[name] = kind
    raw_public_names = [s[0] for s in SPECS]
    # These are separate views, never additional independent corpora.
    frames['prvs-raw-v6-new'] = frames['prvs-raw-v6'].loc[frames['prvs-raw-v6']['split'].eq('new')].copy()
    kinds['prvs-raw-v6-new'] = 'new-prompt-subset-of-prvs-raw'
    for base in ['abdullah-v2','dophuchung-v1']:
        frames[base+'-low'] = frames[base].loc[frames[base]['setting'].eq('low')].copy()
        kinds[base+'-low'] = 'low-reasoning-subset; whitespace-collapsed'
    for kind in ['gap','new']:
        frames['enai-v3-'+kind] = frames['enai-v3'].loc[frames['enai-v3']['kind'].eq(kind)].copy()
        kinds['enai-v3-'+kind] = 'user-only-single-paragraph' if kind == 'gap' else 'system-and-user-single-paragraph'
    prose_categories = set(codex.category.unique()) - {'creative_short_fiction','short_email','reasoning_explanation'}
    frames['codex-prose-subset'] = codex.loc[codex.category.isin(prose_categories)].copy()
    kinds['codex-prose-subset'] = '7-of-10-prompt-categories; subset-of-codex'
    pilot = prepare_frame('kaggle-paired20', read(EXP/'inputs/kaggle-proxy-pilot-20.jsonl','kaggle-paired20').rename(columns={'request_model':'model'}))
    frames['kaggle-paired20'] = pilot
    kinds['kaggle-paired20'] = 'own-small-pilot-not-public-corpus'

    summaries, by_model, standardized, rows_out, overlaps = {}, [], [], [], []
    codex_hashes = set(codex.text_hash_normalized)
    for name, full in frames.items():
        selected = full.loc[full.model.isin(MODELS)].copy()
        summaries[name] = {'total_rows':len(full),'four_model_rows':len(selected),'kind':kinds[name],
                           'selected_model_counts':{str(k):int(v) for k,v in selected.model.value_counts().items()},
                           'all_selected':profile(selected),
                           'balanced_model_count':int(selected.model.nunique()),
                           'model_balanced_rates':{feature:float(selected.groupby('model')[feature].mean().mean()) for feature in FEATURES}}
        for model, part in selected.groupby('model'):
            metric = profile(part)
            by_model.append({'source':name,'model':model,**{k:v for k,v in metric.items() if k not in ['counts','rates','length_bins']},**{f'{k}_count':v for k,v in metric['counts'].items()},**{f'{k}_rate':v for k,v in metric['rates'].items()}})
            target = codex.loc[codex.model.eq(model)]
            common = [b for b in target.length_bin.unique() if (part.length_bin.eq(b).sum()>=5 and target.length_bin.eq(b).sum()>=5)]
            support = target.loc[target.length_bin.isin(common)]
            if len(support):
                weights = support.length_bin.value_counts(normalize=True)
                rec = {'source':name,'model':model,'codex_supported_n':len(support),'codex_support_fraction':len(support)/len(target),'reference_n':int(part.length_bin.isin(common).sum()),'common_bins':';'.join(common)}
                for feature in FEATURES:
                    rec[f'{feature}_reference_rate'] = float(sum(weights[b]*part.loc[part.length_bin.eq(b),feature].mean() for b in common))
                    rec[f'{feature}_codex_rate'] = float(support[feature].mean())
                    rec[f'{feature}_difference_pp'] = 100*(rec[f'{feature}_codex_rate']-rec[f'{feature}_reference_rate'])
                standardized.append(rec)
        if name in raw_public_names+['official-train']:
            matching = full.loc[full.text_hash_normalized.isin(codex_hashes)]
            overlaps.append({'source':name,'all_label_rows':len(full),'normalized_exact_overlap_rows':len(matching),'unique_codex_hashes_matched':int(matching.text_hash_normalized.nunique())})
        rows_out.append(selected[['source','row_id','model','words','length_bin','text_hash_normalized']+FEATURES].assign(source=name))

    pairs = pilot.merge(codex, on=['prompt_id','model'], suffixes=('_kaggle','_codex'),validate='one_to_one')
    assert len(pairs)==20 and pairs.prompt_kaggle.eq(pairs.prompt_codex).all()
    paired = {'n':20,'independent_prompt_groups':int(pairs.prompt_id.nunique()),'mean_abs_relative_length_difference':float(((pairs.words_codex-pairs.words_kaggle).abs()/pairs.words_kaggle).mean()),
              'features':{f:{'kaggle':int(pairs[f+'_kaggle'].sum()),'codex':int(pairs[f+'_codex'].sum()),'codex_only':int((pairs[f+'_codex'] & ~pairs[f+'_kaggle']).sum()),'kaggle_only':int((pairs[f+'_kaggle'] & ~pairs[f+'_codex']).sum())} for f in FEATURES}}
    paired['input_token_offsets'] = {m:sorted(set(int(r.usage_codex[0]['input_tokens'])-int(r.usage_kaggle['prompt_tokens']) for r in pairs.loc[pairs.model.eq(m)].itertuples())) for m in pairs.model.unique()}

    category = codex.groupby(['model','category']).agg(n=('row_id','size'),words_mean=('words','mean'),**{f+'_rate':(f,'mean') for f in FEATURES}).reset_index()
    by_model_df = pd.DataFrame(by_model)
    by_model_df.to_csv(OUT/'by-source-model.csv',index=False)
    pd.DataFrame(standardized).to_csv(OUT/'length-standardized.csv',index=False)
    category.to_csv(OUT/'codex-by-category.csv',index=False)
    pd.concat(rows_out,ignore_index=True).to_csv(OUT/'row-features.csv',index=False)
    save(OUT/'input-integrity.json',{'status':'PASS','files':provenance})
    save(OUT/'source-profiles.json',summaries)
    save(OUT/'exact-overlap.json',overlaps)
    save(OUT/'paired-control.json',paired)
    concise = {name:{'n':s['four_model_rows'],'models':s['balanced_model_count'],'balanced_rates':{f:round(s['model_balanced_rates'][f]*100,2) for f in ['newline','bold','em_dash','preface','code_fence']}} for name,s in summaries.items()}
    save(OUT/'summary.json',{'observed_at':datetime.now(timezone.utc).isoformat(),'comparison':concise,'paired_control':paired,'training_or_generation':False,'interpretation':'Descriptive. Model/length controls do not control original prompt, system instructions, preprocessing or backend snapshot. Public views overlap and are not independent.'})
    print(json.dumps({'comparison':concise,'overlaps':overlaps,'paired_control':paired},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
