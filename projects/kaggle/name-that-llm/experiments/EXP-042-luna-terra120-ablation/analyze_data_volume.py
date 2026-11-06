"""Descriptive inventory/weight analysis only; no training or generation."""
from pathlib import Path
from collections import Counter
import ast,hashlib,json,re
import pandas as pd

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=ROOT/'experiments/EXP-023-codex-subscription-pilot-r1/collection/records.jsonl'
NEW=ROOT/'experiments/EXP-040-codex300-concurrency2/collection/combined-records.jsonl'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def records(p):return [json.loads(x) for x in p.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def main():
    assert sha(OLD)=='cb903279b3727b31ce2125b89c5ce8f530d769a984f49c2cb86e4fbefb1e9722'
    assert sha(NEW)=='2d6c691016a6ae6f789d5c6a625a207a3ec89e5093bf804c5052b11c90a6125b'
    old,new=records(OLD),records(NEW);targets={'gpt-5.6-luna','gpt-5.6-terra'}
    ot=[r for r in old if r['split']=='train'];olt=[r for r in ot if r['model_requested'] in targets]
    nt=[r for r in new if r['split']=='train'];nlt=[r for r in nt if r['model_requested'] in targets]
    oldgroups={r['prompt_group'] for r in olt};newgroups={r['prompt_group'] for r in nlt}
    assert newgroups<=oldgroups
    code=ROOT/'experiments/EXP-033-codex-source-contamination-audit/code/compare_sources.py'
    tree=ast.parse(code.read_text(encoding='utf-8'))
    patterns=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PATTERNS' for t in n.targets))
    format_code=ROOT/'experiments/EXP-034-explicit-format-alignment/code/prepare_alignment.py'
    ftree=ast.parse(format_code.read_text(encoding='utf-8'))
    function=next(n for n in ftree.body if isinstance(n,ast.FunctionDef) and n.name=='format_flags')
    namespace={'re':re};exec(compile(ast.Module(body=[function],type_ignores=[]),str(format_code),'exec'),namespace)
    format_flags=namespace['format_flags']
    def profile(rs):
        return {'answers':len(rs),'prompt_groups':len({r['prompt_group'] for r in rs}),
            'models':dict(Counter(r['model_requested'] for r in rs)),
            'splits':dict(Counter(r['split'] for r in rs)),
            'newline':sum(bool(re.search(patterns['newline'],r['text'])) for r in rs),
            'bold':sum(bool(re.search(patterns['bold'],r['text'])) for r in rs),
            'no_exp034_format_flags':sum(not format_flags(r['text']) for r in rs),
            'no_format_flags_by_model':dict(Counter(r['model_requested'] for r in rs if not format_flags(r['text'])))}
    p=ROOT/'cloud/luna-terra120-042/payload';train=pd.read_csv(p/'train.csv');folds=pd.read_csv(p/'folds.csv');extra=pd.read_csv(p/'extra.csv');additional=pd.read_csv(p/'new120.csv')
    weights=[]
    for fold in range(5):
        tr=train[folds.fold.ne(fold)];counts=tr.model.value_counts();pub=extra.model.value_counts()
        mass=sum(.5*counts[m]/pub[m] for m in additional.model)
        weights.append({'fold':fold,'official_train_rows':len(tr),'baseline_raw_rows':len(tr)+len(extra),
            'added_raw_rows':len(additional),'additional_rows_over_base':len(additional)/(len(tr)+len(extra)),
            'added_sample_weight_mass':mass,'baseline_sample_weight_mass':len(tr),
            'added_fraction_of_final_sample_weight':mass/(len(tr)+mass),
            'target_class_added_mass_over_base':30/694})
    result={'scope':'Read-only inventory and loss-weight accounting; not a learning curve or new training result',
        'inputs':[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for p in [OLD,NEW,code,format_code]],
        'old1000':profile(old),'old_train800':profile(ot),'old_luna_terra400':profile(olt),
        'old_matched120':profile([r for r in olt if r['prompt_group'] in newgroups]),
        'new300':profile(new),'new_train240':profile(nt),'new_luna_terra120':profile(nlt),
        'old_extra_training_prompt_groups':len(oldgroups-newgroups),'old_luna_terra_rows_on_extra_prompts':sum(r['prompt_group'] not in newgroups for r in olt),
        'old_train800_plus_new120_unique_prompt_groups':len({r['prompt_group'] for r in ot+nlt}),
        'holdout_prompt_union':len({r['prompt_group'] for r in old+new if r['split']=='holdout'}),
        'fold_weight_accounting':weights,'weight_fraction_is_not_information_fraction':True,
        'new_generation':False,'classifier_training':False,'formal_pool_ingestion':False}
    (HERE/'metrics/data-volume-analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'old_train':profile(ot),'old_LT':profile(olt),'new_LT':profile(nlt),'extra_groups':len(oldgroups-newgroups),'weight_fraction':weights[0]['added_fraction_of_final_sample_weight']},ensure_ascii=False))
if __name__=='__main__':main()
