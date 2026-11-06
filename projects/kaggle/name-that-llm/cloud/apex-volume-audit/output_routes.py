"""Exercise unchanged final submission writer on three real public rows, including failure paths."""
import ast
import json
import os
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def require(value,message):
    if not value:raise ValueError(message)


def main():
    base=json.loads((ROOT/'cloud/apex-zero-054/kernel/apex-zero.ipynb').read_text())
    real=pd.read_csv(ROOT/'test.csv',dtype={'id':str})
    require(len(real)==3 and real.id.is_unique,'Public example has changed')
    records={}
    for count,n in [(120,56),(240,57)]:
        pkg=ROOT/f'cloud/apex-codex{count}-{n:03}/kernel/apex.ipynb'
        cells=json.loads(pkg.read_text())['cells']
        require(cells[2]==base['cells'][2] and cells[16]==base['cells'][16],'Launch or writer differs from the baseline')
        setup=''.join(cells[2]['source'])
        assignment=next(line for line in setup.splitlines() if line.startswith('FAST_COMMIT = '))
        src=''.join(cells[16]['source']).split("if FAST_COMMIT:   # fork: placeholder",1)
        require(len(src)==2 and '--atlas_budget' in src[0] and "ATLAS_BUDGET = int(8.3 * 3600 - COMP_SEC - 600)" in src[0], 'Budget/writer differs')
        tail="if FAST_COMMIT:   # fork: placeholder"+src[1]
        status={}
        for case in ('public_three_rows','hidden_three_rows','hidden_main_failure','hidden_duplicate_ids'):
            with tempfile.TemporaryDirectory(dir=HERE) as tmp,patch.dict(os.environ):
                wd=Path(tmp);real.to_csv(wd/'test.csv',index=False)
                if case=='public_three_rows':os.environ.pop('KAGGLE_IS_COMPETITION_RERUN',None)
                else:os.environ['KAGGLE_IS_COMPETITION_RERUN']='1'
                env={'os':os,'pd':pd,'_pd':pd,'DATA':tmp,'cmd':['fixed-infer-sentinel'],'subprocess':subprocess}
                exec(compile(assignment,'<candidate FAST declaration>','exec'),env)
                require(env['FAST_COMMIT']==(case=='public_three_rows'),'Wrong fast routing')
                calls=[]
                def child(command,check):
                    require(check and command==['fixed-infer-sentinel'],'Main child failure ignored')
                    calls.append(command)
                    if case=='hidden_main_failure':raise subprocess.CalledProcessError(1,command)
                    ids=real.id.tolist() if case!='hidden_duplicate_ids' else [real.id.iloc[0]]*3
                    pd.DataFrame({'id':ids,'model':['gpt-5.6-luna']*3}).to_parquet(wd/'submission.parquet',index=False)
                caught=None
                try:
                    with patch.object(subprocess,'run',child):
                        exec(compile(tail.replace('/kaggle/working',wd.as_posix()),'<candidate writer>','exec'),env)
                except (subprocess.CalledProcessError,AssertionError) as err:caught=err
                ok=case in ('public_three_rows','hidden_three_rows')
                require((caught is None)==ok and len(calls)==int(case!='public_three_rows'),'Wrong final writer result')
                if ok:
                    got=pd.read_parquet(wd/'submission.parquet')
                    require(len(got)==3 and got.id.astype(str).tolist()==real.id.tolist() and set(got.model)==
                            ({'gpt-5.5'} if case=='public_three_rows' else {'gpt-5.6-luna'}),'Output ID/labels not preserved')
                status[case]='EXPECTED_PASS' if ok else 'EXPECTED_REJECTION'
        records[count]=status
    result={'status':'PASS_FINAL_OUTPUT_ROUTES','public_rows':3,'arms':records,
            'scope':'Real candidate FAST declaration and final writer, real pandas parquet roundtrip. Main child simulated; no hidden model run.'}
    (HERE/'output-route-verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
