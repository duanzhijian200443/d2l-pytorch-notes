"""Verify V2 quick-commit outputs; this is not hidden-inference validation."""
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'outputs-v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))

def main():
    required=['input_preflight.json','reproduction_environment.json','a1_synthetic_parity.json','availability.json','submission_receipt.json','submission.parquet']
    assert all((OUT/n).is_file() for n in required)
    preflight=read('input_preflight.json'); env=read('reproduction_environment.json')
    parity=read('a1_synthetic_parity.json'); receipt=read('submission_receipt.json')
    assert preflight['status']=='PASS_EXP046_INPUT_PREFLIGHT'
    assert preflight['counts']=={'baseline_noncompetition_files':528,'baseline_small_files_hashed':398,'baseline_large_files_size_only':130,'extra_files_hashed':5}
    assert preflight['official_train_sha256']=='40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707'
    assert env['status']=='PASS' and env['cuda_test']=='PASS' and env['versions']['scikit-learn']=='1.6.1'
    assert len(env['gpus'])==2 and all('T4' in name for name in env['gpus'])
    assert parity['status']=='PASS' and parity['argmax_equal'] and parity['both_inference_routes_exercised']
    assert parity['maximum_raw_margin_difference']<=1e-12 and parity['maximum_probability_difference']<=1e-12
    assert receipt['experiment_id']=='EXP-046' and receipt['status']=='PASS'
    assert receipt['fast_commit'] is True and receipt['competition_rerun_flag_present'] is False
    assert receipt['prediction_kind']=='public_placeholder'
    assert receipt['submission_sha256']==sha(OUT/'submission.parquet')
    actual=pd.read_parquet(OUT/'submission.parquet'); test=pd.read_csv(ROOT/'test.csv')
    assert receipt['rows']==preflight['test_rows']==len(actual)==len(test)
    assert list(actual.columns)==['id','model'] and actual.id.is_unique and actual.model.notna().all()
    assert actual.id.astype(str).tolist()==test.id.astype(str).tolist()
    assert set(actual.model)=={'gpt-5.5'}
    result={'status':'PASS_QUICK_COMMIT_ONLY','observed_at':datetime.now(timezone.utc).isoformat(),
            'notebook_ref':'zhjianduan/name-that-llm-a1-public-stylegram-046','version':2,
            'rows':len(actual),'verified':['32 pinned input configuration and mounted content policy','T4 x2 CUDA','A1 synthetic numerical parity','public placeholder format'],
            'not_verified':['full-data A1 fit','complete hidden inference','leaderboard improvement'],
            'outputs':{n:sha(OUT/n) for n in required}}
    (HERE/'quick-commit-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
