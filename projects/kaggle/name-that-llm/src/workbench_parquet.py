"""Verify registered public commit samples, without claiming hidden inference."""
import argparse
import pandas as pd
from workbench_core import *

def verify_extra_receipt(p, extra):
    require(p['id']=='EXP-048','EXTRA_RECEIPT_NOT_REGISTERED')
    require(extra.get('status')=='PASS' and extra.get('new_rows')==120
            and extra.get('new_groups')==60 and extra.get('holdout_overlap')==0,
            'ADDED_DATA_PREFLIGHT_INVALID')
    require(extra.get('new120_sha256')==p['receipt_identity']['added_data_sha256'],
            'ADDED_DATA_HASH_DIFFERS')

def verify(p,collection):
    remote=p['remote'];folder=inside(collection['files_path']);verify_files(collection,folder)
    require(collection['ref']==remote['ref'] and collection['version']==remote['version'] and collection['script_version_id']==remote['script_version_id'],'COLLECTION_TARGET_MISMATCH')
    receipt=read(folder/'submission_receipt.json')
    for key,value in p['receipt_identity'].items():require(receipt.get(key)==value,'SUBMISSION_RECEIPT_IDENTITY_MISMATCH')
    require(receipt.get('status')=='PASS','SUBMISSION_RECEIPT_FAILED')
    require(receipt.get('fast_commit') is True and receipt.get('prediction_kind')=='public_placeholder'
            and receipt.get('competition_rerun_flag_present') is False,'ONLY_PUBLIC_COMMIT_VERIFIER_AVAILABLE')
    for key,value in p['expected_parameters'].items():require(receipt.get(key)==value,'RECEIPT_PARAMETER_MISMATCH')
    path=folder/'submission.parquet';require(sha(path)==receipt['submission_sha256'],'PARQUET_RECEIPT_HASH_MISMATCH')
    train=ROOT/'train.csv';require(sha(train)=='40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707','OFFICIAL_TRAIN_CHANGED')
    labels=set(pd.read_csv(train).model.astype(str));test=pd.read_csv(ROOT/'test.csv');actual=pd.read_parquet(path)
    require(list(actual.columns)==['id','model'] and actual.notna().all().all(),'PARQUET_SCHEMA_INVALID')
    require(actual.id.is_unique and actual.id.astype(str).tolist()==test.id.astype(str).tolist(),'PUBLIC_IDS_DIFFER')
    require(len(actual)==receipt['rows'] and set(actual.model).issubset(labels),'PARQUET_ROWS_OR_LABELS_INVALID')
    require(set(actual.model)=={'gpt-5.5'},'UNEXPECTED_PUBLIC_PLACEHOLDER')
    inputs=read(folder/'input_preflight.json')
    require(inputs.get('status')=='PASS_EXP046_INPUT_PREFLIGHT' and inputs['test_rows']==len(actual),'INPUT_PREFLIGHT_INVALID')
    require(inputs['manifest_sha256']=='de7a305b5d26a5ad0cca0ed41cf31e2af1f85037dd354286aeef77ac67aaabf5','WRONG_INPUT_MANIFEST')
    require(inputs['counts']=={'baseline_noncompetition_files':528,'baseline_small_files_hashed':398,'baseline_large_files_size_only':130,'extra_files_hashed':5},'INPUT_COUNTS_DIFFER')
    if p['id']=='EXP-048':verify_extra_receipt(p,read(folder/'add120_preflight.json'))
    env=read(folder/'reproduction_environment.json');parity=read(folder/'a1_synthetic_parity.json')
    require(env.get('status')=='PASS' and env.get('cuda_test')=='PASS','CUDA_PREFLIGHT_FAILED')
    require(env['versions']['scikit-learn']=='1.6.1' and len(env['gpus'])==2 and all('T4' in g for g in env['gpus']),'ENVIRONMENT_DIFFERS')
    require(parity.get('status')=='PASS' and parity.get('argmax_equal') is True and parity.get('both_inference_routes_exercised') is True,'SYNTHETIC_PARITY_FAILED')
    for name in ['maximum_raw_margin_difference','maximum_probability_difference']:
        value=float(parity[name]);require(math.isfinite(value) and 0<=value<=1e-12,'SYNTHETIC_PARITY_DIFFERS')
    return {'status':'PASS_COMMIT_SAMPLE_ONLY','observed_at':now(),'scope':'public_placeholder_and_input_environment_synthetic_checks',
            'rows':len(actual),'ref':remote['ref'],'version':remote['version'],'script_version_id':remote['script_version_id'],
            'verified':['output hashes and public IDs','registered recipe identity','mounted-input policy','T4 x2 CUDA','synthetic numerical parity'],
            'not_verified':['complete hidden model execution','real-data fitting success','hidden model/ATLAS coverage','leaderboard gain'],
            'large_weights_rehashed':False,'read_only_verification':True}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('experiment');parser.add_argument('destination');args=parser.parse_args()
    dest=inside(args.destination)
    try:
        _,p,_=profile(args.experiment);collection=latest(p['id'],'collect');require(collection is not None,'COLLECT_OUTPUTS_FIRST')
        result=verify(p,collection)
    except Exception as error:result=safe_failure(error)
    atomic_json(dest/'backend.json',result,exclusive=True);print(result['status']);return int(result['status']=='ERROR')

if __name__=='__main__':raise SystemExit(main())
