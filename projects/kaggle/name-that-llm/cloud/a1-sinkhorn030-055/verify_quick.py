"""Accept only the completed three-row public commit and write a submission handoff."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import pandas as pd
import control as c

def main():
    state=c.base.state()
    assert state['experiment_id']=='EXP-055' and state['notebook_ref']==c.REF
    assert state['save_request_count']==state['run_request_count']==1
    assert state['run_version']==state['last_status']['version']==state['downloaded_version']==2
    assert state['last_status']['status']=='COMPLETE'
    out=c.HERE/'outputs-v2'
    binding=state['readback_receipts']['2']
    rb=c.HERE/binding['path']; assert c.base.sha(rb)==binding['sha256']
    readback=c.base.read(rb)
    assert readback['status']=='PASS' and readback['ref']==c.REF and readback['version']==2
    folder,meta,local=c.local_package()
    assert readback['local_bindings']==local
    cloud_meta=c.base.read(rb.parent/'kernel-metadata.json')
    remote=c.base.verify_readback(meta,cloud_meta,folder/meta['code_file'],rb.parent/cloud_meta['code_file'])
    assert remote['source_canonical_json_sha256']==local['source_canonical_json_sha256'] and remote['input_count']==32
    sid=c.base.read(c.HERE/'script-identity.json')
    assert sid['ref']==c.REF and sid['version']==2 and sid['script_version_id']==state['script_version_id']>0
    manifest=c.base.read(c.HERE/'outputs-v2-manifest.json')
    assert c.base.sha(c.HERE/'outputs-v2-manifest.json')==state['outputs_manifest_sha256']
    hashes={item['path']:item['sha256'] for item in manifest['files']}
    for name,digest in hashes.items(): assert c.base.sha(out/name)==digest
    required={'submission.parquet','submission_receipt.json','reproduction_environment.json','a1_synthetic_parity.json','availability.json','input_preflight.json'}
    assert required.issubset(hashes)
    pre,env,parity,avail,receipt=[c.base.read(out/n) for n in
        ('input_preflight.json','reproduction_environment.json','a1_synthetic_parity.json','availability.json','submission_receipt.json')]
    assert pre['status']=='PASS_EXP046_INPUT_PREFLIGHT'
    assert pre['counts']=={'baseline_noncompetition_files':528,'baseline_small_files_hashed':398,
                          'baseline_large_files_size_only':130,'extra_files_hashed':5}
    assert pre['official_train_sha256']=='40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707'
    assert pre['manifest_sha256']=='de7a305b5d26a5ad0cca0ed41cf31e2af1f85037dd354286aeef77ac67aaabf5'
    assert avail['status']=='RESOLVED_PATH_AVAILABILITY' and avail['weights_changed'] is False
    assert env['status']==env['cuda_test']=='PASS' and len(env['gpus'])==2 and all('T4' in v for v in env['gpus'])
    for name,version in {'numpy':'2.1.3','pandas':'2.3.3','scipy':'1.16.3','scikit-learn':'1.6.1'}.items():
        assert env['versions'][name]==version
    assert parity['status']=='PASS' and parity['argmax_equal'] and parity['both_inference_routes_exercised']
    assert max(parity['maximum_raw_margin_difference'],parity['maximum_probability_difference'])<=1e-12
    assert receipt['status']=='PASS' and receipt['experiment_id']=='EXP-049' and receipt['sinkhorn']==.30
    assert receipt['fast_commit'] is True and receipt['competition_rerun_flag_present'] is False
    assert receipt['prediction_kind']=='public_placeholder' and receipt['atlas_w']==.20 and receipt['comp_w']==.15
    parquet=pd.read_parquet(out/'submission.parquet');public=pd.read_csv(c.ROOT/'test.csv')
    assert receipt['rows']==pre['test_rows']==len(parquet)==len(public)==3
    assert list(parquet.columns)==['id','model'] and parquet.id.is_unique and parquet.id.tolist()==public.id.tolist()
    assert set(parquet.model)=={'gpt-5.5'} and receipt['submission_sha256']==hashes['submission.parquet']
    evidence=[rb,c.HERE/'outputs-v2-manifest.json',c.HERE/'script-identity.json',c.HERE/state['last_status']['path']]
    evidence.extend(out/n for n in sorted(required))
    ready={'status':'READY_FOR_SUBMISSION','experiment_id':'EXP-055','notebook_experiment_id':'EXP-049',
           'ref':c.REF,'version':2,'script_version_id':sid['script_version_id'],'source_sha256':c.SOURCE,
           'source_canonical_sha256':remote['source_canonical_json_sha256'],'metadata_sha256':c.METADATA,
           'submission_sha256':hashes['submission.parquet'],'private':True,'internet':False,'asset_count':32,
           'verified_at':c.base.now(),'verified_gpus':env['gpus'],'rows':3,'sinkhorn':.30,'source_scope':'public quick commit',
           'evidence':[{'path':p.relative_to(c.ROOT).as_posix(),'sha256':c.base.sha(p)} for p in evidence],
           'input_hash_policy':pre['counts'],'hidden_run_status':'NOT_SUBMITTED_BY_THIS_CONTROLLER'}
    c.base.save(c.HERE/'ready-for-submission.json',ready,exclusive=True)
    c.base.update(status='READY_FOR_SUBMISSION',gpu_may_be_active=False,submission_request_count=0)
    print(f"READY_FOR_SUBMISSION {c.REF} V2 scriptVersionId={sid['script_version_id']}")

if __name__=='__main__': main()
