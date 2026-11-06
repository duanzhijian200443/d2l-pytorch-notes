"""Build EXP-031 from frozen EXP-029 inputs; no downloads, model runs or saves."""
from datetime import datetime, timezone
import base64
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

import input_policy
import runner

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'cloud/full-stack-capture'
EXP=ROOT/'experiments/EXP-031-full-stack-capture-r1'
PREFLIGHT_SHA='de7a305b5d26a5ad0cca0ed41cf31e2af1f85037dd354286aeef77ac67aaabf5'
PANEL_SHA='8eeec7f64e924f0fd79dca57d324d42d269f400bdfbfff76d60b4c168d9ae6f2'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    origin=OLD/'launch-v1/payload'
    old_plan=json.loads((origin/'plan.json').read_text())
    for name,digest in old_plan['input_hashes'].items():
        assert sha(origin/name)==digest,('Changed frozen old payload',name)
    preflight=origin/'preflight_manifest.json'
    assert sha(preflight)==PREFLIGHT_SHA and sha(origin/'panel/test.csv')==PANEL_SHA
    manifest=json.loads(preflight.read_text())
    assert manifest['status']=='PASS' and len(manifest['files'])==531
    availability=input_policy.resolve_input_policy([x['path'] for x in manifest['files']])
    payload=HERE/'payload';assert not payload.exists(),'Existing payload; refuse to overwrite frozen launch'
    payload.mkdir(parents=True)
    unchanged=['setup.py','model_args.py','infer.py','comp_stack.py','gauss.py',
               'full_stack_cache.py','full_stack_observer.py','full_stack_replay.py','runtime-build.json','preflight_manifest.json']
    for name in unchanged:shutil.copyfile(origin/name,payload/name)
    for name in ['runner.py','input_policy.py']:shutil.copyfile(HERE/name,payload/name)
    (payload/'panel').mkdir();shutil.copyfile(origin/'panel/test.csv',payload/'panel/test.csv')
    save(payload/'availability.json',availability)
    (payload/'ATTRIBUTION.md').write_text((origin/'ATTRIBUTION.md').read_text().replace('EXP-029','EXP-031'),encoding='utf-8')
    plan=dict(old_plan)
    plan.update(experiment_id='EXP-031',run_id='exp031-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'),
                expected_comp_members=availability['expected_comp_members'],allowed_omissions=availability['allowed_omissions'],
                availability_sha256=sha(payload/'availability.json'),
                retry_of='EXP-029',retry_change='Shared CPU/GPU availability policy; original model math unchanged')
    plan['input_hashes']={p.relative_to(payload).as_posix():sha(p) for p in sorted(payload.rglob('*')) if p.is_file()}
    save(payload/'plan.json',plan)
    runner.validate_plan(plan);runner.verify_payload(payload,plan)
    # Replay exactly the pre-launch resolver on the same preflight path inventory.
    input_policy.assert_input_policy_equal(availability,input_policy.resolve_input_policy([x['path'] for x in manifest['files']]))
    archive=io.BytesIO()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(payload.rglob('*')):
            if path.is_file():bundle.write(path,path.relative_to(payload).as_posix())
    blob=archive.getvalue();blob_sha=hashlib.sha256(blob).hexdigest()
    allowed=sorted(plan['input_hashes'])+['plan.json']
    source='''# EXP-031: bounded private cache capture; no submission.\nimport base64,hashlib,io,pathlib,subprocess,sys,zipfile\np=pathlib.Path('/kaggle/working/payload');p.mkdir(exist_ok=False)\nblob=base64.b64decode('''+repr(base64.b64encode(blob).decode('ascii'))+''')\nassert hashlib.sha256(blob).hexdigest()=='''+repr(blob_sha)+'''\nwith zipfile.ZipFile(io.BytesIO(blob)) as z:\n    assert sorted(z.namelist())=='''+repr(sorted(allowed))+'''\n    assert all(not pathlib.PurePosixPath(n).is_absolute() and '..' not in pathlib.PurePosixPath(n).parts for n in z.namelist())\n    z.extractall(p)\nsubprocess.run([sys.executable,'-B','-u',str(p/'runner.py')],check=True)\n'''
    compile(source,'capture.py','exec')
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},
              'cells':[{'cell_type':'code','id':'exp031-capture','metadata':{},'source':source,'outputs':[],'execution_count':None}]}
    kernel=HERE/'kernel';kernel.mkdir(exist_ok=False);save(kernel/'capture.ipynb',notebook)
    metadata=json.loads((OLD/'launch-notebook-v2/kernel/kernel-metadata.json').read_text())
    metadata.update(id='zhjianduan/name-that-llm-cache-capture-031',title='Name That LLM Cache Capture 031',code_file='capture.ipynb',kernel_type='notebook')
    save(kernel/'kernel-metadata.json',metadata)
    proof={'status':'PASS_LOCAL_CPU_PREFLIGHT_AND_PACKAGE','observed_at_utc':datetime.now(timezone.utc).isoformat(),
           'experiment_id':'EXP-031','run_id':plan['run_id'],'preflight_manifest_sha256':PREFLIGHT_SHA,
           'panel_sha256':PANEL_SHA,'availability_sha256':plan['availability_sha256'],'expected_comp_members':plan['expected_comp_members'],
           'allowed_omissions':plan['allowed_omissions'],'unchanged_algorithm_files':{name:sha(payload/name) for name in unchanged},
           'repaired_code':{name:sha(payload/name) for name in ['runner.py','input_policy.py']},'builder_sha256':sha(Path(__file__)),
           'plan_sha256':sha(payload/'plan.json'),'notebook_sha256':sha(kernel/'capture.ipynb'),'metadata_sha256':sha(kernel/'kernel-metadata.json'),
           'payload_zip_sha256':blob_sha,'payload_zip_bytes':len(blob),'file_count':len(allowed),'weights_rehashed':False,
           'labels_uploaded':False,'new_model_execution':False,'gpu_started':False}
    save(EXP/'metrics/package-preflight.json',proof)
    print(json.dumps({k:proof[k] for k in ['status','run_id','file_count','payload_zip_bytes','gpu_started']}))


if __name__=='__main__':main()
