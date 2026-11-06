"""Build one private capture upload after CPU preflight; never launch it here."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

import build_runtime
import runner

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
EXP=ROOT/'experiments/EXP-029-full-stack-capture'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def validate_preflight(preflight,config):
    assert preflight['schema_version']==1 and preflight['status']=='PASS'
    assert not preflight.get('synthetic_test_receipt'),'Synthetic receipts cannot authorize a real package'
    assert preflight['attachment_bindings']==config['attachment_bindings']
    assert preflight['preflight_sha256']==sha(HERE/'preflight.py')
    assert preflight['config_sha256']==sha(HERE/'preflight_config.json')
    files=preflight.get('files')
    assert isinstance(files,list) and files,'No audited input files'
    assert len({x['path'] for x in files})==len(files),'Duplicate audited paths'
    for item in files:
        path=PurePosixPath(item['path'])
        assert path.parts[:3]==('/','kaggle','input') and '..' not in path.parts
        assert type(item['bytes']) is int and item['bytes']>=0 and runner.digest_string(item['sha256'])
    assert sum(x['bytes'] for x in files)==preflight['bytes_total']
    wanted={**config['known_input_sha256'],'name-that-llm/train.csv':config['train_sha256']}
    for suffix,digest in wanted.items():
        hits=[x for x in files if x['path'].endswith('/'+suffix)]
        assert len(hits)==1 and hits[0]['sha256']==digest,('Pinned CPU input differs',suffix)
    for pattern in config['required_globs']:
        assert any(PurePosixPath(x['path']).match(pattern) for x in files),('Missing audited required input',pattern)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight-manifest',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args()
    config=json.loads((HERE/'preflight_config.json').read_text())
    preflight=json.loads(args.preflight_manifest.read_text())
    validate_preflight(preflight,config)
    assert sha(EXP/'panel/test.csv')==runner.PANEL_SHA256
    assert not args.destination.exists(),'New immutable destination required'
    payload=args.destination/'payload'
    build_runtime.build(payload)
    shutil.copyfile(HERE/'runner.py',payload/'runner.py')
    shutil.copyfile(args.preflight_manifest,payload/'preflight_manifest.json')
    (payload/'panel').mkdir()
    shutil.copyfile(EXP/'panel/test.csv',payload/'panel/test.csv')
    (payload/'ATTRIBUTION.md').write_text(
        '# EXP-029 private exploratory capture\n\n'
        'Algorithm: fixed local EXP-004 ATLAS20 source; original authors and licenses retained in project source records. '
        'Reference: https://www.kaggle.com/code/omerfarukmerey/0-663-0-660-v31-atlas-20\n\n'
        'Evaluation text: earthmaker / ntl-own-prompt-generations, fixed Kaggle V1, 729 previously selected rows. '
        'https://www.kaggle.com/datasets/earthmaker/ntl-own-prompt-generations/versions/1\n'
        'Text bytes unchanged; identifiers replaced by text-derived hashes and rows sorted by those hashes. '
        'No truth labels or prompt IDs are provided to model processes. '
        'This is an exploratory panel, not certified held-out full-stack validation.\n',encoding='utf-8')
    hashes={p.relative_to(payload).as_posix():sha(p) for p in sorted(payload.rglob('*')) if p.is_file()}
    plan={'schema_version':1,'experiment_id':'EXP-029','run_id':'exp029-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'),
          'scope':'exploratory_external','train_sha256':runner.TRAIN_SHA256,'panel_sha256':runner.PANEL_SHA256,'panel_rows':729,
          'work_budget_seconds':28200,'platform_wall_seconds':28800,'shutdown_grace_seconds':10,
          'expected_main_members':runner.MAIN_MEMBERS,'expected_comp_members':runner.COMP_MEMBERS,'expected_family_gates':runner.FAMILY_GATES,
          'allowed_omissions':['tokattn_q35'],'required_group_coverage':'At least one eligible comp group; otherwise retain results and report coverage unmet, without changing groups or shrinking inventory.',
          'attachment_bindings':config['attachment_bindings'],'preflight_manifest_sha256':sha(args.preflight_manifest),'input_hashes':hashes,
          'required_versions':{k:v for k,v in preflight['environment']['versions'].items() if v is not None},
          'container':json.loads((HERE/'preflight-kernel/kernel-metadata.json').read_text())['docker_image'],
          'new_gpu_runs':1,'automatic_retry':False,'formal_submission':False,'public_resource':False}
    save(payload/'plan.json',plan)
    runner.validate_plan(plan);runner.verify_payload(payload,plan)
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(payload.rglob('*')):
            if p.is_file():archive.write(p,p.relative_to(payload).as_posix())
    blob=buffer.getvalue();digest=hashlib.sha256(blob).hexdigest()
    encoded=base64.b64encode(blob).decode('ascii')
    allowed=set(hashes)|{'plan.json'}
    source='''# EXP-029 one bounded private GPU run; no competition submission.\nimport base64,hashlib,io,pathlib,subprocess,sys,zipfile\np=pathlib.Path('/kaggle/working/payload');p.mkdir(exist_ok=False)\nblob=base64.b64decode('''+repr(encoded)+''')\nassert hashlib.sha256(blob).hexdigest()=='''+repr(digest)+'''\nwith zipfile.ZipFile(io.BytesIO(blob)) as z:\n    assert set(z.namelist())=='''+repr(allowed)+'''\n    assert all(not pathlib.PurePosixPath(n).is_absolute() and '..' not in pathlib.PurePosixPath(n).parts for n in z.namelist())\n    z.extractall(p)\nsubprocess.run([sys.executable,'-B','-u',str(p/'runner.py')],check=True)\n'''
    compile(source,'capture.py','exec')
    kernel=args.destination/'kernel';kernel.mkdir()
    (kernel/'capture.py').write_text(source,encoding='utf-8')
    metadata=json.loads((HERE/'preflight-kernel/kernel-metadata.json').read_text())
    metadata.update(id='zhjianduan/name-that-llm-cache-capture-029',title='Name That LLM Cache Capture 029',
                    code_file='capture.py',enable_gpu=True,machine_shape='NvidiaTeslaT4')
    save(kernel/'kernel-metadata.json',metadata)
    save(args.destination/'launch-package.json',{'status':'PREPARED_NOT_LAUNCHED','run_id':plan['run_id'],
         'payload_zip_sha256':digest,'payload_zip_bytes':len(blob),'bootstrap_sha256':sha(kernel/'capture.py'),
         'metadata_sha256':sha(kernel/'kernel-metadata.json'),'plan_sha256':sha(payload/'plan.json'),
         'preflight_manifest_sha256':plan['preflight_manifest_sha256'],'panel_truth_uploaded':False,
         'prepare_capture_sha256':sha(Path(__file__))})
    print(json.dumps({'status':'PREPARED_NOT_LAUNCHED','run_id':plan['run_id'],'kernel':str(kernel),'bytes':len(source),'files':len(allowed)}))


if __name__=='__main__':main()
