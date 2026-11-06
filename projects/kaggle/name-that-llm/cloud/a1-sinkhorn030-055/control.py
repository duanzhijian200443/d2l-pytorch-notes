"""EXP055: reuse the EXP046 save/readback mechanism with the frozen EXP049 package."""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import re
import types
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FROZEN = ROOT / 'cloud/a1-sinkhorn030-049'
EXP = ROOT / 'experiments/EXP-055-a1-sinkhorn030-formal'
REF = 'zhjianduan/name-that-llm-a1-sinkhorn030-049'
SOURCE = '35ddcc24900d4bace4b3c4cc167475ad2c119af83a9eb687ef717f7bea1200b5'
METADATA = '8c0a1ad41ecda521e57a0072bb68d93bbfa2ce65014a4ee42ffb0d09d679ba6f'
os.environ['HTTP_PROXY'] = os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

base = load(ROOT/'cloud/a1-public-stylegram-046/control.py', 'exp055_reused_controller')
base.REF, base.HERE = REF, HERE
# The package validator only reads its HERE/kernel. Bind that reader to the
# original frozen EXP049 directory while state/locks/receipts stay in EXP055.
read_frozen = types.FunctionType(base.local_package.__code__, dict(base.local_package.__globals__, HERE=FROZEN))
def local_package():
    folder, meta, binding = read_frozen()
    base.require(binding['source_bytes_sha256']==SOURCE and binding['metadata_sha256']==METADATA, 'Frozen EXP049 hashes changed')
    binding['exp055_controller_sha256'] = base.sha(__file__)
    return folder, meta, binding
base.local_package = local_package

def discover(api):
    matches, count = [], 0
    for page in range(1, 11):
        items = api.kernels_list(mine=True, user='zhjianduan', page=page, page_size=100)
        base.require(items is not None and all(i is not None for i in items), 'Incomplete owned inventory')
        count += len(items)
        matches.extend(i.to_dict() for i in items if i.ref in (REF, '/code/'+REF))
        if len(items) < 100: break
    else: raise RuntimeError('Incomplete owned inventory')
    subs = api.competition_submissions('name-that-llm',page_size=100)
    base.require(subs is not None and len(subs)<100, 'Submission list may be truncated')
    keys = ('ref','description','status','scriptVersionId','publicScore','date','fileName')
    selected = [{k:v for k,v in item.to_dict().items() if k in keys} for item in subs]
    prior = [s for s in selected if any(x in str(s.get('description','')).lower() for x in
             ('exp055','exp-055','exp049','exp-049','a1-sinkhorn030-049'))]
    result = {'ref':REF,'observed_at':base.now(),'owned_notebooks':count,'matching_notebooks':matches,
              'competition_submission_count':len(subs),'possible_prior_submissions':prior,'submissions':selected,
              'local_package':local_package()[2]}
    base.save(HERE/'discovery.json',result,exclusive=True)
    base.update(experiment_id='EXP-055',status='DISCOVERY_COMPLETE',reuse_required=bool(matches or prior))
    print(json.dumps({k:result[k] for k in ('owned_notebooks','matching_notebooks','competition_submission_count','possible_prior_submissions')}))

def script_identity(api, version):
    from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest
    q=ApiListKernelSessionOutputRequest();q.user_name,q.kernel_slug=REF.split('/');q.version_label='v'+str(version);q.page_size=100
    with api.build_kaggle_client() as client:
        files=client.kernels.kernels_api_client.list_kernel_session_output(q).files
    bindings=[]
    for row in files:
        m=re.search(r'/kf/(\d+)/',row.url)
        if m:
            bindings.append({'file':row.file_name,'script_version_id':int(m.group(1))})
    primary=[r for r in bindings if r['file']=='submission.parquet']
    base.require(len(primary)==1 and len({r['script_version_id'] for r in bindings})==1,'Cannot bind fixed version outputs')
    value={'ref':REF,'version':version,'script_version_id':primary[0]['script_version_id'],'observed_at':base.now(),
           'bindings':bindings,'method':'fixed version output object kf ID, with signed URLs omitted'}
    base.save(HERE/'script-identity.json',value,exclusive=True)
    base.update(script_version_id=value['script_version_id'])
    return value

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['discover','save-only','run','status','collect','readback'])
    parser.add_argument('--version',type=int,default=2)
    args=parser.parse_args()
    with base.command_lock():
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi();api.authenticate()
        if args.command=='discover':
            discover(api);return
        if args.command in ('save-only','run'):
            base.require((HERE/'discovery.json').is_file() and not base.state().get('reuse_required'),'Requires reuse review')
            r=base.push_once(api,args.command=='run')
            print(json.dumps({k:v for k,v in r.items() if k!='saved_readback'}));return
        if args.command=='status':
            print(json.dumps(base.observe_status(api,args.version)));return
        if args.command=='readback':
            print(base.readback(api,args.version)['status']);return
        base.download(api,args.version)
        print(json.dumps(script_identity(api,args.version)))

if __name__=='__main__': main()
