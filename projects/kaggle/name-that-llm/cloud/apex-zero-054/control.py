"""EXP054 one-shot private save/run. Only the parent owns submission."""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT / 'experiments/EXP-054-apex-zero-formal'
SOURCE = next((ROOT / 'research/solutions').glob('SRC-067-*'))
REF = 'zhjianduan/ntl-apex-zero-054'
RAW_SHA = 'ca4c9e18ed32dcd8168b8118b192d39d657803d30dcd89a564b1c2d1d9c846dd'
IMAGE = 'gcr.io/kaggle-private-byod/python@sha256:9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435'
MACHINE = 'NvidiaTeslaT4'
os.environ['HTTP_PROXY'] = os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

def now():
    return datetime.now(timezone.utc).isoformat()

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode('utf8')).hexdigest()

def load(path):
    return json.loads(path.read_text(encoding='utf8'))

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf8')
    with path.open('xb') as target:
        target.write(content)
        target.flush()
        os.fsync(target.fileno())

def install():
    orig = (SOURCE / 'source.ipynb').read_bytes()
    require(sha(orig) == RAW_SHA, 'Upstream source drift')
    package = HERE / 'kernel';package.mkdir(exist_ok=True)
    actual = package / 'apex-zero.ipynb'
    if not actual.exists():
        with actual.open('xb') as f:
            f.write(orig)
    require(actual.read_bytes() == orig, 'Local source is not exact upstream bytes')
    source_meta = load(SOURCE / 'metadata.json')
    inventory = load(ROOT / 'experiments/EXP-053-apex-stylegram-volume/inputs/source-snapshot/attachment-inventory.json')
    pins = {x['ref']:x['version'] for x in inventory}
    data = [x + '/' + str(pins[x]) for x in source_meta['datasetDataSources']]
    kernels = [x + '/' + str(pins[x]) for x in source_meta['kernelDataSources']]
    models = source_meta['modelDataSources']
    require(len(data) == 15 and len(kernels) == 8 and len(models) == 8 and source_meta['dockerImage'] == IMAGE, 'Unexpected source attachment inventory')
    meta = {'id': REF, 'title': 'NTL Apex Zero 054', 'code_file': actual.name, 'language': 'python', 'kernel_type': 'notebook',
            'is_private': True, 'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False,'machine_shape': MACHINE,
            'docker_image': IMAGE, 'docker_image_pinning_type': 'original','competition_sources':['name-that-llm'],
            'dataset_sources':data,'kernel_sources':kernels,'model_sources':models}
    path = package / 'kernel-metadata.json'
    if not path.exists():save(path, meta)
    require(load(path) == meta, 'Existing package metadata differs')
    return actual, meta

def get(api, ref, version=None):
    with api.build_kaggle_client() as k:
        f = k.kernels.kernels_api_client.get_kernel
        q = f.__globals__['ApiGetKernelRequest']();q.user_name,q.kernel_slug = ref.split('/')
        if version is not None:q.version_label = 'v' + str(version)
        return json.loads(f(q).to_json())

def discover(api):
    owned = load(HERE / 'dedup-inventory.json')
    require(REF not in [r['ref'] for r in owned['owned']], 'Existing target ref, review before use')
    require(not any('apex-zero-054' in str(s).lower() for s in owned['competition_submissions']), 'Existing submission for target')
    notebooks = api.kernels_list(user='zhjianduan', mine=True, competition='name-that-llm', page_size=100)
    require(notebooks is not None and len(notebooks)<100, 'Incomplete competition notebook list')
    def history(item):
        ref = item.ref
        current = get(api, ref);n = current['metadata'].get('currentVersionNumber', 0)
        results = []
        if n == 0:
            src = current.get('blob',{}).get('source','')
            return [{'ref':ref,'version':None,'source_sha256':sha(src) if src else None,
                     'same_upstream_bytes':bool(src) and sha(src)==RAW_SHA,
                     'note':'No current version number returned by SDK; current source checked if present'}]
        for v in range(1,n+1):
            snap = current if v==n else get(api,ref,v)
            src = snap['blob']['source']; m=snap['metadata']
            results.append({'ref':ref,'version':v,'source_sha256':sha(src),'is_private':m.get('isPrivate'),
                            'same_upstream_bytes':sha(src)==RAW_SHA})
        return results
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        rows = [r for batch in executor.map(history,notebooks) for r in batch]
    matches = [r for r in rows if r['same_upstream_bytes']]
    save(HERE / 'owned-version-source-check.json',{'observed_at':now(),'contest_notebooks':len(notebooks),'fixed_versions_compared':len(rows),'versions':rows,'matches':matches})
    require(not matches,'Existing exact-source version found. Evaluate for reuse before creating.')
    print('DEDUP_PASS',len(notebooks),len(rows),flush=True)

def readback(api, version):
    src, meta = install();d=get(api, REF, version);m=d['metadata'];body=d['blob']['source']
    require(m['ref']==REF and m['currentVersionNumber']>=version,'Wrong saved ref/version')
    require(sha(body)==RAW_SHA and body.encode('utf8')==src.read_bytes(),'Saved source bytes differ from author V1')
    require(m['isPrivate'] is True and m['enableInternet'] is False and m['enableGpu'] is True and m['enableTpu'] is False,'Wrong saved access/device flags')
    require(m['machineShape']==MACHINE and m['dockerImage']==IMAGE,'Wrong image or shape')
    for ours,remote in [('dataset_sources','datasetDataSources'),('kernel_sources','kernelDataSources')]:
        expected = {r.rsplit('/',1)[0]:r for r in meta[ours]}
        actual = m[remote]
        require(len(actual)==len(expected) and {('/'.join(r.split('/')[:2])) for r in actual}==set(expected),'Saved attachment refs differ')
        require(all(len(r.split('/'))==2 or r==expected['/'.join(r.split('/')[:2])] for r in actual),'Saved attachment versions differ')
    require(sorted(m['modelDataSources'])==sorted(meta['model_sources']) and m['competitionDataSources']==['name-that-llm'],'Saved model/competition mismatch')
    obj={'observed_at':now(),'ref':REF,'version':version,'source_bytes_sha256':sha(body),'local_source_bytes_sha256':sha(src.read_bytes()),'source_exact':True,
         'private':True,'internet':False,'asset_count':31,'model_versions_bound':8,'requested_dataset_versions':meta['dataset_sources'],
         'requested_kernel_versions':meta['kernel_sources'],'metadata':m,'limit':'GetKernel readback may omit data/kernel version suffixes. Pinned SDK requests and V13 log counts do not certify every original hidden historical asset hash.'}
    save(HERE/f'readback-v{version}.json',obj)
    return obj

def dispatch(api, version):
    from kaggle.api.kaggle_api_extended import ApiSaveKernelRequest, KernelExecutionType
    source, meta=install()
    require(version in [1,2],'Unexpected dispatch version')
    require((HERE/'owned-version-source-check.json').exists() and not load(HERE/'owned-version-source-check.json')['matches'],'Discovery must finish')
    if version==2:
        require(load(HERE/'readback-v1.json')['source_exact'],'Verified V1 save required')
        current=get(api,REF);require(current['metadata']['currentVersionNumber']==1,'Target changed before execution')
    else:
        current=api.kernels_list(user='zhjianduan',mine=True,page_size=100)
        require(current is not None and len(current)<100 and not any(x.ref==REF for x in current),'Target appeared before initial save')
    request = ApiSaveKernelRequest();request.slug=REF;request.new_title=meta['title'];request.text=source.read_bytes().decode('utf8')
    request.language='python';request.kernel_type='notebook';request.is_private=True;request.enable_gpu=True;request.enable_tpu=False;request.enable_internet=False
    request.dataset_data_sources=meta['dataset_sources'];request.kernel_data_sources=meta['kernel_sources'];request.model_data_sources=meta['model_sources'];request.competition_data_sources=meta['competition_sources']
    request.docker_image=IMAGE;request.docker_image_pinning_type='original';request.machine_shape=MACHINE;request.session_timeout_seconds=32400
    if version==1:request.kernel_execution_type=KernelExecutionType.QUICK_SAVE
    save(HERE/f'v{version}-intent.json',{'at':now(),'ref':REF,'version':version,'quick_save':version==1,'source_sha256':sha(request.text),'metadata':meta,'one_shot':True})
    with api.build_kaggle_client() as k:
        response=k.kernels.kernels_api_client.save_kernel(request)
    save(HERE/f'v{version}-ack.json',{'at':now(),'response':response.to_dict()})
    require(not response.error and not any(getattr(response,n,None) for n in ('invalid_dataset_sources','invalid_kernel_sources','invalid_model_sources','invalid_competition_sources','invalid_tags')),'Server rejected save or input')
    require(response.version_number==version and response.ref in (REF,'/code/'+REF),'Unexpected saved ref/version')
    print('SAVE_ACK',response.ref,version,'QUICK_SAVE' if version==1 else 'RUN_QUEUED',flush=True)
    readback(api,version)
    print('READBACK_PASS',version,RAW_SHA,flush=True)

def status(api):
    result=api.kernels_status(REF+'/2');response=result.to_dict()
    path=HERE/'status'/f'{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")}.json'
    save(path,{'observed_at':now(),'ref':REF,'version':2,'response':response})
    print(response['status'],flush=True)
    return response['status']

def collect(api):
    from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest
    require(status(api)=='COMPLETE','Output not ready')
    q=ApiListKernelSessionOutputRequest();q.user_name,q.kernel_slug=REF.split('/');q.version_label='v2';q.page_size=100
    with api.build_kaggle_client() as k:
        files=k.kernels.kernels_api_client.list_kernel_session_output(q).files
    ids={}
    for row in files:
        match=re.search(r'/kf/(\d+)/',row.url)
        if match:ids[row.file_name]=int(match.group(1))
    require('submission.parquet' in ids and len(set(ids.values()))==1,'Unable to bind global scriptVersionId')
    sid=ids['submission.parquet'];save(HERE/'script-identity.json',{'ref':REF,'version':2,'script_version_id':sid,'file_ids':ids,'signed_urls_omitted':True,'at':now()})
    out=HERE/'outputs-v2';out.mkdir(exist_ok=False);paths,token=api.kernels_output(REF+'/2',str(out),file_pattern=r'^(submission\.parquet|infer\.py|comp_stack\.py|gauss\.py|sibling_style\.py)$',quiet=True)
    require(not token and (out/'submission.parquet').is_file(),'Incomplete outputs')
    print('OUTPUTS',sid,[Path(p).name for p in paths],flush=True)

if __name__=='__main__':
    from kaggle.api.kaggle_api_extended import KaggleApi
    a=argparse.ArgumentParser();a.add_argument('action',choices=['discover','save','run','status','collect']);args=a.parse_args()
    api=KaggleApi();api.authenticate()
    if args.action=='discover':discover(api)
    elif args.action=='save':dispatch(api,1)
    elif args.action=='run':dispatch(api,2)
    elif args.action=='status':status(api)
    else:collect(api)
