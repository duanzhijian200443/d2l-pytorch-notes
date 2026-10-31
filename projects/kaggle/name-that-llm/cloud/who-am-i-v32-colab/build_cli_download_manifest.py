"""Resolve pinned Kaggle inputs locally; never export long-lived credentials.

Temporary signed download URLs are written only to the explicitly supplied
temporary path. Logs contain references, counts and sizes, never signed URLs.
"""
import concurrent.futures
import json
import sys
import time
import threading
from pathlib import Path
from urllib.parse import urlparse
import importlib
k = importlib.import_module('kaggle.api.kaggle_api_extended')

sys.stdout.reconfigure(encoding='utf-8')
root = Path(__file__).resolve().parent
lock = json.loads((root/'input-lock.json').read_text())
api = k.KaggleApi()
api.authenticate()
destination = Path(sys.argv[1])
checkpoint = destination.with_suffix('.checkpoint.json')
cache = json.loads(checkpoint.read_text()) if checkpoint.is_file() else {}
cache_lock = threading.Lock()
request_lock = threading.Lock()
next_request = 0
items = []

def remember(key, value):
    with cache_lock:
        cache[key] = value
        pending = checkpoint.with_suffix('.pending')
        pending.write_text(json.dumps(cache),encoding='utf-8')
        pending.replace(checkpoint)
    return value

def paced(call):
    global next_request
    for attempt in range(4):
        with request_lock:
            time.sleep(max(0, next_request-time.monotonic()))
            next_request = time.monotonic()+2.1
        try:
            return call()
        except Exception as exc:
            response = getattr(exc,'response',None)
            if response is None or response.status_code!=429 or attempt==3:
                raise
            raw = response.headers.get('Retry-After','')
            wait = max(int(raw) if raw.isdigit() else 60,60*2**attempt)
            print(f'Kaggle rate limit: waiting {wait} seconds before retry',flush=True)
            with request_lock:
                next_request = max(next_request,time.monotonic()+wait)
            time.sleep(wait)

def response_file(response, name, kind, expected=None):
    try:
        response.raise_for_status()
        assert urlparse(response.url).hostname == 'storage.googleapis.com', 'Unexpected download host'
        size = int(response.headers.get('Content-Length', 0))
        if expected is not None:
            assert size == expected, f'Size mismatch for {name}'
        return {'name':name, 'url':response.url, 'bytes':size, 'format':kind}
    finally:
        response.close()

def dataset_file(job):
    ref, version, name, size = job
    key = f'dataset:{ref}/{version}/{name}'
    if key in cache:
        return cache[key]
    owner, slug = ref.split('/')
    q = k.ApiDownloadDatasetRequest()
    q.owner_slug, q.dataset_slug = owner, slug
    q.dataset_version_number, q.file_name, q.raw = version, name, True
    with api.build_kaggle_client() as client:
        return remember(key,response_file(paced(lambda:client.datasets.dataset_api_client.download_dataset(q)), name, 'raw', size))

for entry in lock['datasets']:
    ref, version = entry['ref'], entry['pinned_version']
    if ref == 'zhjianduan/llama31-8b-base-verified':
        items.append({'kind':'dataset', 'ref':ref, 'version':version,
                      'existing_drive_path':'models/llama-3.1-8b-base-1f47e50', 'files':[]})
        continue
    archive_key=f'dataset-archive:{ref}/{version}'
    if archive_key not in cache:
        owner,slug=ref.split('/')
        q=k.ApiDownloadDatasetRequest()
        q.owner_slug,q.dataset_slug,q.dataset_version_number=owner,slug,version
        try:
            with api.build_kaggle_client() as client:
                archive=response_file(paced(lambda:client.datasets.dataset_api_client.download_dataset(q)),'dataset.zip','zip')
            remember(archive_key,archive)
        except Exception as exc:
            response=getattr(exc,'response',None)
            if response is None or response.status_code!=404:
                raise
            remember(archive_key,None)
    if cache[archive_key]:
        items.append({'kind':'dataset','ref':ref,'version':version,'files':[cache[archive_key]]})
        print(f'Resolved dataset archive {ref}/{version}',flush=True)
        continue
    page = None
    jobs = []
    while True:
        listing = api.dataset_list_files(f'{ref}/{version}', page_token=page, page_size=200)
        jobs.extend((ref,version,f.name,int(f.total_bytes)) for f in listing.files)
        page = listing.next_page_token
        if not page:
            break
    assert jobs, f'Empty dataset {ref}'
    print(f'Resolving dataset {ref}/{version}: {len(jobs)} files', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        files = list(pool.map(dataset_file, jobs))
    items.append({'kind':'dataset','ref':ref,'version':version,'files':files})

for entry in lock['kernels']:
    ref, version = entry['ref'], entry['version']
    owner, slug = ref.split('/')
    files, page = [], None
    while True:
        q = k.ApiListKernelSessionOutputRequest()
        q.user_name, q.kernel_slug, q.version_label = owner, slug, f'v{version}'
        api._set_paging(q, 200, page)
        with api.build_kaggle_client() as client:
            listing = client.kernels.kernels_api_client.list_kernel_session_output(q)
        for f in listing.files or []:
            assert urlparse(f.url).hostname in ('storage.googleapis.com','www.kaggleusercontent.com'), 'Unexpected notebook output host'
            files.append({'name':f.file_name,'url':f.url,'bytes':None,'format':'raw'})
        page = listing.next_page_token
        if not page:
            break
    assert files, f'Empty notebook {ref}'
    items.append({'kind':'notebook','ref':ref,'version':version,'files':files})
    print(f'Resolved notebook {ref}/{version}: {len(files)} files', flush=True)

for handle in lock['models']:
    owner, slug, framework, instance, version = handle.split('/')
    q = k.ApiDownloadModelInstanceVersionRequest()
    q.owner_slug, q.model_slug, q.instance_slug = owner, slug, instance
    q.framework = api.lookup_enum(k.ModelFramework, k.ModelFramework.MODEL_FRAMEWORK_API, framework)
    q.version_number = int(version)
    with api.build_kaggle_client() as client:
        f = response_file(client.models.model_api_client.download_model_instance_version(q), 'model.tar.gz', 'tar')
    items.append({'kind':'model','ref':handle.lower(),'version':int(version),'files':[f]})
    print(f'Resolved model {handle}: {f["bytes"]} compressed bytes',flush=True)

q = k.ApiDownloadDataFilesRequest()
q.competition_name = 'name-that-llm'
with api.build_kaggle_client() as client:
    f = response_file(client.competitions.competition_api_client.download_data_files(q), 'competition.zip', 'zip')
items.append({'kind':'competition','ref':'name-that-llm','version':None,'files':[f]})
assert len(items) == 31
manifest = {'upstream_version':32, 'inputs':items, 'pinned_csv_sha256':lock['pinned_csv_sha256']}
destination.write_text(json.dumps(manifest), encoding='utf-8')
print(json.dumps({'inputs':len(items),'files':sum(len(i['files']) for i in items),
                  'known_transfer_gib':round(sum(f['bytes'] or 0 for i in items for f in i['files'])/1024**3,2)}),flush=True)
