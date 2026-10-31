"""Download the 31 frozen inputs directly to Drive on a CPU runtime."""
import concurrent.futures
import hashlib
import json
import os
import shutil
import tarfile
import time
import traceback
import zipfile
from pathlib import Path
import requests
from urllib.parse import urlparse

ROOT = Path('/content/v32_payload')
BASE = Path('/content/drive/MyDrive/Kaggle/name-that-llm')
DRIVE = BASE/'v32-colab'
CACHE = DRIVE/'cli-cache'
STATUS = DRIVE/'cli_preparation_result.json'
LOCAL = Path('/content/v32_download')
LOCAL.mkdir(exist_ok=True)
manifest = json.loads(Path('/content/v32-cli-downloads.json').read_text())
inventory = []
started = time.time()
deadline = started + 3*3600

def state(status, **extra):
    obj = {'status':status,'completed_inputs':len(inventory),'inputs':31,
           'elapsed_seconds':round(time.time()-started,1), 'updated_at':time.time(),**extra}
    temp = STATUS.with_suffix('.tmp')
    temp.write_text(json.dumps(obj,indent=2))
    temp.replace(STATUS)
    print(json.dumps(obj), flush=True)

def safe_path(root, name):
    target = (root/name).resolve()
    assert target.is_relative_to(root.resolve()), 'Unsafe input file path'
    return target

def download_ranges(f, local):
    """Kaggle notebook CDN streams can terminate early; validate bounded ranges."""
    with requests.get(f['url'],headers={'Range':'bytes=0-0'},stream=True,timeout=(30,60)) as response:
        assert response.status_code==206, 'Notebook output must support byte ranges'
        total=int(response.headers['Content-Range'].split('/')[-1])
    identity=hashlib.sha256(f['url'].encode()).hexdigest()
    marker=local.with_suffix('.resume.json')
    previous=json.loads(marker.read_text()) if marker.is_file() else {}
    offset=previous.get('completed_bytes',0)
    if previous.get('identity')!=identity or previous.get('total')!=total or not local.exists() or local.stat().st_size<offset:
        offset=0
    with local.open('r+b' if local.exists() else 'w+b') as stream:
        stream.truncate(offset)
        stream.seek(offset)
        while offset<total:
            assert time.time()<deadline, 'CPU preparation deadline reached'
            end=min(total-1,offset+128*1024*1024-1)
            for attempt in range(3):
                try:
                    with requests.get(f['url'],headers={'Range':f'bytes={offset}-{end}'},stream=True,timeout=(30,120)) as response:
                        assert response.status_code==206
                        assert response.headers['Content-Range']==f'bytes {offset}-{end}/{total}'
                        count=0
                        for chunk in response.iter_content(8*1024*1024):
                            stream.write(chunk)
                            count+=len(chunk)
                        assert count==end-offset+1, 'Incomplete range'
                    break
                except Exception as exc:
                    stream.truncate(offset)
                    stream.seek(offset)
                    if attempt==2:
                        raise RuntimeError(f'Range download failed for {f["name"]} at {offset}: {type(exc).__name__}') from None
                    time.sleep(3)
            stream.flush()
            offset=end+1
            marker.write_text(json.dumps({'identity':identity,'total':total,'completed_bytes':offset}))
            if offset%(1024**3)==0 or offset==total:
                print('Downloaded range',f['name'],offset,'/',total,flush=True)
    assert local.stat().st_size==total
    marker.unlink(missing_ok=True)
    return total

def download(f, cache):
    target = safe_path(cache, f['name'])
    if f['format']=='raw' and target.is_file() and f['bytes'] is not None and target.stat().st_size==f['bytes']:
        return
    local = LOCAL/(hashlib.sha256((str(cache)+f['name']).encode()).hexdigest()+'.part')
    total = None
    ranged = urlparse(f['url']).hostname=='www.kaggleusercontent.com'
    if ranged:
        total=download_ranges(f,local)
    for attempt in range(0 if ranged else 3):
        if time.time()>deadline:
            raise TimeoutError('CPU input preparation exceeded three hours')
        try:
            # Avoid printing exception strings: a requests exception can contain its signed URL.
            with requests.get(f['url'],stream=True,timeout=(30,120)) as response:
                if response.status_code != 200:
                    raise RuntimeError(f'Download HTTP {response.status_code}')
                total = int(response.headers.get('Content-Length',0)) or f['bytes']
                if f['bytes'] is not None:
                    assert total == f['bytes'], 'Remote object size changed'
                with local.open('wb') as stream:
                    for chunk in response.iter_content(8*1024*1024):
                        if time.time()>deadline:
                            raise TimeoutError('CPU preparation deadline')
                        stream.write(chunk)
            if total:
                assert local.stat().st_size == total, 'Truncated download'
            break
        except Exception as exc:
            print('Download retry', f['name'], attempt+1, type(exc).__name__, flush=True)
            if attempt==2:
                raise RuntimeError(f'Download failed for {f["name"]}: {type(exc).__name__}') from None
            time.sleep(3)
    if f['format']=='raw':
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(local,target)
        assert target.stat().st_size == local.stat().st_size
    elif f['format']=='tar':
        with tarfile.open(local,'r:gz') as archive:
            archive.extractall(cache,filter='data')
    elif f['format']=='zip':
        with zipfile.ZipFile(local) as archive:
            for name in archive.namelist():
                safe_path(cache,name)
            archive.extractall(cache)
    else:
        raise ValueError('Unknown archive format')
    local.unlink()

try:
    state('RUNNING')
    for item in manifest['inputs']:
        kind,ref,version = item['kind'],item['ref'],item['version']
        if item.get('existing_drive_path'):
            cache=BASE/item['existing_drive_path']
            assert (cache/'transfer_manifest.json').is_file(), 'Verified Llama copy is missing'
        else:
            cache=CACHE/kind/ref/(str(version) if kind!='model' and version is not None else '')
            cache.mkdir(parents=True,exist_ok=True)
            marker=cache/'.v32-download-complete.json'
            identity={'kind':kind,'ref':ref,'version':version,
                      'files':[{k:f[k] for k in ('name','bytes','format')} for f in item['files']]}
            if marker.is_file():
                assert json.loads(marker.read_text())==identity, 'Cache identity mismatch'
            else:
                state('RUNNING',current_input=f'{kind}:{ref}',current_files=len(item['files']))
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    pending=[pool.submit(download,f,cache) for f in item['files']]
                    done=0
                    for future in concurrent.futures.as_completed(pending):
                        future.result()
                        done+=1
                        if done%10==0 or done==len(pending):
                            state('RUNNING',current_input=f'{kind}:{ref}',files_done=done,files_total=len(pending))
                marker.write_text(json.dumps(identity,sort_keys=True))
        inventory.append({'kind':kind,'ref':ref,'version':version,'path':str(cache)})
        state('RUNNING',last_completed=f'{kind}:{ref}')
    checked=[]
    for relative,expected in manifest['pinned_csv_sha256'].items():
        dataset=relative.split('/')[0]
        matches=[]
        for item in inventory:
            if item['kind']=='dataset' and item['ref'].split('/')[-1]==dataset:
                matches.extend(Path(item['path']).rglob(Path(relative).name))
        assert len(matches)==1, f'Historical CSV ambiguous or missing: {relative}'
        actual=hashlib.sha256(matches[0].read_bytes()).hexdigest()
        assert actual==expected, f'Historical input mismatch: {relative}'
        checked.append({'input':relative,'path':str(matches[0]),'sha256':actual})
    (DRIVE/'cli_input_inventory.json').write_text(json.dumps(inventory,indent=2))
    (DRIVE/'cli_pinned_inputs_verified.json').write_text(json.dumps({'status':'PASS','files':checked},indent=2))
    state('READY',historical_hashes='PASS')
    os.sync()
except BaseException as exc:
    state('ERROR',error_type=type(exc).__name__,message=str(exc))
    raise
