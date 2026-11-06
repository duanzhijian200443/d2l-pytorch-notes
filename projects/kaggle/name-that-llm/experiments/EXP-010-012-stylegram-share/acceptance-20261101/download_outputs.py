"""Read fixed Kaggle V2 outputs concurrently; no runtime or training mutations."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time
import requests
from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest

HERE = Path(__file__).resolve().parent
DEST = HERE.parent / 'download-v2'
REF = 'zhjianduan/name-that-llm-stylegram-share-cv/2'
local = threading.local()

def get_file(item):
    dest = (DEST / item.file_name).resolve()
    assert dest.is_relative_to(DEST.resolve()), 'Invalid output path'
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not hasattr(local, 'session'):
        local.session = requests.Session()
    for attempt in range(2):
        try:
            with local.session.get(item.url, timeout=(15, 40)) as response:
                response.raise_for_status()
                data = response.content
                size = response.headers.get('Content-Length')
                if size is not None and not response.headers.get('Content-Encoding'):
                    assert len(data) == int(size), 'Incomplete body'
            dest.write_bytes(data)
            return {'path': item.file_name, 'bytes': len(data),
                    'sha256': hashlib.sha256(data).hexdigest()}
        except Exception as exc:
            # Errors may contain signed URLs: record only their type and filename.
            if attempt:
                raise RuntimeError(f'{item.file_name}: {type(exc).__name__}') from None
            time.sleep(1)

def main():
    api = KaggleApi()
    api.authenticate()
    status = api.kernels_status(REF)
    status_text = str(status)
    assert 'COMPLETE' in status_text.upper(), status_text
    entries = []
    log = ''
    token = None
    with api.build_kaggle_client() as client:
        while True:
            req = ApiListKernelSessionOutputRequest()
            req.user_name = 'zhjianduan'
            req.kernel_slug = 'name-that-llm-stylegram-share-cv'
            req.version_label = 'v2'
            api._set_paging(req, 100, token)
            response = client.kernels.kernels_api_client.list_kernel_session_output(req)
            entries.extend(response.files or [])
            if response.log:
                log = response.log
            token = response.next_page_token
            if not token:
                break
    assert len(entries) == len({x.file_name for x in entries})
    print(f'Listed {len(entries)} fixed V2 files', flush=True)
    manifest = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(get_file, item) for item in entries]
        for f in as_completed(futures):
            manifest.append(f.result())
            if len(manifest) % 25 == 0:
                print(f'Downloaded {len(manifest)}/{len(entries)}', flush=True)
    if log:
        (DEST / 'name-that-llm-stylegram-share-cv.log').write_text(log, encoding='utf-8')
    result = {'status': 'DOWNLOADED', 'notebook_ref': REF,
              'observed_at': datetime.now(timezone.utc).isoformat(),
              'kaggle_status': status_text, 'file_count': len(manifest),
              'files': sorted(manifest, key=lambda x: x['path'])}
    (HERE / 'download-manifest.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f'DOWNLOADED {len(manifest)}; no signed URLs saved', flush=True)

if __name__ == '__main__':
    main()
