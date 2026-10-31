"""Read-only retrieval of public notebook sources; never execute downloaded code."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from kaggle.api.kaggle_api_extended import KaggleApi, ApiGetKernelRequest

ROOT = Path(__file__).resolve().parent
ROWS = json.loads((ROOT / 'search-order.json').read_text(encoding='utf-8'))['kernels'][:20]


def fetch(pair):
    rank, row = pair
    dest = ROOT / f'{rank:02d}-{row["ref"].split("/")[0]}'
    dest.mkdir(exist_ok=True)
    api = KaggleApi(); api.authenticate()
    owner, slug = row['ref'].split('/')
    request = ApiGetKernelRequest(); request.user_name = owner; request.kernel_slug = slug
    with api.build_kaggle_client() as client:
        response = client.kernels.kernels_api_client.get_kernel(request)
    metadata = response.metadata.to_dict()
    assert not metadata.get('isPrivate', metadata.get('is_private', False)), 'Private notebook excluded'
    raw = response.blob.source
    notebook = json.loads(raw)
    (dest / 'source.ipynb').write_text(raw, encoding='utf-8')
    (dest / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    markdown, code = [], []
    for i, cell in enumerate(notebook['cells']):
        src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
        (markdown if cell['cell_type'] == 'markdown' else code).append(f'\n# CELL {i}\n' + src)
    (dest / 'description.md').write_text('\n'.join(markdown), encoding='utf-8')
    (dest / 'code.txt').write_text('\n'.join(code), encoding='utf-8')
    result = {'rank': rank, **row, 'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
              'source_sha256': hashlib.sha256(raw.encode()).hexdigest(), 'directory': dest.name,
              'metadata': metadata, 'markdown_characters': sum(map(len, markdown)),
              'code_characters': sum(map(len, code)), 'status': 'DOWNLOADED_NOT_EXECUTED'}
    (dest / 'receipt.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(fetch, pair): pair for pair in enumerate(ROWS, 1)}
        for job in as_completed(jobs):
            rank, row = jobs[job]
            try:
                result = job.result()
                results.append(result)
                print(json.dumps({'rank': rank, 'ref': row['ref'], 'status': result['status'],
                                  'markdown_chars': result['markdown_characters'], 'code_chars': result['code_characters']}, ensure_ascii=False), flush=True)
            except Exception as exc:
                results.append({'rank': rank, 'ref': row['ref'], 'status': 'ERROR', 'error_type': type(exc).__name__})
                print(json.dumps(results[-1]), flush=True)
    results.sort(key=lambda x: x['rank'])
    (ROOT / 'sources-index.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    if len(results) != 20 or any(x['status'] == 'ERROR' for x in results): sys.exit(1)
