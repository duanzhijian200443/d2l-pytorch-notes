"""Read-only fixed-version downloads. No retry loop; never print signed URLs."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'data/open-pool-v13-audit'
ITEMS = [
    ('crtproduction/name-that-llm-open-pool/13', 'ntl_open_pool.parquet', 'v13', 96388050, None),
    ('crtproduction/name-that-llm-open-pool/13', 'prompts.parquet', 'v13', 1592857, None),
    ('crtproduction/name-that-llm-open-pool/13', 'gemini37_unverified.parquet', 'v13', 5366960, None),
    ('prvsiyan/prvsiyans-extended-name-that-llm-dataset/7', 'passages_clean.csv', 'baseline-prvs-v7', 7519791,
     '502653b8cc053bc1ef08920d0ddda892e8b656f14229e38e0755d8a2f079eaf9'),
]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    receipt_path = DEST / 'download-receipt.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf8')) if receipt_path.exists() else {'files': []}
    known = {x['path']: x for x in receipt['files']}
    api = KaggleApi()
    api.authenticate()
    for ref, name, subdir, size, expected_hash in ITEMS:
        folder = DEST / subdir
        path = folder / name
        key = path.relative_to(ROOT).as_posix()
        folder.mkdir(parents=True, exist_ok=True)
        if key in known:
            if not path.exists() or path.stat().st_size != known[key]['bytes'] or sha(path) != known[key]['sha256']:
                raise RuntimeError('Existing receipt mismatch: ' + key)
            print('VERIFIED_REUSE', key, flush=True)
            continue
        if path.exists():
            raise RuntimeError('Unreceipted existing file; inspect before reuse: ' + key)
        print('DOWNLOAD', ref, name, flush=True)
        try:
            api.dataset_download_file(ref, name, path=str(folder), quiet=True, unzip=True)
        except Exception as exc:
            # Upstream HTTP exception messages can contain temporary credentials.
            raise RuntimeError('Download failed; no retry. ' + type(exc).__name__) from None
        if path.stat().st_size != size:
            raise RuntimeError('Fixed metadata size mismatch: ' + key)
        digest = sha(path)
        if expected_hash and digest != expected_hash:
            raise RuntimeError('Baseline runtime hash mismatch: ' + key)
        row = {'path': key, 'bytes': size, 'sha256': digest, 'ref': ref, 'file_name': name,
               'source_url': 'https://www.kaggle.com/datasets/' + '/'.join(ref.split('/')[:2]) + '/versions/' + ref.split('/')[-1],
               'observed_at': datetime.now(timezone.utc).isoformat(),
               'baseline_runtime_hash_verified': expected_hash is not None}
        receipt['files'].append(row)
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf8')
        print('VERIFIED', key, size, digest, flush=True)


if __name__ == '__main__':
    os.environ.setdefault('HTTP_PROXY', 'http://127.0.0.1:7890')
    os.environ.setdefault('HTTPS_PROXY', 'http://127.0.0.1:7890')
    main()
