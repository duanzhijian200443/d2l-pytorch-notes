"""Download a pinned public model into Drive, verifying every file before upload."""

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


def verify_file(path, entry):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != entry['size']:
        return None
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(f"blob {entry['size']}\0".encode())
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha256.update(chunk)
            git_blob.update(chunk)
    actual = sha256.hexdigest() if entry.get('sha256') else git_blob.hexdigest()
    expected = entry.get('sha256') or entry['git_blob_sha1']
    if actual != expected:
        return None
    return {'file': entry['name'], 'bytes': entry['size'], 'sha256': sha256.hexdigest()}


def download_verified(model_dir, manifest):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    verified = []
    started = time.monotonic()
    total = sum(item['size'] for item in manifest['files'])
    done = 0
    print(f"Downloading {len(manifest['files'])} files; {total / 1e9:.3f} GB total.", flush=True)
    for entry in manifest['files']:
        name = entry['name']
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('Only explicitly listed root-level model files are allowed.')
        dest = model_dir / name
        existing = verify_file(dest, entry)
        if existing:
            verified.append(existing)
            done += entry['size']
            print(f"Verified existing: {name}", flush=True)
            continue
        if dest.exists():
            raise RuntimeError(f"Existing file failed verification; left untouched: {dest}")
        partial = model_dir / (name + '.partial')
        url = f"https://huggingface.co/{manifest['repo']}/resolve/{manifest['revision']}/{name}"
        for attempt in range(1, 5):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > entry['size']:
                raise RuntimeError(f"Oversized partial file left untouched: {partial}")
            if offset == entry['size']:
                break
            headers = {'Accept-Encoding': 'identity', 'User-Agent': 'name-that-llm-model-transfer/1.0'}
            if offset:
                headers['Range'] = f'bytes={offset}-'
            last_report = time.monotonic()
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=180) as response:
                    if offset and response.status == 200:
                        offset = 0  # Server ignored Range: replace only our own incomplete file.
                    elif offset:
                        content_range = response.headers.get('Content-Range', '')
                        if response.status != 206 or not content_range.startswith(f'bytes {offset}-'):
                            raise RuntimeError(f'Unexpected resume response for {name}')
                    if response.status not in (200, 206):
                        raise RuntimeError(f'Unexpected HTTP status {response.status} for {name}')
                    with partial.open('ab' if offset else 'wb') as stream:
                        position = offset
                        while True:
                            chunk = response.read(8 * 1024 * 1024)
                            if not chunk:
                                break
                            if position + len(chunk) > entry['size']:
                                raise RuntimeError(f'Download exceeds expected size: {name}')
                            stream.write(chunk)
                            position += len(chunk)
                            now = time.monotonic()
                            if now - last_report >= 20:
                                print(f"{name}: {position / entry['size']:.1%}; total {(done + position) / total:.1%}", flush=True)
                                last_report = now
                if partial.stat().st_size == entry['size']:
                    break
                raise OSError('Connection ended before the expected file size.')
            except (OSError, urllib.error.URLError) as exc:
                # Do not print exception URLs: redirects can contain expiring signed parameters.
                if attempt == 4:
                    raise RuntimeError(f'Download interrupted for {name}; rerun to resume.') from None
                print(f"Retry {attempt}/4 for {name} ({type(exc).__name__}).", flush=True)
                time.sleep(min(5 * attempt, 15))
        print(f"Checking integrity: {name}", flush=True)
        result = verify_file(partial, entry)
        if result is None:
            raise RuntimeError(f'Integrity check failed; incomplete file preserved: {partial}')
        partial.replace(dest)
        verified.append(result)
        done += entry['size']
        print(f"Verified: {name}; {done / total:.1%} complete", flush=True)
    report = {
        'status': 'VERIFIED',
        'source': manifest,
        'files': verified,
        'bytes': total,
        'elapsed_seconds': round(time.monotonic() - started, 1),
    }
    (model_dir / 'transfer_manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"All files verified. Persistent Drive directory: {model_dir}", flush=True)
    return report


def prepare_kaggle_dataset(model_dir, manifest):
    model_dir = Path(model_dir)
    report = json.loads((model_dir / 'transfer_manifest.json').read_text(encoding='utf-8'))
    assert report['status'] == 'VERIFIED'
    assert report['source'] == manifest
    expected = {item['name'] for item in manifest['files']}
    assert {item['file'] for item in report['files']} == expected
    for item in manifest['files']:
        assert (model_dir / item['name']).stat().st_size == item['size'], item['name']
    notice = 'Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.\n'
    (model_dir / 'NOTICE').write_text(notice, encoding='utf-8')
    metadata = {
        'title': 'Llama 3.1 8B Base Verified Weights',
        'id': 'zhjianduan/llama31-8b-base-verified',
        'licenses': [{'name': 'other'}],
        'description': (
            'Built with Llama. Private transfer of the unmodified Llama 3.1 8B base Transformers weights. '
            'License: Llama 3.1 Community License; see LICENSE, USE_POLICY.md and NOTICE. '
            'Source: https://huggingface.co/' + manifest['repo'] + '/tree/' + manifest['revision'] + '. '
            'Weight SHA-256 metadata matches meta-llama/Llama-3.1-8B at ' + manifest['official_revision'] + '. '
            'Every downloaded file was verified before upload; see transfer_manifest.json. '
            'These are original base weights, not Instruct, GGUF, or competition adapters.'
        ),
    }
    (model_dir / 'dataset-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    allowed = expected | {'transfer_manifest.json', 'NOTICE', 'dataset-metadata.json'}
    unexpected = [p.name for p in model_dir.iterdir() if p.name not in allowed]
    if unexpected:
        raise RuntimeError(f'Unexpected files must be reviewed before upload: {unexpected}')
    return metadata
