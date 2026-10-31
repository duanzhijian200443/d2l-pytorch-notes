import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
specs = [
    ('abdullah1afify', 'ntl-generations', 2, 21008519, 'Abdullah Afify', {'README.md': 2011, 'make_tasks.py': 6160, 'ntl_generations.csv': 870886, 'prompts_used.csv': 32000}),
    ('lucasboesen', 'ntl-free-proxy-generations', 7, 21116786, 'Lucas Boesen', {'README.md': 4718, 'gen_passages.csv': 3835894}),
]
lock = json.loads((ROOT / 'input-lock.json').read_text(encoding='utf-8'))
hashes = {}
for owner, slug, version, version_id, author, expected in specs:
    directory = ROOT / 'snapshots' / slug
    files = []
    for name, size in expected.items():
        data = (directory / name).read_bytes()
        assert len(data) == size, name
        digest = hashlib.sha256(data).hexdigest()
        files.append({'name': name, 'bytes': size, 'sha256': digest})
        if name.endswith('.csv'):
            hashes[slug + '/' + name] = digest
    source_url = f'https://www.kaggle.com/datasets/{owner}/{slug}/versions/{version}'
    provenance = {'source_owner': owner, 'source_dataset': slug, 'source_version': version, 'source_version_id': version_id, 'source_url': source_url, 'author': author, 'license': 'CC BY 4.0', 'files': files}
    (directory / 'snapshot_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    metadata = {
        'id': f'zhjianduan/{slug}',
        'title': f'V30 Snapshot {slug} V{version}'[:50],
        'licenses': [{'name': 'CC-BY-4.0'}],
        'description': f'Private, unmodified historical snapshot for reproducing Who am I V30. Original author: {author} ({owner}). Source: {source_url}; dataset version ID {version_id}. The source files are unchanged and remain licensed under CC BY 4.0. See the original README.md and snapshot_provenance.json for attribution and file hashes. This snapshot is pinned because Notebook uploads currently resolve version-qualified dataset references to the latest version.',
    }
    (directory / 'dataset-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    for item in lock['datasets']:
        if item['ref'] == owner + '/' + slug:
            item['snapshot_ref'] = 'zhjianduan/' + slug
            item['snapshot_version'] = 1
            item['source_version_id'] = version_id
lock['pinned_csv_sha256'] = hashes
(ROOT / 'input-lock.json').write_text(json.dumps(lock, indent=2), encoding='utf-8')
print(json.dumps({'snapshot_files': 6, 'csv_hashes': hashes}))
