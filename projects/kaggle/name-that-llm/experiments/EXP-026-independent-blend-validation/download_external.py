"""Fetch pinned SRC-062 raw files using the already configured official client.

Reuses the prior pinned-version downloader. No credentials/URLs are printed.
No data selection, fitting or scoring is performed here.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
from datetime import datetime,timezone

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
source=ROOT/'experiments/EXP-008-public-pool-audit/download_inputs.py'
spec=importlib.util.spec_from_file_location('existing_pinned_downloader',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
snapshot=ROOT/'research/datasets/SRC-062-enaimaan-ntl-openai-generated-responses-v3/snapshots/20261101T073434Z.json'
raw=json.loads(snapshot.read_text(encoding='utf-8'))
listing=json.loads(raw['files']['content'][0]['text'])
entry={'ref':'enaimaan/ntl-openai-generated-responses','version':3,'file_listing':listing}
(HERE/'input-metadata.json').write_text(json.dumps({'source_id':'SRC-062','source_snapshot':snapshot.relative_to(ROOT).as_posix(),'source_snapshot_sha256':hashlib.sha256(snapshot.read_bytes()).hexdigest(),'live_listing_rechecked':'2026-11-01 via Kaggle MCP list_dataset_files explicit v3; 10 matching names/sizes','entry':entry},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
result=module.download(entry)
directory=ROOT/'data/public-pools'/entry['ref']/'v3'
expected={}
for line in (directory/'SHA256SUMS').read_text().splitlines():
    digest,name=line.split(maxsplit=1);name=name.lstrip('* ')
    path=(directory/name).resolve()
    assert path.is_relative_to(directory.resolve()) and path.is_file()
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
    expected[name]=digest
assert set(expected)=={x['name'] for x in result['files']} - {'SHA256SUMS'}
receipt={'observed_at':datetime.now(timezone.utc).isoformat(),'dataset':result,'publisher_sha256sums':'PASS for all 9 covered files','downloader_source':source.relative_to(ROOT).as_posix(),'downloader_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'data_gate':'PENDING','scoring_performed':False}
(HERE/'download-manifest.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Pinned V3 downloaded and checksums verified; no validation score computed.')
