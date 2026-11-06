"""Fetch only the pinned, sub-MB earthmaker V1 source; no model execution."""
import importlib.util
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
downloader=ROOT/'experiments/EXP-008-public-pool-audit/download_inputs.py'
spec=importlib.util.spec_from_file_location('pinned_downloader',downloader)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
listing=ROOT/'research/datasets/SRC-063-earthmaker-own-prompts-v1/files-v1.json'
entry={'ref':'earthmaker/ntl-own-prompt-generations','version':1,'file_listing':json.loads(listing.read_text())}
result=module.download(entry)
receipt={'observed_at':datetime.now(timezone.utc).isoformat(),'dataset':result,
 'source_id':'SRC-063','pinned_listing_sha256':hashlib.sha256(listing.read_bytes()).hexdigest(),
 'downloader':downloader.relative_to(ROOT).as_posix(),'downloader_sha256':hashlib.sha256(downloader.read_bytes()).hexdigest(),
 'data_gate':'PENDING','training_or_scoring_performed':False}
target=HERE/'download-manifest.json'
with target.open('x',encoding='utf-8') as stream:
 json.dump(receipt,stream,ensure_ascii=False,indent=2)
 stream.write('\n')
