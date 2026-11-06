"""Preserve an actual V30 aggregate cache without promoting it to member evidence."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
from full_stack_cache import CacheWriter, read_cache, LABELS

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    source=HERE/'legacy-v30'
    meta=json.loads((source/'comp_meta.json').read_text(encoding='utf-8'))
    test=pd.read_csv(ROOT/'test.csv',dtype={'id':str},keep_default_na=False)
    assert meta['ids']==test.id.tolist() and meta['labels']==list(LABELS)
    matrix=np.load(source/'comp_logit.npy',allow_pickle=False)
    assert matrix.shape==(len(test),18) and len(test)==3 and np.isfinite(matrix).all()
    provenance={'ref':'zhjianduan/name-that-llm-who-am-i-v30-repro','version':7,
        'readback':'experiments/EXP-028-full-stack-cache/logs/kaggle-v30-v7-output.json',
        'source_files':{p.relative_to(ROOT).as_posix():sha(p) for p in [source/'comp_logit.npy',source/'comp_meta.json',ROOT/'test.csv']},
        'cache_kind':'legacy_post_gate_aggregate_only','not_current_atlas20_member_cache':True,
        'labels_available':False,'training_isolation_certified':False}
    writer=CacheWriter(HERE/'caches','legacy-v30-summary',test.id.tolist(),test.text.tolist(),LABELS,
                       metadata={'scope':'historical_public_sample_aggregate','provenance':provenance})
    writer.array('comp_logit',matrix)
    writer.note('provenance',provenance)
    receipt=writer.finish(['comp_logit'],['provenance'],metadata={'full_model_coverage_verified':False,'replayable_full_baseline':False})
    saved=read_cache(HERE/'caches','legacy-v30-summary',test.id.tolist(),test.text.tolist(),LABELS,
                     required_arrays=['comp_logit'],manifest_sha256=receipt['manifest_sha256'])
    assert saved.arrays['comp_logit'].dtype==matrix.dtype and np.array_equal(saved.arrays['comp_logit'],matrix)
    output={'status':'PASS_LEGACY_AGGREGATE_ROUNDTRIP','observed_at_utc':datetime.now(timezone.utc).isoformat(),
        'rows':3,'classes':18,'dtype':str(matrix.dtype),'max_abs_error':0.0,'provenance':provenance,
        'receipt':receipt,'limitation':'Three public samples, post-gate aggregate from V30; no member LP/gates/main/ATLAS/Sinkhorn/typography. Cannot backfill missing scores or evaluate accuracy.',
        'new_model_execution':False,'gpu_used':False,'script_sha256':sha(Path(__file__))}
    target=HERE/'metrics/legacy-import.json'
    with target.open('x',encoding='utf-8') as stream:json.dump(output,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps({k:output[k] for k in ('status','rows','classes','dtype','max_abs_error')},ensure_ascii=False))

if __name__=='__main__':main()
