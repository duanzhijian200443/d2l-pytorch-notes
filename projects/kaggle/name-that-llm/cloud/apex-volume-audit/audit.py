"""Independent read-only audit of Apex Codex120/240 candidate notebooks."""
import argparse
import ast
import base64
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
import unicodedata
import zlib

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ZERO = ROOT/'cloud/apex-zero-054/kernel/apex-zero.ipynb'
EXP52 = ROOT/'experiments/EXP-052-stylegram-four-model-volume'
EXP53 = ROOT/'experiments/EXP-053-apex-stylegram-volume'
ZERO_SHA = 'ca4c9e18ed32dcd8168b8118b192d39d657803d30dcd89a564b1c2d1d9c846dd'
RAW_HASH = {120:'a4e86314a77f7ed699dbfb107b32a8263a1ff09fc7c3ec92b98d4028d487005d',
            240:'694d91e2536db2d77d81f8d8b32a802763a750a2e9643fdb0ba6a10b76755a01'}
ARMS = {120:'apex-codex120-056',240:'apex-codex240-057'}
MODELS = {'gpt-5.6-luna','gpt-5.6-sol','gpt-5.6-terra','gpt-6-astra'}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(b):
    return hashlib.sha256(b).hexdigest()


def norm(s):
    return ' '.join(unicodedata.normalize('NFKC', s).casefold().split())


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def readrows(path):
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def cells(n):
    return [''.join(c['source']) for c in n['cells']]


def audit_data():
    need(digest(ZERO.read_bytes())==ZERO_SHA, 'EXP054 baseline bytes differ from SRC067')
    gatepath = EXP52/'metrics/data-gate.json'
    need(digest(gatepath.read_bytes())=='6949e5af09d14bc0485dfd8e5d849ebbb55ceae157baf4be15842e3b0f6234ee','Frozen EXP052 gate changed')
    gate = read(gatepath)
    holdout = set(gate['holdout_prompt_group_hashes'])
    records = jsonl(ROOT/'experiments/EXP-040-codex300-concurrency2/collection/combined-records.jsonl')
    elig = jsonl(ROOT/'experiments/EXP-041-aligned-channel-repeat-audit/metrics/training-eligibility.jsonl')
    by_key = {x['job_key']:x for x in records}
    admiss = {x['job_key']:x for x in elig}
    need(len(holdout)==50 and len(by_key)==len(records)==300 and len(admiss)==len(elig), 'Source bank/holdout counts differ')
    held_answers = [x for x in records if x['split']=='holdout']
    need(len(held_answers)==60 and all(digest(norm(x['base_prompt']).encode()) in holdout for x in held_answers),
         'Original holdout ancestors differ')
    arm_rows = {}
    sharedfile = EXP53/'inputs/shared.csv'
    need(digest(sharedfile.read_bytes())=='b98c246578afbc38b9bf3c0be1e666a41cc15d051c64d03e875ab0c85da1bd3b','EXP053 shared input changed')
    train = readrows(ROOT/'train.csv')
    shared = readrows(sharedfile)
    need(len(train)==2699 and len(shared)==10743,'Base inputs differ')
    existing_texts = {norm(x['text']) for x in train+shared}
    existing_groups = {x['prompt_id'] for x in train+shared}
    counts = {}
    for n in ARMS:
        path = EXP52/'data'/f'four{n}.csv'
        need(digest(path.read_bytes()) == RAW_HASH[n], 'Frozen added CSV bytes differ')
        rows = readrows(path)
        arm_rows[n] = rows
        groups = {}
        seen_text, seen_source = set(), set()
        for x in rows:
            key=x['job_key']; s=by_key[key]; e=admiss[key]
            ancestor = digest(norm(x['base_prompt']).encode())
            need(x['row_id']=='codex:'+key and key not in seen_source and x['prompt_id']=='codex:'+s['prompt_group']
                 and x['split']==s['split']==e['split']=='train' and ancestor not in holdout
                 and x['prompt_id'] not in existing_groups and x['model']==s['model_requested']==e['model'],
                 'Unbound source row or leaked ancestor')
            raw = (ROOT/s['attempt_path_project']/'final.txt').read_bytes()
            need(raw==x['text'].encode()==s['text'].encode() and digest(raw)==x['text_sha256']==s['raw_final_sha256']
                 and x['base_prompt']==s['base_prompt'] and x['source_protocol']==s['protocol_version']=='single-paragraph-plain-text-v1',
                 'Raw text/protocol differs')
            need(norm(x['text']) not in existing_texts|seen_text,'Duplicate added text')
            seen_source.add(key);seen_text.add(norm(x['text']))
            groups.setdefault(x['prompt_id'], []).append((x['model'],ancestor))
        need(len(rows)==n and len(groups)==n//4 and
             all({x[0] for x in values}==MODELS and len(values)==4 and len({x[1] for x in values})==1 for values in groups.values()),
             'Incomplete four-label prompt')
        counts[n]={'rows':len(rows),'groups':len(groups),'label_counts':dict(Counter(x['model'] for x in rows)),
                   'raw_downloaded_original_texts_matched':len(seen_source),'holdout_hash_overlap':0}
    keys = {x['job_key'] for x in arm_rows[120]}
    need([x for x in arm_rows[240] if x['job_key'] in keys]==arm_rows[120], '120 rows are not an exact ordered subset of 240')
    return train,shared,arm_rows,counts


def weights(train,shared,new):
    labels=sorted({x['model'] for x in train})
    i={key:j for j,key in enumerate(labels)}
    y,ey,ny=[np.asarray([i[x['model']] for x in table],dtype=int) for table in (train,shared,new)]
    h,k=np.bincount(y,minlength=18),np.bincount(ey,minlength=18)
    per=np.divide(.5*h,k)
    old=np.r_[np.where(k[y]>0,.5,1.),per[ey]]
    return labels,y,ey,ny,old,per[ny]


def save(name,obj):
    target=HERE/name
    target.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':
    train, shared, rows, data=audit_data()
    for size in ARMS:
        labels,y,ey,ny,old,added=weights(train,shared,rows[size])
        data[size].update(old_weight_rows=len(old),new_per_row={labels[int(i)]:float(added[ny==i][0]) for i in np.unique(ny)},
                          added_total_mass=float(added.sum()),input_sha256=RAW_HASH[size])
    save('data-verification.json',{'status':'PASS_RAW_DATA_ONLY','evidence':data,'zero_sha256':ZERO_SHA,
                                   'cross_arm':'four120 is an ordered exact subset of four240',
                                   'limit':'Model identity is requested-only. Exact/known prompt isolation, not exhaustive semantic independence.'})
    print(json.dumps(data,ensure_ascii=False))
