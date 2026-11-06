"""Read, bind and independently replay a captured baseline on CPU; no scoring."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import types

ROOT=Path(__file__).resolve().parents[2]
import pandas as pd
# Execute precisely the hashed bytes rather than a potentially stale .pyc.
HELPER_HASHES={}
for module_name in ('full_stack_cache','full_stack_observer','full_stack_replay'):
    path=ROOT/'src'/(module_name+'.py')
    content=path.read_bytes()
    HELPER_HASHES[str(path)]=hashlib.sha256(content).hexdigest()
    module=types.ModuleType(module_name)
    module.__file__=str(path)
    sys.modules[module_name]=module
    exec(compile(content,str(path),'exec'),module.__dict__)
from full_stack_cache import read_cache, LABELS
from full_stack_observer import decode_events
from full_stack_replay import verify_replay


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root',type=Path,required=True)
    parser.add_argument('--test-csv',type=Path,required=True)
    parser.add_argument('--receipts-json',type=Path,required=True,
                        help='External main/comp manifest SHA256 anchors from captured commit receipts')
    parser.add_argument('--inventory-json',type=Path,required=True,
                        help='Explicit expected_main_members and expected_comp_members lists, frozen before capture')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.test_csv,args.receipts_json,args.inventory_json]}
    test=pd.read_csv(args.test_csv,dtype={'id':str},keep_default_na=False)
    if not {'id','text'}.issubset(test.columns):raise ValueError('test CSV needs id/text')
    receipts=json.loads(args.receipts_json.read_text(encoding='utf-8'))
    inventory=json.loads(args.inventory_json.read_text(encoding='utf-8'))
    for component in ('main','comp'):
        digest=(receipts.get(component) or {}).get('manifest_sha256')
        if not isinstance(digest,str) or re.fullmatch(r'[0-9a-f]{64}',digest) is None:
            raise ValueError('External lowercase SHA256 anchor required: '+component)
    members=[]
    for key in ('expected_main_members','expected_comp_members'):
        values=inventory.get(key)
        if not isinstance(values,list) or not values or not all(isinstance(x,str) for x in values) or len(values)!=len(set(values)):
            raise ValueError('Explicit nonempty unique member list required: '+key)
        members.append(values)
    snapshots={name:read_cache(args.cache_root,name,test.id.tolist(),test.text.tolist(),LABELS,
        required_notes=['event-index'],manifest_sha256=receipts[name]['manifest_sha256']) for name in ('main','comp')}
    run_ids={s.metadata['run_id'] for s in snapshots.values()}
    if len(run_ids)!=1:raise ValueError('Main and comp were captured in different runs')
    contexts={s.metadata.get('context_sha256') for s in snapshots.values()}
    if len(contexts)!=1 or not all(isinstance(x,str) and re.fullmatch(r'[0-9a-f]{64}',x) for x in contexts):
        raise ValueError('Main and comp must bind the same capture context SHA256')
    if snapshots['main'].metadata['context']!=snapshots['comp'].metadata['context']:
        raise ValueError('Main and comp context content differs')
    context=snapshots['main'].metadata['context']
    if context['scope']!='synthetic_math':
        for key in ('expected_main_members','expected_comp_members'):
            if context.get(key)!=inventory[key]:
                raise ValueError('Real member inventory must match the pre-capture context: '+key)
    result=verify_replay(decode_events(snapshots['main']),decode_events(snapshots['comp']),members[0],members[1])
    for path,digest in {**before,**HELPER_HASHES}.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('Input changed while verifying: '+path)
    result.update(verified_at_utc=datetime.now(timezone.utc).isoformat(),input_files_sha256=before,
        manifests={k:s.manifest_sha256 for k,s in snapshots.items()},run_id=next(iter(run_ids)),
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper_files_sha256=HELPER_HASHES,capture_context_sha256=next(iter(contexts)),
        helper_import_policy='Executed opening hashed bytes; verified unchanged at close',
        real_model_execution_authenticated=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status':result['status'],'accepted':result['accepted'],
        'scope':result['scope'],'output':str(args.output)},ensure_ascii=False))
    if not result['accepted']:raise SystemExit(2)


if __name__=='__main__':main()
