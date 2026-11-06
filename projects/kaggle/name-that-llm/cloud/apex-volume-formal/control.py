"""EXP056/057 private save, readback and commit acceptance. No competition submission route."""
import argparse
import ast
import base64
import csv
import concurrent.futures
import hashlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import zlib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMMON = Path(__file__).resolve().parent
BASE = ROOT / 'cloud/apex-zero-054'
SRC067 = next((ROOT / 'research/solutions').glob('SRC-067-*'))
RAW_BASE = 'ca4c9e18ed32dcd8168b8118b192d39d657803d30dcd89a564b1c2d1d9c846dd'
SHARED_SHA = 'b98c246578afbc38b9bf3c0be1e666a41cc15d051c64d03e875ab0c85da1bd3b'
ARMS = {'four120': (56, 120, 'a4e86314a77f7ed699dbfb107b32a8263a1ff09fc7c3ec92b98d4028d487005d'),
        'four240': (57, 240, '694d91e2536db2d77d81f8d8b32a802763a750a2e9643fdb0ba6a10b76755a01')}
LABELS = {'gpt-5.6-luna', 'gpt-5.6-terra', 'gpt-5.6-sol', 'gpt-6-astra'}


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_module('apex_zero_control_reused', BASE / 'control.py')
require, sha, read, now = base.require, base.sha, base.load, base.now


def file_entry(file):
    return {'path': file.relative_to(ROOT).as_posix(), 'sha256': sha(file.read_bytes()), 'bytes': file.stat().st_size}


def emit_once(file, obj, variable_time=None):
    if file.exists():
        old = read(file)
        for key in variable_time or ():
            if key in old:
                obj[key] = old[key]
        require(obj == old, 'Refusing changed receipt: ' + str(file))
    else:
        base.save(file, obj)
    return obj


def configure(arm):
    exp, rows, expected_csv_sha = ARMS[arm]
    folder = ROOT / f'cloud/apex-codex{rows}-0{exp}'
    ref = f'zhjianduan/ntl-apex-codex{rows}-0{exp}'
    folder.mkdir(exist_ok=True)
    return {'arm': arm, 'experiment': exp, 'rows': rows, 'groups': rows // 4, 'csv_sha': expected_csv_sha,
            'folder': folder, 'ref': ref, 'src': folder / 'kernel/apex.ipynb',
            'meta': folder / 'kernel/kernel-metadata.json', 'manifest': COMMON / 'manifest.json'}


def csv_proof(blob, cfg):
    expected = ROOT / f"experiments/EXP-052-stylegram-four-model-volume/data/{cfg['arm']}.csv"
    require(sha(blob) == cfg['csv_sha'] and blob == expected.read_bytes(), 'Embedded CSV differs from frozen train-only input')
    records = list(csv.DictReader(io.StringIO(blob.decode('utf8'), newline='')))
    fields = ['row_id','origin','prompt_id','model','text','base_prompt','text_sha256','job_key','split','source_protocol']
    require(records and list(records[0]) == fields and len(records) == cfg['rows'], 'Unexpected input rows/columns')
    require(all(r['split'] == 'train' and r['origin'] == 'codex_subscription_aligned_v1' and r['source_protocol'] == 'single-paragraph-plain-text-v1' and
                r['prompt_id'].startswith('codex:') and sha(r['text']) == r['text_sha256'] and r['row_id']=='codex:'+r['job_key'] for r in records), 'Changed input identities or raw text')
    require(len({r['row_id'] for r in records}) == len(records), 'Duplicate selected row')
    panels = {}
    for row in records:
        panels.setdefault(row['prompt_id'], []).append(row)
    require(len(panels)==cfg['groups'] and all(len(p)==4 and {x['model'] for x in p}==LABELS and len({x['base_prompt'] for x in p})==1 for p in panels.values()), 'Incomplete four-answer prompt panel')
    return {'rows':len(records), 'groups':len(panels), 'csv_sha256':sha(blob),
            'per_label':dict(sorted(Counter(x['model'] for x in records).items())), 'full_text_hashes_verified':True,
            'holdout_rows':0, 'scope':'materialized embedded training input; fast commit does not fit Stylegram'}


def decoded_csv(module, cfg):
    candidates = []
    for node in ast.walk(ast.parse(module)):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, (str, bytes)) or len(node.value) < 2000:
            continue
        try:
            data = zlib.decompress(base64.b64decode(node.value, validate=True))
        except (ValueError, zlib.error):
            continue
        if data.startswith(b'row_id,origin,prompt_id,model,'):
            candidates.append(data)
    require(len(candidates)==1, 'Expected exactly one embedded training CSV in materialized comp module')
    return csv_proof(candidates[0],cfg)


def package(cfg):
    manifest = read(cfg['manifest'])
    arm = manifest['arms'][cfg['arm']]
    source = cfg['src']; meta_path = cfg['meta']; src = source.read_bytes(); meta = read(meta_path)
    require(sha(src)==arm['notebook_sha256'] and sha(meta_path.read_bytes())==arm['metadata_sha256'], 'Candidate source/metadata hash drift')
    require(arm['added_csv_sha256']==cfg['csv_sha'] and arm['rows']==cfg['rows'] and arm['groups']==cfg['groups'] and arm['shared_sha256']==SHARED_SHA,'Candidate dataset binding mismatch')
    baseline_meta = read(BASE/'kernel/kernel-metadata.json')
    for field in ('competition_sources','dataset_sources','kernel_sources','model_sources','language','kernel_type','is_private','enable_gpu','enable_tpu','enable_internet','machine_shape','docker_image','docker_image_pinning_type'):
        require(meta.get(field)==baseline_meta.get(field),'Candidate changes saved zero-arm metadata: '+field)
    require(meta['id']==cfg['ref'] and meta['code_file']=='apex.ipynb', 'Incorrect candidate destination')
    require(sha((BASE/'kernel/apex-zero.ipynb').read_bytes())==RAW_BASE,'EXP054 baseline bytes drift')
    original = read(BASE/'kernel/apex-zero.ipynb'); candidate = json.loads(src)
    require(original.keys()==candidate.keys() and all(original[k]==candidate[k] for k in original if k!='cells'), 'Non-cell notebook fields changed')
    require(len(original['cells'])==len(candidate['cells']), 'Notebook cell count changed')
    changed = [i for i,(a,b) in enumerate(zip(original['cells'],candidate['cells'])) if a!=b]
    require(changed==[12,14],'Only the approved comp algorithm and execution guard cells may change')
    old_cell, new_cell = original['cells'][12], candidate['cells'][12]
    require(old_cell.keys()==new_cell.keys() and all(old_cell[k]==new_cell[k] for k in old_cell if k!='source'),'Unexpected cell metadata change')
    def src_text(cell):return ''.join(cell['source']) if isinstance(cell['source'],list) else cell['source']
    require(src_text(old_cell).startswith('%%writefile comp_stack.py\n') and src_text(new_cell).startswith('%%writefile comp_stack.py\n'), 'Changed cell is not the comp module')
    guard_old,guard_new=original['cells'][14],candidate['cells'][14]
    require(guard_old.keys()==guard_new.keys() and all(guard_old[k]==guard_new[k] for k in guard_old if k!='source'),'Execution guard changed cell metadata')
    require('volume_preflight.json' in src_text(guard_new) and 'FAST_COMMIT' in src_text(guard_new),'Missing public volume precheck or original fast-commit branch')
    proof=decoded_csv(src_text(new_cell).split('\n',1)[1],cfg)
    return source, meta, arm, proof


def validate_preflight(cfg):
    source,meta,arm,proof=package(cfg)
    old = read(SRC067/'commit-output-comparison.json')['results'][0]['materialized_module_hashes']
    baseline = {name:v['sha256'] for name,v in old.items()}
    new = dict(baseline, **{'comp_stack.py':arm['comp_stack_sha256']})
    require(baseline['comp_stack.py'] != new['comp_stack.py'], 'Comp module must contain the treatment')
    value={'status':'PASS_LOCAL_CANDIDATE_PREFLIGHT','verified_at':now(),'ref':cfg['ref'],'arm':cfg['arm'],
           'source_sha256':sha(source.read_bytes()),'metadata_sha256':sha(cfg['meta'].read_bytes()),'manifest_sha256':sha(cfg['manifest'].read_bytes()),
           'baseline_source_sha256':RAW_BASE,'old_module_hashes':baseline,'expected_new_module_hashes':new,
           'embedded_training_rows':proof,'shared_sha256':SHARED_SHA,'shared_rows':10743,'asset_count':31,'competition_sources':['name-that-llm'],
           'exact_zero_arm_attachment_pins':True,'internet':False,'private':True,'docker_image':meta['docker_image'],'machine_shape':meta['machine_shape']}
    return emit_once(cfg['folder']/'preflight-candidate.json',value,['verified_at'])


def owned(api):
    rows=[]
    for page in range(1,11):
        part=api.kernels_list(mine=True,user='zhjianduan',page=page,page_size=100)
        require(part is not None,'Owned inventory unavailable')
        rows.extend(part)
        if len(part)<100:return rows
    raise ValueError('Owned notebook inventory incomplete')


def source_inventory(api):
    historical=read(BASE/'owned-version-source-check.json')
    seen={(v['ref'],v['version']):v for v in historical['versions'] if v['version']}
    candidates=api.kernels_list(user='zhjianduan',mine=True,competition='name-that-llm',page_size=100)
    require(candidates is not None and len(candidates)<100,'Competition notebook inventory incomplete')
    def one(notebook):
        current=base.get(api,notebook.ref); version=current['metadata'].get('currentVersionNumber',0)
        current_hash=sha(current['blob']['source']) if current.get('blob',{}).get('source') else None
        fresh=[{'ref':notebook.ref,'version':version or None,'sha256':current_hash}]
        # EXP054 already checked immutable historic versions; only new old versions need fetching.
        for v in range(1,version):
            if (notebook.ref,v) in seen:continue
            snap=base.get(api,notebook.ref,v)
            fresh.append({'ref':notebook.ref,'version':v,'sha256':sha(snap['blob']['source'])})
        return fresh
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        fresh=[x for group in executor.map(one,candidates) for x in group]
    return {'observed_at':now(),'current_competition_notebooks':len(candidates),'snapshots':fresh,'existing_exp054_history':file_entry(BASE/'owned-version-source-check.json')}


def discover(api,cfg):
    info=validate_preflight(cfg);items=owned(api)
    target=[item.ref for item in items if item.ref==cfg['ref']]
    subs=api.competition_submissions('name-that-llm',page_size=100)
    require(subs is not None and len(subs)<100,'Competition submission inventory incomplete')
    duplicate_subs=[{k:v for k,v in s.to_dict().items() if k in ('ref','description','url','status','scriptVersionId')} for s in subs
                    if cfg['ref'].lower() in str(s.to_dict()).lower() or ('EXP-0'+str(cfg['experiment'])).lower() in str(s.to_dict()).lower()]
    historical=read(BASE/'owned-version-source-check.json')
    existing_matches=[v for v in historical['versions'] if v['source_sha256']==info['source_sha256']]
    snapshot=COMMON/'pre-release-source-inventory.json'
    if not snapshot.exists():base.save(snapshot,source_inventory(api))
    current=read(snapshot); fresh=current['snapshots']
    for item in items:
        if item.ref.startswith('zhjianduan/ntl-apex-codex') and item.ref not in {v['ref'] for v in fresh}:
            snap=base.get(api,item.ref);body=snap.get('blob',{}).get('source','');fresh.append({'ref':item.ref,'version':snap['metadata'].get('currentVersionNumber'), 'sha256':sha(body) if body else None})
    matches=existing_matches+[x for x in fresh if x['sha256']==info['source_sha256']]
    disc={'observed_at':now(),'ref':cfg['ref'],'owned_count':len(items),'submission_count':len(subs),'target_ref_exists':target,'prior_submission_matches':duplicate_subs,
          'source_sha256':info['source_sha256'],'existing_source_matches':matches,'matches':matches,'existing_exp054_history':file_entry(BASE/'owned-version-source-check.json'),'fresh_snapshots':fresh}
    emit_once(cfg['folder']/'owned-version-source-check.json',disc,['observed_at'])
    require(not target and not duplicate_subs and not matches,'Existing destination, exact source, or submission requires review')
    print('DEDUP_PASS',cfg['ref'],len(fresh),flush=True)


def release(cfg,note):
    require(note and len(note.strip())>=12,'Explicit parent release note required')
    p=validate_preflight(cfg);d=read(cfg['folder']/'owned-version-source-check.json')
    require(not(d['matches'] or d['target_ref_exists'] or d['prior_submission_matches']),'Discovery blocks release')
    receipt={'approved_at':now(),'approval_scope':'private GPU quick-save and run only; no competition submit','approval_note':note,
             'ref':cfg['ref'],'source_sha256':p['source_sha256'],'metadata_sha256':p['metadata_sha256'],
             'preflight_sha256':sha((cfg['folder']/'preflight-candidate.json').read_bytes()),
             'manifest_sha256':p['manifest_sha256'],'discovery_sha256':sha((cfg['folder']/'owned-version-source-check.json').read_bytes())}
    emit_once(cfg['folder']/'remote-release.json',receipt,['approved_at'])


def bind(cfg, require_release):
    p=validate_preflight(cfg)
    if require_release:
        r=read(cfg['folder']/'remote-release.json')
        require(r['ref']==cfg['ref'] and r['source_sha256']==p['source_sha256'] and r['metadata_sha256']==p['metadata_sha256'] and r['manifest_sha256']==p['manifest_sha256'] and
                r['preflight_sha256']==sha((cfg['folder']/'preflight-candidate.json').read_bytes()) and r['discovery_sha256']==sha((cfg['folder']/'owned-version-source-check.json').read_bytes()),'No matching parent release for frozen candidate')
    base.HERE=cfg['folder'];base.REF=cfg['ref'];base.RAW_SHA=p['source_sha256']
    base.install=lambda:package(cfg)[:2]
    return p


def collect(api,cfg):
    from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest
    bind(cfg,True)
    require(base.status(api)=='COMPLETE','Commit not complete; output read skipped')
    q=ApiListKernelSessionOutputRequest();q.user_name,q.kernel_slug=cfg['ref'].split('/');q.version_label='v2';q.page_size=100
    with api.build_kaggle_client() as k:
        output=k.kernels.kernels_api_client.list_kernel_session_output(q)
    require(not getattr(output,'next_page_token',None),'Output listing incomplete')
    ids={}
    for row in output.files:
        m=re.search(r'/kf/(\d+)/',row.url)
        if m:ids[row.file_name]=int(m.group(1))
    require('submission.parquet' in ids and len(set(ids.values()))==1,'Cannot bind global script version')
    identity={'ref':cfg['ref'],'version':2,'script_version_id':ids['submission.parquet'],'file_ids':ids,'signed_urls_omitted':True,'observed_at':now()}
    emit_once(cfg['folder']/'script-identity.json',identity,['observed_at'])
    out=cfg['folder']/'outputs-v2';out.mkdir(exist_ok=True)
    paths,token=api.kernels_output(cfg['ref']+'/2',str(out),file_pattern=r'^(submission\.parquet|infer\.py|comp_stack\.py|gauss\.py|sibling_style\.py|stylegram_volume_receipt\.json|volume_preflight\.json|volume_comp_execution\.json)$',quiet=True)
    require(not token and (out/'submission.parquet').exists(),'Incomplete download')
    print('COLLECTED',identity['script_version_id'],[Path(p).name for p in paths])


def verify(cfg):
    import pandas as pd
    p=bind(cfg,True);folder=cfg['folder'];out=folder/'outputs-v2';ids=read(folder/'script-identity.json');rb=read(folder/'readback-v2.json')
    require(ids['ref']==rb['ref']==cfg['ref'] and ids['version']==rb['version']==2 and rb['source_exact'] and rb['source_bytes_sha256']==p['source_sha256'] and rb['asset_count']==31 and rb['private'] and not rb['internet'],'Invalid V2 readback')
    actual_modules={k:sha((out/k).read_bytes()) for k in p['expected_new_module_hashes']}
    require(actual_modules==p['expected_new_module_hashes'],'Materialized module hash mismatch')
    payload=decoded_csv((out/'comp_stack.py').read_text(encoding='utf8'),cfg)
    # The authoring agent's verifier also proves the unchanged portion of comp and static training connection.
    cmd=[sys.executable,'-B','-X','utf8',str(COMMON/'generate.py'),'verify-materialized',cfg['arm'],str(out/'comp_stack.py')]
    checked=subprocess.run(cmd,cwd=ROOT,encoding='utf8',capture_output=True,check=False)
    require(checked.returncode==0,'Generator failed materialized-script validation: '+(checked.stderr or checked.stdout)[-450:])
    emit_once(folder/'materialized-source-check.json',{'method':'generator verify-materialized plus independent AST/decoded CSV verification',
              'generator_stdout':checked.stdout,'module_hashes':actual_modules,'old_module_hashes':p['old_module_hashes'],'added_rows':payload})
    sub=pd.read_parquet(out/'submission.parquet');test=pd.read_csv(ROOT/'test.csv',dtype={'id':str},keep_default_na=False)
    require(list(sub.columns)==['id','model'] and len(sub)==len(test)==3 and sub.id.is_unique and sub.id.astype(str).tolist()==test.id.tolist() and sub.model.tolist()==['gpt-5.5']*3,'Invalid commit placeholder output')
    log=out/(cfg['ref'].split('/')[1]+'.log'); body=log.read_text(encoding='utf8');signals=['FAST_COMMIT True','llama source alexchilton/llama-3-1-8b-base','variant E dropped flag_gemini37_routing_risk 0','open safe pool 190588']
    for signal in signals: require(signal in body,'Missing fixed public log signal: '+signal)
    volume=read(out/'volume_preflight.json')
    checks={'status':'PASS','rows':cfg['rows'],'groups':cfg['groups'],'source_csv_sha256':cfg['csv_sha'],'shared_rows':10743,'shared_groups':738,
            'shared_sha256':SHARED_SHA,'comp_stack_sha256':actual_modules['comp_stack.py'],'arm':cfg['arm'],
            'fast_commit':True,'competition_rerun_flag_present':False,'fit_count':0}
    for field,expected in checks.items():require(volume.get(field)==expected,'Runtime volume preflight mismatch: '+field)
    require(not any((out/f).exists() for f in ['stylegram_volume_receipt.json','volume_comp_execution.json']),'Unexpected hidden-training receipt on public fast branch; investigate')
    evidence={'status':'PASS_PUBLIC_COMMIT_AND_MATERIALIZED_TRAINING_INPUT','verified_at':now(),'ref':cfg['ref'],'version':2,'script_version_id':ids['script_version_id'],
              'source_sha256':p['source_sha256'],'embedded_input_rows':payload,'expected_shared_sha256':SHARED_SHA,'expected_shared_rows':10743,
              'shared_recomputed_by_this_fast_commit':True,'actual_public_data_preflight':volume,'actual_public_data_preflight_file':file_entry(out/'volume_preflight.json'),'stylegram_fit_observed':False,'hidden_execution_observed':False,
              'module_hashes':actual_modules,'public_ids':test.id.tolist(),'fast_signals':signals,'log':file_entry(log),'submission':file_entry(out/'submission.parquet'),
              'zero_arm_identical_placeholder_sha256':read(BASE/'ready-for-submission.json')['submission_sha256'] == sha((out/'submission.parquet').read_bytes()),
              'source_integrity_validation':file_entry(folder/'materialized-source-check.json'),'readback':file_entry(folder/'readback-v2.json')}
    emit_once(folder/'commit-verification.json',evidence,['verified_at'])
    ready={'status':'READY_FOR_SUBMISSION','ref':cfg['ref'],'version':2,'script_version_id':ids['script_version_id'],'source_sha256':p['source_sha256'],
           'submission_sha256':evidence['submission']['sha256'],'private':True,'internet':False,'asset_count':31,'verified_at':evidence['verified_at'],
           'evidence_paths':[file_entry(folder/f)['path'] for f in ['commit-verification.json','materialized-source-check.json','readback-v2.json','script-identity.json','outputs-v2/submission.parquet']],
           'arm':cfg['arm'],'added_csv_sha256':payload['csv_sha256'],'added_rows_verified_in_materialized_source':payload['rows'],'hidden_fit_status':'NOT_OBSERVED','competition_submission_owner':'root'}
    emit_once(folder/'ready-for-submission.json',ready,['verified_at']);print(json.dumps(ready,ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('arm',choices=ARMS);parser.add_argument('action',choices=['source-inventory','preflight','discover','release','save','run','status','collect','verify']);parser.add_argument('--approval-note')
    args=parser.parse_args();cfg=configure(args.arm)
    if args.action=='preflight':print(json.dumps(validate_preflight(cfg),ensure_ascii=False));return
    if args.action=='release':release(cfg,args.approval_note);return
    if args.action=='verify':verify(cfg);return
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    if args.action=='source-inventory':
        path=COMMON/'pre-release-source-inventory.json'
        if not path.exists():base.save(path,source_inventory(api))
        value=read(path);print('INVENTORY',value['current_competition_notebooks'],len(value['snapshots']));return
    if args.action=='discover':discover(api,cfg);return
    if args.action in ('save','run'):
        bind(cfg,True);base.dispatch(api,1 if args.action=='save' else 2);return
    if args.action=='status':bind(cfg,True);base.status(api);return
    if args.action=='collect':collect(api,cfg)


if __name__=='__main__':main()
