"""Build A1 code submission from the frozen own ATLAS20 V2; no remote action."""
import base64
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT / 'experiments/EXP-046-atlas20-a1-submission'
sys.path.insert(0, str(EXP))
from patch_comp import patch_source

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def main():
    assert not any((HERE/name).exists() for name in ['save-only-intent.json','run-intent.json','submission-intent.json']), 'Cannot rebuild a remotely dispatched candidate'
    baseline = ROOT/'reviews/own-scheme-readiness-20261030/baseline-v2.ipynb'
    assert sha(baseline) == '6c9145f038ccffbe08dc6da6db3d1dcf3d2e0e7d2286927bfc8800ef5148cfb8'
    nb = json.loads(baseline.read_text(encoding='utf-8'))
    original = ROOT/'experiments/EXP-009-baseline-data-lineage/comp_stack.py'
    old_comp = ''.join(nb['cells'][8]['source']).split('\n', 1)[1].encode()
    assert old_comp == original.read_bytes().replace(b'\r\n',b'\n')
    patched, patch_receipt = patch_source(original.read_bytes())
    patched = patched.replace(b'\r\n',b'\n')
    patch_receipt['notebook_source_sha256'] = hashlib.sha256(patched).hexdigest()
    patch_receipt['archive_vs_notebook_only_newlines_differ'] = True
    payload = HERE/'payload'
    payload.mkdir(exist_ok=True)
    for name in ['a1_stylegram.py', 'input_preflight.py', 'extra-inputs.json']:
        shutil.copyfile(EXP/name, payload/name)
    shutil.copyfile(ROOT/'cloud/format120-044/payload/stylegram_core.py', payload/'stylegram_core.py')
    assert sha(payload/'stylegram_core.py') == 'caa46bd121164aba891c3d78e662540db9294d653db09f53d02833c1b4ab77ef'
    shutil.copyfile(ROOT/'cloud/full-stack-capture-r1/payload/preflight_manifest.json', payload/'preflight_manifest.json')
    shutil.copyfile(ROOT/'cloud/full-stack-capture-r1/input_policy.py', payload/'input_policy.py')
    shutil.copyfile(HERE/'runtime_preflight.py', payload/'runtime_preflight.py')
    shutil.copyfile(HERE/'run_receipts.py', payload/'run_receipts.py')
    hashes = {p.name:sha(p) for p in sorted(payload.iterdir()) if p.name != 'payload-hashes.json'}
    assert set(hashes)=={'a1_stylegram.py','input_preflight.py','extra-inputs.json','stylegram_core.py','preflight_manifest.json','input_policy.py','runtime_preflight.py','run_receipts.py'}
    save(payload/'payload-hashes.json', hashes)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(payload.iterdir()):
            entry = zipfile.ZipInfo(p.name, date_time=(2026,11,2,0,0,0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(entry, p.read_bytes())
    blob = archive.getvalue()
    bootstrap = '''# EXP046: frozen A1 implementation and input checks; no cached validation predictions.
import base64, hashlib, io, json, pathlib, sys, zipfile, os
_payload = pathlib.Path('/kaggle/working/ntl_a1_payload')
_payload.mkdir(exist_ok=False)
_blob = base64.b64decode(BLOB)
assert hashlib.sha256(_blob).hexdigest() == DIGEST
with zipfile.ZipFile(io.BytesIO(_blob)) as _z:
    assert sorted(_z.namelist()) == NAMES
    assert all('/' not in n and '\\\\' not in n and n not in ('.','..') for n in _z.namelist())
    _z.extractall(_payload)
for _name, _digest in json.loads((_payload/'payload-hashes.json').read_text()).items():
    assert hashlib.sha256((_payload/_name).read_bytes()).hexdigest() == _digest, _name
sys.path.insert(0, str(_payload))
from runtime_preflight import run_runtime_preflight
_a1_preflight = run_runtime_preflight(_payload)
os.environ['NTL_A1_MODULE_DIR'] = str(_payload)
os.environ['NTL_A1_EXTRA'] = _a1_preflight['extra_path']
'''.replace('BLOB', repr(base64.b64encode(blob).decode())).replace('DIGEST',repr(hashlib.sha256(blob).hexdigest())).replace('NAMES',repr(sorted(p.name for p in payload.iterdir())))
    compile(bootstrap, 'bootstrap', 'exec')
    for cell in nb['cells']:
        if cell['cell_type']=='code':
            cell['outputs']=[]; cell['execution_count']=None
    nb['cells'][0]['source'] = '# EXP046: A1 public-data Stylegram + ATLAS20\n\nPrivate controlled submission from own ATLAS20 V2 (EXP004). Only Stylegram is replaced with the frozen official2699 + public10073/share0.5 recipe. Other models, original family-gate data and ensemble settings are preserved. Public quick commits emit a placeholder; competition reruns execute the full model pipeline.\n\nCredits: original ATLAS20/Who am I solution by omerfarukmerey, romantamrazov, dhruvtibarewal, mercurydust and attached model/data contributors; original source comments and attribution retained. Local cached-panel validation is not a leaderboard score.\n'
    setup = ''.join(nb['cells'][2]['source'])
    old_guard = "FAST_COMMIT = len(_pd.read_csv(DATA + '/test.csv')) < 50   # fork: the 3-row commit run skips the heavy cells; the hidden rerun runs everything"
    assert setup.count(old_guard)==1
    setup = setup.replace(old_guard, "FAST_COMMIT = ('KAGGLE_IS_COMPETITION_RERUN' not in os.environ and 0 < len(_pd.read_csv(DATA + '/test.csv')) < 50)  # hidden rerun ALWAYS executes full inference")
    nb['cells'][2]['source'] = bootstrap + '\n' + setup
    nb['cells'][8]['source'] = '%%writefile comp_stack.py\n'+patched.decode()
    nb['cells'][10]['source'] = '''import time as _t
import os, subprocess, sys, json
import numpy as np
_c0 = _t.time()
os.environ['NTL_DROP_Q35'] = '0'
if not FAST_COMMIT:
    with open('/kaggle/working/comp-process.log', 'w') as _log:
        _process = subprocess.Popen([sys.executable, '-u', 'comp_stack.py'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for _line in _process.stdout:
            _log.write(_line); _log.flush(); print(_line, end='', flush=True)
        _rc = _process.wait()
        if _rc: raise subprocess.CalledProcessError(_rc, _process.args)
    _comp_meta = json.load(open('/kaggle/working/comp_meta.json'))
    _comp_logits = np.load('/kaggle/working/comp_logit.npy', allow_pickle=False)
    from a1_stylegram import LABELS as _labels
    _test_ids = _pd.read_csv(DATA+'/test.csv').id.astype(str).tolist()
    assert _comp_meta['labels'] == list(_labels)
    assert _comp_meta['ids'] == _test_ids and _comp_meta['family_gates_applied'] is True
    assert _comp_logits.shape == (len(_test_ids),18) and np.isfinite(_comp_logits).all()
    from run_receipts import verify_comp_log
    pathlib.Path('/kaggle/working/comp_execution_receipt.json').write_text(json.dumps(verify_comp_log(pathlib.Path('/kaggle/working/comp-process.log').read_text(),len(_test_ids)),indent=2))
    _receipt_lines = [x[len('A1_STYLEGRAM_RECEIPT '):] for x in open('/kaggle/working/comp-process.log') if x.startswith('A1_STYLEGRAM_RECEIPT ')]
    assert len(_receipt_lines)==1
    pathlib.Path('/kaggle/working/a1_training_receipt.json').write_text(_receipt_lines[0])
    del _comp_logits
COMP_SEC = _t.time()-_c0
print('comp stack wall time', round(COMP_SEC))
'''
    main_src = ''.join(nb['cells'][12]['source'])
    assert main_src.count('    subprocess.run(cmd, check=True)')==1
    main_src = main_src.replace('    subprocess.run(cmd, check=True)', '''    with open('/kaggle/working/main-process.log','w') as _main_log:
        _main_process = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for _line in _main_process.stdout:
            _main_log.write(_line); _main_log.flush(); print(_line,end='',flush=True)
        _main_rc=_main_process.wait()
        if _main_rc: raise subprocess.CalledProcessError(_main_rc,cmd)
    from run_receipts import verify_main_log
    pathlib.Path('/kaggle/working/main_execution_receipt.json').write_text(json.dumps(verify_main_log(pathlib.Path('/kaggle/working/main-process.log').read_text(),len(pd.read_csv(DATA+'/test.csv'))),indent=2))''')
    main_src += '''
from a1_stylegram import LABELS as _labels
_test_final = pd.read_csv(os.path.join(DATA,'test.csv'))
assert submission.id.astype(str).tolist() == _test_final.id.astype(str).tolist()
assert set(submission.model).issubset(set(_labels))
_submission_receipt = {'experiment_id':'EXP-046', 'status':'PASS', 'fast_commit':FAST_COMMIT,
    'competition_rerun_flag_present':'KAGGLE_IS_COMPETITION_RERUN' in os.environ,
    'rows':len(submission), 'columns':list(submission.columns),
    'prediction_kind':'public_placeholder' if FAST_COMMIT else 'full_hidden_pipeline',
    'submission_sha256':hashlib.sha256(pathlib.Path('/kaggle/working/submission.parquet').read_bytes()).hexdigest(),
    'atlas_w':.20,'comp_w':.15,'sinkhorn':.45,'stylegram_recipe':'official2699+public10073_share0.5_raw_margin'}
pathlib.Path('/kaggle/working/submission_receipt.json').write_text(json.dumps(_submission_receipt,indent=2))
print(json.dumps(_submission_receipt))
'''
    nb['cells'][12]['source'] = main_src
    for cell in nb['cells']:
        if cell['cell_type']=='code':
            source=''.join(cell['source'])
            if source.startswith('%%writefile '):source=source.split('\n',1)[1]
            compile(source,'candidate-cell','exec')
    kernel = HERE/'kernel'; kernel.mkdir(exist_ok=True)
    save(kernel/'a1.ipynb',nb)
    metadata=json.loads((ROOT/'cloud/full-stack-capture-r1/kernel/kernel-metadata.json').read_text())
    metadata.update(id='zhjianduan/name-that-llm-a1-public-stylegram-046', title='Name That LLM A1 Public Stylegram 046', code_file='a1.ipynb')
    metadata['dataset_sources'].append('zhjianduan/ntl-stylegram-cv-inputs-v1/1')
    assert sum(len(metadata[k]) for k in ['dataset_sources','model_sources','kernel_sources','competition_sources'])==32
    save(kernel/'kernel-metadata.json',metadata)
    save(EXP/'metrics/comp-patch.json',patch_receipt)
    (EXP/'metrics/comp-patch.diff').write_text(patch_receipt['unified_diff'],encoding='utf-8')
    save(EXP/'metrics/package-preflight.json', {'status':'PASS_LOCAL_PACKAGE','baseline_sha256':sha(baseline),
        'notebook_sha256':sha(kernel/'a1.ipynb'),'metadata_sha256':sha(kernel/'kernel-metadata.json'),
        'payload_hashes':hashes, 'payload_zip_sha256':hashlib.sha256(blob).hexdigest(),
        'unchanged_source_cells':[4,6], 'comp_patch_sha256':patch_receipt['patched_sha256'],
        'cloud_runtime_not_yet_checked':True,'attachment_count':32,'private':True,'internet':False})
    print('PASS: A1 package built; 32 fixed inputs; no cloud run started')

if __name__=='__main__':main()
