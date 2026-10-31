"""Run frozen V32 using prepared inputs and persistent, versioned checkpoints."""
import ast
import hashlib
import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from web_checkpoint import Checkpoints, atomic_json, file_hash, signature
from web_instrument import instrument_infer

ROOT = Path(__file__).resolve().parent
WORK = Path(os.environ['V32_WORK_DIR'])
INPUT = os.environ['V32_INPUT_ROOT']
os.chdir(WORK)
assert torch.cuda.device_count() == 1 and 'A100' in torch.cuda.get_device_name(0)
environment = {'python': platform.python_version(), 'gpu_name': torch.cuda.get_device_name(0),
               'gpu_count': 1, 'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
               'packages': {n: md.version(n) for n in ['torch', 'transformers', 'peft', 'numpy', 'pandas',
                            'scipy', 'scikit-learn', 'lightgbm', 'safetensors', 'pyarrow', 'accelerate', 'tokenizers']},
               'platform': 'Colab A100', 'same_container_as_upstream': False}
atomic_json(WORK / 'reproduction_environment.json', environment)
nb = json.loads((ROOT / 'upstream-v32.ipynb').read_text())
source = lambda i: ''.join(nb['cells'][i]['source'])
adapt = lambda text: text.replace('/kaggle/input', INPUT).replace('/kaggle/working', str(WORK))
namespace = {'__name__': '__main__', 'Path': Path, 'INPUT': INPUT}
paths = adapt(source(1))
assert paths.count("BL = mdir('/llama-3.1/', '/8b/')") == 1
paths = paths.replace("BL = mdir('/llama-3.1/', '/8b/')",
                      "BL = str(Path(INPUT) / 'datasets/zhjianduan/llama31-8b-base-verified')")
exec(compile(paths, 'v32_paths.py', 'exec'), namespace)
context_data = {'schema': 1, 'environment': environment,
                'sources': {name: file_hash(ROOT / name) for name in
                    ['upstream-v32.ipynb', 'input-lock.json', 'web_execute.py', 'web_instrument.py', 'web_checkpoint.py']},
                'competition_files': {name: file_hash(Path(namespace['DATA']) / name) for name in ['train.csv', 'test.csv']},
                'verified_inputs': json.loads((WORK / 'pinned_inputs_verified.json').read_text())}
context = signature(context_data)
os.environ['V32_CHECKPOINT_CONTEXT'] = context
atomic_json(WORK / 'checkpoint_context.json', {'fingerprint': context, **context_data})
checkpoints = Checkpoints(os.environ['V32_CHECKPOINT_ROOT'], context)
hashes = {}
for i in (2, 3, 4):
    first, body = source(i).split('\n', 1)
    name = first.split()[-1]
    patched = adapt(body)
    if name == 'infer.py':
        patched = instrument_infer(patched)
    ast.parse(patched)
    (WORK / name).write_text(patched)
    hashes[name] = {'upstream_sha256': hashlib.sha256(body.encode()).hexdigest(),
                    'adapted_sha256': hashlib.sha256(patched.encode()).hexdigest(),
                    'changes': 'input/output paths; stage checkpoint wrappers' if name == 'infer.py' else 'input/output paths only'}
atomic_json(WORK / 'module_provenance.json', hashes)
os.environ['NTL_DROP_Q35'] = '0'
started = time.time()


def complementary_stack():
    start = time.time()
    subprocess.run([sys.executable, '-u', 'comp_stack.py'], check=True)
    return {'logits': np.load(WORK / 'comp_logit.npy', allow_pickle=False),
            'meta': json.loads((WORK / 'comp_meta.json').read_text()), 'wall_seconds': time.time() - start}


comp = checkpoints.call('complementary_stack', complementary_stack)
test = pd.read_csv(Path(namespace['DATA']) / 'test.csv')
assert comp['meta']['ids'] == test.id.astype(str).tolist()
assert comp['logits'].shape == (len(test), 18) and np.isfinite(comp['logits']).all()
np.save(WORK / 'comp_logit.npy', comp['logits'])
atomic_json(WORK / 'comp_meta.json', comp['meta'])
# Preserve the reference's time allowance even when the stack was restored.
namespace['COMP_SEC'] = comp['wall_seconds']
exec(compile(adapt(source(6)), 'v32_main.py', 'exec'), namespace)
submission = pd.read_parquet(WORK / 'submission.parquet')
train = pd.read_csv(Path(namespace['DATA']) / 'train.csv')
assert submission.id.tolist() == test.id.tolist()
assert set(submission.columns) == {'id', 'model'} and submission.id.is_unique
assert submission.model.isin(train.model.unique()).all()
reference = pd.read_parquet(ROOT / 'reference-submission.parquet')
comparison = submission.merge(reference, on='id', suffixes=('_actual', '_reference'), validate='one_to_one')
assert len(comparison) == len(test), 'This launcher is for the visible reference examples'
matched = bool((comparison.model_actual == comparison.model_reference).all())
report = {'status': 'COMPLETE', 'upstream_version': 32, 'upstream_script_version_id': 363222852,
          'test_rows': len(test), 'sample_predictions_match': matched,
          'sample_comparison': comparison.to_dict(orient='records'), 'public_score_verified': False,
          'elapsed_inference_seconds_this_attempt': time.time() - started, 'comp_seconds': namespace['COMP_SEC'],
          'checkpoint_context': context, 'hardware_changed': 'T4 x2 to A100; original ensemble settings retained'}
atomic_json(WORK / 'reproduction_result.json', report)
print('REPRODUCTION RESULT', json.dumps(report), flush=True)
