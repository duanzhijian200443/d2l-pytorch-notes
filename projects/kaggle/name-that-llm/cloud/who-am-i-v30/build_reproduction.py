"""Preserve the V30 model pipeline; replace only the Llama asset and run plumbing."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['preflight', 'pincheck', 'full'], default='preflight')
args = parser.parse_args()
upstream = nbformat.read(ROOT / 'upstream-v30.ipynb', as_version=4)
metadata = json.loads((ROOT / 'upstream-metadata.json').read_text(encoding='utf-8'))
inputs = json.loads((ROOT / 'input-lock.json').read_text(encoding='utf-8'))

llama_replacement = '''# Use the verified private copy of the exact Llama 3.1 8B base checkpoint.
_llama_candidates = glob.glob('/kaggle/input/**/llama31-8b-base-verified/config.json', recursive=True)
assert len(_llama_candidates) == 1, ('Expected exactly one verified Llama dataset', _llama_candidates)
BL = os.path.dirname(_llama_candidates[0])
_llama_manifest = json.load(open(os.path.join(BL, 'transfer_manifest.json')))
assert _llama_manifest['status'] == 'VERIFIED'
assert _llama_manifest['source']['revision'] == '1f47e50cdbe801ad8a5174156ec3a0655108fb9f'
for _file in _llama_manifest['files']:
    assert os.path.getsize(os.path.join(BL, _file['file'])) == _file['bytes'], _file['file']
'''
paths = upstream.cells[1].source
assert paths.count("BL = mdir('/llama-3.1/', '/8b/')") == 1
paths = paths.replace("BL = mdir('/llama-3.1/', '/8b/')", llama_replacement)

environment = '''import json, os, platform, sys, importlib.metadata as _md
from pathlib import Path
import torch
packages = {}
for name in ['torch', 'transformers', 'peft', 'numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'safetensors', 'pyarrow']:
    packages[name] = _md.version(name)
environment = {'python': platform.python_version(), 'packages': packages, 'gpu_count': torch.cuda.device_count(), 'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], 'mode': MODE}
print(json.dumps(environment, indent=2))
Path('/kaggle/working').mkdir(exist_ok=True)
Path('/kaggle/working/reproduction_environment.json').write_text(json.dumps(environment, indent=2))
if MODE == 'full':
    assert torch.cuda.device_count() == 2, 'V30 reproduction expects the original T4 x2 environment'
    assert all('T4' in name for name in environment['gpu_names']), environment['gpu_names']
'''
preflight_checks = '''# All expected base checkpoints and adapter families must be present before inference.
import hashlib
all_configs = []
for path in cfgs:
    try:
        config = json.load(open(path))
        all_configs.append((path, config))
    except (OSError, ValueError):
        continue
for model_type, hidden in [('glm4', 4096), ('qwen3', 4096), ('qwen3', 2560), ('qwen2', 3584)]:
    assert any(c.get('model_type') == model_type and c.get('hidden_size') == hidden for _, c in all_configs), (model_type, hidden)
for directory in ['deberta-v3-large', 'electra-large-discriminator', 'desklib-detector']:
    assert any(path.endswith('/' + directory + '/config.json') for path, _ in all_configs), directory
# V30's saved log explicitly skips lora24_public because it was not attached.
assert not glob.glob('/kaggle/input/**/lora24/final/COMPLETE.json', recursive=True), 'Unexpected extra lora24_public member'
for pattern in ['**/name-that-llm-train-lora24-s7/lora_out/job_final/COMPLETE.json', '**/ntl-qwen3-8b-lora-claude-head/**/final/COMPLETE.json', '**/ntl-train-lora-qwen25-instyle/lora_out/job_final/COMPLETE.json', '**/tokattn_debl23_final.pt']:
    assert glob.glob('/kaggle/input/' + pattern, recursive=True), pattern
assert sum(c.get('model_type') == 'ultrabert' for _, c in all_configs) >= 5, 'UltraBERT fold checkpoints'
base_paths = {'qwen3_8b_base': B8, 'qwen3_14b_base': B14, 'qwen25_7b': B25, 'qwen3_8b': I8, 'qwen3_14b': I14, 'qwen25_14b': B25_14, 'llama31_8b': BL}
adapter_paths = A8 + A14 + A25 + AL + A8S7 + AQ8V5 + AQ14V5 + ALL8V5 + [N8, N14, N14B, Q25_14]
for path in base_paths.values():
    config = json.load(open(os.path.join(path, 'config.json')))
    index_path = os.path.join(path, 'model.safetensors.index.json')
    if os.path.exists(index_path):
        shards = sorted(set(json.load(open(index_path))['weight_map'].values()))
        assert all(os.path.isfile(os.path.join(path, name)) for name in shards), path
    else:
        assert glob.glob(os.path.join(path, '*.safetensors')) or glob.glob(os.path.join(path, '*.bin')), path
for path in adapter_paths:
    assert glob.glob(os.path.join(path, '**', 'adapter_config.json'), recursive=True), path
    assert glob.glob(os.path.join(path, '**', 'adapter_model.safetensors'), recursive=True), path
assert os.path.isfile(os.path.join(ATLAS, 'atlas_bundle.json'))
import pandas as pd
train = pd.read_csv(os.path.join(DATA, 'train.csv'))
test = pd.read_csv(os.path.join(DATA, 'test.csv'))
assert train.model.nunique() == 18 and train.prompt_id.nunique() == 150
assert train.id.is_unique and test.id.is_unique
csv_probes = {}
for slug in ['ntl-generations', 'ntl-free-proxy-generations']:
    candidates = ([entry['path'] for entry in checked if entry['input'].split('/')[0] == slug]
                  if MODE == 'full' else sorted(glob.glob('/kaggle/input/**/' + slug + '/**/*.csv', recursive=True)))
    assert candidates, slug
    csv_probes[slug] = [{'name': os.path.basename(p), 'bytes': os.path.getsize(p)} for p in candidates]
manifest = {'status': 'PASS', 'mode': MODE, 'base_paths': base_paths, 'adapter_directory_count': len(adapter_paths), 'train_rows': len(train), 'test_rows': len(test), 'external_csv_probes': csv_probes, 'llama_source_revision': _llama_manifest['source']['revision']}
Path('/kaggle/working/input_preflight.json').write_text(json.dumps(manifest, indent=2))
print('INPUT PREFLIGHT PASS', json.dumps(manifest, indent=2))
'''
title = '# Who am I V30 — private execution reproduction\n\n'
intro = (
    title + 'Upstream: [Roman Tamrazov, V30](https://www.kaggle.com/code/romantamrazov/name-that-llm-who-am-i?scriptVersionId=362877806). '
    'The three inference modules and all ensemble settings are preserved. '
    'The public Llama cache is replaced with our verified private Llama 3.1 8B base dataset. '
    'Input dataset versions are pinned, including ntl-generations V2 and free-proxy generations V7.\n\n'
    '**Mode: ' + args.mode + '.** ' + ('This CPU run checks inputs only; it does not run model inference or produce a leaderboard score.' if args.mode != 'full' else 'This run executes the original GPU pipeline. The visible test file has three examples; a successful saved run is not proof of reproducing the public leaderboard score. No --sim accuracy is claimed.')
)
cells = [nbformat.v4.new_markdown_cell(intro), nbformat.v4.new_markdown_cell(upstream.cells[0].source), nbformat.v4.new_code_cell('MODE = ' + repr(args.mode) + '\n' + environment)]
if args.mode == 'preflight':
    original_copy = "import shutil; shutil.copytree(DEB, '/kaggle/working/deberta_v3_large', dirs_exist_ok=True); DEB = '/kaggle/working/deberta_v3_large'"
    assert original_copy in paths
    paths = paths.replace(original_copy, '# Preflight only: inspect the source checkpoint without copying its weights.')
cells.append(nbformat.v4.new_code_cell(paths))
cells.append(nbformat.v4.new_code_cell(preflight_checks))
if args.mode == 'full':
    for i in (2, 3, 4):
        cells.append(nbformat.v4.new_code_cell(upstream.cells[i].source))
    cells.append(nbformat.v4.new_code_cell('''# Keep the original process boundary and environment; propagate failures reliably.
import time as _t
_c0 = _t.time()
os.environ['NTL_DROP_Q35'] = '1'
with open('/kaggle/working/comp_stack.log', 'w', buffering=1) as log_file:
    with subprocess.Popen([sys.executable, '-u', 'comp_stack.py'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        for line in process.stdout:
            log_file.write(line)
            if 'Warning' not in line and 'warn' not in line:
                print(line, end='', flush=True)
        returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, 'comp_stack.py')
COMP_SEC = _t.time() - _c0
print('comp stack wall time', round(COMP_SEC))
'''))
    cells.append(nbformat.v4.new_code_cell(upstream.cells[6].source))
    cells.append(nbformat.v4.new_code_cell('''# Validate the artifact without presenting the visible sample as a leaderboard evaluation.
test = pd.read_csv(os.path.join(DATA, 'test.csv'))
assert submission['id'].tolist() == test['id'].tolist()
assert submission['model'].isin(train['model'].unique()).all()
report = {'status': 'COMPLETE', 'upstream_version': 30, 'upstream_script_version_id': 362877806, 'test_rows': len(test), 'label_count': int(train['model'].nunique()), 'comp_seconds': COMP_SEC, 'submission_file': 'submission.parquet', 'public_score_verified': False}
Path('/kaggle/working/reproduction_result.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
'''))

pincheck = 'PINNED_CSV_SHA256 = ' + repr(inputs.get('pinned_csv_sha256', {})) + '''
assert len(PINNED_CSV_SHA256) == 3, 'Historical CSV hashes must be configured'
import hashlib
input_inventory = list(Path('/kaggle/input').rglob('*'))
checked = []
for relative, expected in PINNED_CSV_SHA256.items():
    candidates = [p for p in input_inventory if p.name == Path(relative).name and p.is_file()]
    matches = [p for p in candidates if hashlib.sha256(p.read_bytes()).hexdigest() == expected]
    assert len(matches) == 1, ('Expected one historical CSV with matching hash', relative, [str(p) for p in candidates])
    path = matches[0]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == expected, ('Historical input changed', relative)
    checked.append({'input': relative, 'path': str(path), 'sha256': actual, 'bytes': path.stat().st_size})
Path('/kaggle/working/pinned_inputs_verified.json').write_text(json.dumps({'status': 'PASS', 'files': checked}, indent=2))
print('HISTORICAL INPUT HASHES PASS', json.dumps(checked, indent=2))
'''
if args.mode == 'pincheck':
    component_checks = 'import glob\ncfgs = glob.glob("/kaggle/input/**/config.json", recursive=True)\n' + preflight_checks.split('base_paths = ', 1)[0]
    cells = cells[:3] + [nbformat.v4.new_code_cell(pincheck + '\n' + component_checks + "\nprint('COMPLEMENTARY COMPONENT INPUTS PASS')")]
elif args.mode == 'full':
    cells.insert(3, nbformat.v4.new_code_cell(pincheck))

for cell in cells:
    if cell.cell_type != 'code':
        continue
    text = cell.source
    if text.startswith('%%writefile '):
        text = text.split('\n', 1)[1]
    ast.parse(text)
notebook = nbformat.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}})
nbformat.validate(notebook)
nbformat.write(notebook, ROOT / 'reproduction.ipynb')
kernel_metadata = {
    'id': 'zhjianduan/name-that-llm-who-am-i-v30-repro',
    'title': 'Name That LLM Who Am I V30 Repro',
    'code_file': 'reproduction.ipynb', 'language': 'python', 'kernel_type': 'notebook',
    'is_private': True, 'enable_gpu': args.mode == 'full', 'enable_internet': False,
    'competition_sources': ['name-that-llm'],
    'dataset_sources': [(item['snapshot_ref'] + '/' + str(item['snapshot_version'])) if item.get('snapshot_ref') else (item['ref'] + '/' + str(item['pinned_version'])) for item in inputs['datasets']],
    'kernel_sources': [item['ref'] + '/' + str(item['version']) for item in inputs['kernels']],
    'model_sources': inputs['models'],
    'docker_image': metadata['docker_image'], 'docker_image_pinning_type': 'original',
}
if args.mode == 'full':
    kernel_metadata['machine_shape'] = 'NvidiaTeslaT4'
(ROOT / 'kernel-metadata.json').write_text(json.dumps(kernel_metadata, indent=2), encoding='utf-8')
source_hashes = {name: hashlib.sha256(upstream.cells[i].source.split('\n', 1)[1].encode()).hexdigest() for i, name in [(2, 'gauss.py'), (3, 'infer.py'), (4, 'comp_stack.py')]}
(ROOT / 'upstream-module-hashes.json').write_text(json.dumps(source_hashes, indent=2), encoding='utf-8')
print(json.dumps({'mode': args.mode, 'cells': len(cells), 'dataset_sources': len(kernel_metadata['dataset_sources']), 'kernel_sources': len(kernel_metadata['kernel_sources']), 'model_sources': len(kernel_metadata['model_sources'])}))
