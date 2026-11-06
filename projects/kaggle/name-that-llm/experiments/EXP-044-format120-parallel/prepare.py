"""Freeze EXP044 and package a private CPU notebook; does not launch."""
from pathlib import Path
from datetime import datetime, timezone
import base64
import hashlib
import io
import json
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CLOUD = ROOT / 'cloud/format120-044'
BASE = ROOT / 'cloud/luna-terra120-042/payload'
REF = 'zhjianduan/name-that-llm-format120-044'
IMAGE = 'gcr.io/kaggle-images/python@sha256:17b2fc7f27c25c6c2fd4230ec8e66097ab2b89fc88e1e4d6f338b72e1ee0bc3e'
PAYLOAD_NAMES = {'train.csv', 'folds.csv', 'extra.csv', 'stylegram_core.py', 'legacy_runner.py',
                 'runner.py', 'guard.py', 'test_runner.py', 'raw120.csv', 'clean120.csv',
                 'data-gate.json', 'config.json', 'protocol.json'}


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def entry(p):
    return {'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p), 'bytes': p.stat().st_size}


def main():
    state_path = CLOUD / 'execution-state.json'
    previous = read(state_path) if state_path.exists() else {}
    assert not previous or (previous['status'] == 'PREPARED' and previous['cloud_run_requests'] == 0), 'Already dispatched; cannot rebuild'
    gate = read(HERE / 'metrics/data-review.json')
    assert gate['status'].startswith('PASS'), 'Data review not accepted'
    for name, item in gate['inputs'].items():
        assert sha(ROOT / name) == item['sha256'], 'Data review stale: ' + name
    payload = CLOUD / 'payload'
    payload.mkdir(parents=True, exist_ok=True)
    assert all(p.is_dir() and p.name == '__pycache__' or p.is_file() and p.name in PAYLOAD_NAMES
               for p in payload.iterdir()), 'Unexpected payload entry; refusing to package'
    for name in ('raw120.csv', 'clean120.csv'):
        item = gate['outputs'][name]
        assert item['path'] == (payload / name).relative_to(ROOT).as_posix()
        assert sha(payload / name) == item['sha256'] and (payload / name).stat().st_size == item['bytes'], 'Reviewed CSV changed: ' + name
    sources = {n: BASE / n for n in ('train.csv', 'folds.csv', 'extra.csv', 'stylegram_core.py')}
    sources.update({n: HERE / n for n in ('runner.py', 'guard.py', 'test_runner.py')})
    sources['legacy_runner.py'] = ROOT / 'experiments/EXP-042-luna-terra120-ablation/runner.py'
    for name, source in sources.items():
        (payload / name).write_bytes(source.read_bytes())
    assert sha(payload / 'legacy_runner.py') == 'de60b60bfd138b834d8723bbd832fd0522453f549c4050256197f6babe4ed45a'
    for name in ('raw120.csv', 'clean120.csv'):
        assert (payload / name).is_file(), name
    save(payload / 'data-gate.json', {
        'status': 'PASS', 'local_review_sha256': sha(HERE / 'metrics/data-review.json'),
        'candidate_rows_per_arm': 120, 'candidate_groups_per_arm': 60,
        'source': 'EXP023 original outputs and EXP043 derived formatting view',
        'scope': 'Known/exact isolation only; semantic independence and backend snapshot are not certified.',
        'holdout_text_uploaded': False, 'formal_pool_accepted': False,
    })
    old = read(BASE / 'config.json')
    assert gate['holdout_prompt_group_hashes'] == old['holdout_prompt_group_hashes'], 'Holdout group inventory differs'
    for name in ('train.csv', 'folds.csv', 'extra.csv', 'stylegram_core.py'):
        assert sha(payload / name) == old['input_hashes'][name], name
    now = datetime.now(timezone.utc)
    runid = previous.get('run_id', 'exp044-' + now.strftime('%Y%m%dT%H%M%SZ'))
    cfg = {k: old[k] for k in ('schema_version', 'labels', 'new_labels', 'validation_scheme', 'C', 'dual', 'max_iter',
                             'seed', 'share', 'sparse_per_prompt', 'sparse_repeats', 'threads', 'holdout_prompt_group_hashes')}
    cfg.update(experiment_id='EXP-044', run_id=runid, baseline_id='EXP-042', formal_ensemble_baseline_id='EXP-004',
               arms=['raw120', 'clean120'], parallel_workers=2,
               weight_policy='Both arms add the identical 120 old rows with frozen EXP042 public-per-class weights. No reweighting; only added text formatting differs.',
               input_hashes={name: sha(payload / name) for name in sorted(PAYLOAD_NAMES - {'config.json', 'protocol.json'})},
               identity={'added_source': 'EXP023 old Codex original120 / EXP043 formatting view', 'model_observation': 'requested_only',
                         'snapshot': 'unknown', 'original_rows': 120, 'original_prompt_groups': 60, 'preserve_raw': True,
                         'formal_pool_acceptance': False, 'no_holdout_or_paired_diagnostic_text_uploaded': True},
               environment=old['environment'],
               budget={'hard_seconds': 6600, 'platform_wall_seconds': 7200, 'rss_limit_bytes': 20 * 1024**3,
                       'new_cloud_runs': 1, 'gpu_seconds': 0, 'api_usd': 0, 'colab_cu': 0},
               evaluation={'primary': '18-class pooled official OOF balanced accuracy', 'comparison': 'clean120 minus raw120',
                           'bootstrap_draws': 2000, 'bootstrap_seed': 42, 'bootstrap_unit': 'original official prompt_id',
                           'sparse_secondary': 'm2/3/6 x repeats0..4, regroup selected text before centering',
                           'holdout_evaluated': False, 'retune': False},
               resume={'explicit_only': True, 'completed_job_hashes_required': True, 'automatic_cloud_retry': False,
                       'cloud_outputs_must_be_recovered_before_new_session': True},
               limits=['Official folds are previously inspected development validation, not an independent holdout.',
                       'Old raw120 differs from EXP042 aligned new120; both current arms require new training.',
                       'Formatting changes candidate vocabulary and IDF as intended; other data and parameters remain fixed.',
                       'Only Stylegram OOF, not ATLAS20 or a leaderboard result.',
                       'A hard cloud failure may prevent output publication; saved local jobs do not guarantee cloud recovery.'])
    save(HERE / 'config.json', cfg)
    save(payload / 'config.json', cfg)
    protocol = {'experiment_id': 'EXP-044', 'run_id': runid, 'work_budget_seconds': 6600,
                'rss_limit_bytes': 20 * 1024**3, 'input_hashes': {name: sha(payload / name) for name in sorted(PAYLOAD_NAMES - {'protocol.json'})}}
    save(payload / 'protocol.json', protocol)
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(PAYLOAD_NAMES - {'train.csv', 'folds.csv', 'extra.csv'}):
            archive.writestr(name, (payload / name).read_bytes())
    encoded = base64.b64encode(memory.getvalue()).decode('ascii')
    code = """import base64, hashlib, io, json, os, pathlib, subprocess, sys, zipfile
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='2'
target=pathlib.Path('/kaggle/working/exp044-payload')
assert not target.exists(), 'Repeated execution requires explicit checkpoint recovery, not Run All again'
target.mkdir()
raw=base64.b64decode(ENCODED)
assert hashlib.sha256(raw).hexdigest()==BUNDLE_SHA
with zipfile.ZipFile(io.BytesIO(raw)) as archive:
 for name in archive.namelist():
  assert pathlib.PurePosixPath(name).name==name and ':' not in name
  (target/name).write_bytes(archive.read(name))
protocol=json.loads((target/'protocol.json').read_text())
for name in ('train.csv','extra.csv','folds.csv'):
 matches=[p for p in pathlib.Path('/kaggle/input').rglob(name) if hashlib.sha256(p.read_bytes()).hexdigest()==protocol['input_hashes'][name]]
 assert len(matches)==1, ('Expected exactly one frozen input',name,len(matches))
 (target/name).write_bytes(matches[0].read_bytes())
for name,digest in protocol['input_hashes'].items(): assert hashlib.sha256((target/name).read_bytes()).hexdigest()==digest,name
root=pathlib.Path('/kaggle/working/results')
print('EXP044 frozen. CPU only; 2 workers; full 5 folds; no network. Synthetic parity gate runs before training.',flush=True)
try:
 subprocess.run([sys.executable,'-u',str(target/'guard.py'),'--payload',str(target),'--output',str(root)],check=True)
finally:
 if root.exists():
  with zipfile.ZipFile('/kaggle/working/exp044-results.zip','w',zipfile.ZIP_DEFLATED) as archive:
   for p in sorted(root.rglob('*')):
    if p.is_file(): archive.write(p,p.relative_to(root).as_posix())
""".replace('ENCODED', repr(encoded)).replace('BUNDLE_SHA', repr(hashlib.sha256(memory.getvalue()).hexdigest()))
    nb = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
          'cells': [{'cell_type': 'markdown', 'metadata': {}, 'source': ['# EXP044: raw120 vs clean120\nTwo CPU workers; fixed five-fold Stylegram; no competition submission.\n'], 'id': 'exp044-intro'},
                    {'cell_type': 'code', 'metadata': {}, 'source': code.splitlines(True), 'outputs': [], 'execution_count': None, 'id': 'exp044-run'}]}
    save(CLOUD / 'kernel/experiment.ipynb', nb)
    save(CLOUD / 'kernel/kernel-metadata.json', {'id': REF, 'title': 'Name That LLM Format120 044',
         'code_file': 'experiment.ipynb', 'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
         'enable_gpu': False, 'enable_tpu': False, 'enable_internet': False,
         'dataset_sources': ['zhjianduan/ntl-stylegram-cv-inputs-v1/1'], 'competition_sources': ['name-that-llm'],
         'kernel_sources': [], 'model_sources': [], 'docker_image_pinning_type': 'original', 'docker_image': IMAGE})
    save(HERE / 'manifest.json', {'schema_version': 1, 'experiment_id': 'EXP-044', 'freeze_status': 'FROZEN_BEFORE_TRAINING',
         'observed_at': now.isoformat(), 'run_id': runid, 'config': entry(HERE / 'config.json'),
         'inputs': [entry(p) for p in (payload / 'raw120.csv', payload / 'clean120.csv', payload / 'train.csv', payload / 'folds.csv', payload / 'extra.csv', HERE / 'metrics/data-review.json')],
         'code': [entry(p) for p in (HERE / 'prepare.py', HERE / 'prepare_inputs.py', HERE / 'launch.py', HERE / 'validate_local.py', HERE / 'runner.py', HERE / 'test_runner.py', HERE / 'guard.py',
                                    payload / 'legacy_runner.py', payload / 'stylegram_core.py', CLOUD / 'kernel/experiment.ipynb', CLOUD / 'kernel/kernel-metadata.json')],
         'protocol': entry(payload / 'protocol.json'), 'new_model_calls': 0, 'formal_pool_ingestion': False})
    save(state_path, {'experiment_id': 'EXP-044', 'status': 'PREPARED', 'observed_at': now.isoformat(), 'run_id': runid,
         'notebook_ref': REF, 'version': None, 'cloud_run_requests': 0, 'gpu': False, 'internet': False,
         'platform_wall_seconds': 7200, 'work_budget_seconds': 6600, 'parallel_workers': 2,
         'automatic_retry_allowed': False, 'config_sha256': sha(payload / 'config.json'), 'protocol_sha256': sha(payload / 'protocol.json')})
    print(json.dumps({'status': 'FROZEN', 'run_id': runid, 'embedded_bytes': len(memory.getvalue()), 'training_started': False}))


if __name__ == '__main__':
    main()
