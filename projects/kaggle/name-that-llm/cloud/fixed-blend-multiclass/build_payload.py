"""Prepare, but never upload or execute, the accepted EXP-027 CPU payload."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT / 'experiments/EXP-027-multiclass-external-validation'
PREVIOUS = ROOT / 'cloud/fixed-blend-external/payload'
SVC = ROOT / 'cloud/svc4-raw-public-r1'
EXTERNAL = ROOT / 'data/public-pools/earthmaker/ntl-own-prompt-generations/v1'
REF = 'zhjianduan/name-that-llm-fixed-blend-multiclass-027'
DATASET = 'zhjianduan/ntl-fixed-blend-multiclass-inputs-027'
GATE_STATUS = 'PASS_FOR_EXTERNAL_SEVENTEEN_CLASS_CHECK'
TRAIN_SHA256 = '40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707'
GUARD_SHA256 = 'eaf2c31758045488eb53ba3f9c7f824c848bfe0a33698fab816d96485f9b6d3b'
VERSIONS = {'numpy': '2.1.3', 'pandas': '2.3.3', 'scipy': '1.16.3', 'sklearn': '1.6.1'}
CONTAINER = 'gcr.io/kaggle-images/python@sha256:17b2fc7f27c25c6c2fd4230ec8e66097ab2b89fc88e1e4d6f338b72e1ee0bc3e'
FROZEN_INPUTS = {
    'style_extra.csv': '1e9125fd49b19084842332ede84ea65bc0174ee1bb28c8d48acc8ddbd5e36c0e',
    'public.csv': '0a1e31a0f8b0ece7ca44a961075252fe275107f3710804971838d714a4e77aaa',
    'stylegram_core.py': 'caa46bd121164aba891c3d78e662540db9294d653db09f53d02833c1b4ab77ef',
    'svc4.py': '80f66aedad27e8017defe31de7f0286c1b54efa5e96ba6e2cb42516a86e73c18',
    'margin_blend.py': '8c4288447de755d9bd160e5646794b2307f56ae00ae11104d062d9697ef5dc83',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def validate_readiness(gate, cfg, frame, training_labels):
    """Pure preflight; no output writes or model prediction is performed."""
    require(gate.get('status') == GATE_STATUS, 'External data not accepted')
    covered = [name for name in training_labels if name != 'gemini-3.7-flash']
    require(len(training_labels) == len(set(training_labels)) == 18 and len(covered) == 17,
            'Changed training label universe')
    require(cfg['evaluation']['labels'] == covered, 'Changed fixed evaluation label order')
    metadata = gate.get('evaluation', {})
    require(metadata.get('labels') == covered, 'Gate label order changed')
    require(set(frame.columns) == {'id', 'prompt_id', 'model', 'text'}, 'Changed evaluation schema')
    require(not frame.isna().any().any(), 'Missing evaluation values')
    for name in ('id', 'prompt_id', 'text'):
        require(frame[name].map(lambda value: isinstance(value, str) and bool(value.strip())).all(),
                'Blank/non-string evaluation ' + name)
    require(frame.id.is_unique, 'Repeated evaluation sample ID')
    require(set(frame.model) == set(covered), 'Changed evaluation class coverage')
    require(0 < len(frame) <= cfg['evaluation']['max_rows'], 'Evaluation row budget exceeded')
    require(metadata.get('rows') == len(frame), 'Gate row count changed')
    require(metadata.get('groups') == int(frame.prompt_id.nunique()), 'Gate group count changed')
    require(metadata.get('class_counts') == frame.model.value_counts().to_dict(), 'Gate class counts changed')
    require(cfg['model']['parameters']['alpha'] == cfg['evaluation']['alpha'] == .10, 'Changed fixed alpha')
    require(cfg['model']['parameters']['retune'] is False and cfg['evaluation']['retune'] is False,
            'Retuning is not authorized')
    require(cfg['evaluation']['output_classes'] == 18 and cfg['evaluation']['sparse_sweep'] is False,
            'Changed evaluation output/routing scope')
    require(cfg['evaluation']['excluded_label'] == 'gemini-3.7-flash', 'Changed predeclared label exclusion')
    require(cfg['evaluation']['bootstrap_draws'] == 2000 and cfg['evaluation']['bootstrap_seed'] == 42,
            'Changed prompt bootstrap')
    training = cfg['training']
    require((training['official_rows'], training['stylegram_extra_rows'], training['svc4_public_rows'],
             training['svm_fits']) == (2699, 10073, 5400, 3), 'Changed training population')
    require(training['stylegram'] == {'C': .5, 'max_iter': 4000, 'dual': True,
            'random_state': 42, 'share': .5, 'dtype': 'float64'}, 'Changed Stylegram recipe')
    require(training['svc4'] == {'C': .5, 'max_iter': 5000, 'dual': 'auto', 'class_weight': 'balanced',
            'random_state': 42, 'public_row_weight': .35, 'dtype': 'float32'}, 'Changed svc4 recipe')
    require(cfg['environment']['required_versions'] == VERSIONS, 'Changed required versions')
    require(cfg['environment']['container'] == CONTAINER, 'Changed frozen container')
    require(cfg['environment']['threads'] == 2 and cfg['environment']['accelerator'] is None
            and cfg['environment']['internet'] is False, 'Changed private CPU runtime')
    for key, expected in (('work_budget_seconds', 3300), ('platform_wall_seconds', 3600),
                          ('rss_limit_bytes', 12 * 1024**3)):
        require(cfg['budget'][key] == expected, 'Changed budget: ' + key)
    return metadata


def main():
    require(not (HERE / 'execution-state.json').exists(),
            'Controller already prepared; inspect its state instead of creating a duplicate')
    manifest = load(EXP / 'manifest.json')
    require(manifest['freeze_status'] == 'DRAFT', 'Experiment manifest is already frozen')
    gate = load(EXP / 'metrics/data-gate.json')
    cfg = load(EXP / 'config.json')
    split = load(ROOT / 'validation/prompt-group-v1/manifest.json')
    frame = pd.read_csv(EXP / 'evaluation.csv', dtype=str, keep_default_na=False)
    metadata = validate_readiness(gate, cfg, frame, split['labels'])
    require(sha(EXP / 'evaluation.csv') == metadata['sha256'], 'Evaluation changed after audit')
    for entry in gate['inputs']:
        require(sha(ROOT / entry['path']) == entry['sha256'], 'Gate input changed: ' + entry['path'])
    require(sha(ROOT / 'train.csv') == split['train_sha256'] == TRAIN_SHA256, 'Official input changed')
    for name, digest in FROZEN_INPUTS.items():
        require(sha(PREVIOUS / name) == digest, 'Frozen EXP-026 training/core input changed: ' + name)
    require(sha(SVC / 'guard.py') == GUARD_SHA256, 'Frozen guard changed')
    download = load(EXP / 'download-manifest.json')
    require(download['dataset']['ref'] == 'earthmaker/ntl-own-prompt-generations'
            and download['dataset']['version'] == 1, 'Changed external source/version')
    readme = next(entry for entry in download['dataset']['files'] if entry['name'] == 'README.md')
    require(sha(EXTERNAL / 'README.md') == readme['sha256'], 'Source notice changed')
    license_source = PREVIOUS / 'CC-BY-4.0.txt'
    require(sha(license_source) == load(PREVIOUS / 'protocol.json')['input_hashes']['CC-BY-4.0.txt'],
            'Generic license text changed')
    require((EXP / 'audit_external.py').is_file(), 'Missing data-gate audit code')
    # All readiness checks above precede payload or controller mutations.
    payload, dataset, kernel = (HERE / name for name in ('payload', 'dataset', 'kernel'))
    for folder in (payload, dataset, kernel):
        require(not folder.exists(), 'Refusing an existing preparation directory: ' + str(folder))
    for folder in (payload, dataset, kernel):
        folder.mkdir(parents=True)
    sources = {name: PREVIOUS / name for name in FROZEN_INPUTS}
    sources.update({'runner.py': HERE / 'runner.py', 'guard.py': SVC / 'guard.py',
        'evaluation.csv': EXP / 'evaluation.csv', 'data_gate.json': EXP / 'metrics/data-gate.json',
        'config.json': EXP / 'config.json', 'own-README.md': EXTERNAL / 'README.md',
        'CC-BY-4.0.txt': license_source})
    for name, source in sources.items():
        (payload / name).write_bytes(source.read_bytes())
        require(sha(payload / name) == sha(source), 'Copy mismatch: ' + name)
    save(payload / 'own-attribution.json', {
        'dataset': 'earthmaker/ntl-own-prompt-generations', 'version': 1,
        'url': 'https://www.kaggle.com/datasets/earthmaker/ntl-own-prompt-generations/versions/1',
        'attribution': 'Name That LLM own-prompt generations (pilot + v22), Kaggle user earthmaker, October 2026',
        'publisher_declared_license': 'CC BY 4.0', 'original_notice': 'own-README.md',
        'original_notice_sha256': readme['sha256'],
        'scope': 'Publisher declaration retained; model outputs may also be subject to provider terms. No independent blanket rights clearance is asserted.',
        'identity_limit': 'Labels identify requested models as reported by publisher; raw provider logs/serving weights were not independently authenticated.',
        'excluded_label': 'gemini-3.7-flash',
    })
    (payload / 'attribution.md').write_text('''# Private multiclass external validation inputs

This package is private research material for EXP-027; no competition submission or public redistribution is performed.

- Stylegram and svc4 public training files and all three core modules are byte-identical to EXP-026, which inherits EXP-010/018/024. Original training source attributions remain in the project.
- External text is from earthmaker/ntl-own-prompt-generations V1. Suggested attribution: "Name That LLM own-prompt generations (pilot + v22), Kaggle user earthmaker, October 2026". The original README and declared CC BY4.0 notice are retained. Model outputs may also be subject to provider terms; this package does not assert blanket clearance or relicense the text.
- The fixed gate excludes the unresolved gemini-3.7-flash serving label and any detected task/text ancestry groups before scoring. Original response text is not rewritten. Requested model labels are publisher claims, not independent authentication of serving weights.
- Official training text is mounted separately from the competition. Evaluation has 17 covered labels with unequal supports and incomplete prompt panels; all predictions remain unrestricted across 18 classes.

This is a scoped external check of the fixed alpha=.10 lightweight blend, not a full ATLAS20 or leaderboard result.
''', encoding='utf-8')
    now = datetime.now(timezone.utc).isoformat()
    hashes = {path.name: sha(path) for path in payload.iterdir() if path.is_file()}
    protocol = {'schema_version': 1, 'experiment_id': 'EXP-027', 'created_at': now,
        'scope': 'fixed alpha0.10 external seventeen-class imbalanced check with 18-way predictions; not full ATLAS20 or leaderboard',
        'labels': split['labels'], 'train_sha256': TRAIN_SHA256, 'official_rows': 2699, 'style_extra_rows': 10073, 'public_rows': 5400,
        'evaluation_accepted': True, 'evaluation_gate_status': GATE_STATUS,
        'evaluation_rows': metadata['rows'], 'evaluation_groups': metadata['groups'],
        'evaluation_labels': metadata['labels'], 'evaluation_class_counts': metadata['class_counts'],
        'evaluation_sha256': metadata['sha256'], 'data_gate_sha256': sha(EXP / 'metrics/data-gate.json'),
        'alpha': .10, 'style_share': .5, 'public_row_weight': .35, 'required_versions': VERSIONS,
        'bootstrap_draws': 2000, 'bootstrap_seed': 42,
        'bootstrap_missing_class_policy': 'null interval if any draw lacks any of the 17 covered labels; no dropped/replaced draws',
        'input_hashes': hashes, 'work_budget_seconds': 3300, 'platform_wall_seconds': 3600,
        'rss_limit_bytes': 12 * 1024**3, 'threads': 2, 'gpu': False, 'internet': False,
        'limits': ['Seventeen covered true labels; excluded Gemini3.7 remains a selectable output among18 classes.',
                   'Prompt groups and class supports are unequal; covered-label macro recall is distinct from accuracy.',
                   'Independence is bounded by the accepted known ancestry and exact/normalized checks; semantic overlap is not exhaustively ruled out.',
                   'Publisher-requested model identities are not independent authentication of serving weights.',
                   'No alpha tuning, sparse sweep, formal submission, full-ensemble execution or baseline change.']}
    save(payload / 'protocol.json', protocol)
    with zipfile.ZipFile(dataset / 'external_payload.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(payload.iterdir()):
            archive.write(path, path.name)
    save(dataset / 'dataset-metadata.json', {'id': DATASET, 'title': 'NTL Fixed Blend Multiclass Inputs 027',
        'licenses': [{'name': 'other'}], 'description': 'Private fixed alpha0.10 seventeen-label external check. Original Earthmaker notice and license scope retained; unchanged training inputs; no competition submission.'})
    code = """import hashlib,json,pathlib,runpy,sys,os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
matches=[p for p in pathlib.Path('/kaggle/input').rglob('protocol.json') if hashlib.sha256(p.read_bytes()).hexdigest()==PROTOCOL_HASH]
assert len(matches)==1,'Expected one pinned protocol'
target=matches[0].parent
protocol=json.loads(matches[0].read_text())
for name,digest in protocol['input_hashes'].items():assert hashlib.sha256((target/name).read_bytes()).hexdigest()==digest,name
sys.path.insert(0,str(target))
runpy.run_path(str(target/'guard.py'),run_name='__main__')
""".replace('PROTOCOL_HASH', repr(sha(payload / 'protocol.json')))
    (kernel / 'bootstrap.py').write_text(code, encoding='utf-8')
    save(kernel / 'kernel-metadata.json', {'id': REF, 'title': 'Name That LLM Fixed Blend Multiclass 027',
        'code_file': 'bootstrap.py', 'language': 'python', 'kernel_type': 'script', 'is_private': True,
        'enable_gpu': False, 'enable_tpu': False, 'enable_internet': False, 'dataset_sources': [DATASET + '/1'],
        'competition_sources': ['name-that-llm'], 'kernel_sources': [], 'model_sources': [],
        'docker_image_pinning_type': 'original', 'docker_image': cfg['environment']['container']})
    manifest.update(freeze_status='FROZEN', frozen_at=now,
        note='Accepted imbalanced17-label row set frozen before scoring; EXP-026 training bytes and alpha=.10 unchanged.',
        config={'path': (EXP / 'config.json').relative_to(ROOT).as_posix(), 'sha256': sha(EXP / 'config.json')},
        inputs=[{'path': (payload / name).relative_to(ROOT).as_posix(), 'sha256': digest} for name, digest in sorted(hashes.items())],
        code=[{'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path)} for path in
              [HERE / 'runner.py', Path(__file__), EXP / 'audit_external.py']],
        protocol={'path': (payload / 'protocol.json').relative_to(ROOT).as_posix(), 'sha256': sha(payload / 'protocol.json')})
    save(EXP / 'manifest.json', manifest)
    save(HERE / 'preflight.json', {'status': 'FROZEN', 'observed_at': now,
        'protocol_sha256': sha(payload / 'protocol.json'), 'payload_sha256': sha(dataset / 'external_payload.zip'),
        'payload_bytes': (dataset / 'external_payload.zip').stat().st_size,
        'evaluation_rows': len(frame), 'evaluation_groups': metadata['groups'],
        'evaluation_class_counts': metadata['class_counts'], 'training_started': False})
    save(HERE / 'execution-state.json', {'status': 'PREPARED', 'observed_at': now, 'experiment': 'EXP-027',
        'notebook_ref': REF, 'version': None, 'dataset_ref': DATASET, 'dataset_version': 1, 'gpu': False, 'internet': False,
        'work_budget_seconds': 3300, 'platform_wall_seconds': 3600, 'rss_limit_gib': 12,
        'duplicate_run_allowed': False, 'automatic_retry_allowed': False,
        'protocol_sha256': sha(payload / 'protocol.json'), 'result_status': 'NOT_RUN'})
    print(json.dumps({'status': 'FROZEN', 'rows': len(frame), 'groups': metadata['groups'],
                      'payload_bytes': (dataset / 'external_payload.zip').stat().st_size}))


if __name__ == '__main__':
    main()
