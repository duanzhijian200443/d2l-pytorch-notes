"""EXP044: bounded two-process fold jobs with sealed, explicit checkpoints.

Formal execution is supervised by guard.py. Each completed fold/arm is immutable;
resume only reuses validated receipts and creates new directories for unfinished
jobs. No fitted models/TF-IDF caches, network, GPU, retries, or data rewriting.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '2'

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
import psutil
import scipy
from scipy.sparse import vstack
import sklearn
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ARMS = ('raw120', 'clean120')
HARD_SECONDS = 6600
RSS_LIMIT = 20 * 1024**3
LEGACY_SHA256 = 'de60b60bfd138b834d8723bbd832fd0522453f549c4050256197f6babe4ed45a'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def runtime():
    return {'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__,
            'sklearn': sklearn.__version__, 'python': platform.python_version()}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def modules(payload, fallback=False):
    legacy_path, core_path = payload / 'legacy_runner.py', payload / 'stylegram_core.py'
    if fallback:
        if not legacy_path.exists():
            legacy_path = HERE.parent / 'EXP-042-luna-terra120-ablation/runner.py'
        if not core_path.exists():
            core_path = HERE.parent / 'EXP-010-012-stylegram-share/payload/stylegram_core.py'
    need(sha(legacy_path) == LEGACY_SHA256, 'Frozen EXP042 legacy runner changed')
    return load_module(legacy_path, 'exp044_legacy'), load_module(core_path, 'exp044_core'), legacy_path, core_path


def process_record(pid=None):
    process = psutil.Process(pid or os.getpid())
    return {'pid': process.pid, 'create_time': process.create_time(), 'host': platform.node(), 'boot_time': psutil.boot_time()}


def process_live(record):
    """Fail closed when an old process belongs to an unobservable host/boot."""
    need(record['host'] == platform.node() and abs(record['boot_time'] - psutil.boot_time()) < 2,
         'Cannot confirm process inactivity across host/boot without a terminal guard receipt')
    try:
        process = psutil.Process(record['pid'])
        return abs(process.create_time() - record['create_time']) < .01 and process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def stop_tree(pid):
    """Kill all descendants even if they opened their own process sessions."""
    try:
        parent = psutil.Process(pid)
        parent.suspend()
        processes = parent.children(recursive=True) + [parent]
    except psutil.NoSuchProcess:
        return
    for process in reversed(processes):
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=15)
    need(not [p for p in alive if p.status() != psutil.STATUS_ZOMBIE], 'Process tree did not stop')


def validate_pair(raw, clean, legacy, train, extra, holdout):
    for frame in (raw, clean):
        legacy.validate_new_rows(frame, train, extra, holdout)
    need(raw.columns.tolist() == clean.columns.tolist(), 'Raw/clean columns differ')
    need('raw_text_sha256' in raw, 'Missing raw text mapping anchor')
    for column in raw.columns:
        if column not in ('text', 'text_sha256'):
            need(raw[column].tolist() == clean[column].tolist(), f'Raw/clean metadata mismatch: {column}')
    need(raw.text_sha256.tolist() == raw.raw_text_sha256.tolist(), 'Raw source hash mapping mismatch')
    need(raw.text.tolist() != clean.text.tolist(), 'Raw/clean arms are identical')


def preflight(payload, strict=True):
    config = read(payload / 'config.json')
    legacy, core, legacy_path, core_path = modules(payload)
    legacy.validate_config(config)
    need(config.get('experiment_id') == 'EXP-044' and config.get('arms') == list(ARMS), 'Experiment/arms changed')
    need(config.get('share') == .5 and config.get('parallel_workers') == 2 and config.get('threads') == 2, 'Fixed execution/weight parameters')
    need(config['budget']['hard_seconds'] == HARD_SECONDS and config['budget']['rss_limit_bytes'] == RSS_LIMIT
         and config['budget']['platform_wall_seconds'] == 7200, 'Fixed cumulative budgets')
    required = {*legacy.FROZEN_HASHES, 'raw120.csv', 'clean120.csv', 'runner.py', 'guard.py', 'legacy_runner.py'}
    need(required.issubset(config['input_hashes']), 'Missing required input/code hashes')
    for name, digest in config['input_hashes'].items():
        need(Path(name).name == name and '\\' not in name and ':' not in name, 'Unsafe input name')
        need(sha(payload / name) == digest, 'Input/code hash mismatch: ' + name)
    need(sha(__file__) == config['input_hashes']['runner.py'], 'Executing runner is not frozen payload runner')
    for name, digest in legacy.FROZEN_HASHES.items():
        need(config['input_hashes'][name] == digest, 'Frozen Stylegram input/core changed: ' + name)
    need(config['environment']['required_versions'] == legacy.VERSIONS, 'Required versions changed')
    if strict:
        need(all(runtime()[k] == value for k, value in legacy.VERSIONS.items()), 'Strict runtime versions mismatch')
        need(not Path('/dev/nvidia0').exists(), 'GPU device detected')
    frames = [pd.read_csv(payload / name, dtype=str, keep_default_na=False)
              for name in ('train.csv', 'folds.csv', 'extra.csv', 'raw120.csv', 'clean120.csv')]
    train, folds, extra, raw, clean = frames
    need(len(train) == len(folds) == 2699 and train.id.is_unique, 'Official count/IDs')
    for col in ('id', 'prompt_id', 'model'):
        need(train[col].tolist() == folds[col].tolist(), 'Official fold alignment: ' + col)
    need(train.text.map(legacy.text_hash).tolist() == folds.text_sha256.tolist(), 'Official text hashes')
    folds['fold'] = pd.to_numeric(folds.fold, errors='raise')
    need(set(folds.fold) == set(range(5)) and folds.groupby('prompt_id').fold.nunique().eq(1).all(), 'Prompt fold leakage')
    need(set(train.model) == set(extra.model) == set(legacy.LABELS), 'Baseline labels')
    need(len(extra) == 10073 and extra.row_id.is_unique and extra.prompt_id.nunique() == 694, 'Public counts')
    need(extra.groupby('prompt_id').canonical_prompt.nunique().eq(1).all()
         and not extra.canonical_prompt.str.startswith('official:').any(), 'Public prompt groups')
    need(set(train.text.map(legacy.normalize)).isdisjoint(extra.text.map(legacy.normalize)), 'Baseline text overlap')
    validate_pair(raw, clean, legacy, train, extra, config['holdout_prompt_group_hashes'])
    identity = {'schema_version': 1, 'run_id': config['run_id'], 'config_sha256': sha(payload / 'config.json'),
                'runner_sha256': sha(__file__), 'guard_sha256': sha(payload / 'guard.py'),
                'legacy_runner_sha256': sha(legacy_path), 'core_sha256': sha(core_path),
                'input_hashes': config['input_hashes'], 'environment': runtime(), 'labels': config['labels'],
                'arms': list(ARMS), 'validation_scheme': config['validation_scheme'], 'folds': 5,
                'strict_environment': strict}
    return config, frames, legacy, core, identity


def flat_identity(identity):
    return {k: identity[k] for k in ('run_id', 'config_sha256', 'runner_sha256', 'legacy_runner_sha256', 'core_sha256')} | {'identity_sha256': canonical_hash(identity)}


def compute_job(config, frames, legacy, core, identity, fold, arm, output):
    """The EXP042 add120 numerical path, unchanged, for exactly one fold/arm."""
    train, folds, extra, raw, clean = frames
    added = raw if arm == 'raw120' else clean
    lookup = {label: i for i, label in enumerate(config['labels'])}
    y, ey, ny = [frame.model.map(lookup).to_numpy(dtype=int) for frame in (train, extra, added)]
    fi = folds.fold.to_numpy(dtype=int)
    ti, vi = np.flatnonzero(fi != fold), np.flatnonzero(fi == fold)
    valid = train.iloc[vi].reset_index(drop=True)
    ev = legacy.Evidence(output, config, Path(config['_config_path']))
    ev.identity = flat_identity(identity)
    _, weight, masses = legacy.fixed_weights(y[ti], ey, ny)
    ev.save('weights.json', {'fold': fold, 'arm': arm,
            'classes': [dict(row, label=config['labels'][row['class_index']]) for row in masses],
            'old_row_weights_identical': True, 'normalization_denominator': 'base_extra_only'})
    texts = train.text.iloc[ti].tolist() + extra.text.tolist() + added.text.tolist()
    target = np.r_[y[ti], ey, ny]
    with ev.stage(fold, arm, 'tfidf_fit_transform'):
        grams = core.StyleGrams().fit(texts)
        x, xv = grams.transform(texts), grams.transform(valid.text.tolist())
        features = x.shape[1]
        del grams
        xc = legacy.centre_training(core, x, train.prompt_id.iloc[ti].to_numpy(), extra.prompt_id.to_numpy(), added.prompt_id.to_numpy())
    plain = legacy.fit_model(x, target, weight, ev, fold, arm, 'plain')
    del x
    centred = legacy.fit_model(xc, target, weight, ev, fold, arm, 'centred')
    del xc
    with ev.stage(fold, arm, 'dense_prediction'):
        groups = core.normalised_content_groups(valid.text.tolist())
        mask = core.eligible(groups)
        ps = plain.decision_function(xv)
        scores = ps.copy()
        if mask.any():
            scores[mask] = centred.decision_function(core.sparse_group_centre(xv[mask], groups[mask]))
        ev.matrix('dense.npz', ids=valid.id.to_numpy(dtype=str), scores=scores, plain_scores=ps,
                  true=y[vi], group_ids=groups, centred_mask=mask, route=np.where(mask, 'centred', 'plain'),
                  fold=np.full(len(vi), fold), arm=arm)
    with ev.stage(fold, arm, 'sparse_prediction'):
        for m in (2, 3, 6):
            for repeat in range(5):
                ev.check_budget()
                selected = legacy.sparse_indices(valid, m, repeat)
                group = core.normalised_content_groups(valid.text.iloc[selected].tolist())
                mk = core.eligible(group)
                sp = ps[selected].copy()
                if mk.any():
                    sp[mk] = centred.decision_function(core.sparse_group_centre(xv[selected[mk]], group[mk]))
                ev.matrix(f'sparse/m{m}-r{repeat}.npz', ids=valid.id.iloc[selected].to_numpy(dtype=str), scores=sp,
                          plain_scores=ps[selected], true=y[vi[selected]], group_ids=group, centred_mask=mk,
                          route=np.where(mk, 'centred', 'plain'), fold=np.full(len(selected), fold), arm=arm, m=m, repeat=repeat)
    ev.save('state.json', {'status': 'COMPLETE', 'fold': fold, 'arm': arm})
    ev.save('result.json', {'status': 'COMPLETE', 'fold': fold, 'arm': arm, 'features': features,
                          'rows': len(vi), 'balanced_accuracy': float(balanced_accuracy_score(y[vi], scores.argmax(1))),
                          'trained_svms': 2, 'seconds': time.monotonic()-ev.started})


def inventory(directory, exclude=frozenset()):
    return [{'path': p.relative_to(directory).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)}
            for p in sorted(directory.rglob('*')) if p.is_file() and p.relative_to(directory).as_posix() not in exclude]


def validate_inventory(directory, files, exclude=frozenset()):
    names = set()
    need(isinstance(files, list) and files, 'Empty checkpoint inventory')
    for entry in files:
        name = entry['path']
        path = PurePosixPath(name)
        need(name and '\\' not in name and ':' not in name and not path.is_absolute() and '..' not in path.parts
             and path.as_posix() == name and name not in names and name not in exclude, 'Unsafe/duplicate checkpoint entry')
        target = (directory / name).resolve(strict=True)
        need(target.is_relative_to(directory.resolve()) and target.is_file() and not target.is_symlink(), 'Checkpoint path escapes directory')
        need(type(entry['bytes']) is int and target.stat().st_size == entry['bytes'] and sha(target) == entry['sha256'], 'Checkpoint file hash mismatch: ' + name)
        names.add(name)
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file() and p.relative_to(directory).as_posix() not in exclude}
    need(actual == names, 'Checkpoint inventory is not closed')


def validate_job(directory, identity, fold, arm):
    receipt = read(directory / 'completion.json')
    need(receipt['status'] == 'COMPLETE' and receipt['identity'] == identity
         and receipt['fold'] == fold and receipt['arm'] == arm, 'Checkpoint identity/fold/arm mismatch')
    validate_inventory(directory, receipt['files'], {'completion.json'})
    check_job_products(directory, identity, fold, arm)
    return receipt


def check_job_products(directory, identity, fold, arm):
    expected = {'dense.npz', 'weights.json', 'result.json', 'state.json', 'identity.json', 'process.json', 'training.log', 'timings.json'}
    expected |= {f'sparse/m{m}-r{r}.npz' for m in (2, 3, 6) for r in range(5)}
    expected |= {f'fits/f{fold}-{arm}-{kind}.json' for kind in ('plain', 'centred')}
    expected.add(f'folds/f{fold}-{arm}-stages.json')
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file() and p.name != 'completion.json'}
    need(actual == expected, 'Missing/unexpected job output')
    job_identity = read(directory / 'identity.json')
    need(job_identity == {'identity': identity, 'fold': fold, 'arm': arm}, 'Job start identity mismatch')
    result = read(directory / 'result.json')
    need(result['status'] == 'COMPLETE' and result['fold'] == fold and result['arm'] == arm and result['trained_svms'] == 2, 'Job incomplete')
    need(read(directory / 'state.json')['status'] == 'COMPLETE', 'Job state incomplete')
    with np.load(directory / 'dense.npz', allow_pickle=False) as dense_file:
        dense = {key: dense_file[key] for key in dense_file.files}
    dense_lookup = {value: i for i, value in enumerate(dense['ids'].tolist())}
    for p in directory.rglob('*.npz'):
        with np.load(p, allow_pickle=False) as data:
            for key, value in flat_identity(identity).items():
                need(data[key].item() == value, 'NPZ identity mismatch')
            need(data['labels'].tolist() == identity['labels'], 'NPZ class order')
            n = len(data['ids'])
            need(n > 0 and len(set(data['ids'])) == n and data['scores'].shape == (n, 18)
                 and data['plain_scores'].shape == (n, 18), 'NPZ row/score shape')
            need(data['scores'].dtype == np.float64 and np.isfinite(data['scores']).all()
                 and data['plain_scores'].dtype == np.float64 and np.isfinite(data['plain_scores']).all(), 'NPZ finite float64 scores')
            need(data['fold'].shape == (n,) and data['fold'].dtype.kind in 'iu'
                 and np.all(data['fold'] == fold) and data['arm'].item() == arm, 'NPZ fold/arm')
            truth, groups, mask = data['true'], data['group_ids'], data['centred_mask']
            need(truth.shape == (n,) and truth.dtype.kind in 'iu' and ((truth >= 0) & (truth < 18)).all(), 'NPZ true labels')
            need(groups.shape == (n,) and groups.dtype.kind in 'iu' and (groups >= 1).all(), 'NPZ inference groups')
            need(mask.shape == (n,) and mask.dtype == np.bool_ and data['route'].shape == (n,), 'NPZ route shapes')
            _, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
            need(np.array_equal(mask, (counts[inv] >= 2) & (counts[inv] <= 18)), 'NPZ eligibility mask')
            need(np.array_equal(data['route'], np.where(mask, 'centred', 'plain'))
                 and np.array_equal(data['scores'][~mask], data['plain_scores'][~mask]), 'NPZ route consistency')
            if p.parent.name == 'sparse':
                mpart, rpart = p.stem.split('-')
                need(data['m'].item() == int(mpart[1:]) and data['repeat'].item() == int(rpart[1:]), 'Sparse scenario mismatch')
                need(set(data['ids']).issubset(dense_lookup), 'Sparse IDs outside dense fold')
                indices = np.array([dense_lookup[value] for value in data['ids']])
                need(np.array_equal(truth, dense['true'][indices])
                     and np.array_equal(data['plain_scores'], dense['plain_scores'][indices]), 'Sparse truth/plain cache mismatch')
    for kind in ('plain', 'centred'):
        fit = read(directory / f'fits/f{fold}-{arm}-{kind}.json')
        need(fit['C'] == .5 and fit['dual'] is True and fit['seed'] == 42 and fit['max_iter'] == 4000
             and fit['classes'] == list(range(18)) and fit['dtype'] == 'float64', 'Fit parameters changed')
        need(not any(w['category'] == 'ConvergenceWarning' for w in fit['warnings']), 'Convergence warning')


def seal_job(directory, identity, fold, arm):
    need(not (directory / 'completion.json').exists(), 'Cannot overwrite sealed job')
    check_job_products(directory, identity, fold, arm)
    receipt = {'status': 'COMPLETE', 'identity': identity, 'fold': fold, 'arm': arm,
               'files': inventory(directory), 'completed_at': time.time()}
    save(directory / 'completion.json', receipt)
    validate_job(directory, identity, fold, arm)


def find_completed(output, identity, plan):
    completed = {}
    allowed = {f'f{fold}-{arm}' for fold, arm in plan}
    root = output / 'jobs'
    if not root.exists():
        return completed
    need({p.name for p in root.iterdir() if p.is_dir()}.issubset(allowed), 'Unknown job directory')
    for fold, arm in plan:
        parent = root / f'f{fold}-{arm}'
        if not parent.exists():
            continue
        for directory in sorted(parent.iterdir()):
            need(directory.is_dir() and directory.name.startswith('attempt-'), 'Unexpected job attempt path')
            need(read(directory / 'identity.json') == {'identity': identity, 'fold': fold, 'arm': arm}, 'Incomplete job identity changed')
            if (directory / 'completion.json').exists():
                validate_job(directory, identity, fold, arm)
                need((fold, arm) not in completed, 'Multiple complete attempts for a job')
                completed[(fold, arm)] = directory
    return completed


def schedule(payload, output, identity, plan, attempt, synthetic=False, resume=False):
    """A single coordinator owns global status and commits each exited job."""
    completed = find_completed(output, identity, plan)
    need(resume or not completed, 'Existing completed jobs require --resume')
    pending = [job for job in plan if job not in completed]
    active = {}
    max_active = 0
    try:
        while pending or active:
            while pending and len(active) < 2:
                failed = [job for job, values in active.items()
                          if values[0].poll() is not None and values[0].returncode != 0]
                need(not failed, 'JOB_FAILED before refill: ' + repr(failed))
                fold, arm = pending.pop(0)
                parent = output / 'jobs' / f'f{fold}-{arm}'
                previous = list(parent.glob('attempt-*')) if parent.exists() else []
                directory = parent / f'attempt-{len(previous)+1:04d}'
                directory.mkdir(parents=True, exist_ok=False)
                save(directory / 'identity.json', {'identity': identity, 'fold': fold, 'arm': arm})
                command = [sys.executable, '-X', 'utf8', '-u', str(Path(__file__).resolve()), '--payload', str(payload), '--output', str(output),
                           '--_worker', str(directory), '--_fold', str(fold), '--_arm', arm]
                if synthetic:
                    command.append('--_synthetic')
                stream = (directory / 'training.log').open('x', encoding='utf-8')
                try:
                    child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                             creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                except BaseException:
                    stream.close()
                    raise
                save(directory / 'process.json', process_record(child.pid))
                active[(fold, arm)] = (child, stream, directory)
                max_active = max(max_active, len(active))
            save(output / 'state.json', {'status': 'RUNNING', 'identity_sha256': canonical_hash(identity), 'attempt': attempt,
                    'completed': [[f, a] for f, a in completed], 'active': [[f, a] for f, a in active]})
            finished = [job for job, values in active.items() if values[0].poll() is not None]
            # Detect every failure before launching replacements or sealing success.
            failures = [job for job in finished if active[job][0].returncode != 0]
            if failures:
                raise RuntimeError('JOB_FAILED: ' + repr(failures))
            for job in finished:
                child, stream, directory = active.pop(job)
                child.wait()
                stream.close()
                seal_job(directory, identity, *job)
                completed[job] = directory
                save(output / 'checkpoint.json', {'identity_sha256': canonical_hash(identity),
                    'completed': [{'fold': f, 'arm': a, 'path': p.relative_to(output).as_posix(),
                                   'completion_sha256': sha(p / 'completion.json')} for (f, a), p in completed.items()]})
            if active:
                time.sleep(.1)
    finally:
        for child, stream, directory in active.values():
            try:
                if child.poll() is None:
                    stop_tree(child.pid)
                child.wait(timeout=20)
            finally:
                stream.close()
    return completed, max_active


def synthetic_frames(legacy):
    rows, extras, additions = [], [], []
    subjects = ['astronomy orbit star planet', 'garden flower soil water', 'music melody tempo violin',
                'kitchen bread yeast oven', 'travel train station route']
    for p, subject in enumerate(subjects):
        for label, model in enumerate(legacy.LABELS):
            rows.append({'id': f'p{p}-m{label}', 'prompt_id': f'p{p}', 'model': model,
                         'text': f'{subject}. Example answer {label} uses phrase{label} and thoughtful details; group {p}.'})
    for p in range(2):
        for label, model in enumerate(legacy.LABELS):
            extras.append({'row_id': f'e{p}-{label}', 'prompt_id': f'e{p}', 'model': model,
                           'text': f'Public explanation phrase{label} carries details and reasoning. Topic {p}, answer {label}.'})
        for label in legacy.NEW_LABELS:
            additions.append({'row_id': f'n{p}-{label}', 'prompt_id': f'codex:n{p}', 'model': label,
                              'text': f'**Fresh topic {p}**\n\n- Thoughtful new example for {label}: direct reasons and details.'})
    train, extra, raw = pd.DataFrame(rows), pd.DataFrame(extras), pd.DataFrame(additions)
    folds = train[['id', 'prompt_id', 'model']].copy()
    folds['fold'] = np.repeat(np.arange(5), 18)
    clean = raw.copy()
    clean['text'] = clean.text.str.replace('**', '', regex=False).str.replace('\n\n- ', ' ', regex=False)
    return [train, folds, extra, raw, clean]


def synthetic_setup(payload, identity=None):
    legacy, core, legacy_path, core_path = modules(payload, fallback=True)
    config = {'run_id': 'exp044-synthetic', 'labels': legacy.LABELS, 'budget': {'hard_seconds': HARD_SECONDS},
              '_config_path': str(legacy_path)}
    identity = identity or {'run_id': 'exp044-synthetic', 'config_sha256': canonical_hash({'test': 'synthetic-v1'}),
                'runner_sha256': sha(__file__), 'legacy_runner_sha256': sha(legacy_path), 'core_sha256': sha(core_path),
                'environment': runtime(), 'labels': legacy.LABELS, 'purpose': 'synthetic execution equivalence only'}
    return config, synthetic_frames(legacy), legacy, core, identity


def legacy_reference(config, frames, legacy, core, arm):
    """Independent serial transcription of the frozen EXP042 add120 branch."""
    from sklearn.svm import LinearSVC
    from sklearn.exceptions import ConvergenceWarning
    import warnings
    train, folds, extra, raw, clean = frames
    added = raw if arm == 'raw120' else clean
    lookup = {label: i for i, label in enumerate(config['labels'])}
    y, ey, ny = [frame.model.map(lookup).to_numpy(dtype=int) for frame in (train, extra, added)]
    ti, vi = np.flatnonzero(folds.fold.to_numpy() != 0), np.flatnonzero(folds.fold.to_numpy() == 0)
    valid = train.iloc[vi].reset_index(drop=True)
    gram = core.StyleGrams().fit(train.text.iloc[ti].tolist() + extra.text.tolist() + added.text.tolist())
    x = gram.transform(train.text.iloc[ti].tolist() + extra.text.tolist() + added.text.tolist())
    xv = gram.transform(valid.text.tolist())
    xc = vstack([core.sparse_group_centre(x[:len(ti)], train.prompt_id.iloc[ti].to_numpy()),
                 core.sparse_group_centre(x[len(ti):len(ti)+len(extra)], extra.prompt_id.to_numpy(), 2, 10**6),
                 core.sparse_group_centre(x[len(ti)+len(extra):], added.prompt_id.to_numpy(), 2, 10**6)], format='csr')
    tr, ex = np.bincount(y[ti], minlength=18), np.bincount(ey, minlength=18)
    weight = np.r_[np.full(len(ti), .5), .5*tr[ey]/ex[ey], .5*tr[ny]/ex[ny]]
    target = np.r_[y[ti], ey, ny]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        plain = LinearSVC(C=.5, max_iter=4000, random_state=42, dual=True).fit(x, target, sample_weight=weight)
        centre = LinearSVC(C=.5, max_iter=4000, random_state=42, dual=True).fit(xc, target, sample_weight=weight)
    need(not any(issubclass(w.category, ConvergenceWarning) for w in caught), 'Synthetic serial convergence warning')
    ps = plain.decision_function(xv)
    outputs = {}
    for name, selected in [('dense.npz', np.arange(len(valid)))] + [(f'sparse/m{m}-r{r}.npz', legacy.sparse_indices(valid, m, r)) for m in (2, 3, 6) for r in range(5)]:
        groups = core.normalised_content_groups(valid.text.iloc[selected].tolist())
        mask = core.eligible(groups)
        scores = ps[selected].copy()
        if mask.any():
            scores[mask] = centre.decision_function(core.sparse_group_centre(xv[selected[mask]], groups[mask]))
        outputs[name] = {'scores': scores, 'plain_scores': ps[selected], 'group_ids': groups, 'centred_mask': mask}
    return outputs


def self_test(payload, output):
    need(not output.exists(), 'Self-test output must be fresh')
    output.mkdir(parents=True)
    config, frames, legacy, core, identity = synthetic_setup(payload)
    with threadpool_limits(limits=2):
        references = {arm: legacy_reference(config, frames, legacy, core, arm) for arm in ARMS}
    plan = [(0, arm) for arm in ARMS]
    completed, maximum = schedule(payload, output, identity, plan, 'selftest', synthetic=True)
    need(maximum == 2, 'Two-worker path was not exercised')
    maximum_difference = 0.
    for (fold, arm), directory in completed.items():
        for name, reference in references[arm].items():
            with np.load(directory / name, allow_pickle=False) as actual:
                for key, expected in reference.items():
                    if expected.dtype.kind == 'f':
                        diff = float(np.max(np.abs(actual[key] - expected)))
                        maximum_difference = max(maximum_difference, diff)
                        need(np.allclose(actual[key], expected, rtol=0, atol=1e-12), 'Serial/parallel numerical mismatch')
                    else:
                        need(np.array_equal(actual[key], expected), 'Serial/parallel routing mismatch')
                need(np.array_equal(actual['scores'].argmax(1), reference['scores'].argmax(1)), 'Serial/parallel predictions differ')
    before = {str(p): sha(p) for p in output.rglob('completion.json')}
    reused, resumed_workers = schedule(payload, output, identity, plan, 'selftest-resume', synthetic=True, resume=True)
    need(resumed_workers == 0 and reused == completed and before == {str(p): sha(p) for p in output.rglob('completion.json')}, 'Resume did not reuse unchanged completed jobs')
    # Corruption check uses a separate tiny receipt tree and never damages evidence.
    probe = output / 'corruption-probe'
    probe.mkdir()
    (probe / 'value').write_text('original', encoding='utf-8')
    files = inventory(probe)
    (probe / 'value').write_text('corrupted', encoding='utf-8')
    refused = False
    try:
        validate_inventory(probe, files)
    except ValueError:
        refused = True
    need(refused, 'Corruption was not rejected')
    result = {'status': 'PASS', 'environment': runtime(), 'runner_sha256': sha(__file__),
              'legacy_runner_sha256': identity['legacy_runner_sha256'], 'core_sha256': identity['core_sha256'],
              'classes': 18, 'parallel_workers_observed': maximum, 'max_abs_score_difference': maximum_difference,
              'completed_resume_reused': True, 'corruption_rejected': refused,
              'scope': 'small real StyleGrams numerical/concurrency test; not full model training or accuracy evidence'}
    save(output / 'selftest.json', result)
    save(output / 'selftest-receipt.json', {'files': inventory(output, {'selftest-receipt.json'}), 'result': result})
    return result


def aggregate(output, completed, frames, config, identity, attempt):
    train, folds = frames[:2]
    aggregate_dir = output / 'aggregates' / attempt
    aggregate_dir.mkdir(parents=True, exist_ok=False)
    labels = config['labels']
    summaries, jobs = [], []
    for arm in ARMS:
        fields = {key: [] for key in ('ids', 'scores', 'plain_scores', 'true', 'group_ids', 'centred_mask', 'route', 'fold')}
        positions = []
        for fold in range(5):
            directory = completed[(fold, arm)]
            validate_job(directory, identity, fold, arm)
            vi = np.flatnonzero(folds.fold.to_numpy() == fold)
            with np.load(directory / 'dense.npz', allow_pickle=False) as cache:
                need(cache['ids'].tolist() == train.id.iloc[vi].tolist(), 'Completed fold IDs mismatch official input')
                expected_y = train.model.iloc[vi].map({v: i for i, v in enumerate(labels)}).to_numpy(dtype=int)
                need(np.array_equal(cache['true'], expected_y), 'Completed fold labels mismatch official input')
                for key in fields:
                    fields[key].append(cache[key])
            positions.extend(vi.tolist())
            jobs.append({'fold': fold, 'arm': arm, 'path': directory.relative_to(output).as_posix(),
                         'completion_sha256': sha(directory / 'completion.json')})
        order = np.argsort(positions)
        arrays = {key: np.concatenate(parts)[order] for key, parts in fields.items()}
        need(len(positions) == 2699 and len(set(positions)) == 2699, 'Incomplete OOF')
        np.savez_compressed(aggregate_dir / f'oof-{arm}.npz', **arrays, labels=np.array(labels), arm=arm, **flat_identity(identity))
        pred = arrays['scores'].argmax(1)
        csv = train[['id', 'prompt_id', 'model']].copy()
        csv['fold'], csv['predicted_model'], csv['correct'] = arrays['fold'], np.array(labels)[pred], pred == arrays['true']
        csv.to_csv(aggregate_dir / f'oof-{arm}.csv', index=False)
        summaries.append({'arm': arm, 'balanced_accuracy': float(balanced_accuracy_score(arrays['true'], pred)),
                          'accuracy': float((pred == arrays['true']).mean()), 'correct': int((pred == arrays['true']).sum())})
    result = {'status': 'COMPLETE', **flat_identity(identity), 'identity': identity, 'summaries': summaries,
              'trained_svms': 20, 'completed_jobs': jobs, 'aggregate_path': aggregate_dir.relative_to(output).as_posix(),
              'gpu_used': False, 'scope': 'fixed raw120 versus clean120 Stylegram OOF only'}
    save(aggregate_dir / 'result.json', result)
    save(output / 'result.json', result)
    save(output / 'state.json', {'status': 'COMPLETE', 'identity_sha256': canonical_hash(identity), 'completed_jobs': jobs})


def coordinated(args):
    # Only the live guard holding the matching nonce may start/resume scheduling.
    ticket = read(args.output / '.guard.lock')
    need(ticket['nonce'] == args._ticket and process_live(ticket['process']), 'No matching live guard')
    config, frames, legacy, core, identity = preflight(args.payload, strict=True)
    config['_config_path'] = str(args.payload / 'config.json')
    run_path = args.output / 'run.json'
    if run_path.exists():
        need(args.resume and read(run_path)['identity'] == identity, 'Resume run/config/code/input/environment identity mismatch')
    else:
        need(not args.resume, 'Resume requested without original run identity')
        save(run_path, {'identity': identity, 'config': read(args.payload / 'config.json')})
    attempt = ticket['attempt']
    save(args.output / 'attempts' / attempt / 'preflight.json', {'status': 'PASS', 'identity': identity})
    gates = sorted((args.output / 'selftests').glob('attempt-*'))
    passed_gate = False
    for gate in gates:
        need(args.resume, 'Existing self-test requires explicit resume')
        if not (gate / 'selftest-receipt.json').exists():
            continue  # Incomplete gate is preserved; new attempt gets a fresh gate.
        receipt = read(gate / 'selftest-receipt.json')
        validate_inventory(gate, receipt['files'], {'selftest-receipt.json'})
        result = receipt['result']
        need(result['status'] == 'PASS' and result['environment'] == runtime() and result['runner_sha256'] == sha(__file__)
             and result['legacy_runner_sha256'] == identity['legacy_runner_sha256'] and result['core_sha256'] == identity['core_sha256'], 'Self-test gate identity mismatch')
        passed_gate = True
    if not passed_gate:
        self_test(args.payload, args.output / 'selftests' / attempt)
    completed, max_active = schedule(args.payload, args.output, identity, [(fold, arm) for fold in range(5) for arm in ARMS], attempt, resume=args.resume)
    aggregate(args.output, completed, frames, config, identity, attempt)
    save(args.output / 'attempts' / attempt / 'scheduler.json', {'status': 'COMPLETE', 'max_active_workers': max_active})


def worker(args):
    directory = args._worker
    start = read(directory / 'identity.json')
    # Parent writes the immutable process receipt immediately after spawn.
    deadline = time.monotonic() + 10
    while not (directory / 'process.json').exists() and time.monotonic() < deadline:
        time.sleep(.02)
    need((directory / 'process.json').exists(), 'Worker start receipt missing')
    need(read(directory / 'process.json')['pid'] == os.getpid(), 'Worker PID receipt mismatch')
    try:
        if args._synthetic:
            config, frames, legacy, core, identity = synthetic_setup(args.payload)
        else:
            config, frames, legacy, core, identity = preflight(args.payload, strict=True)
            config['_config_path'] = str(args.payload / 'config.json')
        need(start == {'identity': identity, 'fold': args._fold, 'arm': args._arm}, 'Worker identity mismatch')
        with threadpool_limits(limits=2):
            compute_job(config, frames, legacy, core, identity, args._fold, args._arm, directory)
    except BaseException:
        save(directory / 'failure.json', {'status': 'FAILED', 'traceback': traceback.format_exc()})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload', type=Path, default=HERE)
    parser.add_argument('--output', type=Path, default=Path(os.environ.get('NTL_OUTPUT_DIR', '/kaggle/working/results')))
    parser.add_argument('--resume', action='store_true')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--preflight-only', action='store_true')
    modes.add_argument('--self-test', action='store_true')
    parser.add_argument('--_ticket')
    parser.add_argument('--_worker', type=Path)
    parser.add_argument('--_fold', type=int)
    parser.add_argument('--_arm', choices=ARMS)
    parser.add_argument('--_synthetic', action='store_true')
    args = parser.parse_args(argv)
    args.payload, args.output = args.payload.resolve(), args.output.resolve()
    if args._worker:
        worker(args)
    elif args._ticket:
        coordinated(args)
    elif args.preflight_only:
        need(not args.resume and not args.output.exists(), 'Preflight requires fresh output and no resume')
        config, frames, legacy, core, identity = preflight(args.payload, strict=False)
        save(args.output / 'preflight.json', {'status': 'PASS', 'identity': identity, 'training_started': False})
    elif args.self_test:
        need(not args.resume, 'Standalone self-test requires fresh output')
        print(json.dumps(self_test(args.payload, args.output)))
    else:
        command = [sys.executable, '-u', str(args.payload / 'guard.py'), '--payload', str(args.payload), '--output', str(args.output)]
        if args.resume:
            command.append('--resume')
        raise SystemExit(subprocess.call(command))


if __name__ == '__main__':
    main()
