"""EXP-042: fixed Stylegram control versus 120 additional raw training rows.

No network, installation, GPU code, retry, or resume. The external guard enforces
wall/RSS limits during native fit calls; this runner checks elapsed time at stages.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '2')

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import time
import traceback
import unicodedata
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy.sparse import vstack
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

LABELS = ['claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8',
          'claude-opus-5', 'claude-sonnet-5', 'gemini-3.1-pro-preview',
          'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
          'glm-5.3', 'glm-5.3-flash', 'gpt-5.5', 'gpt-5.6-luna',
          'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra', 'kimi-k3', 'qwen3.8-max']
NEW_LABELS = ['gpt-5.6-luna', 'gpt-5.6-terra']
VERSIONS = {'numpy': '2.1.3', 'pandas': '2.3.3', 'scipy': '1.16.3', 'sklearn': '1.6.1'}
FROZEN_HASHES = {
    'train.csv': '40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707',
    'extra.csv': '1e9125fd49b19084842332ede84ea65bc0174ee1bb28c8d48acc8ddbd5e36c0e',
    'folds.csv': 'e8ba1e3712852168dc57f56960d4f3e00be612d639ade8a7dcb0f0d8335ce551',
    'stylegram_core.py': 'caa46bd121164aba891c3d78e662540db9294d653db09f53d02833c1b4ab77ef',
}
DYNAMIC_NAMES = {'state.json', 'guard.json', 'training.log', '.run.lock', 'artifacts.json'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def stamp():
    return datetime.now(timezone.utc).isoformat()


def validate_config(config):
    require(isinstance(config.get('run_id'), str) and bool(config['run_id']), 'Missing run_id')
    require(config.get('labels') == LABELS, 'Fixed label order mismatch')
    require(config.get('new_labels') == NEW_LABELS, 'Additional labels must be Luna/Terra')
    require(config.get('validation_scheme') == 'prompt-group-v1', 'Wrong validation scheme')
    for key, expected in {'C': .5, 'max_iter': 4000, 'seed': 42, 'dual': True,
                          'sparse_per_prompt': [2, 3, 6], 'sparse_repeats': 5}.items():
        require(config.get(key) == expected, f'Fixed parameter mismatch: {key}')
    require(isinstance(config.get('identity'), dict) and bool(config['identity']), 'Missing source identity')
    seconds = config.get('budget', {}).get('hard_seconds')
    require(isinstance(seconds, (int, float)) and not isinstance(seconds, bool)
            and math.isfinite(seconds) and 0 < seconds <= 7200, 'Invalid hard budget')
    held = config.get('holdout_prompt_group_hashes')
    require(isinstance(held, list) and len(held) > 0 and len(set(held)) == len(held)
            and all(isinstance(v, str) and len(v) == 64 and all(c in '0123456789abcdef' for c in v)
                    for v in held), 'Missing or invalid holdout group hashes')


def verify_hashes(payload, config):
    hashes = config.get('input_hashes', {})
    require(set(FROZEN_HASHES).union({'new120.csv'}).issubset(hashes), 'Missing required input hashes')
    for name, digest in hashes.items():
        require(Path(name).name == name and ':' not in name and '\\' not in name, 'Unsafe input name')
        require((payload / name).is_file() and sha(payload / name) == digest, f'Input hash mismatch: {name}')
    for name, digest in FROZEN_HASHES.items():
        require(hashes[name] == digest, f'Frozen baseline changed: {name}')


def validate_new_rows(new, train, extra, holdout):
    required = {'row_id', 'origin', 'prompt_id', 'model', 'text', 'base_prompt',
                'text_sha256', 'job_key', 'split', 'source_protocol'}
    require(required.issubset(new.columns), 'Missing new120 columns')
    require(len(new) == 120 and new.row_id.is_unique and new.job_key.is_unique, 'new120 row/job cardinality')
    require(all(new[c].map(lambda x: isinstance(x, str) and bool(x)).all() for c in required), 'Empty new120 fields')
    require(new.split.eq('train').all(), 'Non-train row rejected')
    require(new.model.value_counts().to_dict() == {label: 60 for label in NEW_LABELS}, 'new120 labels/counts')
    require(new.prompt_id.nunique() == 60 and new.groupby('prompt_id').size().eq(2).all(), 'new120 group counts')
    require(new.groupby('prompt_id').model.nunique().eq(2).all(), 'Each group needs both new labels')
    hashes = new.text.map(text_hash)
    require(hashes.tolist() == new.text_sha256.tolist(), 'Raw text hash mismatch')
    require(hashes.is_unique, 'Duplicate new text')
    require(set(hashes).isdisjoint(set(train.text.map(text_hash)) | set(extra.text.map(text_hash))), 'New text overlaps baseline')
    groups = new.base_prompt.map(lambda t: text_hash(normalize(t)))
    require(groups.nunique() == 60, 'Normalized new prompt group count')
    pairs = pd.DataFrame({'id': new.prompt_id, 'hash': groups})
    require(pairs.groupby('id')['hash'].nunique().eq(1).all()
            and pairs.groupby('hash').id.nunique().eq(1).all(), 'Prompt ID/hash mapping is not one-to-one')
    require(set(groups).isdisjoint(holdout), 'Holdout prompt group in training')
    return groups


def fixed_weights(official_y, base_extra_y, added_y, classes=18):
    """Old row weights are identical in both arms; added mass is not renormalized."""
    counts = np.bincount(official_y, minlength=classes).astype(float)
    base_counts = np.bincount(base_extra_y, minlength=classes).astype(float)
    require((counts > 0).all() and (base_counts > 0).all(), 'Missing training class')
    public_per_row = .5 * counts / base_counts
    control = np.r_[np.full(len(official_y), .5), public_per_row[base_extra_y]]
    add = np.r_[control, public_per_row[added_y]]
    added_counts = np.bincount(added_y, minlength=classes)
    audit = []
    for label in range(classes):
        audit.append({'class_index': label, 'official_rows': int(counts[label]),
                      'base_extra_rows': int(base_counts[label]), 'added_rows': int(added_counts[label]),
                      'official_per_row_weight': .5, 'public_per_row_weight': float(public_per_row[label]),
                      'official_mass': float(.5 * counts[label]), 'base_extra_mass': float(.5 * counts[label]),
                      'control_mass': float(counts[label]),
                      'added_mass': float(added_counts[label] * public_per_row[label]),
                      'add120_mass': float(counts[label] + added_counts[label] * public_per_row[label])})
    return control, add, audit


def sparse_indices(valid, m, repeat):
    keys = valid.id.map(lambda value: text_hash(f'42:{repeat}:{value}'))
    frame = pd.DataFrame({'prompt': valid.prompt_id.to_numpy(), 'key': keys.to_numpy(), 'index': np.arange(len(valid))})
    return np.sort(frame.sort_values(['prompt', 'key']).groupby('prompt', sort=True).head(m)['index'].to_numpy())


def centre_training(core, x, official_groups, public_groups, new_groups):
    n, e = len(official_groups), len(public_groups)
    blocks = [core.sparse_group_centre(x[:n], official_groups),
              core.sparse_group_centre(x[n:n+e], public_groups, 2, 10**6)]
    if len(new_groups):
        # Separate block plus namespace: public group IDs cannot absorb new rows.
        codex_groups = np.array(['codex:' + str(g).removeprefix('codex:') for g in new_groups])
        blocks.append(core.sparse_group_centre(x[n+e:], codex_groups, 2, 10**6))
    return vstack(blocks, format='csr')


class Evidence:
    def __init__(self, out, config, config_path):
        self.out, self.config = out, config
        self.started = time.monotonic()
        self.identity = {'run_id': config.get('run_id'), 'config_sha256': sha(config_path), 'runner_sha256': sha(__file__)}
        self.current = None
        self.completed = []
        self.stages = []

    def save(self, name, data):
        target = self.out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(target.suffix + '.tmp')
        temp.write_text(json.dumps({**self.identity, **data}, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temp.replace(target)

    def check_budget(self):
        if time.monotonic() - self.started >= self.config['budget']['hard_seconds']:
            raise TimeoutError('WALL_BUDGET_EXCEEDED')

    @contextmanager
    def stage(self, fold, arm, name):
        self.check_budget()
        begin = time.monotonic()
        self.current = {'fold': fold, 'arm': arm, 'stage': name, 'started_at': stamp()}
        self.save('state.json', {'status': 'RUNNING', 'current': self.current, 'completed': self.completed})
        print(json.dumps({**self.identity, **self.current, 'status': 'STARTED'}), flush=True)
        try:
            yield
            self.check_budget()
        except BaseException:
            self.stages.append({**self.current, 'status': 'FAILED', 'seconds': time.monotonic() - begin})
            self.save('timings.json', {'stages': self.stages})
            raise
        else:
            self.stages.append({**self.current, 'status': 'COMPLETE', 'seconds': time.monotonic() - begin})
            self.save('timings.json', {'stages': self.stages})
            self.save(f'folds/f{fold}-{arm}-stages.json', {'stages': [s for s in self.stages if s['fold'] == fold and s['arm'] == arm]})

    def matrix(self, name, **arrays):
        for key in ('scores', 'plain_scores'):
            if key in arrays:
                require(np.isfinite(arrays[key]).all(), f'Nonfinite {key}')
        target = self.out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.with_suffix('.npz.tmp').open('wb') as stream:
            np.savez_compressed(stream, **arrays, labels=np.array(LABELS), **self.identity)
        target.with_suffix('.npz.tmp').replace(target)

    def manifest(self):
        files = [{'path': p.relative_to(self.out).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)}
                 for p in sorted(self.out.rglob('*')) if p.is_file()
                 and p.name not in DYNAMIC_NAMES and not p.name.endswith('.tmp')]
        self.save('artifacts.json', {'files': files})


def preflight(payload, config, strict):
    validate_config(config)
    verify_hashes(payload, config)
    frames = [pd.read_csv(payload / name, dtype=str, keep_default_na=False)
              for name in ('train.csv', 'folds.csv', 'extra.csv', 'new120.csv')]
    train, folds, extra, new = frames
    require(len(train) == len(folds) == 2699 and train.id.is_unique, 'Official row count/IDs')
    for column in ('id', 'prompt_id', 'model'):
        require(train[column].tolist() == folds[column].tolist(), f'Fold alignment: {column}')
    require(train.text.map(text_hash).tolist() == folds.text_sha256.tolist(), 'Fold text hashes')
    folds['fold'] = pd.to_numeric(folds.fold, errors='raise')
    require(set(folds.fold) == set(range(5)) and folds.groupby('prompt_id').fold.nunique().eq(1).all(), 'Fold leakage')
    require(set(train.model) == set(LABELS) and set(extra.model) == set(LABELS), 'Baseline labels')
    require(len(extra) == 10073 and extra.row_id.is_unique and extra.prompt_id.nunique() == 694, 'Baseline public counts')
    require(extra.groupby('prompt_id').canonical_prompt.nunique().eq(1).all()
            and not extra.canonical_prompt.str.startswith('official:').any(), 'Baseline public grouping')
    require(set(train.text.map(normalize)).isdisjoint(extra.text.map(normalize)), 'Baseline text overlap')
    prompt_hashes = validate_new_rows(new, train, extra, config['holdout_prompt_group_hashes'])
    actual = {'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__}
    mismatch = {k: {'expected': v, 'actual': actual[k]} for k, v in VERSIONS.items() if actual[k] != v}
    if strict:
        require(not mismatch, f'Strict environment mismatch: {mismatch}')
        require(not Path('/dev/nvidia0').exists(), 'GPU device detected; CPU-only execution required')
    return frames, {'status': 'PASS', 'strict_environment': strict, 'environment': actual,
                    'environment_mismatches': mismatch, 'python': platform.python_version(),
                    'rows': {'official': len(train), 'base_extra': len(extra), 'new': len(new)},
                    'new_prompt_groups': int(prompt_hashes.nunique()), 'holdout_overlap': 0,
                    'input_hashes': config['input_hashes'], 'gpu_requested': False}


def fit_model(x, target, weight, ev, fold, arm, kind):
    with ev.stage(fold, arm, kind + '_fit'):
        require(x.dtype == np.float64, 'Stylegram must remain float64')
        model = LinearSVC(C=.5, dual=True, max_iter=4000, random_state=42)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.fit(x, target, sample_weight=weight)
        warning_records = [{'category': w.category.__name__, 'message': str(w.message)} for w in caught]
        ev.save(f'fits/f{fold}-{arm}-{kind}.json', {'fold': fold, 'arm': arm, 'kind': kind,
                'shape': list(x.shape), 'dtype': str(x.dtype), 'C': .5, 'dual': True, 'max_iter': 4000,
                'seed': 42, 'classes': model.classes_.tolist(), 'n_iter': int(model.n_iter_), 'warnings': warning_records})
        require(not any(issubclass(w.category, ConvergenceWarning) for w in caught), 'SVM convergence warning')
        require(model.classes_.tolist() == list(range(18)), 'SVM class order')
    return model


def run_training(payload, config, frames, ev):
    spec = importlib.util.spec_from_file_location('exp042_stylegram_core', payload / 'stylegram_core.py')
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    train, folds, extra, new = frames
    lookup = {label: i for i, label in enumerate(LABELS)}
    y, ey, ny = [f.model.map(lookup).to_numpy(dtype=int) for f in (train, extra, new)]
    fold_ids = folds.fold.to_numpy(dtype=int)
    oof = {arm: np.full((len(train), 18), np.nan) for arm in ('control', 'add120')}
    oof_plain = {arm: np.full((len(train), 18), np.nan) for arm in oof}
    oof_groups = {arm: np.zeros(len(train), dtype=int) for arm in oof}
    oof_masks = {arm: np.zeros(len(train), dtype=bool) for arm in oof}
    records = []
    for fold in range(5):
        ti, vi = np.flatnonzero(fold_ids != fold), np.flatnonzero(fold_ids == fold)
        valid = train.iloc[vi].reset_index(drop=True)
        control_w, added_w, masses = fixed_weights(y[ti], ey, ny)
        ev.save(f'weights/f{fold}.json', {'classes': [dict(row, label=LABELS[row['class_index']]) for row in masses],
                'old_row_weights_identical': True, 'new_weight_rule': 'same class base-extra per-row weight',
                'normalization_denominator': 'base extra 10073 rows only; additional mass is retained'})
        scenarios = []
        with ev.stage(fold, 'shared', 'unlabelled_sparse_grouping'):
            for m in (2, 3, 6):
                for repeat in range(5):
                    ev.check_budget()
                    selected = sparse_indices(valid, m, repeat)
                    groups = core.normalised_content_groups(valid.text.iloc[selected].tolist())
                    scenarios.append((m, repeat, selected, groups, core.eligible(groups)))
        for arm in ('control', 'add120'):
            is_added = arm == 'add120'
            added = new if is_added else new.iloc[:0]
            texts = train.text.iloc[ti].tolist() + extra.text.tolist() + added.text.tolist()
            target = np.r_[y[ti], ey, ny if is_added else np.array([], dtype=int)]
            weight = added_w if is_added else control_w
            with ev.stage(fold, arm, 'tfidf_fit_transform'):
                grams = core.StyleGrams().fit(texts)
                x, xv = grams.transform(texts), grams.transform(valid.text.tolist())
                features = x.shape[1]
                del grams
                xc = centre_training(core, x, train.prompt_id.iloc[ti].to_numpy(), extra.prompt_id.to_numpy(), added.prompt_id.to_numpy())
            plain = fit_model(x, target, weight, ev, fold, arm, 'plain')
            del x
            centred = fit_model(xc, target, weight, ev, fold, arm, 'centred')
            del xc
            with ev.stage(fold, arm, 'dense_prediction'):
                groups = core.normalised_content_groups(valid.text.tolist())
                mask = core.eligible(groups)
                plain_scores = plain.decision_function(xv)
                scores = plain_scores.copy()
                if mask.any():
                    scores[mask] = centred.decision_function(core.sparse_group_centre(xv[mask], groups[mask]))
                oof[arm][vi], oof_plain[arm][vi] = scores, plain_scores
                oof_groups[arm][vi], oof_masks[arm][vi] = groups, mask
                ev.matrix(f'folds/f{fold}-{arm}.npz', ids=valid.id.to_numpy(dtype=str), scores=scores,
                          plain_scores=plain_scores, true=y[vi], group_ids=groups, centred_mask=mask,
                          route=np.where(mask, 'centred', 'plain'), fold=np.full(len(vi), fold), arm=arm)
            with ev.stage(fold, arm, 'sparse_prediction'):
                for m, repeat, selected, groups, mask in scenarios:
                    ev.check_budget()
                    sp = plain_scores[selected].copy()
                    if mask.any():
                        sp[mask] = centred.decision_function(core.sparse_group_centre(xv[selected[mask]], groups[mask]))
                    ev.matrix(f'sparse/f{fold}-{arm}-m{m}-r{repeat}.npz', ids=valid.id.iloc[selected].to_numpy(dtype=str),
                              scores=sp, plain_scores=plain_scores[selected], true=y[vi[selected]], group_ids=groups,
                              centred_mask=mask, route=np.where(mask, 'centred', 'plain'),
                              fold=np.full(len(selected), fold), arm=arm, m=m, repeat=repeat)
            records.append({'fold': fold, 'arm': arm, 'features': features, 'rows': len(vi),
                            'balanced_accuracy': float(balanced_accuracy_score(y[vi], scores.argmax(1)))})
            ev.completed.append({'fold': fold, 'arm': arm})
            ev.save('fold_metrics.json', {'records': records})
            ev.save('state.json', {'status': 'RUNNING', 'completed': ev.completed, 'current': None})
            del plain, centred, xv, texts
            gc.collect()
    summaries = []
    for arm in oof:
        with ev.stage('all', arm, 'save_oof'):
            scores, mask = oof[arm], oof_masks[arm]
            ev.matrix(f'oof/{arm}.npz', ids=train.id.to_numpy(dtype=str), scores=scores,
                      plain_scores=oof_plain[arm], true=y, group_ids=oof_groups[arm], centred_mask=mask,
                      route=np.where(mask, 'centred', 'plain'), fold=fold_ids, arm=arm)
            table = train[['id', 'prompt_id', 'model']].copy()
            table['fold'], table['predicted_model'] = fold_ids, np.array(LABELS)[scores.argmax(1)]
            table['correct'] = scores.argmax(1) == y
            for key, value in ev.identity.items():
                table[key] = value
            table.to_csv(ev.out / f'oof/{arm}.csv', index=False)
            summaries.append({'arm': arm, 'balanced_accuracy': float(balanced_accuracy_score(y, scores.argmax(1))),
                              'accuracy': float((scores.argmax(1) == y).mean())})
    ev.check_budget()
    ev.save('result.json', {'status': 'COMPLETE', 'summaries': summaries, 'trained_svms': 20, 'folds': 5,
            'total_seconds': time.monotonic() - ev.started, 'gpu_used': False,
            'scope': 'Stylegram grouped OOF only; not leaderboard or full ensemble'})
    ev.manifest()
    ev.save('state.json', {'status': 'COMPLETE', 'completed': ev.completed, 'observed_at': stamp()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, default=Path(os.environ.get('NTL_OUTPUT_DIR', '/kaggle/working/results')))
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    allowed = {'.run.lock', 'guard.json', 'training.log'}
    require(all(p.name in allowed for p in args.output.iterdir()), 'Output already contains results; no retry or resume')
    config_path = args.payload / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    ev = Evidence(args.output, config, config_path)
    try:
        validate_config(config)
        ev.save('source.json', {'config': config, 'observed_at': stamp()})
        frames, report = preflight(args.payload, config, strict=not args.preflight_only)
        ev.save('preflight.json', report)
        ev.check_budget()
        if args.preflight_only:
            ev.save('state.json', {'status': 'PREFLIGHT_COMPLETE', 'training_started': False})
            ev.manifest()
            return
        with threadpool_limits(limits=2):
            run_training(args.payload, config, frames, ev)
    except BaseException:
        ev.save('failure.json', {'status': 'FAILED', 'current': ev.current, 'completed': ev.completed,
                'elapsed_seconds': time.monotonic() - ev.started, 'traceback': traceback.format_exc(), 'observed_at': stamp()})
        ev.save('state.json', {'status': 'ERROR', 'current': ev.current, 'completed': ev.completed})
        ev.manifest()
        raise


if __name__ == '__main__':
    main()
