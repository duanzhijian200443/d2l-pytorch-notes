"""Read-only EXP042 output verification and fixed paired post-analysis.

No model fitting, network, alternate weights, retries, or holdout evaluation.
The artifacts SHA must be acquired separately from the downloaded artifact tree.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import traceback

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CLOUD = ROOT / 'cloud/luna-terra120-042'
BASE = ROOT / 'experiments/EXP-010-012-stylegram-share'
ARMS = ('control', 'add120')
DYNAMIC = {'artifacts.json', 'guard.json', 'state.json', 'training.log', '.run.lock'}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def equal(left, right, name):
    need(np.array_equal(left, right), f'Array mismatch: {name}')


def near(left, right, name):
    need(np.isclose(left, right, rtol=0, atol=1e-12), f'Numeric mismatch: {name}')


def integer_vector(value, n, name, minimum=0, maximum=None):
    value = np.asarray(value)
    need(value.shape == (n,) and value.dtype.kind in 'iu', f'Invalid integer vector: {name}')
    need((value >= minimum).all() and (maximum is None or (value <= maximum).all()), f'Integer range: {name}')
    return value


def score_matrix(value, n, classes=18):
    value = np.asarray(value)
    need(value.shape == (n, classes) and value.dtype == np.float64 and np.isfinite(value).all(), 'Invalid decision matrix')
    return value


def metrics(y, pred, classes=18):
    y = integer_vector(y, len(y), 'truth', maximum=classes-1)
    pred = integer_vector(pred, len(y), 'prediction', maximum=classes-1)
    support = np.bincount(y, minlength=classes)
    correct = np.bincount(y[pred == y], minlength=classes)
    recall = np.divide(correct, support, out=np.full(classes, np.nan), where=support > 0)
    # A missing class never silently changes the fixed 18-class estimand.
    return {'rows': len(y), 'balanced_accuracy': float(recall.mean()) if (support > 0).all() else None,
            'accuracy': float((pred == y).mean()) if len(y) else None, 'correct': int(correct.sum()),
            'support': support.tolist(), 'class_correct': correct.tolist(),
            'recall': [float(v) if np.isfinite(v) else None for v in recall],
            'missing_class_indices': np.flatnonzero(support == 0).tolist()}


def paired_metrics(y, control, candidate, groups, classes=18, draws=2000, seed=42):
    need(len(y) == len(control) == len(candidate) == len(groups), 'Pairing length mismatch')
    base = metrics(y, control, classes)
    added = metrics(y, candidate, classes)
    _, gi = np.unique(np.asarray(groups), return_inverse=True)
    ng = int(gi.max()) + 1
    support = np.zeros((ng, classes), dtype=np.int64)
    delta_correct = np.zeros_like(support)
    np.add.at(support, (gi, y), 1)
    np.add.at(delta_correct, (gi, y), (candidate == y).astype(int) - (control == y).astype(int))
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, ng, size=(draws, ng))
    den = support[selected].sum(axis=1)
    num = delta_correct[selected].sum(axis=1)
    valid = (den > 0).all(axis=1)
    deltas = np.full(draws, np.nan)
    deltas[valid] = (num[valid] / den[valid]).mean(axis=1)
    ok = deltas[valid]
    result = {'delta_ba': None if base['balanced_accuracy'] is None or added['balanced_accuracy'] is None
              else added['balanced_accuracy'] - base['balanced_accuracy'],
              'fixed_errors': int(((control != y) & (candidate == y)).sum()),
              'introduced_errors': int(((control == y) & (candidate != y)).sum()),
              'prediction_disagreements': int((control != candidate).sum()),
              'bootstrap': {'unit': 'official prompt_id', 'draws': draws, 'seed': seed,
                            'valid_draws': int(valid.sum()), 'null_draws': int((~valid).sum()), 'redraws': 0,
                            'missing_class_policy': 'store null; no redraw; percentile interval uses valid draws only',
                            'ci95': np.quantile(ok, [.025, .975]).tolist() if len(ok) else None,
                            'deltas': [float(v) if np.isfinite(v) else None for v in deltas]}}
    return result


def check_weights(receipt, official_y, extra_y, new_y, labels):
    classes = len(labels)
    official = np.bincount(official_y, minlength=classes)
    extra = np.bincount(extra_y, minlength=classes)
    added = np.bincount(new_y, minlength=classes)
    need((official > 0).all() and (extra > 0).all(), 'Weight class absent')
    need(receipt['old_row_weights_identical'] is True, 'Old weights not declared identical')
    need(len(receipt['classes']) == classes, 'Weight class cardinality')
    for i, row in enumerate(receipt['classes']):
        need(row['class_index'] == i and row['label'] == labels[i], 'Weight label order')
        for key, expected in {'official_rows': int(official[i]), 'base_extra_rows': int(extra[i]),
                              'added_rows': int(added[i])}.items():
            need(type(row[key]) is int and row[key] == expected, f'Weight count: {key}')
        public_weight = .5 * official[i] / extra[i]
        expected = {'official_per_row_weight': .5, 'public_per_row_weight': public_weight,
                    'official_mass': .5 * official[i], 'base_extra_mass': .5 * official[i],
                    'control_mass': float(official[i]), 'added_mass': added[i] * public_weight,
                    'add120_mass': official[i] + added[i] * public_weight}
        for key, value in expected.items():
            near(row[key], value, 'weight ' + key)
    # Same frozen code plus per-class receipts attest the fit weight rule. Models
    # and training weight arrays were not saved; no byte-level fit replay claim.


def verify_manifest(results, anchor):
    need(len(anchor) == 64 and all(c in '0123456789abcdef' for c in anchor), 'Invalid external SHA256')
    need(sha(results / 'artifacts.json') == anchor, 'External artifacts anchor mismatch')
    manifest = read(results / 'artifacts.json')
    need(isinstance(manifest.get('files'), list) and manifest['files'], 'Invalid artifact list')
    listed = set()
    for entry in manifest['files']:
        name = entry['path']
        p = PurePosixPath(name)
        need(name and '\\' not in name and ':' not in name and not p.is_absolute()
             and '..' not in p.parts and p.as_posix() == name and name not in listed, 'Unsafe/duplicate artifact path')
        need(p.name not in DYNAMIC and not p.name.endswith('.tmp'), 'Dynamic file in stable manifest')
        target = (results / name).resolve(strict=True)
        need(target.is_relative_to(results.resolve()) and target.is_file(), 'Artifact outside results')
        need(type(entry['bytes']) is int and target.stat().st_size == entry['bytes'] and sha(target) == entry['sha256'], 'Artifact bytes mismatch: ' + name)
        listed.add(name)
    actual = {p.relative_to(results).as_posix() for p in results.rglob('*') if p.is_file() and p.name not in DYNAMIC and not p.name.endswith('.tmp')}
    need(actual == listed and 'result.json' in listed and 'failure.json' not in listed, 'Stable artifact inventory mismatch/failure')
    need(not any(p.name.endswith('.tmp') for p in results.rglob('*')), 'Incomplete temporary output')
    return manifest, listed


def check_identity(document, expected, name):
    for key, value in expected.items():
        need(document.get(key) == value, f'{name}: identity mismatch {key}')


def load_npz(path, identity, labels):
    with np.load(path, allow_pickle=False) as z:
        data = {key: z[key] for key in z.files}
    for key, value in identity.items():
        need(data[key].shape == () and data[key].item() == value, f'{path.name}: {key}')
    equal(data['labels'], labels, path.name + ' labels')
    return data


def check_prediction(data, ids, y, folds, arm, classes=18):
    n = len(ids)
    need(len(set(ids)) == n, 'Duplicate expected IDs')
    equal(data['ids'], ids, 'ids/order')
    integer_vector(data['true'], n, 'true', maximum=classes-1)
    equal(data['true'], y, 'truth')
    integer_vector(data['fold'], n, 'fold', maximum=4)
    equal(data['fold'], folds, 'fold')
    need(data['arm'].shape == () and data['arm'].item() == arm, 'Arm mismatch')
    score_matrix(data['scores'], n, classes)
    score_matrix(data['plain_scores'], n, classes)
    groups = integer_vector(data['group_ids'], n, 'inference group', minimum=1)
    mask = data['centred_mask']
    need(mask.shape == (n,) and mask.dtype == np.bool_, 'Invalid route mask')
    expected_mask = np.zeros(n, dtype=bool)
    for fold in np.unique(folds):
        selected = np.flatnonzero(folds == fold)
        _, inv, counts = np.unique(groups[selected], return_inverse=True, return_counts=True)
        expected_mask[selected] = (counts[inv] >= 2) & (counts[inv] <= 18)
    equal(mask, expected_mask, 'eligible group routing')
    equal(data['route'], np.where(mask, 'centred', 'plain'), 'route names')
    equal(data['scores'][~mask], data['plain_scores'][~mask], 'plain route scores')
    return data['scores'].argmax(axis=1)


def sparse_selection(valid, m, repeat):
    frame = valid.copy()
    frame['_key'] = frame.id.map(lambda value: hashlib.sha256(f'42:{repeat}:{value}'.encode()).hexdigest())
    selected_ids = set(frame.sort_values(['prompt_id', '_key']).groupby('prompt_id', sort=True).head(m).id)
    return np.flatnonzero(valid.id.isin(selected_ids).to_numpy())


def old_baseline_check(control, train, fold_ids, y, labels, out):
    download = read(BASE / 'acceptance-20261101/download-manifest.json')
    need(read(BASE / 'acceptance-20261101/acceptance-status.json')['status'] == 'PASS', 'Old baseline not accepted')
    need(download['notebook_ref'] == 'zhjianduan/name-that-llm-stylegram-share-cv/2', 'Old baseline version')
    entries = {e['path']: e for e in download['files']}
    oldroot = BASE / 'download-v2/results'
    def anchored(name):
        p = oldroot / name
        e = entries['results/' + name]
        need(p.stat().st_size == e['bytes'] and sha(p) == e['sha256'], 'Old baseline bytes: ' + name)
        return p
    scores = score_matrix(np.load(anchored('oof-share0.5.npy'), allow_pickle=False), len(train))
    csv = pd.read_csv(anchored('oof-share0.5.csv'), dtype={'id': str, 'prompt_id': str})
    for column in ('id', 'prompt_id', 'model'):
        equal(csv[column], train[column], 'old baseline ' + column)
    equal(csv.fold, fold_ids, 'old baseline folds')
    equal(csv.predicted_model, np.array(labels)[scores.argmax(1)], 'old baseline CSV argmax')
    equal(csv.correct, scores.argmax(1) == y, 'old baseline CSV correct')
    for fold in range(5):
        with np.load(anchored(f'fold{fold}-share0.5.npz'), allow_pickle=False) as z:
            vi = np.flatnonzero(fold_ids == fold)
            equal(z['ids'], train.id.iloc[vi], 'old fold IDs')
            equal(z['labels'], labels, 'old fold labels')
            equal(z['true'], y[vi], 'old fold truth')
            equal(z['scores'], scores[vi], 'old fold decision matrix')
    mismatches = scores.argmax(1) != control.argmax(1)
    diff = train[['id', 'prompt_id', 'model']].copy()
    diff['old_pred'] = np.array(labels)[scores.argmax(1)]
    diff['control_pred'] = np.array(labels)[control.argmax(1)]
    diff['max_abs_margin_delta'] = np.max(np.abs(scores - control), axis=1)
    diff.loc[mismatches].to_csv(out / 'baseline_prediction_differences.csv', index=False)
    return {'status': 'PASS' if not mismatches.any() else 'FAIL', 'prediction_disagreements': int(mismatches.sum()),
            'matrix_byte_values_equal': bool(np.array_equal(scores, control)),
            'max_abs_margin_delta': float(np.max(np.abs(scores - control))),
            'old_download_manifest_sha256': sha(BASE / 'acceptance-20261101/download-manifest.json'),
            'old_oof_sha256': sha(oldroot / 'oof-share0.5.npy'),
            'requirement': 'same predictions required; margin differences reported explicitly'}


def verify(results, anchor, output):
    manifest, listed = verify_manifest(results, anchor)
    config = read(HERE / 'config.json')
    frozen = read(HERE / 'manifest.json')
    need(frozen['freeze_status'] == 'FROZEN_BEFORE_TRAINING', 'EXP042 not frozen')
    need(frozen['config']['sha256'] == sha(HERE / 'config.json'), 'Frozen config changed')
    need(frozen['run_id'] == config['run_id'], 'Manifest run identity')
    need(config['experiment_id'] == 'EXP-042' and config['arms'] == list(ARMS), 'Experiment/arms')
    need(config['validation_scheme'] == 'prompt-group-v1', 'Validation scheme')
    need(config['evaluation']['bootstrap_draws'] == 2000 and config['evaluation']['bootstrap_seed'] == 42, 'Bootstrap protocol')
    payload = CLOUD / 'payload'
    need(sha(payload / 'config.json') == sha(HERE / 'config.json'), 'Payload config changed')
    for name, digest in config['input_hashes'].items():
        need(Path(name).name == name and sha(payload / name) == digest, 'Frozen payload hash: ' + name)
    code = next(e for e in frozen['code'] if e['path'] == (HERE / 'runner.py').relative_to(ROOT).as_posix())
    need(code['sha256'] == config['input_hashes']['runner.py'] == sha(HERE / 'runner.py'), 'Runner differs from frozen source')
    identity = {'run_id': config['run_id'], 'config_sha256': sha(HERE / 'config.json'), 'runner_sha256': code['sha256']}
    check_identity(manifest, identity, 'artifacts')
    for name in listed:
        if name.endswith('.json'):
            check_identity(read(results / name), identity, name)
    guard, state, result = [read(results / name) for name in ('guard.json', 'state.json', 'result.json')]
    need(all(doc['status'] == 'COMPLETE' for doc in (guard, state, result)), 'Non-COMPLETE terminal status')
    check_identity(state, identity, 'state')
    need(guard['child_exit'] == 0 and guard['artifacts_verified'] is True and guard['termination_reason'] is None, 'Guard did not validate success')
    need(guard['elapsed_seconds'] < config['budget']['hard_seconds'] and guard['peak_rss_bytes'] <= config['budget']['rss_limit_bytes'], 'Guard budget violation')
    need(guard['wall_seconds'] == config['budget']['hard_seconds'] and guard['rss_limit_bytes'] == config['budget']['rss_limit_bytes'], 'Guard configured budget mismatch')
    need(result['trained_svms'] == 20 and result['folds'] == 5 and result['gpu_used'] is False, 'Training scope mismatch')
    need(0 < result['total_seconds'] < config['budget']['hard_seconds'], 'Runner budget violation')
    preflight = read(results / 'preflight.json')
    need(preflight['status'] == 'PASS' and preflight['strict_environment'] is True and preflight['environment_mismatches'] == {}, 'Runtime preflight not strict PASS')
    need(preflight['environment'] == config['environment']['required_versions'], 'Runtime versions differ')
    need(preflight['input_hashes'] == config['input_hashes'] and preflight['gpu_requested'] is False, 'Runtime inputs/GPU differ')
    need(read(results / 'source.json')['config'] == config, 'Runtime source config mismatch')
    execution = read(CLOUD / 'execution-state.json')
    need(execution['run_id'] == config['run_id'] and isinstance(execution['version'], int) and execution['version'] > 0, 'Cloud version/run receipt missing')
    need(execution['notebook_ref'] == 'zhjianduan/name-that-llm-luna-terra120-042', 'Cloud notebook mismatch')
    labels = config['labels']
    need(len(labels) == 18 and len(set(labels)) == 18, 'Label count/order')
    train = pd.read_csv(payload / 'train.csv', dtype=str, keep_default_na=False)
    folds = pd.read_csv(payload / 'folds.csv', dtype={'id': str, 'prompt_id': str})
    extra = pd.read_csv(payload / 'extra.csv', dtype=str, keep_default_na=False)
    new = pd.read_csv(payload / 'new120.csv', dtype=str, keep_default_na=False)
    need(len(train) == 2699 and len(extra) == 10073 and len(new) == 120, 'Training input counts')
    need(new.model.value_counts().to_dict() == {'gpt-5.6-luna': 60, 'gpt-5.6-terra': 60} and new.split.eq('train').all(), 'New training class/split counts')
    need(new.prompt_id.nunique() == 60 and new.groupby('prompt_id').size().eq(2).all(), 'New prompt pair count')
    for col in ('id', 'prompt_id', 'model'):
        equal(train[col], folds[col], 'input alignment ' + col)
    need(folds.groupby('prompt_id').fold.nunique().eq(1).all(), 'Official prompt leakage')
    lookup = {label: i for i, label in enumerate(labels)}
    y, ey, ny = [frame.model.map(lookup).to_numpy(dtype=int) for frame in (train, extra, new)]
    fold_ids = folds.fold.to_numpy(dtype=int)
    completed = [{'fold': f, 'arm': a} for f in range(5) for a in ARMS]
    need(state['completed'] == completed, 'Completion order/count')
    stage_rows = read(results / 'timings.json')['stages']
    expected_stages = [(f, 'shared', 'unlabelled_sparse_grouping') for f in range(5)]
    expected_stages += [(f, a, s) for f in range(5) for a in ARMS for s in ('tfidf_fit_transform', 'plain_fit', 'centred_fit', 'dense_prediction', 'sparse_prediction')]
    expected_stages += [('all', a, 'save_oof') for a in ARMS]
    need(sorted(map(str, [(r['fold'], r['arm'], r['stage']) for r in stage_rows])) == sorted(map(str, expected_stages)), 'Stage coverage')
    need(all(r['status'] == 'COMPLETE' and np.isfinite(r['seconds']) and r['seconds'] >= 0 for r in stage_rows), 'Failed/invalid stage')
    expected_fold_names = {f'folds/f{f}-{a}.npz' for f in range(5) for a in ARMS}
    expected_sparse_names = {f'sparse/f{f}-{a}-m{m}-r{r}.npz' for f in range(5) for a in ARMS for m in (2, 3, 6) for r in range(5)}
    need({n for n in listed if n.startswith('folds/') and n.endswith('.npz')} == expected_fold_names, 'Dense fold cache count')
    need({n for n in listed if n.startswith('sparse/') and n.endswith('.npz')} == expected_sparse_names, 'Sparse cache count')
    need({n for n in listed if n.startswith('fits/')} == {f'fits/f{f}-{a}-{k}.json' for f in range(5) for a in ARMS for k in ('plain', 'centred')}, 'Fit log inventory')
    dense, dense_parts, perclass, fold_metrics = {}, {}, [], []
    reported_folds = read(results / 'fold_metrics.json')['records']
    need(len(reported_folds) == 10 and len({(r['fold'], r['arm']) for r in reported_folds}) == 10, 'Reported fold metric count')
    for fold in range(5):
        check_weights(read(results / f'weights/f{fold}.json'), y[fold_ids != fold], ey, ny, labels)
    for arm in ARMS:
        data = load_npz(results / f'oof/{arm}.npz', identity, labels)
        pred = check_prediction(data, train.id.to_numpy(), y, fold_ids, arm)
        dense_parts[arm] = data
        dense[arm] = metrics(y, pred)
        reported = [r for r in result['summaries'] if r['arm'] == arm]
        need(len(reported) == 1, 'Result summary arm count')
        for key in ('balanced_accuracy', 'accuracy'):
            near(dense[arm][key], reported[0][key], 'Reported dense ' + key)
        csv = pd.read_csv(results / f'oof/{arm}.csv', dtype={'id': str, 'prompt_id': str})
        for col in ('id', 'prompt_id', 'model'):
            equal(csv[col], train[col], 'OOF CSV ' + col)
        equal(csv.fold, fold_ids, 'OOF CSV folds')
        equal(csv.predicted_model, np.array(labels)[pred], 'OOF CSV argmax')
        equal(csv.correct, pred == y, 'OOF CSV correctness')
        for key, value in identity.items():
            need(csv[key].eq(value).all(), 'OOF CSV identity')
        for i, label in enumerate(labels):
            perclass.append({'arm': arm, 'model': label, 'support': dense[arm]['support'][i],
                             'correct': dense[arm]['class_correct'][i], 'recall': dense[arm]['recall'][i]})
        for fold in range(5):
            vi = np.flatnonzero(fold_ids == fold)
            cache = load_npz(results / f'folds/f{fold}-{arm}.npz', identity, labels)
            fpred = check_prediction(cache, train.id.iloc[vi].to_numpy(), y[vi], fold_ids[vi], arm)
            for key in ('scores', 'plain_scores', 'true', 'group_ids', 'centred_mask', 'route', 'ids', 'fold'):
                equal(cache[key], data[key][vi], 'Fold vs OOF ' + key)
            fm = metrics(y[vi], fpred)
            fold_metrics.append({'fold': fold, 'arm': arm, **fm})
            rec = next(r for r in reported_folds if r['fold'] == fold and r['arm'] == arm)
            near(rec['balanced_accuracy'], fm['balanced_accuracy'], 'Fold BA')
            for kind in ('plain', 'centred'):
                fit = read(results / f'fits/f{fold}-{arm}-{kind}.json')
                need(fit['fold'] == fold and fit['arm'] == arm and fit['kind'] == kind, 'Fit identity')
                need(fit['C'] == .5 and fit['dual'] is True and fit['max_iter'] == 4000 and fit['seed'] == 42
                     and fit['dtype'] == 'float64' and fit['classes'] == list(range(18)), 'Fit protocol')
                need(0 < fit['n_iter'] <= 4000 and not any(w['category'] == 'ConvergenceWarning' for w in fit['warnings']), 'Fit convergence')
                need(fit['shape'] == [len(train)-len(vi)+10073+(120 if arm == 'add120' else 0), rec['features']], 'Fit rows/features')
    equal(dense_parts['control']['group_ids'], dense_parts['add120']['group_ids'], 'Dense arm inference grouping')
    equal(dense_parts['control']['centred_mask'], dense_parts['add120']['centred_mask'], 'Dense arm routing')
    basepred, addpred = [dense_parts[a]['scores'].argmax(1) for a in ARMS]
    paired = paired_metrics(y, basepred, addpred, train.prompt_id.to_numpy())
    paired['fold_deltas'] = {str(f): next(r['balanced_accuracy'] for r in fold_metrics if r['fold'] == f and r['arm'] == 'add120')
                            - next(r['balanced_accuracy'] for r in fold_metrics if r['fold'] == f and r['arm'] == 'control')
                            for f in range(5)}
    sparse = []
    for m in (2, 3, 6):
        for repeat in range(5):
            joined = {}
            for arm in ARMS:
                parts = []
                for fold in range(5):
                    vi = np.flatnonzero(fold_ids == fold)
                    valid = train.iloc[vi].reset_index(drop=True)
                    selected = sparse_selection(valid, m, repeat)
                    global_ix = vi[selected]
                    cache = load_npz(results / f'sparse/f{fold}-{arm}-m{m}-r{repeat}.npz', identity, labels)
                    need(cache['m'].item() == m and cache['repeat'].item() == repeat, 'Sparse scenario identity')
                    check_prediction(cache, train.id.iloc[global_ix].to_numpy(), y[global_ix], fold_ids[global_ix], arm)
                    equal(cache['plain_scores'], dense_parts[arm]['plain_scores'][global_ix], 'Sparse plain cache')
                    parts.append((cache, global_ix))
                ids = np.concatenate([p[0]['ids'] for p in parts])
                need(len(set(ids)) == len(ids), 'Duplicate pooled sparse IDs')
                joined[arm] = {key: np.concatenate([p[0][key] for p in parts]) for key in ('ids', 'scores', 'true', 'group_ids', 'centred_mask')}
                joined[arm]['indices'] = np.concatenate([p[1] for p in parts])
            for key in ('ids', 'true', 'indices', 'group_ids', 'centred_mask'):
                equal(joined['control'][key], joined['add120'][key], 'Paired sparse ' + key)
            sy = joined['control']['true']
            bp, ap = [joined[a]['scores'].argmax(1) for a in ARMS]
            bm, am = metrics(sy, bp), metrics(sy, ap)
            need(bm['balanced_accuracy'] is not None and am['balanced_accuracy'] is not None, 'Sparse pooled class absent')
            sparse.append({'m': m, 'repeat': repeat, 'control': bm, 'add120': am,
                           'delta_ba': am['balanced_accuracy']-bm['balanced_accuracy'],
                           'fixed_errors': int(((bp != sy) & (ap == sy)).sum()),
                           'introduced_errors': int(((bp == sy) & (ap != sy)).sum())})
    baseline = old_baseline_check(dense_parts['control']['scores'], train, fold_ids, y, labels, output)
    predictions = train[['id', 'prompt_id', 'model']].copy()
    predictions['fold'] = fold_ids
    predictions['control'] = np.array(labels)[basepred]
    predictions['add120'] = np.array(labels)[addpred]
    predictions['fixed_error'] = (basepred != y) & (addpred == y)
    predictions['introduced_error'] = (basepred == y) & (addpred != y)
    predictions.to_csv(output / 'predictions.csv', index=False)
    pd.DataFrame(perclass).to_csv(output / 'dense_perclass.csv', index=False)
    save(output / 'fold_metrics.json', fold_metrics)
    save(output / 'sparse_summary.json', {'scenarios': sparse, 'aggregates': [
        {'m': m, 'mean_delta_ba': float(np.mean([r['delta_ba'] for r in sparse if r['m'] == m])),
         'positive_repeats': sum(r['delta_ba'] > 0 for r in sparse if r['m'] == m),
         'negative_repeats': sum(r['delta_ba'] < 0 for r in sparse if r['m'] == m)} for m in (2, 3, 6)]})
    save(output / 'summary.json', {'dense': dense, 'paired': paired, 'historical_control': baseline,
        'scope': 'Fixed Stylegram development OOF only; conditional fitted-model bootstrap; no new holdout/leaderboard claim',
        'weight_evidence_limit': 'Recomputed class weight receipts and frozen runner; fitted models and per-row training weight arrays were not saved.'})
    verification = {'status': 'PASS' if baseline['status'] == 'PASS' else 'FAIL', **identity,
        'external_artifacts_sha256': anchor, 'stable_files': len(listed), 'runtime_environment': preflight['environment'],
        'runtime_environment_source': 'results/preflight.json.environment',
        'cloud_version_receipt': execution, 'cloud_version_receipt_sha256': sha(CLOUD / 'execution-state.json'),
        'cloud_version_limit': 'Local execution receipt linked by run_id; external artifacts anchor supplied separately by caller.',
        'official_rows': 2699, 'dense_fold_caches': 10, 'sparse_fold_caches': 150, 'trained_svms': 20,
        'historical_control': baseline, 'verifier_sha256': sha(__file__),
        'observed_at': datetime.now(timezone.utc).isoformat()}
    save(output / 'verification.json', verification)
    need(baseline['status'] == 'PASS', 'Historical control predictions differ; fairness baseline is not PASS')
    return verification


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', type=Path, required=True)
    ap.add_argument('--artifacts-sha256', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    need(not args.output.exists(), 'Analysis output must be fresh; no overwrite')
    need(not args.output.resolve().is_relative_to(args.results.resolve()), 'Analysis cannot be written inside results')
    need(not args.results.resolve().is_relative_to(args.output.resolve()), 'Analysis cannot contain the source results')
    args.output.mkdir(parents=True)
    try:
        result = verify(args.results, args.artifacts_sha256, args.output)
        print(json.dumps(result, ensure_ascii=False))
    except BaseException:
        failure = {'status': 'FAIL', 'traceback': traceback.format_exc(),
                   'external_artifacts_sha256': args.artifacts_sha256, 'verifier_sha256': sha(__file__)}
        save(args.output / 'failure.json', failure)
        if not (args.output / 'verification.json').exists():
            save(args.output / 'verification.json', failure)
        raise


if __name__ == '__main__':
    main()
