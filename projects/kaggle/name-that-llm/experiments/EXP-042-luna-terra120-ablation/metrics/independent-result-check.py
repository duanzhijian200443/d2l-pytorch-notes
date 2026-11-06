"""Independent CSV/NPZ arithmetic; no experiment metric helper, training or network."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = ROOT / 'cloud/luna-terra120-042/outputs-v2/results'
OLD = ROOT / 'experiments/EXP-010-012-stylegram-share/download-v2/results'
ANCHOR = 'bd4d9c2913924659fa0b83af99e08e0aa65393ee8848af54b2889939086ec2bb'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(RESULTS / 'artifacts.json') == ANCHOR
manifest = json.loads((RESULTS / 'artifacts.json').read_text(encoding='utf-8'))
for entry in manifest['files']:
    path = RESULTS / entry['path']
    assert path.resolve().is_relative_to(RESULTS.resolve())
    assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], entry['path']


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def metrics(truth, prediction):
    counts = np.bincount(truth, minlength=18)
    correct = np.bincount(truth[prediction == truth], minlength=18)
    assert (counts > 0).all(), 'Missing true class'
    return {'rows': len(truth), 'correct': int(correct.sum()),
            'accuracy': float(correct.sum() / counts.sum()),
            'balanced_accuracy': float((correct / counts).mean())}


def compare(truth, baseline, candidate):
    a, b = metrics(truth, baseline), metrics(truth, candidate)
    ca, cb = baseline == truth, candidate == truth
    return {'control': a, 'add120': b,
            'delta_ba': b['balanced_accuracy'] - a['balanced_accuracy'],
            'delta_correct': b['correct'] - a['correct'],
            'both_correct': int((ca & cb).sum()), 'fixed_errors': int((~ca & cb).sum()),
            'new_errors': int((ca & ~cb).sum()), 'both_wrong': int((~ca & ~cb).sum()),
            'changed_predictions': int((baseline != candidate).sum())}


fold_path = ROOT / 'validation/prompt-group-v1/folds.csv'
assert sha(fold_path) == 'e8ba1e3712852168dc57f56960d4f3e00be612d639ade8a7dcb0f0d8335ce551'
official = list(csv.DictReader(fold_path.open(encoding='utf-8', newline='')))
ids = np.array([row['id'] for row in official])
folds = np.array([int(row['fold']) for row in official])
prompts = np.array([row['prompt_id'] for row in official])
assert len(official) == 2699 and len(set(ids)) == 2699
arms = {arm: load(RESULTS / f'oof/{arm}.npz') for arm in ['control', 'add120']}
labels = arms['control']['labels'].tolist()
assert len(labels) == len(set(labels)) == 18
truth = np.array([labels.index(row['model']) for row in official])
for arm, data in arms.items():
    assert np.array_equal(data['ids'], ids) and np.array_equal(data['true'], truth)
    assert np.array_equal(data['fold'], folds) and data['labels'].tolist() == labels
    assert data['scores'].shape == (2699, 18) and np.isfinite(data['scores']).all()
    for key in ['run_id', 'config_sha256', 'runner_sha256']:
        assert data[key].item() == manifest[key]
pred = {arm: data['scores'].argmax(axis=1) for arm, data in arms.items()}
overall = compare(truth, pred['control'], pred['add120'])
per_class = []
for i, label in enumerate(labels):
    mask = truth == i
    ca = int((pred['control'][mask] == i).sum())
    cb = int((pred['add120'][mask] == i).sum())
    per_class.append({'label': label, 'support': int(mask.sum()), 'correct_control': ca, 'correct_add120': cb,
                      'recall_control': ca / int(mask.sum()), 'recall_add120': cb / int(mask.sum()),
                      'delta_correct': cb - ca, 'delta_recall': (cb - ca) / int(mask.sum()),
                      'predicted_control': int((pred['control'] == i).sum()),
                      'predicted_add120': int((pred['add120'] == i).sum()),
                      'false_positives_control': int((pred['control'] == i).sum()) - ca,
                      'false_positives_add120': int((pred['add120'] == i).sum()) - cb,
                      'precision_control': ca / int((pred['control'] == i).sum()) if (pred['control'] == i).any() else None,
                      'precision_add120': cb / int((pred['add120'] == i).sum()) if (pred['add120'] == i).any() else None})
per_fold = []
for f in range(5):
    selected = folds == f
    per_fold.append({'fold': f, **compare(truth[selected], pred['control'][selected], pred['add120'][selected])})

# Paired cluster bootstrap: 150 canonical official prompts, with replacement.
groups = sorted(set(prompts))
assert len(groups) == 150
den = np.zeros((150, 18), dtype=np.int64)
diff = np.zeros_like(den)
for g, group in enumerate(groups):
    use = prompts == group
    assert len(set(folds[use])) == 1
    den[g] = np.bincount(truth[use], minlength=18)
    diff[g] = (np.bincount(truth[use & (pred['add120'] == truth)], minlength=18)
               - np.bincount(truth[use & (pred['control'] == truth)], minlength=18))
draws = np.random.default_rng(42).integers(0, 150, size=(2000, 150))
values, invalid = [], 0
for draw in draws:
    multiplicity = np.bincount(draw, minlength=150)
    support = multiplicity @ den
    if np.any(support == 0):
        invalid += 1
    else:
        values.append(float(np.mean((multiplicity @ diff) / support)))
interval = None if invalid else np.quantile(values, [.025, .975], method='linear').tolist()

old_scores = np.load(OLD / 'oof-share0.5.npy', allow_pickle=False)
old_rows = list(csv.DictReader((OLD / 'oof-share0.5.csv').open(encoding='utf-8', newline='')))
assert [row['id'] for row in old_rows] == ids.tolist()
control_parity = {'oof_scores_exact': bool(np.array_equal(old_scores, arms['control']['scores'])),
                  'oof_max_absolute_error': float(np.max(np.abs(old_scores - arms['control']['scores']))),
                  'oof_predictions_exact': bool(np.array_equal(old_scores.argmax(1), pred['control']))}
fold_parity = []
for f in range(5):
    old = load(OLD / f'fold{f}-share0.5.npz')
    new = load(RESULTS / f'folds/f{f}-control.npz')
    equal = {k: bool(np.array_equal(old[k], new[k])) for k in ['ids', 'labels', 'true', 'scores', 'plain_scores', 'group_ids', 'centred_mask']}
    fold_parity.append({'fold': f, 'fields_exact': equal})

sparse, sparse_parity = [], []
for size in [2, 3, 6]:
    for repeat in range(5):
        pooled = {arm: {'true': [], 'pred': [], 'ids': []} for arm in arms}
        for f in range(5):
            current = {}
            for arm in arms:
                data = load(RESULTS / f'sparse/f{f}-{arm}-m{size}-r{repeat}.npz')
                assert data['labels'].tolist() == labels and np.isfinite(data['scores']).all()
                assert data['run_id'].item() == manifest['run_id']
                pooled[arm]['true'].append(data['true'])
                pooled[arm]['pred'].append(data['scores'].argmax(1))
                pooled[arm]['ids'].append(data['ids'])
                current[arm] = data
            for k in ['ids', 'true', 'group_ids', 'centred_mask']:
                assert np.array_equal(current['control'][k], current['add120'][k]), (f, size, repeat, k)
            old = load(OLD / f'sparse-f{f}-a0.5-m{size}-r{repeat}.npz')
            equal = {k: bool(np.array_equal(old[k], current['control'][k])) for k in ['ids', 'true', 'scores', 'group_ids', 'centred_mask']}
            sparse_parity.append({'fold': f, 'm': size, 'repeat': repeat, 'fields_exact': equal})
        for arm in arms:
            pooled[arm] = {k: np.concatenate(v) for k, v in pooled[arm].items()}
            assert len(set(pooled[arm]['ids'])) == len(pooled[arm]['ids'])
        sparse.append({'m': size, 'repeat': repeat, **compare(pooled['control']['true'], pooled['control']['pred'], pooled['add120']['pred'])})
sparse_summary = []
for size in [2, 3, 6]:
    selected = [row for row in sparse if row['m'] == size]
    sparse_summary.append({'m': size,
        'mean_control_ba': float(np.mean([row['control']['balanced_accuracy'] for row in selected])),
        'mean_add120_ba': float(np.mean([row['add120']['balanced_accuracy'] for row in selected])),
        'mean_delta_ba': float(np.mean([row['delta_ba'] for row in selected])),
        'positive_repeats': sum(row['delta_ba'] > 0 for row in selected),
        'negative_repeats': sum(row['delta_ba'] < 0 for row in selected),
        'zero_repeats': sum(row['delta_ba'] == 0 for row in selected)})
control_parity['all_fold_fields_exact'] = all(all(row['fields_exact'].values()) for row in fold_parity)
control_parity['all_75_sparse_fields_exact'] = all(all(row['fields_exact'].values()) for row in sparse_parity)
published = json.loads((RESULTS / 'result.json').read_text(encoding='utf-8'))
assert all(abs(next(row for row in published['summaries'] if row['arm'] == arm)['balanced_accuracy']
               - overall[arm]['balanced_accuracy']) < 1e-14 for arm in arms)
report = {'status': 'PASS_INDEPENDENT_ARITHMETIC', 'run_id': manifest['run_id'], 'artifact_anchor': ANCHOR,
    'script_sha256': sha(Path(__file__)), 'folds_sha256': sha(fold_path),
    'source_manifest_files_verified': len(manifest['files']), 'numpy_version': np.__version__,
    'overall': overall, 'per_class': per_class, 'per_fold': per_fold,
    'prompt_bootstrap': {'groups': 150, 'draws': 2000, 'seed': 42, 'valid': len(values), 'invalid': invalid,
       'ci95_delta_ba': interval, 'ci95_percentage_points': None if interval is None else [100 * x for x in interval],
       'group_order': 'lexicographically sorted official prompt_id', 'quantile_method': 'linear',
       'scope': 'paired group resampling, fixed trained predictions, no retraining uncertainty'},
    'sparse': sparse, 'sparse_summary': sparse_summary, 'historical_control_parity': control_parity,
    'historical_fold_parity': fold_parity, 'historical_sparse_parity': sparse_parity,
    'limitations': ['Independent result arithmetic and anchored file verification; no classifier fitting or network.',
       'OOF development folds have been inspected previously; not a fresh holdout.',
       'Per-class and sparse gains cannot be selected after observing results to override the frozen overall primary endpoint.',
       'Luna/Terra labels are requested_only; no authenticated backend snapshot claim.',
       'Stylegram-only result does not establish complete-ensemble or leaderboard change.']}
Path(__file__).with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: report[k] for k in ['status', 'overall', 'per_class', 'per_fold', 'prompt_bootstrap', 'sparse_summary', 'historical_control_parity']}, ensure_ascii=False, indent=2))
